"""
pmbm6 — a 6-DOF projective motion blur model with an exact autograd adjoint.

Extends the rotation-only PMBM of MATH_FOUNDATION.md §2-§5 to a full SE(3)
twist (§13 of the addendum), keeping every structural property intact:

    B = (1/T) sum_n  T_{H_n}        linear in u        B^T exact via autograd

Conventions
-----------
Camera-frame point velocity under a body twist xi = (nu, omega):

    dX/dt = -nu - omega x X          =>   d/dt [X;1] = A [X;1],
    A = [[-hat(omega), -nu], [0, 0]]

Pixel i of the BLURRY image accumulates the latent pixel that, at time t,
sits at i.  Back-project i at depth Z, propagate to the exposure centre, and
reproject:

    Delta(t)(i) = pi( K exp(-A t) Z(i) K^{-1} i ) - i
    v(i)        = (1/T) sum_n u( i + Delta(t_n) ),   t_n uniform on [-tau/2, tau/2]

so d/dt Delta |_0 = -L(i) xi with L the interaction matrix of eq. (23).
The sign is verified in `selftest_interaction_matrix`.

Everything below is numpy/torch only; no learned components.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "interaction_matrix", "stacked_interaction", "admissibility",
    "se3_exp_matrix", "offset_field", "PMBM6", "estimate_twist_lstsq",
]

# ----------------------------------------------------------------------------
# 1. Interaction matrix  (addendum eq. 23)
# ----------------------------------------------------------------------------

def interaction_matrix(u, v, Z, f):
    """L(u,v,Z) in pixels, columns [nu_x, nu_y, nu_z, w_x, w_y, w_z].

    u, v are offsets from the principal point (pixels). Broadcasts over arrays.
    Returns shape (..., 2, 6).
    """
    u, v, Z = np.broadcast_arrays(np.asarray(u, float),
                                  np.asarray(v, float),
                                  np.asarray(Z, float))
    zero = np.zeros_like(u)
    row0 = np.stack([-f / Z, zero, u / Z, u * v / f, -(f + u * u / f), v], -1)
    row1 = np.stack([zero, -f / Z, v / Z, f + v * v / f, -u * v / f, -u], -1)
    return np.stack([row0, row1], -2)


def stacked_interaction(K, Z, tau, stride=8):
    """L stacked over pixels, scaled to pixels-of-displacement per unit twist.

    Z : (H, W) depth map in metres.  Returns (2*n, 6).
    """
    f = 0.5 * (K[0, 0] + K[1, 1])
    H, W = Z.shape
    yy, xx = np.mgrid[0:H:stride, 0:W:stride]
    L = interaction_matrix(xx - K[0, 2], yy - K[1, 2], Z[::stride, ::stride], f)
    return L.reshape(-1, 6) * tau


def admissibility(K, Z, tau, sigma_meas=0.5, stride=8, tol=None):
    """Per-frame 6-DOF admissibility gate (addendum §15.3).

    The meaningful quantity is the achievable precision along the WEAKEST twist
    direction, which for the least-squares estimator of eq. (26) is

        delta_xi_worst = sigma_meas / sigma_min(L_stacked)

    with L_stacked *unnormalised* (so that adding pixels genuinely helps:
    sigma_min grows like sqrt(n)).  `sigma6_rms` is the per-pixel-RMS singular
    value, which is scale-free and comparable across image sizes.

    tol : optional dict {'nu': m/s, 'omega': rad/s}; sets `passes`.
    """
    Lm = stacked_interaction(K, Z, tau, stride)
    n = Lm.shape[0] // 2
    U, s, Vt = np.linalg.svd(Lm, full_matrices=False)
    worst = float(sigma_meas / s[-1])
    w = Vt[-1]
    # split the weakest direction into its translational / rotational parts
    frac_nu = float(np.linalg.norm(w[:3]))
    lim = None
    if tol is not None:
        lim = frac_nu * tol.get("nu", np.inf) + (1 - frac_nu) * tol.get("omega", np.inf)
    return dict(sigma=s / np.sqrt(n), sigma_raw=s, cond=float(s[0] / s[-1]),
                sigma6_rms=float(s[-1] / np.sqrt(n)),
                delta_xi_worst=worst, weakest=w, frac_translational=frac_nu,
                n_pixels=n, passes=(None if lim is None else bool(worst < lim)))


# ----------------------------------------------------------------------------
# 2. SE(3) flow
# ----------------------------------------------------------------------------

def _hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def se3_exp_matrix(xi, t):
    """4x4 exp(A t) with A = [[-hat(omega), -nu],[0,0]]; xi = (nu, omega)."""
    nu, om = np.asarray(xi[:3], float), np.asarray(xi[3:], float)
    A = np.zeros((4, 4))
    A[:3, :3] = -_hat(om)
    A[:3, 3] = -nu
    from scipy.linalg import expm
    return expm(A * t)


def offset_field(xi, K, Z, tau, T, xidot=None, first_order=False,
                 row_time=0.0):
    """Delta(t_n)(i) for n = 0..T-1, shape (T, H, W, 2), in pixels.

    xidot     : optional (6,) twist derivative -> second-order term (eq. 22).
    first_order : if True use Delta = -t L xi instead of the exact flow, so the
                  truncation error of the linear model can be measured.
    row_time  : rolling-shutter line time alpha (s/row); 0 = global shutter.
                Implements the left action of addendum eq. (30).
    """
    H, W = Z.shape
    f = 0.5 * (K[0, 0] + K[1, 1])
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    du, dv = xx - K[0, 2], yy - K[1, 2]
    ts = (np.arange(T) + 0.5) / T * tau - tau / 2.0          # centred on 0

    if first_order:
        L = interaction_matrix(du, dv, Z, f)                  # (H,W,2,6)
        out = np.empty((T, H, W, 2))
        for n, t in enumerate(ts):
            teff = t + row_time * yy if row_time else t
            xi_t = np.asarray(xi, float)
            d = -np.einsum('hwij,j->hwi', L, xi_t)
            out[n] = d * (teff[..., None] if row_time else t)
        return out

    X = np.stack([du * Z / f, dv * Z / f, Z], -1)             # (H,W,3)
    Xh = np.concatenate([X, np.ones_like(Z)[..., None]], -1)
    base_i = np.stack([xx, yy], -1)
    out = np.empty((T, H, W, 2))

    for n, t in enumerate(ts):
        if not row_time:
            Xt = Xh @ _flow(xi, xidot, -t).T
            out[n] = _project_h(Xt, K) - base_i
            continue
        # Rolling shutter, addendum eq. (30): row r's exposure centre shifts by
        # alpha*r, and exp(-A(t + alpha r)) = exp(-A t) @ exp(-A alpha)^r since
        # A commutes with itself.  One matrix power per row -- one extra scalar
        # in the model, not a new model class.
        M = _flow(xi, xidot, -t)
        step = _flow(xi, xidot, -row_time)
        acc = np.eye(4)
        for r in range(H):
            Xt = Xh[r] @ (M @ acc).T
            out[n, r] = _project_h(Xt, K) - base_i[r]
            acc = acc @ step
    return out


def _flow(xi, xidot, t):
    """exp(A t) with the optional second-order twist term folded in."""
    if xidot is None:
        return se3_exp_matrix(xi, t)
    xi_eff = np.asarray(xi, float) + 0.5 * np.asarray(xidot, float) * t
    return se3_exp_matrix(xi_eff, t)


def _project_h(Xt, K):
    z = np.clip(Xt[..., 2], 1e-6, None)
    return np.stack([K[0, 0] * Xt[..., 0] / z + K[0, 2],
                     K[1, 1] * Xt[..., 1] / z + K[1, 2]], -1)


# ----------------------------------------------------------------------------
# 3. The blur operator
# ----------------------------------------------------------------------------

class PMBM6:
    """v = (1/T) sum_n u(i + Delta_n).  Linear in u; adjoint by autograd."""

    def __init__(self, offsets, device="cpu", dtype=torch.float64):
        self.d = torch.as_tensor(np.asarray(offsets), dtype=dtype, device=device)
        self.T, self.H, self.W, _ = self.d.shape
        yy, xx = torch.meshgrid(torch.arange(self.H, dtype=dtype, device=device),
                                torch.arange(self.W, dtype=dtype, device=device),
                                indexing="ij")
        base = torch.stack([xx, yy], -1)
        g = base[None] + self.d
        gx = 2.0 * g[..., 0] / (self.W - 1) - 1.0
        gy = 2.0 * g[..., 1] / (self.H - 1) - 1.0
        self.grid = torch.stack([gx, gy], -1)                 # (T,H,W,2)

    def forward(self, u):
        """u: (H,W) or (C,H,W) tensor -> blurred image, same shape."""
        squeeze = (u.dim() == 2)
        x = u[None] if squeeze else u
        C = x.shape[0]
        rep = x[None].expand(self.T, C, self.H, self.W)
        out = F.grid_sample(rep, self.grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        out = out.mean(0)
        return out[0] if squeeze else out

    __call__ = forward

    def adjoint(self, w):
        """B^T w, exact to machine precision (MATH_FOUNDATION §5.3)."""
        u = torch.zeros(w.shape, dtype=w.dtype, device=w.device,
                        requires_grad=True)
        (self.forward(u) * w).sum().backward()
        return u.grad

    def local_kernel_pca(self):
        """Per-pixel first-moment blur direction and extent, from the offsets.

        Returns (direction (H,W,2) unit, extent (H,W) in pixels) -- the straight
        segment of addendum §13.2.  This is what a kernel estimator measures.
        """
        d = self.d.cpu().numpy()
        d = d - d.mean(0, keepdims=True)
        # principal direction of the T offsets at each pixel
        cxx = (d[..., 0] ** 2).mean(0)
        cyy = (d[..., 1] ** 2).mean(0)
        cxy = (d[..., 0] * d[..., 1]).mean(0)
        tr, det = cxx + cyy, cxx * cyy - cxy ** 2
        lam = tr / 2 + np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
        vx, vy = cxy, lam - cxx
        nrm = np.hypot(vx, vy) + 1e-12
        ext = (d.max(0) - d.min(0))
        return np.stack([vx / nrm, vy / nrm], -1), np.hypot(ext[..., 0], ext[..., 1])


# ----------------------------------------------------------------------------
# 4. Closed-form twist estimation (addendum §16, eq. 26)
# ----------------------------------------------------------------------------

def estimate_twist_lstsq(displacements, K, Z, tau, stride=8, omega_known=None):
    """Least-squares xi from measured per-pixel displacement (2n,) or (H,W,2).

    omega_known : if given (3,), solves only for nu -- the gyro-constrained
                  problem of addendum §17.
    Returns (xi_hat, covariance).
    """
    Lm = stacked_interaction(K, Z, tau, stride)
    d = np.asarray(displacements)
    if d.ndim == 3:
        d = d[::stride, ::stride].reshape(-1)
    if omega_known is None:
        xi, *_ = np.linalg.lstsq(Lm, d, rcond=None)
        cov = np.linalg.inv(Lm.T @ Lm)
        return xi, cov
    Lv, Lw = Lm[:, :3], Lm[:, 3:]
    nu, *_ = np.linalg.lstsq(Lv, d - Lw @ np.asarray(omega_known), rcond=None)
    return np.concatenate([nu, omega_known]), np.linalg.inv(Lv.T @ Lv)


# ----------------------------------------------------------------------------
# 5. Self-tests
# ----------------------------------------------------------------------------

def selftest_interaction_matrix(seed=0):
    """d/dt Delta|_0 == -L xi  (validates sign convention and eq. 23)."""
    rng = np.random.default_rng(seed)
    K = np.array([[458., 0, 376.], [0, 458., 240.], [0, 0, 1.]])
    Z = np.full((9, 13), 4.0)
    xi = rng.normal(size=6) * np.array([1, 1, 1, .5, .5, .5])
    h = 1e-6
    Xh = _backproject(K, Z)
    d_num = (_project_h(Xh @ se3_exp_matrix(xi, -h).T, K)
             - _project_h(Xh @ se3_exp_matrix(xi, +h).T, K)) / (2 * h)
    f = 0.5 * (K[0, 0] + K[1, 1])
    yy, xx = np.mgrid[0:9, 0:13].astype(float)
    L = interaction_matrix(xx - K[0, 2], yy - K[1, 2], Z, f)
    d_ana = -np.einsum('hwij,j->hwi', L, xi)   # sign convention: dDelta/dt = -L xi
    scale = np.abs(d_ana).max()
    return float(np.abs(d_num - d_ana).max() / scale)


def _backproject(K, Z):
    H, W = Z.shape
    f = 0.5 * (K[0, 0] + K[1, 1])
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    X = np.stack([(xx - K[0, 2]) * Z / f, (yy - K[1, 2]) * Z / f, Z], -1)
    return np.concatenate([X, np.ones_like(Z)[..., None]], -1)


def selftest_adjoint(seed=0, H=64, W=80, T=17):
    """<Bu, w> == <u, B^T w>  (MATH_FOUNDATION §5.3)."""
    rng = np.random.default_rng(seed)
    K = np.array([[458., 0, W / 2], [0, 458., H / 2], [0, 0, 1.]])
    Z = 2.0 + 6.0 * rng.random((H, W))
    xi = np.array([1.0, -0.5, 1.2, 0.4, -0.7, 0.2])
    off = offset_field(xi, K, Z, tau=0.01, T=T)
    B = PMBM6(off)
    u = torch.tensor(rng.normal(size=(H, W)), dtype=torch.float64)
    w = torch.tensor(rng.normal(size=(H, W)), dtype=torch.float64)
    lhs = (B(u) * w).sum().item()
    rhs = (u * B.adjoint(w)).sum().item()
    return abs(lhs - rhs) / max(abs(lhs), 1e-30)


if __name__ == "__main__":
    print(f"interaction matrix vs finite differences : {selftest_interaction_matrix():.3e} (relative)")
    print(f"adjoint <Bu,w> vs <u,B^T w> rel. error   : {selftest_adjoint():.3e}")
