"""
occlusion — the P0 extensions promised but not yet built: occlusion-aware
depth warping, sensor saturation, and noise.

`PMBM6` in pmbm6.py is a GATHER: for each destination pixel i, it samples the
latent image at i + Delta(i). That is correct when depth is smooth, but wrong
at an occlusion boundary -- multiple source points can compete for the same
destination pixel (the near surface should win), and some destination pixels
have no valid source at all (disocclusion holes, revealed background). A
gather can't represent either; it just blends across the boundary, which is
exactly the "soft depth boundary" failure mode flagged in the research plan
as the one place existing theory gives no cover.

This module implements the SCATTER + z-buffer alternative:

    for each time sample t_n:
        for each source pixel j, compute its destination i = pi(...)
        keep the source with smallest depth (nearest wins) at each i
        pixels with no winner are holes
    v = average over t_n of the winner images, renormalised by valid count

Linearity in u is preserved (visibility depends on Z, not u), so PMBM6's
adjoint-by-autograd argument still holds if you differentiate through this;
what's added here is the forward rendering path used to BUILD ground-truth
blurry images for the synthesiser (P1), where getting occlusion right is
what makes the planted ground truth trustworthy.

Also here: the paper's saturation response R(x) (eq. 13), and a basic sensor
noise model (read + shot), both needed for physically-plausible synthesis.
"""
from __future__ import annotations
import numpy as np

__all__ = ["scatter_blur_occluded", "saturation_response", "add_sensor_noise",
          "compare_gather_vs_scatter"]


def _flow_matrix(xi, t):
    from pmbm6 import se3_exp_matrix
    return se3_exp_matrix(xi, t)


def scatter_blur_occluded(u, Z, K, xi, tau, T, xidot=None, bg_value=None):
    """Occlusion-correct forward render.  u, Z: (H,W).  Returns dict with:
        v            -- (H,W) blurred image, holes renormalised
        coverage     -- (H,W) fraction of the T samples with a valid source
                        (0 = fully disoccluded hole, 1 = fully covered)
        hole_mask    -- coverage < 0.5: genuine per-frame disocclusion holes
        z_range      -- (H,W) max - min depth of the WINNING source across the
                        T samples. This is the real occlusion-boundary
                        signature: a destination pixel where the z-buffer
                        winner switches between near and far surfaces at
                        different instants within the exposure. GATHER
                        cannot represent this at all -- it looks up one
                        Z(i) at the destination and uses it for every
                        sample, so it necessarily picks one surface (or an
                        interpolated blend) for the whole exposure.
        boundary_mask -- z_range above a threshold: where winner identity
                        changes during the exposure

    bg_value: fill for fully-uncovered pixels (default: u's own mean).
    """
    H, W = u.shape
    f = 0.5 * (K[0, 0] + K[1, 1])
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    du, dv = xx - K[0, 2], yy - K[1, 2]
    X = np.stack([du * Z / f, dv * Z / f, Z], -1)
    Xh = np.concatenate([X, np.ones_like(Z)[..., None]], -1).reshape(-1, 4)
    u_flat = u.reshape(-1)
    ts = (np.arange(T) + 0.5) / T * tau - tau / 2.0

    accum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    zmin = np.full((H, W), np.inf)
    zmax = np.full((H, W), -np.inf)

    for t in ts:
        xi_eff = xi if xidot is None else (np.asarray(xi, float) + 0.5 * np.asarray(xidot, float) * t)
        M = _flow_matrix(xi_eff, -t)               # source -> observed-frame pose at time t
        Xt = Xh @ M.T
        zc = np.clip(Xt[:, 2], 1e-6, None)          # depth AFTER motion: the z-buffer key
        ix = K[0, 0] * Xt[:, 0] / zc + K[0, 2]
        iy = K[1, 1] * Xt[:, 1] / zc + K[1, 2]
        px, py = np.round(ix).astype(int), np.round(iy).astype(int)
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)

        # z-buffer: process far-to-near so the last (nearest) write wins.
        order = np.argsort(-zc[valid])
        pxi, pyi = px[valid][order], py[valid][order]
        vals = u_flat[valid][order]
        zvals = zc[valid][order]
        flat_dst = pyi * W + pxi

        frame = np.full(H * W, np.nan)
        frame_z = np.full(H * W, np.nan)
        frame[flat_dst] = vals                      # later (nearer) overwrites earlier
        frame_z[flat_dst] = zvals
        frame = frame.reshape(H, W)
        frame_z = frame_z.reshape(H, W)
        covered = ~np.isnan(frame)
        accum[covered] += frame[covered]
        count += covered
        zmin = np.where(covered, np.minimum(zmin, frame_z), zmin)
        zmax = np.where(covered, np.maximum(zmax, frame_z), zmax)

    coverage = count / T
    with np.errstate(invalid='ignore', divide='ignore'):
        v = accum / np.maximum(count, 1)
    fill = float(np.mean(u)) if bg_value is None else bg_value
    v[count == 0] = fill
    z_range = np.where(count > 0, zmax - zmin, 0.0)
    return dict(v=v, coverage=coverage, hole_mask=coverage < 0.5,
                z_range=z_range, boundary_mask=z_range > 0.25 * np.nanmax(z_range + 1e-9))


def saturation_response(x, a=50.0):
    """Eq. (13)/(34): smooth surrogate for sensor clipping.
    R(x) ~ x for x << 1, R(x) -> 1 for x >> 1."""
    return x - (1.0 / a) * np.log1p(np.exp(a * (x - 1.0)))


def add_sensor_noise(v, sigma_read=0.01, shot_gain=0.0, rng=None):
    """Additive Gaussian read noise plus optional signal-dependent shot noise
    (variance proportional to signal, the usual CMOS approximation).
    v assumed in [0,1]-ish range (pre- or post-saturation, either is fine)."""
    rng = rng or np.random.default_rng()
    out = v + rng.normal(0, sigma_read, v.shape)
    if shot_gain > 0:
        out += rng.normal(0, 1, v.shape) * np.sqrt(np.maximum(v, 0) * shot_gain)
    return out


def compare_gather_vs_scatter(u, Z, K, xi, tau, T):
    """Side-by-side: PMBM6's gather vs. the occlusion-correct scatter, on the
    SAME depth-discontinuous scene.  Returns both images plus the pixel-wise
    disagreement map -- this is where occlusion artefacts concentrate, and it
    is the concrete diagnostic for the 'soft depth boundary' risk."""
    import torch
    from pmbm6 import offset_field, PMBM6

    off = offset_field(xi, K, Z, tau, T)
    gather_v = PMBM6(off)(torch.tensor(u, dtype=torch.float64)).numpy()
    scat = scatter_blur_occluded(u, Z, K, xi, tau, T)
    diff = np.abs(gather_v - scat['v'])
    bmask = scat['boundary_mask']
    return dict(gather=gather_v, scatter=scat['v'], coverage=scat['coverage'],
                diff=diff, boundary_mask=bmask,
                diff_at_boundary=float(diff[bmask].mean()) if bmask.any() else 0.0,
                diff_elsewhere=float(diff[~bmask].mean()) if (~bmask).any() else 0.0,
                diff_at_holes=float(diff[scat['hole_mask']].mean())
                if scat['hole_mask'].any() else 0.0)


if __name__ == "__main__":
    # A foreground square (near) sliding over a far background: the canonical
    # occlusion test case. Confirms (a) the scatter method produces visible
    # disocclusion holes where the gather method cannot, (b) the disagreement
    # concentrates at the boundary rather than being uniform, (c) saturation
    # and noise run without blowing up.
    H, W = 96, 128
    K = np.array([[200., 0, W/2], [0, 200., H/2], [0, 0, 1.]])
    yy, xx = np.mgrid[0:H, 0:W]
    u = 0.5 + 0.3 * np.sin(xx / 6.0) * np.cos(yy / 6.0)          # far texture
    Z = np.full((H, W), 8.0)
    fg = (np.abs(xx - W*0.4) < 18) & (np.abs(yy - H*0.5) < 18)
    Z[fg] = 2.0
    u[fg] = 0.9

    # v*tau chosen so the foreground square (Z=2) moves ~15 px -- enough to
    # visibly uncover background behind it -- while the background (Z=8)
    # moves only ~4 px, so the disocclusion trail is unambiguous.
    xi = np.array([6.0, 0, 0, 0, 0, 0])                          # pure lateral pan
    res = compare_gather_vs_scatter(u, Z, K, xi, tau=0.05, T=41)
    print(f"mean |gather - scatter| at occlusion-boundary pixels  : {res['diff_at_boundary']:.4f}")
    print(f"mean |gather - scatter| elsewhere                     : {res['diff_elsewhere']:.4f}")
    print(f"boundary pixels (winner switches fg/bg during exposure): "
          f"{res['boundary_mask'].sum()} / {H*W}")
    ratio = res['diff_at_boundary'] / max(res['diff_elsewhere'], 1e-9)
    print(f"concentration ratio (boundary / elsewhere)             : {ratio:.1f}x")
    assert res['boundary_mask'].sum() > 0, "scene didn't produce any boundary-sweep pixels"
    assert ratio > 3, "expected occlusion disagreement to concentrate at boundaries"

    Rx = saturation_response(np.array([0.0, 0.5, 1.0, 1.5, 3.0]))
    print(f"saturation R([0, .5, 1, 1.5, 3]) = {np.round(Rx, 3)}")

    noisy = add_sensor_noise(res['scatter'], sigma_read=0.01, shot_gain=0.002)
    print(f"noise added, std of (noisy - clean) = {(noisy - res['scatter']).std():.4f}")
    print("\nself-test passed.")
