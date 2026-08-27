"""
The exposure-time observability experiment.

Question: if you build a custom model on "trajectory + exposure-time/blur
relation", can tau and the trajectory be separated?

Claim under test (addendum to MATH_FOUNDATION.md):

    RESULT: no.  Rescaling time by alpha (tau -> alpha*tau, and the whole
    trajectory jet xi -> xi/alpha, xidot -> xidot/alpha^2, ...) leaves the
    blurry image BIT-IDENTICAL.  tau is unobservable from one image at every
    order, not merely to first order -- it is a property of the trajectory's
    parameterisation, and the image sees only the pose SET (MATH_FOUNDATION
    §3.1).  Only an external rate (gyro) or a second frame breaks the tie.

Run:  python3 exp_exposure.py
"""
import numpy as np, torch
from pmbm6 import (offset_field, PMBM6, stacked_interaction, admissibility,
                   estimate_twist_lstsq)

rng = np.random.default_rng(7)
H, W, T = 192, 256, 41
K = np.array([[458., 0, W / 2], [0, 458., H / 2], [0, 0, 1.]])
TAU = 0.010


def texture(H, W, seed=3):
    """Broadband 1/f test image with edges -- realistic deblurring statistics."""
    r = np.random.default_rng(seed)
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2); f[0, 0] = 1
    img = np.real(np.fft.ifft2(np.fft.fft2(r.normal(size=(H, W))) / f))
    img = (img - img.min()) / np.ptp(img)
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(8):                                   # hard edges
        a, b, c = r.normal(), r.normal(), r.normal() * 40
        img += 0.25 * ((a * (xx - W / 2) + b * (yy - H / 2) + c) > 0)
    return np.clip((img - img.min()) / np.ptp(img), 0, 1)


U = texture(H, W)
Zmap = 2.0 + 8.0 * (np.mgrid[0:H, 0:W][0] / H)           # 2 - 10 m ramp
u_t = torch.tensor(U, dtype=torch.float64)


def blur(xi, tau, xidot=None, Z=Zmap):
    off = offset_field(xi, K, Z, tau, T, xidot=xidot)
    return PMBM6(off)(u_t).numpy(), off


def psnr(a, b):
    m = np.mean((a - b) ** 2)
    return 99.0 if m < 1e-20 else 10 * np.log10(1.0 / m)


print("=" * 74)
print("EXPERIMENT 1 -- is (tau, xi) separable from a single image?")
print("  render with (tau, xi) and with (a*tau, xi/a): same product tau*xi.")
print("=" * 74)
xi0 = np.array([1.2, -0.6, 0.8, 0.5, -0.9, 0.3])         # m/s ; rad/s
ref, _ = blur(xi0, TAU)
print(f"  reference: tau={TAU*1e3:.0f} ms, |omega|={np.linalg.norm(xi0[3:]):.2f} rad/s")
print(f"  {'alpha':>7} {'tau (ms)':>9} {'PSNR vs ref':>13} {'RMS diff (grey lv)':>20}")
for a in [0.25, 0.5, 2.0, 4.0]:
    alt, _ = blur(xi0 / a, TAU * a)
    rms = np.sqrt(np.mean((alt - ref) ** 2)) * 255
    print(f"  {a:7.2f} {TAU*a*1e3:9.1f} {psnr(alt, ref):13.1f} {rms:20.3f}")
print("  (kernel-measurement noise floor is ~0.5 px; 8-bit quantisation is 1 grey level)")

print()
print("=" * 74)
print("EXPERIMENT 2 -- does intra-exposure ACCELERATION break the tie?")
print("  Correct time rescaling t -> t/alpha requires xi -> xi/alpha AND")
print("  xidot -> xidot/alpha^2 (the whole Taylor jet rescales).  If that is")
print("  bit-identical, tau is unobservable to ALL orders, not just the first.")
print("=" * 74)
print(f"  {'omega_dot':>13} {'consistent jet':>26} {'naive xidot/alpha':>22}")
print(f"  {'':13} {'PSNR':>12} {'RMS':>13} {'PSNR':>10} {'RMS':>11}")
for wd in [0, 50, 300, 1000, 3000]:
    xd = np.array([0, 0, 0, float(wd), 0, 0.0])
    r0, _ = blur(xi0, TAU, xidot=xd)
    ok, _ = blur(xi0 / 2, TAU * 2, xidot=xd / 4)          # consistent: /alpha^2
    bad, _ = blur(xi0 / 2, TAU * 2, xidot=xd / 2)         # naive:      /alpha
    print(f"  {wd:8d} r/s^2 {psnr(ok, r0):12.1f} {np.sqrt(np.mean((ok-r0)**2))*255:12.2e}"
          f" {psnr(bad, r0):10.1f} {np.sqrt(np.mean((bad-r0)**2))*255:11.3f}")
print("  -> the consistent rescaling is bit-identical at every acceleration.")
print("     tau is NOT identifiable from one image, at any order.  It is a")
print("     property of the parameterisation; the image sees only the pose SET.")

print()
print("=" * 74)
print("EXPERIMENT 3 -- the routes to tau that actually exist, with budgets")
print("=" * 74)
f = 458.0
for name, tau, om, fps in [("EuRoC-like  ", 0.010, 2.0, 20),
                           ("fast flight ", 0.005, 6.0, 30),
                           ("racing drone", 0.002, 15.0, 60)]:
    ext = f * om * tau                                   # blur extent, px
    inter = f * om / fps                                 # inter-frame flow, px
    e_gyro = 0.5 / max(ext, 1e-9)                         # blur extent to +-0.5 px
    e_flow = np.hypot(0.5 / max(ext, 1e-9), 1.0 / max(inter, 1e-9))
    print(f"  {name}: tau={tau*1e3:4.1f} ms  blur extent={ext:6.2f} px  "
          f"inter-frame flow={inter:7.2f} px")
    print(f"      (a) gyro-anchored, tau = extent/(f|omega|)  -> {e_gyro*100:5.1f} %")
    print(f"      (b) blur/flow ratio, tau/T = extent/flow    -> {e_flow*100:5.1f} % "
          f" [Korcak & Matas 2023]")
    print(f"      (c) single image, no external rate          ->  UNOBSERVABLE")

print()
print("=" * 74)
print("EXPERIMENT 4 -- admissibility gate and closed-form twist recovery")
print("=" * 74)
for label, Z in [("planar Z=5 m      ", np.full((H, W), 5.0)),
                 ("depth ramp 2-10 m ", Zmap)]:
    a = admissibility(K, Z, TAU, sigma_meas=0.5, stride=8,
                      tol=dict(nu=0.05, omega=0.02))
    print(f"  {label} sigma6_rms={a['sigma6_rms']:6.3f} px  cond={a['cond']:7.1f}  "
          f"n={a['n_pixels']}")
    print(f"      worst-direction precision = {a['delta_xi_worst']:.3f} "
          f"({a['frac_translational']*100:.0f}% translational)  "
          f"{'PASS' if a['passes'] else 'FAIL'} vs tol(0.05 m/s, 0.02 rad/s)")
    Lm = stacked_interaction(K, Z, TAU, 8)
    d = Lm @ xi0
    errs, errs_g = [], []
    for _ in range(300):
        dn = d + rng.normal(0, 0.5, d.shape)
        errs.append(estimate_twist_lstsq(dn, K, Z, TAU, 8)[0] - xi0)
        errs_g.append(estimate_twist_lstsq(dn, K, Z, TAU, 8,
                                           omega_known=xi0[3:])[0] - xi0)
    e, eg = np.array(errs).std(0), np.array(errs_g).std(0)
    print(f"      free 6-DOF  sigma(nu)={np.round(e[:3],3)} m/s  "
          f"sigma(omega)={np.round(e[3:],4)} rad/s")
    print(f"      gyro-fixed  sigma(nu)={np.round(eg[:3],3)} m/s")

print()
print("=" * 74)
print("EXPERIMENT 5 -- rolling shutter costs one scalar (addendum eq. 30)")
print("=" * 74)
gs, _ = blur(xi0, TAU)
for alpha_us in [0, 10, 30, 60]:
    off = offset_field(xi0, K, Zmap, TAU, T, row_time=alpha_us * 1e-6)
    rs = PMBM6(off)(u_t).numpy()
    readout = alpha_us * 1e-6 * H * 1e3
    print(f"  line time {alpha_us:3d} us (frame readout {readout:5.1f} ms): "
          f"PSNR vs global shutter {psnr(rs, gs):5.1f} dB, "
          f"RMS {np.sqrt(np.mean((rs-gs)**2))*255:6.2f} grey levels")
