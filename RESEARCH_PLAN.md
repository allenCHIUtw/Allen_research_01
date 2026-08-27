# Research Plan — 6-DoF Trajectory-Recovery Deblurring for Drones

Your stated plan: *start from the paper's trajectory motion model, add the
relation between exposure time and blur generation, build a custom model.*

The instinct is right and the gap is real. But one result has to be settled
before anything else is built, because it changes what the second ingredient
actually contributes.

---

## 0. The finding that reshapes the plan

> **Exposure time and the trajectory are not two unknowns. They are one.**

Rescale time by any $\alpha>0$:

$$
\tau\mapsto\alpha\tau,\qquad
\xi\mapsto\xi/\alpha,\qquad
\dot\xi\mapsto\dot\xi/\alpha^2,\qquad
\ddot\xi\mapsto\ddot\xi/\alpha^3,\ \dots
$$

The blurry image is **bit-identical**. Not approximately — exactly, and at every
order. It follows directly from §3.1 of your document: $\mathbf{B}$ depends on the
poses only as a weighted *multiset*, and rescaling time permutes nothing and
moves no pose. It only relabels them.

I verified this with the forward model in `pmbm6/`: rendering at
$\alpha\in\{0.25,0.5,2,4\}$ gives RMS difference **0.000 grey levels**, and it
stays bit-identical with angular acceleration up to $3000$ rad/s². (Intra-exposure
acceleration breaking the tie is the natural first guess. It is wrong — $\dot\xi$
rescales too.)

**What this does to your plan.** "The relation between exposure time and blur
generation" is not a source of extra information. It is a *scale constraint*, and
scale must come from outside the image:

| Route to $\tau$ | Requires | Precision ($f{=}458$, kernel to $\pm0.5$ px) |
|---|---|---|
| $\tau = \text{blur extent}/(f\lVert\omega_{\text{IMU}}\rVert)$ | synchronised gyro | $5.5\%$ at $\tau{=}10$ ms; $3.6\%$ at $\tau{=}5$ ms |
| $\tau/T_{\text{frame}} = \text{extent}/\text{inter-frame flow}$ | a second frame | $5.9\%$ / $3.8\%$ |
| single image, no external rate | — | **unobservable** |

This is not bad news. It is the sharpest thing you can say early, it is provable,
and — importantly — **it is a gap in the current literature**, because the closest
prior work assumes $\tau$ known without saying why it must.

---

## 1. Where your plan sits, as of August 2026

I surveyed the current state. Read this before choosing your claim.

| Work | What it does | What it leaves you |
|---|---|---|
| **Blur2Seq** (arXiv:2510.20539) — [code](https://github.com/GuillermoCarbajal/Blur2Seq) | single image → **rotation-only** trajectory + model-based restoration, differentiable PMBM | translation, depth, $\tau$ |
| **Image as an IMU**, ICCV 2025 (arXiv:2503.17358) — [code](https://github.com/jerredchen/image-as-an-imu) | single image → **full 6-DoF velocity** via dense motion flow + mono depth + least squares | **assumes $\tau$ known.** This is your opening — and note their estimator is close to §16 of the addendum, so that part is *not* your novelty |
| **Korčák & Matas** (arXiv:2303.10247) — [code](https://github.com/edavidk7/exposure_fraction_estimation) | shutter angle from $\lVert K\rVert/\lVert f\rVert$ — blur extent over optical flow | **the blur/flow ratio is taken.** Scalar, 2-frame, forensics-motivated; not 6-DoF, not metric, not inertial |
| **BAD-NeRF / BAD-Gaussians / MBA-SLAM** (WU-CVGL) | SE(3)-interpolated exposure integration, multi-frame bundle adjustment | **the SE(3) exposure parameterisation is taken** — do not claim §13 as novel |
| **TRGS-SLAM** (arXiv:2603.20443, 2026) | continuous-time dual B-spline, per-pixel readout time → **RS + blur + IMU jointly** | thermal domain; fixed sensor time constant, no $\tau$ estimation |
| **GAMD** (arXiv:2402.06854) | gyro-derived exposure pose → per-point blur trajectory | rotation-dominant; translation treated as negligible; datasets not locatable |

**Do not claim as novel:** the twist/SE(3) exposure parameterisation; the
interaction-matrix least-squares velocity estimator; the blur-to-flow exposure
ratio. All three are published.

**What is genuinely unclaimed:**

1. The time-scale degeneracy stated and proved, with the observability conditions
   that follow. Nobody states it; "Image as an IMU" silently depends on it.
2. **Joint** estimation of $(\xi,\ \tau,\ t_{\text{offset}})$ rather than assuming
   any of them — with the identifiability argument for why the combination is
   well-posed and the single image is not.
3. A per-frame **admissibility gate** ($\delta\xi_{\text{worst}}$, addendum §15.3)
   computed *before* solving. Everyone reports post-hoc error; nobody predicts
   which frames carry the information.
4. The parallax/depth duality (addendum §18) used constructively — turning the
   term that breaks the homography into the depth estimate.
5. Any of this validated on a **real drone with a rolling shutter**. No published
   work covers that combination, and no dataset supports it (see §4).

### A defensible thesis statement

> *Motion blur is an inertial measurement only up to a time scale. We characterise
> the exact degeneracy, establish the two conditions that resolve it, and build a
> 6-DoF trajectory-recovery deblurring pipeline for drones that estimates exposure
> time and IMU–shutter offset jointly with the trajectory, gated by a
> computable per-frame admissibility criterion.*

That is one paper. Items 4 and 5 are a second.

---

## 2. The model to build

Everything below is implemented in the accompanying `pmbm6/` code.

**State.** Per frame $k$: twist $\xi_k=(\nu_k,\omega_k)\in\mathfrak{se}(3)$,
optionally $\dot\xi_k$. Global per-camera: exposure $\tau$, IMU–shutter offset
$t_{\text{off}}$, rolling-shutter line time $\alpha$. Per frame: depth $Z_k$
(stereo, or from eq. 29).

**Forward operator.**

$$
\Delta^{(t)}(\mathbf{i}) = \pi\Bigl(K\exp\bigl(-A(\xi)\,(t+\alpha r_{\mathbf{i}})\bigr)\,Z(\mathbf{i})K^{-1}\mathbf{i}\Bigr)-\mathbf{i},
\qquad
A(\xi)=\begin{bmatrix}-\widehat{\omega} & -\nu\\ 0 & 0\end{bmatrix}
$$

$$
v(\mathbf{i}) = R\Bigl(\tfrac1T\textstyle\sum_n u\bigl(\mathbf{i}+\Delta^{(t_n)}(\mathbf{i})\bigr)\Bigr)+\varepsilon,
\qquad t_n \text{ uniform on } [-\tfrac\tau2,\tfrac\tau2]
$$

$\mathbf{B}$ stays linear in $\mathbf{u}$, so $\mathbf{B}^\top$ is exact by autograd
(verified: $1.2\times10^{-16}$). $R$ is the saturation surrogate of your §6.

**Objective.**

$$
\min_{\{\mathbf{u}_k\},\{\xi_k\},\tau,t_{\text{off}},Z}
\underbrace{\sum_k\tfrac12\lVert R(\mathbf{B}_k\mathbf{u}_k)-\mathbf{v}_k\rVert^2}_{\text{data}}
+\lambda\sum_k\mathcal{R}(\mathbf{u}_k)
+\underbrace{\gamma\sum_k\lVert\omega_k-\omega_{\text{IMU}}(t_k+t_{\text{off}})\rVert^2_{\Sigma}}_{\text{inertial}}
+\underbrace{\mu\sum_k\lVert\mathcal{W}_{k\to k+1}(\mathbf{u}_k)-\mathbf{u}_{k+1}\rVert^2}_{\text{cross-frame}}
$$

The last two terms are what make the problem well-posed. The inertial term fixes
the time scale (§0) and excludes $\xi\equiv0$ from the feasible set; the
cross-frame term makes $\mathbf{u}_k=\mathbf{v}_k$ not a global optimum (your §7.1's
degeneracy, removed rather than escaped). Solve with linearised ADMM (your §7.3)
in $\mathbf{u}$, Gauss–Newton on $\mathfrak{se}(3)$ for $\xi$, and a coarse grid
plus refinement for $(\tau,t_{\text{off}})$ using the whiteness objective of §10.

**Initialisation.** Closed-form: measure local kernel direction and extent, solve
$\hat\xi=(\mathbb{L}^\top\mathbb{L})^{-1}\mathbb{L}^\top\hat{\mathbf d}$ (addendum
eq. 26). Non-iterative, so it cannot land in the delta-kernel basin.

---

## 3. Data — the honest position

**No existing dataset has real drone blur + sharp ground truth + ground-truth
6-DoF sub-exposure trajectory + logged exposure time.** Plan around that.

| Dataset | Real blur | Sharp GT | IMU | Depth | Shutter | $\tau$ logged | Pose GT | Use it for |
|---|---|---|---|---|---|---|---|---|
| **[Gen3-DroneFlight](https://github.com/uzh-rpg/event-sharp-nerf-drones)** (UZH-RPG 2026) | ✅ | ✅ (few/seq) | ✅ | ✗ (events) | global | ✅ **10/30/50 ms set** | ✅ mocap | **primary real benchmark.** Small; drone ≤2 m/s |
| **[TUM VI](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)** | incidental | ✗ | ✅ HW-sync | ✅ stereo | global | ✅ **per-frame, ns** | partial mocap | **validating the $\tau$ estimator** — the only widely-used set with logged exposure + photometric calib |
| **[EuRoC MAV](https://projects.asl.ethz.ch/datasets/euroc-mav/)** | incidental | ✗ | ✅ | ✅ stereo + Leica scan | global | ✗ **auto-exposure** | ✅ Vicon | trajectory work only. Auto-exposure makes it useless for $\tau$ |
| **[UZH-FPV](https://fpv.ifi.uzh.ch/)** | ✅ heavy, to 100 km/h | ✗ | ✅ | events | global | ✗ | ✅ laser tracker | qualitative stress test at extreme blur |
| **[Blackbird](https://arxiv.org/abs/1810.01987)** | ✗ rendered | ✅ | ✅ | ✅ | n/a | n/a | ✅ mocap | **synthesising blur on real flight dynamics with exact GT** |
| **[GS-Blur](https://github.com/dongwoohhh/GS-Blur)** | ✗ 3DGS | ✅ | ✗ | ✅ | global | n/a | Bézier (release unconfirmed) | controllable synthetic at scale |
| **[Köhler](https://webdav.tuebingen.mpg.de/pixel/benchmark4camerashake/)** | ✅ | ✅ | ✗ | ✗ | global | — | ✅ **true 6-DoF** (robot replay) | the only real GT-trajectory set; tiny, not drone |
| **[BSD](https://github.com/zzh-tech/RSCD)** | ✅ | ✅ | ✗ | ✗ | global | ✅ **1–24 ms** | ✗ | $\tau$-estimator sanity check (what Korčák & Matas used) |

Confirmed: **GoPro, REDS, DVD, RealBlur, RSBlur ship no camera-pose ground truth.**
The Blur2Seq authors say the same — Köhler is the only real set with a GT trajectory.

**Recommended stack, in order:** (1) your own synthetic renderer for controlled
identifiability studies; (2) Blackbird or GS-Blur re-rendering for real flight
dynamics with exact sub-exposure GT; (3) TUM VI to validate $\tau$; (4)
Gen3-DroneFlight as the real drone benchmark; (5) UZH-FPV qualitative.

**The dataset gap is itself a contribution.** A drone flight with a *rolling-shutter*
camera at *fixed, logged* exposure, hardware-synced IMU, stereo, and a
beam-splitter short-exposure reference would be the first of its kind. If you have
access to a flight cage and a mocap system, seriously consider it.

---

## 4. What you need to have in place

**Hardware (if collecting).** Fixed exposure, never auto — this is what makes
EuRoC unusable for your second ingredient. Hardware trigger between camera and
IMU. Stereo or depth. If you want the rolling-shutter result, deliberately use an
RS sensor and read its line time from the datasheet.

**Calibration, in this order.**

1. Intrinsics + lens distortion. Eq. (23) is pinhole; wide drone optics need
   $\mathcal{D}\circ\exp(\hat\xi t)\circ\mathcal{D}^{-1}$ or undistortion first.
2. Camera–IMU extrinsics **and time offset** (Kalibr). At $\omega=2$ rad/s,
   $1$ ms of offset is $0.9$ px — your entire error budget. Treat $t_{\text{off}}$
   as a state, not a constant.
3. Rolling-shutter line time $\alpha$ (Kalibr supports this). At $30\ \mu$s/row
   the difference from a global-shutter model is $5.6$ grey levels RMS — far
   above noise, so it cannot be ignored.
4. Photometric: response curve and vignetting, for the saturation model §6.

**Baselines.** Blur2Seq (public), Image-as-an-IMU (public), Whyte 2010 (MATLAB,
public), MBA-SLAM / BAD-Gaussians (public) for the multi-frame comparison, and
Korčák & Matas (public) as the $\tau$ baseline you must beat or match.

**Validation protocol.**

- Run the **admissibility gate first**. Report $\delta\xi_{\text{worst}}$ per frame
  and exclude — or flag — frames that fail. Nobody does this; it will make your
  error statistics honest and is itself a result.
- Use the **whiteness test** (your §10) as the model-adequacy criterion, and
  respect its one-sidedness: sweep $\lambda$, conclude "inadequate" only if *no*
  $\lambda$ whitens. Report the attribution correlations —
  $\mathrm{corr}(\text{row},\lVert r\rVert)$ for RS/$\tau$ error,
  $\mathrm{corr}(\lVert\nabla v\rVert,\lVert r\rVert)$ for parallax.
- Measure trajectory error in the induced metric
  $\mathbf{G}=\sum\mathbf{L}^\top\mathbf{L}$ (addendum eq. 25), not Euclidean on
  Euler angles.
- Ablate in this order: rotation-only → 6-DoF → 6-DoF + RS → + joint $\tau$.

---

## 5. Milestones, each with a falsifiable check

| # | Work | Falsifiable check | Est. |
|---|---|---|---|
| **M0** | Reproduce the time-scale degeneracy | rescaled render is bit-identical | done — `exp_exposure.py` |
| **M1** | 6-DoF forward model + exact adjoint | $\langle\mathbf{Bu},\mathbf{w}\rangle=\langle\mathbf{u},\mathbf{B}^\top\mathbf{w}\rangle$ to $10^{-16}$ | done — `pmbm6.py` |
| **M2** | Synthetic generator with controllable $(\xi,\tau,\alpha,Z)$ | recover planted $\xi$ within predicted covariance | 2 wk |
| **M3** | **Non-blind** deconvolution with IMU-supplied $\xi$ on EuRoC | does 6-DoF whiten the residual where rotation-only does not? **This is your first real result** — and if rotation-only already whitens on EuRoC, your premise is weaker than assumed and you should know that in month 1 | 4 wk |
| **M4** | Joint $(\tau,t_{\text{off}})$ estimation | error vs TUM VI's logged per-frame exposure; beat/match Korčák & Matas on BSD | 6 wk |
| **M5** | Full blind pipeline vs baselines | PSNR/SSIM on Gen3-DroneFlight + trajectory error in $\mathbf{G}$ | 8 wk |
| **M6** | Rolling shutter | RS-aware model whitens where global-shutter model cannot | 4 wk |

M3 is the decision point. Run it before committing to the rest.

---

## 6. Risks worth naming now

- **$\nu_z$ is the weakest DoF and forward flight is the commonest drone mode.**
  Radial displacement $r v_z\tau/Z$ has no $f$ amplification: $1.2$ px versus
  $9$ px for yaw. Even gyro-constrained, my Monte Carlo gives $\sigma(\nu_z)=0.09$
  m/s against $0.016$ m/s laterally. Do not promise metric forward velocity.
- **Occlusion.** A depth-aware warp is a $z$-buffered scatter; gradients are
  discontinuous at depth edges and holes need renormalising. Linearity in
  $\mathbf{u}$ survives, so the adjoint is fine — but this is the one place your
  existing theory gives no cover.
- **Gen3-DroneFlight is slow ($\le2$ m/s) and small.** It may not exhibit the
  parallax regime your model is built for. Check $Z_{\min}$ against
  $Z^\star=fv\tau$ before relying on it.
- **Blur2Seq has no license file.** Check before building on the code.

---

## 7. What is provided

`pmbm6/pmbm6.py` — 6-DoF projective motion blur model:

- `interaction_matrix` — eq. (23), validated against central differences ($7\times10^{-10}$)
- `offset_field` — exact $SE(3)$ flow, optional $\dot\xi$, optional rolling shutter via eq. (30)
- `PMBM6.forward` / `.adjoint` — grid-sample blur, autograd adjoint exact to $1.2\times10^{-16}$
- `admissibility` — the per-frame gate, $\delta\xi_{\text{worst}}=\sigma_{\text{meas}}/\sigma_{\min}(\mathbb{L})$
- `estimate_twist_lstsq` — closed-form $\xi$, with and without a known gyro
- `PMBM6.local_kernel_pca` — per-pixel blur direction and extent

`pmbm6/exp_exposure.py` — the five observability experiments, reproducing every
number quoted above.

**What I cannot provide:** the data, the drone, and the flight cage. Everything
else above is either in the code or is a decision for you.

---

*Companion documents:* `MATH_FOUNDATION.md` (yours) and
`MATH_FOUNDATION_6DOF.md` (§13–§23, the mathematics behind this plan; §23 is the
degeneracy proof).
