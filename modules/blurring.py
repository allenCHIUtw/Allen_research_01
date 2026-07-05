"""
Unified drone motion-blur simulation framework.

Replaces treating rotation / translation / vibration / rolling-shutter as
four independent, addable effects. Instead: sample the camera's 6DOF pose
over the exposure window (with rolling-shutter row offsets baked in), project
each pixel's depth through that trajectory to get its true image-space path,
then turn that path into a blur via either PSF convolution or warp-and-average.

Everything below is a full implementation of the layered pipeline. Each piece
can be tested in isolation before wiring them into simulate_drone_motion_blur().
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List


# ---------------------------------------------------------------------------
# Shared camera / exposure configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraParams:
    intrinsic_matrix: np.ndarray                 # 3x3 K, shared by every blur type
    dist_coeffs: Optional[np.ndarray] = None      # k1,k2,p1,p2,k3 lens distortion
    row_readout_time: float = 1e-5                # sec/row; 0 = global shutter
    pixel_pitch_um: Optional[float] = None        # for physical-unit conversions


@dataclass
class ExposureParams:
    exposure_time: float = 1 / 500
    n_samples: int = 16                           # sub-exposure temporal samples
    # time-varying motion, so acceleration/gust effects aren't lost to a
    # single constant velocity/omega assumption
    omega_fn: Optional[Callable[[float], np.ndarray]] = None      # t -> 3-vec rad/s
    velocity_fn: Optional[Callable[[float], np.ndarray]] = None   # t -> 3-vec units/s
    vibration_fn: Optional[Callable[[float], np.ndarray]] = None  # t -> (dx,dy) px


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _skew(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix of a 3-vector."""
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]], dtype=np.float64)


def _exp_so3(w: np.ndarray) -> np.ndarray:
    """Exponential map so(3) -> SO(3) via Rodrigues. `w` is an axis-angle 3-vec."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        # first-order approximation is exact enough at the limit
        return np.eye(3) + _skew(w)
    K = _skew(w / theta)
    return (np.eye(3)
            + np.sin(theta) * K
            + (1.0 - np.cos(theta)) * (K @ K))


# ---------------------------------------------------------------------------
# 1. Pose trajectory sampler
#    Ties rotation + translation + vibration + rolling shutter together
#    instead of computing each blur type's displacement independently.
# ---------------------------------------------------------------------------

def sample_pose_trajectory(cam: CameraParams, exp: ExposureParams,
                            row: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build exp.n_samples sub-exposure camera poses over [0, exposure_time].
    If `row` is given, shift the sample window by the rolling-shutter delay
    (row * cam.row_readout_time) so each scanline integrates a different
    slice of the same underlying trajectory.

    Returns: list of (R_i [3x3], t_i [3,], vib_offset_i [2,]) for each sample.
    """
    n = int(exp.n_samples)
    if n < 1:
        n = 1

    # Absolute time offset for this scanline (rolling shutter).
    row_delay = 0.0
    if row is not None:
        row_delay = row * cam.row_readout_time

    # Sub-exposure sample times, midpoint-sampled inside each sub-interval so
    # the average over samples approximates the integral over the window.
    if n == 1:
        rel_times = np.array([exp.exposure_time * 0.5])
    else:
        dt = exp.exposure_time / n
        rel_times = (np.arange(n) + 0.5) * dt   # midpoints of n equal bins

    poses: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # Accumulators for incremental integration.
    R_acc = np.eye(3, dtype=np.float64)
    t_acc = np.zeros(3, dtype=np.float64)
    prev_t = 0.0

    for k in range(n):
        t_rel = float(rel_times[k])
        t_abs = t_rel + row_delay
        step = t_rel - prev_t   # integration step for this sample

        # --- rotation: integrate omega(t) as incremental exponential-map steps
        if exp.omega_fn is not None:
            omega = np.asarray(exp.omega_fn(t_abs), dtype=np.float64).reshape(3)
            dR = _exp_so3(omega * step)
            R_acc = dR @ R_acc
        # else: R stays identity (static)

        # --- translation: integrate velocity(t)
        if exp.velocity_fn is not None:
            vel = np.asarray(exp.velocity_fn(t_abs), dtype=np.float64).reshape(3)
            t_acc = t_acc + vel * step
        # else: t stays zero

        # --- vibration: already a displacement, sample directly (no integrate)
        if exp.vibration_fn is not None:
            vib = np.asarray(exp.vibration_fn(t_abs), dtype=np.float64).reshape(2)
        else:
            vib = np.zeros(2, dtype=np.float64)

        poses.append((R_acc.copy(), t_acc.copy(), vib.copy()))
        prev_t = t_rel

    return poses


# ---------------------------------------------------------------------------
# 2. Per-pixel image-space path from a sampled trajectory
# ---------------------------------------------------------------------------

def project_trajectory_to_pixel_path(cam: CameraParams,
                                      poses: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                      depth_map: np.ndarray,
                                      pixel_xy: Tuple[int, int]) -> np.ndarray:
    """
    For pixel (u, v) at depth Z = depth_map[v, u], project it through every
    sampled pose to get its path in image space. Returns an (n_samples, 2)
    array of (u_i, v_i).

    This is the step that makes rotation + translation + vibration + depth
    physically consistent, instead of three separate shift formulas summed
    after the fact.
    """
    u, v = int(pixel_xy[0]), int(pixel_xy[1])
    K = np.asarray(cam.intrinsic_matrix, dtype=np.float64)
    Kinv = np.linalg.inv(K)

    Z = float(depth_map[v, u])
    if not np.isfinite(Z) or Z <= 0:
        # Degenerate depth: return a static path at the original location.
        return np.tile(np.array([u, v], dtype=np.float64), (len(poses), 1))

    # Unproject (u, v, Z) to a 3D camera-space point.
    # If distortion coeffs are present, undistort the normalized coords first
    # so the back-projection ray corresponds to the true scene direction.
    if cam.dist_coeffs is not None:
        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
        norm = cv2.undistortPoints(pts, K, np.asarray(cam.dist_coeffs, np.float64))
        xn, yn = float(norm[0, 0, 0]), float(norm[0, 0, 1])
        X = np.array([xn * Z, yn * Z, Z], dtype=np.float64)
    else:
        ray = Kinv @ np.array([u, v, 1.0], dtype=np.float64)
        X = ray * Z   # ray already has unit z, so scaling by Z gives depth Z

    path = np.empty((len(poses), 2), dtype=np.float64)
    for i, (R_i, t_i, vib_i) in enumerate(poses):
        # Camera moves by (R_i, t_i); the scene point in the new camera frame:
        Xc = R_i @ X + t_i
        z = Xc[2]
        if abs(z) < 1e-9:
            path[i] = [u, v]
            continue
        proj = K @ Xc
        up = proj[0] / proj[2]
        vp = proj[1] / proj[2]
        # add vibration offset in pixel space after reprojection
        path[i, 0] = up + vib_i[0]
        path[i, 1] = vp + vib_i[1]

    return path


# ---------------------------------------------------------------------------
# 3. PSF kernel from a pixel path (fast, approximate route)
# ---------------------------------------------------------------------------

def build_local_psf(path_uv: np.ndarray, kernel_size: int,
                     sub_pixel: bool = True) -> np.ndarray:
    """
    Rasterize a pixel-space path into a small normalized (sum=1) kernel_size x
    kernel_size PSF, anti-aliased so curved/jittered paths (rotation +
    vibration combined) don't collapse into a straight-line kernel.
    """
    ks = int(kernel_size)
    if ks < 1:
        ks = 1
    if ks % 2 == 0:
        ks += 1   # keep an unambiguous center

    psf = np.zeros((ks, ks), dtype=np.float64)
    center = (ks - 1) / 2.0

    path = np.asarray(path_uv, dtype=np.float64)
    if path.shape[0] == 0:
        psf[int(center), int(center)] = 1.0
        return psf

    # Recenter path around its own centroid, then place at kernel center.
    centroid = path.mean(axis=0)
    local = path - centroid + center   # (N,2) in (x, y) == (col, row)

    def splat(x: float, y: float, w: float):
        """Bilinear splat of weight w at fractional (x=col, y=row)."""
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        x0 = int(np.floor(x)); y0 = int(np.floor(y))
        fx = x - x0; fy = y - y0
        for dy in (0, 1):
            for dx in (0, 1):
                xx = x0 + dx; yy = y0 + dy
                if 0 <= xx < ks and 0 <= yy < ks:
                    wx = (1 - fx) if dx == 0 else fx
                    wy = (1 - fy) if dy == 0 else fy
                    psf[yy, xx] += w * wx * wy

    if local.shape[0] == 1:
        splat(local[0, 0], local[0, 1], 1.0)
    else:
        # Walk each segment between consecutive samples, splatting sub-pixel
        # steps so curved paths keep their shape instead of collapsing.
        for i in range(local.shape[0] - 1):
            p0 = local[i]; p1 = local[i + 1]
            seg = p1 - p0
            length = float(np.hypot(seg[0], seg[1]))
            steps = max(1, int(np.ceil(length * (2 if sub_pixel else 1))))
            for s in range(steps):
                a = (s + 0.5) / steps
                pt = p0 + a * seg
                splat(pt[0], pt[1], 1.0 / steps)

    total = psf.sum()
    if total > 0:
        psf /= total
    else:
        psf[int(center), int(center)] = 1.0
    return psf


def apply_spatially_variant_psf(img: np.ndarray, cam: CameraParams,
                                 exp: ExposureParams, depth_map: np.ndarray,
                                 kernel_size: int = 31,
                                 depth_layers: Optional[int] = 16) -> np.ndarray:
    """
    Fast path: bucket depth_map into `depth_layers` bands, build one
    representative PSF per band (via sample_pose_trajectory +
    project_trajectory_to_pixel_path + build_local_psf at a representative
    pixel/depth), convolve each band with its own kernel, and composite by
    depth (near-to-far) using soft masks at band boundaries.
    """
    img_f = img.astype(np.float64)
    H, W = depth_map.shape[:2]
    n_layers = int(depth_layers) if depth_layers else 1

    valid = np.isfinite(depth_map) & (depth_map > 0)
    if not np.any(valid):
        return img.copy()

    zmin = float(depth_map[valid].min())
    zmax = float(depth_map[valid].max())
    if zmax <= zmin:
        zmax = zmin + 1e-6

    # Quantize depth into bands (linear in depth). Band edges shared for masks.
    edges = np.linspace(zmin, zmax, n_layers + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # A representative pixel: image center, where the path is well-defined.
    rep_u, rep_v = W // 2, H // 2

    # One representative trajectory for the representative scanline. Rolling
    # shutter is applied at the representative row so the band PSFs reflect it.
    poses = sample_pose_trajectory(cam, exp, row=rep_v)

    # Accumulators for near-to-far compositing with soft band boundaries.
    out = np.zeros_like(img_f)
    weight = np.zeros((H, W), dtype=np.float64)

    # Soft mask half-width in depth units (a fraction of band spacing).
    band_w = (edges[1] - edges[0]) if n_layers > 1 else (zmax - zmin + 1e-6)
    feather = 0.5 * band_w + 1e-9

    # Process near-to-far so nearer bands take precedence at soft overlaps.
    order = np.argsort(centers)   # ascending depth = near to far
    for li in order:
        z_rep = float(centers[li])

        # Build a per-band PSF using a depth map that is constant at this band.
        rep_depth = np.full((H, W), z_rep, dtype=np.float64)
        path = project_trajectory_to_pixel_path(cam, poses, rep_depth, (rep_u, rep_v))
        psf = build_local_psf(path, kernel_size)

        blurred = np.empty_like(img_f)
        if img_f.ndim == 2:
            blurred = cv2.filter2D(img_f, -1, psf, borderType=cv2.BORDER_REFLECT)
        else:
            for c in range(img_f.shape[2]):
                blurred[..., c] = cv2.filter2D(img_f[..., c], -1, psf,
                                               borderType=cv2.BORDER_REFLECT)

        # Soft membership: 1 inside the band, feathering across the edges.
        d = np.abs(depth_map - z_rep)
        mask = np.clip(1.0 - (d - 0.5 * band_w) / feather, 0.0, 1.0)
        mask[~valid] = 0.0

        if img_f.ndim == 3:
            m3 = mask[..., None]
            out += blurred * m3
        else:
            out += blurred * mask
        weight += mask

    # Normalize by accumulated weight; fall back to original where no band hit.
    w_safe = np.where(weight > 1e-8, weight, 1.0)
    if img_f.ndim == 3:
        result = out / w_safe[..., None]
    else:
        result = out / w_safe
    fallback = weight <= 1e-8
    if np.any(fallback):
        if img_f.ndim == 3:
            result[fallback] = img_f[fallback]
        else:
            result[fallback] = img_f[fallback]

    return np.clip(result, 0, 255).astype(img.dtype) if img.dtype == np.uint8 else result


# ---------------------------------------------------------------------------
# 4. Occlusion-aware forward warp (slower, more correct route)
#    Needed because depth-varying translation causes parallax: near objects
#    reveal/hide background differently than far ones.
# ---------------------------------------------------------------------------

def forward_warp_with_zbuffer(img: np.ndarray, depth_map: np.ndarray,
                               cam: CameraParams, R: np.ndarray,
                               t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp img for one sub-exposure pose (R, t), splatting each source pixel to
    its new location and resolving overlaps with a z-buffer test so nearer
    surfaces correctly occlude farther ones. Prevents the ghosting/tearing a
    naive per-pixel shift-and-average produces at depth discontinuities.

    Returns: (warped_img, valid_mask)
    """
    H, W = depth_map.shape[:2]
    K = np.asarray(cam.intrinsic_matrix, dtype=np.float64)
    Kinv = np.linalg.inv(K)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    img_f = img.astype(np.float64)
    channels = 1 if img_f.ndim == 2 else img_f.shape[2]

    # Build a grid of all pixel coordinates.
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    u = uu.ravel().astype(np.float64)
    v = vv.ravel().astype(np.float64)
    Z = depth_map.ravel().astype(np.float64)

    good = np.isfinite(Z) & (Z > 0)

    # Unproject every valid pixel with its own depth to 3D.
    rays = Kinv @ np.vstack([u, v, np.ones_like(u)])   # (3, N)
    X = rays * Z[None, :]                               # (3, N) camera-space

    # Transform by (R, t).
    Xc = R @ X + t[:, None]                             # (3, N)
    zc = Xc[2]
    good = good & (zc > 1e-9)

    proj = K @ Xc
    ud = proj[0] / proj[2]
    vd = proj[1] / proj[2]

    ui = np.round(ud).astype(np.int64)
    vi = np.round(vd).astype(np.int64)
    inb = good & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    warped = np.zeros((H, W, channels), dtype=np.float64)
    zbuf = np.full((H, W), np.inf, dtype=np.float64)
    valid = np.zeros((H, W), dtype=bool)

    src_idx = np.nonzero(inb)[0]
    dst_u = ui[inb]
    dst_v = vi[inb]
    dst_z = zc[inb]

    src_colors = img_f.reshape(-1, channels)[src_idx]

    # Z-buffer splat: for each destination pixel keep the nearest source depth.
    # Sort by depth descending so nearer writes land last and win.
    order = np.argsort(-dst_z)
    du = dst_u[order]; dv = dst_v[order]; dz = dst_z[order]; dc = src_colors[order]

    warped[dv, du, :] = dc
    zbuf[dv, du] = dz
    valid[dv, du] = True

    # Fill disocclusion holes: inpaint the invalid regions from neighbors.
    hole = ~valid
    if np.any(hole):
        mask_u8 = (hole.astype(np.uint8)) * 255
        base = warped
        # cv2.inpaint needs 8-bit or 3-channel; normalize per-channel.
        filled = np.empty_like(base)
        for c in range(channels):
            ch = base[..., c]
            lo, hi = float(ch.min()), float(ch.max())
            scale = (hi - lo) if hi > lo else 1.0
            ch8 = np.clip((ch - lo) / scale * 255.0, 0, 255).astype(np.uint8)
            inp = cv2.inpaint(ch8, mask_u8, 3, cv2.INPAINT_TELEA)
            filled[..., c] = inp.astype(np.float64) / 255.0 * scale + lo
        warped = np.where(hole[..., None], filled, warped)

    if channels == 1:
        warped = warped[..., 0]

    if img.dtype == np.uint8:
        warped = np.clip(warped, 0, 255).astype(np.uint8)

    return warped, valid


def blur_via_warp_average(img: np.ndarray, depth_map: np.ndarray,
                           cam: CameraParams, exp: ExposureParams,
                           row_block: int = 1) -> np.ndarray:
    """
    Slow/accurate path: for each row-block, get its rolling-shutter-shifted
    trajectory via sample_pose_trajectory(cam, exp, row=...), forward-warp
    the sharp image at each sub-exposure pose via forward_warp_with_zbuffer,
    and average the n_samples warped frames for that row-block.
    """
    H, W = depth_map.shape[:2]
    img_f = img.astype(np.float64)
    channels = 1 if img_f.ndim == 2 else img_f.shape[2]

    out = np.zeros_like(img_f)

    for r0 in range(0, H, row_block):
        r1 = min(r0 + row_block, H)
        rep_row = (r0 + r1 - 1) // 2   # representative scanline for this block

        poses = sample_pose_trajectory(cam, exp, row=rep_row)

        # Accumulate warped frames and per-pixel valid counts for this block.
        accum = np.zeros((H, W, channels), dtype=np.float64) if channels > 1 \
            else np.zeros((H, W), dtype=np.float64)
        count = np.zeros((H, W), dtype=np.float64)

        for (R_i, t_i, vib_i) in poses:
            warped, valid = forward_warp_with_zbuffer(img, depth_map, cam, R_i, t_i)
            wf = warped.astype(np.float64)

            # Apply the whole-frame pixel vibration offset for this sample.
            if np.any(vib_i):
                Mv = np.array([[1, 0, vib_i[0]], [0, 1, vib_i[1]]], dtype=np.float64)
                if channels > 1:
                    wf = cv2.warpAffine(wf, Mv, (W, H), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)
                else:
                    wf = cv2.warpAffine(wf, Mv, (W, H), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)

            if channels > 1:
                accum += wf * valid[..., None]
            else:
                accum += wf * valid
            count += valid.astype(np.float64)

        # Divide by valid sample count per pixel (handle holes -> fall back).
        c_safe = np.where(count > 0, count, 1.0)
        if channels > 1:
            block_avg = accum / c_safe[..., None]
        else:
            block_avg = accum / c_safe
        empty = count <= 0
        if np.any(empty):
            block_avg[empty] = img_f[empty]

        # Stitch this row-block back into the full frame.
        out[r0:r1] = block_avg[r0:r1]

    if img.dtype == np.uint8:
        return np.clip(out, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# 5. Sensor noise / exposure-level realism
# ---------------------------------------------------------------------------

def apply_sensor_noise(img: np.ndarray, exposure_time: float,
                        iso_gain: float = 1.0,
                        read_noise_std: float = 1.0) -> np.ndarray:
    """
    Add shot noise (signal-dependent, ~Poisson) and read noise (Gaussian),
    and renormalize brightness for the virtual exposure time so a longer
    integration window doesn't just look like a scaled-up clean image.
    """
    orig_dtype = img.dtype
    max_val = 255.0 if orig_dtype == np.uint8 else float(max(1.0, img.max()))

    img_f = img.astype(np.float64)

    # Convert to a photon-count-like scale. Longer exposure -> more photons,
    # then we renormalize back so brightness stays comparable while noise
    # statistics reflect the true integration.
    # Reference exposure keeps the scene near its original brightness.
    ref_exposure = 1.0 / 500.0
    exposure_scale = exposure_time / ref_exposure if ref_exposure > 0 else 1.0

    # photons ~ signal * gain * exposure_scale (in arbitrary electrons)
    photon_scale = iso_gain * exposure_scale
    # Map intensity [0, max] to an electron count with a reasonable full-well.
    full_well = 10000.0
    electrons = (img_f / max_val) * full_well * photon_scale
    electrons = np.clip(electrons, 0, None)

    # Poisson shot noise on the electron scale.
    rng = np.random.default_rng()
    noisy = rng.poisson(electrons).astype(np.float64)

    # Gaussian read noise (in electrons).
    noisy += rng.normal(0.0, read_noise_std, size=noisy.shape)

    # Convert back to intensity, undoing the exposure/gain scaling so the
    # mean brightness is preserved (noise character is what changes).
    back = noisy / (full_well * photon_scale) * max_val
    back = np.clip(back, 0, max_val)

    if orig_dtype == np.uint8:
        return np.round(back).astype(np.uint8)
    return back.astype(orig_dtype)


# ---------------------------------------------------------------------------
# 6. Unified orchestrator — main entry point replacing the four
#    independent blur functions
# ---------------------------------------------------------------------------

def simulate_drone_motion_blur(img: np.ndarray, depth_map: np.ndarray,
                                cam: CameraParams, exp: ExposureParams,
                                method: str = "psf",
                                add_noise: bool = True) -> np.ndarray:
    """
    method="psf"  -> apply_spatially_variant_psf   (fast, approximate)
    method="warp" -> blur_via_warp_average          (slow, occlusion-correct)
    """
    if method == "psf":
        blurred = apply_spatially_variant_psf(img, cam, exp, depth_map)
    elif method == "warp":
        blurred = blur_via_warp_average(img, depth_map, cam, exp)
    else:
        raise ValueError(f"Unknown method: {method!r} (expected 'psf' or 'warp')")

    if add_noise:
        blurred = apply_sensor_noise(blurred, exp.exposure_time)

    return blurred


# ---------------------------------------------------------------------------
# 7. Validation harness — run before trusting the pipeline on real footage
# ---------------------------------------------------------------------------

def validate_known_case(cam: CameraParams) -> None:
    """
    Sanity check: render a single point at known depth Z, apply pure lateral
    translation velocity v with no rotation/vibration, and confirm the
    resulting blur length in pixels matches the analytic prediction:

        blur_px ~= f_pixels * v / Z * exposure_time
    """
    H, W = 200, 200
    Z = 10.0                     # depth in world units
    v = 2.0                      # lateral velocity (world units / s), +x
    exposure_time = 1.0 / 100.0

    f_pixels = float(cam.intrinsic_matrix[0, 0])
    analytic = f_pixels * v / Z * exposure_time

    # Single bright point at image center on a black background.
    img = np.zeros((H, W), dtype=np.float64)
    cx = int(round(cam.intrinsic_matrix[0, 2]))
    cy = int(round(cam.intrinsic_matrix[1, 2]))
    cx = min(max(cx, 0), W - 1)
    cy = min(max(cy, 0), H - 1)
    img[cy, cx] = 255.0

    depth_map = np.full((H, W), Z, dtype=np.float64)

    exp = ExposureParams(
        exposure_time=exposure_time,
        n_samples=32,
        omega_fn=None,
        velocity_fn=lambda t: np.array([v, 0.0, 0.0]),
        vibration_fn=None,
    )

    # Use the warp path (occlusion-correct, exact for a translating point).
    out = simulate_drone_motion_blur(img, depth_map, cam, exp,
                                     method="warp", add_noise=False)

    out_f = out.astype(np.float64)
    thresh = 0.05 * out_f.max() if out_f.max() > 0 else 0.5
    ys, xs = np.nonzero(out_f > thresh)
    if xs.size == 0:
        raise AssertionError("No blur streak detected in output.")

    measured = float(xs.max() - xs.min())

    print(f"Analytic blur length : {analytic:.3f} px")
    print(f"Measured blur length : {measured:.3f} px")

    # Allow tolerance for rasterization / streak-endpoint quantization.
    tol = max(2.0, 0.25 * analytic)
    assert abs(measured - analytic) <= tol, (
        f"Blur length mismatch: measured {measured:.2f}, "
        f"expected ~{analytic:.2f} (tol {tol:.2f})"
    )
    print("validate_known_case PASSED")


if __name__ == "__main__":
    K = np.array([[500.0, 0.0, 100.0],
                  [0.0, 500.0, 100.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    cam = CameraParams(intrinsic_matrix=K, row_readout_time=1e-5)
    validate_known_case(cam)