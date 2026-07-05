"""
drone_motion_blur_run.py

Runnable single-image implementation of the drone motion-blur model:
reads one sharp image (+ optional depth map), simulates combined 6DOF
camera motion blur (rotation + translation + vibration + rolling shutter),
and writes one blurred image.

Usage:
    python drone_motion_blur_run.py --input sharp.jpg --output blurred.jpg \
        [--depth depth.png] [--depth-value 20] [--depth-scale 1.0] \
        [--fx 800] [--fy 800] \
        [--omega 0 0 0] [--velocity 0 0 5] [--exposure 0.002] \
        [--vib-hz 120] [--vib-amp-px 1.5] [--rolling-shutter 1e-5] \
        [--n-samples 24] [--row-block 16] [--noise]

Math + approximations (read before trusting results near strong depth edges):

1. Image formation: B(x) = (1/T) * integral_0^T S(W_t(x)) dt
   -- the blurred image is the average of the sharp image warped through the
   camera trajectory over the exposure window. Approximated here with
   `n_samples` discrete sub-exposure steps (see sample_trajectory()).

2. Rotation: R(t) = Rodrigues(omega * t), the exact solution of the rigid-
   body rotation ODE starting from R(0) = I. Translation: C(t) = velocity*t
   (both assume constant omega/velocity across the exposure -- swap in
   time-varying values from real IMU/gimbal logs if you have them).

3. Depth-aware projection (displacement_field): each pixel is unprojected
   with its own depth, moved by (R, C), reprojected. This is what makes
   translation blur depth-dependent (parallax) while rotation-only blur
   stays depth-independent (homography), matching the framework discussion.

4. Backward-warp approximation: to render the frame at sub-exposure time
   t_i, the "correct" operation is an inverse mapping (solve for source
   pixel given destination pixel). This script instead evaluates the
   forward displacement field at the destination grid coordinates and
   samples the source there -- a standard first-order approximation, valid
   when per-sample displacement is small (true for n_samples >= ~16) but
   increasingly approximate at sharp depth discontinuities. For occlusion-
   correct results at strong depth edges, use a full forward z-buffer warp
   instead (see forward_warp_with_zbuffer in drone_motion_blur_framework.py).

5. Rolling shutter: image is processed in horizontal row-blocks; each block
   gets its own trajectory sampled with a time offset of
   row * row_readout_time (row = block's middle row). Set --row-block 1 for
   an exact per-row shutter model (slower).

6. Sensor noise: Poisson (shot) + Gaussian (read) noise, applied after blur.
"""

import argparse
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Camera / trajectory math
# ---------------------------------------------------------------------------

def rodrigues(omega_vec):
    """3-vector (axis*angle) -> 3x3 rotation matrix."""
    R, _ = cv2.Rodrigues(np.asarray(omega_vec, dtype=np.float64))
    return R


def build_intrinsics(width, height, fx, fy, cx=None, cy=None):
    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy
    return np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)


def sample_trajectory(exposure_time, n_samples, omega, velocity,
                       vib_hz=0.0, vib_amp_px=0.0,
                       row=0, row_readout_time=0.0):
    """
    Returns:
      R_list   : list of n_samples 3x3 rotation matrices, cumulative from t=0
      C_list   : (n_samples, 3) cumulative translation, world/initial-camera units
      vib_list : (n_samples, 2) pixel-space vibration offset (dx, dy)
    """
    t0 = row * row_readout_time
    times = t0 + np.linspace(0.0, exposure_time, n_samples)

    omega = np.asarray(omega, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    R_list = [rodrigues(omega * t) for t in times]
    C_list = np.stack([velocity * t for t in times], axis=0)

    if vib_hz > 0 and vib_amp_px > 0:
        phase = 2 * np.pi * vib_hz * times
        vib_x = vib_amp_px * np.sin(phase)
        vib_y = vib_amp_px * np.sin(phase * 1.3 + 0.7)  # detuned 2nd axis, avoids a pure line
        vib_list = np.stack([vib_x, vib_y], axis=1)
    else:
        vib_list = np.zeros((n_samples, 2))

    return R_list, C_list, vib_list


def displacement_field(K, depth, R, C, row_offset=0):
    """
    For a (h, w) depth block starting at absolute image row `row_offset`,
    compute where each pixel's 3D point projects to after the camera has
    moved by (R, C) relative to t=0.

    Returns (du, dv): destination-minus-source pixel coordinates, shape (h, w).
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u_grid, v_grid = np.meshgrid(
        np.arange(w, dtype=np.float64),
        np.arange(row_offset, row_offset + h, dtype=np.float64))

    Z = np.clip(depth, 1e-3, None)
    X = (u_grid - cx) / fx * Z
    Y = (v_grid - cy) / fy * Z
    P = np.stack([X, Y, Z], axis=-1)          # h x w x 3, point coords at t=0

    P_t = (P - C) @ R                          # point coords in the moved camera frame
    Zt = np.clip(P_t[..., 2], 1e-3, None)

    u_t = fx * P_t[..., 0] / Zt + cx
    v_t = fy * P_t[..., 1] / Zt + cy

    return u_t - u_grid, v_t - v_grid


# ---------------------------------------------------------------------------
# Blur synthesis
# ---------------------------------------------------------------------------

def simulate_motion_blur(img, depth, K, exposure_time, omega, velocity,
                          vib_hz=0.0, vib_amp_px=0.0, row_readout_time=0.0,
                          n_samples=24, row_block=16):
    """
    img   : HxWx3 uint8 sharp image
    depth : HxW float depth map, same physical units as `velocity`
    """
    h, w = depth.shape
    img_f = img.astype(np.float32)
    acc = np.zeros_like(img_f)
    weight = np.zeros((h, w, 1), dtype=np.float32)

    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))

    for row_start in range(0, h, row_block):
        row_end = min(row_start + row_block, h)
        row_mid = (row_start + row_end) // 2

        R_list, C_list, vib_list = sample_trajectory(
            exposure_time, n_samples, omega, velocity,
            vib_hz, vib_amp_px, row=row_mid, row_readout_time=row_readout_time)

        block_depth = depth[row_start:row_end, :]
        block_u = u_grid[row_start:row_end, :]
        block_v = v_grid[row_start:row_end, :]

        for R, C, vib in zip(R_list, C_list, vib_list):
            du, dv = displacement_field(K, block_depth, R, C, row_offset=row_start)
            du = (du + vib[0]).astype(np.float32)
            dv = (dv + vib[1]).astype(np.float32)

            # first-order backward sample: output(u,v) <- source(u - du, v - dv)
            map_x = block_u - du
            map_y = block_v - dv

            sample = cv2.remap(img_f, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)

            valid = ((map_x >= 0) & (map_x < w - 1) &
                     (map_y >= 0) & (map_y < h - 1)).astype(np.float32)[..., None]

            acc[row_start:row_end, :, :] += sample * valid
            weight[row_start:row_end, :, :] += valid

    weight = np.clip(weight, 1e-6, None)
    blurred = acc / weight
    return np.clip(blurred, 0, 255).astype(np.uint8)


def apply_sensor_noise(img, gain=20.0, read_noise_std=2.0):
    """
    Poisson (shot) + Gaussian (read) noise.
    gain: virtual photons-per-pixel-level; higher gain -> relatively less
    shot noise (brighter / longer-effective-exposure regime).
    """
    img_f = np.clip(img.astype(np.float32), 0, None)
    photon_counts = img_f * gain
    shot = np.random.poisson(photon_counts).astype(np.float32) / gain
    noisy = shot + np.random.normal(0, read_noise_std, img_f.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Simulate drone camera motion blur on a single image.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--depth", default=None,
                     help="depth map file (.png/.tif/etc, or .npy raw float array); "
                          "if omitted, a constant depth plane is used")
    ap.add_argument("--depth-value", type=float, default=20.0,
                     help="constant depth (same units as --velocity) if --depth not given")
    ap.add_argument("--depth-scale", type=float, default=1.0,
                     help="multiply loaded depth map values by this to convert to --velocity's units")
    ap.add_argument("--fx", type=float, default=800.0)
    ap.add_argument("--fy", type=float, default=800.0)
    ap.add_argument("--omega", type=float, nargs=3, default=[0, 0, 0], metavar=("WX", "WY", "WZ"),
                     help="rad/s around x y z (camera frame)")
    ap.add_argument("--velocity", type=float, nargs=3, default=[0, 0, 5], metavar=("VX", "VY", "VZ"),
                     help="units/s along x y z (camera frame); +z = toward the scene")
    ap.add_argument("--exposure", type=float, default=1 / 500)
    ap.add_argument("--vib-hz", type=float, default=0.0)
    ap.add_argument("--vib-amp-px", type=float, default=0.0)
    ap.add_argument("--rolling-shutter", type=float, default=0.0, help="sec/row, 0 = global shutter")
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--row-block", type=int, default=16, help="1 = exact per-row rolling shutter (slower)")
    ap.add_argument("--noise", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read input image: {args.input}")
    h, w = img.shape[:2]

    if args.depth:
        if args.depth.lower().endswith(".npy"):
            # raw float array from a model export (e.g. DepthAnything3's
            # `_depth.npy` output) -- no quantization loss, unlike a PNG
            depth = np.load(args.depth).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0] if depth.shape[-1] <= 4 else depth.squeeze()
        else:
            depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"could not read depth map: {args.depth}")
            depth = depth.astype(np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = depth * args.depth_scale
    else:
        depth = np.full((h, w), args.depth_value, dtype=np.float32)

    K = build_intrinsics(w, h, args.fx, args.fy)

    blurred = simulate_motion_blur(
        img, depth, K,
        exposure_time=args.exposure,
        omega=args.omega,
        velocity=args.velocity,
        vib_hz=args.vib_hz,
        vib_amp_px=args.vib_amp_px,
        row_readout_time=args.rolling_shutter,
        n_samples=args.n_samples,
        row_block=args.row_block,
    )

    if args.noise:
        blurred = apply_sensor_noise(blurred)

    cv2.imwrite(args.output, blurred)
    print(f"wrote {args.output}  ({w}x{h}, n_samples={args.n_samples}, exposure={args.exposure}s)")


if __name__ == "__main__":
    main()
