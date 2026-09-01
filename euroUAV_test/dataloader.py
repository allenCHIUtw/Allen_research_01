"""
dataloader.py
=============
EuRoC MAV (ASL format) loader built for 6-DoF motion-deblurring experiments.

Targeted at ``V1_02_medium`` and ``V1_03_difficult`` (Vicon Room 1) -- the two
sequences that ship a Leica point cloud alongside full 6-DoF ground truth.


Design notes
------------
Timestamps
    Everything is ``int64`` **nanoseconds**. Never cast these to float32, and be
    careful even with float64: EuRoC absolute timestamps are ~1.4e18 ns, which is
    past float64's exact-integer range (2^53 ~ 9e15), so absolute ns in float64
    quantise to ~256 ns. Every routine here subtracts a reference epoch before
    going to float, so precision stays at the sub-nanosecond level. This matters
    because camera/IMU time-offset calibration for deblurring needs ~50-150 us.

Time offset
    First-class parameter, not a baked-in constant. Convention::

        t_imu_equivalent = t_cam + time_offset_ns

    A positive offset means camera timestamps run *early* relative to the IMU
    clock. On EuRoC the true value is ~0 (hardware sync), which makes it a good
    place to validate an offset estimator by injecting a known value.

Shutter
    EuRoC is GLOBAL SHUTTER (Aptina MT9V034). There is no line delay. A
    ``line_delay_ns`` argument exists so you can *synthesise* rolling shutter for
    experiments, but it defaults to 0.

Exposure
    EuRoC does not log exposure time, and the VI-sensor runs independent
    auto-exposure per camera (so cam0 and cam1 have different blur magnitudes for
    the same motion). ``exposure_ns`` is therefore always something you supply or
    estimate. There is no ground truth for it.

Frame timestamp semantics
    ASL describes the rig as using "shutter-centric temporal alignment", so the
    default exposure anchor here is ``"mid"``. If your residuals show a systematic
    half-exposure bias, try ``"start"``.


Quick start
-----------
    from dataloader import load_sequence

    seq = load_sequence("/data/EuRoC/V1_02_medium")
    print(seq.summary())

    img   = seq.load_image(120)
    Hs    = seq.blur_homographies(120, exposure_ns=10_000_000, n_samples=24)
    stats = seq.parallax_budget(120, exposure_ns=10_000_000, z_min=1.5)
"""

from __future__ import annotations

import os
import csv
import glob
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Dict, Any, List

import numpy as np

try:
    import yaml
except ImportError as _e:  # pragma: no cover
    raise ImportError("dataloader.py needs PyYAML: pip install pyyaml") from _e

try:
    from scipy.spatial.transform import Rotation, Slerp
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


NS = 1_000_000_000  # nanoseconds per second


# --------------------------------------------------------------------------- #
# Small SO(3) helpers
# --------------------------------------------------------------------------- #

def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """(...,4) quaternion in EuRoC's [w,x,y,z] order -> (...,3,3) rotation."""
    q = np.asarray(q, dtype=np.float64)
    if _HAVE_SCIPY:
        flat = q.reshape(-1, 4)
        xyzw = flat[:, [1, 2, 3, 0]]  # EuRoC is wxyz; scipy wants xyzw
        R = Rotation.from_quat(xyzw).as_matrix()
        return R.reshape(q.shape[:-1] + (3, 3))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: (...,3) rotation vector -> (...,3,3)."""
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w, axis=-1, keepdims=True)
    small = theta < 1e-12
    axis = np.where(small, 0.0, w / np.where(small, 1.0, theta))
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -axis[..., 2], axis[..., 1]
    K[..., 1, 0], K[..., 1, 2] = axis[..., 2], -axis[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -axis[..., 1], axis[..., 0]
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    s = np.sin(theta)[..., None]
    c = (1 - np.cos(theta))[..., None]
    return I + s * K + c * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,3) rotation vector."""
    if _HAVE_SCIPY:
        flat = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
        return Rotation.from_matrix(flat).as_rotvec().reshape(R.shape[:-2] + (3,))
    R = np.asarray(R, dtype=np.float64)
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    theta = np.arccos(cos)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2],
                  R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    denom = 2 * np.sin(theta)
    scale = np.where(np.abs(denom) < 1e-12, 0.5, theta / np.where(np.abs(denom) < 1e-12, 1.0, denom))
    return v * scale[..., None]


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a (...,4,4) homogeneous transform."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Ti = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -np.einsum('...ij,...j->...i', Rt, t)
    Ti[..., 3, 3] = 1.0
    return Ti


# --------------------------------------------------------------------------- #
# Calibration containers
# --------------------------------------------------------------------------- #

@dataclass
class CameraCalib:
    name: str
    resolution: Tuple[int, int]          # (width, height)
    rate_hz: float
    K: np.ndarray                        # 3x3
    dist: np.ndarray                     # radial-tangential [k1,k2,p1,p2]
    T_BS: np.ndarray                     # 4x4, body -> sensor(camera)
    camera_model: str
    distortion_model: str

    @property
    def fx(self) -> float: return float(self.K[0, 0])
    @property
    def fy(self) -> float: return float(self.K[1, 1])
    @property
    def cx(self) -> float: return float(self.K[0, 2])
    @property
    def cy(self) -> float: return float(self.K[1, 2])
    @property
    def width(self) -> int: return int(self.resolution[0])
    @property
    def height(self) -> int: return int(self.resolution[1])

    def undistort_maps(self):
        """cv2 remap tables for this camera. Requires opencv."""
        import cv2
        return cv2.initUndistortRectifyMap(
            self.K, self.dist, np.eye(3), self.K,
            (self.width, self.height), cv2.CV_32FC1)


@dataclass
class ImuCalib:
    rate_hz: float
    T_BS: np.ndarray
    gyro_noise_density: float
    gyro_random_walk: float
    accel_noise_density: float
    accel_random_walk: float


def _read_sensor_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_camera_calib(cam_dir: str) -> CameraCalib:
    y = _read_sensor_yaml(os.path.join(cam_dir, "sensor.yaml"))
    intr = np.asarray(y["intrinsics"], dtype=np.float64)
    K = np.array([[intr[0], 0.0, intr[2]],
                  [0.0, intr[1], intr[3]],
                  [0.0, 0.0, 1.0]])
    T_BS = np.asarray(y["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    return CameraCalib(
        name=os.path.basename(cam_dir.rstrip("/")),
        resolution=tuple(int(v) for v in y["resolution"]),
        rate_hz=float(y.get("rate_hz", 20)),
        K=K,
        dist=np.asarray(y.get("distortion_coefficients", [0, 0, 0, 0]), dtype=np.float64),
        T_BS=T_BS,
        camera_model=str(y.get("camera_model", "pinhole")),
        distortion_model=str(y.get("distortion_model", "radial-tangential")),
    )


def load_imu_calib(imu_dir: str) -> ImuCalib:
    y = _read_sensor_yaml(os.path.join(imu_dir, "sensor.yaml"))
    T_BS = np.asarray(y["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    return ImuCalib(
        rate_hz=float(y.get("rate_hz", 200)),
        T_BS=T_BS,
        gyro_noise_density=float(y.get("gyroscope_noise_density", np.nan)),
        gyro_random_walk=float(y.get("gyroscope_random_walk", np.nan)),
        accel_noise_density=float(y.get("accelerometer_noise_density", np.nan)),
        accel_random_walk=float(y.get("accelerometer_random_walk", np.nan)),
    )


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #

def _load_csv_numeric(path: str) -> np.ndarray:
    """Load an ASL data.csv (one '#' header line) as float64, keeping col 0 exact."""
    rows: List[List[str]] = []
    with open(path, "r", newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([c for c in line.split(",") if c != ""])
    if not rows:
        raise ValueError(f"no data rows in {path}")
    return rows


def _split_ts_and_values(rows: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    t_ns = np.array([int(r[0]) for r in rows], dtype=np.int64)
    vals = np.array([[float(c) for c in r[1:]] for r in rows], dtype=np.float64)
    order = np.argsort(t_ns, kind="stable")
    return t_ns[order], vals[order]


@dataclass
class ImuData:
    t_ns: np.ndarray      # (N,) int64
    gyro: np.ndarray      # (N,3) rad/s, IMU frame
    accel: np.ndarray     # (N,3) m/s^2, IMU frame
    calib: ImuCalib


def load_imu(imu_dir: str) -> ImuData:
    rows = _load_csv_numeric(os.path.join(imu_dir, "data.csv"))
    t_ns, v = _split_ts_and_values(rows)
    if v.shape[1] < 6:
        raise ValueError(f"imu0/data.csv expected 6 value columns, got {v.shape[1]}")
    return ImuData(t_ns=t_ns, gyro=v[:, 0:3], accel=v[:, 3:6], calib=load_imu_calib(imu_dir))


@dataclass
class GroundTruth:
    """state_groundtruth_estimate0: 17 columns, ~200 Hz, aligned to the sensor clock."""
    t_ns: np.ndarray      # (N,)
    p: np.ndarray         # (N,3) body position in world
    q_wxyz: np.ndarray    # (N,4) body orientation, world<-body
    v: np.ndarray         # (N,3) body velocity in world
    bg: np.ndarray        # (N,3) gyro bias, IMU frame
    ba: np.ndarray        # (N,3) accel bias, IMU frame

    @property
    def R(self) -> np.ndarray:
        if not hasattr(self, "_R"):
            object.__setattr__(self, "_R", quat_wxyz_to_matrix(self.q_wxyz))
        return self._R

    def gaps(self, max_gap_ns: int) -> np.ndarray:
        """(M,2) array of [start_ns, end_ns] intervals where GT is missing."""
        d = np.diff(self.t_ns)
        idx = np.nonzero(d > max_gap_ns)[0]
        return np.stack([self.t_ns[idx], self.t_ns[idx + 1]], axis=1) if idx.size else np.zeros((0, 2), np.int64)

    def covers(self, t_ns: np.ndarray, max_gap_ns: int) -> np.ndarray:
        """Boolean mask: which query times sit inside a well-sampled GT span."""
        t_ns = np.atleast_1d(np.asarray(t_ns, dtype=np.int64))
        ok = (t_ns >= self.t_ns[0]) & (t_ns <= self.t_ns[-1])
        for lo, hi in self.gaps(max_gap_ns):
            ok &= ~((t_ns > lo) & (t_ns < hi))
        return ok

    def interpolate(self, t_ns) -> Dict[str, np.ndarray]:
        """Linear on p/v/bg/ba, SLERP on orientation. Times are clamped to range."""
        t_ns = np.atleast_1d(np.asarray(t_ns, dtype=np.int64))
        epoch = self.t_ns[0]
        ts = (self.t_ns - epoch).astype(np.float64) / NS
        tq = np.clip((t_ns - epoch).astype(np.float64) / NS, ts[0], ts[-1])

        out = {k: np.stack([np.interp(tq, ts, getattr(self, k)[:, i]) for i in range(3)], axis=-1)
               for k in ("p", "v", "bg", "ba")}

        if _HAVE_SCIPY:
            xyzw = self.q_wxyz[:, [1, 2, 3, 0]]
            out["R"] = Slerp(ts, Rotation.from_quat(xyzw))(tq).as_matrix()
        else:
            j = np.clip(np.searchsorted(ts, tq, side="right") - 1, 0, len(ts) - 2)
            a = ((tq - ts[j]) / (ts[j + 1] - ts[j]))[:, None]
            R0, R1 = self.R[j], self.R[j + 1]
            out["R"] = R0 @ so3_exp(a * so3_log(np.swapaxes(R0, -1, -2) @ R1))

        out["T_WB"] = se3(out["R"], out["p"])
        out["t_ns"] = t_ns
        return out


def load_groundtruth(gt_dir: str) -> GroundTruth:
    rows = _load_csv_numeric(os.path.join(gt_dir, "data.csv"))
    t_ns, v = _split_ts_and_values(rows)
    if v.shape[1] < 16:
        raise ValueError(
            f"state_groundtruth_estimate0 expected 17 columns total, got {v.shape[1] + 1}")
    return GroundTruth(t_ns=t_ns, p=v[:, 0:3], q_wxyz=v[:, 3:7],
                       v=v[:, 7:10], bg=v[:, 10:13], ba=v[:, 13:16])


def load_frame_index(cam_dir: str) -> Tuple[np.ndarray, List[str]]:
    """Timestamps and absolute image paths, sorted. Falls back to globbing data/."""
    csv_path = os.path.join(cam_dir, "data.csv")
    data_dir = os.path.join(cam_dir, "data")
    if os.path.exists(csv_path):
        rows = _load_csv_numeric(csv_path)
        t_ns = np.array([int(r[0]) for r in rows], dtype=np.int64)
        names = [r[1].strip() for r in rows]
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        t_ns = np.array([int(os.path.splitext(os.path.basename(f))[0]) for f in files], dtype=np.int64)
        names = [os.path.basename(f) for f in files]
    order = np.argsort(t_ns, kind="stable")
    t_ns = t_ns[order]
    paths = [os.path.join(data_dir, names[i]) for i in order]
    return t_ns, paths


def load_pointcloud(root: str) -> Optional[np.ndarray]:
    """(N,3) Leica scan from pointcloud0/data.ply. Vicon Room sequences only."""
    for cand in (os.path.join(root, "mav0", "pointcloud0", "data.ply"),
                 os.path.join(root, "pointcloud0", "data.ply")):
        if os.path.exists(cand):
            return _read_ply_xyz(cand)
    return None


def _read_ply_xyz(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header, n_vertex, fmt, props = [], 0, "ascii", []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_vertex = int(line.split()[2])
            elif line.startswith("property") and len(header) and n_vertex:
                props.append(line.split())
            elif line == "end_header":
                break
        if fmt == "ascii":
            pts = np.empty((n_vertex, 3), np.float64)
            for i in range(n_vertex):
                pts[i] = [float(x) for x in f.readline().split()[:3]]
            return pts
        dtmap = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
                 "uchar": "u1", "uint8": "u1", "int": "i4", "int32": "i4",
                 "short": "i2", "ushort": "u2"}
        endian = "<" if "little" in fmt else ">"
        dt = np.dtype([(p[2], endian + dtmap.get(p[1], "f4")) for p in props])
        arr = np.frombuffer(f.read(n_vertex * dt.itemsize), dtype=dt, count=n_vertex)
        return np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float64)


# --------------------------------------------------------------------------- #
# Sequence
# --------------------------------------------------------------------------- #

@dataclass
class EuRoCSequence:
    root: str
    cam_name: str
    cam: CameraCalib
    cam_other: CameraCalib
    imu: ImuData
    gt: GroundTruth
    frame_t_ns: np.ndarray
    frame_paths: List[str]
    time_offset_ns: int = 0
    max_gt_gap_ns: int = 50_000_000      # 50 ms; GT is ~200 Hz so this is generous
    _pc: Optional[np.ndarray] = field(default=None, repr=False)

    # ---- basics ---------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.frame_t_ns)

    @property
    def T_BS_cam(self) -> np.ndarray:
        return self.cam.T_BS

    @property
    def T_cam0_cam1(self) -> np.ndarray:
        """Stereo extrinsic (~0.11 m baseline), regardless of which cam is primary."""
        c0 = self.cam if self.cam_name == "cam0" else self.cam_other
        c1 = self.cam_other if self.cam_name == "cam0" else self.cam
        return se3_inv(c0.T_BS) @ c1.T_BS

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.T_cam0_cam1[:3, 3]))

    @property
    def pointcloud(self) -> Optional[np.ndarray]:
        if self._pc is None:
            object.__setattr__(self, "_pc", load_pointcloud(self.root))
        return self._pc

    def load_image(self, i: int) -> np.ndarray:
        """uint8 (H,W) grayscale."""
        path = self.frame_paths[i]
        try:
            import cv2
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"cv2 could not read {path}")
            return img
        except ImportError:
            from PIL import Image
            return np.array(Image.open(path))

    # ---- time ------------------------------------------------------------ #

    def cam_to_imu_time(self, t_ns) -> np.ndarray:
        """Apply the camera->IMU clock offset. t_imu = t_cam + time_offset_ns."""
        return np.atleast_1d(np.asarray(t_ns, dtype=np.int64)) + np.int64(self.time_offset_ns)

    def with_time_offset(self, time_offset_ns: int) -> "EuRoCSequence":
        """Cheap copy with a different offset -- for sweeping during calibration."""
        import copy
        s = copy.copy(self)
        s.time_offset_ns = int(time_offset_ns)
        return s

    def exposure_window(self, i: int, exposure_ns: int,
                        anchor: str = "mid") -> Tuple[int, int]:
        """(t_start, t_end) in IMU-clock nanoseconds for frame i."""
        t = int(self.cam_to_imu_time(self.frame_t_ns[i])[0])
        e = int(exposure_ns)
        if anchor == "mid":
            return t - e // 2, t - e // 2 + e
        if anchor == "start":
            return t, t + e
        if anchor == "end":
            return t - e, t
        raise ValueError("anchor must be 'mid', 'start' or 'end'")

    def valid_frames(self, exposure_ns: int = 0, anchor: str = "mid") -> np.ndarray:
        """Indices whose whole exposure window is covered by gap-free ground truth."""
        idx = np.arange(len(self))
        if exposure_ns == 0:
            t = self.cam_to_imu_time(self.frame_t_ns)
            return idx[self.gt.covers(t, self.max_gt_gap_ns)]
        starts = np.array([self.exposure_window(i, exposure_ns, anchor)[0] for i in idx], np.int64)
        ends = np.array([self.exposure_window(i, exposure_ns, anchor)[1] for i in idx], np.int64)
        ok = self.gt.covers(starts, self.max_gt_gap_ns) & self.gt.covers(ends, self.max_gt_gap_ns)
        return idx[ok]

    # ---- poses ----------------------------------------------------------- #

    def pose_body(self, t_ns) -> np.ndarray:
        """(...,4,4) T_WB at IMU-clock times."""
        return self.gt.interpolate(t_ns)["T_WB"]

    def pose_camera(self, t_ns) -> np.ndarray:
        """(...,4,4) T_WC = T_WB @ T_BS."""
        return self.pose_body(t_ns) @ self.cam.T_BS

    def gyro_at(self, t_ns, bias_corrected: bool = True) -> np.ndarray:
        """(...,3) interpolated angular velocity in the IMU frame, IMU-clock times."""
        t_ns = np.atleast_1d(np.asarray(t_ns, dtype=np.int64))
        epoch = self.imu.t_ns[0]
        ts = (self.imu.t_ns - epoch).astype(np.float64) / NS
        tq = np.clip((t_ns - epoch).astype(np.float64) / NS, ts[0], ts[-1])
        w = np.stack([np.interp(tq, ts, self.imu.gyro[:, i]) for i in range(3)], axis=-1)
        if bias_corrected:
            w = w - self.gt.interpolate(t_ns)["bg"]   # bias is in the IMU frame too
        return w

    def gyro_body(self, t_ns, bias_corrected: bool = True) -> np.ndarray:
        """Angular velocity rotated into the body frame (identity on stock EuRoC)."""
        R_B_imu = self.imu.calib.T_BS[:3, :3]
        return self.gyro_at(t_ns, bias_corrected) @ R_B_imu.T

    def integrate_gyro_rotation(self, t0_ns: int, t1_ns: int,
                                n_steps: int = 64,
                                bias_corrected: bool = True) -> np.ndarray:
        """
        Relative rotation R(t0)^T R(t1) of the *IMU body* by integrating gyro.

        This is the practical path (no ground truth needed). Compare against
        ``pose_body`` to sanity-check your time offset and bias handling.
        """
        edges = np.linspace(int(t0_ns), int(t1_ns), n_steps + 1)
        mids = ((edges[:-1] + edges[1:]) / 2).astype(np.int64)
        dt = np.diff(edges) / NS
        w = self.gyro_body(mids, bias_corrected=bias_corrected)
        R = np.eye(3)
        for k in range(n_steps):
            R = R @ so3_exp(w[k] * dt[k])
        return R

    # ---- the blur operator ----------------------------------------------- #

    def blur_poses(self, i: int, exposure_ns: int, n_samples: int = 24,
                   anchor: str = "mid", source: str = "gt") -> Dict[str, np.ndarray]:
        """
        Sample the camera trajectory across frame i's exposure window.

        Returns
        -------
        t_ns   : (n,) sample times
        T_WC   : (n,4,4) absolute camera poses (source='gt' only)
        dR     : (n,3,3) R_WC(t_k)^T R_WC(t_ref) -- reference-to-sample rotation
        dp     : (n,3)   camera-frame translation of the reference origin
        weights: (n,) uniform quadrature weights summing to 1

        Sampling density matters: consecutive warps should be under ~1 px apart,
        so n_samples should be at least the blur extent in pixels. Undersampling
        gives you a comb-shaped PSF instead of a smooth streak.
        """
        t0, t1 = self.exposure_window(i, exposure_ns, anchor)
        t_ns = np.linspace(t0, t1, n_samples).astype(np.int64)
        t_ref = int(self.cam_to_imu_time(self.frame_t_ns[i])[0])

        if source == "gt":
            T_WC = self.pose_camera(t_ns)                       # (n,4,4)
            T_ref = self.pose_camera(np.array([t_ref]))[0]      # (4,4)
            T_rel = se3_inv(T_WC) @ T_ref                       # ref -> sample
            return dict(t_ns=t_ns, T_WC=T_WC, dR=T_rel[:, :3, :3], dp=T_rel[:, :3, 3],
                        weights=np.full(n_samples, 1.0 / n_samples))

        if source == "gyro":
            # EuRoC T_BS maps sensor coords into the body frame, so R_WC = R_WB @ R_BC
            # and a body-frame relative rotation conjugates as R_BC^T (.) R_BC.
            R_BC = self.cam.T_BS[:3, :3]                        # camera -> body
            dR = np.empty((n_samples, 3, 3))
            for k, tk in enumerate(t_ns):
                R_body = self.integrate_gyro_rotation(tk, t_ref)  # sample -> ref, body frame
                dR[k] = R_BC.T @ R_body @ R_BC
            return dict(t_ns=t_ns, T_WC=None, dR=dR, dp=np.zeros((n_samples, 3)),
                        weights=np.full(n_samples, 1.0 / n_samples))

        raise ValueError("source must be 'gt' or 'gyro'")

    def blur_homographies(self, i: int, exposure_ns: int, n_samples: int = 24,
                          anchor: str = "mid", source: str = "gt") -> np.ndarray:
        """
        (n,3,3) rotation-only homographies H_k with  x_k ~ H_k x_ref.

        To render blur, sample the latent image at H_k^{-1} x for each output
        pixel x and average. Depth-independent, so this is exact only for pure
        rotation; use ``blur_flow`` when translation matters.
        """
        s = self.blur_poses(i, exposure_ns, n_samples, anchor, source)
        Kmat, Kinv = self.cam.K, np.linalg.inv(self.cam.K)
        return Kmat @ s["dR"] @ Kinv

    def blur_flow(self, i: int, exposure_ns: int, inv_depth: np.ndarray,
                  n_samples: int = 24, anchor: str = "mid",
                  scale: float = 1.0, shift: float = 0.0) -> np.ndarray:
        """
        (n,H,W,2) full depth-dependent displacement field over the exposure.

        ``inv_depth`` is an (H,W) affine-invariant inverse-depth map -- e.g. raw
        Depth Anything output. Metric inverse depth is recovered as
        ``(inv_depth - shift) / scale``; per the identifiability argument, only
        the ratio ``translation / scale`` is observable, so if you have GT
        velocity you can solve for ``scale`` instead of optimising it.
        """
        s = self.blur_poses(i, exposure_ns, n_samples, anchor, source="gt")
        H, W = self.cam.height, self.cam.width
        if inv_depth.shape != (H, W):
            raise ValueError(f"inv_depth must be {(H, W)}, got {inv_depth.shape}")
        z_inv = (np.asarray(inv_depth, np.float64) - shift) / scale

        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        rays = np.stack([(u - self.cam.cx) / self.cam.fx,
                         (v - self.cam.cy) / self.cam.fy,
                         np.ones_like(u)], axis=-1)                # (H,W,3)

        out = np.empty((n_samples, H, W, 2))
        for k in range(n_samples):
            X = rays + s["dp"][k][None, None, :] * z_inv[..., None]
            Xr = np.einsum('ij,hwj->hwi', s["dR"][k], X)
            zz = np.where(np.abs(Xr[..., 2]) < 1e-9, 1e-9, Xr[..., 2])
            out[k, ..., 0] = self.cam.fx * Xr[..., 0] / zz + self.cam.cx - u
            out[k, ..., 1] = self.cam.fy * Xr[..., 1] / zz + self.cam.cy - v
        return out

    # ---- diagnostics ------------------------------------------------------ #

    def parallax_budget(self, i: int, exposure_ns: int, z_min: float,
                        z_max: Optional[float] = None) -> Dict[str, float]:
        """
        How much displacement the depth-dependent term contributes, in pixels.

        If ``px_parallax`` is small relative to your error budget, the rotation-only
        gyro model is enough and you can drop the depth field entirely. If it is
        large, you are in the hard regime and need depth.
        """
        t = self.cam_to_imu_time(self.frame_t_ns[i])
        g = self.gt.interpolate(t)
        speed = float(np.linalg.norm(g["v"][0]))
        omega = float(np.linalg.norm(self.gyro_at(t)[0]))
        dt = exposure_ns / NS
        f = self.cam.fx
        near = f * speed * dt / max(z_min, 1e-6)
        far = f * speed * dt / max(z_max, 1e-6) if z_max else 0.0
        return dict(speed_mps=speed, omega_rads=omega, omega_degs=np.degrees(omega),
                    px_rotation=f * omega * dt,
                    px_translation_near=near,
                    px_parallax=near - far)

    def time_offset_residual(self, offsets_ns: Sequence[int],
                             stride: int = 5, n_steps: int = 16,
                             reference: str = "gt") -> np.ndarray:
        """
        Sweep candidate offsets; return mean rotation disagreement (rad) between
        the gyro stream and a reference rotation over each frame interval.

        The offset is applied to the *gyro lookup only* -- the reference clock
        stays fixed. Shifting both sides together is a no-op and yields a flat
        curve, which is an easy mistake to make.

        ``reference='gt'`` uses ground-truth pose, which stands in for the
        image-derived rotation you would use on real footage where no GT exists.
        Substitute your own frame-to-frame rotation estimates to calibrate a rig.

        The trough sharpens with angular rate, so run this on your most
        aggressive segments. A flat curve means the segment has too little
        rotation to constrain the offset.

        Note this compares *interval-integrated* rotation, which is a weaker
        signal than instantaneous pointing error: it scales with how much omega
        changes across the interval, not with omega itself. Expect it to localise
        the offset to roughly a millisecond. To reach the ~50-150 us that
        deblurring actually needs, refine against a photometric residual -- render
        blur with the candidate offset and match it to the observed frame.
        """
        valid = set(int(v) for v in self.valid_frames())
        # Pair each sampled frame with its IMMEDIATE successor. Pairing
        # consecutive *sampled* frames instead would make every interval
        # `stride` frames long and trip the dropout guard below.
        pairs = [(i, i + 1) for i in sorted(valid)[::stride]
                 if (i + 1) in valid and i + 1 < len(self)]
        if len(pairs) < 2:
            raise ValueError("not enough consecutive frames with ground truth to sweep")
        max_gap = 3 * NS // int(self.cam.rate_hz)
        res = np.full(len(offsets_ns), np.nan)
        for j, off in enumerate(offsets_ns):
            errs = []
            for i, i_next in pairs:
                ta = int(self.cam_to_imu_time(self.frame_t_ns[i])[0])
                tb = int(self.cam_to_imu_time(self.frame_t_ns[i_next])[0])
                if tb - ta > max_gap:
                    continue  # straddles a dropout
                R_gyro = self.integrate_gyro_rotation(
                    ta + int(off), tb + int(off), n_steps=n_steps)
                if reference == "gt":
                    T = self.pose_body(np.array([ta, tb]))
                    R_ref = T[0, :3, :3].T @ T[1, :3, :3]
                else:
                    raise ValueError("reference must be 'gt'")
                errs.append(np.linalg.norm(so3_log(R_gyro.T @ R_ref)))
            if errs:
                res[j] = float(np.mean(errs))
        return res

    def summary(self) -> str:
        t = self.frame_t_ns
        dur = (t[-1] - t[0]) / NS
        gaps = self.gt.gaps(self.max_gt_gap_ns)
        w = self.gyro_at(self.cam_to_imu_time(t))
        sp = np.linalg.norm(self.gt.v, axis=1)
        lines = [
            f"EuRoC sequence : {os.path.basename(self.root.rstrip('/'))}",
            f"primary camera : {self.cam_name}  {self.cam.width}x{self.cam.height} "
            f"@ {self.cam.rate_hz:g} Hz  f=({self.cam.fx:.1f}, {self.cam.fy:.1f})",
            f"frames         : {len(self)}  over {dur:.1f} s",
            f"imu            : {len(self.imu.t_ns)} samples @ {self.imu.calib.rate_hz:g} Hz",
            f"gt             : {len(self.gt.t_ns)} samples, {len(gaps)} gap(s) > "
            f"{self.max_gt_gap_ns/1e6:g} ms",
            f"frames with gt : {len(self.valid_frames())} / {len(self)}",
            f"stereo baseline: {self.baseline_m*100:.1f} cm",
            f"time offset    : {self.time_offset_ns} ns (applied)",
            f"angular rate   : median {np.degrees(np.median(np.linalg.norm(w,axis=1))):.1f} "
            f"deg/s, p95 {np.degrees(np.percentile(np.linalg.norm(w,axis=1),95)):.1f} deg/s",
            f"speed          : median {np.median(sp):.2f} m/s, p95 {np.percentile(sp,95):.2f} m/s",
            f"pointcloud     : {'yes' if self.pointcloud is not None else 'no'}",
            "shutter        : GLOBAL (MT9V034) -- line delay is 0",
            "exposure       : not logged; auto-exposure differs per camera",
        ]
        return "\n".join(lines)


def load_sequence(root: str, cam: str = "cam0", time_offset_ns: int = 0,
                  max_gt_gap_ns: int = 50_000_000) -> EuRoCSequence:
    """
    Load one EuRoC sequence in ASL format.

    ``root`` may be the sequence directory (containing ``mav0/``) or ``mav0``
    itself. Tested against V1_02_medium and V1_03_difficult.
    """
    base = os.path.join(root, "mav0") if os.path.isdir(os.path.join(root, "mav0")) else root
    if not os.path.isdir(base):
        raise FileNotFoundError(f"no mav0/ under {root}")
    other = "cam1" if cam == "cam0" else "cam0"
    for sub in (cam, other, "imu0", "state_groundtruth_estimate0"):
        if not os.path.isdir(os.path.join(base, sub)):
            raise FileNotFoundError(f"missing {sub}/ under {base}")

    t_ns, paths = load_frame_index(os.path.join(base, cam))
    return EuRoCSequence(
        root=root,
        cam_name=cam,
        cam=load_camera_calib(os.path.join(base, cam)),
        cam_other=load_camera_calib(os.path.join(base, other)),
        imu=load_imu(os.path.join(base, "imu0")),
        gt=load_groundtruth(os.path.join(base, "state_groundtruth_estimate0")),
        frame_t_ns=t_ns,
        frame_paths=paths,
        time_offset_ns=int(time_offset_ns),
        max_gt_gap_ns=int(max_gt_gap_ns),
    )


def load_vicon_room(root_dir: str, sequences=("V1_02_medium", "V1_03_difficult"),
                    **kwargs) -> Dict[str, EuRoCSequence]:
    """Load several sequences at once, keyed by name."""
    return {s: load_sequence(os.path.join(root_dir, s), **kwargs) for s in sequences}


# --------------------------------------------------------------------------- #
# Optional torch wrapper
# --------------------------------------------------------------------------- #

class EuRoCTorchDataset:
    """
    torch.utils.data.Dataset over frames with valid ground truth.

    Each item is a dict with the image, the sampled blur trajectory, and the
    rotation-only homographies. Instantiated lazily so importing this module
    never requires torch.
    """

    def __init__(self, seq: EuRoCSequence, exposure_ns: int = 10_000_000,
                 n_samples: int = 24, anchor: str = "mid", source: str = "gt",
                 normalise: bool = True, indices: Optional[Sequence[int]] = None):
        import torch  # noqa: F401
        self.seq = seq
        self.exposure_ns = int(exposure_ns)
        self.n_samples = int(n_samples)
        self.anchor = anchor
        self.source = source
        self.normalise = normalise
        self.indices = (np.asarray(indices, dtype=int) if indices is not None
                        else seq.valid_frames(exposure_ns, anchor))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> Dict[str, Any]:
        import torch
        i = int(self.indices[k])
        img = self.seq.load_image(i).astype(np.float32)
        if self.normalise:
            img = img / 255.0
        s = self.seq.blur_poses(i, self.exposure_ns, self.n_samples, self.anchor, self.source)
        H = self.seq.blur_homographies(i, self.exposure_ns, self.n_samples,
                                       self.anchor, self.source)
        t = self.seq.cam_to_imu_time(self.seq.frame_t_ns[i])
        g = self.seq.gt.interpolate(t)
        return {
            "index": i,
            "t_ns": int(self.seq.frame_t_ns[i]),
            "image": torch.from_numpy(img)[None],
            "dR": torch.from_numpy(s["dR"]).float(),
            "dp": torch.from_numpy(s["dp"]).float(),
            "H": torch.from_numpy(H).float(),
            "weights": torch.from_numpy(s["weights"]).float(),
            "velocity": torch.from_numpy(g["v"][0]).float(),
            "gyro": torch.from_numpy(self.seq.gyro_at(t)[0]).float(),
            "K": torch.from_numpy(self.seq.cam.K).float(),
        }


__all__ = [
    "EuRoCSequence", "EuRoCTorchDataset", "CameraCalib", "ImuCalib",
    "ImuData", "GroundTruth",
    "load_sequence", "load_vicon_room", "load_camera_calib", "load_imu_calib",
    "load_imu", "load_groundtruth", "load_frame_index", "load_pointcloud",
    "quat_wxyz_to_matrix", "so3_exp", "so3_log", "se3", "se3_inv",
]