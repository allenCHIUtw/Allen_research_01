"""
euroc_io — ASL-format EuRoC loader and a one-shot integrity verifier.

Run `python3 euroc_io.py verify /path/to/V2_01_easy/mav0` the moment the zip
finishes extracting. It settles, from your own files rather than from docs,
every item flagged unconfirmed earlier:

  - does mav0/pointcloud0/ exist, and under what filename
  - does mav0/vicon0/ exist (vs. only state_groundtruth_estimate0/)
  - the actual sample rate of state_groundtruth_estimate0/data.csv
  - whether cam*/data.csv really carries only [timestamp, filename]
    (confirms there is no exposure column anywhere)
  - cross-checks the shipped sensor.yaml intrinsics against the published
    EuRoC constants, so a bad unzip or a swapped file is caught immediately

Also provides plain ASL-format readers for the P1 synthesiser: camera
intrinsics/extrinsics, IMU stream, ground-truth pose stream (nearest-neighbour
interpolated to any query time), and a .ply point-cloud reader with no
external dependency beyond numpy (ASCII and binary_little_endian PLY).
"""
from __future__ import annotations
import csv, os, struct, sys
import numpy as np

# ---------------------------------------------------------------------------
# Published EuRoC calibration constants (Burri et al. 2016; mirrored verbatim
# in OKVIS/ORB-SLAM3/VINS-Mono configs). Used only to sanity-check that the
# sensor.yaml you actually unzipped matches the documented dataset.
# ---------------------------------------------------------------------------
REFERENCE_CAM0 = dict(
    fx=458.654880721, fy=457.296696463, cx=367.215803962, cy=248.37534061,
    dist=[-0.28340811217, 0.0739590738929, 0.000193595028569, 1.76187114545e-05],
)
REFERENCE_CAM1 = dict(
    fx=457.587426604, fy=456.13442556, cx=379.99944652, cy=255.238185386,
    dist=[-0.283683654496, 0.0745128430929, -0.000104738949098, -3.55590700274e-05],
)
REFERENCE_BASELINE_M = 0.110074137800478
REFERENCE_IMU = dict(
    rate_hz=200.0,
    gyro_noise_density=1.6968e-04, gyro_random_walk=1.9393e-05,
    accel_noise_density=2.0000e-03, accel_random_walk=3.0000e-03,
)


# ---------------------------------------------------------------------------
# YAML reading without a dependency: EuRoC's sensor.yaml is a flat subset of
# YAML 1.1 (scalars, flow-style lists, one nested `T_BS:` block). A tiny
# hand-rolled parser avoids pulling in pyyaml for something this constrained;
# falls back to pyyaml if it's already installed.
# ---------------------------------------------------------------------------

def read_sensor_yaml(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    out, cur_key, cur_list = {}, None, None
    with open(path) as f:
        for raw in f:
            line = raw.split('#', 1)[0].rstrip()
            if not line.strip() or line.strip() == '%YAML:1.0' or line.strip().startswith('---'):
                continue
            if line.startswith('  - ') and cur_key is not None:
                cur_list.append(_scalar(line.strip()[2:]))
                out[cur_key] = cur_list
                continue
            if ':' in line and not line.startswith(' '):
                key, _, val = line.partition(':')
                key, val = key.strip(), val.strip()
                if val == '' or val is None:
                    cur_key, cur_list = key, []
                    out[key] = cur_list
                elif val.startswith('[') and val.endswith(']'):
                    out[key] = [_scalar(v.strip()) for v in val[1:-1].split(',') if v.strip()]
                    cur_key = None
                else:
                    out[key] = _scalar(val)
                    cur_key = None
    return out


def _scalar(s):
    s = s.strip().strip('"').strip("'")
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

def read_imu_csv(path):
    """imu0/data.csv -> dict(t (N,) seconds, gyro (N,3) rad/s, accel (N,3) m/s^2)."""
    t, w, a = [], [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(int(row[0]) * 1e-9)
            w.append([float(row[1]), float(row[2]), float(row[3])])
            a.append([float(row[4]), float(row[5]), float(row[6])])
    return dict(t=np.array(t), gyro=np.array(w), accel=np.array(a))


def read_cam_csv(path):
    """cam*/data.csv -> (t (N,) seconds, filenames list). Also confirms the
    two-column-only claim: raises if extra columns are present, which would
    mean this particular mirror DOES carry exposure metadata after all."""
    t, names = [], []
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        if len(header) != 2:
            raise RuntimeError(
                f"cam data.csv has {len(header)} columns ({header}) -- "
                "expected exactly 2. This EuRoC mirror may carry exposure "
                "metadata that prior research did not find. Investigate "
                "before assuming exposure is unavailable."
            )
        for row in r:
            t.append(int(row[0]) * 1e-9)
            names.append(row[1])
    return np.array(t), names


def read_groundtruth_csv(path):
    """state_groundtruth_estimate0/data.csv ->
    dict(t, pos (N,3), quat_wxyz (N,4), vel (N,3), bw (N,3), ba (N,3)).
    Column count varies by EuRoC mirror; handle the common 17-column layout
    (t, p, q_wxyz, v, bw, ba) and degrade gracefully otherwise."""
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            rows.append([float(x) for x in row])
    arr = np.array(rows)
    t = arr[:, 0] * 1e-9
    out = dict(t=t, pos=arr[:, 1:4], quat_wxyz=arr[:, 4:8], n_cols=arr.shape[1], header=header)
    if arr.shape[1] >= 17:
        out['vel'] = arr[:, 8:11]
        out['bw'] = arr[:, 11:14]
        out['ba'] = arr[:, 14:17]
    return out


def read_ply(path):
    """Minimal PLY reader (ASCII or binary_little_endian), vertex x,y,z only.
    No trimesh/open3d dependency. Returns (N,3) float32."""
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'ply'
        fmt = None
        n_verts = 0
        props = []
        line = f.readline()
        while not line.strip().startswith(b'end_header'):
            parts = line.split()
            if parts[0] == b'format':
                fmt = parts[1].decode()
            elif parts[:2] == [b'element', b'vertex']:
                n_verts = int(parts[2])
                reading_vertex_props = True
            elif parts[0] == b'property' and 'reading_vertex_props' in dir():
                props.append(parts[-1].decode())
            elif parts[0] == b'element':
                reading_vertex_props = False
            line = f.readline()

        xi, yi, zi = props.index('x'), props.index('y'), props.index('z')
        if fmt == 'ascii':
            pts = np.zeros((n_verts, 3), dtype=np.float32)
            for i in range(n_verts):
                vals = f.readline().split()
                pts[i] = [float(vals[xi]), float(vals[yi]), float(vals[zi])]
            return pts
        elif fmt == 'binary_little_endian':
            stride = len(props) * 4
            buf = f.read(n_verts * stride)
            pts = np.zeros((n_verts, 3), dtype=np.float32)
            for i in range(n_verts):
                rec = struct.unpack_from(f'<{len(props)}f', buf, i * stride)
                pts[i] = [rec[xi], rec[yi], rec[zi]]
            return pts
        else:
            raise NotImplementedError(f"PLY format '{fmt}' not handled (need ascii or "
                                      "binary_little_endian)")


def interp_pose(gt, t_query):
    """Linear position + slerp orientation at t_query (seconds), from a
    read_groundtruth_csv() dict. Returns (pos (3,), quat_wxyz (4,))."""
    t = gt['t']
    idx = np.searchsorted(t, t_query)
    idx = np.clip(idx, 1, len(t) - 1)
    t0, t1 = t[idx - 1], t[idx]
    a = 0.0 if t1 == t0 else (t_query - t0) / (t1 - t0)
    pos = (1 - a) * gt['pos'][idx - 1] + a * gt['pos'][idx]
    q0, q1 = gt['quat_wxyz'][idx - 1], gt['quat_wxyz'][idx]
    if np.dot(q0, q1) < 0:
        q1 = -q1
    quat = (1 - a) * q0 + a * q1
    quat = quat / np.linalg.norm(quat)
    return pos, quat


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def verify_sequence(mav0_dir):
    print(f"Verifying: {mav0_dir}\n" + "=" * 70)
    entries = sorted(os.listdir(mav0_dir))
    print(f"Top-level entries: {entries}\n")

    # --- point cloud / vicon0 presence (previously unconfirmed) ---
    pc_candidates = [e for e in entries if 'pointcloud' in e.lower() or 'leica' in e.lower()
                    or 'pcl' in e.lower()]
    vicon_dirs = [e for e in entries if e.lower().startswith('vicon')]
    print(f"[point cloud / structure]  candidates found: {pc_candidates or 'NONE'}")
    print(f"[vicon0/]                  present: {bool(vicon_dirs)} ({vicon_dirs})")
    for cand in pc_candidates:
        sub = os.path.join(mav0_dir, cand)
        if os.path.isdir(sub):
            files = os.listdir(sub)
            print(f"    {cand}/ contains: {files}")
            for fn in files:
                if fn.lower().endswith('.ply'):
                    pts = read_ply(os.path.join(sub, fn))
                    print(f"    -> {fn}: {len(pts):,} points, "
                          f"bbox = {pts.min(0)} .. {pts.max(0)}")
    print()

    # --- camera CSV: confirm two-column, no exposure field ---
    for cam in ('cam0', 'cam1'):
        p = os.path.join(mav0_dir, cam, 'data.csv')
        if not os.path.exists(p):
            continue
        t, names = read_cam_csv(p)
        dt = np.diff(t)
        print(f"[{cam}] {len(t)} frames, {1/np.median(dt):.2f} Hz median "
              f"(jitter std {dt.std()*1e6:.1f} us) -- two-column CSV confirmed, "
              "no exposure field present")
        y = read_sensor_yaml(os.path.join(mav0_dir, cam, 'sensor.yaml'))
        intr = y.get('intrinsics', [])
        ref = REFERENCE_CAM0 if cam == 'cam0' else REFERENCE_CAM1
        if len(intr) == 4:
            ok = np.allclose(intr, [ref['fx'], ref['fy'], ref['cx'], ref['cy']], atol=0.5)
            print(f"    intrinsics {intr}  vs reference {[ref['fx'],ref['fy'],ref['cx'],ref['cy']]}"
                  f"  -> {'MATCH' if ok else 'MISMATCH -- check this file'}")
    print()

    # --- ground truth rate (previously unconfirmed) ---
    gtp = os.path.join(mav0_dir, 'state_groundtruth_estimate0', 'data.csv')
    if os.path.exists(gtp):
        gt = read_groundtruth_csv(gtp)
        dt = np.diff(gt['t'])
        print(f"[state_groundtruth_estimate0] {len(gt['t'])} rows, "
              f"{gt['n_cols']} columns, rate = {1/np.median(dt):.2f} Hz "
              f"(median dt = {np.median(dt)*1e3:.3f} ms)")
        print(f"    header: {gt['header']}")
        print(f"    duration: {gt['t'][-1]-gt['t'][0]:.1f} s, "
              f"path length approx {np.sum(np.linalg.norm(np.diff(gt['pos'],axis=0),axis=1)):.1f} m")
    else:
        print("[state_groundtruth_estimate0] NOT FOUND")
    print()

    # --- Leica (Machine Hall only) ---
    lp = os.path.join(mav0_dir, 'leica0', 'data.csv')
    print(f"[leica0/] present: {os.path.exists(lp)}  "
          f"(expected on Machine Hall sequences only, absent on Vicon Room)")
    print()

    # --- IMU noise params vs reference ---
    ip = os.path.join(mav0_dir, 'imu0', 'sensor.yaml')
    if os.path.exists(ip):
        y = read_sensor_yaml(ip)
        rate = y.get('rate_hz')
        ok_rate = (rate == REFERENCE_IMU['rate_hz'])
        print(f"[imu0] rate_hz = {rate}  -> {'MATCH' if ok_rate else 'CHECK'}")
        for k in ('gyroscope_noise_density', 'gyroscope_random_walk',
                  'accelerometer_noise_density', 'accelerometer_random_walk'):
            if k in y:
                print(f"    {k} = {y[k]}")

    print("\n" + "=" * 70)
    print("Done. Compare the [point cloud], [vicon0/], and [ground truth rate]")
    print("lines above against build_order.html's reference table -- those")
    print("were the three items flagged unconfirmed from documentation alone.")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "verify":
        print("Usage: python3 euroc_io.py verify /path/to/<SEQ>/mav0")
        sys.exit(1)
    verify_sequence(sys.argv[2])
