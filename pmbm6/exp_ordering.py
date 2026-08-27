"""
Where does non-linear motion actually break Blur2Seq?

The Trajectory Prediction Network predicts T FREE poses (r in R^T for roll,
[p,y] in R^{Tx2} for pitch/yaw) -- it is NOT constrained to linear motion, and
unlike Zhang et al.'s ETR it does not interpolate from two endpoints.  So the
forward model handles arbitrary trajectories exactly.

The linearity assumption is in the VIDEO GENERATION ORDERING HEURISTIC, which
the paper states on p.12:

    "we find an extreme point as the farthest from all the others.  Then, we
     generate the trajectory by ordering the points according to the distance
     to the farthest one.  Although this heuristic may not work for sinuous or
     noisy trajectories..."

This script quantifies "may not work", and compares against the exact
minimum-length Hamiltonian path ordering (addendum §14), which is correct for
ANY injective arc, not just monotone ones.

Run:  python3 exp_ordering.py
"""
import numpy as np
from itertools import permutations

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- orderings

def heuristic_blur2seq(P):
    """Paper p.12: extreme point = farthest from all others; sort by distance."""
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    extreme = int(np.argmax(D.sum(1)))
    return np.argsort(D[extreme])


def min_length_path(P):
    """Exact minimum-length Hamiltonian path (Held-Karp). T <= ~14."""
    T = len(P)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    INF = np.inf
    dp = np.full((1 << T, T), INF)
    par = np.full((1 << T, T), -1, dtype=int)
    for i in range(T):
        dp[1 << i, i] = 0.0
    for mask in range(1 << T):
        for last in range(T):
            c = dp[mask][last]
            if c == INF or not (mask >> last) & 1:
                continue
            for nxt in range(T):
                if (mask >> nxt) & 1:
                    continue
                nm = mask | (1 << nxt)
                v = c + D[last, nxt]
                if v < dp[nm][nxt]:
                    dp[nm][nxt] = v
                    par[nm][nxt] = last
    full = (1 << T) - 1
    last = int(np.argmin(dp[full]))
    order, mask = [], full
    while last != -1:
        order.append(last)
        p = par[mask][last]
        mask ^= (1 << last)
        last = p
    return np.array(order[::-1])


def score(order, T):
    """Fraction of adjacent pairs in the correct sequence (either direction)."""
    o = np.asarray(order)
    fwd = np.mean(np.abs(np.diff(o)) == 1)
    return max(fwd, np.mean(np.abs(np.diff(o[::-1])) == 1))


# ------------------------------------------------------------- trajectories

def traj(kind, T=12, tau=0.010, seed=0):
    """Angular trajectory (pitch,yaw,roll) in radians over the exposure."""
    r = np.random.default_rng(seed)
    t = np.linspace(-tau / 2, tau / 2, T)
    A = 0.02                                        # ~1.1 deg amplitude
    if kind == "linear":
        return np.outer(t / tau, r.normal(size=3)) * A * 40
    if kind == "tremor_10Hz":                       # physiological hand tremor
        ph = r.uniform(0, 2 * np.pi, 3)
        return A * np.stack([np.sin(2*np.pi*10*t + p) for p in ph], -1)
    if kind == "body_50Hz":                         # drone attitude dynamics
        ph = r.uniform(0, 2 * np.pi, 3)
        return A * np.stack([np.sin(2*np.pi*50*t + p) for p in ph], -1)
    if kind == "vib_150Hz":                         # airframe / rotor vibration
        ph = r.uniform(0, 2 * np.pi, 3)
        return A * np.stack([np.sin(2*np.pi*150*t + p) for p in ph], -1)
    if kind == "vib_400Hz":                         # blade-pass, small prop
        ph = r.uniform(0, 2 * np.pi, 3)
        return A * np.stack([np.sin(2*np.pi*400*t + p) for p in ph], -1)
    if kind == "sinuous":                           # S-curve with a reversal
        s = t / tau
        return A * 30 * np.stack([s, np.sin(3*np.pi*s) / 6, s * 0 + s**2], -1)
    raise ValueError(kind)


KINDS = ["linear", "tremor_10Hz", "body_50Hz", "vib_150Hz", "vib_400Hz", "sinuous"]
T, TAU, F = 12, 0.010, 458.0

print("=" * 78)
print("EXPERIMENT A -- ordering accuracy: paper heuristic vs min-length path")
print(f"  T={T} poses, tau={TAU*1e3:.0f} ms, f={F:.0f} px.  100 random draws each.")
print("=" * 78)
print(f"  {'trajectory':<14} {'cycles in tau':>14} {'blur (px)':>10} "
      f"{'heuristic':>11} {'min-length':>12}")
for kind in KINDS:
    hs, ms, ext = [], [], []
    for s in range(100):
        P = traj(kind, T, TAU, seed=s)
        hs.append(score(heuristic_blur2seq(P), T))
        ms.append(score(min_length_path(P), T))
        ext.append(F * np.ptp(np.linalg.norm(P, axis=1)))
    cyc = {"linear": 0.0, "tremor_10Hz": 0.1, "body_50Hz": 0.5,
           "vib_150Hz": 1.5, "vib_400Hz": 4.0, "sinuous": 1.5}[kind]
    print(f"  {kind:<14} {cyc:>14.1f} {np.mean(ext):>10.1f} "
          f"{np.mean(hs)*100:>10.0f}% {np.mean(ms)*100:>11.0f}%")

print()
print("=" * 78)
print("EXPERIMENT B -- how non-linear is the training distribution?")
print("  Blur2Seq trains on physiological hand-tremor trajectories (Sec 5.1).")
print("  Curvature measured as max deviation from the chord, in pixels.")
print("=" * 78)
print(f"  {'trajectory':<14} {'sagitta (px)':>14} {'self-intersects?':>18} "
      f"{'monotone from end?':>20}")
for kind in KINDS:
    sag, si, mono = [], [], []
    for s in range(200):
        P = traj(kind, 41, TAU, seed=s) * F           # to pixels
        chord = P[-1] - P[0]
        n = np.linalg.norm(chord)
        if n < 1e-9:
            d = np.linalg.norm(P - P[0], axis=1)
        else:
            proj = np.outer((P - P[0]) @ chord / n**2, chord)
            d = np.linalg.norm((P - P[0]) - proj, axis=1)
        sag.append(d.max())
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        ex = int(np.argmax(D.sum(1)))
        r = D[ex]
        mono.append(np.all(np.diff(r[np.argsort(np.arange(len(P)))]) >= 0)
                    or np.all(np.diff(r[::-1]) >= 0))
        # crude self-intersection: any non-adjacent pair closer than the step
        step = np.median(np.linalg.norm(np.diff(P, axis=0), axis=1))
        M = D + np.eye(len(P)) * 1e9
        for k in range(1, 3):
            M += np.diag(np.ones(len(P)-k) * 1e9, k) + np.diag(np.ones(len(P)-k) * 1e9, -k)
        si.append(M.min() < 0.5 * step)
    print(f"  {kind:<14} {np.mean(sag):>14.2f} {np.mean(si)*100:>17.0f}% "
          f"{np.mean(mono)*100:>19.0f}%")

print()
print("=" * 78)
print("EXPERIMENT C -- does the EMD trajectory loss see ordering at all?")
print("=" * 78)


def emd_cost(A, B, f=F):
    """Paper eq. (36)-(37): grid transformed by each pose, image-plane cost,
    optimal transport.  Small T -> exact assignment by brute force."""
    g = np.stack(np.meshgrid(np.linspace(-200, 200, 4),
                             np.linspace(-200, 200, 4)), -1).reshape(-1, 2)
    def proj(P):                       # small-angle: pitch/yaw shift, roll rotate
        out = []
        for th in P:
            c, s = np.cos(th[2]), np.sin(th[2])
            R = np.array([[c, -s], [s, c]])
            out.append(g @ R.T + f * np.array([th[1], th[0]]))
        return np.stack(out)
    GA, GB = proj(A), proj(B)
    M = ((GA[:, None] - GB[None]) ** 2).sum(-1).mean(-1)
    n = len(A)
    best = min(sum(M[i, p[i]] for i in range(n)) for p in permutations(range(n)))
    return best / n


P = traj("sinuous", 7, TAU, seed=1)
shuffles = [np.arange(7), np.array([0, 1, 2, 3, 4, 5, 6])[::-1],
            np.array([3, 1, 5, 0, 6, 2, 4]), np.array([6, 0, 5, 1, 4, 2, 3])]
names = ["correct order", "reversed", "shuffled A", "shuffled B"]
print(f"  {'ordering':<16} {'EMD cost vs correct':>22} {'adjacency score':>18}")
for nm, sh in zip(names, shuffles):
    print(f"  {nm:<16} {emd_cost(P, P[sh]):>22.6f} {score(sh, 7)*100:>17.0f}%")
print("  -> EMD is exactly 0 for every permutation.  The trajectory loss cannot")
print("     supervise ordering, by construction (paper Sec 5.2, and §3.1 of")
print("     MATH_FOUNDATION).  Order is imposed post hoc by the heuristic alone.")
