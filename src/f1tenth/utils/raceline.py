'''
    Minimum-curvature race line + speed profile.

    The race line is the closed path that stays inside the track (minus a safety margin)
    and has the smallest total squared curvature. That gives the classic racing lines
    (file/optimal racing line): enter wide, clip the apex, exit wide, straighten S-bends.
    Minimum curvature is direction independent, so the reverse track uses the same line.

        p_i = c_i + a_i * n_i                        (c: centerline, n: left normal)
        min_a  sum_i | p_{i-1} - 2 p_i + p_{i+1} |^2  (curvature * ds^2)
        s.t.   -(w_right_i - margin) <= a_i <= (w_left_i - margin)

    p is linear in a, so this is an exact bounded linear least-squares problem (one solve).
'''

import numpy as np
from numba import njit
from scipy import sparse
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline


def resample_closed(xy, ds):
    '''Resample a closed polyline (no duplicated end point) at ~uniform spacing ds.'''
    pts = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(s[-1] / ds)), 8)
    cs = CubicSpline(s, pts, bc_type='periodic')
    s_new = np.linspace(0.0, s[-1], n, endpoint=False)
    return cs(s_new), s[-1]


def tangents_normals(xy):
    '''Unit tangent and left normal of a closed polyline (central differences).'''
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return t, n


def curvature(xy):
    '''Signed curvature of a closed polyline from three consecutive points.'''
    p0, p1, p2 = np.roll(xy, 1, axis=0), xy, np.roll(xy, -1, axis=0)
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    cross = (p1 - p0)[:, 0] * (p2 - p1)[:, 1] - (p1 - p0)[:, 1] * (p2 - p1)[:, 0]
    return 2.0 * cross / (a * b * c + 1e-12)


def min_curvature_line(center, w_left, w_right, margin, reg=1e-7, w_length=0.0):
    '''
        center:  (N, 2) closed centerline, ~uniform spacing (no duplicated end point)
        w_left, w_right: (N,) distance from centerline to the left / right wall [m]
        margin:  distance the line keeps from each wall [m] (car half width + safety)
        returns: (N, 2) race line, (N,) its lateral offset from the centerline
    '''
    N = center.shape[0]
    _, n = tangents_normals(center)
    lo = -(w_right - margin)
    hi = (w_left - margin)
    bad = lo > hi                       # narrower than 2*margin: stay on the centerline
    lo[bad] = hi[bad] = 0.0
    hi = np.maximum(hi, lo + 1e-6)

    # second-difference operator on a closed loop
    D = sparse.diags([np.ones(N), -2.0 * np.ones(N), np.ones(N)], [-1, 0, 1], shape=(N, N), format='lil')
    D[0, N - 1] = 1.0
    D[N - 1, 0] = 1.0
    D = D.tocsr()

    # D p = D c + D diag(n) a ; small ridge term keeps the solution unique on straights
    # w_length > 0 adds the path length (sum of squared first differences): shortest-path blend
    D1 = sparse.diags([-np.ones(N), np.ones(N)], [0, 1], shape=(N, N), format='lil')
    D1[N - 1, 0] = 1.0
    D1 = D1.tocsr()
    wl1 = np.sqrt(w_length)
    A = sparse.vstack([D @ sparse.diags(n[:, 0]), D @ sparse.diags(n[:, 1]),
                       wl1 * (D1 @ sparse.diags(n[:, 0])), wl1 * (D1 @ sparse.diags(n[:, 1])),
                       np.sqrt(reg) * sparse.identity(N)]).tocsr()
    b = -np.concatenate([D @ center[:, 0], D @ center[:, 1],
                         wl1 * (D1 @ center[:, 0]), wl1 * (D1 @ center[:, 1]), np.zeros(N)])
    # bounded convex quadratic: L-BFGS-B with the exact gradient (lsq_linear/trf was minutes per track)
    At = A.T.tocsr()
    def f(a):
        r = A @ a - b
        return float(r @ r), 2.0 * (At @ r)
    x0 = np.clip(np.zeros(N), lo, hi)
    res = minimize(f, x0, jac=True, method='L-BFGS-B', bounds=list(zip(lo, hi)),
                   options={'maxiter': 20000, 'ftol': 1e-14, 'gtol': 1e-10})
    off = np.clip(res.x, lo, hi)
    return center + off[:, None] * n, off


def speed_profile(xy, v_min, v_max, a_lat, a_acc, a_brake):
    '''
        Max speed along a closed line: lateral-acceleration limit in corners, then a backward
        (braking) and a forward (acceleration) pass so the profile is drivable.
    '''
    k = np.abs(curvature_smooth(xy))
    v = np.clip(np.sqrt(a_lat / (k + 1e-6)), v_min, v_max)
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    return np.maximum(_passes(v, seg, a_acc, a_brake), v_min)


@njit(cache=True)
def _passes(v, seg, a_acc, a_brake):
    n = len(v)
    for _ in range(2):
        for i in range(n - 1, -1, -1):            # braking
            j = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2.0 * a_brake * seg[i]))
        for i in range(n):                        # acceleration
            j = (i - 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2.0 * a_acc * seg[j]))
    return v


def curvature_smooth(xy, span=1.0):
    '''Curvature from points ~span meters apart (robust to small waypoint noise).'''
    seg = np.mean(np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1))
    k = max(1, int(round(span / seg)))
    p0, p1, p2 = np.roll(xy, k, axis=0), xy, np.roll(xy, -k, axis=0)
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    cross = (p1 - p0)[:, 0] * (p2 - p1)[:, 1] - (p1 - p0)[:, 1] * (p2 - p1)[:, 0]
    return 2.0 * cross / (a * b * c + 1e-9)


def lap_time(xy, v):
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    return float(np.sum(seg / np.maximum(v, 1e-3)))


def min_time_line(center, w_left, w_right, margin, v_min, v_max, a_lat, a_acc, a_brake,
                  weights=(0.0, 0.001, 0.003, 0.01, 0.03, 0.1)):
    '''
        Approximate minimum-lap-time line: blend minimum curvature (fast corners) with the
        shortest path (less distance), and keep the blend with the best lap time under the car's
        lateral / acceleration / braking limits. With a weak motor (a_acc ~3 m/s^2) the shortest-path
        share matters more than for a full-size race car. Returns line, lap time, chosen weight.
    '''
    best = None
    for w in weights:
        line, _ = min_curvature_line(center, w_left, w_right, margin, w_length=w)
        dense, _ = resample_closed(line, 0.1)
        t = lap_time(dense, speed_profile(dense, v_min, v_max, a_lat, a_acc, a_brake))
        if best is None or t < best[1]:
            best = (line, t, w)
    return best


def refine_min_time(center, w_left, w_right, margin, off0, v_min, v_max, a_lat, a_acc, a_brake,
                    bump_spacing=4.0, bump_width=3.0, iters=60):
    '''
        Minimum-lap-time refinement of a line given as centerline offsets off0 (e.g. the
        minimum-curvature solution). The offset is changed by a sum of smooth bumps along the
        track (one every bump_spacing m), and their amplitudes are optimized for the lap time
        computed with the car's lateral / acceleration / braking limits. This captures what
        minimum curvature ignores, e.g. late apexes before straights (exit speed matters).
        Returns (line, offsets, lap time).
    '''
    from scipy.optimize import minimize
    N = len(center)
    _, n = tangents_normals(center)
    lo, hi = -(w_right - margin), (w_left - margin)
    hi = np.maximum(hi, lo)
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    L = float(np.sum(seg))
    K = max(8, int(L / bump_spacing))
    centers = np.linspace(0, L, K, endpoint=False)
    d = np.abs(s[:, None] - centers[None, :])
    d = np.minimum(d, L - d)
    B = np.exp(-0.5 * (d / bump_width) ** 2)                 # (N, K)

    def offsets(c):
        return np.clip(off0 + B @ c, lo, hi)

    def cost(c):
        line = center + offsets(c)[:, None] * n
        return lap_time(line, speed_profile(line, v_min, v_max, a_lat, a_acc, a_brake))

    def fg(c, h=0.02):
        f0 = cost(c)
        g = np.zeros(K)
        for k in range(K):                                    # forward differences per bump
            cc = c.copy()
            cc[k] += h
            g[k] = (cost(cc) - f0) / h
        return f0, g

    t0 = cost(np.zeros(K))
    res = minimize(fg, np.zeros(K), jac=True, method='L-BFGS-B', bounds=[(-1.0, 1.0)] * K,
                   options={'maxiter': iters})
    c = res.x if res.fun < t0 else np.zeros(K)
    off = offsets(c)
    return center + off[:, None] * n, off, min(res.fun, t0)
