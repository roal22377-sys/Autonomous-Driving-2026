'''
    Scan-to-map localization for plan B (MAP_TOOL.md section 4): the car's pose on a map made by
    tools/map_offline.py, at every scan (40 Hz).

        prediction  wheel odometry: the odom motion since the last scan, applied to the last pose
        correction  point-to-line ICP of the scan (every `stride`-th beam) onto the map's wall cells
        check       accepted when the mean residual is small, enough beams match and the correction is
                    plausible; otherwise the prediction is kept and the pose counts as "not valid" after
                    `hold` scans in a row (the policy then gets valid = 0 and drives on LiDAR alone)
        start       global_init(): ICP from candidate poses along the course (every 0.5 m, both headings)

    ROS-free, so the car (agent_node) and the offline test (tools/loc_test.py) run the same code.
    The ICP is the one of tools/map_offline.py (copied so the car does not import the tools).
    Poses are base_link poses in the map frame (laser `laser_x` m ahead of base_link, as in the map tool).
'''
import os

import numpy as np
from scipy.spatial import cKDTree


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def relative(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])


def transform(pts, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + pose[:2]


def read_map(map_dir):
    '''map.yaml + map.pgm of tools/map_offline.py -> (occupied cell centers (N, 2) [m], resolution, origin)'''
    meta = {}
    for line in open(os.path.join(map_dir, 'map.yaml')):
        if ':' in line:
            k, v = line.split(':', 1)
            meta[k.strip()] = v.strip()
    res = float(meta['resolution'])
    org = np.array([float(x) for x in meta['origin'].strip('[]').split(',')][:2])
    raw = open(os.path.join(map_dir, meta.get('image', 'map.pgm')), 'rb').read()
    hdr = raw.split(b'\n', 3)
    W, H = map(int, hdr[1].split())
    img = np.frombuffer(hdr[3][:W * H], np.uint8).reshape(H, W)[::-1]       # row 0 = lowest y
    iy, ix = np.nonzero(img < 50)
    return np.column_stack([org[0] + (ix + 0.5) * res, org[1] + (iy + 0.5) * res]), res, org


def normals(pts, tree, k=16, flat=0.5):
    '''wall cells of a 5 cm grid: k 16 / flat 0.5 keeps 65 % of the cells (11/24 map; k 10 / 0.3: 31 %, the
    scan then matched only 15-45 % of its beams). Same accuracy (3.5 cm median vs the map tool's poses)'''
    _, idx = tree.query(pts, k=min(k, len(pts)))
    nb = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum('nki,nkj->nij', nb, nb))
    return v[:, :, 0], w[:, 0] / (w[:, 1] + 1e-9) < flat


def icp(src, tgt, tree, tgt_n, tgt_ok, init, iters=10, d_max=(0.5, 0.25, 0.12)):
    '''point-to-line ICP (tools/map_offline.py). returns pose, mean |residual| [m], inliers, constraint'''
    pose = np.array(init, float)
    res, n_in, cons = np.inf, 0, 0.0
    for it in range(iters):
        dm = d_max[min(it * len(d_max) // iters, len(d_max) - 1)]
        P = transform(src, pose)
        d, j = tree.query(P, distance_upper_bound=dm)
        m = np.isfinite(d)
        m[m] = tgt_ok[j[m]]
        if m.sum() < 30:
            return pose, np.inf, 0, 0.0
        p, q, n = P[m], tgt[j[m]], tgt_n[j[m]]
        r = np.einsum('ij,ij->i', p - q, n)
        keep = np.abs(r) <= np.quantile(np.abs(r), 0.85)
        p, n, r = p[keep], n[keep], r[keep]
        rel = p - pose[:2]
        J = np.column_stack([n[:, 0], n[:, 1], n[:, 0] * -rel[:, 1] + n[:, 1] * rel[:, 0]])
        H = J.T @ J + np.eye(3) * 1e-3 * len(r)
        dx = -np.linalg.solve(H, J.T @ r)
        pose = np.array([pose[0] + dx[0], pose[1] + dx[1], wrap(pose[2] + dx[2])])
        res, n_in = float(np.mean(np.abs(r))), int(keep.sum())
        if np.abs(dx[:2]).max() < 2e-4 and abs(dx[2]) < 2e-4:
            break
    Ht = J[:, :2].T @ J[:, :2]
    cons = float(np.linalg.eigvalsh(Ht)[0] / (np.trace(Ht) + 1e-9))
    return pose, res, n_in, cons


class ScanLocalizer:
    DEFAULT = dict(stride=3, r_min=0.10, r_max=10.0, iters=10,
                   res_ok=0.03,         # [m] mean |point-to-line residual| of an accepted match (11/24 log: 1.0 cm median)
                   min_inliers=60,      # matched beams (of ~360; 11/24 log: 100-300)
                   jump_max=0.5,        # [m] correction vs the odom prediction above this -> rejected
                   hold=8,              # scans in a row without an accepted match before valid = False (0.2 s)
                   local_tries=14,      # lost: this many scans of local search (one ICP per scan), then the
                                        # start candidates along the course nearest the prediction first, one per scan
                   explained_ok=0.8,    # re-found pose: share of the scan within 10 cm of a map wall (correct poses:
                                        # median 1.00; wrong ones that passed residual + inliers: median 0.53, 11/24 log)
                   confirm_local=8,     # ... then tracked this many scans before it counts as valid (0.2 s)
                   confirm_global=40,   # ... 1 s when found by the course-wide search (a straight looks the same
                                        # a few meters further: one scan cannot tell, the motion through a corner can)
                   ambiguous_gap=0.05,  # start: another pose > 1.5 m away explaining the scan within this -> refuse
                   confirm_cons=0.1,    # ... and only scans that fix both directions count (not along a straight:
                                        # a pose re-found 1 m off along a straight stayed "right" there, sim test 11/25)
                   explained_track=0.45)  # tracking: running mean (10 scans) of the explained share below this -> lost.
                                        # 11/24 log, correct poses: never below 0.58; a wrong pose that kept matching
                                        # (forced 6 m / 115 deg off): median 0.33

    def __init__(self, map_dir, laser_x=0.13, **cfg):
        self.cfg = dict(self.DEFAULT, **cfg)
        self.laser_x = float(laser_x)
        self.pts, self.res, self.org = read_map(map_dir)
        self.tree = cKDTree(self.pts)
        self.n, self.ok = normals(self.pts, self.tree)
        self.pose, self.last_odom = None, None
        self.bad, self.valid, self.confirm = 0, False, 0
        self.lost, self.k_lost, self.ema = False, 0, 1.0
        self.last_cons = 0.0
        self.candidates, self.sweep = None, None
        self.stats = dict(steps=0, accepted=0, recovered=0)

    # -------------------------------------------------------------- scan -> points (base_link frame)
    def points(self, ranges, angles):
        st = self.cfg['stride']
        r, a = np.asarray(ranges, float)[::st], np.asarray(angles, float)[::st]
        ok = np.isfinite(r) & (r > self.cfg['r_min']) & (r < self.cfg['r_max'])
        return np.column_stack([self.laser_x + r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])])

    def explained(self, src, pose, d=0.10):
        '''share of the scan points within d of a map wall cell'''
        dist, _ = self.tree.query(transform(src, pose), distance_upper_bound=d)
        return float(np.isfinite(dist).mean()) if len(dist) else 0.0

    def match(self, src, init):
        c = self.cfg
        pose, res, n_in, cons = icp(src, self.pts, self.tree, self.n, self.ok, init, iters=c['iters'])
        self.last_cons = cons
        good = res < c['res_ok'] and n_in >= c['min_inliers']
        return pose, res, n_in, good

    YAW_TRIES = (0.0, 0.1, -0.1, 0.25, -0.25, 0.45, -0.45)

    def recover(self, src, pred, k):
        '''lost for k scans past `hold`: ONE wide ICP per scan, so a lost scan costs about what a normal one does
        (all tries in one scan took 19 ms on the PC, 11/25: the policy would have slowed from 40 Hz while lost).
        First around the prediction with heading offsets (the odom heading drifts), then the course candidates.'''
        local = k < self.cfg['local_tries'] or self.candidates is None
        if local:
            p0 = pred + np.array([0.0, 0.0, self.YAW_TRIES[k % len(self.YAW_TRIES)]])
        else:
            j = (k - self.cfg['local_tries']) % len(self.candidates)
            if j == 0 or self.sweep is None:      # a new sweep: nearest to the prediction first (position, then heading)
                c = self.candidates
                d = np.hypot(c[:, 0] - pred[0], c[:, 1] - pred[1]) + 1.0 * np.abs(wrap(c[:, 2] - pred[2]))
                self.sweep = np.argsort(d)
            p0 = self.candidates[self.sweep[j]]
        p, res, n_in, _ = icp(src, self.pts, self.tree, self.n, self.ok, p0, iters=12, d_max=(1.0, 0.5, 0.25, 0.12))
        if (res < self.cfg['res_ok'] and n_in >= self.cfg['min_inliers'] * 1.5        # stricter than tracking
                and self.explained(src, p) >= self.cfg['explained_ok']):
            return p, res, n_in, self.cfg['confirm_local' if local else 'confirm_global']
        return None

    # -------------------------------------------------------------- start
    def global_init(self, ranges, angles, candidates, odom=None):
        '''best ICP match from candidate poses (N, 3), e.g. course_candidates(); sets the pose.
        returns (pose, residual, explained share, alternative): alternative = (pose, explained) of the best pose
        > 1.5 m / 45 deg away when it explains the scan almost as well (ambiguous start: valid stays False)'''
        src = self.points(ranges, angles)
        self.candidates = np.asarray(candidates, float)
        hyp = []
        for c0 in candidates:
            p, res, n_in, _ = icp(src, self.pts, self.tree, self.n, self.ok, c0, iters=6, d_max=(1.0, 0.5, 0.25))
            if np.isfinite(res) and res < 2 * self.cfg['res_ok'] and n_in >= self.cfg['min_inliers']:
                hyp.append((self.explained(src, p) - 2.0 * res, p))
        if not hyp:
            return None, np.inf, 0.0, None
        hyp.sort(key=lambda h: -h[0])
        p, res, n_in, good = self.match(src, hyp[0][1])
        ex = self.explained(src, p)
        alt = None
        for sc, q in hyp[1:]:
            if np.hypot(*(q[:2] - p[:2])) > 1.5 or abs(wrap(q[2] - p[2])) > np.pi / 4:
                if sc >= hyp[0][0] - self.cfg['ambiguous_gap']:
                    alt = (q, sc)
                break
        self.reset(p, odom)
        self.valid = bool(good) and ex >= self.cfg['explained_ok'] and alt is None
        return p, res, ex, alt

    def reset(self, pose, odom=None):
        self.pose = np.array(pose, float)
        self.last_odom = None if odom is None else np.array(odom, float)
        self.bad, self.valid, self.confirm = 0, True, 0
        self.lost, self.k_lost, self.ema = False, 0, 1.0

    # -------------------------------------------------------------- every scan
    def step(self, ranges, angles, odom):
        '''odom: (x, y, yaw) of the wheel odometry at this scan. returns (pose, valid, info)'''
        if self.pose is None:
            return None, False, {}
        odom = np.array(odom, float)
        pred = self.pose if self.last_odom is None else compose(self.pose, relative(self.last_odom, odom))
        self.last_odom = odom
        src = self.points(ranges, angles)
        self.stats['steps'] += 1
        info = dict(n=len(src))
        good, pose, res, n_in = False, pred, np.inf, 0
        if len(src) < 50:
            pass                                            # no usable scan: odom only
        elif not self.lost:                                 # tracking
            pose, res, n_in, good = self.match(src, pred)
            jump = float(np.hypot(*(pose[:2] - pred[:2])))
            info['jump'] = jump
            if jump > self.cfg['jump_max']:
                good = False
            if good:
                self.ema = 0.9 * self.ema + 0.1 * self.explained(src, pose)
                if self.ema < self.cfg['explained_track']:  # keeps matching, but the scan does not fit the map: wrong place
                    good, self.lost = False, True
                    info['mismatch'] = True
        else:                                               # lost: one search try per scan
            rec = self.recover(src, pred, self.k_lost)
            self.k_lost += 1
            if rec is not None:
                pose, res, n_in, self.confirm = rec
                good, self.lost, self.k_lost = True, False, 0
                self.ema = self.explained(src, pose)
                info['recovered'] = True
                self.stats['recovered'] += 1
        info.update(res=res, inliers=n_in, accepted=bool(good))
        if good:
            if not info.get('recovered') and self.last_cons >= self.cfg['confirm_cons']:
                self.confirm = max(self.confirm - 1, 0)
            self.pose, self.bad, self.sweep = pose, 0, None
            self.stats['accepted'] += 1
        else:
            self.pose, self.bad = pred, self.bad + 1
            if self.bad >= self.cfg['hold'] or self.confirm > 0:   # (a re-found pose that does not hold up: lost again)
                if not self.lost:
                    self.lost, self.k_lost = True, 0
                self.confirm = 0
        self.valid = not self.lost and self.bad < self.cfg['hold'] and self.confirm == 0
        info['confirming'] = self.confirm
        return self.pose.copy(), self.valid, info


def course_candidates(track_npz, step=0.5, yaw_offsets=(0.0, np.pi)):
    '''start candidates: the course centerline every `step` m, along it and reversed'''
    z = np.load(track_npz)
    c = np.asarray(z['centerline'], float)
    t = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    h = np.arctan2(t[:, 1], t[:, 0])
    ds = np.median(np.hypot(*np.diff(c, axis=0).T))
    k = max(int(round(step / max(ds, 1e-3))), 1)
    return np.array([[c[i, 0], c[i, 1], wrap(h[i] + o)] for i in range(0, len(c), k) for o in yaw_offsets])


class CarPreview:
    '''
        car side of plan B: localization on the course map + the track preview features (utils/preview.py),
        built exactly like the simulator's (utils/env.py). One per run.

        course_npz  the course from tools/map_to_track.py (map frame), in the driving direction; if the car
                    stands the other way round, <name>_rev.npz is used (and the switch is reported)
        laser_x     base_link -> laser on the car (bringup_launch.py: 0.13 m)
        lidar_pos_offset  the simulator's pose -> laser (dynamic.yaml: 0.17 m): the preview uses the same reference point
    '''

    def __init__(self, course_npz, map_dir, preview_cfg, laser_x=0.13, lidar_pos_offset=0.17, log=print, **loc_cfg):
        from utils.preview import TrackPreview
        self.log = log
        self.cfg = preview_cfg
        self.course = course_npz
        self.loc = ScanLocalizer(map_dir, laser_x=laser_x, **loc_cfg)
        self.shift = float(laser_x) - float(lidar_pos_offset)      # base_link -> sim pose reference, along x
        self.tp = TrackPreview(np.asarray(np.load(course_npz)['raceline'], float), preview_cfg)
        self.started = False
        self.n, self.n_valid = 0, 0

    def ref_pose(self, pose):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return pose[0] + self.shift * c, pose[1] + self.shift * s, pose[2]

    def start(self, ranges, angles, odom):
        '''global start along the course; returns (ok, message)'''
        from utils.preview import TrackPreview
        p, res, ex, alt = self.loc.global_init(ranges, angles, course_candidates(self.course), odom)
        if p is None:
            return False, 'localization: no start pose found (no candidate matched the scan)'
        if alt is not None:
            return False, (f'localization: ambiguous start, x {p[0]:.2f} y {p[1]:.2f} or x {alt[0][0]:.2f} y {alt[0][1]:.2f} '
                           'explain the scan about equally: move the car where it sees a corner')
        if not self.loc.valid:
            return False, f'localization: no reliable start pose (residual {res*100:.1f} cm, {ex*100:.0f}% of the scan on the map)'
        f = self.tp.features(*self.ref_pose(p), None, True)
        msg = (f'localization: start at x {p[0]:.2f} y {p[1]:.2f} yaw {np.degrees(p[2]):.0f} deg (residual {res*100:.1f} cm, '
               f'{ex*100:.0f}% of the scan on the map)')
        if abs(f[1]) > np.pi / 2:
            rev = self.course[:-4] + '_rev.npz' if not self.course.endswith('_rev.npz') else self.course[:-8] + '.npz'
            if not os.path.exists(rev):
                return False, msg + f'; the car faces against {os.path.basename(self.course)} and {rev} does not exist'
            self.course = rev
            self.tp = TrackPreview(np.asarray(np.load(rev)['raceline'], float), self.cfg)
            msg += f'; car faces the other way: using {os.path.basename(rev)}'
        self.started = True
        return True, msg

    def features(self, ranges, angles, odom, speed_cap=None):
        '''-> (preview features, base_link pose or None, valid, info)'''
        pose, valid, info = self.loc.step(ranges, angles, odom)
        self.n += 1
        self.n_valid += int(bool(valid))
        if pose is None:
            return self.tp.features(0, 0, 0, None, False), None, False, info
        return self.tp.features(*self.ref_pose(pose), speed_cap, valid), pose, valid, info
