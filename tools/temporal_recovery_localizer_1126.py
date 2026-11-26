"""SIMULATION-ONLY delayed recovery prototype; never imported by vehicle code.

Normal tracking delegates to the unchanged ScanLocalizer. After loss, a full
search is modeled as a delayed result while the vehicle continues on fallback.
Four globally ranked modes are then checked against subsequent scans. Fixed
latency is an experimental assumption, not measured real-time scheduling.
"""
import time
import numpy as np

from utils.localizer import ScanLocalizer, compose, relative, icp, wrap
from analyze_localizer_hypotheses_1126 import rank_candidates


class TemporalRecoveryLocalizer(ScanLocalizer):
    def __init__(self, *args, search_delay=.6, margin=.05, confirmation=.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not all(np.isfinite(x) and x >= 0 for x in (search_delay, margin, confirmation)):
            raise ValueError('Invalid temporal recovery settings')
        self.search_delay, self.margin, self.confirmation = search_delay, margin, confirmation
        self.now = 0.
        self.recovering = False
        self.pending = None
        self.hypotheses = []
        self.frames = 0
        self.leader, self.leader_since = None, None
        self.released_at = None
        self.events = []
        self.tracking_runtime_ms = []

    def set_time(self, seconds):
        if not np.isfinite(seconds) or seconds < self.now:
            raise ValueError('Simulation time must be finite and monotonic')
        self.now = float(seconds)

    def _search(self, ranges, angles, odom):
        initial = np.vstack([self.pose, self.candidates])
        start = time.perf_counter()
        ranked, accepted = rank_candidates(self, ranges, angles, initial)
        runtime_ms = (time.perf_counter()-start)*1000
        self.pending = dict(release=self.now+self.search_delay, odom=odom.copy(), rows=ranked[:4])
        self.events.append(dict(kind='search', time=self.now, release_time=self.pending['release'],
            full_seed_count=len(initial), accepted_seeds=accepted, distinct_modes=len(ranked),
            retained_modes=len(ranked[:4]), actual_compute_ms=runtime_ms,
            compute_exceeded_assumed_delay=runtime_ms > self.search_delay*1000))

    def step(self, ranges, angles, odom):
        odom = np.asarray(odom, float).copy()
        if not np.isfinite(odom).all():
            odom = self.last_odom.copy() if self.last_odom is not None else np.zeros(3)
        if not self.recovering:
            pose, valid, info = super().step(ranges, angles, odom)
            if valid or pose is None:
                return pose, valid, info
            self.recovering = True
            self.events.append(dict(kind='loss', time=self.now))
            # Parent already predicted this scan, so no second odometry step.
            delta = np.zeros(3)
        else:
            delta = relative(self.last_odom, odom)
            self.pose = compose(self.pose, delta)
            self.last_odom = odom.copy()
            self.stats['steps'] += 1
        self.valid, self.lost = False, True
        src = self.points(ranges, angles)
        if len(src) < 50:
            # A pending search must not recover on an absent current scan.
            self.pending, self.hypotheses = None, []
            self.leader, self.leader_since = None, None
            return self.pose.copy(), False, dict(temporal='no_scan')
        if self.pending is None and not self.hypotheses:
            self._search(ranges, angles, odom)
        if self.pending is not None:
            if self.now < self.pending['release']-1e-9:
                return self.pose.copy(), False, dict(temporal='search_delay')
            motion = relative(self.pending['odom'], odom)
            self.hypotheses = [dict(id=i, pose=compose(np.asarray(r['pose']), motion), total=0.)
                               for i,r in enumerate(self.pending['rows'])]
            self.pending = None
            self.frames = 0
            self.released_at = self.now
            self.leader, self.leader_since = None, None
            delta = np.zeros(3)  # capture-to-now motion was applied above
        begin = time.perf_counter()
        self.frames += 1
        for h in self.hypotheses:
            prediction = compose(h['pose'], delta)
            pose, residual, inliers, _ = icp(src, self.pts, self.tree, self.n, self.ok,
                prediction, iters=self.cfg['iters'])
            explained = self.explained(src, pose)
            good = (residual < self.cfg['res_ok'] and inliers >= self.cfg['min_inliers']
                    and np.linalg.norm(pose[:2]-prediction[:2]) <= self.cfg['jump_max']
                    and explained >= self.cfg['explained_refind'])
            h.update(pose=pose if good else prediction, good=bool(good),
                     total=h['total']+(float(explained-2*residual) if good else 0.))
        ranked = sorted(self.hypotheses, key=lambda h:h['total'], reverse=True)
        distinct = []
        for h in ranked:
            if all(np.linalg.norm(h['pose'][:2]-other['pose'][:2]) > .5 or
                   abs(wrap(h['pose'][2]-other['pose'][2])) > np.deg2rad(20) for other in distinct):
                distinct.append(h)
        best = distinct[0] if distinct else None
        # No surviving rival is an experimental heuristic, not global uniqueness.
        gap = ((best['total']-distinct[1]['total'])/self.frames if len(distinct)>1
               else 1. if best is not None else 0.)
        qualifies = best is not None and best['good'] and gap >= self.margin
        if not qualifies:
            self.leader, self.leader_since = None, None
        elif self.leader != best['id']:
            self.leader, self.leader_since = best['id'], self.now
        self.tracking_runtime_ms.append((time.perf_counter()-begin)*1000)
        if self.leader_since is not None and self.now-self.leader_since >= self.confirmation-1e-9:
            accepted_pose = best['pose'].copy()
            self.events.append(dict(kind='accepted', time=self.now, pose=accepted_pose.tolist(),
                frames=self.frames, mean_score=best['total']/self.frames, gap=gap,
                retained_modes=len(self.hypotheses), distinct_modes=len(distinct)))
            super().reset(accepted_pose, odom)
            self.recovering = False
            self.hypotheses = []
            return self.pose.copy(), True, dict(temporal='accepted', recovered=True)
        if self.released_at is not None and self.now-self.released_at >= 2.-1e-9:
            # A stale/pruned set is not allowed to remain the only set forever.
            self.events.append(dict(kind='retry', time=self.now))
            self.hypotheses = []
            self.released_at = None
        return self.pose.copy(), False, dict(temporal='confirming', gap=gap)
