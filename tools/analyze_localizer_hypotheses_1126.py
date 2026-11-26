#!/usr/bin/env python3
"""Offline full-course hypothesis sweep; truth only annotates ranked results.

This is a recoverability diagnostic, not a real-time estimator implementation.
It deliberately measures the full search cost before considering an online form.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/f1tenth'))
from utils.localizer import ScanLocalizer, course_candidates, icp, wrap, compose, relative
from benchmark_localizer_truth_1126 import sha


def rank_candidates(loc, scan, angles, candidates):
    """Uses only current scan, map and supplied map/estimator hypotheses."""
    src = loc.points(scan, angles)
    ranked = []
    for index, initial in enumerate(candidates):
        pose, residual, inliers, constraint = icp(src, loc.pts, loc.tree, loc.n, loc.ok,
            initial, iters=12, d_max=(1., .5, .25, .12))
        if residual >= loc.cfg['res_ok'] or inliers < loc.cfg['min_inliers'] * 1.5:
            continue
        explained = loc.explained(src, pose)
        if explained < loc.cfg['explained_refind']:
            continue
        ranked.append(dict(index=index, pose=pose.tolist(), residual=float(residual),
            inliers=int(inliers), constraint=float(constraint), explained=float(explained),
            score=float(explained - 2 * residual)))
    ranked.sort(key=lambda r: r['score'], reverse=True)
    distinct = []
    for row in ranked:
        pose = np.asarray(row['pose'])
        if all(np.linalg.norm(pose[:2] - np.asarray(other['pose'])[:2]) > .5 or
               abs(wrap(pose[2] - other['pose'][2])) > np.deg2rad(20) for other in distinct):
            distinct.append(row)
    return distinct, len(ranked)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording-dir', required=True)
    p.add_argument('--track', required=True)
    p.add_argument('--condition', default='scan_loss4')
    p.add_argument('--times', nargs='+', type=float, required=True)
    p.add_argument('--candidate-limit', type=int, default=0,
                   help='0: full sweep; positive: nearest map seeds by existing position+heading prior')
    p.add_argument('--output', required=True)
    cli = p.parse_args()
    root = Path(cli.recording_dir)
    metadata = json.loads((root / 'report.json').read_text())
    assert metadata['status'] == 'complete'
    assert sha(root / 'inputs.npz') == metadata['inputs_sha256']
    assert sha(cli.track) == metadata['recording']['track_sha256']
    assert sha(ROOT / 'src/f1tenth/utils/localizer.py') == metadata['localizer_sha256']
    with np.load(root / 'inputs.npz') as z:
        t, truth, scans, angles, odom = [z[k].copy() for k in ('t', 'truth', 'ranges', 'angles', 'odom')]
    with np.load(root / (cli.condition + '.npz')) as z:
        estimate = z['pose'].copy()
    loc = ScanLocalizer(str(root / 'map'), laser_x=metadata['laser_offset_m'])
    seeds = course_candidates(cli.track)
    result = dict(status='running', source_report_sha256=sha(root / 'report.json'),
        input_sha256=metadata['inputs_sha256'], candidate_limit=cli.candidate_limit, cases=[],
        localizer_sha256=metadata['localizer_sha256'], estimate_sha256=sha(root / (cli.condition + '.npz')),
        qualification='Offline scan matching: no truth in candidate generation or ranking. Exact simulation map; an optional candidate limit uses the odometry-predicted position+heading prior. No temporal hypothesis verification or closed-loop control. A score gap inside a pruned candidate set does not prove global uniqueness. Runtime is measured, not assumed to fit 40Hz.')
    dest = Path(cli.output)
    if dest.exists():
        raise FileExistsError('Do not overwrite a diagnostic')
    for requested in cli.times:
        index = int(np.searchsorted(t, requested - 1e-9))
        if index >= len(t):
            raise ValueError('Time outside recorded trajectory')
        # The currently estimated pose is an observable candidate, never truth.
        indices = np.arange(len(seeds))
        if cli.candidate_limit > 0:
            prior = compose(estimate[index-1], relative(odom[index-1], odom[index])) if index else estimate[0]
            distance = np.linalg.norm(seeds[:, :2] - prior[:2], axis=1) + abs(wrap(seeds[:, 2] - prior[2]))
            indices = np.argsort(distance)[:cli.candidate_limit]
        candidates = np.vstack([estimate[index], seeds[indices]])
        begin = time.perf_counter()
        ranked, accepted = rank_candidates(loc, scans[index], angles, candidates)
        runtime_ms = (time.perf_counter() - begin) * 1000
        # Annotation starts AFTER ranking. Never feed these errors to selection.
        for row in ranked:
            pose = np.asarray(row['pose'])
            row['course_seed_index'] = int(indices[row['index']-1]) if row['index'] else None
            row['truth_error_m'] = float(np.linalg.norm(pose[:2] - truth[index, :2]))
            row['truth_yaw_error_deg'] = float(np.degrees(abs(wrap(pose[2] - truth[index, 2]))))
        row = dict(requested_time=requested, observed_time=float(t[index]),
            candidates=len(candidates), accepted_candidates=accepted, distinct_hypotheses=len(ranked),
            baseline_error_m=float(np.linalg.norm(estimate[index, :2] - truth[index, :2])),
            search_runtime_ms=float(runtime_ms), top_hypotheses=ranked[:10])
        result['cases'].append(row)
        dest.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(time=row['observed_time'], baseline_error_m=row['baseline_error_m'],
            candidates=row['candidates'], search_runtime_ms=runtime_ms,
            best=ranked[0] if ranked else None)), flush=True)
    result['status'] = 'complete'
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
