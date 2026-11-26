#!/usr/bin/env python3
"""Offline temporal ranking of a full-course scan-matching hypothesis set.

Truth is used only after the estimator-only computation has returned. This is
not a production localization implementation or a closed-loop recovery test.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/f1tenth'))
from utils.localizer import ScanLocalizer, course_candidates, icp, compose, relative, wrap
from analyze_localizer_hypotheses_1126 import rank_candidates
from benchmark_localizer_truth_1126 import sha, first_stable_time


def temporal_rank(loc, scans, angles, odom, candidates, indices, max_hypotheses=0):
    """Only scans, imperfect odometry and map hypotheses enter the calculation."""
    begin = time.perf_counter()
    initial, accepted = rank_candidates(loc, scans[indices[0]], angles, candidates)
    init_ms = 1000 * (time.perf_counter() - begin)
    retained = initial[:max_hypotheses] if max_hypotheses else initial
    hypotheses = [dict(id=i, pose=np.asarray(r['pose']), total=r['score'], good=True,
                       score=r['score']) for i, r in enumerate(retained)]
    history = []
    for step, index in enumerate(indices):
        start = time.perf_counter()
        if step:
            src = loc.points(scans[index], angles)
            delta = relative(odom[indices[step-1]], odom[index])
            for h in hypotheses:
                prediction = compose(h['pose'], delta)
                pose, residual, inliers, _ = icp(src, loc.pts, loc.tree, loc.n, loc.ok,
                    prediction, iters=loc.cfg['iters'])
                explained = loc.explained(src, pose)
                good = (residual < loc.cfg['res_ok'] and inliers >= loc.cfg['min_inliers']
                        and np.linalg.norm(pose[:2]-prediction[:2]) <= loc.cfg['jump_max']
                        and explained >= loc.cfg['explained_refind'])
                # Failed matches contribute zero rather than disappearing. Their
                # previous cumulative confidence cannot survive a poor sequence.
                score = float(explained - 2 * residual) if good else 0.
                h.update(pose=pose if good else prediction, good=bool(good), score=score,
                         total=h['total'] + score)
        ranked = sorted(hypotheses, key=lambda h:h['total'], reverse=True)
        distinct = []
        for h in ranked:
            if all(np.linalg.norm(h['pose'][:2]-o['pose'][:2]) > .5 or
                   abs(wrap(h['pose'][2]-o['pose'][2])) > np.deg2rad(20) for o in distinct):
                distinct.append(h)
        top = [dict(id=h['id'], pose=h['pose'].tolist(), good=h['good'],
                    mean_score=float(h['total']/(step+1)), current_score=h['score']) for h in distinct[:3]]
        history.append(dict(index=int(index), hypotheses=len(hypotheses), distinct=len(distinct),
            runtime_ms=1000*(time.perf_counter()-start), top=top,
            mean_score_gap=top[0]['mean_score']-top[1]['mean_score'] if len(top)>1 else None))
    return dict(initial_search_ms=init_ms, initial_candidates=len(candidates),
                initially_accepted_seeds=accepted, initial_distinct_hypotheses=len(initial),
                retained_hypotheses=len(retained), initial_modes=initial, frames=history)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording-dir', required=True)
    p.add_argument('--track', required=True)
    p.add_argument('--condition', required=True)
    p.add_argument('--starts', nargs='+', type=float, required=True)
    p.add_argument('--duration', type=float, default=1.5)
    p.add_argument('--max-hypotheses', type=int, default=0,
                   help='0 keeps all modes; positive retains highest full-global-scan scores')
    p.add_argument('--output', required=True)
    cli = p.parse_args()
    root = Path(cli.recording_dir)
    dest = Path(cli.output)
    if dest.exists() or cli.duration <= 0 or cli.max_hypotheses < 0:
        raise ValueError('Use a new report path and positive window duration')
    meta = json.loads((root/'report.json').read_text())
    assert meta['status'] == 'complete'
    assert sha(root/'inputs.npz') == meta['inputs_sha256']
    assert sha(cli.track) == meta['recording']['track_sha256']
    assert sha(ROOT/'src/f1tenth/utils/localizer.py') == meta['localizer_sha256']
    with np.load(root/'inputs.npz') as z:
        t, truth, scans, angles, odom = [z[k].copy() for k in ('t','truth','ranges','angles','odom')]
    with np.load(root/(cli.condition+'.npz')) as z:
        estimates = z['pose'].copy()
    loc = ScanLocalizer(str(root/'map'), laser_x=meta['laser_offset_m'])
    seeds = course_candidates(cli.track)
    report = dict(status='running', source_report_sha256=sha(root/'report.json'),
        input_sha256=meta['inputs_sha256'], condition=cli.condition,
        estimate_sha256=sha(root/(cli.condition+'.npz')), localizer_sha256=meta['localizer_sha256'],
        implementation_sha256=sha(__file__),
        method='Full global search at window start; keep all accepted distinct modes or an explicit highest-score limit. Predict with odometry and locally refine retained modes each scan. Rank by average accepted explained-minus2residual score; invalid match contributes zero.',
        max_hypotheses=cli.max_hypotheses,
        windows=[], qualification='Offline fixed motion, exact simulation map. No truth in search, tracking or ranking. Initial hard match filtering and optional score-based pruning can omit the correct mode; no new hypotheses enter after initialization. Score gaps after pruning do not prove global uniqueness. No acceptance/validity policy, guaranteed runtime, full fallback loop or real-car qualification. Windows are diagnostic, not untouched tests.')
    for requested in cli.starts:
        start = int(np.searchsorted(t, requested-1e-9))
        if start >= len(t) or t[start]+cli.duration > t[-1]+1e-9:
            raise ValueError('Window outside recorded motion')
        indices = np.flatnonzero((np.arange(len(t))>=start)&(t<=t[start]+cli.duration+1e-9))
        if len(indices) < 2:
            raise ValueError('Window must contain at least two observed scans')
        result = temporal_rank(loc, scans, angles, odom, np.vstack([estimates[start], seeds]), indices,
                               max_hypotheses=cli.max_hypotheses)
        correct = []
        # Ground-truth annotation occurs only after all sequential ranking is done.
        true_ranks = [i+1 for i,h in enumerate(result['initial_modes'])
                      if np.linalg.norm(np.asarray(h['pose'])[:2]-truth[start,:2]) <= .2
                      and abs(wrap(h['pose'][2]-truth[start,2])) <= np.deg2rad(5)]
        result['best_true_initial_rank'] = min(true_ranks) if true_ranks else None
        result['true_initial_mode_retained'] = any(i<=result['retained_hypotheses'] for i in true_ranks)
        for row in result['frames']:
            index = row['index']; row['time']=float(t[index])
            for h in row['top']:
                pose = np.asarray(h['pose'])
                h['truth_error_m']=float(np.linalg.norm(pose[:2]-truth[index,:2]))
                h['truth_yaw_error_deg']=float(np.degrees(abs(wrap(pose[2]-truth[index,2]))))
            winner = row['top'][0] if row['top'] else None
            correct.append(bool(winner and winner['good'] and winner['truth_error_m']<=.2
                                and winner['truth_yaw_error_deg']<=5.))
        result.update(requested_start=requested, start_time=float(t[start]), duration=cli.duration,
            winner_correct_fraction=float(np.mean(correct)),
            first_correct_winner_sustained_half_second=first_stable_time(t[indices], correct, t[start]),
            tracking_runtime_p99_ms=float(np.quantile([r['runtime_ms'] for r in result['frames'][1:]], .99)),
            initial_winner=result['frames'][0]['top'][0] if result['frames'][0]['top'] else None,
            final_winner=result['frames'][-1]['top'][0] if result['frames'][-1]['top'] else None)
        report['windows'].append(result)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('frames', 'initial_modes')}),flush=True)
    report['status']='complete'
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
