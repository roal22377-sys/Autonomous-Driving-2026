#!/usr/bin/env python3
"""One-setting-at-a-time estimator recovery ablations on frozen simulator data."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/f1tenth'))
from utils.localizer import ScanLocalizer, course_candidates, wrap
from benchmark_localizer_truth_1126 import sha, first_stable_time
from audit_localizer_recovery_1126 import invalid_intervals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording-dir', required=True)
    p.add_argument('--track', required=True)
    p.add_argument('--output-dir', required=True)
    cli = p.parse_args()
    source, dest = Path(cli.recording_dir).resolve(), Path(cli.output_dir).resolve()
    report = json.loads((source / 'report.json').read_text())
    assert report['status'] == 'complete'
    assert sha(cli.track) == report['recording']['track_sha256']
    assert sha(source / 'inputs.npz') == report['inputs_sha256']
    assert sha(ROOT / 'src/f1tenth/utils/localizer.py') == report['localizer_sha256']
    if dest.exists():
        raise FileExistsError('Use a new output directory')
    dest.mkdir(parents=True)
    with np.load(source / 'inputs.npz') as z:
        t, true, scans, angles, odom = [z[k].copy() for k in ('t', 'truth', 'ranges', 'angles', 'odom')]
    candidates = course_candidates(cli.track)
    event_time = report['fault_time_requested']
    conditions = [('control', 0., None), ('scan_loss2', 2., None), ('scan_loss4', 4., None),
                  ('wrong_pose_small', 0., [.8, 0., .35]), ('wrong_pose_large', 0., [6., 2., 2.])]
    settings = [('current', {}), ('confirm_local40', {'confirm_local': 40}),
                ('confirm_local80', {'confirm_local': 80}),
                ('confirm_cons02', {'confirm_cons': .2}),
                ('refind08', {'explained_refind': .8})]
    audit = dict(status='running', cases=[], source_report_sha256=sha(source / 'report.json'),
        input_sha256=report['inputs_sha256'], track_sha256=sha(cli.track),
        map_sha256={name: sha(source / 'map' / name) for name in ('map.yaml', 'map.pgm')},
        localizer_sha256=report['localizer_sha256'], settings=dict(settings),
        qualification='Frozen motion and exact simulation map; exploratory diagnostic tuning set. Each alternative changes one existing localizer parameter only. Runtime configuration and learned policies are unchanged. Validate any candidate on other geometries and recorded car logs before adoption.')
    weights = np.diff(t, append=t[-1])
    def save():
        tmp = dest / 'report.tmp.json'
        tmp.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        tmp.replace(dest / 'report.json')
    for label, cfg in settings:
        for name, duration, kick in conditions:
            loc = ScanLocalizer(str(source / 'map'), laser_x=report['laser_offset_m'], **cfg)
            loc.candidates = candidates.copy()
            loc.reset(true[0, :3], odom[0])
            poses, valid, runtimes, transitions = [], [], [], []
            injected, last_valid, last_false = False, True, False
            for i, now in enumerate(t):
                if kick is not None and now >= event_time and not injected:
                    loc.pose += np.asarray(kick)
                    loc.pose[2] = wrap(loc.pose[2])
                    injected = True
                scan = np.full_like(scans[i], np.inf) if event_time <= now < event_time + duration else scans[i]
                start = time.perf_counter()
                pose, ok, info = loc.step(scan, angles, odom[i])
                runtimes.append(time.perf_counter() - start)
                poses.append(pose)
                valid.append(ok)
                error = float(np.linalg.norm(pose[:2] - true[i, :2]))
                yaw_error = float(abs(wrap(pose[2] - true[i, 2])))
                false = bool(ok and (error > .5 or yaw_error > np.deg2rad(15)))
                if ok != last_valid or false != last_false or info.get('recovered'):
                    transitions.append(dict(t=float(now), valid=bool(ok), false_valid=false,
                        position_error_m=error, yaw_error_deg=float(np.degrees(yaw_error)),
                        recovered=bool(info.get('recovered')), confirming=int(loc.confirm),
                        constraint=float(loc.last_cons), explained_ema=float(loc.ema),
                        explained_now=float(loc.explained(loc.points(scan, angles), pose)),
                        residual=float(info['res']) if np.isfinite(info.get('res', np.inf)) else None,
                        inliers=int(info.get('inliers', 0))))
                last_valid, last_false = ok, false
            poses, valid = np.asarray(poses), np.asarray(valid, bool)
            error = np.linalg.norm(poses[:, :2] - true[:, :2], axis=1)
            yaw_error = np.abs(wrap(poses[:, 2] - true[:, 2]))
            correct = valid & (error <= .2) & (yaw_error <= np.deg2rad(5))
            false = valid & ((error > .5) | (yaw_error > np.deg2rad(15)))
            stable = first_stable_time(t, correct, event_time + duration)
            if label == 'current':
                with np.load(source / (name + '.npz')) as original:
                    assert np.array_equal(poses, original['pose']) and np.array_equal(valid, original['valid'])
            np.savez_compressed(dest / (label + '_' + name + '.npz'), t=t, pose=poses,
                                valid=valid, position_error=error, yaw_error=yaw_error)
            row = dict(setting=label, condition=name, valid_fraction=float(valid.mean()),
                correct_valid_fraction=float(correct.mean()), false_valid_scans=int(false.sum()),
                false_valid_observed_seconds=float(weights[false].sum()),
                stable_recovery_time=stable,
                stable_recovery_after_fault_end_s=None if stable is None else stable - event_time - duration,
                invalid_intervals=invalid_intervals(t, valid),
                false_valid_intervals=invalid_intervals(t, ~false),
                position_error_p95_m=float(np.quantile(error, .95)),
                runtime_p99_ms=float(np.quantile(runtimes, .99) * 1000), transitions=transitions)
            audit['cases'].append(row)
            save()
            print(label, name, 'false-valid scans', row['false_valid_scans'],
                  'stable recovery after end', row['stable_recovery_after_fault_end_s'], flush=True)
    audit.update(status='complete', current_replay_bitwise_matches_all_five_original_conditions=True)
    save()


if __name__ == '__main__':
    main()
