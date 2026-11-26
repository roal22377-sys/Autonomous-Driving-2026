#!/usr/bin/env python3
"""Replay actual ScanLocalizer against simulated scan/pose truth, offline.

The policy drives independently of the tested estimator. This measures estimator
errors, not closed-loop fallback safety. Initialization uses the known true pose;
startup ambiguity and map reconstruction error require separate tests.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_stable_time(t, good, after, seconds=.5):
    """First observed start of a continuously good interval of sufficient length."""
    start = None
    for now, ok in zip(t, good):
        if now < after:
            continue
        if not ok:
            start = None
        elif start is None:
            start = float(now)
        if start is not None and now - start >= seconds - 1e-9:
            return start
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model', default='ppo_s2r/25')
    p.add_argument('--tag', default='3080000')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--cap', type=float, default=8.)
    p.add_argument('--fault-time', type=float, default=3.)
    cli = p.parse_args()
    os.chdir(SRC)
    import numpy as np
    import random
    import torch
    import train_il_s2r as IL
    from utils.env import F1Wrapper
    from utils.vectorize import RunningMeanStd
    from utils.localizer import ScanLocalizer, course_candidates, wrap, read_map
    from audit_localizer_recovery_1126 import invalid_intervals
    dest = Path(cli.output_dir).resolve()
    if dest.exists() or SRC / 'installation' in dest.parents:
        raise ValueError('Use a new isolated results directory')
    dest.mkdir(parents=True)
    torch.set_num_threads(1)
    np.random.seed(cli.seed)
    random.seed(cli.seed)
    args = IL.get_args(['--task_config', 'results/' + cli.model + '/backup/f1tenth.yaml'])
    args.seed = cli.seed
    checkpoint = Path('results/' + cli.model + '/checkpoint/model_' + cli.tag + '.pt')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    actor = IL.actor_from_state(ck['actor'])
    actor.eval()
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load('results/' + cli.model + '/sim2real', required=True)
    path = str(Path(cli.track).resolve())
    env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode='eval', tracks=[path])
    env.fixed_speed_cap = cli.cap
    env.start_mode = 'center'
    frames, truth, scans = [], [], []
    try:
        obs, _ = env.reset(options={'track': path})
        while True:
            with torch.no_grad():
                action = actor(torch.as_tensor(rms.normalize(obs), dtype=torch.float32)).loc.numpy()
            obs, _, term, trunc, info = env.step(action)
            st = env.vehicle_state()
            frames.append(float(env.elapsed))
            truth.append([st[0], st[1], st[4], st[3]])
            # Read an existing scan: never raycast again or consume simulator RNG.
            scans.append(np.asarray(info['obs_dict']['scans'][0], np.float32).copy())
            if term or trunc:
                break
        record = dict(complete=bool(env.success), elapsed=float(env.elapsed), seed=cli.seed,
                      cap=cli.cap, checkpoint_sha256=sha(checkpoint), track_sha256=sha(path))
        car = env._env.unwrapped.sim.agents[0]
        laser_x = float(car.lidar_pos_offset)
        angles = np.asarray(car.scan_angles, float).copy()
    finally:
        env.close()
    t, true, ranges = np.asarray(frames), np.asarray(truth), np.asarray(scans)
    assert ranges.shape[1] == len(angles)
    assert np.all(np.diff(t) > 0)
    if not record['complete'] or t[-1] < cli.fault_time + 6.:
        raise ValueError('Need a complete trajectory with enough post-fault observations')
    # Deliberately imperfect odometry; assumptions, not fitted vehicle parameters.
    rng = np.random.default_rng(cli.seed + 991)
    dt = np.diff(t, prepend=t[0])
    yr = np.gradient(np.unwrap(true[:, 2]), t)
    yaw = true[0, 2] + np.cumsum((.85 * yr + .02 + rng.normal(0, .02, len(t))) * dt)
    speed = 1.03 * true[:, 3]
    odom = np.column_stack([true[0, 0] + np.cumsum(speed * np.cos(yaw) * dt),
                           true[0, 1] + np.cumsum(speed * np.sin(yaw) * dt), wrap(yaw)])
    np.savez_compressed(dest / 'inputs.npz', t=t, truth=true, ranges=ranges, angles=angles, odom=odom)
    # Export the exact simulation grid, retaining its metric frame and y direction.
    map_dir = dest / 'map'
    map_dir.mkdir()
    with np.load(path, allow_pickle=False) as z:
        image = z['occupancy'].astype(np.uint8)
        origin = np.asarray(z['origin'], float)
        resolution = float(z['resolution'])
    h, w = image.shape
    (map_dir / 'map.pgm').write_bytes(('P5\n%d %d\n255\n' % (w, h)).encode() + image[::-1].tobytes())
    (map_dir / 'map.yaml').write_text('image: map.pgm\nresolution: %r\norigin: [%r, %r, 0.0]\n' %
                                     (resolution, float(origin[0]), float(origin[1])))
    pts, res, org = read_map(str(map_dir))
    iy, ix = np.nonzero(image < 50)
    expected = np.column_stack([origin[0] + (ix + .5) * resolution, origin[1] + (iy + .5) * resolution])
    assert np.array_equal(pts, expected) and res == resolution and np.array_equal(org, origin[:2])
    candidates = course_candidates(path)
    conditions = [('control', 0., None), ('scan_loss2', 2., None), ('scan_loss4', 4., None),
                  ('wrong_pose_small', 0., [.8, 0., .35]), ('wrong_pose_large', 0., [6., 2., 2.])]
    report = dict(status='running', recording=record, cases=[],
        inputs_sha256=sha(dest / 'inputs.npz'), localizer_sha256=sha(SRC / 'utils/localizer.py'),
        fault_time_requested=cli.fault_time, laser_offset_m=laser_x,
        odometry_assumptions=dict(speed_scale=1.03, yaw_rate_scale=.85, yaw_rate_bias=.02, yaw_rate_noise_std=.02),
        qualification='Offline fixed motion. Exact known grid and true initial CG pose; laser offset uses simulation CG convention, not ROS base_link. Tests actual ScanLocalizer recovery and false-valid errors, not startup, map-error robustness, fallback policy or closed-loop safety.',
        stable_recovery_definition='valid AND position error <=.2m AND yaw error <=5deg continuously >=.5s',
        false_valid_definition='valid AND (position error >.5m OR yaw error >15deg); diagnostic thresholds, not certified safety limits')
    for name, duration, kick in conditions:
        loc = ScanLocalizer(str(map_dir), laser_x=laser_x)
        loc.candidates = candidates.copy()
        loc.reset(true[0, :3], odom[0])
        poses, valid, runtime = [], [], []
        injected = False
        onset = float(t[np.searchsorted(t, cli.fault_time)])
        for i, now in enumerate(t):
            if kick is not None and now >= cli.fault_time and not injected:
                loc.pose += np.asarray(kick)
                loc.pose[2] = wrap(loc.pose[2])
                injected = True
            scan = np.full_like(ranges[i], np.inf) if cli.fault_time <= now < cli.fault_time + duration else ranges[i]
            begin = time.perf_counter()
            pose, ok, _ = loc.step(scan, angles, odom[i])
            runtime.append(time.perf_counter() - begin)
            poses.append(pose)
            valid.append(ok)
        poses, valid = np.asarray(poses), np.asarray(valid, bool)
        error = np.linalg.norm(poses[:, :2] - true[:, :2], axis=1)
        yaw_error = np.abs(wrap(poses[:, 2] - true[:, 2]))
        correct = valid & (error <= .2) & (yaw_error <= np.deg2rad(5))
        false_valid = valid & ((error > .5) | (yaw_error > np.deg2rad(15)))
        after = cli.fault_time + duration
        stable = first_stable_time(t, correct, after)
        intervals = invalid_intervals(t, valid)
        np.savez_compressed(dest / (name + '.npz'), t=t, pose=poses, valid=valid,
                            position_error=error, yaw_error=yaw_error, false_valid=false_valid)
        row = dict(condition=name, fault_onset_observed=onset if name != 'control' else None,
            valid_fraction=float(valid.mean()), correct_valid_fraction=float(correct.mean()),
            position_error_median_m=float(np.median(error)), position_error_p95_m=float(np.quantile(error, .95)),
            position_error_max_m=float(error.max()), false_valid_scans=int(false_valid.sum()),
            post_event_false_valid_scans=int(false_valid[t >= onset].sum()),
            stable_recovery_time=stable, stable_recovery_after_fault_end_s=None if stable is None else stable - after,
            no_stable_recovery_observed=stable is None, invalid_intervals=intervals,
            runtime_p99_ms=float(np.quantile(runtime, .99) * 1000))
        report['cases'].append(row)
        (dest / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    report['status'] = 'complete'
    (dest / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
