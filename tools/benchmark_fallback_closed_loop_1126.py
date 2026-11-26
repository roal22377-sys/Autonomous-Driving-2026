#!/usr/bin/env python3
"""Offline closed-loop CarPreview + primary policy + optional LiDAR fallback.

Known true initialization and exact simulation map isolate recovery after startup.
This adapts the car's switch/history semantics to simulator command coordinates;
it does not execute ROS, actuator drivers, or real-time hardware scheduling.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = Path(__file__).resolve()
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]


def steering_scales(args):
    gain = float(args.get('steer_cmd_gain', 1.))
    return tuple(float(args['max_steer']) * float(args.get('steer_cmd_gain_' + side, gain))
                 for side in ('left', 'right'))


def convert_steering(value, source_scales, target_scales):
    side = 0 if value > 0 else 1
    return value * source_scales[side] / target_scales[side]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--condition', choices=['control', 'scan_loss4', 'wrong_pose_large'], required=True)
    p.add_argument('--fallback', choices=['on', 'off'], required=True)
    p.add_argument('--return-confirm', type=float, default=0.,
                   help='Experimental continuous-valid time before leaving fallback; production baseline is0')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--cap', type=float, default=8.)
    p.add_argument('--model', default='ppo_s2r/25')
    p.add_argument('--tag', default='3080000')
    p.add_argument('--recovery', choices=['baseline', 'temporal'], default='baseline')
    p.add_argument('--search-delay', type=float, default=.6,
                   help='Simulation-only delay before a temporal global-search result is available')
    cli = p.parse_args()
    if not 0. <= cli.return_confirm <= 5.:
        raise ValueError('return-confirm must be between0 and5 seconds')
    os.chdir(SRC)
    import numpy as np
    import torch
    from ruamel.yaml import YAML
    import train_il_s2r as IL
    from utils.env import F1Wrapper, normalize_speed, unnormalize_speed
    from utils.localizer import CarPreview, course_candidates, read_map, wrap
    from utils.observation import get_obs_config
    from utils.vectorize import RunningMeanStd
    from benchmark_localizer_truth_1126 import first_stable_time

    dest = Path(cli.output_dir).resolve()
    if dest.exists() or SRC / 'installation' in dest.parents:
        raise ValueError('Use a new isolated output directory')
    torch.set_num_threads(1)
    np.random.seed(cli.seed)
    random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    config = Path('results/' + cli.model + '/backup/f1tenth.yaml')
    metadata = json.loads((config.parent / 'ppo_args.json').read_text())
    if metadata.get('il') != 'imitation/1607':
        raise ValueError('This benchmark requires the frozen IL1607 normalizer lineage')
    args = IL.get_args(['--task_config', str(config)])
    args.seed = cli.seed
    ckpath = Path('results/' + cli.model + '/checkpoint/model_' + cli.tag + '.pt')
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    actor = IL.actor_from_state(ck['actor'])
    actor.eval()
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load('results/imitation/1607/sim2real', required=True)
    fbroot = Path('results/ppo_s2r/12190')
    fbargs = YAML().load(fbroot / 'backup/f1tenth.yaml')
    fbcfg = get_obs_config(type(args)(fbargs))
    pcfg = get_obs_config(args)
    canonical = lambda value: json.dumps(value, sort_keys=True, default=lambda x: x.tolist())
    for key in ('lidar', 'speed', 'history', 'speed_limit'):
        if canonical(pcfg[key]) != canonical(fbcfg[key]):
            raise ValueError('Shared sensor prefix requires matching ' + key)
    if fbcfg['preview']['enabled'] or fbargs['obs_dim'] != args.obs_dim - 18:
        raise ValueError('Expected LiDAR-only114 and preview132 policy pair')
    for key in ('min_speed', 'max_speed', 'start_kick'):
        if args.get(key) != fbargs.get(key):
            raise ValueError('Command conversion requires matching ' + key)
    fbactor = torch.load(fbroot / 'sim2real/actor.pt', map_location='cpu', weights_only=False)
    fbactor.eval()
    fbrms = RunningMeanStd('agent_obs', fbargs['obs_dim'])
    fbrms.load(str(fbroot / 'sim2real'), required=True)
    primary_scales, fallback_scales = steering_scales(args), steering_scales(fbargs)
    track = str(Path(cli.track).resolve())
    dest.mkdir(parents=True)
    source = Path(track)
    stem = source.stem
    reverse_candidates = ([source.with_name(stem[:-4] + '.npz')] if stem.endswith('_rev') else
                          [source.with_name(stem.replace('_rev_', '_', 1) + '.npz')] if '_rev_' in stem else
                          [source.with_name(stem + '_rev.npz'),
                           source.with_name(stem.replace('_s', '_rev_s', 1) + '.npz')])
    reverse_candidates = sorted(set(x for x in reverse_candidates if x != source and x.exists()))
    if len(reverse_candidates) != 1:
        raise ValueError('Need one unambiguous reverse track for CarPreview orientation recovery')
    reverse_source = reverse_candidates[0]
    course = dest / 'course.npz'
    shutil.copyfile(source, course)
    shutil.copyfile(reverse_source, dest / 'course_rev.npz')
    map_dir = dest / 'map'
    map_dir.mkdir()
    with np.load(track, allow_pickle=False) as z:
        grid = z['occupancy'].astype(np.uint8)
        origin, resolution = np.asarray(z['origin'], float), float(z['resolution'])
    with np.load(reverse_source, allow_pickle=False) as z:
        same_grid = (np.array_equal(grid, z['occupancy']) and np.array_equal(origin, z['origin'])
                     and resolution == float(z['resolution']))
    reverse_mode = 'existing counterpart on identical grid'
    if not same_grid:
        # Generated forward/reverse tracks may have independently rasterized
        # grids. Orient on exactly this map, using this same feasible line in
        # reverse, rather than silently mixing two metric grids.
        with np.load(track, allow_pickle=False) as z:
            payload = {key: z[key] for key in z.files}
        payload['raceline'] = payload['raceline'][::-1].copy()
        np.savez_compressed(dest / 'course_rev.npz', **payload)
        reverse_mode = 'input raceline reversed on identical input grid'
    h, w = grid.shape
    (map_dir / 'map.pgm').write_bytes(('P5\n%d %d\n255\n' % (w, h)).encode() + grid[::-1].tobytes())
    (map_dir / 'map.yaml').write_text('image: map.pgm\nresolution: %r\norigin: [%r, %r, 0.0]\n' %
                                     (resolution, float(origin[0]), float(origin[1])))
    pts, res, org = read_map(str(map_dir))
    iy, ix = np.nonzero(grid < 50)
    expected = np.column_stack([origin[0] + (ix + .5) * resolution, origin[1] + (iy + .5) * resolution])
    assert np.array_equal(pts, expected) and res == resolution and np.array_equal(org, origin[:2])
    env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode='eval', tracks=[track])
    env.fixed_speed_cap = cli.cap
    env.start_mode = 'center'
    rows, transitions = [], []
    digest = hashlib.sha256()
    rng = np.random.default_rng(cli.seed + 991)
    try:
        obs, info = env.reset(options={'track': track})
        car = env._env.unwrapped.sim.agents[0]
        angles = np.asarray(car.scan_angles, float)
        laser = float(car.lidar_pos_offset)
        pv = CarPreview(str(course), str(map_dir), pcfg['preview'], laser_x=laser,
                        lidar_pos_offset=laser, log=lambda *_: None)
        if cli.recovery == 'temporal':
            from temporal_recovery_localizer_1126 import TemporalRecoveryLocalizer
            pv.loc = TemporalRecoveryLocalizer(str(map_dir), laser_x=laser, search_delay=cli.search_delay)
        assert len(pv.tps) == 2
        state = env.vehicle_state()
        odom = np.asarray([state[0], state[1], state[4]], float)
        pv.loc.candidates = course_candidates(track)
        pv.loc.reset(odom.copy(), odom.copy())
        pv.started = True
        pv.orient(pv.ref_pose(odom))
        previous_yaw, previous_t = float(state[4]), 0.
        lost_since, injected, previous_fb, valid_since = None, False, False, None
        while True:
            now = float(env.elapsed)
            state = env.vehicle_state()
            dt = now - previous_t
            if dt > 0:
                yr = float(wrap(state[4] - previous_yaw)) / dt
                odom[2] = wrap(odom[2] + (.85 * yr + .02 + rng.normal(0, .02)) * dt)
                odom[:2] += 1.03 * float(state[3]) * dt * np.array([np.cos(odom[2]), np.sin(odom[2])])
            previous_t, previous_yaw = now, float(state[4])
            scan = np.asarray(info['obs_dict']['scans'][0], float)
            if cli.condition == 'wrong_pose_large' and now >= 3. and not injected:
                pv.loc.pose += np.array([6., 2., 2.])
                pv.loc.pose[2] = wrap(pv.loc.pose[2])
                injected = True
            if cli.condition == 'scan_loss4' and 3. <= now < 7.:
                scan = np.full_like(scan, np.inf)
                injected = True
            if cli.recovery == 'temporal':
                pv.loc.set_time(now)
            preview, pose, valid, _ = pv.features(scan, angles, odom, cli.cap)
            if valid:
                lost_since = None
            elif lost_since is None:
                lost_since = now
            lost_for = 0. if lost_since is None else now - lost_since
            fb = cli.fallback == 'on' and lost_for > .3
            if cli.fallback == 'on' and previous_fb:
                if not valid:
                    valid_since = None
                    fb = True
                else:
                    if valid_since is None:
                        valid_since = now
                    fb = now - valid_since < cli.return_confirm - 1e-9
            else:
                valid_since = None
            primary_obs = obs.copy()
            primary_obs[-18:] = preview
            if fb:
                policy_obs = primary_obs[:fbargs['obs_dim']].copy()
                # Both policies remember the same executed servo command in their
                # own action coordinates, matching agent_node.pushHistories.
                start = pcfg['lidar']['num_beams'] + int(pcfg['speed']['enabled'])
                for i in range(start, start + 2 * pcfg['history']['length'], 2):
                    policy_obs[i] = np.clip(convert_steering(policy_obs[i], primary_scales, fallback_scales), -1, 1)
                policy_obs[-1] = min(cli.cap, 4.)
                chosen, normalizer = fbactor, fbrms
            else:
                policy_obs, chosen, normalizer = primary_obs, actor, rms
            with torch.no_grad():
                action = chosen(torch.as_tensor(normalizer.normalize(policy_obs), dtype=torch.float32)).loc.numpy()
            applied = action.copy()
            if fb:
                applied[0] = convert_steering(action[0], fallback_scales, primary_scales)
                v = min(float(unnormalize_speed(action[1], args.min_speed, args.max_speed)), cli.cap, 4.)
                applied[1] = normalize_speed(v, args.min_speed, args.max_speed)
            applied = np.clip(applied, -1, 1)
            if fb != previous_fb:
                transitions.append(dict(time=now, fallback=bool(fb), localizer_valid=bool(valid)))
            previous_fb = fb
            error = float(np.linalg.norm(np.asarray(pose[:2]) - state[:2]))
            yaw_error = float(abs(wrap(pose[2] - state[4])))
            rows.append([now, error, yaw_error, bool(valid), fb, applied[0], applied[1], state[0], state[1], state[4]])
            digest.update(policy_obs.tobytes())
            digest.update(applied.tobytes())
            digest.update(state.tobytes())
            obs, _, term, trunc, info = env.step(applied)
            if term or trunc:
                break
        data = np.asarray(rows, float)
        good = (data[:, 3] > .5) & (data[:, 1] <= .2) & (data[:, 2] <= np.deg2rad(5))
        false_valid = (data[:, 3] > .5) & ((data[:, 1] > .5) | (data[:, 2] > np.deg2rad(15)))
        after = 7. if cli.condition == 'scan_loss4' else 3.
        np.savez_compressed(dest / 'trajectory.npz', values=data,
            columns=np.asarray(['time', 'position_error', 'yaw_error', 'valid', 'fallback',
                                'steer', 'speed', 'true_x', 'true_y', 'true_yaw']))
        report = dict(status='complete', condition=cli.condition, fallback=cli.fallback,
            recovery_method=cli.recovery,
            return_confirm_seconds=cli.return_confirm,
            seed=cli.seed, cap=cli.cap, success=bool(env.success), collision=bool(env.collision),
            elapsed=float(env.elapsed), fault_triggered=injected, transitions=transitions,
            fallback_steps=int(data[:, 4].sum()), false_valid_scans=int(false_valid.sum()),
            correct_valid_fraction=float(good.mean()),
            stable_correct_recovery_time=first_stable_time(data[:, 0], good, after),
            checkpoint_sha256=sha(ckpath), fallback_actor_sha256=sha(fbroot / 'sim2real/actor.pt'),
            primary_normalizer_sha256=sha('results/imitation/1607/sim2real/agent_obs_scale.pkl'),
            fallback_normalizer_sha256=sha(fbroot / 'sim2real/agent_obs_scale.pkl'),
            config_sha256=sha(config), track_sha256=sha(track),
            reverse_track_sha256=sha(dest / 'course_rev.npz'), reverse_track_mode=reverse_mode,
            reverse_track_source=str(reverse_source) if same_grid else track,
            direction_variants_loaded=len(pv.tps),
            dynamics_sha256=sha('configs/task/dynamic.yaml'),
            localizer_sha256=sha('utils/localizer.py'), code_sha256=sha(TOOL_PATH),
            fallback_config_sha256=sha(fbroot / 'backup/f1tenth.yaml'),
            observation_action_state_sha256=digest.hexdigest(),
            steering_scales=dict(primary=primary_scales, fallback=fallback_scales),
            odometry_assumptions=dict(speed_scale=1.03, yaw_rate_scale=.85, yaw_rate_bias=.02, yaw_rate_noise_std=.02),
            qualification=('Closed-loop simulation, actual CarPreview/ScanLocalizer and frozen fallback12190. '
                'True initial CG pose and exact map; no startup test. scan_loss4 affects estimator input only, '
                'not policy LiDAR. Off means primary-only, not production emergency-stop mode. '
                'Simulator command conversion approximates car switch/history semantics; not ROS or actuator qualification. '
                'Original simulator observation augmentation is retained; no extra raycasts/noise draws for policy sensors.'))
        if cli.recovery == 'temporal':
            report['temporal_recovery'] = dict(search_delay_seconds=cli.search_delay,
                margin=pv.loc.margin, confirmation_seconds=pv.loc.confirmation,
                events=pv.loc.events,
                tracking_runtime_p99_ms=(float(np.quantile(pv.loc.tracking_runtime_ms,.99))
                                         if pv.loc.tracking_runtime_ms else None),
                implementation_sha256=sha(ROOT/'tools/temporal_recovery_localizer_1126.py'),
                qualification='Experimental simulation-only replacement after loss. Search latency is fixed and explicitly modeled; measured compute overruns are logged. Capture-to-result odometry compensation, then current-scan temporal confirmation; no past-scan replay. Four retained modes do not prove global uniqueness. During recovery raw odometry is used without the production skid extrapolator. Not ROS, real-time or vehicle qualification.')
        (dest / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
