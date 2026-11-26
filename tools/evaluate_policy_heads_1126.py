#!/usr/bin/env python3
"""Matched frozen-policy steering/speed combinations; no privileged controls.

Both actors receive the same observation and actual applied-command history.
This tests compatibility before considering any ensemble teacher or distillation.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]
ACTORS = {}
RMS = None
TAGS = ('440000', '3080000')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize():
    global RMS
    import torch
    import train_il_s2r as IL
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    args = IL.get_args(['--task_config', 'results/ppo_s2r/25/backup/f1tenth.yaml'])
    RMS = RunningMeanStd('agent_obs', args.obs_dim)
    RMS.load('results/imitation/1607/sim2real', required=True)
    for tag in TAGS:
        ck = torch.load('results/ppo_s2r/25/checkpoint/model_' + tag + '.pt',
                        map_location='cpu', weights_only=False)
        ACTORS[tag] = IL.actor_from_state(ck['actor'])
        ACTORS[tag].eval()


def drive(job):
    import numpy as np
    import torch
    import train_il_s2r as IL
    from utils.env import F1Wrapper
    from utils.driving_metrics import DrivingMetrics
    args = IL.get_args([])
    args.seed = job['seed']
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode=job['mode'], tracks=[job['path']])
    env.fixed_speed_cap = job['cap']
    env.start_mode = 'center'
    digest = hashlib.sha256()
    metrics = DrivingMetrics()
    try:
        obs, _ = env.reset(options={'track': job['path']})
        while True:
            normalized = torch.as_tensor(RMS.normalize(obs), dtype=torch.float32)
            with torch.no_grad():
                steer_policy = ACTORS[job['steer_tag']](normalized).loc.numpy()
                speed_policy = (steer_policy if job['speed_tag'] == job['steer_tag'] else
                                ACTORS[job['speed_tag']](normalized).loc.numpy())
            action = np.asarray([steer_policy[0], speed_policy[1]], dtype=np.float32)
            action = np.clip(action, -1, 1)
            digest.update(obs.tobytes())
            digest.update(env.vehicle_state().tobytes())
            digest.update(action.tobytes())
            obs, _, term, trunc, _ = env.step(action)
            metrics.update(env, action)
            if term or trunc:
                break
        return dict(job, complete=bool(env.success), collision=bool(env.collision),
            elapsed=float(env.elapsed), lap_times=[float(t) for t in env.lap_times],
            progress_m=float(env.progress), phases=metrics.result(),
            observation_action_state_sha256=digest.hexdigest())
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--mode', choices=['eval', 'worst'], default='eval')
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--cap', type=float, default=8.)
    cli = p.parse_args()
    os.chdir(SRC)
    cases = [
        ('installation/maps_hall/Hall3000_rev_s54_w3000.npz', 7),
        ('results/extended_hall_1126/v1/eval/HallExtended510000_s137_w510000.npz', 19),
        ('installation/maps_real/Practice1126b.npz', 17),
        ('installation/maps_open_test/big/Shanghai_w901.npz', 33),
        ('installation/maps_open_test/big/YasMarina_rev_w902.npz', 44),
        ('installation/maps_gen/Monza.npz', 7),
    ]
    expected = {}
    if cli.mode == 'eval' and cli.seed_offset == 0 and cli.cap == 8.:
        original = json.loads(Path('results/analysis1126/preview_transition_seed7.json').read_text())
        for row in original['runs']:
            if 'Hall3000_rev' in row['path'] and row['condition'] == 'control':
                expected[row['tag']] = row['observation_action_state_sha256']
        assert set(expected) == set(TAGS)
    jobs = []
    for path, seed in cases:
        for steer in TAGS:
            for speed in TAGS:
                jobs.append(dict(path=str(Path(path).resolve()), seed=seed + cli.seed_offset,
                    mode=cli.mode, cap=cli.cap, steer_tag=steer, speed_tag=speed,
                    controller='steer' + steer + '_speed' + speed))
    dest = Path(cli.output)
    if dest.exists():
        raise FileExistsError('Do not overwrite an evaluation')
    report = dict(status='running', runs=[], total=len(jobs),
        config_sha256=sha('results/ppo_s2r/25/backup/f1tenth.yaml'),
        dynamics_sha256=sha('configs/task/dynamic.yaml'),
        normalizer_sha256=sha('results/imitation/1607/sim2real/agent_obs_scale.pkl'),
        checkpoints={tag: sha('results/ppo_s2r/25/checkpoint/model_' + tag + '.pt') for tag in TAGS},
        tracks={str(Path(path).resolve()): sha(path) for path, _ in cases},
        qualification='Selected diagnostic cohort, including known failure cases. Both actors use ordinary identical observations and applied-command history; no expert or true-state intervention. Frozen head combinations may be incompatible. Results do not qualify a deployable ensemble or distilled model.')
    def save():
        tmp = dest.with_suffix('.tmp.json')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(dest)
    save()
    with mp.get_context('spawn').Pool(cli.workers, initializer=initialize) as pool:
        for row in pool.imap_unordered(drive, jobs):
            if expected and 'Hall3000_rev' in row['path'] and row['steer_tag'] == row['speed_tag']:
                assert row['observation_action_state_sha256'] == expected[row['steer_tag']], 'Control instrumentation changed the trajectory'
                row['prior_control_bitwise_match'] = True
            report['runs'].append(row)
            save()
            print(len(report['runs']), '/', len(jobs), Path(row['path']).stem,
                  row['controller'], row['complete'], round(row['elapsed'], 3), flush=True)
    report['status'] = 'complete'
    report['summary'] = {}
    for steer in TAGS:
        for speed in TAGS:
            label = 'steer' + steer + '_speed' + speed
            rows = [r for r in report['runs'] if r['controller'] == label]
            report['summary'][label] = dict(complete=sum(r['complete'] for r in rows), total=len(rows))
    save()
    print('DONE', json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
