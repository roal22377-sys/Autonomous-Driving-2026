#!/usr/bin/env python3
"""Bounded simulation-only whole-policy handoffs after an exactly replayed prefix.

The selected Shanghai validation case is diagnostic, never a source of training
observations. No live learner, cache, vehicle, or production fallback is changed.
"""
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.chdir(SRC)
    import numpy as np
    import torch
    from ruamel.yaml import YAML
    import train_il_s2r as IL
    from utils.env import F1Wrapper
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    output = Path('results/analysis1126/ppo29_u10_shanghai_handoff')
    if output.exists():
        raise FileExistsError('Preserve previous diagnostic')
    stem = Path('results/analysis1126/traces/ppo_s2r_29_440000_Shanghai_cap8_center_seed31_preview_audit_physics_audit_observations_audit')
    meta = json.loads(stem.with_suffix('.json').read_text())
    archive = Path(meta['observation_audit']['file'])
    assert sha(archive) == meta['observation_audit']['sha256']
    with np.load(archive, allow_pickle=False) as z:
        recorded_obs, recorded_actions, times = z['O'], z['policy_action'], z['command_time']
    with stem.with_suffix('.csv').open() as f:
        recorded = list(csv.DictReader(f))
    assert len(recorded) == len(recorded_obs) == len(recorded_actions) == 6346
    config = Path('results/ppo_s2r/29/backup/f1tenth.yaml')
    args = IL.get_args(['--task_config', str(config)])
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load('results/imitation/1607/sim2real', required=True)
    models = dict(teacher2130=Path('results/imitation/14/checkpoint/model_init.pt'),
                  source25_u70=Path('results/ppo_s2r/25/checkpoint/model_3080000.pt'))
    policies = {}
    for label, path in models.items():
        ck = torch.load(path, map_location='cpu', weights_only=False)
        actor = IL.actor_from_state(ck['actor']); actor.eval()
        mean, var = (ck['obs_mean'], ck['obs_var']) if 'obs_mean' in ck else (rms.mean, rms.var)
        policies[label] = (actor, mean, np.sqrt(var+1e-8))
    output.mkdir()
    report = dict(status='running', baseline_observations_sha256=sha(archive),
        baseline_csv_sha256=sha(stem.with_suffix('.csv')), config_sha256=sha(config),
        dynamics_sha256=sha('configs/task/dynamic.yaml'),
        teacher_sha256={k: sha(p) for k,p in models.items()},
        cases=[], qualification=('Selected deterministic simulation only. Each handoff preserves '
            'all pending actuator commands and vehicle state by replaying the exact candidate '
            'prefix from reset. Both steering and speed are handed off together. A successful '
            'handoff is not a deployable switching rule, a globally superior policy, or a reason '
            'to use these validation observations as gradients.'))
    destination = output/'report.json'
    def save():
        tmp = output/'report.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        tmp.replace(destination)
    save()
    for teacher, requested in [(None, None), ('teacher2130',156.), ('teacher2130',157.),
                               ('source25_u70',156.), ('source25_u70',157.)]:
        args = IL.get_args(['--task_config', str(config)]); args.seed = 31
        np.random.seed(31); random.seed(31)
        track = str(Path('installation/maps_gen/Shanghai.npz').resolve())
        env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode='eval', tracks=[track])
        env.expert_cfg = dict(YAML().load(Path('configs/task/f1tenth_preview_1126.yaml'))['expert_plan'])
        env.fixed_speed_cap = 8.; env.start_mode = 'center'
        name = 'baseline' if teacher is None else '%s_after%g' % (teacher, requested)
        prefix = 0; actual_switch = None; after = []
        try:
            obs, _ = env.reset(options={'track':track})
            step = 0
            while True:
                command_time = float(env.elapsed)
                switched = requested is not None and command_time >= requested
                if not switched or actual_switch is None:
                    assert step < len(recorded_obs)
                    np.testing.assert_array_equal(obs, recorded_obs[step])
                    assert command_time == times[step]
                    if step:
                        state = env.vehicle_state()
                        for j, field in ((0,'x'), (1,'y'), (2,'wheel_actual'), (3,'speed')):
                            assert float(state[j]) == float(recorded[step-1][field])
                    prefix += 1
                if switched:
                    if actual_switch is None:
                        actual_switch = command_time
                    actor, mean, std = policies[teacher]
                    with torch.no_grad():
                        action = actor(torch.as_tensor((obs-mean)/std, dtype=torch.float32)).loc.numpy()
                    action = np.clip(action, -1, 1)
                else:
                    action = recorded_actions[step]
                obs, _, term, trunc, _ = env.step(action)
                if switched:
                    state = env.vehicle_state(); geometry = env.driving_geometry()
                    after.append(dict(command_time=command_time, state_time=float(env.elapsed),
                        x=float(state[0]), y=float(state[1]), wheel=float(state[2]), speed=float(state[3]),
                        steer_command=float(action[0]), normalized_speed_command=float(action[1]),
                        lateral=float(geometry['lateral']), reference_heading=float(geometry['heading'])))
                step += 1
                if term or trunc:
                    break
            row = dict(name=name, teacher=teacher, requested_switch_time=requested,
                actual_switch_time=actual_switch, exact_prefix_observations=prefix,
                success=bool(env.success), collision=bool(env.collision),
                elapsed=float(env.elapsed), steps=step, lap_times=[float(t) for t in env.lap_times])
            if teacher is None:
                assert row['success'] == meta['success'] and row['collision'] == meta['collision']
                assert row['elapsed'] == meta['elapsed'] and row['lap_times'] == meta['lap_times']
                assert step == len(recorded_obs)
                row['original_failure_exactly_reproduced'] = True
            else:
                assert actual_switch is not None and after
                path = output/(name+'.csv')
                with path.open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=list(after[0])); writer.writeheader(); writer.writerows(after)
                row.update(after_handoff_csv=str(path), after_handoff_sha256=sha(path))
            report['cases'].append(row); save(); print(json.dumps(row), flush=True)
        finally:
            env.close()
    report['status'] = 'complete_read_only'; save()


if __name__ == '__main__':
    main()
