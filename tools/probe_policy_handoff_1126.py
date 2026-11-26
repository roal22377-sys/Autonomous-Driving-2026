#!/usr/bin/env python3
"""Bounded simulation-only whole-policy handoffs after an exactly replayed prefix.

The selected validation case is diagnostic, never a source of training
observations. No live learner, cache, vehicle, or production fallback is changed.
"""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', default='results/analysis1126/traces/ppo_s2r_29_440000_Shanghai_cap8_center_seed31_preview_audit_physics_audit_observations_audit')
    parser.add_argument('--track', default='installation/maps_gen/Shanghai.npz')
    parser.add_argument('--switch_times', type=float, nargs='+', default=[156.,157.])
    parser.add_argument('--teachers', choices=['teacher2130','source25_u70','expert'], nargs='+',
                        default=['teacher2130','source25_u70'])
    parser.add_argument('--output', default='results/analysis1126/ppo29_u10_shanghai_handoff')
    cli = parser.parse_args()
    os.chdir(SRC)
    import numpy as np
    import torch
    from ruamel.yaml import YAML
    import train_il_s2r as IL
    from utils.env import F1Wrapper, normalize_speed
    from utils.rule_based import RacelinePlanner
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    output = Path(cli.output)
    if output.exists():
        raise FileExistsError('Preserve previous diagnostic')
    stem = Path(cli.trace)
    meta = json.loads(stem.with_suffix('.json').read_text())
    assert 1 <= len(cli.switch_times) <= 3 and len(set(cli.switch_times)) == len(cli.switch_times)
    assert len(set(cli.teachers)) == len(cli.teachers)
    assert all(0 < t < meta['elapsed'] for t in cli.switch_times)
    track = str(Path(cli.track).resolve())
    assert Path(track).name == meta['track']
    archive = Path(meta['observation_audit']['file'])
    assert sha(archive) == meta['observation_audit']['sha256']
    with np.load(archive, allow_pickle=False) as z:
        recorded_obs, recorded_actions, times = z['O'], z['policy_action'], z['command_time']
    with stem.with_suffix('.csv').open() as f:
        recorded = list(csv.DictReader(f))
    assert len(recorded) == len(recorded_obs) == len(recorded_actions) == meta['observation_audit']['rows']
    config = Path('results')/meta['model']/'backup/f1tenth.yaml'
    args = IL.get_args(['--task_config', str(config)])
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load('results/imitation/1607/sim2real', required=True)
    models = dict(teacher2130=Path('results/imitation/14/checkpoint/model_init.pt'),
                  source25_u70=Path('results/ppo_s2r/25/checkpoint/model_3080000.pt'))
    models = {k:v for k,v in models.items() if k in cli.teachers}
    policies = {}
    for label, path in models.items():
        ck = torch.load(path, map_location='cpu', weights_only=False)
        actor = IL.actor_from_state(ck['actor']); actor.eval()
        mean, var = (ck['obs_mean'], ck['obs_var']) if 'obs_mean' in ck else (rms.mean, rms.var)
        policies[label] = (actor, mean, np.sqrt(var+1e-8))
    output.mkdir()
    report = dict(status='running', baseline_observations_sha256=sha(archive),
        baseline_csv_sha256=sha(stem.with_suffix('.csv')), config_sha256=sha(config),
        baseline_model=meta['model'], baseline_tag=meta['tag'], track=track, track_sha256=sha(track),
        seed=meta['seed'], cap=meta['cap'], start=meta['start'],
        normalizer_sha256=sha('results/imitation/1607/sim2real/agent_obs_scale.pkl'),
        dynamics_sha256=sha('configs/task/dynamic.yaml'),
        teacher_sha256={k: sha(p) for k,p in models.items()},
        privileged_expert=('expert' in cli.teachers),
        expert_configuration_sha256=sha('configs/task/f1tenth_preview_1126.yaml'),
        planner_code_sha256=sha('utils/rule_based.py'),
        cases=[], qualification=('Selected deterministic simulation only. Each handoff preserves '
            'all pending actuator commands and vehicle state by replaying the exact candidate '
            'prefix from reset. Both steering and speed are handed off together. A successful '
            'handoff is not a deployable switching rule, a globally superior policy, or a reason '
            'to use these validation observations as gradients. The optional rule-based expert '
            'uses privileged simulator pose and pending actuator commands; it is not a deployable '
            'learned policy or evidence of observability from actor inputs.'))
    destination = output/'report.json'
    def save():
        tmp = output/'report.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        tmp.replace(destination)
    save()
    cases = [(None,None)] + [(teacher,t) for teacher in cli.teachers for t in cli.switch_times]
    for teacher, requested in cases:
        args = IL.get_args(['--task_config', str(config)]); args.seed = meta['seed']
        np.random.seed(meta['seed']); random.seed(meta['seed'])
        env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode='eval', tracks=[track])
        env.expert_cfg = dict(YAML().load(Path('configs/task/f1tenth_preview_1126.yaml'))['expert_plan'])
        env.fixed_speed_cap = meta['cap']; env.start_mode = meta['start']
        name = 'baseline' if teacher is None else '%s_after%g' % (teacher, requested)
        prefix = 0; actual_switch = None; after = []
        try:
            obs, _ = env.reset(options={'track':track})
            planner = None
            if teacher == 'expert':
                planner = RacelinePlanner(env._env.unwrapped.track, args.length_f+args.length_r,
                    v_min=env.min_speed, v_max=env.speed_cap, max_steer=env.max_steer,
                    max_steer_left=env.steer_left_max, max_steer_right=env.steer_right_max,
                    **IL.expert_kwargs(env))
            step = 0
            while True:
                command_time = float(env.elapsed)
                if planner is not None:
                    steer_queue, speed_queue = env.pending_commands()
                    expert_speed, expert_wheel = planner.plan(env.vehicle_state(), steer_queue, speed_queue, env.timestep)
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
                    if teacher == 'expert':
                        action = np.asarray([env.steer_command(expert_wheel),
                            normalize_speed(expert_speed,env.min_speed,env.max_speed)], dtype=np.float32)
                    else:
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
