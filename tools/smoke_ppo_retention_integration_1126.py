#!/usr/bin/env python3
"""Real tiny updates for an isolated PPO copy; evaluations are synthetic."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

__file__ = str(Path(__file__).resolve())
ROOT = Path(__file__).parents[1]
SRC = ROOT / 'src/f1tenth'
STAGE = SRC / 'results/preflight_1126/retention_staged'
extra = [os.environ['PPO_RETENTION_STAGE']] if os.environ.get('PPO_RETENTION_STAGE') else []
sys.path[:0] = extra + [str(SRC), str(SRC / 'installation')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['original', 'disabled', 'active'], required=True)
    parser.add_argument('--trial-id', type=int, required=True)
    cli = parser.parse_args()
    if cli.mode != 'original':
        os.environ['PPO_RETENTION_STAGE'] = str(STAGE)
        sys.path.insert(0, str(STAGE))
    elif os.environ.get('PPO_RETENTION_STAGE'):
        raise ValueError('Original comparison must run without staged module environment')
    os.chdir(SRC)
    import torch
    import train_ppo_s2r as PPO
    from ruamel.yaml import YAML
    expected = (SRC if cli.mode == 'original' else STAGE) / 'train_ppo_s2r.py'
    assert Path(PPO.__file__).resolve() == expected
    run = Path('results/code_smoke_retention_integration1126') / str(cli.trial_id)
    if run.exists():
        raise FileExistsError('Preserve prior smoke runs')
    yaml = YAML()
    override = Path('results/preflight_1126/retention_staged/smoke_override.yaml')
    cfg = yaml.load(Path('configs/task/sim_override_1126_ppo.yaml'))
    cfg['training_mix'] = {k: .2 for k in ('hall', 'short', 'regular', 'open', 'real')}
    yaml.dump(cfg, override)

    def synthetic_evaluate(args, policy, tracks, mode='eval', settings=PPO.IL.EVAL_SETTINGS):
        rows = []
        for caps, start in settings:
            for i, path in enumerate(sorted(tracks)):
                cap = caps[i % len(caps)] if isinstance(caps, (tuple, list)) else caps
                rows.append((f'{Path(path).stem}@{cap:g}/{start}', True, 100., .01, {}))
        return 1., 100., rows

    PPO.IL.evaluate = synthetic_evaluate
    argv = [__file__, '--il', 'imitation/1607', '--name', 'code_smoke_retention_integration1126',
        '--algo_idx', str(cli.trial_id), '--init', 'preflight_1126/ppo25_u70_actor_only.pt',
        '--sim_override', str(override), '--workers', '5', '--eval_workers', '1',
        '--n_steps', '500', '--total_steps', '1000', '--epochs', '1', '--minibatch', '250',
        '--eval_every', '1', '--ckpt_every', '1', '--short_eval', '1', '--worst_eval', '1',
        '--lost_eval', '1', '--family_select', '1', '--critic_warmup', '1', '--max_warmup', '1',
        '--warmup_ev=-1000000000', '--cost_critic', '1', '--crash_adapt', '0',
        '--collision_cost', '5000', '--crash_start', '5000', '--ret_norm', '1',
        '--log_std_steer', '-3.4', '--log_std_speed', '-2.6', '--bc_lost_off', '1',
        '--bc_steer_weight', '15', '--line_cost', '10', '--lat_acc_mode', 'path_normal', '--lr', '2e-5']
    if cli.mode == 'active':
        argv += ['--retention_file', 'results/ppo_retention_candidate_1126/replay_cache.npz',
                 '--retention_manifest', 'results/ppo_retention_candidate_1126/preparation.json',
                 '--retention_weight', '.1', '--retention_share', '.125']
    sys.argv = argv
    PPO.main()
    path = run / 'checkpoint/resume.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['update'] == 2 and state['actor_start'] == 2
    if cli.mode == 'active':
        assert state['retention'] is not None
        assert state['log'][-1]['retention_minibatches'] == 2
        assert state['log'][-1]['retention_loss'] > 0
        assert state['log'][1]['retention_minibatches'] == 0
        initial_cache = state['retention']
        sys.argv = argv.copy()
        sys.argv[sys.argv.index('--total_steps') + 1] = '1500'
        sys.argv.append('--resume')
        PPO.main()
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['update'] == 3 and state['retention']['metadata'] == initial_cache['metadata']
        assert not torch.equal(state['retention']['rng'], initial_cache['rng'])
        assert state['log'][-1]['retention_minibatches'] == 2
        teacher = torch.load('results/preflight_1126/ppo25_u70_actor_only.pt', map_location='cpu', weights_only=False)['actor']
        assert all(torch.equal(v, state['teacher'][k]) for k, v in teacher.items())
    else:
        assert state.get('retention') is None
    for key in ('actor', 'critic', 'ccritic', 'teacher'):
        assert all(torch.isfinite(v).all() for v in state[key].values())
    report = dict(status='passed', mode=cli.mode, trial_id=cli.trial_id, update=state['update'],
        implementation_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),
        snapshot=str(path), retention_minibatches=state['log'][-1].get('retention_minibatches', 0),
        retention_loss=state['log'][-1].get('retention_loss'),
        qualification='Synthetic scoring and tiny real updates only. No deployment/performance result; .1 weight and .125 share are smoke inputs, not adopted hyperparameters.')
    (STAGE / ('smoke_' + cli.mode + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print('INTEGRATION SMOKE PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
