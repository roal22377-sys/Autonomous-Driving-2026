#!/usr/bin/env python3
"""Real PPO warm-up/update/resume smoke with synthetic evaluation scores.

Verifies actor-only restart, fresh critic statistics, frozen starting-policy BC
teacher, and teacher preservation on resume. Never a performance evaluation.
"""
import hashlib
import argparse
import json
import os
import sys
from pathlib import Path

__file__ = str(Path(__file__).resolve())  # spawn must still find this script after chdir
ROOT = Path(__file__).parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-id', type=int, default=1)
    cli = parser.parse_args()
    os.chdir(SRC)
    import torch
    import numpy as np
    from ruamel.yaml import YAML
    import train_ppo_s2r as PPO

    run = Path('results/code_smoke_actor_restart_1126') / str(cli.trial_id)
    if run.exists():
        raise FileExistsError('Preserve existing smoke evidence: ' + str(run))
    init = Path('results/preflight_1126/ppo25_u70_actor_only.pt')
    source = torch.load(init, map_location='cpu', weights_only=False)
    assert set(source) == {'actor'}
    yaml = YAML()
    cfg = yaml.load(Path('configs/task/sim_override_1126_ppo.yaml'))
    cfg['training_mix'] = {k: .2 for k in ('hall', 'short', 'regular', 'open', 'real')}
    override = Path('results/preflight_1126/actor_restart_smoke_override.yaml')
    yaml.dump(cfg, override)
    evaluations = []

    def synthetic_evaluate(args, policy, tracks, mode='eval', settings=PPO.IL.EVAL_SETTINGS):
        evaluations.append(len(tracks))
        rows = []
        for caps, start in settings:
            for i, path in enumerate(sorted(tracks)):
                cap = caps[i % len(caps)] if isinstance(caps, (list, tuple)) else caps
                rows.append((f'{Path(path).stem}@{cap:g}/{start}', True, 100., .01, {}))
        return 1., 100., rows

    PPO.IL.evaluate = synthetic_evaluate
    argv = [str(__file__), '--il', 'imitation/1607', '--name', 'code_smoke_actor_restart_1126',
        '--algo_idx', str(cli.trial_id), '--init', str(init.relative_to('results')),
        '--sim_override', str(override), '--workers', '5', '--eval_workers', '1',
        '--n_steps', '500', '--total_steps', '1000', '--epochs', '1', '--minibatch', '250',
        '--eval_every', '1', '--ckpt_every', '1', '--short_eval', '1', '--worst_eval', '1',
        '--lost_eval', '1', '--family_select', '1', '--critic_warmup', '2', '--max_warmup', '2',
        '--warmup_ev=-1000000000', '--cost_critic', '1', '--crash_adapt', '0',
        '--collision_cost', '5000', '--crash_start', '5000', '--ret_norm', '1',
        '--log_std_steer', '-3.4', '--log_std_speed', '-2.6', '--bc_lost_off', '1',
        '--bc_steer_weight', '15', '--line_cost', '10', '--lat_acc_mode', 'path_normal']
    print('CODE SMOKE ONLY: synthetic evaluation; outputs cannot qualify a driving policy', flush=True)
    sys.argv = argv
    PPO.main()
    checkpoint = run / 'checkpoint/resume.pt'
    warm = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert warm['update'] == 2 and warm['actor_start'] is None
    assert all(torch.equal(warm['actor'][k], v) for k, v in source['actor'].items() if 'log_std' not in k)
    assert all(torch.equal(warm['teacher'][k], v) for k, v in source['actor'].items())
    assert not warm['opt_actor']['state']
    assert warm['opt_critic']['state'] and warm['opt_ccritic']['state']
    assert warm['priv_rms'][2] > 1 and warm['ret_rms'][2] > 1
    # A real resumed actor update must not silently replace the PPO teacher with IL.
    sys.argv = argv.copy()
    sys.argv[sys.argv.index('--total_steps') + 1] = '1500'
    sys.argv.append('--resume')
    PPO.main()
    resumed = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert resumed['update'] == 3 and resumed['actor_start'] == 3
    assert all(torch.equal(resumed['teacher'][k], v) for k, v in source['actor'].items())
    assert any(not torch.equal(resumed['actor'][k], v) for k, v in source['actor'].items() if 'log_std' not in k)
    assert resumed['opt_actor']['state']
    assert resumed['il']['res'] == warm['il']['res']
    for key in ('actor', 'critic', 'ccritic'):
        assert all(torch.isfinite(v).all() for v in resumed[key].values())
    assert all(np.isfinite(v).all() for key in ('priv_rms', 'ret_rms') for v in resumed[key])
    result = dict(status='passed', trial_id=cli.trial_id, source_actor_sha256=hashlib.sha256(init.read_bytes()).hexdigest(),
        real_rollout_steps=1500, warmup_updates=2, resumed_actor_updates=1,
        policy_mean_weights_frozen_during_warmup=True, starting_ppo_teacher_preserved_across_resume=True,
        critic_normalizers_new_and_updated=True, original_il_baseline_preserved=True,
        qualification='Synthetic evaluation scores, five equal-share workers, tiny code smoke only. Not policy performance, not production warmup sufficiency or source qualification.')
    Path('results/preflight_1126/actor_restart_smoke_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print('SMOKE PASSED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
