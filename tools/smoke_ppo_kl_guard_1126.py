#!/usr/bin/env python3
"""Exercise staged KL-guard training/save/resume with tiny real rollouts.

All evaluations are synthetic and must never be used as driving evidence.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

__file__ = str(Path(__file__).resolve())
ROOT = Path(__file__).parents[1]
SRC = ROOT/'src/f1tenth'
STAGE = SRC/'results/preflight_1126/kl_guard_staged'
os.environ['PPO_KL_SMOKE_STAGE'] = str(STAGE)
sys.path[:0] = [str(STAGE), str(SRC), str(SRC/'installation')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-id', type=int, required=True)
    cli = parser.parse_args()
    os.chdir(SRC)
    import torch
    import train_ppo_s2r as PPO
    from ruamel.yaml import YAML
    torch.set_num_threads(1)
    assert Path(PPO.__file__).resolve() == STAGE/'train_ppo_s2r.py'
    run = Path('results/code_smoke_kl_guard1126')/str(cli.trial_id)
    if run.exists():
        raise FileExistsError('Preserve previous smoke runs')
    yaml = YAML()
    override = STAGE/('smoke_override_%s.yaml' % cli.trial_id)
    cfg = yaml.load(Path('configs/task/sim_override_1126_ppo.yaml'))
    cfg['training_mix'] = {k: .2 for k in ('hall', 'short', 'regular', 'open', 'real')}
    yaml.dump(cfg, override)

    def synthetic_evaluate(args, policy, tracks, mode='eval', settings=PPO.IL.EVAL_SETTINGS):
        rows = []
        for caps, start in settings:
            for i, path in enumerate(sorted(tracks)):
                cap = caps[i % len(caps)] if isinstance(caps, (tuple,list)) else caps
                rows.append((f'{Path(path).stem}@{cap:g}/{start}', True, 100., .01, {}))
        return 1., 100., rows

    PPO.IL.evaluate = synthetic_evaluate
    argv = [__file__, '--il', 'imitation/1607', '--name', 'code_smoke_kl_guard1126',
        '--algo_idx', str(cli.trial_id), '--init', 'preflight_1126/ppo25_u70_actor_only.pt',
        '--sim_override', str(override), '--workers', '5', '--eval_workers', '1',
        '--n_steps', '500', '--total_steps', '1500', '--epochs', '1', '--minibatch', '250',
        '--eval_every', '1', '--ckpt_every', '1', '--short_eval', '1', '--worst_eval', '1',
        '--lost_eval', '1', '--family_select', '1', '--critic_warmup', '1', '--max_warmup', '1',
        '--warmup_ev=-1000000000', '--cost_critic', '1', '--crash_adapt', '0',
        '--collision_cost', '5000', '--crash_start', '5000', '--ret_norm', '1',
        '--log_std_steer', '-3.4', '--log_std_speed', '-2.6', '--bc_lost_off', '1',
        '--bc_steer_weight', '15', '--line_cost', '10', '--lat_acc_mode', 'path_normal', '--lr', '2e-5',
        '--retention_file', 'results/ppo_retention_candidate_1126/replay_cache.npz',
        '--retention_manifest', 'results/ppo_retention_candidate_1126/preparation.json',
        '--retention_weight', '5', '--retention_share', '.125', '--kl_guard', 'minibatch_rollback']
    sys.argv = argv.copy()
    PPO.main()
    resume = run/'checkpoint/resume.pt'
    state = torch.load(resume, map_location='cpu', weights_only=False)
    assert state['update']==3 and state['actor_start']==2
    assert state['args']['kl_guard']=='minibatch_rollback'
    assert state['log'][1]['actor_accepted_minibatches']==0
    metadata = state['retention']['metadata']
    sys.argv = argv.copy()+['--resume']
    sys.argv[sys.argv.index('--total_steps')+1] = '2000'
    PPO.main()
    state = torch.load(resume, map_location='cpu', weights_only=False)
    assert state['update']==4 and state['retention']['metadata']==metadata
    for row in state['log']:
        if row.get('update',0)>0:
            assert row['kl_guard']=='minibatch_rollback' and row['kl']<=.015+1e-9
            assert row['retention_minibatches']==row['actor_accepted_minibatches']
    for key in ('actor','critic','ccritic','teacher'):
        assert all(torch.isfinite(v).all().item() for v in state[key].values())
    teacher = torch.load('results/preflight_1126/ppo25_u70_actor_only.pt', map_location='cpu', weights_only=False)['actor']
    assert all(torch.equal(v,state['teacher'][k]) for k,v in teacher.items())
    # Attempt a forbidden mode change in a separate copy; preserve the original
    # smoke run's backup arguments as well as its complete resume.
    rejected = run.parent/str(cli.trial_id+1000)
    assert not rejected.exists()
    (rejected/'checkpoint').mkdir(parents=True)
    (rejected/'checkpoint/resume.pt').write_bytes(resume.read_bytes())
    sys.argv = argv.copy()+['--resume']
    sys.argv[sys.argv.index('--algo_idx')+1] = str(cli.trial_id+1000)
    sys.argv[sys.argv.index('--kl_guard')+1] = 'epoch'
    try:
        PPO.main()
    except ValueError as exc:
        assert 'KL guard mode changed during resume' in str(exc)
    else:
        raise AssertionError('A guard-mode change during resume was accepted')
    report = dict(status='passed', trial_id=cli.trial_id, update=4,
        staged_sha256=hashlib.sha256((STAGE/'train_ppo_s2r.py').read_bytes()).hexdigest(),
        resume=str(resume), teacher_preserved=True, mode_change_rejected=True,
        finite_weights=True, accepted_retention_counters_consistent=True,
        qualification='Tiny real simulator updates plus synthetic evaluation scores. Exercises lifecycle only; neither completion nor lap-time evidence. Production learner remains unchanged.')
    (STAGE/('smoke_%s.json'%cli.trial_id)).write_text(json.dumps(report,indent=2)+'\n')
    print('KL GUARD LIFECYCLE PASSED',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
