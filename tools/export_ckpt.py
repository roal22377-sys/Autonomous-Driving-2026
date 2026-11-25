#!/usr/bin/env python3
'''
Export a PPO checkpoint (results/<run>/checkpoint/model_<steps>.pt) as a deployable model directory
results/<dest>/{sim2real/actor.pt + agent_obs_scale.pkl + agent_reward_scale.pkl, backup/, README.md}, the layout
agent_node.py, compare_models.py and check_track.py load (like results/ppo_s2r/12190).

    python3 tools/export_ckpt.py ppo_s2r/16 model_1760000.pt ppo_s2r/1640 [--note "..."]

The observation normalizer is the IL model's (PPO keeps it frozen: train_ppo_s2r.py), read from the run's
backup/ppo_args.yaml `il`. The update number is steps / n_steps.
'''
import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'src/f1tenth')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='e.g. ppo_s2r/16')
    ap.add_argument('ckpt', help='e.g. model_1760000.pt')
    ap.add_argument('dest', help='e.g. ppo_s2r/1640')
    ap.add_argument('--note', default='')
    a = ap.parse_args()
    os.chdir(SRC)
    run, dest = f'results/{a.run}', f'results/{a.dest}'
    if os.path.exists(dest):
        raise SystemExit(f'{dest} exists')
    pa = dict(line.split(': ', 1) for line in open(f'{run}/backup/ppo_args.yaml').read().splitlines() if ': ' in line)
    il, n_steps = pa['il'].strip(), int(float(pa['n_steps']))
    os.environ['F1_TASK_CONFIG'] = f'{run}/backup/f1tenth.yaml'      # the actor's input size comes from it
    sys.path.insert(0, SRC)
    import torch
    import train_il_s2r as IL
    sd = torch.load(f'{run}/checkpoint/{a.ckpt}', map_location='cpu')['actor']
    actor = IL.actor_from_state(sd)
    os.makedirs(f'{dest}/sim2real'), os.makedirs(f'{dest}/backup')
    torch.save(actor, f'{dest}/sim2real/actor.pt')
    for f in ('agent_obs_scale.pkl', 'agent_reward_scale.pkl'):
        shutil.copy(f'results/{il}/sim2real/{f}', f'{dest}/sim2real/{f}')
    for f in ('f1tenth.yaml', 'dynamic.yaml', 'ppo_args.yaml'):
        shutil.copy(f'{run}/backup/{f}', f'{dest}/backup/{f}')
    steps = int(a.ckpt.split('_')[-1].split('.')[0])
    with open(f'{dest}/README.md', 'w') as f:
        f.write(f'{a.run} checkpoint at update {steps // n_steps} ({a.ckpt}), IL {il} normalizer. {a.note}\n')
    print(f'[export] {run}/checkpoint/{a.ckpt} (update {steps // n_steps}) -> {dest}')


if __name__ == '__main__':
    main()
