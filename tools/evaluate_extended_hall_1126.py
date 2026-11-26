#!/usr/bin/env python3
"""Feasibility and heldout policy screen for isolated longer narrow layouts."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]
POLICIES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(tags, model='ppo_s2r/25', normalizer_dir='results/ppo_s2r/25/sim2real'):
    import torch
    import train_il_s2r as IL
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    args = IL.get_args(['--task_config', 'results/' + model + '/backup/f1tenth.yaml'])
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load(normalizer_dir, required=True)
    for tag in tags:
        if tag == 'expert':
            continue
        ck = torch.load('results/' + model + '/checkpoint/model_' + tag + '.pt',
                        map_location='cpu', weights_only=False)
        POLICIES[tag] = IL.portable_policy((ck['actor'], rms.mean, rms.var))


def worker(job):
    import train_il_s2r as IL
    from evaluate_expert_1126 import drive
    row = dict(job)
    if row['controller'] == 'expert':
        raw = drive(('extended_hall', row['path'], row['seed'], row['cap'], 'center', row['mode']))
        row.update(complete=raw['complete'], elapsed=raw['time'], failure_tail=raw['failure_tail'])
    else:
        raw = IL.eval_worker((row['path'], POLICIES[row['controller']], row['mode'],
                              row['seed'], row['cap'], 'center'))
        row.update(complete=bool(raw[1]), elapsed=float(raw[2]), phases=raw[4])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--mode', choices=['eval', 'worst'], default='eval')
    p.add_argument('--seeds', type=int, nargs='+', default=[7])
    p.add_argument('--caps', type=float, nargs='+', default=[8., 5.])
    p.add_argument('--controllers', nargs='+', default=['expert', '440000', '3080000'])
    p.add_argument('--model', default='ppo_s2r/25', help='PPO run, e.g. ppo_s2r/26; checkpoints stay explicit')
    cli = p.parse_args()
    os.chdir(SRC)
    model_dir = Path('results') / cli.model
    launch_args = json.loads((model_dir / 'backup/ppo_args.json').read_text())
    normalizer_dir = model_dir / 'sim2real'
    if not (normalizer_dir / 'agent_obs_scale.pkl').exists():
        # Rejected early checkpoints have no promoted sim2real export. PPO keeps
        # the source IL normalizer frozen; never use another run's by accident.
        normalizer_dir = Path('results') / launch_args['il'] / 'sim2real'
    normalizer_hash = sha(normalizer_dir / 'agent_obs_scale.pkl')
    manifest = Path(cli.manifest)
    generation = json.loads(manifest.read_text())
    assert generation['status'] == 'generated_static_checks_recorded_not_applied'
    assert generation['base_splits_disjoint']
    train = {r['base_geometry_sha256'] for r in generation['variants'] if r['split'] == 'train'}
    heldout = {r['base_geometry_sha256'] for r in generation['variants'] if r['split'] == 'eval'}
    assert train.isdisjoint(heldout)
    for path, digest in generation['source_sha256'].items():
        assert sha(path) == digest, path
    bases = sorted(b['seed'] for b in generation['bases'])
    jobs = []
    for item in generation['variants']:
        assert sha(item['path']) == item['sha256'], item['path']
        for seed_base in cli.seeds:
            # Fixed base+direction index independent of completion or pool order.
            seed = seed_base + bases.index(item['seed']) * 2 + int(item['reverse'])
            for cap in cli.caps:
                for controller in cli.controllers:
                    if controller != 'expert' and item['split'] != 'eval':
                        continue
                    jobs.append(dict(path=item['path'], split=item['split'],
                        base_geometry_sha256=item['base_geometry_sha256'],
                        base_seed=item['seed'], reverse=item['reverse'], length_m=item['length_m'],
                        seed=seed, seed_base=seed_base, cap=cap, controller=controller, model=cli.model, mode=cli.mode))
    dest = Path(cli.output)
    if dest.exists():
        raise FileExistsError('Do not overwrite a previous evaluation')
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', runs=[], total=len(jobs), mode=cli.mode, seeds=cli.seeds,
        model=cli.model, normalizer_directory=str(normalizer_dir), normalizer_sha256=normalizer_hash,
        manifest_sha256=sha(manifest),
        config_sha256=sha(model_dir / 'backup/f1tenth.yaml'),
        dynamics_sha256=sha('configs/task/dynamic.yaml'),
        checkpoint_sha256={tag: sha(model_dir / 'checkpoint' / ('model_' + tag + '.pt'))
                           for tag in cli.controllers if tag != 'expert'},
        qualification='Untrained synthetic base geometries; single-seed screens are diagnostics, not real-car qualification. Expert uses privileged state. No layouts excluded for poor controller performance.')
    def save():
        temp = dest.with_suffix('.tmp.json')
        temp.write_text(json.dumps(result, indent=2) + '\n')
        os.replace(temp, dest)
    save()
    with mp.get_context('spawn').Pool(cli.workers, initializer=initialize,
                                     initargs=(cli.controllers, cli.model, str(normalizer_dir))) as pool:
        for row in pool.imap_unordered(worker, jobs):
            result['runs'].append(row)
            save()
            if len(result['runs']) % 4 == 0 or not row['complete']:
                print(len(result['runs']), '/', len(jobs), row['controller'],
                      Path(row['path']).name, row['cap'], row['complete'], flush=True)
    result['status'] = 'complete'
    result['summary'] = {}
    for controller in cli.controllers:
        result['summary'][controller] = {}
        for split in ('train', 'eval'):
            rows = [r for r in result['runs'] if r['controller'] == controller and r['split'] == split]
            if rows:
                result['summary'][controller][split] = dict(complete=sum(r['complete'] for r in rows), total=len(rows))
    save()
    print('DONE', json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
