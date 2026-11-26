#!/usr/bin/env python3
"""Measure action drift on frozen successful old-policy trajectories; not lap evaluation."""
import hashlib
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', type=int, nargs='+', default=[10, 20, 30])
    parser.add_argument('--output', default='results/analysis1126/ppo25_frozen_trajectory_retention_audit.json')
    cli = parser.parse_args()
    if any(u <= 0 for u in cli.updates) or len(set(cli.updates)) != len(cli.updates):
        raise ValueError('Checkpoint updates must be distinct positive integers')
    os.chdir(SRC)
    output = Path(cli.output)
    if output.exists():
        raise FileExistsError('Preserve prior evidence; choose a new --output')
    import numpy as np
    import torch
    import train_il_s2r as IL
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    args = IL.get_args(['--task_config', 'results/imitation/1607/backup/f1tenth.yaml'])
    assert (args.obs_dim, args.min_speed, args.max_speed) == (132, .5, 8.)
    rms = RunningMeanStd('agent_obs', 132)
    rms.load('results/imitation/1607/sim2real', required=True)
    models = {'source2130_transferred': Path('results/imitation/14/checkpoint/model_init.pt'),
              'frozenIL1607': Path('results/imitation/16/checkpoint/model_round7.pt')}
    models.update({f'PPO25u{u}': Path(f'results/ppo_s2r/25/checkpoint/model_{u*44000}.pt')
                   for u in cli.updates})
    loaded = {}
    for name, path in models.items():
        ck = torch.load(path, map_location='cpu', weights_only=False)
        actor = IL.actor_from_state(ck['actor']); actor.eval()
        mean, var = ck.get('obs_mean', rms.mean), ck.get('obs_var', rms.var)
        loaded[name] = actor, np.asarray(mean), np.sqrt(np.asarray(var)+1e-8)
    report = dict(status='running', model_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest()
                  for k,p in models.items()}, families={}, limitations=[
        'Successful source-policy training trajectories only; no new completion or lap-time evidence.',
        'Action disagreement is not proof the new action is worse or that BC caused the drift.',
        'Equal deterministic sample count per successful trajectory; no held-out tracks used for training.',
        'Steering bins use source commands, not actual curvature/straight-phase labels.'])
    for family in ('regular', 'open'):
        root = Path(f'results/{family}_replay_1126')
        collection = json.loads((root/'collection.json').read_text())
        assert collection['status'] == 'complete'
        assert collection['source_sha256'] == report['model_sha256']['source2130_transferred']
        rng = np.random.default_rng(1126)
        obs, targets, tracks = [], [], []
        for row in sorted(collection['runs'], key=lambda r:r['track']):
            if not row['complete']:
                continue
            with np.load(root/'shards'/(Path(row['track']).stem+'.npz')) as z:
                ix = rng.choice(len(z['O']), min(512,len(z['O'])), replace=False)
                obs.append(z['O'][ix].astype(np.float32)); targets.append(z['Y'][ix])
                tracks.append(Path(row['track']).name)
        O, Y = np.concatenate(obs), np.concatenate(targets)
        assert O.shape[1] == 132 and np.isfinite(O).all() and np.isfinite(Y).all()
        stats = {}; frozen_prediction = None
        for name, (actor,mean,std) in loaded.items():
            preds = []
            with torch.no_grad():
                for start in range(0,len(O),4096):
                    X = torch.as_tensor(((O[start:start+4096]-mean)/std).astype(np.float32))
                    preds.append(actor(X).loc.clamp(-1,1).numpy())
            prediction = np.concatenate(preds)
            signed = prediction-Y
            error = np.abs(signed)
            assert np.isfinite(error).all()
            stats[name] = dict(steer_normalized_mae=float(error[:,0].mean()),
                              speed_mps_mae=float(3.75*error[:,1].mean()),
                              speed_mps_bias=float(3.75*signed[:,1].mean()),
                              steer_normalized_p99=float(np.quantile(error[:,0],.99)),
                              action_abs_error_p99=float(np.quantile(error,.99)))
            if name == 'frozenIL1607':
                frozen_prediction = prediction
            elif name.startswith('PPO'):
                drift = np.abs(prediction-frozen_prediction)
                stats[name]['versus_IL1607'] = dict(steer_normalized_mae=float(drift[:,0].mean()),
                                                   speed_mps_mae=float(3.75*drift[:,1].mean()))
            if name == 'source2130_transferred':
                assert stats[name]['action_abs_error_p99'] < .01, 'Cache/source mapping mismatch'
        report['families'][family] = dict(samples=len(O),tracks=tracks,statistics=stats)
        print(family, json.dumps(stats), flush=True)
    report['status'] = 'complete'
    report['decision'] = 'Use drift as one input to later retention ablation; require matched completion/pace tests before changing PPO.'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
