#!/usr/bin/env python3
"""Measure action-cache fit separately from closed-loop driving performance."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=int, default=28)
    parser.add_argument('--control', type=int, default=27)
    parser.add_argument('--update', required=True, type=int)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    os.chdir(SRC)
    import numpy as np
    import torch
    import train_ppo_s2r as PPO
    from utils.ppo_retention import RetentionCache
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    out = Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve previous diagnostic: ' + str(out))
    args = PPO.IL.get_args(['--task_config', 'results/ppo_s2r/%d/backup/f1tenth.yaml' % cli.run])
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load('results/imitation/1607/sim2real', required=True)
    cache_path = Path('results/ppo_retention_candidate_1126/replay_cache.npz')
    manifest = Path('results/ppo_retention_candidate_1126/preparation.json')
    cache = RetentionCache(args, rms, cache_path, manifest,
                           PPO.training_tracks(True, short=True), minibatch=2048, share=.125, seed=1127)
    warmup = json.loads(Path('results/preflight_1126/ppo%d_production_warmup_review.json' % cli.run).read_text())
    if cache.metadata != warmup['retention_metadata']:
        raise ValueError('Diagnostic split/normalization differs from the production cache')
    with np.load(cache_path, allow_pickle=False) as z:
        families = z['family'].copy()
    sources = dict(source_u70=Path('results/ppo_s2r/25/checkpoint/model_3080000.pt'),
        control=Path('results/ppo_s2r/%d/checkpoint/model_%d.pt' % (cli.control, cli.update * 44000)),
        candidate=Path('results/ppo_s2r/%d/checkpoint/model_%d.pt' % (cli.run, cli.update * 44000)))
    if not all(path.is_file() for path in sources.values()):
        raise FileNotFoundError('Wait for the complete checkpoint evaluation; no partial policy substitution')
    results = {}
    for label, path in sources.items():
        state = torch.load(path, map_location='cpu', weights_only=False)
        actor = PPO.IL.actor_from_state(state['actor'])
        actor.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(families), 2048):
                predictions.append(actor(cache.data.X[i:i+2048].float()).loc.numpy())
        prediction = np.concatenate(predictions)
        target = cache.data.T.numpy()
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite actor predictions')
        rows = {}
        for split, indices in (('train_rows', cache.data.train.numpy()), ('withheld_rows', cache.data.val.numpy())):
            rows[split] = {}
            for family in ('all', 'regular', 'open'):
                selected = indices if family == 'all' else indices[families[indices] == family]
                if not len(selected):
                    raise ValueError('Empty cache stratum: ' + split + '/' + family)
                raw = ((prediction[selected] - target[selected]) ** 2).mean(axis=0)
                clipped = ((np.clip(prediction[selected], -1, 1) - target[selected]) ** 2).mean(axis=0)
                rows[split][family] = dict(samples=len(selected), steer_mse=float(raw[0]),
                    speed_mse=float(raw[1]), weighted_loss=float((raw[0] + .25 * raw[1]) / 2),
                    clipped_steer_mse=float(clipped[0]), clipped_speed_mse=float(clipped[1]))
        results[label] = dict(checkpoint=str(path), checkpoint_sha256=sha(path), strata=rows)
    report = dict(status='complete_read_only', update=cli.update, cache_metadata=cache.metadata, results=results,
        qualification=('Fixed normalized observations/action targets from successful training trajectories. '
                       'The2% withheld rows share trajectories and geometries with training rows; they '
                       'are only a supervised-fit diagnostic, never unseen-track validation. Lower MSE '
                       'does not establish better steering dynamics, completion, or lap time. '
                       'No optimizer step, environment rollout, or live learner mutation.'))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for label, row in results.items():
        print(label, row['strata']['withheld_rows']['all'])


if __name__ == '__main__':
    main()
