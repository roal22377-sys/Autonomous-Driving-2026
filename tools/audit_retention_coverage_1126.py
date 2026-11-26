#!/usr/bin/env python3
"""Read-only coverage and production-loader audit of two frozen replay banks."""
import hashlib
import json
import os
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
    import train_ppo_s2r as PPO
    from utils.observation import get_obs_config
    from utils.ppo_retention import RetentionCache
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    out = Path('results/analysis1126/variant_retention_coverage_and_production_audit.json')
    if out.exists():
        raise FileExistsError('Preserve existing audit')
    task = Path('results/ppo_s2r/29/backup/f1tenth.yaml')
    args = PPO.IL.get_args(['--task_config', str(task)])
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load('results/imitation/1607/sim2real', required=True)
    warm = json.loads(Path('results/preflight_1126/ppo29_production_warmup_review.json').read_text())
    paths = dict(control=Path('results/ppo_retention_candidate_1126'),
                 candidate=Path('results/ppo_variant_retention_candidate_1126'))
    results = {}
    for label, directory in paths.items():
        cache_path, manifest_path = directory/'replay_cache.npz', directory/'preparation.json'
        manifest = json.loads(manifest_path.read_text())
        cache = RetentionCache(args, rms, cache_path, manifest_path,
            PPO.training_tracks(True, short=True), minibatch=2048, share=.125, seed=1127)
        if label == 'control':
            assert cache.metadata == warm['retention_metadata']
        else:
            for key, value in warm['retention_metadata'].items():
                if key not in ('cache_sha256', 'manifest_sha256'):
                    assert cache.metadata[key] == value, key
        with np.load(cache_path, allow_pickle=False) as z:
            O, Y = z['O'].astype(np.float32), z['Y'].astype(np.float32)
            columns, family, track_id = list(z['columns']), z['family'], z['track_id']
        # Compare every row, rather than a sample, under the actual learner task.
        expected = torch.as_tensor(rms.normalize(O), dtype=torch.float16)
        assert torch.equal(cache.data.X, expected)
        assert np.array_equal(cache.data.T.numpy(), Y)
        col = lambda name: O[:, columns.index(name)]
        curvature = np.max(np.abs(np.stack([col('preview/curv/'+d) for d in ('0.5','1','2')])), axis=0)
        heading = np.abs(col('preview/e_psi'))
        previous = col('history/0/steer')
        masks = dict(all=np.ones(len(O), dtype=bool),
            near_curvature_ge045=curvature >= .45,
            near_curvature_ge1=curvature >= 1.,
            reference_heading_gt03=heading > .3,
            reference_heading_gt08=heading > .8,
            lateral_offset_gt05=np.abs(col('preview/e_y')) > .5,
            large_steer_target=np.abs(Y[:, 0]) >= .8,
            strong_command_sign_reversal=(previous*Y[:, 0] < 0) & (np.abs(previous) > .25) & (np.abs(Y[:, 0]) > .25),
            missing_preview=col('preview/valid') <= .5,
            lower_cap=col('speed_cap') < 8.)
        families = {}
        for name in ('regular', 'open'):
            eligible = family == name
            families[name] = {}
            for stratum, mask in masks.items():
                chosen = eligible & mask
                families[name][stratum] = dict(rows=int(chosen.sum()),
                    fraction=float(chosen.sum()/eligible.sum()),
                    distinct_tracks=int(len(np.unique(track_id[chosen]))))
        origins = {}
        for i, row in enumerate(manifest['tracks']):
            origin = row.get('origin', 'original_'+row['family'])
            counts = origins.setdefault(origin, dict(rows=0, tracks=0, high_curvature_rows=0, large_heading_rows=0))
            mask = track_id == i
            counts['rows'] += int(mask.sum()); counts['tracks'] += 1
            counts['high_curvature_rows'] += int((mask & masks['near_curvature_ge045']).sum())
            counts['large_heading_rows'] += int((mask & masks['reference_heading_gt08']).sum())
        results[label] = dict(metadata=cache.metadata, normalized_rows_verified=len(O),
            families=families, origins=origins, cache_sha256_after=sha(cache_path))
        assert results[label]['cache_sha256_after'] == cache.metadata['cache_sha256']
    il_args = PPO.IL.get_args(['--task_config', 'results/imitation/1607/backup/f1tenth.yaml'])
    canonical = lambda x: json.loads(json.dumps(get_obs_config(x), default=lambda v: v.tolist()))
    old, production = canonical(il_args), canonical(args)
    differences = {}
    def compare(a, b, prefix=''):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a.keys() | b.keys():
                compare(a.get(key), b.get(key), prefix+'/'+key)
        elif a != b:
            differences[prefix] = [a,b]
    compare(old, production)
    assert set(differences) == {'/preview/noise/lost_full_prob', '/preview/noise/lost_len', '/preview/noise/lost_episode_prob'}
    report = dict(status='passed_read_only', task_config=str(task), task_sha256=sha(task),
        initial_builder_vs_production_observation_config=differences, results=results,
        qualification=('All rows loaded using the actual PPO29 task and frozen IL1607 normalization; '
            'only cache and manifest identities differ from production sampler metadata. The initial '
            'candidate builder used the IL task, whose three outage-randomization settings differ; '
            'fixed cached observations and action units are unchanged, and this audit verifies their '
            'production loading separately without rewriting the old audit or data. Strata overlap; '
            'curvature is a preview feature, not an identified180-degree hairpin. Reference heading '
            'is not localization error and does not by itself prove recovery. Strong sign reversal '
            'compares the next target with the last command, not actual wheel response. Successful '
            'training-only teacher trajectories do not cover all failed/off-policy states or prove '
            'better closed-loop completion. No optimizer, sampling-policy or live-cache changes.'))
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for label, row in results.items():
        print(label, json.dumps(row['families']['regular']), flush=True)


if __name__ == '__main__':
    main()
