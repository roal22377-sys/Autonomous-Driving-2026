#!/usr/bin/env python3
"""Validate training-only adverse-dynamics source trajectories before rehearsal."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation'), str(ROOT / 'tools')]
from collect_expert_recovery_probe_1126 import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    os.chdir(SRC)
    source = Path(cli.collection); out = Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve earlier audit')
    doc = json.loads(source.read_text())
    assert doc['status'] == 'complete_not_applied'
    assert doc['simulation_mode'] == 'worst' and doc['cap'] == 8.
    for artifact in doc['artifacts'].values():
        assert sha(artifact['path']) == artifact['sha256']
    import collect_expert_recovery_probe_1126 as C
    C.initialize(str(source.parent), True, doc['source_checkpoint'])
    np, torch = C.np, C.torch
    from utils.training_mix import groups_for
    args = C.IL.get_args(['--task_config', C.CONFIG])
    assert sha(C.CONFIG) == doc['artifacts']['config']['sha256']
    variants = [p for p in C.IL.list_variants(C.IL.MAPS) if C.IL.HOLDOUT not in p]
    allowed = {str(Path(p).resolve()) for p in groups_for(args, variants)['regular']}
    tracks = sorted(Path(p).resolve() for p in variants if Path(p).stem.endswith(('_w101', '_w103')))
    assert len(tracks) == 92 and all(str(p) in allowed for p in tracks)
    expected = [(str(p), start) for p in tracks for start in ('center', 'random')]
    jobs = {(r['track'], r['seed'], r['start']): r for r in doc['jobs']}
    assert len(jobs) == len(doc['jobs']) == len(doc['runs']) == 184
    assert [(r['track'], r['start'], r['seed']) for r in doc['jobs']] == [(p, start, 65000+i) for i, (p, start) in enumerate(expected)]
    assert {(r['track'], r['seed'], r['start']) for r in doc['runs']} == set(jobs)
    summaries = []
    total_rows = success_rows = 0
    maximum_batch_error = maximum_single_error = 0.
    rng = np.random.default_rng(112672)
    for row in sorted(doc['runs'], key=lambda r: r['seed']):
        job = jobs[row['track'], row['seed'], row['start']]
        assert row['track_sha256'] == job['track_sha256'] == sha(row['track'])
        assert row['shard'] == job['shard'] and sha(row['shard']) == row['shard_sha256']
        assert row['admissible_for_rehearsal'] == row['complete']
        assert abs(row['seconds'] - row['steps'] * args.timestep) < 1e-7
        if row['complete']:
            assert not row['collision'] and len(row['lap_times']) == 2
            assert row['lap_times'][1] == row['seconds'] and row['lap_times'][0] < row['lap_times'][1]
        with np.load(row['shard'], allow_pickle=False) as z:
            O, T, Y = z['O'], z['T'], z['Y']
            n = row['steps']
            assert O.shape == (n, 132) and T.shape == Y.shape == (n, 2)
            assert O.dtype == T.dtype == Y.dtype == np.float32
            assert list(z['columns']) == C.COLS
            assert np.isfinite(O).all() and np.isfinite(T).all() and np.isfinite(Y).all()
            np.testing.assert_array_equal(np.clip(T, -1, 1), Y)
            assert np.all(O[:, C.COLS.index('speed_cap')] == 8)
            X = torch.as_tensor((O-C.MEAN)/C.STD, dtype=torch.float32)
            with torch.no_grad():
                predictions = torch.cat([C.ACTOR(X[i:i+2048]).loc for i in range(0, n, 2048)]).numpy()
                error = abs(predictions-T)
                # Batch GEMM and rollout vector GEMV have different float32 rounding.
                # Check all rows tightly, then reproduce random and maximal-error rows
                # using the exact single-row inference shape used by the collector.
                assert float(error.max()) < 1e-5
                selected = np.unique(np.r_[rng.choice(n, min(n, 128), replace=False), np.unravel_index(error.argmax(), error.shape)[0]])
                exact = np.stack([C.ACTOR(X[i]).loc.numpy() for i in selected])
                np.testing.assert_array_equal(exact, T[selected])
            maximum_batch_error = max(maximum_batch_error, float(error.max()))
            maximum_single_error = max(maximum_single_error, float(abs(exact-T[selected]).max()))
            total_rows += n
            success_rows += n if row['complete'] else 0
            summaries.append(dict(track=row['track'], seed=row['seed'], start=row['start'],
                                  complete=row['complete'], rows=n, shard=row['shard'], shard_sha256=row['shard_sha256'],
                                  raw_action_batch_max_error=float(error.max()), exact_single_rows=len(selected)))
    result = dict(status='passed_not_applied', collection_sha256=sha(source),
                  source_sha256=doc['artifacts']['source']['sha256'], cases=184,
                  completed=sum(r['complete'] for r in doc['runs']), total_rows=total_rows,
                  admissible_success_rows=success_rows, maximum_batch_label_error=maximum_batch_error,
                  maximum_single_label_error=maximum_single_error, cases_audited=summaries,
                  qualification='All training cases, artifacts, raw labels, applied actions, timing and successful two-lap outcomes verified. Float32 batch roundoff checked against exact single-row reconstruction. Only successful episodes eligible for future rehearsal. No cache or model adoption; known-base training augmentation, not unseen-course or hardware qualification.')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ['status', 'cases', 'completed', 'total_rows', 'admissible_success_rows', 'maximum_batch_label_error']}))


if __name__ == '__main__':
    main()
