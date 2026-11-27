#!/usr/bin/env python3
"""Compare frozen candidate starting policies on the audited training failures."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC/'installation'), str(ROOT/'tools')]
from collect_expert_recovery_probe_1126 import sha

POLICIES = {
    'source25u70': 'results/ppo_s2r/25/checkpoint/model_3080000.pt',
    'source29u20': 'results/ppo_s2r/29/checkpoint/model_880000.pt',
    'original_recovery100': 'results/recovery_adaptation_1126/checkpoint/model_100.pt',
    'stress_recovery100': 'results/stress_recovery_adaptation_1126/stress_rehearsal/checkpoint/model_100.pt',
}


def initialize():
    import collect_expert_recovery_probe_1126 as C
    C.initialize('/tmp/recovery_corpus_evaluation', True)


def drive(job):
    import collect_expert_recovery_probe_1126 as C
    name, row = job
    np, torch = C.np, C.torch
    ck = torch.load(POLICIES[name], map_location='cpu', weights_only=False)
    if 'obs_mean' in ck or 'obs_var' in ck:
        np.testing.assert_array_equal(ck['obs_mean'], C.MEAN)
        np.testing.assert_array_equal(np.sqrt(ck['obs_var']+1e-8), C.STD)
    actor = C.IL.actor_from_state(ck['actor']); actor.eval()
    _, env, obs = C.environment(row['track'], row['seed'])
    try:
        while True:
            with torch.no_grad():
                action = np.clip(actor(torch.as_tensor((obs-C.MEAN)/C.STD, dtype=torch.float32)).loc.numpy(), -1, 1)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        result = dict(policy=name, case_id=row['case_id'], track=row['track'], seed=row['seed'],
                      complete=bool(env.success), collision=bool(env.collision), seconds=float(env.elapsed),
                      lap_times=[float(t) for t in env.lap_times])
        # Reproduce the actual student failures that generated both collections.
        if sha(POLICIES[name]) == row['source_actor_sha256']:
            assert not result['complete'] and abs(result['seconds']-row['baseline_seconds']) < 1e-8
            result['collection_baseline_reproduced'] = True
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    cli = parser.parse_args()
    if not 1 <= cli.workers <= 8:
        raise ValueError('Use1to8 diagnostic workers')
    os.chdir(SRC)
    out = Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve previous evaluation')
    corpus_path = Path('results/recovery_corpus_round2_1126/manifest.json')
    corpus = json.loads(corpus_path.read_text())
    assert corpus['status'] == 'prepared_not_applied' and corpus['total_cases'] == len(corpus['cases']) == 13
    for collection in corpus['collections']:
        assert sha(collection['path']) == collection['sha256']
        assert sha(collection['audit']) == collection['audit_sha256']
    allowed = {str(p.resolve()) for p in Path('installation/maps_open_train').glob('*.npz')}
    for row in corpus['cases']:
        assert row['track'] in allowed and sha(row['track']) == row['track_sha256']
        assert sha(row['shard']) == row['shard_sha256']
    # Validate every checkpoint in the parent before creating the process pool.
    import collect_expert_recovery_probe_1126 as C
    for name, path in POLICIES.items():
        C.initialize('/tmp/recovery_corpus_evaluation', True, path)
    report = dict(status='running', corpus_sha256=sha(corpus_path),
                  policies={k: dict(path=p, sha256=sha(p)) for k, p in POLICIES.items()},
                  config_sha256=sha(C.CONFIG), normalizer_sha256=sha('results/imitation/1607/sim2real/agent_obs_scale.pkl'),
                  dynamics_sha256=sha('configs/task/dynamic.yaml'), cases=13, runs=[],
                  qualification='Frozen-policy from-reset controls on13 audited training failures; cap8 nominal dynamics, exact per-case seeds and shared frozenIL1607 normalization. Case selection was based on two students, so this is diagnostic, not independent qualification or global model ranking. No teacher handoff, gradients, model selection automation or vehicle actuation.')
    out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temp = out.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); temp.replace(out)
    save()
    jobs = [(name, row) for name in POLICIES for row in corpus['cases']]
    with mp.get_context('spawn').Pool(cli.workers, initializer=initialize) as pool:
        for result in pool.imap_unordered(drive, jobs):
            report['runs'].append(result); save(); print('CASE', result, flush=True)
    assert len(report['runs']) == 52
    assert len({(r['policy'], r['case_id']) for r in report['runs']}) == 52
    assert sum(r.get('collection_baseline_reproduced', False) for r in report['runs']) == 13
    assert all(sha(p) == report['policies'][k]['sha256'] for k, p in POLICIES.items())
    report['summary'] = {name: dict(complete=sum(r['complete'] for r in report['runs'] if r['policy'] == name), total=13,
                                   failures=[r['case_id'] for r in report['runs'] if r['policy'] == name and not r['complete']]) for name in POLICIES}
    report['status'] = 'complete_read_only_not_selected'; save(); print('DONE', report['summary'], flush=True)


if __name__ == '__main__':
    main()
