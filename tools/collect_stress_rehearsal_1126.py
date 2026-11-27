#!/usr/bin/env python3
"""Collect frozen-source actions on training variants under adverse dynamics.

Separate raw diagnostic shards; only full two-lap successes are eligible for
future rehearsal. Does not construct a cache, train, or promote a model.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation'), str(ROOT / 'tools')]
from collect_expert_recovery_probe_1126 import sha


def initialize(destination, source):
    import collect_expert_recovery_probe_1126 as C
    C.initialize(destination, True, source)


def drive(job):
    import collect_expert_recovery_probe_1126 as C
    from utils.env import F1Wrapper
    import random
    np, torch = C.np, C.torch
    path, seed, start, destination = job
    args = C.IL.get_args(['--task_config', C.CONFIG]); args.seed = seed
    np.random.seed(seed); random.seed(seed)
    env = F1Wrapper(args, C.IL.MAPS, obs_noise=True, sim_mode='worst', tracks=[path])
    env.fixed_speed_cap = 8.; env.start_mode = start
    observations, raw_actions, applied_actions = [], [], []
    try:
        obs, _ = env.reset(options={'track': path})
        while True:
            with torch.no_grad():
                action = C.ACTOR(torch.as_tensor((obs-C.MEAN)/C.STD, dtype=torch.float32)).loc.numpy()
            applied = np.clip(action, -1, 1)
            observations.append(obs.copy()); raw_actions.append(action.copy()); applied_actions.append(applied.copy())
            obs, _, term, trunc, _ = env.step(applied)
            if term or trunc:
                break
        O = np.asarray(observations, dtype=np.float32)
        T = np.asarray(raw_actions, dtype=np.float32)
        Y = np.asarray(applied_actions, dtype=np.float32)
        assert O.shape == (len(O), 132) and T.shape == Y.shape == (len(O), 2)
        assert np.isfinite(O).all() and np.isfinite(T).all() and np.isfinite(Y).all()
        np.testing.assert_array_equal(np.clip(T, -1, 1), Y)
        assert np.all(O[:, C.COLS.index('speed_cap')] == 8)
        if env.success:
            assert not env.collision and len(env.lap_times) == 2
        shard = Path(destination)
        np.savez_compressed(shard, O=O, T=T, Y=Y, columns=np.asarray(C.COLS))
        return dict(track=path, track_sha256=sha(path), seed=seed, start=start,
                    complete=bool(env.success), collision=bool(env.collision),
                    steps=len(O), seconds=float(env.elapsed), lap_times=[float(t) for t in env.lap_times],
                    shard=str(shard), shard_sha256=sha(shard),
                    admissible_for_rehearsal=bool(env.success))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--source', default='results/ppo_s2r/29/checkpoint/model_880000.pt')
    parser.add_argument('--destination', required=True)
    cli = parser.parse_args()
    if not 1 <= cli.workers <= 8:
        raise ValueError('Use between1 and8 diagnostic workers')
    os.chdir(SRC)
    dest = Path(cli.destination)
    if dest.exists():
        raise FileExistsError('Inspect prior collection; never overwrite or duplicate')
    # Validate the actor and embedded normalizer in the parent before spawning.
    initialize(str(dest), cli.source)
    import collect_expert_recovery_probe_1126 as C
    from utils.training_mix import groups_for
    args = C.IL.get_args(['--task_config', C.CONFIG])
    variants = [p for p in C.IL.list_variants(C.IL.MAPS) if C.IL.HOLDOUT not in p]
    training = groups_for(args, variants)
    allowed = {str(Path(p).resolve()) for p in training['regular']}
    tracks = sorted(Path(p).resolve() for p in variants if Path(p).stem.endswith(('_w101', '_w103')))
    expected = {m+d+'_w'+str(s)+'.npz' for m in C.IL.MAPS for d in ('', '_rev') for s in (101, 103)}
    assert len(tracks) == 92 and {p.name for p in tracks} == expected
    assert all(str(p) in allowed and C.IL.HOLDOUT not in str(p) for p in tracks)
    C.IL.require_fresh_variants([str(p) for p in tracks])
    jobs = []
    for path in tracks:
        for start in ('center', 'random'):
            jobs.append((str(path), 65000+len(jobs), start, str((dest/'shards'/(path.stem+'_'+start+'.npz')).resolve())))
    artifacts = dict(source=Path(cli.source), config=Path(C.CONFIG),
                     normalizer=Path('results/imitation/1607/sim2real/agent_obs_scale.pkl'),
                     dynamics=Path('configs/task/dynamic.yaml'),
                     simulator=Path('installation/f1tenth_gym/envs/dynamic_models.py'),
                     wrapper=Path('utils/env.py'), collector=Path(__file__))
    report = dict(status='running', simulation_mode='worst', cap=8.,
                  source_checkpoint=cli.source,
                  artifacts={k: dict(path=str(p), sha256=sha(p)) for k, p in artifacts.items()},
                  selection='Every training regular variant w101/w103, both directions, center/random starts; no heldoutw105 or case-selected evaluation observations.',
                  jobs=[dict(track=p, track_sha256=sha(p), seed=s, start=start, shard=shard) for p, s, start, shard in jobs],
                  runs=[], qualification='Frozen-source behavior under adverse simulator parameters; training geometry only, not a novel-layout or real-hardware result. Successful complete trajectories eligible for future rehearsal after raw-label/provenance audit. Failed episodes remain diagnostic. No cache adoption or learner mutation.')
    (dest/'shards').mkdir(parents=True)
    def save():
        temp = dest/'collection.tmp'; temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); temp.replace(dest/'collection.json')
    save(); begun = time.monotonic()
    with mp.get_context('spawn').Pool(cli.workers, initializer=initialize, initargs=(str(dest), cli.source)) as pool:
        for result in pool.imap_unordered(drive, jobs):
            report['runs'].append(result); save()
            if len(report['runs']) % 10 == 0 or not result['complete']:
                print('COLLECT', len(report['runs']), '/', len(jobs), 'complete', sum(r['complete'] for r in report['runs']), 'elapsed', round(time.monotonic()-begun), flush=True)
    assert len(report['runs']) == len(jobs) == 184
    assert len({(r['track'], r['seed'], r['start']) for r in report['runs']}) == 184
    assert all(sha(p) == report['artifacts'][k]['sha256'] for k, p in artifacts.items())
    report.update(status='complete_not_applied', completed=sum(r['complete'] for r in report['runs']), elapsed_seconds=time.monotonic()-begun)
    save(); print('DONE', report['completed'], '/', len(jobs), flush=True)


if __name__ == '__main__':
    main()
