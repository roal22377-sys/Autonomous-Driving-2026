#!/usr/bin/env python3
"""Collect actual missing-preview trajectories on all training short/hall maps."""
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


def initialize(source):
    import collect_expert_recovery_probe_1126 as C
    C.initialize('/tmp/lost_rehearsal', True, source)


def drive(job):
    import collect_expert_recovery_probe_1126 as C
    np, torch = C.np, C.torch
    family, track, seed, shard = job
    _, env, obs = C.environment(track, seed)
    observations, targets = [], []
    try:
        while True:
            assert obs[C.COLS.index('preview/valid')] == 0
            with torch.no_grad():
                target = C.ACTOR(torch.as_tensor((obs-C.MEAN)/C.STD, dtype=torch.float32)).loc.numpy()
            observations.append(obs.copy()); targets.append(target.copy())
            obs, _, term, trunc, _ = env.step(np.clip(target, -1, 1))
            if term or trunc:
                break
        O, T = np.asarray(observations, np.float32), np.asarray(targets, np.float32)
        assert O.shape == (len(O), 132) and T.shape == (len(O), 2)
        assert np.isfinite(O).all() and np.isfinite(T).all()
        if env.success:
            assert not env.collision and len(env.lap_times) == 2
        np.savez_compressed(shard, O=O, T=T, Y=np.clip(T, -1, 1), columns=np.asarray(C.COLS))
        return dict(family=family, track=track, track_sha256=sha(track), seed=seed,
                    complete=bool(env.success), collision=bool(env.collision), steps=len(O),
                    seconds=float(env.elapsed), lap_times=[float(t) for t in env.lap_times],
                    shard=shard, shard_sha256=sha(shard), admissible_for_rehearsal=bool(env.success))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='results/ppo_s2r/25/checkpoint/model_3080000.pt')
    parser.add_argument('--destination', required=True)
    parser.add_argument('--workers', type=int, default=4)
    cli = parser.parse_args()
    if not 1 <= cli.workers <= 8:
        raise ValueError('Use1to8 workers')
    os.chdir(SRC)
    dest = Path(cli.destination)
    if dest.exists():
        raise FileExistsError('Preserve earlier collection')
    os.environ['F1_LOC_LOST'] = '1'
    initialize(cli.source)
    import collect_expert_recovery_probe_1126 as C
    from utils.training_mix import groups_for
    args = C.IL.get_args(['--task_config', C.CONFIG])
    groups = groups_for(args, [p for p in C.IL.list_variants(C.IL.MAPS) if C.IL.HOLDOUT not in p])
    manifest = Path('results/imitation/14/backup/track_manifest.json')
    training = json.loads(manifest.read_text())['training']
    for family in ['short', 'hall']:
        assert {str(Path(p).resolve()) for p in groups[family]} == {str(Path(p).resolve()) for p in training[family]}
    jobs = []
    for family in ['short', 'hall']:
        for path in sorted(Path(p).resolve() for p in groups[family]):
            jobs.append((family, str(path), 66000+len(jobs), str((dest/'shards'/(family+'_'+path.stem+'.npz')).resolve())))
    assert len(groups['short']) == 90 and len(groups['hall']) == 101 and len(jobs) == 191
    artifacts = dict(source=Path(cli.source), config=Path(C.CONFIG),
                     normalizer=Path('results/imitation/1607/sim2real/agent_obs_scale.pkl'),
                     dynamics=Path('configs/task/dynamic.yaml'), wrapper=Path('utils/env.py'),
                     collector=Path(__file__), training_manifest=manifest)
    report = dict(status='running', source_checkpoint=cli.source,
                  artifacts={k:dict(path=str(p), sha256=sha(p)) for k,p in artifacts.items()},
                  simulation_mode='eval', cap=8., start='center', missing_preview=True,
                  jobs=[dict(family=f, track=p, track_sha256=sha(p), seed=s, shard=z) for f,p,s,z in jobs], runs=[],
                  qualification='All191predeclared training short/hall variants under actual full-episode missingpreview with LiDAR retained. Source25u70 from-reset frozen-policy behavior; only successful complete trajectories eligible for future rehearsal after audit. Failed trajectories arediagnostic. No heldout evaluation observations, actual relocalization, gradientupdate, replaycache adoption or real-car operation.')
    (dest/'shards').mkdir(parents=True)
    def save():
        temp=dest/'collection.tmp'; temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); temp.replace(dest/'collection.json')
    save()
    with mp.get_context('spawn').Pool(cli.workers, initializer=initialize, initargs=(cli.source,)) as pool:
        for result in pool.imap_unordered(drive, jobs):
            report['runs'].append(result); save()
            if len(report['runs']) % 10 == 0 or not result['complete']:
                print('COLLECT',len(report['runs']),len(jobs),'success',sum(r['complete'] for r in report['runs']),flush=True)
    assert len(report['runs']) == len({r['seed'] for r in report['runs']}) == 191
    assert all(sha(p) == report['artifacts'][k]['sha256'] for k,p in artifacts.items())
    report.update(status='complete_not_applied', completed=sum(r['complete'] for r in report['runs']))
    save(); print('DONE',report['completed'],'/191',flush=True)


if __name__ == '__main__':
    main()
