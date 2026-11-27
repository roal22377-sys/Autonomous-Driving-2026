#!/usr/bin/env python3
"""Return control to the student after a bounded training-only expert correction.

Use every rescued case in a completed, audited collection. Recorded expert
actions are used only while every observation and true state exactly reproduces
the saved expert continuation. No new labels, gradients, or vehicle actions.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import collect_expert_recovery_probe_1126 as collector


def initialize():
    collector.initialize('results/expert_recovery_failures_1126', True)


def run(job):
    row, seconds = job
    np, torch = collector.np, collector.torch
    assert collector.sha(row['shard']) == row['shard_sha256']
    assert collector.sha(row['track']) == row['track_sha256']
    with np.load(row['shard'], allow_pickle=False) as z:
        observation, action, state, times = [z[k].copy() for k in ('O','Y','state','command_time')]
    _, env, obs = collector.environment(row['track'], row['seed'])
    start = row['switch_step']
    count = int(round(seconds/env.timestep))
    end = start + count
    assert 0 < count < len(action)
    step = verified = 0
    returned = False
    try:
        while True:
            if start <= step <= end:
                i = step-start
                np.testing.assert_array_equal(obs, observation[i])
                np.testing.assert_array_equal(env.vehicle_state(), state[i])
                assert float(env.elapsed) == times[i]
                verified += 1
            if start <= step < end:
                chosen = action[step-start]
            else:
                with torch.no_grad():
                    X = torch.as_tensor((obs-collector.MEAN)/collector.STD, dtype=torch.float32)
                    chosen = np.clip(collector.ACTOR(X).loc.numpy(), -1, 1)
                if step == end:
                    returned = True
            obs, _, term, trunc, _ = env.step(chosen)
            step += 1
            if term or trunc:
                break
        assert returned and verified == count+1
        return dict(track=Path(row['track']).name, track_sha256=row['track_sha256'], seed=row['seed'],
            handoff_at_reset=row['handoff_at_reset'],
            expert_seconds=seconds, switch_time=row['switch_time'],
            returned_at=row['switch_time']+count*env.timestep,
            verified_expert_states=verified, student_baseline=row['baseline'], full_expert=row['handoff'],
            bounded_expert=dict(success=bool(env.success), collision=bool(env.collision),
                seconds=float(env.elapsed), lap_times=[float(t) for t in env.lap_times],
                terminal_xy=env.vehicle_state()[:2].tolist()), shard_sha256=row['shard_sha256'])
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    os.chdir(collector.SRC)
    out = Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve previous probe')
    directory = Path('results/expert_recovery_failures_1126')
    path = directory/'collection.json'
    meta = json.loads(path.read_text())
    audit_path = directory/'audit.json'
    audit = json.loads(audit_path.read_text())
    assert meta['status'] == 'complete_not_applied'
    assert audit['status'] == 'passed_not_applied' and audit['collection_sha256'] == collector.sha(path)
    assert collector.sha(collector.STUDENT) == meta['student_sha256']
    for key, artifact in [('config',collector.CONFIG), ('dynamics','configs/task/dynamic.yaml'),
                          ('normalizer','results/imitation/1607/sim2real/agent_obs_scale.pkl')]:
        assert collector.sha(artifact) == meta[key+'_sha256']
    rows = sorted((r for r in meta['runs'] if r.get('outcome') == 'rescued'), key=lambda r:(r['track'],r['seed']))
    assert rows
    jobs = [(row, seconds) for row in rows for seconds in (1., 2.)]
    report = dict(status='running', collection_sha256=collector.sha(path), audit_sha256=collector.sha(audit_path),
        student_sha256=meta['student_sha256'], collector_sha256=collector.sha(Path(collector.__file__)),
        tool_sha256=collector.sha(Path(__file__).resolve()), planned=len(jobs), runs=[],
        qualification='All rescued training cases, fixed1s/2s correction windows, same seeds and nominal dynamics. Exact saved expert states/observations checked through the return instant. Student resumes for the rest of both laps. This scripted privileged correction probes persistence of recovery; it is not an autonomous recovery policy, new model or held-out performance claim. No gradient updates or deployment.')
    out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temp = out.with_suffix('.tmp')
        temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temp.replace(out)
    save()
    with mp.get_context('spawn').Pool(2, initializer=initialize) as pool:
        for row in pool.imap_unordered(run,jobs):
            report['runs'].append(row)
            save()
            print(row['track'],row['expert_seconds'],row['bounded_expert'],flush=True)
    report['status'] = 'complete_read_only'
    save()


if __name__ == '__main__':
    main()
