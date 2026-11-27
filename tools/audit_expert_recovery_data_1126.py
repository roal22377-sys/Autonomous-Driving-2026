#!/usr/bin/env python3
"""Audit frozen training-failure continuations and same-input actor disagreement.

Requires completed collection. Never updates a model or constructs a replay bank.
"""
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', default='results/expert_recovery_failures_1126/collection.json')
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    os.chdir(SRC)
    import numpy as np
    import torch
    import train_il_s2r as IL
    from ruamel.yaml import YAML
    from utils.policy_transfer import columns
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    out = Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve previous audit')
    source = Path(cli.collection)
    raw = source.read_bytes()
    report = json.loads(raw)
    assert report['status'] == 'complete_not_applied'
    assert report['protocol'] == 'all_training_open_failures'
    config = Path('results/ppo_s2r/29/backup/f1tenth.yaml')
    artifacts = dict(student=Path(report['student_checkpoint']), config=config,
        normalizer=Path('results/imitation/1607/sim2real/agent_obs_scale.pkl'),
        dynamics=Path('configs/task/dynamic.yaml'),
        expert_config=Path('configs/task/f1tenth_preview_1126.yaml'),
        planner_code=Path('utils/rule_based.py'))
    for label, path in artifacts.items():
        assert sha(path) == report[label + '_sha256'], label
    cfg = YAML().load(config.read_text())
    args = IL.get_args(['--task_config', str(config)])
    expected_columns = columns(args)
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load('results/imitation/1607/sim2real', required=True)
    paths = dict(source_u70=Path('results/ppo_s2r/25/checkpoint/model_3080000.pt'),
        student_u20=artifacts['student'], candidate_u30=Path('results/ppo_s2r/29/checkpoint/model_1320000.pt'))
    actors = {}
    for name, path in paths.items():
        actor = IL.actor_from_state(torch.load(path, map_location='cpu', weights_only=False)['actor'])
        actor.eval()
        actors[name] = actor
    allowed = {str(p.resolve()) for p in Path('installation/maps_open_train').glob('*.npz')}
    jobs = {(r['track'], r['seed']): r for r in report['jobs']}
    runs = report['runs']
    assert len(jobs) == len(runs) == 184
    assert len({(r['track'], r['seed']) for r in runs}) == 184
    assert all(track in allowed for track, _ in jobs)
    assert [(r['track'], r['seed']) for r in report['jobs']] == [
        (track, 64000+i) for i, track in enumerate(sorted(r['track'] for r in report['jobs']))]
    summaries = []
    rows = 0
    maximum_steer_error = maximum_speed_error = 0.
    dt = float(cfg['timestep'])
    s2r = cfg['sim2real']
    for r in runs:
        assert (r['track'], r['seed']) in jobs
        assert r['track_sha256'] == sha(r['track']) == jobs[(r['track'], r['seed'])]['track_sha256']
        if r['baseline']['success']:
            assert r['status'] == 'baseline_complete_no_handoff' and 'shard' not in r
            assert not r['admissible_for_future_review']
            continue
        assert r['switch_step'] == max(0, r['baseline']['steps'] - round(2/dt))
        assert r['handoff_at_reset'] == (r['switch_step'] == 0)
        assert r['prefix_observations_and_states_exact'] == r['switch_step'] + 1
        assert r['admissible_for_future_review'] == r['handoff']['success']
        if r['handoff']['success']:
            assert not r['handoff']['collision'] and len(r['handoff']['lap_times']) == 2
        assert sha(r['shard']) == r['shard_sha256']
        with np.load(r['shard'], allow_pickle=False) as z:
            O, Y = z['O'], z['Y']
            n = len(O)
            assert z['columns'].tolist() == expected_columns
            assert O.shape == (n, 132) and Y.shape == (n, 2) and z['state'].shape == (n, 7)
            assert O.dtype == Y.dtype == np.float32
            assert all(np.isfinite(z[key]).all() for key in z.files if key != 'columns')
            assert np.max(abs(Y)) <= 1
            assert np.all(O[:, expected_columns.index('preview/valid')] > .5)
            assert np.all(O[:, expected_columns.index('speed_cap')] == 8)
            assert n == r['rows'] == r['handoff']['steps'] - r['switch_step']
            np.testing.assert_allclose(z['command_time'], r['switch_time'] + np.arange(n)*dt, rtol=0, atol=1e-7)
            np.testing.assert_allclose(z['seconds_since_handoff'], z['command_time']-r['switch_time'], rtol=0, atol=1e-10)
            early = z['seconds_since_handoff'] <= 2
            assert int(early.sum()) == r['first_two_second_rows']
            y = Y.astype(np.float64)
            x = abs(y[:, 0])
            lock = np.where(y[:, 0] > 0, min(s2r['steer_left_max_nominal'], cfg['max_steer']),
                            min(s2r['steer_right_max_nominal'], cfg['max_steer']))
            k = np.where(y[:, 0] > 0, s2r['steer_scale_left_nominal'], s2r['steer_scale_right_nominal'])
            wheel = np.sign(y[:, 0])*lock*x*(k+(1-k)*x*x)
            speed = (y[:, 1]*(cfg['max_speed']-cfg['min_speed'])+cfg['max_speed']+cfg['min_speed'])/2
            se = float(np.max(abs(wheel-z['teacher_wheel'])))
            ve = float(np.max(abs(speed-z['teacher_speed'])))
            assert se < 1e-6 and ve < 1e-6
            maximum_steer_error = max(maximum_steer_error, se)
            maximum_speed_error = max(maximum_speed_error, ve)
            fit = {}
            # Match rollout normalization exactly, not the float16 replay-cache encoding.
            X = torch.as_tensor((O-rms.mean)/np.sqrt(rms.var+1e-8), dtype=torch.float32)
            with torch.no_grad():
                for name, actor in actors.items():
                    prediction = np.concatenate([actor(X[i:i+2048]).loc.numpy() for i in range(0,n,2048)])
                    assert np.isfinite(prediction).all()
                    delta = np.clip(prediction,-1,1)-Y
                    fit[name] = {}
                    for label, mask in [('first_two_seconds',early), ('remaining_continuation',~early)]:
                        if not mask.any():
                            continue
                        fit[name][label] = dict(rows=int(mask.sum()),
                            normalized_action_mse=(delta[mask]**2).mean(axis=0).tolist(),
                            steering_absolute_error_p95=float(np.quantile(abs(delta[mask,0]),.95)))
            summaries.append(dict(track=Path(r['track']).name, seed=r['seed'], outcome=r['outcome'],
                handoff_at_reset=r['handoff_at_reset'], switch_time=r['switch_time'],
                admissible=r['admissible_for_future_review'], rows=n, early_rows=int(early.sum()),
                shard=r['shard'], shard_sha256=r['shard_sha256'], same_input_fit=fit))
            rows += n
    result = dict(status='passed_not_applied', collection_sha256=hashlib.sha256(raw).hexdigest(),
        cases=len(runs), baseline_complete=sum(r['baseline']['success'] for r in runs),
        rescued=sum(r.get('outcome') == 'rescued' for r in runs), rows=rows,
        successful_post_prefix_recoveries=sum(r.get('outcome') == 'rescued' and not r['handoff_at_reset'] for r in runs),
        successful_from_reset_interventions=sum(r.get('outcome') == 'rescued' and r['handoff_at_reset'] for r in runs),
        maximum_steering_label_error_rad=maximum_steer_error, maximum_speed_label_error_mps=maximum_speed_error,
        actors={name:dict(path=str(path),sha256=sha(path)) for name,path in paths.items()},
        cases_with_handoff=summaries,
        qualification='Training maps only; all184 declared cases, source hashes, raw inputs, whole-action units, timing and full2lap outcomes audited. Prefix equality was checked during collection. Actor disagreement on privileged teacher continuations is diagnostic, not a counterfactual rollout or evidence of partial-observation learnability. Rows within trajectories are correlated. No cache adoption, gradient update or held-out observation reuse.')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','cases','baseline_complete','rescued','rows')}))


if __name__ == '__main__':
    main()
