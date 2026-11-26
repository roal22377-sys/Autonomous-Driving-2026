#!/usr/bin/env python3
"""Compare epoch versus minibatch KL stopping on one isolated fresh rollout.

Runs the actual production optimizer block on private copies. No production
source, checkpoint, training process or map is changed.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path

__file__ = str(Path(__file__).resolve())
ROOT = Path(__file__).parents[1]
SRC = ROOT / 'src/f1tenth'
extra = [os.environ['PPO_KL_STAGE']] if os.environ.get('PPO_KL_STAGE') else []
sys.path[:0] = extra + [str(SRC), str(SRC / 'installation')]
BASE_SHA = '8efbabe8723b8307f29ed397a387ebea9a65ab816cfe967b3838f203963580d9'


class ProbeDone(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-id', type=int, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--staged-learner', type=Path, help='Also verify the actual staged optimizer block')
    cli = parser.parse_args()
    staged_path = cli.staged_learner.resolve() if cli.staged_learner else None
    if cli.steps < 2048 or cli.steps % 5:
        raise ValueError('Use >=2048 steps divisible by five')
    os.chdir(SRC)
    import torch
    import numpy as np
    from ruamel.yaml import YAML
    torch.set_num_threads(1)
    run = Path('results/code_probe_kl_granularity1126') / str(cli.trial_id)
    if run.exists():
        raise FileExistsError('Preserve earlier probes')
    original = Path('train_ppo_s2r.py').read_bytes()
    if hashlib.sha256(original).hexdigest() != BASE_SHA:
        raise ValueError('Review injection after production code changes')
    source_text = original.decode()
    marker = '        with torch.no_grad():\n            old_dist = actor(S)'
    assert source_text.count(marker) == 1
    begin = source_text.index(marker)
    end = source_text.index('        clip_frac = clip_frac / max(1, n_mb)', begin)
    original_block = textwrap.dedent(source_text[begin:end])
    staged_block = None
    if staged_path:
        staged_text = staged_path.read_text()
        stage_begin = staged_text.index(marker)
        stage_end = staged_text.index('        clip_frac = clip_frac / max(1, n_mb)', stage_begin)
        staged_block = textwrap.dedent(staged_text[stage_begin:stage_end])
    assert original_block.count('        opt_actor.step()') == 1
    instrumented = original_block.replace('        opt_actor.step()', '''        opt_actor.step()
        with torch.no_grad():
            step_kl = torch.distributions.kl_divergence(old_dist, actor(S)).sum(-1).mean().item()
        _trace.append(dict(epoch=ep, minibatch=n_mb, kl=step_kl))
        if _mode == 'minibatch_stop' and step_kl > args.max_kl:
            break''')
    rollback_block = instrumented.replace('        d = actor(S[idx])', '''        _before_actor = copy.deepcopy((actor.state_dict(), opt_actor.state_dict(), retention.state_dict()))
        d = actor(S[idx])''').replace("        if _mode == 'minibatch_stop' and step_kl > args.max_kl:", '''        if step_kl > args.max_kl:
            actor.load_state_dict(_before_actor[0])
            opt_actor.load_state_dict(_before_actor[1])
            retention.load_state_dict(_before_actor[2])
            opt_actor.zero_grad()
            _rejected = True''').replace('        if kl > args.max_kl:', '        if _rejected or kl > args.max_kl:')
    stage = run / 'isolated_module'
    stage.mkdir(parents=True)
    (stage / 'train_ppo_s2r.py').write_text(source_text.replace(marker,
        '        _kl_probe(locals())\n        raise _ProbeDone()\n' + marker))
    os.environ['PPO_KL_STAGE'] = str(stage.resolve())
    sys.path.insert(0, str(stage.resolve()))
    import train_ppo_s2r as PPO
    assert Path(PPO.__file__).resolve() == (stage/'train_ppo_s2r.py').resolve()
    src_snapshot = Path('results/ppo_s2r/28/checkpoint/diagnostic_before_kl_spike_u35.pt')
    payload = src_snapshot.read_bytes()
    (run / 'checkpoint').mkdir()
    snapshot = run / 'checkpoint/resume.pt'
    snapshot.write_bytes(payload)
    source = torch.load(snapshot, map_location='cpu', weights_only=False)
    assert source['update'] == 35
    yaml = YAML()
    cfg = yaml.load(Path('configs/task/sim_override_1126_ppo.yaml'))
    cfg['training_mix'] = {k: .2 for k in ('hall', 'short', 'regular', 'open', 'real')}
    override = run / 'probe_override.yaml'
    yaml.dump(cfg, override)
    launch = json.loads(Path('results/preflight_1126/ppo28_launch_manifest.json').read_text())['argv']
    assert launch[1:3] == ['-u', 'train_ppo_s2r.py']
    argv = launch[3:]
    for flag, value in {
            '--name': 'code_probe_kl_granularity1126', '--algo_idx': str(cli.trial_id),
            '--workers': '5', '--n_steps': str(cli.steps), '--sim_override': str(override),
            '--total_steps': str(cli.steps * (source['args']['total_steps']//source['args']['n_steps']))}.items():
        argv[argv.index(flag)+1] = value
    sys.argv = [str(stage/'train_ppo_s2r.py')] + argv + ['--resume']

    def same(a, b):
        if torch.is_tensor(a):
            return torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b

    def probe(frame):
        try:
            assert frame['update'] == 36 and not frame['warm']
            assert same(frame['actor'].state_dict(), source['actor'])
            names = ('actor', 'teacher', 'critic', 'ccritic', 'opt_actor', 'opt_critic', 'opt_ccritic',
                     'actor_params', 'retention', 'args', 'T', 'N', 'S', 'C', 'A', 'LP', 'ADV', 'R',
                     'RC', 'BCM', 'bc_dim', 'bc_w', 'warm')
            initial = {k: frame[k] for k in names}
            rng = torch.get_rng_state().clone()
            from utils.policy_transfer import columns
            feature_names = columns(frame['args'])
            raw_obs = frame['B']['obs'].reshape(len(frame['S']), -1)
            states, results = {}, {}
            modes = ['unmodified', 'epoch_instrumented', 'minibatch_stop', 'minibatch_rollback']
            if staged_block:
                modes += ['staged_epoch', 'staged_rollback']
            for mode in modes:
                # Copy the connected object graph together, preserving the
                # optimizer-to-parameter and actor_params aliases.
                scope = copy.deepcopy(initial)
                scope.update(torch=torch, np=np, copy=copy, _mode=mode, _trace=[], _rejected=False)
                scope['args'].kl_guard = 'minibatch_rollback' if mode == 'staged_rollback' else 'epoch'
                assert {id(p) for g in scope['opt_actor'].param_groups for p in g['params']} == {id(p) for p in scope['actor_params']}
                torch.set_rng_state(rng)
                block = (original_block if mode == 'unmodified' else
                         rollback_block if mode == 'minibatch_rollback' else instrumented)
                if mode.startswith('staged_'):
                    block = staged_block
                exec(block, scope)
                state = {k: scope[k].state_dict() for k in
                         ('actor', 'critic', 'ccritic', 'teacher', 'opt_actor', 'opt_critic', 'opt_ccritic', 'retention')}
                state['torch_rng'] = torch.get_rng_state().clone()
                for name in ('actor', 'critic', 'ccritic', 'teacher'):
                    assert all(torch.isfinite(v).all().item() for v in state[name].values())
                states[mode] = state
                with torch.no_grad():
                    after, before = scope['actor'](scope['S']), frame['actor'](frame['S'])
                    change = after.loc - before.loc
                    per_state_kl = torch.distributions.kl_divergence(before, after).sum(-1)
                    lost = frame['BCM'] == 0
                strata = {}
                for label, mask in (('all', torch.ones_like(lost, dtype=torch.bool)), ('valid', ~lost), ('lost', lost)):
                    values = per_state_kl[mask].numpy()
                    if len(values):
                        strata[label] = dict(rows=len(values), mean=float(values.mean()),
                            p95=float(np.percentile(values,95)), p99=float(np.percentile(values,99)), maximum=float(values.max()))
                largest = []
                for index in torch.argsort(per_state_kl, descending=True)[:16].tolist():
                    feature_indices = torch.argsort(frame['S'][index].abs(), descending=True)[:6].tolist()
                    largest.append(dict(row=index, worker=index % frame['N'],
                        family=frame['families'][index % frame['N']], rollout_step=index // frame['N'],
                        kl=float(per_state_kl[index]), before_mean=before.loc[index].tolist(),
                        after_mean=after.loc[index].tolist(),
                        clipped_action_delta=(after.loc[index].clamp(-1,1)-before.loc[index].clamp(-1,1)).tolist(),
                        raw_speed=float(raw_obs[index,feature_names.index('speed')]),
                        preview_valid=bool(raw_obs[index,-1]>.5),
                        largest_normalized_features=[dict(name=feature_names[j],
                            normalized=float(frame['S'][index,j]), raw=float(raw_obs[index,j])) for j in feature_indices]))
                rejected = scope.get('guard_rejected', scope['_rejected'])
                results[mode] = dict(final_kl=float(scope['kl']), actor_minibatches=int(scope['n_mb']),
                    rejected_actor_minibatches=int(rejected),
                    accepted_actor_minibatches=int(scope['n_mb'])-int(rejected),
                    retention_minibatches=int(scope['retention_steps']), per_minibatch_kl=scope['_trace'],
                    per_state_kl=strata,
                    largest_state_changes=largest,
                    mean_abs_action_delta=change.abs().mean(0).tolist(),
                    lost_mean_abs_action_delta=change[lost].abs().mean(0).tolist() if lost.any() else None)
            assert same(states['unmodified'], states['epoch_instrumented']), 'Instrumentation changed optimizer result'
            if staged_block:
                assert same(states['unmodified'], states['staged_epoch']), 'Staged default changes optimizer behavior'
                assert same(states['minibatch_rollback'], states['staged_rollback']), 'Staged rollback differs from the verified prototype'
                assert results['staged_rollback']['retention_minibatches'] == results['staged_rollback']['accepted_actor_minibatches']
            assert same(frame['actor'].state_dict(), source['actor']), 'Source actor mutated'
            assert snapshot.read_bytes() == payload and src_snapshot.read_bytes() == payload
            assert results['minibatch_rollback']['final_kl'] <= frame['args'].max_kl
            report = dict(status='complete_isolated_probe', source_snapshot=str(src_snapshot),
                source_snapshot_sha256=hashlib.sha256(payload).hexdigest(), production_sha256=BASE_SHA,
                rollout_steps=cli.steps, workers=5, family_mix=cfg['training_mix'],
                missing_preview_rows=int((frame['BCM'] == 0).sum()), update=36,
                max_kl=float(frame['args'].max_kl), bc_weight=float(frame['bc_w']),
                observation_matrix_sha256=hashlib.sha256(frame['S'].numpy().tobytes()).hexdigest(),
                instrumentation_bit_exact=True, live_source_unchanged=True, results=results,
                staged_implementation_sha256=hashlib.sha256(staged_path.read_bytes()).hexdigest() if staged_path else None,
                staged_default_and_rollback_bit_exact=bool(staged_block),
                qualification=('Actual optimizer block on private copies of one identical fresh training-only '
                    'rollout. Five balanced-family workers differ from production20-worker quotas. '
                    'Environment state is not serialized, so this is not an exact replay of update39. '
                    'Minibatch stopping checks after an optimizer step and can still overshoot. Rollback '
                    'restores the rejected actor/optimizer/cache-RNG step; that minibatch critic update '
                    'is retained. A bound on rollout-mean KL is not a per-state or unseen-state bound. '
                    'None of these probes proves better completion, pace, or localization. '
                    'No checkpoint from this probe is promoted or used to initialize training.'))
            (run/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k!='results'}), flush=True)
            for mode, row in results.items():
                print(mode, 'KL', row['final_kl'], 'actor minibatches', row['actor_minibatches'], flush=True)
        finally:
            frame['venv'].close()

    PPO._kl_probe = probe
    PPO._ProbeDone = ProbeDone
    try:
        PPO.main()
    except ProbeDone:
        return
    raise RuntimeError('The pre-update probe hook was not reached')


if __name__ == '__main__':
    main()
