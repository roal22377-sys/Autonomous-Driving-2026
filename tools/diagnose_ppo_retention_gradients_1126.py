#!/usr/bin/env python3
"""Measure auxiliary/BC/PPO gradients on an isolated training-only rollout.

The probe restores a full snapshot, collects balanced-family observations and
stops before any optimizer update. It is not a training or performance trial.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

__file__ = str(Path(__file__).resolve())
ROOT = Path(__file__).parents[1]
SRC = ROOT / 'src/f1tenth'
extra = [os.environ['PPO_GRADIENT_STAGE']] if os.environ.get('PPO_GRADIENT_STAGE') else []
sys.path[:0] = extra + [str(SRC), str(SRC / 'installation')]
BASE_SHA = '153f80f245ff5919c2f115ff47cca2230cb688036db697862fee866f88f3609b'


class ProbeDone(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-id', type=int, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    cli = parser.parse_args()
    if cli.steps < 2048 or cli.steps % 5:
        raise ValueError('Use >=2048 steps divisible by five')
    os.chdir(SRC)
    import torch
    import numpy as np
    from ruamel.yaml import YAML
    from utils.ppo_retention import RetentionCache
    run = Path('results/code_probe_retention_gradients1126') / str(cli.trial_id)
    if run.exists():
        raise FileExistsError('Preserve prior probes')
    original = Path('train_ppo_s2r.py').read_bytes()
    if hashlib.sha256(original).hexdigest() != BASE_SHA:
        raise ValueError('Review the probe injection after learner changes')
    marker = '        with torch.no_grad():\n            old_dist = actor(S)'
    code = original.decode()
    assert code.count(marker) == 1
    code = code.replace(marker, '        _gradient_probe(locals())\n        raise _ProbeDone()\n' + marker)
    stage = run / 'isolated_module'
    stage.mkdir(parents=True)
    (stage / 'train_ppo_s2r.py').write_text(code)
    os.environ['PPO_GRADIENT_STAGE'] = str(stage.resolve())
    sys.path.insert(0, str(stage.resolve()))
    import train_ppo_s2r as PPO
    assert Path(PPO.__file__).resolve() == (stage / 'train_ppo_s2r.py').resolve()
    payload = Path('results/ppo_s2r/27/checkpoint/resume.pt').read_bytes()
    (run / 'checkpoint').mkdir()
    snapshot = run / 'checkpoint/resume.pt'
    snapshot.write_bytes(payload)
    source = torch.load(snapshot, map_location='cpu', weights_only=False)
    yaml = YAML()
    override = run / 'probe_override.yaml'
    cfg = yaml.load(Path('configs/task/sim_override_1126_ppo.yaml'))
    cfg['training_mix'] = {k: .2 for k in ('hall', 'short', 'regular', 'open', 'real')}
    yaml.dump(cfg, override)
    plan = json.loads(Path('results/preflight_1126/robust_start_line_ablation_plan.json').read_text())
    launch = plan['trials'][0]['argv']
    assert launch[1:3] == ['-u', 'train_ppo_s2r.py']
    argv = launch[3:]
    # Preserve update-based exploration/BC schedules when reducing rollout size.
    total_steps = cli.steps * (source['args']['total_steps'] // source['args']['n_steps'])
    for flag, value in {'--name': 'code_probe_retention_gradients1126', '--algo_idx': str(cli.trial_id),
                        '--total_steps': str(total_steps),
                        '--workers': '5', '--n_steps': str(cli.steps), '--sim_override': str(override)}.items():
        argv[argv.index(flag) + 1] = value
    sys.argv = [str(stage / 'train_ppo_s2r.py')] + argv + ['--resume']

    def probe(frame):
        try:
            actor, teacher, args = (frame[k] for k in ('actor', 'teacher', 'args'))
            params = frame['actor_params']
            assert all(torch.equal(v, actor.state_dict()[k]) for k, v in source['actor'].items())
            cache = RetentionCache(args, frame['rms'],
                'results/ppo_retention_candidate_1126/replay_cache.npz',
                'results/ppo_retention_candidate_1126/preparation.json',
                PPO.training_tracks(True, args.real_tracks, True), seed=1127)
            generator = torch.Generator().manual_seed(271126)
            bc_weight = float(source['log'][-1]['bc_w'])

            def grad(loss):
                values = torch.autograd.grad(loss, params, allow_unused=True)
                v = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                               for g, p in zip(values, params)])
                assert torch.isfinite(v).all()
                return v

            def norm(v):
                return float(torch.linalg.vector_norm(v))

            def cosine(a, b):
                denom = norm(a) * norm(b)
                return float(torch.dot(a, b)) / denom if denom else None

            rows = []
            for i in range(12):
                idx = torch.randperm(len(frame['S']), generator=generator)[:args.minibatch]
                inputs = frame['S'][idx]
                dist = actor(inputs)
                ratio = torch.exp(dist.log_prob(frame['A'][idx]).sum(-1) - frame['LP'][idx])
                advantage = frame['ADV'][idx]
                pg = -torch.min(ratio * advantage, ratio.clamp(1-args.clip, 1+args.clip) * advantage).mean()
                pg_grad = grad(pg)
                with torch.no_grad():
                    target = teacher(inputs).loc
                weights = torch.stack([1 + args.bc_steer_weight * target[:, 0].clamp(-1, 1).square(),
                                       torch.full_like(target[:, 0], args.bc_speed)], -1)
                mask = frame['BCM'][idx, None] if frame['BCM'] is not None else 1.
                bc = ((actor(inputs).loc-target).square() * weights * mask).mean()
                bc_grad = grad(bc)
                retention, components = cache.loss(actor, speed_weight=args.bc_speed)
                retention_grad = grad(retention)
                base_grad = pg_grad + bc_weight * bc_grad
                row = dict(batch=i, pg_loss=float(pg), bc_loss=float(bc), retention_loss=float(retention),
                    pg_gradient_norm=norm(pg_grad), bc_gradient_norm=norm(bc_grad),
                    base_gradient_norm=norm(base_grad), retention_gradient_norm=norm(retention_grad),
                    retention_vs_pg_cosine=cosine(retention_grad, pg_grad),
                    retention_vs_bc_cosine=cosine(retention_grad, bc_grad), component_mse=components,
                    weighted_ratios={str(w): w * norm(retention_grad)/max(norm(base_grad), 1e-12)
                                     for w in (.1, .5, 1., 5., 10.)})
                rows.append(row)
            assert all(torch.equal(v, actor.state_dict()[k]) for k, v in source['actor'].items())
            assert snapshot.read_bytes() == payload
            report = dict(status='passed', source_run='ppo_s2r/27', source_update=source['update'],
                source_snapshot_sha256=hashlib.sha256(payload).hexdigest(), source_bc_weight=bc_weight,
                production_source_sha256=BASE_SHA, rollout_steps=cli.steps, workers=5,
                family_mix=cfg['training_mix'], optimizer_steps=0, actor_weights_unchanged=True,
                cache=cache.metadata, rows=rows,
                median_auxiliary_to_base_gradient_ratio={str(w): float(np.median([r['weighted_ratios'][str(w)] for r in rows]))
                                                       for w in (.1, .5, 1., 5., 10.)},
                qualification=('Scale diagnostic only. Fresh reset of five balanced training-family workers; '
                    'not the production20-worker step mix, not an exact resumed environment trajectory. '
                    'PPO/BC gradients use current-policy states; retention uses separate successful old-policy states. '
                    'No optimizer step, coefficient selection, performance or causal claim.'))
            (run / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print('GRADIENT PROBE PASSED', json.dumps({k:v for k,v in report.items() if k not in ('rows','cache')}), flush=True)
        finally:
            frame['venv'].close()

    PPO._gradient_probe = probe
    PPO._ProbeDone = ProbeDone
    try:
        PPO.main()
    except ProbeDone:
        return
    raise RuntimeError('Probe did not reach the pre-update hook')


if __name__ == '__main__':
    main()
