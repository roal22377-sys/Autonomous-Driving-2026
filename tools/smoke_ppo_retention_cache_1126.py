#!/usr/bin/env python3
"""Check an optional PPO retention cache without updating any policy weights."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]


def main():
    os.chdir(SRC)
    import torch
    import train_ppo_s2r as PPO
    from utils.vectorize import RunningMeanStd
    from utils.ppo_retention import RetentionCache
    torch.set_num_threads(1)
    args = PPO.IL.get_args(['--task_config', 'results/ppo_s2r/27/backup/f1tenth.yaml'])
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load('results/imitation/1607/sim2real', required=True)
    tracks = PPO.training_tracks(True, short=True)
    path = 'results/ppo_retention_candidate_1126/replay_cache.npz'
    manifest = 'results/ppo_retention_candidate_1126/preparation.json'
    torch.manual_seed(2731)
    rng = torch.get_rng_state().clone()
    cache = RetentionCache(args, rms, path, manifest, tracks)
    assert torch.equal(rng, torch.get_rng_state()), 'Construction perturbed learner RNG'
    cache.sample()
    state = cache.state_dict()
    expected = cache.sample()
    restored = RetentionCache(args, rms, path, manifest, tracks)
    restored.load_state_dict(state)
    actual = restored.sample()
    assert all(torch.equal(x, y) for x, y in zip(expected, actual))
    assert torch.equal(rng, torch.get_rng_state()), 'Sampling perturbed learner RNG'
    changed = copy.deepcopy(state)
    changed['metadata']['cache_sha256'] = 'wrong-source'
    before = restored.generator.get_state().clone()
    try:
        restored.load_state_dict(changed)
        raise AssertionError('Changed source accepted on resume')
    except ValueError:
        pass
    assert torch.equal(before, restored.generator.get_state())
    first = json.loads(Path(manifest).read_text())['tracks'][0]['track']
    restricted = [p for p in tracks if Path(p).resolve() != Path(first).resolve()]
    try:
        RetentionCache(args, rms, path, manifest, restricted)
        raise AssertionError('Non-training trajectory accepted')
    except ValueError:
        pass
    changed_args = copy.deepcopy(args)
    changed_args.steer_cmd_gain_left *= 1.01
    try:
        RetentionCache(changed_args, rms, path, manifest, tracks)
        raise AssertionError('Changed steering command units accepted')
    except ValueError:
        pass
    ck = torch.load('results/preflight_1126/ppo25_u70_actor_only.pt', map_location='cpu', weights_only=False)
    actor = PPO.IL.actor_from_state(ck['actor'])
    actor.eval()
    weights = {k: v.clone() for k, v in actor.state_dict().items()}
    rng = torch.get_rng_state().clone()
    loss, metrics = cache.loss(actor)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g).item() > 0 for g in grads)
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in weights.items())
    assert torch.equal(rng, torch.get_rng_state()), 'Loss perturbed learner RNG'
    report = dict(status='passed', cache=cache.metadata, training_tracks=len(tracks),
        learner_rng_unchanged=True, exact_sampler_resume=True, changed_source_rejected=True,
        nontraining_trajectory_rejected=True, policy_weights_unchanged=True,
        changed_steering_command_gain_rejected=True,
        finite_nonzero_actor_gradient=True, auxiliary_loss=float(loss), component_mse=metrics,
        helper_sha256=hashlib.sha256(Path('utils/ppo_retention.py').read_bytes()).hexdigest(),
        qualification='No optimizer step and no rollout performance test. Optional helper is not wired into PPO27. Cache batch/share and speed weight used here are smoke settings, not adopted learner hyperparameters.')
    Path('results/preflight_1126/ppo_retention_cache_smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
