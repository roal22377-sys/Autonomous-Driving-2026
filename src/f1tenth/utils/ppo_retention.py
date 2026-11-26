"""Optional supervised retention cache; never supplies PPO advantages or returns.

Not wired into the learner by this module. Its sampler owns its RNG so enabling
diagnostics cannot silently perturb rollout/minibatch randomness.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from ruamel.yaml import YAML

from .replay import load_replay
from .observation import get_obs_config


class RetentionCache:
    def __init__(self, args, rms, cache_path, manifest_path, training_tracks,
                 minibatch=2048, share=.125, seed=1126):
        path = Path(cache_path)
        manifest = json.loads(Path(manifest_path).read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest['cache_sha256']:
            raise ValueError('Retention cache does not match its audited manifest')
        allowed = {str(Path(p).resolve()) for p in training_tracks}
        tracks = manifest['tracks']
        if (not tracks or any(not r['source_episode_complete'] or r['family'] not in ('regular', 'open')
                              or str(Path(r['track']).resolve()) not in allowed for r in tracks)):
            raise ValueError('Retention requires successful training-only regular/open trajectories')
        with np.load(path, allow_pickle=False) as z:
            family, ids = z['family'], z['track_id']
            teacher = str(z['teacher'])
            if (len(z['O']) != manifest['rows'] or ids.shape != family.shape
                    or ids.shape != (manifest['rows'],) or not np.issubdtype(ids.dtype, np.integer)
                    or ids.min() < 0 or ids.max() >= len(tracks)):
                raise ValueError('Invalid retention provenance arrays')
            for i, row in enumerate(tracks):
                mask = ids == i
                if int(mask.sum()) != row['rows'] or not np.all(family[mask] == row['family']):
                    raise ValueError('Retention rows disagree with trajectory provenance')
        teacher_args = YAML().load(Path('results') / teacher / 'backup/f1tenth.yaml')
        def gains(cfg):
            default = float(cfg.get('steer_cmd_gain', 1.))
            return [float(cfg.get('steer_cmd_gain_' + side, default)) for side in ('left', 'right')]
        if not np.allclose(gains(args), gains(teacher_args), rtol=0, atol=1e-8):
            raise ValueError('Retention teacher uses different steering command gains')
        # load_replay validates schema, action units, finiteness and normalization.
        # Its random train/diagnostic split must not consume the learner's RNG.
        loader_args = EasyDict(dict(args, replay_file=str(path), replay_share=share, batch=minibatch))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.data = load_replay(loader_args, rms)
        self.generator = torch.Generator().manual_seed(seed + 1)
        gain = float(args.get('steer_cmd_gain', 1.))
        semantics = dict(observation=get_obs_config(args), min_speed=float(args.min_speed),
            max_speed=float(args.max_speed), max_steer=float(args.max_steer),
            steering_gains=[float(args.get('steer_cmd_gain_' + side, gain)) for side in ('left', 'right')])
        encoded = json.dumps(semantics, sort_keys=True, default=lambda x: x.tolist()).encode()
        self.metadata = dict(cache_sha256=digest,
            policy_semantics_sha256=hashlib.sha256(encoded).hexdigest(),
            manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            normalizer_sha256=hashlib.sha256(np.asarray(rms.mean).tobytes() + np.asarray(rms.var).tobytes()).hexdigest(),
            split_seed=int(seed), minibatch=int(minibatch), share=float(share),
            batch=int(self.data.batch), rows=int(manifest['rows']))

    def sample(self):
        choice = torch.randint(len(self.data.train), (self.data.batch,), generator=self.generator)
        indices = self.data.train[choice]
        return self.data.X[indices].float(), self.data.T[indices]

    def loss(self, actor, speed_weight=.25):
        if not np.isfinite(speed_weight) or speed_weight < 0:
            raise ValueError('Invalid retention speed weight')
        inputs, targets = self.sample()
        device = next(actor.parameters()).device
        prediction = actor(inputs.to(device)).loc
        error = prediction - targets.to(device)
        weights = error.new_tensor([1., speed_weight])
        value = (error.square() * weights).mean()
        return value, dict(steer_mse=float(error[:, 0].detach().square().mean()),
                           speed_mse=float(error[:, 1].detach().square().mean()))

    def state_dict(self):
        return dict(metadata=dict(self.metadata), rng=self.generator.get_state().clone())

    def load_state_dict(self, state):
        if state['metadata'] != self.metadata:
            raise ValueError('Retention source, normalization or sampler configuration changed on resume')
        self.generator.set_state(state['rng'])
