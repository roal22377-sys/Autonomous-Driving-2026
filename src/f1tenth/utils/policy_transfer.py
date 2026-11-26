"""Expand a policy with new preview features without discarding learned inputs."""
import os
import numpy as np
import torch
from ruamel.yaml import YAML
from easydict import EasyDict
from .observation import get_obs_config
from .vectorize import RunningMeanStd


def columns(args):
    cfg = get_obs_config(EasyDict(args))
    names = ['lidar/%d' % i for i in range(cfg['lidar']['num_beams'])]
    if cfg['speed']['enabled']:
        names += ['speed']
    names += ['history/%d/%s' % (i, key) for i in range(cfg['history']['length']) for key in ('steer', 'speed')]
    if cfg['speed_limit']['enabled']:
        names += ['speed_cap']
    pv = cfg['preview']
    if pv['enabled']:
        names += ['preview/e_y', 'preview/e_psi']
        names += ['preview/curv/%g' % x for x in pv['curv_ahead']]
        names += ['preview/speed/%g' % x for x in pv['speed_ahead']]
        names += ['preview/valid']
    if len(names) != args['obs_dim'] or len(set(names)) != len(names):
        raise ValueError('invalid observation column names')
    return names


def expand_state(old_state, old_names, new_names, old_mean, old_var, new_mean, new_var):
    """Rebase first layer onto new normalization; new feature weights start at 0."""
    if set(old_names) - set(new_names):
        raise ValueError('transfer would drop existing input features')
    state = {k: v.detach().clone() for k, v in old_state.items()}
    weight, bias = state['fc1.weight'], state['fc1.bias']
    new_w = torch.zeros((weight.shape[0], len(new_names)), dtype=weight.dtype)
    new_b = bias.clone()
    old_std, new_std = np.sqrt(np.asarray(old_var) + 1e-8), np.sqrt(np.asarray(new_var) + 1e-8)
    for j, name in enumerate(old_names):
        k = new_names.index(name)
        new_w[:, k] = weight[:, j] * float(new_std[k] / old_std[j])
        new_b += weight[:, j] * float((new_mean[k] - old_mean[j]) / old_std[j])
    state['fc1.weight'], state['fc1.bias'] = new_w, new_b
    return state


def initialize(actor, args, rms, source, observations):
    old_cfg = YAML().load(open(os.path.join(source, 'backup/f1tenth.yaml')))
    old_rms = RunningMeanStd('agent_obs', old_cfg['obs_dim'])
    old_rms.load(os.path.join(source, 'sim2real'), required=True)
    checkpoint = os.path.join(source, 'checkpoint/model.pt')
    if os.path.isfile(checkpoint):
        old_state = torch.load(checkpoint, map_location='cpu')['actor']
    else:
        old_state = torch.load(os.path.join(source, 'sim2real/actor.pt'), map_location='cpu').state_dict()
    rms.update(observations)
    before, after = columns(old_cfg), columns(args)
    state = expand_state(old_state, before, after, old_rms.mean, old_rms.var, rms.mean, rms.var)
    actor.load_state_dict(state)
    # Exact affine equivalence on shared inputs before any gradient update.
    sample = np.asarray(observations[:512], np.float32)
    idx = [after.index(n) for n in before]
    old_z = (sample[:, idx] - old_rms.mean) / np.sqrt(old_rms.var + 1e-8)
    new_z = rms.normalize(sample)
    old_y = torch.as_tensor(old_z, dtype=torch.float32) @ old_state['fc1.weight'].T + old_state['fc1.bias']
    new_y = torch.as_tensor(new_z, dtype=torch.float32) @ state['fc1.weight'].T + state['fc1.bias']
    error = float(torch.max(torch.abs(old_y - new_y)))
    if error > 2e-4:
        raise ValueError('policy transfer equivalence failed: %g' % error)
    return dict(source=source, old_dim=len(before), new_dim=len(after),
                added=[n for n in after if n not in before], first_layer_max_error=error)
