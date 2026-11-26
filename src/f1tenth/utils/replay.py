"""Frozen-policy rehearsal, kept separate from new environment experience."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from ruamel.yaml import YAML
from .policy_transfer import columns


def load_replay(args, rms):
    path = args.get('replay_file')
    if not path:
        return None
    share = float(args.get('replay_share', .5))
    if not 0 < share < 1:
        raise ValueError('replay_share must leave both new and replay samples in each batch')
    with np.load(path, allow_pickle=False) as z:
        names, old_names = z['columns'].tolist(), z['old_columns'].tolist()
        if names != columns(args) or len(set(old_names)) != len(old_names) or not set(old_names) <= set(names):
            raise ValueError('replay observation schema does not match this policy')
        O, Y, missing = z['O'], z['Y'], z['missing_new_features']
        teacher = str(z['teacher'])
    n = len(O)
    if (n < 2 or O.shape != (n,args.obs_dim) or Y.shape != (n,2) or missing.shape != (n,)
            or missing.dtype != np.bool_ or not np.isfinite(Y).all() or np.max(np.abs(Y)) > 1.00001):
        raise ValueError('invalid replay arrays or action labels')
    cfg = YAML().load(Path('results')/teacher/'backup/f1tenth.yaml')
    if any(abs(float(args[k])-float(cfg[k])) > 1e-8 for k in ('min_speed','max_speed','max_steer')):
        raise ValueError('replay teacher uses different action scaling')
    X = torch.empty((n,args.obs_dim),dtype=torch.float16)
    for i in range(0,n,100000):
        values = rms.normalize(O[i:i+100000].astype(np.float32))
        if not np.isfinite(values).all() or np.max(np.abs(values)) > np.finfo(np.float16).max:
            raise ValueError('replay observations are not finite after normalization')
        X[i:i+100000] = torch.as_tensor(values,dtype=torch.float16)
    # Historical records have no near-preview inputs. Zero their normalized
    # contribution rather than inventing a curvature/speed value for them.
    mask = torch.as_tensor(missing)
    for j,name in enumerate(names):
        if name not in old_names:
            X[mask,j] = 0
    T = torch.as_tensor(Y,dtype=torch.float32)
    perm = torch.randperm(n);nv = max(1,min(n-1,n//50))
    count = int(round(args.batch*share))
    if count < 1 or count >= args.batch:
        raise ValueError('batch is too small for the replay share')
    print(f'[BC replay] {n} cached states from {teacher}; {count}/{args.batch} samples per batch; '
          'new-experience quotas unchanged',flush=True)
    return SimpleNamespace(X=X,T=T,train=perm[nv:],val=perm[:nv],batch=count,teacher=teacher)
