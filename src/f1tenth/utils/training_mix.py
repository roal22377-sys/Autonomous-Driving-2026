"""Track families and collection-step budgets shared by IL and PPO.

The 2025 photo reconstruction is explicitly trained; it is excluded from the
hall validation set. Other block-3 hall courses remain validation-only.
"""
import glob
import os
import numpy as np
from .track_gen import GEN_DIR

ROOT = os.path.dirname(GEN_DIR)


def family(path):
    name = os.path.basename(path)
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if folder == 'maps_hall' or name.startswith(('Hall', 'Course2025')):
        return 'hall'
    if folder == 'maps_short':
        return 'short'
    if folder == 'maps_real' or name.startswith('Practice'):
        return 'real'
    if folder == 'maps_open_train':
        return 'open'
    return 'regular'


def hall_split():
    from .track_hall import hall_tracks
    train = set(hall_tracks(False))
    valid = set(hall_tracks(True))
    copies = {p for p in valid if os.path.basename(p).startswith('Course2025')}
    train.update(copies)
    valid.difference_update(copies)
    if train & valid:
        raise ValueError('hall train/validation overlap')
    return sorted(train), sorted(valid)


def groups_for(args, regular):
    from .track_short import short_tracks
    hall, _ = hall_split()
    real = [os.path.abspath(p) for p in args.get('training_real_tracks', [])]
    groups = dict(regular=list(regular), open=sorted(glob.glob(os.path.join(ROOT, 'maps_open_train', '*.npz'))),
                  short=short_tracks(False), hall=hall, real=real)
    shares = dict(args.get('training_mix', {}) or {})
    if shares:
        if any(float(s) < 0 for s in shares.values()) or abs(sum(shares.values()) - 1.0) > 1e-6:
            raise ValueError('training_mix must be nonnegative and sum to one')
        for group, share in shares.items():
            if group not in groups or (share > 0 and not groups[group]):
                raise ValueError('missing training family: ' + group)
        files = [os.path.abspath(p) for group, ps in groups.items() if shares.get(group, 0) > 0 for p in ps]
        if len(files) != len(set(files)):
            raise ValueError('duplicate training track paths')
        for p in files:
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
    return groups


def step_budgets(shares, total):
    keys = [k for k, v in shares.items() if v > 0]
    raw = np.asarray([shares[k] * int(total) for k in keys])
    counts = np.floor(raw).astype(int)
    for j in np.argsort(-(raw - counts))[:int(total) - int(counts.sum())]:
        counts[j] += 1
    return dict(zip(keys, map(int, counts)))


def step_weights(groups, shares):
    """Inverse-length episode weights approximate target proportions of STEPS.

    Measured PPO step shares are also logged; policy speeds/failures can change
    episode duration, so these weights are a starting estimate, not a guarantee.
    IL uses exact collection quotas instead.
    """
    weights = {}
    for group, share in shares.items():
        if share <= 0:
            continue
        ps = groups[group]
        for p in ps:
            with np.load(p) as z:
                length = max(float(z['length']), 10.0)
            weights[os.path.basename(p)] = float(share) / (len(ps) * length)
    return weights


def evaluation_sets(args):
    _, hall = hall_split()
    real = [os.path.abspath(p) for p in args.get('training_real_tracks', [])]
    return dict(hall=hall, real=real)
