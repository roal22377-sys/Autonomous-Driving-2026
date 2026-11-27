"""Deterministic rollout-level annealing of reference-line guidance only."""
import math


def validate(start, end, decay_updates):
    if not math.isfinite(start) or start < 0:
        raise ValueError('Line guidance start must be finite and nonnegative')
    if end is not None and (not math.isfinite(end) or not 0 <= end <= start):
        raise ValueError('Line guidance end must be finite and between0 andstart')
    if not isinstance(decay_updates, int) or decay_updates <= 0:
        raise ValueError('Line guidance decay requires a positive integer actor-update count')


def coefficient(start, end, decay_updates, completed_update, actor_start):
    """Coefficient for the NEXT rollout, after completed_update optimizer updates.

    Warm-up keeps the initial value. The first actor update therefore still
    uses a rollout at the initial coefficient; each actor-training update
    iteration then advances the schedule. This counts actor update iterations,
    including an iteration whose KL guard rejects some/all minibatches.
    No wall-clock or mutable schedule counter is needed across resume.
    """
    validate(start, end, decay_updates)
    if end is None or actor_start is None:
        return float(start)
    completed_actor_updates = max(0, completed_update-actor_start+1)
    fraction = min(1., completed_actor_updates/decay_updates)
    return float(start+(end-start)*fraction)


def check_resume(saved, current):
    """Legacy constant-guidance resumes are unchanged; annealing is immutable."""
    if saved.get('line_cost_end') is None and current.get('line_cost_end') is None:
        return
    keys = [('line_cost', 0.), ('line_cost_end', None), ('line_cost_decay_updates', 40)]
    if any(saved.get(k, default) != current.get(k, default) for k, default in keys):
        raise ValueError('Line guidance schedule changed during resume; start a declared new trial')
