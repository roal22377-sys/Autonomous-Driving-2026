"""Select a mixed-course policy by completion first, then matched relative pace."""
import math


def choose_families(current, reference, shares):
    """Known practice geometry is a completion gate, never an unseen-course score.

    Every family must keep its completed-run count. Among such candidates, prefer
    higher weighted completion, then faster relative times on uncapped runs both
    policies finished. Episode durations cannot change a family's weight.
    """
    groups = [g for g, w in shares.items() if w > 0 and g != 'real']
    if not groups or any(g not in current or g not in reference for g in groups):
        raise ValueError('family selection needs every active validation family')
    total = sum(float(shares[g]) for g in groups)
    rates, old_rates, gains, regressions = {}, {}, {}, []
    for group in groups + (['real'] if 'real' in reference else []):
        if group not in current:
            raise ValueError('missing current family: ' + group)
        now, old = current[group], reference[group]
        rate, base = float(now['success']), float(old['success'])
        if not (math.isfinite(rate) and math.isfinite(base) and 0 <= rate <= 1 and 0 <= base <= 1):
            raise ValueError('invalid family success: ' + group)
        rates[group], old_rates[group] = rate, base
        if rate < base - 1e-9:
            regressions.append(group)
        nr, rr = now.get('res', []), old.get('res', [])
        if bool(nr) != bool(rr):
            raise ValueError('missing paired runs: ' + group)
        if nr:
            nk, rk = [r[0] for r in nr], [r[0] for r in rr]
            if len(nk) != len(set(nk)) or len(rk) != len(set(rk)) or set(nk) != set(rk):
                raise ValueError('different or duplicate evaluation cases: ' + group)
            for rows, claimed in ((nr, rate), (rr, base)):
                if abs(sum(bool(r[1]) for r in rows)/len(rows) - claimed) > 1e-9:
                    raise ValueError('completion count disagrees with success: ' + group)
            # All caps remain completion gates; lap-time optimization uses cap 8.
            good = lambda rows: {r[0]: float(r[2]) for r in rows if r[1] and r[0].split('@')[-1].split('/')[0] == '8'}
            a, b = good(nr), good(rr)
            common = set(a) & set(b)
            if common:
                if any(not (0 < a[k] < math.inf and 0 < b[k] < math.inf) for k in common):
                    raise ValueError('invalid completed-run time: ' + group)
                gains[group] = sum((b[k]-a[k])/b[k] for k in common)/len(common)
    score = sum(float(shares[g])*rates[g] for g in groups)/total
    old_score = sum(float(shares[g])*old_rates[g] for g in groups)/total
    pace_groups = [g for g in groups if g in gains]
    pace_weight = sum(float(shares[g]) for g in pace_groups)
    pace = sum(float(shares[g])*gains[g] for g in pace_groups)/pace_weight if pace_weight else float('nan')
    complete_better = score > old_score + 1e-9
    faster = math.isfinite(pace) and pace > 1e-6
    known_better = 'real' in rates and rates['real'] > old_rates['real'] + 1e-9
    selected = not regressions and (complete_better or (abs(score-old_score) <= 1e-9 and (known_better or faster)))
    reason = ('completion regressed: '+','.join(regressions) if regressions else
              'higher completion' if complete_better else 'higher practice completion (known geometry)' if known_better
              else 'faster matched pace' if faster else 'no improvement')
    return dict(selected=selected, reason=reason, completion=score, reference_completion=old_score,
                relative_pace_gain=pace, family_pace_gains=gains)
