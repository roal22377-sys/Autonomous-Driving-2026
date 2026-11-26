#!/usr/bin/env python3
"""Review completed PPO26 evaluations against same-update PPO25 and its u10 best.

Old rejected PPO25 checkpoints lack some per-case records. Report their counts
without inventing paired pace or failure identities for those cohorts.
"""
import argparse
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path.insert(0, str(SRC))
from utils.model_selection import choose_families

COHORTS = {
    'regular': ('eval_success', 'res', 176),
    'open': ('open', 'open_res', 88),
    'short': ('short', 'short_res', 32),
    'hall': ('hall', 'hall_res', 102),
    'real': ('real_success', 'real_res', 8),
    'worst': ('worst', 'worst_res', 132),
    'lost_short': ('lost_short', 'lost_short_res', 32),
    'lost_general': ('lost_eval', 'lost_eval_res', 44),
}


def matched(now, old):
    if [r[0] for r in now] != [r[0] for r in old]:
        raise ValueError('Different case ordering; seed correspondence not established')
    if len(set(r[0] for r in now)) != len(now):
        raise ValueError('Duplicate case identifiers')
    common = [(a, b) for a, b in zip(now, old) if a[1] and b[1]
              and a[0].split('@')[-1].split('/')[0] == '8']
    for a, b in common:
        if not all(math.isfinite(float(r[2])) and r[2] > 0 for r in (a, b)):
            raise ValueError('Invalid completed-run duration')
    return dict(
        common_completed_uncapped=len(common),
        relative_pace_gain=(sum((b[2] - a[2]) / b[2] for a, b in common)
                            / len(common) if common else None),
        new_failures=[a[0] for a, b in zip(now, old) if not a[1] and b[1]],
        recoveries=[a[0] for a, b in zip(now, old) if a[1] and not b[1]])


def completed_row(path, update):
    with path.open('rb') as f:
        rows = pickle.load(f)
    found = [r for r in rows if r.get('update') == update and 'eval_success' in r]
    if len(found) != 1:
        raise ValueError('Complete evaluation not available: %s update %s' % (path, update))
    return found[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--update', type=int, required=True)
    p.add_argument('--output', required=True)
    cli = p.parse_args()
    import torch
    torch.set_num_threads(1)
    base = SRC / 'results/ppo_s2r'
    args = [json.loads((base / str(n) / 'backup/ppo_args.json').read_text())
            for n in (25, 26)]
    differences = {k: [args[0].get(k), args[1].get(k)]
                   for k in args[0].keys() | args[1].keys()
                   if args[0].get(k) != args[1].get(k)}
    if set(differences) != {'save_dir', 'baseline_from', 'algo_idx', 'line_cost'}:
        raise ValueError('Unexpected experiment configuration differences')
    old, now = [completed_row(base / str(n) / 'train_log.pkl', cli.update)
                for n in (25, 26)]
    snapshot = base / '25/checkpoint/resume_before_line_ablation_u95.pt'
    control = torch.load(snapshot, map_location='cpu', weights_only=False)
    if control['best']['name'] != 'update 10':
        raise ValueError('Reference snapshot best changed')
    reference = control['best']['families']
    current, cohorts = {}, {}
    for group, (rate_key, rows_key, count) in COHORTS.items():
        rows = now[rows_key]
        successes = sum(bool(r[1]) for r in rows)
        old_successes = round(float(old[rate_key]) * count)
        if (len(rows) != count or abs(successes/count - now[rate_key]) > 1e-9
                or abs(old_successes/count - old[rate_key]) > 1e-9):
            raise ValueError('Cohort/count mismatch: ' + group)
        cohorts[group] = dict(candidate_completed=successes, total=count,
                              same_update_control_completed=old_successes,
                              same_update_paired=(matched(rows, old[rows_key])
                                  if rows_key in old else None))
        if group in reference:
            cohorts[group]['vs_pace_reference_u10'] = matched(rows, reference[group]['res'])
            current[group] = dict(success=now[rate_key], res=rows)
    checkpoint = base / ('26/checkpoint/model_%s.pt' % now['step'])
    weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not all(torch.isfinite(t).all().item() for name in ('actor', 'critic', 'ccritic')
               for t in weights[name].values()):
        raise ValueError('Nonfinite checkpoint weights')
    report = dict(run='ppo_s2r/26', update=cli.update, steps=now['step'],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        configuration_differences=differences, cohorts=cohorts,
        own_run_selection=now['family_selection'], selected_within_own_run=now['saved'],
        comparison_vs_pace_reference_u10=choose_families(current, reference, args[1]['training_mix']),
        qualification=('Fixed evaluation seed schedule. Missing same-update per-case records '
                       'are null, never inferred. u10 is a pace reference, not an overall '
                       'winner on extended-layout/localization validation. No real-car qualification.'))
    out = Path(cli.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cohorts'}, indent=2))
    for group, row in cohorts.items():
        print(group, '%s/%s vs same-update %s/%s' % (row['candidate_completed'], row['total'],
              row['same_update_control_completed'], row['total']))


if __name__ == '__main__':
    main()
