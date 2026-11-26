#!/usr/bin/env python3
"""Review PPO27 against both its starting policy and the faster PPO25u10.

These are fixed-seed diagnostic comparisons, not deployment qualification.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path.insert(0, str(SRC))
from utils.model_selection import choose_families
from review_ppo_line_ablation_1126 import COHORTS, completed_row, matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update', type=int, required=True)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    base = SRC / 'results/ppo_s2r'
    old_args, args = [json.loads((base / str(n) / 'backup/ppo_args.json').read_text())
                      for n in (25, 27)]
    differences = {k: [old_args.get(k), args.get(k)]
                   for k in old_args.keys() | args.keys() if old_args.get(k) != args.get(k)}
    if set(differences) != {'save_dir', 'baseline_from', 'algo_idx', 'init'}:
        raise ValueError('Unexpected robust-start experiment differences')
    if args['line_cost'] != 10 or args['init'] != 'preflight_1126/ppo25_u70_actor_only.pt':
        raise ValueError('Unexpected starting model or trajectory constraint')
    if (base / '25/backup/dynamic.yaml').read_bytes() != (base / '27/backup/dynamic.yaml').read_bytes():
        raise ValueError('Different dynamics configurations')
    now = completed_row(base / '27/train_log.pkl', cli.update)
    source_row = completed_row(base / '25/train_log.pkl', 70)
    reference_path = SRC / 'results/analysis1126/ppo25_u70_full_family_reference.json'
    source = json.loads(reference_path.read_text())
    source_checkpoint = base / '25/checkpoint/model_3080000.pt'
    if hashlib.sha256(source_checkpoint.read_bytes()).hexdigest() != source['checkpoint_sha256']:
        raise ValueError('Starting source identity changed')
    # Evaluation deliberately uses base7 independently of the training seed.
    if now['evaluation_seed_base'] != 7 or args['seed'] != old_args['seed']:
        raise ValueError('Different evaluation seed schedules')
    snapshot = torch.load(base / '25/checkpoint/resume_before_line_ablation_u95.pt',
                          map_location='cpu', weights_only=False)
    if snapshot['best']['name'] != 'update 10':
        raise ValueError('Pace-reference identity changed')
    references = {'source_u70': source['families'], 'pace_u10': snapshot['best']['families']}
    current, cohorts = {}, {}
    for group, (rate_key, rows_key, count) in COHORTS.items():
        rows = now[rows_key]
        successes = sum(bool(r[1]) for r in rows)
        source_count = round(float(source_row[rate_key]) * count)
        if (len(rows) != count or abs(successes / count - now[rate_key]) > 1e-9
                or abs(source_count / count - source_row[rate_key]) > 1e-9):
            raise ValueError('Cohort/count mismatch: ' + group)
        cohorts[group] = dict(candidate_completed=successes, total=count,
                              source_completed=source_count)
        for label, families in references.items():
            if group in families:
                cohorts[group]['vs_' + label] = matched(rows, families[group]['res'])
                current[group] = dict(success=now[rate_key], res=rows)
                if label == 'source_u70' and abs(families[group]['success'] - source_row[rate_key]) > 1e-9:
                    raise ValueError('Restored source reference disagrees with training log')
        if group not in source['families']:
            cohorts[group]['vs_source_u70'] = (matched(rows, source_row[rows_key])
                                               if rows_key in source_row else None)
    checkpoint = base / ('27/checkpoint/model_%s.pt' % now['step'])
    weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not all(torch.isfinite(v).all().item() for key in ('actor', 'critic', 'ccritic')
               for v in weights[key].values()):
        raise ValueError('Nonfinite policy weights')
    comparisons = {label: choose_families(current, families, args['training_mix'])
                   for label, families in references.items()}
    stress_regressions = [g for g in ('worst', 'lost_short', 'lost_general')
                          if cohorts[g]['candidate_completed'] < cohorts[g]['source_completed']]
    report = dict(run='ppo_s2r/27', update=cli.update, steps=now['step'],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        configuration_differences=differences, cohorts=cohorts,
        original_il_selection=now['family_selection'], selected_within_own_run=now['saved'],
        comparisons=comparisons,
        source_stress_and_lost_guard=dict(passed=not stress_regressions,
                                         completion_regressions=stress_regressions),
        eligible_for_expanded_validation=(not stress_regressions and
            all(result['selected'] for result in comparisons.values())),
        expanded_validation_qualified=False,
        qualification=('Same case ordering and fixed per-track seed schedule; source u70 restored '
                       'open/short records reproduce its original counts. Missing source per-case '
                       'stress/lost records remain null. Pace uses jointly completed cap8 runs only. '
                       'Stress/lost counts additionally must not regress from sourceu70 before '
                       'expanded candidate validation. This report alone cannot qualify new-layout, '
                       'correct relocalization, or real-car performance. Neither reference is an overall winner.'))
    out = Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve prior review: ' + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['comparisons'], indent=2))
    for group, row in cohorts.items():
        print(group, '%s/%s vs source %s/%s' % (row['candidate_completed'], row['total'],
              row['source_completed'], row['total']))


if __name__ == '__main__':
    main()
