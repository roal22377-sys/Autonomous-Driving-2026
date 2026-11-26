#!/usr/bin/env python3
"""Review the declared retention-only PPO28 pilot against control27 and sources.

This reads completed fixed-cohort evaluations; it does not select exports or
modify the running learner. Expanded-layout and real-car qualification remain
separate even when all fixed-cohort gates pass.
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update', required=True, type=int)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    out = Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve previous review: ' + str(out))
    import torch
    torch.set_num_threads(1)
    base = SRC / 'results/ppo_s2r'
    control_args, args = [json.loads((base / str(n) / 'backup/ppo_args.json').read_text())
                          for n in (27, 28)]
    differences = {k: [control_args.get(k), args.get(k)]
                   for k in control_args.keys() | args.keys()
                   if control_args.get(k) != args.get(k)}
    expected = {'save_dir', 'algo_idx', 'retention_file', 'retention_manifest',
                'retention_weight', 'retention_share'}
    if set(differences) != expected:
        raise ValueError('Unexpected retention-only experiment differences: ' + str(differences))
    if (args['line_cost'] != 10 or args['retention_weight'] != 5
            or args['retention_share'] != .125
            or args['init'] != 'preflight_1126/ppo25_u70_actor_only.pt'):
        raise ValueError('Parameters differ from the declared pilot')
    for name in ('dynamic.yaml', 'f1tenth.yaml'):
        if (base / '27/backup' / name).read_bytes() != (base / '28/backup' / name).read_bytes():
            raise ValueError('Different task/dynamics: ' + name)
    manifest_path = SRC / args['retention_manifest']
    manifest = json.loads(manifest_path.read_text())
    if sha(SRC / args['retention_file']) != manifest['cache_sha256']:
        raise ValueError('Retention cache identity changed')
    launch = json.loads((SRC / 'results/preflight_1126/ppo28_launch_manifest.json').read_text())
    if (launch['retention_manifest_sha256'] != sha(manifest_path)
            or launch['retention_cache_sha256'] != manifest['cache_sha256']):
        raise ValueError('Retention lineage differs from the launch')
    now, control = [completed_row(base / str(n) / 'train_log.pkl', cli.update)
                    for n in (28, 27)]
    source_row = completed_row(base / '25/train_log.pkl', 70)
    source_path = SRC / 'results/analysis1126/ppo25_u70_full_family_reference.json'
    source = json.loads(source_path.read_text())
    if sha(base / '25/checkpoint/model_3080000.pt') != source['checkpoint_sha256']:
        raise ValueError('Source actor identity changed')
    if now['evaluation_seed_base'] != 7 or control['evaluation_seed_base'] != 7:
        raise ValueError('Different evaluation seed schedule')
    snapshot = torch.load(base / '25/checkpoint/resume_before_line_ablation_u95.pt',
                          map_location='cpu', weights_only=False)
    if snapshot['best']['name'] != 'update 10':
        raise ValueError('Pace reference identity changed')
    references = dict(source_u70=source['families'], pace_u10=snapshot['best']['families'])
    current, control_families, cohorts = {}, {}, {}
    for group, (rate_key, rows_key, count) in COHORTS.items():
        rows, old = now[rows_key], control[rows_key]
        successes, old_successes = [sum(bool(r[1]) for r in rr) for rr in (rows, old)]
        source_successes = round(float(source_row[rate_key]) * count)
        if (len(rows) != count or len(old) != count
                or abs(successes / count - now[rate_key]) > 1e-9
                or abs(old_successes / count - control[rate_key]) > 1e-9
                or abs(source_successes / count - source_row[rate_key]) > 1e-9):
            raise ValueError('Cohort/count mismatch: ' + group)
        row = dict(candidate_completed=successes, total=count,
                   source_completed=source_successes, control_completed=old_successes,
                   vs_same_update_control=matched(rows, old))
        for label, families in references.items():
            if group in families:
                row['vs_' + label] = matched(rows, families[group]['res'])
                current[group] = dict(success=now[rate_key], res=rows)
                control_families[group] = dict(success=control[rate_key], res=old)
        if group not in source['families']:
            row['vs_source_u70'] = (matched(rows, source_row[rows_key])
                                    if rows_key in source_row else None)
        cohorts[group] = row
    references['same_update_control'] = control_families
    comparisons = {label: choose_families(current, families, args['training_mix'])
                   for label, families in references.items()}
    regressions = [g for g in ('worst', 'lost_short', 'lost_general')
                   if cohorts[g]['candidate_completed'] < cohorts[g]['source_completed']]
    ckpath = base / ('28/checkpoint/model_%s.pt' % now['step'])
    weights = torch.load(ckpath, map_location='cpu', weights_only=False)
    if not all(torch.isfinite(v).all().item() for name in ('actor', 'critic', 'ccritic')
               for v in weights[name].values()):
        raise ValueError('Nonfinite candidate weights')
    if now['retention_weight'] != 5 or not now['retention_minibatches'] > 0:
        raise ValueError('Expected active retention at the reviewed actor update')
    report = dict(run='ppo_s2r/28', update=cli.update, steps=now['step'],
        checkpoint_sha256=sha(ckpath), source_reference_sha256=sha(source_path),
        configuration_differences=differences, cohorts=cohorts, comparisons=comparisons,
        retention=dict(loss=now['retention_loss'], minibatches=now['retention_minibatches'],
                       weight=now['retention_weight'], cache_sha256=manifest['cache_sha256'],
                       manifest_sha256=sha(manifest_path)),
        source_stress_and_lost_guard=dict(passed=not regressions, completion_regressions=regressions),
        eligible_for_expanded_validation=(not regressions and all(
            comparisons[label]['selected'] for label in ('source_u70', 'pace_u10'))),
        original_il_selection=now['family_selection'], selected_within_own_run=now['saved'],
        expanded_validation_qualified=False,
        qualification=('Matched case order and fixed seeds; jointly completed cap8 pace only. '
                       'Missing source stress/lost per-case records remain null. Same-update control '
                       'measures this seed-specific experiment, not statistical significance. '
                       'Passing fixed cohorts does not qualify unseen layouts, correct localization '
                       'recovery, or real-car performance. No checkpoint is promoted by this tool.'))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(comparisons, indent=2))
    for group, row in cohorts.items():
        print(group, '%s/%s; control%s, source%s' % (row['candidate_completed'], row['total'],
              row['control_completed'], row['source_completed']))


if __name__ == '__main__':
    main()
