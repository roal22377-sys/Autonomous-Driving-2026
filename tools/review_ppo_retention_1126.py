#!/usr/bin/env python3
"""Review declared retention/KL-guard pilots against their controls and sources.

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


def per_cap_comparison(candidate, reference):
    """Expose high-cap failures that an aggregate completion count can hide."""
    matched(candidate, reference)  # Validate identity, order and uniqueness.
    cap = lambda row: float(row[0].split('@')[-1].split('/')[0])
    result = {}
    for value in sorted({cap(row) for row in candidate}):
        pairs = [(a, b) for a, b in zip(candidate, reference) if cap(a) == value]
        completed = [(a, b) for a, b in pairs if a[1] and b[1]]
        if any(not (0 < float(row[2]) < float('inf'))
               for pair in completed for row in pair):
            raise ValueError('Invalid paired completed duration')
        result['%g' % value] = dict(
            total=len(pairs), candidate_completed=sum(bool(a[1]) for a, _ in pairs),
            reference_completed=sum(bool(b[1]) for _, b in pairs),
            paired_completed=len(completed),
            relative_pace_gain=(sum((b[2] - a[2]) / b[2] for a, b in completed)
                                / len(completed) if completed else None),
            new_failures=[a[0] for a, b in pairs if not a[1] and b[1]],
            recoveries=[a[0] for a, b in pairs if a[1] and not b[1]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=int, choices=[28, 29, 30], default=28)
    parser.add_argument('--update', required=True, type=int)
    parser.add_argument('--control_update', type=int,
                        help='Explicit historical checkpoint; never labelled a same-update control')
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    out = Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve previous review: ' + str(out))
    import torch
    torch.set_num_threads(1)
    base = SRC / 'results/ppo_s2r'
    candidate_run, control_run = cli.run, {28: 27, 29: 28, 30: 29}[cli.run]
    control_args, args = [json.loads((base / str(n) / 'backup/ppo_args.json').read_text())
                          for n in (control_run, candidate_run)]
    differences = {k: [control_args.get(k), args.get(k)]
                   for k in control_args.keys() | args.keys()
                   if control_args.get(k) != args.get(k)}
    expected = {'save_dir', 'algo_idx', 'retention_file', 'retention_manifest',
                'retention_weight', 'retention_share'}
    if candidate_run == 29:
        expected = {'save_dir', 'algo_idx', 'kl_guard'}
        if args['kl_guard'] != 'minibatch_rollback' or control_args.get('kl_guard', 'epoch') != 'epoch':
            raise ValueError('Different KL-guard intervention')
    if candidate_run == 30:
        expected = {'save_dir', 'algo_idx', 'line_cost_end', 'line_cost_decay_updates',
                    'anneal_imitation_with_line', 'straight_offset_cost'}
        if (args['line_cost_end'] != 0 or args['line_cost_decay_updates'] != 40
                or args['anneal_imitation_with_line'] != 1 or args['straight_offset_cost'] != 0
                or args['kl_guard'] != 'minibatch_rollback'
                or control_args.get('kl_guard') != 'minibatch_rollback'):
            raise ValueError('Different declared joint-guidance curriculum')
    if set(differences) != expected:
        raise ValueError('Unexpected single-factor experiment differences: ' + str(differences))
    if (args['line_cost'] != 10 or args['retention_weight'] != 5
            or args['retention_share'] != .125
            or args['init'] != 'preflight_1126/ppo25_u70_actor_only.pt'):
        raise ValueError('Parameters differ from the declared pilot')
    for name in ('dynamic.yaml', 'f1tenth.yaml'):
        if (base / str(control_run) / 'backup' / name).read_bytes() != (base / str(candidate_run) / 'backup' / name).read_bytes():
            raise ValueError('Different task/dynamics: ' + name)
    manifest_path = SRC / args['retention_manifest']
    manifest = json.loads(manifest_path.read_text())
    if sha(SRC / args['retention_file']) != manifest['cache_sha256']:
        raise ValueError('Retention cache identity changed')
    launch = json.loads((SRC / ('results/preflight_1126/ppo%d_launch_manifest.json' % candidate_run)).read_text())
    if (launch['retention_manifest_sha256'] != sha(manifest_path)
            or launch['retention_cache_sha256'] != manifest['cache_sha256']):
        raise ValueError('Retention lineage differs from the launch')
    control_update = cli.update if cli.control_update is None else cli.control_update
    control_label = 'same_update_control' if control_update == cli.update else 'historical_control'
    now = completed_row(base / str(candidate_run) / 'train_log.pkl', cli.update)
    control = completed_row(base / str(control_run) / 'train_log.pkl', control_update)
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
    restored_worst, restored_worst_provenance = None, None
    if candidate_run in (29, 30):
        path = SRC / 'results/analysis1126/ppo25_u70_worst_regular_reference_seed7.json'
        restored = json.loads(path.read_text())
        if (restored['checkpoint_sha256'] != source['checkpoint_sha256']
                or restored['simulation_mode'] != 'worst' or restored['localization_forced_lost']
                or restored['seeds'] != [7] or restored['diagnostic_caps'] is not None
                or restored['diagnostic_cycle_caps'] is not None or restored['regular_scope'] != 'standard'
                or restored['normalizer_file_sha256'] != launch['observation_normalizer_sha256']):
            raise ValueError('Different restored worst-case source protocol')
        for key, filename in (('config_sha256', 'f1tenth.yaml'), ('dynamics_sha256', 'dynamic.yaml')):
            if restored[key] != sha(base / str(candidate_run) / 'backup' / filename):
                raise ValueError('Different restored worst-case task/dynamics')
        cohort = restored['families']['regular']
        track_order = [Path(p).stem for p in cohort['track_names']]
        if track_order != now['evaluation_track_order']['worst_res']:
            raise ValueError('Different restored worst-case seed order')
        for entry in cohort['runs']:
            if entry['seed'] != 7 + track_order.index(entry['result'][0].split('@')[0]):
                raise ValueError('Different restored worst-case per-case seed')
        restored_worst = [r['result'] for r in cohort['runs']]
        if len(restored_worst) != 132 or sum(bool(r[1]) for r in restored_worst) != 120 or abs(source_row['worst'] - 120/132) > 1e-9:
            raise ValueError('Source worst-case history was not reproduced')
        restored_worst_provenance = dict(report=str(path.relative_to(ROOT)), sha256=sha(path))
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
                   source_completed=source_successes, control_completed=old_successes)
        row['vs_' + control_label] = matched(rows, old)
        for label, families in references.items():
            if group in families:
                row['vs_' + label] = matched(rows, families[group]['res'])
                current[group] = dict(success=now[rate_key], res=rows)
                control_families[group] = dict(success=control[rate_key], res=old)
        if group not in source['families']:
            row['vs_source_u70'] = (matched(rows, source_row[rows_key])
                                    if rows_key in source_row else None)
        if group == 'worst' and restored_worst is not None:
            row['vs_source_u70'] = matched(rows, restored_worst)
        row['per_cap_comparisons'] = {control_label: per_cap_comparison(rows, old)}
        for label, families in references.items():
            if group in families:
                row['per_cap_comparisons'][label] = per_cap_comparison(rows, families[group]['res'])
        if group == 'worst' and restored_worst is not None:
            row['per_cap_comparisons']['source_u70'] = per_cap_comparison(rows, restored_worst)
        cohorts[group] = row
    references[control_label] = control_families
    comparisons = {label: choose_families(current, families, args['training_mix'])
                   for label, families in references.items()}
    regressions = [g for g in ('worst', 'lost_short', 'lost_general')
                   if cohorts[g]['candidate_completed'] < cohorts[g]['source_completed']]
    ckpath = base / ('%d/checkpoint/model_%s.pt' % (candidate_run, now['step']))
    weights = torch.load(ckpath, map_location='cpu', weights_only=False)
    if not all(torch.isfinite(v).all().item() for name in ('actor', 'critic', 'ccritic')
               for v in weights[name].values()):
        raise ValueError('Nonfinite candidate weights')
    expected_retention_weight = 5.
    guidance_review = None
    if candidate_run == 30:
        import pickle
        import time
        from utils.line_guidance import coefficient, imitation_multiplier
        for attempt in range(10):
            try:
                with (base / '30/train_log.pkl').open('rb') as f:
                    history = pickle.load(f)
                break
            except (EOFError, pickle.UnpicklingError):
                if attempt == 9:
                    raise
                time.sleep(.1)
        starts = [r['update'] for r in history if r.get('update', 0) > 0 and not r.get('warm', True)]
        if not starts:
            raise ValueError('No actor-training iteration in candidate history')
        actor_start = min(starts)
        line = coefficient(args['line_cost'], args['line_cost_end'], args['line_cost_decay_updates'],
                           cli.update-1, actor_start)
        factor = imitation_multiplier(args['line_cost'], line, True)
        if abs(now['line_guidance_coefficient']-line) > 1e-12 or abs(now['imitation_guidance_multiplier']-factor) > 1e-12:
            raise ValueError('Runtime guidance does not match rollout schedule')
        expected_retention_weight = args['retention_weight']*factor
        guidance_review = dict(actor_start=actor_start, line_coefficient=line, imitation_multiplier=factor,
                               effective_retention_weight=expected_retention_weight, effective_bc_weight=now['bc_w'])
        if factor == 0 and now['bc_w'] != 0:
            raise ValueError('Teacher BC remains active after joint guidance reached zero')
    if (abs(now['retention_weight']-expected_retention_weight) > 1e-12 or now['retention_minibatches'] < 0
            or (candidate_run == 28 and not now['retention_minibatches'] > 0)):
        raise ValueError('Expected active retention at the reviewed actor update')
    guard_review = None
    if candidate_run in (29, 30):
        if (now['kl_guard'] != 'minibatch_rollback' or not 0 <= now['kl'] <= args['max_kl'] + 1e-9
                or now['actor_accepted_minibatches'] + now['kl_guard_rejected_minibatches'] != now['actor_attempted_minibatches']
                or now['actor_accepted_minibatches'] != now['retention_minibatches']):
            raise ValueError('Production KL guard/counter invariant failed')
        guard_review = {k: now[k] for k in ('kl', 'kl_guard', 'kl_guard_attempted_kl',
            'actor_attempted_minibatches', 'actor_accepted_minibatches', 'kl_guard_rejected_minibatches')}
    report = dict(run='ppo_s2r/%d' % candidate_run, update=cli.update, steps=now['step'],
        checkpoint_sha256=sha(ckpath), source_reference_sha256=sha(source_path),
        configuration_differences=differences, cohorts=cohorts, comparisons=comparisons,
        retention=dict(loss=now['retention_loss'], minibatches=now['retention_minibatches'],
                       weight=now['retention_weight'], cache_sha256=manifest['cache_sha256'],
                       manifest_sha256=sha(manifest_path)),
        kl_guard_review=guard_review,
        restored_worst_source=restored_worst_provenance,
        source_stress_and_lost_guard=dict(passed=not regressions, completion_regressions=regressions),
        eligible_for_expanded_validation=(not regressions and all(
            comparisons[label]['selected'] for label in ('source_u70', 'pace_u10'))),
        original_il_selection=now['family_selection'], selected_within_own_run=bool(now['saved']),
        expanded_validation_qualified=False,
        qualification=('Matched case order and fixed seeds; jointly completed cap8 pace only. '
                       'Missing source stress/lost per-case records remain null. Same-update control '
                       'measures this seed-specific experiment, not statistical significance. '
                       'Passing fixed cohorts does not qualify unseen layouts, correct localization '
                       'recovery, or real-car performance. No checkpoint is promoted by this tool.'))
    if candidate_run == 30:
        report['joint_guidance_review'] = guidance_review
        report['qualification'] += (' Run30 changes reference guidance, BC/retention coupling and straight offset objective together. '
                                    'Run29 is a matched historical driving comparator; this comparison cannot isolate the causal effect of each changed term.')
    if control_update != cli.update:
        report['historical_control'] = dict(run=control_run, update=control_update,
            checkpoint_sha256=sha(base / str(control_run) / 'checkpoint' / ('model_%s.pt' % control['step'])))
        report['qualification'] = report['qualification'].replace(
            'Same-update control measures this seed-specific experiment, not statistical significance.',
            'Historical control uses a different training update; it is a matched driving reference, not a same-budget causal comparison.')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(comparisons, indent=2))
    for group, row in cohorts.items():
        print(group, '%s/%s; control%s, source%s' % (row['candidate_completed'], row['total'],
              row['control_completed'], row['source_completed']))


if __name__ == '__main__':
    main()
