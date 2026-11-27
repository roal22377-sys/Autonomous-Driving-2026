#!/usr/bin/env python3
"""Compare phase telemetry only on matched, jointly completed cap8 runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from review_ppo_line_ablation_1126 import COHORTS, completed_row, matched

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
PROTOCOL = 'smoothed_raceline_008_v1'


def compare(current, reference):
    match = matched(current, reference)  # Also checks ordering/unique case identities.
    common = [(a, b) for a, b in zip(current, reference) if a[1] and b[1]
              and a[0].split('@')[-1].split('/')[0] == '8']
    phases = {}
    for phase in ('straight', 'entry', 'hairpin', 'exit', 'other'):
        pairs = [(a[4][phase], b[4][phase]) for a, b in common
                 if phase in a[4] and phase in b[4]]
        if not pairs:
            continue
        if any(row['protocol'] != PROTOCOL or row['samples'] < 1
               for pair in pairs for row in pair):
            raise ValueError('Different phase protocol or empty measurements')
        fields = {}
        for metric in ('speed', 'lateral_rms', 'heading_rms', 'steer_weave_rms', 'seconds'):
            values = np.array([[a[metric], b[metric]] for a, b in pairs], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite phase measurement')
            fields[metric] = dict(candidate_mean=float(values[:, 0].mean()),
                reference_mean=float(values[:, 1].mean()),
                paired_mean_delta=float((values[:, 0] - values[:, 1]).mean()))
        phases[phase] = dict(paired_cases=len(pairs), metrics=fields)
    return dict(jointly_completed_cap8_cases=match['common_completed_uncapped'], phases=phases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=int, default=28)
    p.add_argument('--control', type=int, default=27)
    p.add_argument('--update', type=int, required=True)
    p.add_argument('--control_update', type=int)
    p.add_argument('--output', required=True)
    cli = p.parse_args()
    out = Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve previous phase review')
    base = SRC / 'results/ppo_s2r'
    control_update = cli.update if cli.control_update is None else cli.control_update
    control_label = 'same_update_control' if control_update == cli.update else 'historical_control'
    now = completed_row(base / str(cli.run) / 'train_log.pkl', cli.update)
    control = completed_row(base / str(cli.control) / 'train_log.pkl', control_update)
    if now['evaluation_seed_base'] != 7 or control['evaluation_seed_base'] != 7:
        raise ValueError('Unexpected fixed seed schedule')
    source_path = SRC / 'results/analysis1126/ppo25_u70_full_family_reference.json'
    source = json.loads(source_path.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(base / '25/checkpoint/model_3080000.pt') != source['checkpoint_sha256']:
        raise ValueError('Starting source identity changed')
    result = {}
    for family, reference in source['families'].items():
        _, key, _ = COHORTS[family]
        result[family] = dict(vs_source_u70=compare(now[key], reference['res']))
        result[family]['vs_' + control_label] = compare(now[key], control[key])
    report = dict(status='complete_read_only', run=cli.run, control=cli.control, update=cli.update,
        checkpoint_sha256=digest(base / str(cli.run) / 'checkpoint' / ('model_%s.pt' % now['step'])),
        source_reference_sha256=digest(source_path), protocol=PROTOCOL, families=result,
        qualification=('Equally weighted paired case means of existing phase telemetry; failed runs '
                       'are excluded here and remain counted in the full completion review. Bins '
                       'use projected smoothed-reference curvature, so their occupancy can change '
                       'with trajectory. Phase seconds are not independently timed fixed spatial '
                       'sectors. Hairpin means the high-curvature proxy, not a180-degree turn label. '
                       'Steer weave uses commanded action*nominal steering lock, not measured wheel '
                       'motion. Reference-line offset is not an optimal-raceline objective. This '
                       'diagnostic cannot override completion/lap-time selection or qualify hardware.'))
    if control_update != cli.update:
        report['historical_control_update'] = control_update
        report['qualification'] += ' Historical control uses a different training update, not an equal-budget causal comparison.'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for family, row in result.items():
        for phase in ('straight', 'hairpin'):
            metrics = row['vs_source_u70']['phases'].get(phase)
            if metrics:
                print(family, phase, 'pairs', metrics['paired_cases'],
                      {k: round(v['paired_mean_delta'], 6) for k, v in metrics['metrics'].items()})


if __name__ == '__main__':
    main()
