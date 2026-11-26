#!/usr/bin/env python3
"""Compare candidate outages with restored per-case robust-teacher references."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from review_ppo_line_ablation_1126 import completed_row, matched

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=int, default=28)
    parser.add_argument('--update', type=int, required=True)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    out = Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve previous review')
    base = SRC / 'results/ppo_s2r'
    candidate = base / str(cli.run)
    now = completed_row(candidate / 'train_log.pkl', cli.update)
    source_row = completed_row(base / '25/train_log.pkl', 70)
    source_checkpoint = base / '25/checkpoint/model_3080000.pt'
    normalizer = SRC / 'results/imitation/1607/sim2real/agent_obs_scale.pkl'
    if sha(base / '25/sim2real/agent_obs_scale.pkl') != sha(normalizer):
        raise ValueError('Source export no longer matches the frozen IL normalizer')
    args = json.loads((candidate / 'backup/ppo_args.json').read_text())
    if not args['lost_eval'] or args['il'] != 'imitation/1607' or now['evaluation_seed_base'] != 7:
        raise ValueError('Unexpected candidate outage/normalizer protocol')
    cohorts = {}
    for family, rate_key, rows_key, count in (
            ('general', 'lost_eval', 'lost_eval_res', 44),
            ('short', 'lost_short', 'lost_short_res', 32)):
        path = SRC / ('results/analysis1126/ppo25_u70_lost_%s_reference_seed7.json' % family)
        reference = json.loads(path.read_text())
        key = 'regular' if family == 'general' else family
        scores = reference['families'][key]
        if (reference['checkpoint_sha256'] != sha(source_checkpoint)
                or reference['config_sha256'] != sha(candidate / 'backup/f1tenth.yaml')
                or reference['dynamics_sha256'] != sha(candidate / 'backup/dynamic.yaml')
                or not reference['localization_forced_lost'] or reference['seeds'] != [7]
                or reference['regular_scope'] != 'standard'):
            raise ValueError('Different restored evaluation protocol: ' + family)
        if reference['diagnostic_caps'] != ([5.] if family == 'general' else None):
            raise ValueError('Different restored cap schedule')
        old = [r['result'] for r in scores['runs']]
        rows = now[rows_key]
        successes = sum(bool(r[1]) for r in old)
        candidate_successes = sum(bool(r[1]) for r in rows)
        if (len(old) != count or len(rows) != count
                or successes != scores['complete']
                or abs(candidate_successes / count - now[rate_key]) > 1e-9
                or abs(successes / count - source_row[rate_key]) > 1e-9):
            raise ValueError('Restored reference does not reproduce historical counts')
        ordered_tracks = [Path(name).stem for name in scores['track_names']]
        for item in scores['runs']:
            if item['seed'] != 7 + ordered_tracks.index(item['result'][0].split('@')[0]):
                raise ValueError('Restored per-case seed mismatch')
        if ordered_tracks != now['evaluation_track_order'][rows_key]:
            raise ValueError('Different candidate per-track seed order')
        comparison = matched(rows, old)
        pace = {}
        for cap in sorted({r[0].split('@')[1].split('/')[0] for r in old}):
            pairs = [(a, b) for a, b in zip(rows, old) if a[1] and b[1]
                     and a[0].split('@')[1].split('/')[0] == cap]
            if any(not math.isfinite(float(r[2])) or r[2] <= 0
                   for pair in pairs for r in pair):
                raise ValueError('Invalid completed-run duration at cap ' + cap)
            pace[cap] = dict(common_completed=len(pairs), relative_pace_gain=(
                sum((b[2] - a[2]) / b[2] for a, b in pairs) / len(pairs) if pairs else None))
        cohorts[family] = dict(candidate_complete=candidate_successes,
            source_complete=successes, total=count, comparison=comparison, per_cap_pace=pace,
            source_report=str(path.relative_to(ROOT)), source_report_sha256=sha(path),
            historical_count_reproduced=True)
    report = dict(status='complete_read_only', run=cli.run, update=cli.update, cohorts=cohorts,
        source_checkpoint_sha256=sha(source_checkpoint),
        candidate_checkpoint_sha256=sha(candidate / 'checkpoint' / ('model_%s.pt' % now['step'])),
        observation_normalizer_sha256=sha(normalizer),
        qualification=('Source reruns explicitly used frozen IL1607, whose current file also exactly '
                       'matches the source export. The original rerun JSON does not record its '
                       'normalizer hash; this field records verified lineage, not a retroactive '
                       'measurement. Preview is invalid throughout each run while policy LiDAR '
                       'remains present. This tests the primary policy without map preview; it '
                       'does not test sensor loss, actual localization recovery, or fallback12190. '
                       'Pace includes only paired complete runs; completion failures remain visible. '
                       'Diagnostic cases never become gradient data.'))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for family, row in cohorts.items():
        print(family, '%s/%s vs source%s/%s' % (row['candidate_complete'], row['total'],
              row['source_complete'], row['total']), row['comparison'])


if __name__ == '__main__':
    main()
