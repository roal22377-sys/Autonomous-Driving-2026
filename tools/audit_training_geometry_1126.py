#!/usr/bin/env python3
"""Read-only geometry coverage of active and prepared training map families."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]


def cyclic_runs(mask):
    """Return (start, number of samples), joining a run across the array seam."""
    import numpy as np
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    if mask.all():
        return [(0, len(mask))]
    starts = np.flatnonzero(mask & ~np.roll(mask, 1))
    ends = np.flatnonzero(~mask & np.roll(mask, 1))
    return [(int(s), int(np.min((ends-s) % len(mask)))) for s in starts]


def describe(path):
    import numpy as np
    from utils.raceline import resample_closed, curvature_smooth
    with np.load(path, allow_pickle=False) as z:
        original = np.asarray(z['raceline'], dtype=np.float64)
        width = np.asarray(z['w_left'] + z['w_right'], dtype=np.float64)
        runoffs = np.asarray(z['runoffs'], dtype=np.float64).reshape(-1, 6)
        signature = str(z['gen_sig']) if 'gen_sig' in z else None
    xy, length = resample_closed(original, .1)
    ds = length / len(xy)
    k = curvature_smooth(xy, span=1.)
    straight = [(s, n) for s, n in cyclic_runs(abs(k) < .08) if n * ds >= 10.]
    tight = [(s, n) for s, n in cyclic_runs(abs(k) > .45) if n * ds >= .5]
    # This is a geometric tight-bend proxy, not a 180-degree hairpin classifier.
    sharp_after_straight = 0
    for start, n in straight:
        ahead = (start + n + np.arange(int(round(12./ds)))) % len(k)
        sharp_after_straight += int(np.any(abs(k[ahead]) > .45))
    gaps = [((tight[(i+1) % len(tight)][0] - (s+n)) % len(k)) * ds
            for i, (s, n) in enumerate(tight)] if len(tight) > 1 else []
    return dict(path=str(Path(path).resolve()), generator_signature=signature,
        reference_geometry_sha256=hashlib.sha256(original.tobytes() + width.tobytes() + runoffs.tobytes()).hexdigest(),
        length_m=float(length), width_min_m=float(width.min()), width_median_m=float(np.median(width)),
        tight_distance_fraction=float(np.mean(abs(k) > .45)), tight_bend_sections=len(tight),
        adjacent_tight_pairs_gap_le10m=int(sum(g <= 10. for g in gaps)),
        long_straight_sections=len(straight), longest_straight_m=max([n*ds for _, n in cyclic_runs(abs(k) < .08)], default=0.),
        straight_ge10m_then_tight_within12m=sharp_after_straight,
        annotated_runoff_count=len(runoffs),
        annotated_runoff_reach_max_m=float(runoffs[:, 3].max()) if len(runoffs) else 0.)


def aggregate(rows):
    import numpy as np
    fields = ('length_m', 'width_min_m', 'width_median_m', 'tight_distance_fraction',
              'tight_bend_sections', 'longest_straight_m', 'annotated_runoff_count')
    return dict(files=len(rows),
        distributions={k:dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                                 map(float, np.quantile([r[k] for r in rows], [0, .25, .5, .75, 1])))) for k in fields},
        files_with_long_straight_then_tight=sum(r['straight_ge10m_then_tight_within12m'] > 0 for r in rows),
        files_with_adjacent_tight_sections=sum(r['adjacent_tight_pairs_gap_le10m'] > 0 for r in rows),
        files_with_annotated_runoff=sum(r['annotated_runoff_count'] > 0 for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    out = Path(cli.output).resolve()
    if out.exists():
        raise FileExistsError('Preserve prior audit')
    os.chdir(SRC)
    import train_il_s2r as IL
    from utils.training_mix import groups_for, step_budgets
    args = IL.get_args(['--task_config', 'results/ppo_s2r/27/backup/f1tenth.yaml'])
    groups = groups_for(args, [p for p in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in p])
    active = {str(Path(p).resolve()) for ps in groups.values() for p in ps}
    assert len(active) == sum(map(len, groups.values()))
    for label, path in [('extended_hall', 'results/extended_hall_1126/v1/manifest.json'),
                        ('target_open', 'results/target_open_maps_1126/v1/manifest.json')]:
        manifest = json.loads(Path(path).read_text())
        for split in ('train', 'eval'):
            paths = [r['path'] for r in manifest['variants'] if r['split'] == split]
            assert not active.intersection(str(Path(p).resolve()) for p in paths)
            groups['prepared_' + label + '_' + split] = paths
    rows = {g: [describe(p) for p in ps] for g, ps in groups.items()}
    report = dict(status='complete_read_only', protocol='reference_geometry_1126_v1',
        definitions=dict(sample_spacing_m=.1, curvature_span_m=1., tight_threshold_per_m=.45,
                         min_tight_section_m=.5, straight_threshold_per_m=.08,
                         min_long_straight_m=10., sharp_turn_lookahead_after_straight_m=12.,
                         adjacent_tight_gap_m=10.),
        active_training_files=len(active), actual_worker_allocation=step_budgets(args.training_mix, 20),
        summary={g:aggregate(v) for g,v in rows.items()}, tracks=rows,
        qualification=('File/direction/variant counts are not independent base layouts or observed rollout exposure. '
                       'Tight sections are curvature proxies, not semantic hairpin labels. Stored widths describe '
                       'the reference corridor, not occupancy everywhere. Runoff annotations do not exhaust all '
                       'free-space geometry. No collision/dynamic feasibility or performance claim. Prepared eval '
                       'geometries remain excluded from gradients; known-base opening variants are not unseen layouts. '
                       'Reference hash covers the analyzed arrays, not occupancy. No live maps or sampler changed.'))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tracks'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
