#!/usr/bin/env python3
"""Time fixed geometric gates using recorded simulator XY, independently of preview.

This is a diagnostic of selected traces, never a replacement for complete laps.
Corner regions are selected from map curvature before inspecting policy timing.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/f1tenth'))
from utils.raceline import curvature_smooth, resample_closed


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crossing_times(xy, times, center, tangent, half_width):
    """Interpolate forward/backward crossings of a finite gate segment."""
    displacement = xy - center
    along = displacement @ tangent
    normal = np.array([-tangent[1], tangent[0]])
    lateral = displacement @ normal
    nonzero = np.flatnonzero(np.abs(along) > 1e-10)
    left, right = nonzero[:-1], nonzero[1:]
    result = {}
    for label, mask in (
            ('forward', (along[left] < 0) & (along[right] > 0)),
            ('backward', (along[left] > 0) & (along[right] < 0))):
        # Touching then returning to the same side is not a crossing. A single
        # exact on-gate sample has an exact time; prolonged on-gate intervals
        # have no unique crossing time and are deliberately left untimed.
        a, b = left[mask & (right-left <= 2)], right[mask & (right-left <= 2)]
        fraction = -along[a] / (along[b] - along[a])
        offset = lateral[a] + fraction * (lateral[b] - lateral[a])
        values = times[a] + fraction * (times[b] - times[a])
        exact = b-a == 2
        offset[exact] = lateral[a[exact]+1]
        values[exact] = times[a[exact]+1]
        selected = np.abs(offset) <= half_width
        result[label] = values[selected].tolist()
    return result


def corner_gates(xy, ds, threshold, padding):
    curvature = curvature_smooth(xy, span=1.0)
    mask = np.abs(curvature) > threshold
    if mask.all():
        raise ValueError('The whole course exceeds the corner threshold')
    starts = np.flatnonzero(mask & ~np.roll(mask, 1))
    n = len(xy)
    pad = max(1, int(round(padding / ds)))
    tangent_span = max(1, int(round(.5 / ds)))
    regions = []
    for start in starts:
        length = 1
        while mask[(start + length) % n]:
            length += 1
        # Very short spikes are excluded by a map-only, fixed length criterion.
        if length * ds < .5:
            continue
        end = int(start + length)
        indices = [(int(start) - pad) % n, int(start), end % n, (end + pad) % n]
        gates = []
        for label, index in zip(('entry', 'core_start', 'core_end', 'exit'), indices):
            tangent = xy[(index + tangent_span) % n] - xy[(index - tangent_span) % n]
            tangent = tangent / np.linalg.norm(tangent)
            gates.append(dict(name=label, reference_s_m=float(index * ds),
                              center=xy[index].tolist(), tangent=tangent.tolist()))
        regions.append(dict(id=len(regions), core_length_m=float(length * ds),
                            maximum_abs_curvature=float(np.abs(curvature[(start + np.arange(length)) % n]).max()),
                            gates=gates))
    return regions


def measure(stem, regions, half_width):
    meta_path, csv_path = stem.with_suffix('.json'), stem.with_suffix('.csv')
    meta = json.loads(meta_path.read_text())
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    xy = np.column_stack((data['x'], data['y']))
    times = data['state_time']
    if not np.isfinite(xy).all() or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('Invalid trajectory or sample times')
    if abs(times[-1] - meta['elapsed']) > 1e-6:
        raise ValueError('CSV and summary duration differ')
    boundaries = [0.] + list(meta['lap_times'])
    if times[-1] > boundaries[-1] + 1e-6:
        boundaries.append(float(times[-1]))
    rows = []
    for region in regions:
        crossings = [crossing_times(xy, times, np.array(g['center']), np.array(g['tangent']), half_width)
                     for g in region['gates']]
        for lap, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
            forward = [[t for t in c['forward'] if begin <= t < end] for c in crossings]
            backward = [[t for t in c['backward'] if begin <= t < end] for c in crossings]
            usable = all(len(t) == 1 for t in forward) and not any(backward)
            ordered = usable and all(b[0] > a[0] for a, b in zip(forward[:-1], forward[1:]))
            row = dict(corner=region['id'], lap=lap, forward_crossings=forward,
                       backward_crossings=backward, timed=bool(ordered),
                       lap_completed=lap <= len(meta['lap_times']))
            if ordered:
                t = np.array([v[0] for v in forward])
                row['seconds'] = dict(entry=float(t[1]-t[0]), core=float(t[2]-t[1]),
                                      exit=float(t[3]-t[2]), total=float(t[3]-t[0]))
            else:
                row['reason'] = 'Missing, repeated, backward, or out-of-order crossings; no inferred sector time'
            rows.append(row)
    return dict(meta=meta, csv_sha256=digest(csv_path), summary_sha256=digest(meta_path), sectors=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True, help='Trace path without suffix')
    parser.add_argument('--reference', type=Path, required=True, help='Trace path without suffix')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=.45)
    parser.add_argument('--padding', type=float, default=2.)
    parser.add_argument('--half-width', type=float, default=1.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous sector review')
    if min(args.threshold, args.padding, args.half_width) <= 0:
        raise ValueError('Geometry parameters must be positive')
    with np.load(args.track, allow_pickle=False) as z:
        xy, length = resample_closed(z['raceline'], ds=.1)
    regions = corner_gates(xy, float(length)/len(xy), args.threshold, args.padding)
    candidate, reference = [measure(path, regions, args.half_width) for path in (args.candidate, args.reference)]
    for field in ('track', 'cap', 'start', 'seed'):
        if candidate['meta'][field] != reference['meta'][field]:
            raise ValueError('Different trace case: ' + field)
    if candidate['meta']['track'] != args.track.name:
        raise ValueError('Trace track differs from gate map')
    pairs = []
    old = {(r['corner'], r['lap']): r for r in reference['sectors']}
    for row in candidate['sectors']:
        prior = old.get((row['corner'], row['lap']))
        if row['timed'] and prior and prior['timed']:
            pairs.append(dict(corner=row['corner'], lap=row['lap'],
                both_laps_completed=row['lap_completed'] and prior['lap_completed'],
                candidate_seconds=row['seconds'], reference_seconds=prior['seconds'],
                delta_seconds={k: row['seconds'][k]-prior['seconds'][k] for k in row['seconds']}))
    report = dict(status='complete_read_only', track_sha256=digest(args.track),
        protocol=dict(curvature_threshold=args.threshold, padding_m=args.padding,
                      gate_half_width_m=args.half_width, curvature_span_m=1.),
        regions=regions, candidate=candidate, reference=reference, paired_sectors=pairs,
        qualification=('Finite transverse gates fixed from map curvature, crossed by recorded true XY; '
                       'no policy localization or projection index is used. Times are interpolated between '
                       'simulator samples. High curvature is not a180-degree hairpin label. All gates '
                       'must be crossed once, forward and in order within the same lap; start-line '
                       'straddling or ambiguous sectors are not timed. Width can miss a valid alternate '
                       'line; missing gates do not themselves mean collision. Partial failed laps '
                       'remain labeled and cannot establish completion or overall pace improvement. '
                       'Overlapping padded regions must not be summed. Selected-case simulation only.'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Regions:', len(regions), 'paired sectors:', len(pairs))
    for row in pairs:
        print(row['corner'], 'lap', row['lap'], 'completed', row['both_laps_completed'], row['delta_seconds'])


if __name__ == '__main__':
    main()
