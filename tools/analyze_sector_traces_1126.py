#!/usr/bin/env python3
"""Compare fixed spatial sector times, including approach and corner exit."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'src/f1tenth'
sys.path.insert(0, str(SRC))


def sectors(curvature, ds, before=3., after=5., threshold=.45):
    """Partition the entire closed path; nearby tight corners merge naturally."""
    core = np.abs(curvature) > threshold
    mask = np.zeros(len(core), bool)
    for offset in range(-int(np.ceil(before/ds)), int(np.ceil(after/ds))+1):
        mask |= np.roll(core, offset)
    transitions = np.flatnonzero(mask != np.roll(mask, 1))
    if not len(transitions):
        transitions = np.array([0])
    result = []
    for j, start in enumerate(transitions):
        end = transitions[(j+1) % len(transitions)]
        count = int((end-start) % len(mask)) or len(mask)
        result.append(dict(id=j, start_m=float(start*ds), length_m=float(count*ds),
                           kind='tight_corner_sequence' if mask[start] else 'connecting_section'))
    assert abs(sum(s['length_m'] for s in result)-len(core)*ds) < 1e-8
    return result


def crossing(time, progress, level):
    """First forward crossing; never extrapolate an unobserved boundary."""
    hits = np.flatnonzero((progress[:-1] < level) & (progress[1:] >= level))
    if not len(hits):
        return None
    i = int(hits[0])
    fraction = (level-progress[i])/(progress[i+1]-progress[i])
    return float(time[i]+fraction*(time[i+1]-time[i])), i


def sector_times(time, progress, length, definitions, bad_steps=None):
    bad_steps = np.zeros(len(time)-1, bool) if bad_steps is None else bad_steps
    result = []
    for sector in definitions:
        for cycle in range(int(np.floor(progress.min()/length))-1,
                           int(np.ceil(progress.max()/length))+1):
            start = sector['start_m']+cycle*length
            entry = crossing(time, progress, start)
            if entry is None:
                continue
            # A reversal before entry must not supply this sector's exit.
            offset = entry[1]
            exit_ = crossing(time[offset:], progress[offset:], start+sector['length_m'])
            if exit_ is None:
                continue
            t_exit, j_exit = exit_[0], exit_[1]+offset
            if t_exit <= entry[0]:
                continue
            window = slice(offset, j_exit+1)
            delta = np.diff(progress)[window]
            result.append(dict(sector_id=sector['id'], cycle=cycle, kind=sector['kind'],
                               entry_time=entry[0], exit_time=t_exit, seconds=t_exit-entry[0],
                               backwards_m=float(-delta[delta < 0].sum()),
                               projection_valid=not bool(np.any(bad_steps[window]))))
    return result


def analyze(path):
    from utils.preview import TrackPreview, get_preview_config
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta.get('diagnostic_override'):
        raise ValueError('Use unmodified policy traces for performance comparisons')
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    data = {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
    paths = [p for folder in ('maps_gen','maps_real','maps_hall','maps_short','maps_open_test/big')
             for p in (SRC/'installation'/folder).glob(meta['track'])]
    if len(paths) != 1:
        raise ValueError('Ambiguous or missing source track: '+meta['track'])
    with np.load(paths[0]) as z:
        line = TrackPreview(z['raceline'], get_preview_config({'track_preview':{'enabled':True}}))
    hint = None; positions = []; errors = []
    for x,y,k in zip(data['x'], data['y'], data['curvature']):
        hint = line.project(x,y,hint=hint)
        errors.append(abs(line.k[hint]-k))
        tangent = np.array([np.cos(line.heading[hint]), np.sin(line.heading[hint])])
        fraction = np.clip((np.array([x,y])-line.xy[hint])@tangent, -line.ds/2, line.ds/2)
        positions.append((hint*line.ds+fraction) % line.L)
    if max(errors) > 1e-10:
        raise ValueError('Offline projection does not match recorded geometry')
    positions = np.asarray(positions)
    delta = (np.diff(positions)+line.L/2) % line.L-line.L/2
    progress = np.r_[positions[0], positions[0]+np.cumsum(delta)]
    motion = np.hypot(np.diff(data['x']), np.diff(data['y']))
    bad = np.abs(delta) > np.maximum(1., 2*motion+.3)
    definitions = sectors(line.k,line.ds)
    measured = sector_times(data['state_time'], progress, line.L, definitions, bad)
    return dict(model=meta['model'],tag=meta['tag'],track=meta['track'],seed=meta['seed'],cap=meta['cap'],
                start=meta.get('start','center'),success=meta['success'],elapsed=meta['elapsed'],
                checkpoint_sha256=meta['checkpoint_sha256'],track_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                trace=str(path),sample_dt_s=float(np.median(np.diff(data['state_time']))),
                reference_length_m=line.L,definitions=definitions,passes=measured,
                suspicious_projection_steps=int(bad.sum()),projection_curvature_max_error=max(errors))


def compare(baseline, candidate):
    for key in ('track_sha256','track','seed','cap','start','definitions'):
        if baseline[key] != candidate[key]:
            raise ValueError('Unmatched comparison: '+key)
    # Do not hide failures by comparing only the successful prefix of a crash.
    if not baseline['success'] or not candidate['success']:
        return dict(qualified=False,reason='At least one full episode failed',paired_passes=[])
    a = {(r['sector_id'],r['cycle']):r for r in baseline['passes'] if r['projection_valid']}
    b = {(r['sector_id'],r['cycle']):r for r in candidate['passes'] if r['projection_valid']}
    pairs = [dict(sector_id=k[0],cycle=k[1],kind=a[k]['kind'],baseline_seconds=a[k]['seconds'],
                  candidate_seconds=b[k]['seconds'],seconds_faster=a[k]['seconds']-b[k]['seconds'])
             for k in sorted(a.keys() & b.keys())]
    return dict(qualified=True,full_episode_seconds_faster=baseline['elapsed']-candidate['elapsed'],
                paired_passes=pairs,unmatched_passes=len(a.keys() ^ b.keys()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    baseline,candidate=[analyze(Path(x).resolve()) for x in (args.baseline,args.candidate)]
    result=dict(protocol='fixed_spatial_sectors_v1',baseline=baseline,candidate=candidate,
                comparison=compare(baseline,candidate),
                definitions='Tight curvature>.45/m expanded3m before/5m after; merge overlaps. Remaining path is connecting sections. Boundaries fixed from map, independent of policy.',
                limitations=['Known-case diagnostic; does not establish broad model superiority.',
                             'Times interpolated from sampled true simulated poses, not real localization estimates.',
                             'Partial initial/final sectors omitted. Cycles refer to reference-line wrap crossings.',
                             'Average speed or exit speed alone is not the selection objective.',
                             'Projection anomalies invalidate affected sectors; full episode outcome remains reported.'])
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['comparison'],indent=2))


if __name__=='__main__':main()
