#!/usr/bin/env python3
"""Separate corner settling, sustained straight tracking and steering command motion."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path.insert(0, str(SRC))


def analyze(path):
    import numpy as np
    from utils.preview import TrackPreview, get_preview_config
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta.get('diagnostic_override'):
        raise ValueError('Use unmodified policy traces for straight comparisons')
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    data = {key:np.array([float(r[key]) for r in rows]) for key in rows[0]}
    files = [p for folder in ('maps_gen','maps_real','maps_hall','maps_short')
             for p in (SRC/'installation'/folder).glob(meta['track'])]
    if len(files) != 1:
        raise ValueError(f'Expected exactly one track: {meta["track"]}')
    with np.load(files[0]) as z:
        line = TrackPreview(z['raceline'], get_preview_config({'track_preview':{'enabled':True}}))
    future_offsets = np.round(np.array([0., .5, 1., 2.])/line.ds).astype(int)
    past_offsets = np.arange(int(round(1./line.ds))+1)
    hint = None; gate = []; projection_errors = []
    for x,y,k in zip(data['x'],data['y'],data['curvature']):
        hint = line.project(x,y,hint=hint)
        projection_errors.append(abs(line.k[hint]-k))
        gate.append(max(np.abs(line.k[(hint+future_offsets)%line.n]).max(),
                        np.abs(line.k[(hint-past_offsets)%line.n]).max()) < .08)
    assert max(projection_errors) < 1e-10, 'Offline projection does not match recorded reward geometry'
    gate = np.asarray(gate)
    dt = float(np.median(np.diff(data['state_time'])))
    assert np.max(np.abs(np.diff(data['state_time'])-dt)) < 1e-8
    age = 0.; settled = []
    for active in gate:
        age = age+dt if active else 0.
        settled.append(active and age >= .5-1e-9)
    settled = np.asarray(settled)
    steer = data['policy_steer']*.5
    command_rate = np.diff(steer,prepend=steer[0])/dt
    wheel_rate = np.diff(data['wheel_actual'],prepend=data['wheel_actual'][0])/dt
    def metrics(mask):
        n = int(mask.sum())
        if not n:
            return dict(samples=0,seconds=0.)
        return dict(samples=n,seconds=n*dt,speed_mean=float(data['speed'][mask].mean()),
                    lateral_rms=float(np.sqrt(np.mean(data['lateral'][mask]**2))),
                    lateral_abs_p95=float(np.quantile(np.abs(data['lateral'][mask]),.95)),
                    heading_rms=float(np.sqrt(np.mean(data['heading'][mask]**2))),
                    command_rate_rms=float(np.sqrt(np.mean(command_rate[mask]**2))),
                    wheel_rate_rms=float(np.sqrt(np.mean(wheel_rate[mask]**2))))
    transitions = np.diff(np.r_[False,settled,False].astype(int))
    segments = []; detrended_sum = 0.; detrended_n = 0
    for a,b in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)):
        if (b-a)*dt < 2.-1e-9:
            continue
        t = np.arange(b-a)*dt; y = data['lateral'][a:b]
        residual = y-np.polyval(np.polyfit(t,y,1),t)
        detrended_sum += float(residual@residual); detrended_n += len(y)
        segments.append(dict(start_time=float(data['state_time'][a]),seconds=len(y)*dt,
                             speed_mean=float(data['speed'][a:b].mean()),
                             lateral_mean=float(y.mean()),lateral_rms=float(np.sqrt(np.mean(y*y))),
                             detrended_lateral_rms=float(np.sqrt(np.mean(residual**2))),
                             wheel_rate_rms=float(np.sqrt(np.mean(wheel_rate[a:b]**2)))))
    return dict(model=meta['model'],tag=meta['tag'],track=meta['track'],seed=meta['seed'],cap=meta['cap'],
                success=meta['success'],elapsed=meta['elapsed'],checkpoint_sha256=meta['checkpoint_sha256'],
                trace=str(path.relative_to(ROOT)),projection_curvature_max_error=max(projection_errors),
                straight_gate=metrics(gate),settled_straight=metrics(settled),long_segments=segments,
                long_segment_detrended_lateral_rms=(detrended_sum/detrended_n)**.5 if detrended_n else None,
                phase_metrics=meta['phases'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces',nargs='+',required=True)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    report = dict(protocol='offline_settled_straight_v2',cases=[analyze(Path(f).resolve()) for f in args.traces],
                  definitions={'straight':'smoothedraceline absolute curvature<.08, ahead0/.5/1/2m and densepast1m',
                               'settled':'straightgate true continuously>=.5s',
                               'long_segments':'continuous settledstraight>=2s',
                               'detrended':'Remove each segment linear lateral drift; also report rawoffset/error',
                               'steering':'command uses normalizedaction*.5rad; wheelrate uses actualsimulatedwheelangle'},
                  limitations=['Diagnostic subset; no claim of universal straight stability.',
                               'Models can traverse a segment at different speeds and lateral offsets.',
                               'Detrended lateral RMS does not by itself identify an oscillation frequency.',
                               'Simulated wheelangle is not measured real-car servo response.'])
    dest = Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+'\n')
    for case in report['cases']:
        print(case['model'],case['tag'],case['track'],case['settled_straight'],flush=True)


if __name__ == '__main__':
    main()
