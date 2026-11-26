#!/usr/bin/env python3
"""Read-only reference feasibility screen; not a dynamic tracking qualification."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]


def screen(xy, limits, wheelbase, max_steer, max_rate):
    import numpy as np
    from utils.raceline import curvature_smooth, speed_profile, lap_time
    k = curvature_smooth(xy)
    v = speed_profile(xy, **limits)
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    dt = 2 * seg / np.maximum(v + np.roll(v, -1), 1e-6)
    steer = np.arctan(wheelbase * k)
    rate = np.diff(np.r_[steer, steer[0]]) / np.maximum(dt, 1e-6)
    ax = (np.roll(v, -1)**2 - v**2) / np.maximum(2*seg, 1e-6)
    ay = v**2 * k
    return dict(lap_time_s=float(lap_time(xy, v)), max_speed=float(v.max()),
                max_lateral_acc=float(np.abs(ay).max()),
                max_kinematic_steer_rad=float(np.abs(steer).max()),
                max_kinematic_steer_rate_rad_s=float(np.abs(rate).max()),
                steer_limit_fraction=float(np.mean(np.abs(steer) > max_steer)),
                steer_rate_limit_fraction=float(np.mean(np.abs(rate) > max_rate)),
                max_combined_acc_proxy=float(np.hypot(ax, ay).max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks', nargs='+', required=True)
    p.add_argument('--output', required=True)
    cli = p.parse_args()
    os.chdir(SRC)
    import numpy as np
    from ruamel.yaml import YAML
    from utils.raceline import resample_closed
    from utils.track_gen import LAP_TIME_LIMITS
    cfg = YAML().load(Path('configs/task/f1tenth_preview_1126.yaml').read_text())
    dynamics = YAML().load(Path('configs/task/dynamic.yaml').read_text())
    plan = cfg['track_preview']['plan']
    current = dict(v_min=plan['v_min'], v_max=plan['v_max'],
                   a_lat=plan['grip']*plan['grip_share'], a_acc=plan['a_acc'], a_brake=plan['a_brake'])
    report = dict(status='reference screen only', geometry_generator_limits=dict(LAP_TIME_LIMITS),
                  current_speed_limits=current, tracks=[], limitations=[
                      'Uses existing stored path with legacy/current speed plans; no line reoptimization yet.',
                      'Kinematic feedforward angle and rate ignore slip, tracking error and actuator dynamics.',
                      'Combined acceleration is a geometric proxy, not a measured tire limit.',
                      'No footprint collision check; all candidates need dense clearance and closed-loop validation.',
                      'Generator source settings do not establish provenance of every stored map.'])
    for item in cli.tracks:
        path = Path(item)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            xy, _ = resample_closed(z['raceline'].astype(np.float64), .1)
            sig = str(z['gen_sig']) if 'gen_sig' in z else None
        kwargs = dict(wheelbase=float(dynamics['length_f'] + dynamics['length_r']),
                      max_steer=float(cfg['max_steer']), max_rate=float(dynamics['sv_max']))
        row = dict(track=str(path), sha256=before, stored_generator_signature=sig,
                   legacy=screen(xy, dict(LAP_TIME_LIMITS), **kwargs),
                   current=screen(xy, current, **kwargs))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        report['tracks'].append(row)
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
