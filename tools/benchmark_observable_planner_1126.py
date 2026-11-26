#!/usr/bin/env python3
"""Offline planner prototype using estimated pose, observed speed and command history.

No deployment: missing localization ends a diagnostic run without a fallback.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]


def observation_tap_class(base):
    class ObservationTap(base):
        def getObs(self, obs_dict, reset=False):
            # Capture the very same noisy pose already passed to preview.features.
            # No additional RNG draw or substitution of the true vehicle pose.
            features = self.preview.features
            def capture(x, y, yaw, speed_cap=None, valid=True, hint=None):
                self.estimated_pose = (float(x), float(y), float(yaw), bool(valid))
                return features(x, y, yaw, speed_cap, valid, hint)
            self.preview.features = capture
            try:
                return super().getObs(obs_dict, reset)
            finally:
                self.preview.features = features

    return ObservationTap


def pending_from_history(hist, steps, steer_delay, speed_delay, wheel, speed):
    """Reset padding is not a command: normalized speed zero means 4.25 m/s.

    Before a queue fills, its older slots are physical zero, as on env reset.
    The step counter is local controller state, not privileged simulator data.
    """
    if max(steer_delay, speed_delay) > len(hist):
        raise ValueError('Command history is shorter than the assumed delay')
    sq = [wheel(float(hist[k, 0])) if k < steps else 0.
          for k in reversed(range(steer_delay))]
    vq = [speed(float(hist[k, 1])) if k < steps else 0.
          for k in reversed(range(speed_delay))]
    return sq, vq


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tracks', nargs='+', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[7])
    p.add_argument('--caps', nargs='+', type=float, default=[8, 4])
    p.add_argument('--mode', choices=['eval', 'worst'], default='eval')
    p.add_argument('--diagnostic-oracle', choices=['none', 'pose', 'wheel', 'both'], default='none',
                   help='Privileged diagnostic only; never qualifies an observable controller')
    p.add_argument('--corner-lookahead', type=float, default=.65)
    p.add_argument('--output', required=True)
    cli = p.parse_args()
    if not 0 < cli.corner_lookahead < float('inf'):
        p.error('--corner-lookahead must be positive and finite')
    os.chdir(SRC)
    import numpy as np
    import random
    import torch
    import train_il_s2r as IL
    from utils.env import F1Wrapper, normalize_speed, unnormalize_speed
    from utils.rule_based import RacelinePlanner
    from utils.driving_metrics import DrivingMetrics

    ObservationTap = observation_tap_class(F1Wrapper)

    args = IL.get_args(['--task_config', 'configs/task/f1tenth_preview_1126.yaml'])
    torch.set_num_threads(1)
    from utils.observation import get_obs_config
    oc = get_obs_config(args)
    if oc['lidar']['num_beams'] != 88 or not oc['speed']['enabled'] or oc['history']['length'] != 12:
        raise ValueError('Prototype requires 88 LiDAR bins, speed and 12 command pairs')
    nominal = args.sim2real
    dt = float(args.timestep)
    ns = int(round(nominal['steer_delay_nominal']/dt))
    nv = int(round(nominal['speed_delay_nominal']/dt))
    locks = [float(nominal[f'steer_{side}_max_nominal']) for side in ('left','right')]
    scales = [float(nominal[f'steer_scale_{side}_nominal']) for side in ('left','right')]
    def wheel(command):
        side = 0 if command > 0 else 1
        return command*locks[side]*(scales[side]+(1-scales[side])*command**2)
    def command(angle):
        side = 0 if angle > 0 else 1
        target = np.clip(abs(angle)/locks[side],0.,1.)
        lo, hi = 0., 1.
        for _ in range(35):
            mid = (lo+hi)/2
            if mid*(scales[side]+(1-scales[side])*mid**2) < target: lo = mid
            else: hi = mid
        return float(np.copysign((lo+hi)/2, angle))
    report = dict(status='running', mode=cli.mode, diagnostic_oracle=cli.diagnostic_oracle,
                  corner_lookahead=cli.corner_lookahead,
                  eligible_observable_benchmark=cli.diagnostic_oracle == 'none',
                  history_protocol='physical_zero_reset_padding_v2', runs=[], limitations=[
        'Prototype; no car operation. No proof of hardware-equivalent actuator estimation.',
        'Controller only receives static map, noisy estimated pose, observed speed, command history and nominal constants.',
        'Wheel angle is estimated by a nominal rate model; true wheel/slip/grip/delay are not supplied.',
        'Localization loss ends the run as incomplete; LiDAR fallback is not implemented.',
        'Only requested known courses; this does not establish broad robustness.'])
    if cli.diagnostic_oracle != 'none':
        report['limitations'].append('PRIVILEGED DIAGNOSTIC: selected pose/wheel inputs use simulator truth.')
    dest = Path(cli.output); dest.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted(cli.tracks):
        with np.load(path) as z:
            track = SimpleNamespace(raceline=SimpleNamespace(xs=z['raceline'][:,0].copy(), ys=z['raceline'][:,1].copy()))
        for seed in cli.seeds:
            for cap in cli.caps:
                np.random.seed(seed); random.seed(seed); args.seed=seed
                env = ObservationTap(args, IL.MAPS, obs_noise=True, sim_mode=cli.mode, tracks=[path])
                env.fixed_speed_cap=cap; env.start_mode='center'
                try:
                    obs,_=env.reset(options={'track':path})
                    planner=RacelinePlanner(track,args.length_f+args.length_r,v_min=args.min_speed,v_max=cap,
                        a_lat=6.,a_acc=nominal['a_max_nominal'],a_brake=nominal['a_brake_nominal'],
                        max_steer=args.max_steer,max_steer_left=locks[0],max_steer_right=locks[1],
                        sv_max=args.sv_max,sim_agent=None,a_lat_steer=6.,steer_speed_mode='hold',
                        lad_base=.6,lad_gain=.22,corner_lad_cap=cli.corner_lookahead,corner_lad_kappa=.5)
                    estimated_wheel=0.; metrics=DrivingMetrics(); latencies=[]; reason='episode_end'
                    steps=0; wheel_errors=[]; pose_errors=[]
                    while True:
                        x,y,yaw,valid=env.estimated_pose
                        if not valid: reason='localization_unavailable'; break
                        # Truth is logged only, except in explicitly labelled diagnostic runs.
                        truth=env.vehicle_state()
                        pose_errors.append(float(np.hypot(x-truth[0],y-truth[1])))
                        if cli.diagnostic_oracle in ('pose','both'):
                            x,y,yaw=map(float,(truth[0],truth[1],truth[4]))
                        start=time.perf_counter()
                        # Fixed schema of this prototype: 88 LiDAR + speed + 12 pairs.
                        speed=float(obs[88]); hist=obs[89:113].reshape(12,2)
                        sq,vq=pending_from_history(hist,steps,ns,nv,wheel,
                            lambda c: float(unnormalize_speed(c,args.min_speed,args.max_speed)))
                        control_wheel=float(truth[2]) if cli.diagnostic_oracle in ('wheel','both') else estimated_wheel
                        state=np.array([x,y,control_wheel,speed,yaw,0.,0.])
                        v,angle=planner.plan(state,sq,vq,dt)
                        action=np.clip([command(angle),normalize_speed(v,args.min_speed,args.max_speed)],-1,1)
                        target=sq[0] if sq else wheel(action[0])
                        estimated_wheel+=float(np.clip(target-estimated_wheel,-args.sv_max*dt,args.sv_max*dt))
                        latencies.append(time.perf_counter()-start)
                        obs,_,term,trunc,_=env.step(action)
                        steps+=1
                        wheel_errors.append(float(estimated_wheel-env.vehicle_state()[2]))
                        metrics.update(env,action)
                        if term or trunc:
                            reason='complete' if env.success else 'collision' if env.collision else 'truncated' if trunc else 'terminated'
                            break
                    row=dict(track=Path(path).name,seed=seed,cap=cap,complete=bool(env.success),
                             elapsed=float(env.elapsed),reason=reason,phases=metrics.result(),
                             estimated_wheel_rmse=float(np.sqrt(np.mean(np.square(wheel_errors)))) if wheel_errors else None,
                             estimated_wheel_max_error=float(np.max(np.abs(wheel_errors))) if wheel_errors else None,
                             estimated_pose_rmse=float(np.sqrt(np.mean(np.square(pose_errors)))) if pose_errors else None,
                             controller_p95_ms=float(np.quantile(latencies,.95)*1000) if latencies else None)
                    report['runs'].append(row)
                    dest.write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps({k:v for k,v in row.items() if k!='phases'}),flush=True)
                finally:env.close()
    report['status']='complete';dest.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
