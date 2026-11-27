#!/usr/bin/env python3
"""Offline paired source/geometry/joint-profile tracking test. Never trains.

An experimental wrapper supplies the exact optimized speed/curvature preview;
the teacher uses the same profile. Production/car code remains untouched.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'src/f1tenth'
sys.path[:0]=[str(SRC),str(SRC/'installation'),str(ROOT/'tools')]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=[7])
    ap.add_argument('--modes',nargs='+',choices=['eval','worst'],default=['eval'])
    ap.add_argument('--model',default='ppo_s2r/25')
    ap.add_argument('--tag',default='3080000')
    ap.add_argument('--caps',nargs='+',type=float,default=[8.])
    ap.add_argument('--variants',nargs='+',choices=['source','geometry_only','speed_only','curvature_only','joint_profile'],default=['source','geometry_only','joint_profile'])
    ap.add_argument('--controllers',nargs='+',choices=['policy','expert'],default=['policy','expert'])
    ap.add_argument('--heading-merge-gate',action='store_true',help='Offline teacher diagnostic: reduce speed during path-heading mismatch; not production adoption')
    ap.add_argument('--teacher-episode-limits',action='store_true',help='Retain privileged teacher adaptation to episode grip/acceleration/braking when loading a nominal profile')
    ap.add_argument('--teacher-controller',choices=['pure_pursuit','feedforward_feedback'],default='pure_pursuit',help='Offline diagnostic only; no car or production teacher change')
    ap.add_argument('--teacher-lad-off-gain',type=float,default=None,help='Offline teacher diagnostic: add lookahead in proportion to lateral line error; production default remains unchanged')
    cli=ap.parse_args();os.chdir(SRC)
    import numpy as np
    import torch
    import train_il_s2r as il
    from utils.env import F1Wrapper,normalize_speed
    from utils.rule_based import RacelinePlanner
    from utils.vectorize import RunningMeanStd
    from utils.driving_metrics import DrivingMetrics
    from compare_reference_limits_1126 import footprint_screen
    torch.set_num_threads(1)
    generation=json.loads(Path(cli.report).read_text())
    rows=generation.get('tracks',[])+([generation['in_progress_track']] if generation.get('in_progress_track') else [])
    accepted=[]
    for row in rows:
        source=Path(row['source']);assert sha(source)==row['source_sha256']
        for name,c in row['candidates'].items():
            if c['eligible_for_closed_loop_review']:
                candidate=Path(c['path']);assert sha(candidate)==c['sha256']
                with np.load(candidate,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                with np.load(source,allow_pickle=False) as z:
                    assert np.array_equal(a['occupancy'],z['occupancy'])
                    assert np.array_equal(a['centerline'],z['centerline'])
                static=footprint_screen(a['raceline'],a)
                if static['poses_touching_occupied_samples']!=0:continue
                accepted.append((row,name,c,a,static))
    if not accepted:raise ValueError('No dense/static-feasible candidate available')
    output=Path(cli.output).resolve()
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    args=il.get_args(['--task_config',f'results/{cli.model}/backup/f1tenth.yaml'])
    ckpath=Path(f'results/{cli.model}/checkpoint/model_{cli.tag}.pt')
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    actor=il.actor_from_state(ck['actor']);actor.eval()
    rms=RunningMeanStd('agent_obs',args.obs_dim);rms.load(f'results/{cli.model}/sim2real',required=True)
    result=dict(status='running',source_report_sha256=sha(cli.report),checkpoint_sha256=sha(ckpath),
                teacher_heading_merge_gate=cli.heading_merge_gate,
                teacher_episode_limits=cli.teacher_episode_limits,
                teacher_controller=cli.teacher_controller,
                teacher_lad_off_gain=cli.teacher_lad_off_gain,
                training_started=False,runs=[],qualification='Known training-map closed-loop diagnostic. Frozen actor and privileged teacher; not trained improvement or broad qualification.')
    for row,name,c,a,static in accepted:
        xy=a['raceline'].astype(float);speed=a['joint_speed'].astype(float)
        curvature=np.tan(a['joint_steering'].astype(float))/.34
        segment=np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1)
        arc=np.r_[0,np.cumsum(segment)]
        def interpolate(values,s):return np.interp(s%arc[-1],arc,np.r_[values,values[0]])
        class ExperimentalWrapper(F1Wrapper):
            def _reset_preview(self):
                super()._reset_preview()
                if self.reference_mode in ('speed_only','curvature_only','joint_profile'):
                    s=np.arange(self.preview.n)*self.preview.ds
                    # The preview resampler starts at the first source point.
                    assert np.linalg.norm(self.preview.xy[0]-xy[0])<1e-5
                    if self.reference_mode in ('speed_only','joint_profile'):self.preview.v=interpolate(speed,s)
                    if self.reference_mode in ('curvature_only','joint_profile'):self.preview.k=interpolate(curvature,s)
                    self.profile_audit=dict(max_speed=float(self.preview.v.max()),
                                           source_min_speed=float(speed.min()),
                                           dimensions_unchanged=self.preview.dim==18)
        for variant in cli.variants:
            path=row['source'] if variant=='source' else c['path']
            for mode in cli.modes:
                for seed in cli.seeds:
                    for cap in cli.caps:
                        for controller in cli.controllers:
                            np.random.seed(seed);random.seed(seed);args.seed=seed
                            env=ExperimentalWrapper(args,il.MAPS,obs_noise=True,sim_mode=mode,tracks=[path])
                            env.reference_mode=variant;env.fixed_speed_cap=cap;env.start_mode='center'
                            try:
                                obs,_=env.reset(options={'track':path})
                                planner=None
                                if controller=='expert':
                                    planner=RacelinePlanner(env._env.unwrapped.track,args.length_f+args.length_r,
                                        v_min=env.min_speed,v_max=env.speed_cap,max_steer=env.max_steer,
                                        max_steer_left=env.steer_left_max,max_steer_right=env.steer_right_max,
                                        **il.expert_kwargs(env))
                                    if cli.teacher_lad_off_gain is not None:
                                        planner.lad_off_gain=cli.teacher_lad_off_gain
                                    from utils.raceline import curvature_smooth
                                    planner.feedback_curvature=(curvature if variant!='source' else curvature_smooth(planner.xy,span=.5))
                                    if variant in ('speed_only','joint_profile'):
                                        assert planner.n==len(speed)
                                        assert np.allclose(planner.xy,xy,atol=1e-5)
                                        scale=1.
                                        if cli.teacher_episode_limits:
                                            ratios=[float(env.params['mu'])/float(args.mu),
                                                    float(env.params['a_max'])/float(row['limits']['a_acc']),
                                                    float(env.a_brake)/float(row['limits']['a_brake']),1.]
                                            scale=float(np.sqrt(max(0.,min(ratios))))
                                        planner.v_profile=np.minimum(speed*scale,env.speed_cap)
                                        env.teacher_profile_scale=scale
                                metrics=DrivingMetrics();trace=[]
                                while True:
                                    if controller=='expert':
                                        sq,vq=env.pending_commands();v,d=planner.plan(env.vehicle_state(),sq,vq,env.timestep)
                                        if cli.teacher_controller=='feedforward_feedback':
                                            state=env.vehicle_state()
                                            px,py,yaw,pv=planner.predict(state,sq,vq,env.timestep)
                                            i=planner.last_index
                                            tangent=planner.xy[(i+1)%planner.n]-planner.xy[(i-1)%planner.n]
                                            desired_yaw=np.arctan2(tangent[1],tangent[0])
                                            ey=-np.sin(desired_yaw)*(px-planner.xy[i,0])+np.cos(desired_yaw)*(py-planner.xy[i,1])
                                            motion_yaw=yaw+float(state[6])
                                            epsi=(desired_yaw-motion_yaw+np.pi)%(2*np.pi)-np.pi
                                            # Small-error bicycle poles: damping .9,
                                            # natural frequency2.5rad/s. Test, not a
                                            # stability claim for the slipping plant.
                                            speed_scale=max(pv,1.);wb=planner.wheelbase
                                            kh=2*.9*2.5*wb/speed_scale
                                            ke=2.5**2*wb/speed_scale
                                            ff=np.arctan(wb*planner.feedback_curvature[i])
                                            d=float(np.clip(ff+kh*epsi-np.arctan(ke*ey/speed_scale),-planner.max_steer_right,planner.max_steer_left))
                                            planner._last_cmd=(d,v)
                                        if cli.heading_merge_gate:
                                            _,_,yaw,_=planner.predict(env.vehicle_state(),sq,vq,env.timestep)
                                            i=planner.last_index
                                            tangent=planner.xy[(i+1)%planner.n]-planner.xy[(i-1)%planner.n]
                                            reference_yaw=np.arctan2(tangent[1],tangent[0])
                                            error=abs((yaw-reference_yaw+np.pi)%(2*np.pi)-np.pi)
                                            # Full planned speed <=5deg; merge speed at>=20deg.
                                            alignment=float(np.clip((np.radians(20)-error)/np.radians(15),0,1))
                                            v=min(v,planner.merge_speed+max(v-planner.merge_speed,0)*alignment)
                                            planner._last_cmd=(d,v)
                                        action=np.clip([env.steer_command(d),normalize_speed(v,env.min_speed,env.max_speed)],-1,1)
                                    else:
                                        with torch.no_grad():action=actor(torch.as_tensor(rms.normalize(obs),dtype=torch.float32)).loc.numpy()
                                    obs,_,term,trunc,_=env.step(action);metrics.update(env,action)
                                    trace.append([float(env.elapsed)]+env.vehicle_state().tolist())
                                    if term or trunc:break
                                record=dict(source=Path(row['source']).name,candidate=name,variant=variant,
                                    mode=mode,seed=seed,cap=cap,controller=controller,complete=bool(env.success),
                                    elapsed=float(env.elapsed),metrics=metrics.result(),static_screen=static,
                                    profile_override=variant in ('speed_only','curvature_only','joint_profile'),preview_audit=env.__dict__.get('profile_audit'))
                                record['teacher_profile_scale']=env.__dict__.get('teacher_profile_scale')
                                trace_dir=output.parent/(output.stem+'_traces');trace_dir.mkdir(exist_ok=True)
                                trace_path=trace_dir/f'{len(result["runs"]):04d}.npz'
                                np.savez_compressed(trace_path,trace=np.asarray(trace))
                                record['trace']=str(trace_path);result['runs'].append(record)
                                output.write_text(json.dumps(result,indent=2)+'\n')
                                print(json.dumps({k:v for k,v in record.items() if k not in ('metrics','static_screen')}),flush=True)
                            finally:env.close()
    result['status']='complete';output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
