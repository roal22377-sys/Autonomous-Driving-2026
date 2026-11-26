#!/usr/bin/env python3
"""Inject timed preview faults; tests one policy, NOT the full car localizer/fallback."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'src/f1tenth'
sys.path[:0]=[str(SRC),str(SRC/'installation')]
POLICIES={}


class PreviewFault:
    def __init__(self, original, env, phase, condition):
        self.original,self.env,self.phase,self.condition=original,env,phase,condition
        self.started=None;self.hint=None;self.active=False;self.trigger_pose=None;self.trigger_s=None
        self.duration={'control':0.,'loss05':.5,'loss2':2.,'loss4':4.,'loss8':8.,'wrong_pose1':1.}[condition]

    def step(self,x,y,yaw):
        result=self.original(x,y,yaw)  # exactly one original noise draw, even during faults
        line=self.env.preview
        self.hint=line.project(x,y,hint=self.hint)
        ahead=(self.hint+np.round(np.array([0.,.5,1.,2.])/line.ds).astype(int))%line.n
        future=float(np.abs(line.k[ahead]).max())
        phase_ok=(future>.45 and abs(line.k[self.hint])<.45) if self.phase=='entry' else future<.08
        if self.started is None and self.env.elapsed>=2. and phase_ok:
            self.started=float(self.env.elapsed)
            self.trigger_pose=[float(x),float(y),float(yaw)]
            self.trigger_s=float(self.hint*line.ds)
        self.active=self.started is not None and self.started<=self.env.elapsed<self.started+self.duration-1e-9
        if not self.active or self.condition=='control':return result
        px,py,ph,valid=result
        if self.condition.startswith('loss'):return px,py,ph,False
        # Deliberately wrong yet valid localization: +0.5 m lateral, +0.15 rad yaw.
        return px-.5*np.sin(yaw),py+.5*np.cos(yaw),ph+.15,valid


def initialize():
    import torch
    import train_il_s2r as IL
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    args=IL.get_args(['--task_config','results/ppo_s2r/25/backup/f1tenth.yaml'])
    rms=RunningMeanStd('agent_obs',args.obs_dim);rms.load('results/imitation/1607/sim2real',required=True)
    for tag in ('440000','3080000'):
        ck=torch.load(f'results/ppo_s2r/25/checkpoint/model_{tag}.pt',map_location='cpu',weights_only=False)
        actor=IL.actor_from_state(ck['actor']);actor.eval()
        POLICIES[tag]=(actor,rms)


def drive(job):
    import torch
    import train_il_s2r as IL
    from utils.env import F1Wrapper,unnormalize_speed
    args=IL.get_args([]);args.seed=job['seed']
    np.random.seed(args.seed);random.seed(args.seed);torch.manual_seed(args.seed)
    actor,rms=POLICIES[job['tag']]
    env=F1Wrapper(args,IL.MAPS,obs_noise=True,sim_mode=job.get('mode','eval'),tracks=[job['path']])
    env.fixed_speed_cap=8.;env.start_mode='center';obs,_=env.reset(options={'track':job['path']})
    fault=PreviewFault(env.loc_noise.step,env,job['phase'],job['condition'])
    if not job.get('bare_control'):env.loc_noise.step=fault.step
    digest=hashlib.sha256();transitions=[];previous_action=None;previous_active=False
    invalid_steps=0;recover_after=None;peak_lateral=0.;peak_heading=0.
    try:
        while True:
            with torch.no_grad():act=actor(torch.as_tensor(rms.normalize(obs),dtype=torch.float32)).loc.numpy()
            act=np.clip(act,-1,1)
            digest.update(obs.tobytes());digest.update(env.vehicle_state().tobytes());digest.update(act.tobytes())
            if obs[-1]<.5:
                assert np.all(obs[-env.preview.dim:]==0), 'Invalid preview must be raw zero'
                invalid_steps+=1
            if fault.active!=previous_active:
                transitions.append(dict(time=float(env.elapsed),active=bool(fault.active),
                    steer_command_change_rad=(float(abs(act[0]-previous_action[0]))*.5 if previous_action is not None else None),
                    speed_command_change_mps=(float(abs(unnormalize_speed(act[1],.5,8.)-unnormalize_speed(previous_action[1],.5,8.))) if previous_action is not None else None)))
                if not fault.active:recover_after=float(env.elapsed)
            previous_action=act.copy();previous_active=fault.active
            obs,_,term,trunc,_=env.step(act)
            if recover_after is not None and env.elapsed-recover_after<=1.:
                geometry=env.driving_geometry()
                peak_lateral=max(peak_lateral,abs(geometry['lateral']))
                peak_heading=max(peak_heading,abs(geometry['heading']))
            if term or trunc:break
        return dict(job,success=bool(env.success),collision=bool(env.collision),elapsed=float(env.elapsed),
                    observation_action_state_sha256=digest.hexdigest(),triggered=fault.started is not None,
                    injected_start_time=fault.started,injected_duration=fault.duration,invalid_input_steps=invalid_steps,
                    injected_true_pose=fault.trigger_pose,injected_reference_s_m=fault.trigger_s,
                    restoration_observed=recover_after is not None,transitions=transitions,
                    post_restoration_observed_seconds=min(1.,float(env.elapsed)-recover_after) if recover_after is not None else 0.,
                    first_second_after_restoration_max_lateral_m=peak_lateral if recover_after is not None else None,
                    first_second_after_restoration_max_heading_rad=peak_heading if recover_after is not None else None)
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seeds',type=int,nargs='+',default=[7])
    p.add_argument('--mode',choices=['eval','worst'],default='eval')
    p.add_argument('--conditions',nargs='+',choices=['control','loss05','loss2','loss4','loss8','wrong_pose1'],
                   default=['control','loss05','loss2','wrong_pose1'])
    p.add_argument('--verify_noop',action='store_true')
    cli=p.parse_args();os.chdir(SRC)
    initialize()
    from utils.training_mix import hall_split
    from utils.track_short import short_tracks
    hall=hall_split()[1];short=short_tracks(True)
    tracks=[hall[0],hall[len(hall)//2],short[0],short[len(short)//2]]
    tracks+=sorted(str(p) for p in Path('installation/maps_real').glob('Practice1126b*.npz'))
    tracks=[str(Path(p).resolve()) for p in tracks]
    dest=Path(cli.output)
    if dest.exists():raise FileExistsError('Do not overwrite previous evidence')
    if cli.verify_noop:
        job=dict(path=tracks[-2],seed=17,tag='440000',phase='entry',condition='control')
        a,b=drive(job),drive(dict(job,bare_control=True))
        assert a['observation_action_state_sha256']==b['observation_action_state_sha256']
        assert a['success']==b['success'] and a['elapsed']==b['elapsed']
        dest.write_text(json.dumps(dict(status='passed',bitwise_noop_equivalence=True,steps_or_outcome='entire_two_lap_rollout',control=a),indent=2)+'\n')
        print('NOOP EQUIVALENCE PASSED',flush=True);return
    jobs=[dict(path=path,seed=seed+i,tag=tag,phase=phase,condition=condition,mode=cli.mode)
          for seed in cli.seeds for i,path in enumerate(tracks) for tag in POLICIES
          for phase in ('entry','straight') for condition in cli.conditions]
    result=dict(status='running',total=len(jobs),runs=[],seeds=cli.seeds,mode=cli.mode,
                checkpoints={tag:hashlib.sha256(Path(f'results/ppo_s2r/25/checkpoint/model_{tag}.pt').read_bytes()).hexdigest() for tag in POLICIES},
                track_sha256={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in tracks},
                config_sha256=hashlib.sha256(Path('results/ppo_s2r/25/backup/f1tenth.yaml').read_bytes()).hexdigest(),
                dynamics_sha256=hashlib.sha256(Path('configs/task/dynamic.yaml').read_bytes()).hexdigest(),
                qualification='Synthetic observation faults, same-policy control only. No ScanLocalizer, no separate fallback12190, no measured relocalization latency. Fault restoration is prescribed. Cases without triggered faults do not prove resilience.',
                trigger='First entry ahead curvature>.45 withcurrent<.45, or ahead curvature<.08 forstraight, after2s. Phase conditioned: physical trigger position can differ by policy. Compare faults with each policy own matched control; not an exact common-location cross-policy experiment.')
    with mp.get_context('spawn').Pool(cli.workers,initializer=initialize) as pool:
        for row in pool.imap_unordered(drive,jobs):
            result['runs'].append(row)
            dest.write_text(json.dumps(result,indent=2)+'\n')
            if len(result['runs'])%8==0 or not row['success']:
                print(len(result['runs']),'/',len(jobs),row['tag'],Path(row['path']).name,row['phase'],row['condition'],
                      'triggered',row['triggered'],'complete',row['success'],flush=True)
    result['status']='complete';dest.write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',len(jobs),flush=True)


if __name__=='__main__':main()
