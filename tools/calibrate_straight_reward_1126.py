#!/usr/bin/env python3
"""Measure straight penalties along identical expert trajectories before PPO."""
import os,sys,json,csv,multiprocessing as mp
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]
SCHEMES={'future_only':(0.,False),'entry_reset':(0.,True),'past05_reset':(.5,True),
         'past10_reset':(1.,True),'past15_reset':(1.5,True)}


def drive(job):
    os.chdir(S)
    import numpy as np,torch
    import train_il_s2r as IL
    from utils.env import F1Wrapper,normalize_speed
    from utils.rule_based import RacelinePlanner
    torch.set_num_threads(1);np.random.seed(0)
    path,cap,production=job;args=IL.get_args(['--task_config','configs/task/f1tenth_preview_1126.yaml']);args.seed=0
    env=F1Wrapper(args,IL.MAPS,obs_noise=True,sim_mode='eval',tracks=[path])
    env.fixed_speed_cap=cap;env.start_mode='center'
    env.PROGRESS_COEFF=10.;env.STRAIGHT_STEER_COST=5.;env.STRAIGHT_V_REF=4.
    env.STRAIGHT_HEADING_COST=.5;env.STRAIGHT_OFFSET_COST=2.;env.STEER_RATE_COST=3.
    env.STRAIGHT_BACK=1. if production else 0.;env.STRAIGHT_ENTRY_RESET=production
    env.reset(options={'track':path})
    planner=RacelinePlanner(env._env.unwrapped.track,args.length_f+args.length_r,v_min=env.min_speed,v_max=env.speed_cap,
                           max_steer=env.max_steer,max_steer_left=env.steer_left_max,max_steer_right=env.steer_right_max,
                           **IL.expert_kwargs(env))
    state={k:dict(ema=0.,gate=False,cost=0.,distance=0.,max_ratio=0.,samples=0) for k in SCHEMES}
    rows=[];distance=0.
    while True:
        sq,vq=env.pending_commands();v,d=planner.plan(env.vehicle_state(),sq,vq,env.timestep)
        act=np.clip([env.steer_command(d),normalize_speed(v,env.min_speed,env.max_speed)],-1,1)
        _,_,term,trunc,info=env.step(act);g=env.driving_geometry();line=env._line_ref();j=env._reward_line_hint
        ds=max(0.,float(env.delta_s));distance+=ds
        row=dict(t=float(env.elapsed),v=float(env.vehicle_state()[3]),steer=float(act[0]),ds=ds,
                 curvature=g['curvature'],heading=g['heading'],lateral=g['lateral'])
        for name,(back,reset) in SCHEMES.items():
            st=state[name]
            behind=(j-np.arange(int(round(back/line.ds))+1))%line.n
            gate=g['future_curvature']<.08 and (back==0 or float(np.max(np.abs(line.k[behind])))<.08)
            if reset and gate and not st['gate']:st['ema']=float(act[0])
            st['ema']+=(float(act[0])-st['ema'])*min(1.,env.timestep/env.STRAIGHT_EMA_TIME)
            cost=5.*(float(act[0])-st['ema'])**2*ds/(4.*env.timestep) if gate else 0.
            st['cost']+=cost;st['distance']+=ds if gate else 0.;st['samples']+=int(gate)
            if ds>.001:st['max_ratio']=max(st['max_ratio'],cost/(10.*ds))
            st['gate']=gate;row[name+'_cost']=cost;row[name+'_gate']=int(gate)
            if name==('past10_reset' if production else 'future_only'):
                measured=float(info['straight_cost'])-min(.1,float(info['straight_heading_cost'])+float(info['straight_offset_cost']))
                assert abs(measured-cost)<1e-5,(measured,cost)
        rows.append(row)
        if term or trunc:break
    result=dict(track=Path(path).name,cap=cap,complete=bool(env.success),time=float(env.elapsed),distance=distance,
                schemes={k:dict(cost=v['cost'],straight_distance=v['distance'],coverage=v['distance']/distance,
                                cost_per_straight_m=v['cost']/v['distance'] if v['distance'] else None,
                                max_cost_over_progress=v['max_ratio'],samples=v['samples']) for k,v in state.items()})
    out=Path('results/preflight_1126/straight_calibration');out.mkdir(exist_ok=True)
    with (out/f'{Path(path).stem}_cap{cap:g}{"_production" if production else ""}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    env.close();return result


def main():
    os.chdir(S);os.environ['F1_TASK_CONFIG']='configs/task/f1tenth_preview_1126.yaml'
    paths=sorted((S/'installation/maps_real').glob('Practice1126b*.npz'))+sorted((S/'installation/maps_hall').glob('Course2025*.npz'))
    production='--production' in sys.argv
    jobs=[(str(p),cap,production) for p in paths for cap in [8.,3.]]
    with mp.get_context('spawn').Pool(2) as pool:results=pool.map(drive,jobs)
    summary={}
    for name in SCHEMES:
        a=[r['schemes'][name] for r in results];dist=sum(x['straight_distance'] for x in a)
        summary[name]=dict(total_cost=sum(x['cost'] for x in a),straight_distance=dist,
                           coverage=dist/sum(r['distance'] for r in results),
                           cost_per_straight_m=sum(x['cost'] for x in a)/dist if dist else None,
                           maximum_cost_over_progress=max(x['max_cost_over_progress'] for x in a))
    out=dict(qualification='Expert reward calibration, not a trained-policy qualification',complete=sum(r['complete'] for r in results),total=len(results),
             settings=dict(coefficient=5.,v_ref=4.,nominal_lateral=6.,threshold=.08,production=production,lookbehind_sampling='dense'),summary=summary,results=results)
    Path(f'results/preflight_1126/straight_reward_calibration{"_production" if production else ""}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__':main()
