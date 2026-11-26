#!/usr/bin/env python3
"""Measure straight penalties along identical expert trajectories before PPO."""
import os,sys,json,csv,multiprocessing as mp
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]
SCHEMES={'future_only':(0.,False),'entry_reset':(0.,True),'past05_reset':(.5,True),
         'past10_reset':(1.,True),'past15_reset':(1.5,True)}
FULL_REWARD=dict(LINE_COST=10.,LINE_MARGIN=.2,LAT_COST=.5,LAT_GRIP_SHARE=.9,
                 WALL_COST=5.,WALL_MARGIN=.3,TIME_COST=.1,BAY_COST=2.)


def drive(job):
    os.chdir(S)
    import numpy as np,torch
    import train_il_s2r as IL
    from utils.env import F1Wrapper,normalize_speed
    from utils.rule_based import RacelinePlanner
    torch.set_num_threads(1);np.random.seed(0)
    path,cap,production,full_reward,lat_mode=job;args=IL.get_args(['--task_config','configs/task/f1tenth_preview_1126.yaml']);args.seed=0
    env=F1Wrapper(args,IL.MAPS,obs_noise=True,sim_mode='eval',tracks=[path])
    env.fixed_speed_cap=cap;env.start_mode='center'
    env.PROGRESS_COEFF=10.;env.STRAIGHT_STEER_COST=5.;env.STRAIGHT_V_REF=4.
    env.STRAIGHT_HEADING_COST=.5;env.STRAIGHT_OFFSET_COST=2.;env.STEER_RATE_COST=3.
    env.STRAIGHT_BACK=1. if production else 0.;env.STRAIGHT_ENTRY_RESET=production
    if full_reward:
        for key,value in FULL_REWARD.items():setattr(env,key,value)
    env.LAT_ACC_MODE=lat_mode
    env.reset(options={'track':path})
    planner=RacelinePlanner(env._env.unwrapped.track,args.length_f+args.length_r,v_min=env.min_speed,v_max=env.speed_cap,
                           max_steer=env.max_steer,max_steer_left=env.steer_left_max,max_steer_right=env.steer_right_max,
                           **IL.expert_kwargs(env))
    state={k:dict(ema=0.,gate=False,cost=0.,distance=0.,max_ratio=0.,samples=0) for k in SCHEMES}
    rows=[];distance=0.
    agent=env._env.unwrapped.sim.agents[0]
    from utils.driving_physics import path_normal_acceleration
    while True:
        sq,vq=env.pending_commands();v,d=planner.plan(env.vehicle_state(),sq,vq,env.timestep)
        act=np.clip([env.steer_command(d),normalize_speed(v,env.min_speed,env.max_speed)],-1,1)
        before=env.vehicle_state()
        _,_,term,trunc,info=env.step(act);g=env.driving_geometry();line=env._line_ref();j=env._reward_line_hint
        ds=max(0.,float(env.delta_s));distance+=ds
        row=dict(t=float(env.elapsed),v=float(env.vehicle_state()[3]),steer=float(act[0]),ds=ds,
                 curvature=g['curvature'],heading=g['heading'],lateral=g['lateral'])
        if full_reward:
            row.update({key:float(info[key]) for key in ('forward_reward','steer_cost','straight_cost',
                       'straight_heading_cost','straight_offset_cost','line_cost','lat_cost','wall_cost','time_cost','bay_cost')})
            row.update(lat_acc=float(env.lat_acc),grip=float(env.params['mu'])*9.81)
            row['lat_cost_share08']=.5*max(0.,abs(row['lat_acc'])-.8*row['grip'])
            after=env.vehicle_state();mid=(before+after)*.5
            theta0,theta1=before[4]+before[6],after[4]+after[6]
            theta=theta0+.5*np.arctan2(np.sin(theta1-theta0),np.cos(theta1-theta0))
            velocity0=before[3]*np.array([np.cos(theta0),np.sin(theta0)])
            velocity1=after[3]*np.array([np.cos(theta1),np.sin(theta1)])
            normal_fd=float(np.dot((velocity1-velocity0)/env.timestep,[-np.sin(theta),np.cos(theta)]))
            row.update(normal_acc=path_normal_acceleration(agent,after),normal_acc_mid=path_normal_acceleration(agent,mid),
                       normal_acc_fd=normal_fd,slip=float(after[6]))
            row['lat_cost_normal']=.5*max(0.,abs(row['normal_acc'])-.9*row['grip'])
            if lat_mode=='path_normal':assert abs(row['lat_cost']-row['lat_cost_normal'])<1e-6
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
    if full_reward:
        keys=('forward_reward','steer_cost','straight_cost','straight_heading_cost','straight_offset_cost',
              'line_cost','lat_cost','wall_cost','time_cost','bay_cost','lat_cost_share08','lat_cost_normal')
        result['reward_totals']={key:sum(r[key] for r in rows) for key in keys}
        result['peak_lat_acc']=max(abs(r['lat_acc']) for r in rows)
        moving=[r for r in rows if r['v']>.6]
        result['acceleration_diagnostic']=dict(peak_normal_acc=max(abs(r['normal_acc']) for r in rows),
            normal_mid_vs_velocity_fd_rms=float(np.sqrt(np.mean([(r['normal_acc_mid']-r['normal_acc_fd'])**2 for r in moving]))),
            proxy_vs_normal_rms=float(np.sqrt(np.mean([(r['lat_acc']-r['normal_acc'])**2 for r in moving]))),
            reward_mode=lat_mode,note='Path-normal acceleration from the simulator derivative, cross-checked against world velocity differences.')
    out=Path('results/preflight_1126/straight_calibration');out.mkdir(exist_ok=True)
    suffix=('_production' if production else '')+('_full_reward' if full_reward else '')
    if lat_mode!='yaw_proxy':suffix+='_'+lat_mode
    with (out/f'{Path(path).stem}_cap{cap:g}{suffix}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    env.close();return result


def main():
    os.chdir(S);os.environ['F1_TASK_CONFIG']='configs/task/f1tenth_preview_1126.yaml'
    paths=sorted((S/'installation/maps_real').glob('Practice1126b*.npz'))+sorted((S/'installation/maps_hall').glob('Course2025*.npz'))
    production='--production' in sys.argv
    full_reward='--full_reward' in sys.argv
    lat_mode='path_normal' if '--path_normal' in sys.argv else 'yaw_proxy'
    if full_reward and not production:raise ValueError('--full_reward requires --production')
    jobs=[(str(p),cap,production,full_reward,lat_mode) for p in paths for cap in [8.,3.]]
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
    if full_reward:
        totals={key:sum(r['reward_totals'][key] for r in results) for key in results[0]['reward_totals']}
        out['full_reward']=dict(settings=FULL_REWARD,totals=totals,
                                cost_to_progress={key:value/totals['forward_reward'] for key,value in totals.items() if key!='forward_reward'})
    suffix=('_production' if production else '')+('_full_reward' if full_reward else '')
    if lat_mode!='yaw_proxy':suffix+='_'+lat_mode
    Path(f'results/preflight_1126/straight_reward_calibration{suffix}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__':main()
