#!/usr/bin/env python3
"""Replay one checkpoint/case and retain state, commands and shadow-expert labels."""
import argparse,csv,json,os,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='imitation/14');p.add_argument('--tag',default='round0')
    p.add_argument('--normalizer_model',help='Frozen normalizer source for a checkpoint without its own sim2real export')
    p.add_argument('--track',required=True);p.add_argument('--cap',type=float,default=8.)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--start',default='center',choices=['center','random'])
    p.add_argument('--diagnostic_override',choices=['expert_speed','expert_steer','expert_both'],
                   help='privileged counterfactual for diagnosis; never a policy qualification')
    p.add_argument('--audit_preview',action='store_true',help='record the projection index and actual preview inputs')
    cli=p.parse_args();os.chdir(S)
    import numpy as np,torch,random
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ruamel.yaml import YAML
    import train_il_s2r as IL
    from utils.env import F1Wrapper,unnormalize_speed,normalize_speed
    from utils.rule_based import RacelinePlanner
    from utils.vectorize import RunningMeanStd
    from utils.driving_metrics import DrivingMetrics
    torch.set_num_threads(1);np.random.seed(cli.seed);random.seed(cli.seed)
    args=IL.get_args(['--task_config',f'results/{cli.model}/backup/f1tenth.yaml']);args.seed=cli.seed
    ckpath=Path(f'results/{cli.model}/checkpoint/model{("_"+cli.tag) if cli.tag!="best" else ""}.pt')
    if not ckpath.exists() and cli.tag=='best':
        ckpath=Path(f'results/{cli.model}/sim2real/actor.pt')
        ck={'actor':torch.load(ckpath,map_location='cpu',weights_only=False).state_dict()}
    else:ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    rms=RunningMeanStd('agent_obs',args.obs_dim)
    if 'obs_mean' in ck:rms.mean,rms.var=ck['obs_mean'],ck['obs_var']
    else:rms.load(f'results/{cli.normalizer_model or cli.model}/sim2real',required=True)
    actor=IL.actor_from_state(ck['actor']);actor.eval()
    path=str(Path(cli.track).resolve())
    env=F1Wrapper(args,IL.MAPS,obs_noise=True,sim_mode='eval',tracks=[path])
    # Only the shadow labeler uses the new expert. The tested policy keeps its own observations.
    env.expert_cfg=dict(YAML().load(Path('configs/task/f1tenth_preview_1126.yaml'))['expert_plan'])
    env.fixed_speed_cap=cli.cap;env.start_mode=cli.start;obs,_=env.reset(options={'track':path})
    planner=RacelinePlanner(env._env.unwrapped.track,args.length_f+args.length_r,v_min=env.min_speed,
                           v_max=env.speed_cap,max_steer=env.max_steer,max_steer_left=env.steer_left_max,
                           max_steer_right=env.steer_right_max,**IL.expert_kwargs(env))
    from utils.policy_transfer import columns
    preview_columns=[(i,name) for i,name in enumerate(columns(args)) if name.startswith('preview/')]
    if cli.audit_preview and env.preview is None:raise ValueError('--audit_preview requires track preview')
    previous_i=getattr(env.preview,'last_i',None);previous_valid=bool(getattr(env,'_pv',True))
    rows=[];metrics=DrivingMetrics()
    while True:
        with torch.no_grad():act=actor(torch.as_tensor(rms.normalize(obs),dtype=torch.float32)).loc.numpy()
        act=np.clip(act,-1,1);sq,vq=env.pending_commands()
        ev,ed=planner.plan(env.vehicle_state(),sq,vq,env.timestep)
        command_time=float(env.elapsed);valid=bool(getattr(env,'_pv',True))
        audit={}
        if cli.audit_preview:
            current_i=getattr(env.preview,'last_i',None);jump=float('nan')
            if valid and previous_valid and current_i is not None and previous_i is not None:
                jump=((current_i-previous_i+env.preview.n//2)%env.preview.n-env.preview.n//2)*env.preview.ds
            audit=dict(observed_preview_index=current_i if current_i is not None else float('nan'),observed_preview_delta_s=jump)
            audit.update({'observed_'+name:float(obs[i]) for i,name in preview_columns})
            previous_i,previous_valid=current_i,valid
        applied=act.copy()
        if cli.diagnostic_override in ('expert_speed','expert_both'):
            applied[1]=normalize_speed(ev,env.min_speed,env.max_speed)
        if cli.diagnostic_override in ('expert_steer','expert_both'):
            applied[0]=env.steer_command(ed)
        side=0 if act[0]>0 else 1;k=env.steer_scale[side];lock=env.steer_left_max if side==0 else env.steer_right_max
        target=float(act[0]*lock*(k+(1-k)*act[0]**2))
        obs,_,term,trunc,info=env.step(applied);state=env.vehicle_state();g=env.driving_geometry();metrics.update(env,applied)
        rows.append(dict(command_time=command_time,state_time=float(env.elapsed),x=float(state[0]),y=float(state[1]),
                         speed=float(state[3]),policy_speed=min(float(unnormalize_speed(act[1],env.min_speed,env.max_speed)),env.speed_cap),
                         expert_speed=float(ev),policy_steer=float(act[0]),policy_wheel_target=target,wheel_actual=float(state[2]),
                         expert_wheel_target=float(ed),expert_steer_command=env.steer_command(ed),lateral=g['lateral'],
                         heading=g['heading'],curvature=g['curvature'],lat_acc=float(env.lat_acc),preview_valid=int(valid),
                         applied_steer=float(applied[0]),applied_speed=float(unnormalize_speed(applied[1],env.min_speed,env.max_speed)),**audit))
        if term or trunc:break
    dest=Path('results/analysis1126/traces');dest.mkdir(parents=True,exist_ok=True)
    stem=f'{cli.model.replace("/","_")}_{cli.tag}_{Path(path).stem}_cap{cli.cap:g}_{cli.start}_seed{cli.seed}'
    if cli.diagnostic_override:stem+='_'+cli.diagnostic_override
    if cli.audit_preview:stem+='_preview_audit'
    with open(dest/(stem+'.csv'),'w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(model=cli.model,tag=cli.tag,track=Path(path).name,cap=cli.cap,start=cli.start,seed=cli.seed,
                 normalizer_source='embedded_checkpoint' if 'obs_mean' in ck else (cli.normalizer_model or cli.model),
                 success=bool(env.success),collision=bool(env.collision),elapsed=float(env.elapsed),phases=metrics.result(),
                 diagnostic_override=cli.diagnostic_override,policy_qualification=not bool(cli.diagnostic_override),
                 progress_m=float(env.progress),lap_times=[float(t) for t in env.lap_times],
                 checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),
                 shadow_expert='6.0 nominal reference, corner lookahead .65, response inversion; diagnostic only')
    if cli.audit_preview:
        jumps=np.array([r['observed_preview_delta_s'] for r in rows]);jumps=jumps[np.isfinite(jumps)]
        summary['preview_projection_audit']=dict(maximum_step_m=float(np.abs(jumps).max()) if len(jumps) else None,steps_over_1m=int((np.abs(jumps)>1.).sum()),
             note='Projection index and raw preview values actually presented to the policy; logging does not alter observations.')
    (dest/(stem+'.json')).write_text(json.dumps(summary,indent=2)+'\n');env.close()
    data={k:np.array([r[k] for r in rows]) for k in rows[0]};t=data['state_time']
    fig,axes=plt.subplots(2,2,figsize=(13,8));ax=axes[0,0]
    with np.load(path) as z:
        occ,org,res=z['occupancy'],z['origin'],float(z['resolution']);h,w=occ.shape
        ax.imshow(occ,origin='lower',cmap='gray',extent=[org[0],org[0]+w*res,org[1],org[1]+h*res])
        line=z['raceline'];ax.plot(line[:,0],line[:,1],color='0.6',lw=.8)
    ax.plot(data['x'],data['y'],lw=1.2);ax.scatter(data['x'][-1],data['y'][-1],c='green' if summary['success'] else 'red')
    ax.set_aspect('equal');ax.set_title('Trajectory and final position');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
    ax=axes[0,1]
    for key,label in [('speed','actual'),('policy_speed','policy (capped, before start kick)'),('expert_speed','shadow expert')]:ax.plot(t,data[key],label=label,lw=.9)
    ax.set_ylabel('speed [m/s]');ax.legend(fontsize=8)
    ax=axes[1,0]
    for key,label in [('wheel_actual','actual'),('policy_wheel_target','policy target before delay'),('expert_wheel_target','shadow expert')]:ax.plot(t,data[key],label=label,lw=.8)
    ax.set_ylabel('wheel angle [rad]');ax.legend(fontsize=8)
    ax=axes[1,1];ax.plot(t,data['lateral'],label='lateral error [m]');ax.plot(t,data['heading'],label='heading error [rad]',alpha=.7)
    ax.axhline(0,color='0.5',lw=.5);ax.legend(fontsize=8)
    for ax in [axes[0,1],axes[1,0],axes[1,1]]:ax.set_xlabel('state time [s]');ax.grid(alpha=.2)
    fig.suptitle(f'{cli.model} {cli.tag}: {Path(path).stem}, cap {cli.cap:g}, seed {cli.seed}: '+('complete' if summary['success'] else 'failed'))
    fig.tight_layout();fig.savefig(dest/(stem+'.png'),dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='phases'}));print(dest/stem)


if __name__=='__main__':main()
