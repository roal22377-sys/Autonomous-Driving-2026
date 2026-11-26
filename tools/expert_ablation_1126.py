#!/usr/bin/env python3
import os,sys,json,multiprocessing as mp,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def run(job):
    name,path,mu,share,cap,inverse=job
    os.chdir(S)
    import numpy as np,torch
    import train_il_s2r as IL
    from utils.env import F1Wrapper,normalize_speed
    from utils.rule_based import RacelinePlanner
    torch.set_num_threads(1);np.random.seed(0)
    a=IL.get_args([]);a.mu=mu;a.seed=0
    a.expert_plan=dict(grip_share=share,steer_grip_share=share,steer_speed_mode='hold',corner_lad_cap=cap,
                       invert_steer_response=inverse)
    env=F1Wrapper(a,IL.MAPS,obs_noise=True,sim_mode='eval',tracks=[path]);env.fixed_speed_cap=8.;env.start_mode='center'
    env.reset(options={'track':path})
    p=RacelinePlanner(env._env.unwrapped.track,.34,v_min=env.min_speed,v_max=env.speed_cap,max_steer=env.max_steer,
                     max_steer_left=env.steer_left_max,max_steer_right=env.steer_right_max,**IL.expert_kwargs(env))
    last=[]
    while True:
        sq,vq=env.pending_commands();v,d=p.plan(env.vehicle_state(),sq,vq,env.timestep)
        action=np.clip([env.steer_command(d),normalize_speed(v,env.min_speed,env.max_speed)],-1,1)
        _,_,term,trunc,_=env.step(action)
        st=env.vehicle_state();g=env.driving_geometry()
        last.append([float(env.elapsed),float(st[3]),float(d),float(st[2]),g['lateral'],g['heading'],g['curvature']])
        if term or trunc:break
    result=dict(case=name,track=os.path.basename(path),ok=bool(env.success),time=float(env.elapsed),last=last[-20:])
    env.close();return result


def main():
    os.chdir(S);os.environ['F1_TASK_CONFIG']='configs/task/f1tenth_preview_1126.yaml'
    old=json.loads((S/'results/preflight_1126/expert_feasibility.json').read_text())['results']
    bad={x['track']+'.npz' for x in old if not x['ok']}
    full='--full' in sys.argv
    paths=[str(p.resolve()) for p in (S/'installation/maps_hall').glob('*.npz') if full or p.name in bad]
    cases=([(f'cap_{c}',.75,6/(.75*9.81),c) for c in [.75,.65,.55]] if full else
           [('old_grip',.65,.7,None),('lower_budget',.75,.7,None),('short_lad',.75,6/(.75*9.81),.75)])
    inverse='--inverse' in sys.argv
    jobs=[(n,p,mu,sh,c,inverse) for n,mu,sh,c in cases for p in paths]
    results=[];start=time.time()
    with mp.get_context('spawn').Pool(8) as pool:
        for row in pool.imap_unordered(run,jobs):
            results.append(row)
            if len(results)%20==0:print(len(results),'/',len(jobs),round(time.time()-start),'s',flush=True)
    suffix=('_full' if full else '')+('_inverse' if inverse else '')
    (S/f'results/preflight_1126/expert_ablation{suffix}.json').write_text(json.dumps(results,indent=2)+'\n')
    for n,_,_,_ in cases:
        q=[x for x in results if x['case']==n];print(n,sum(x['ok'] for x in q),'/',len(q),flush=True)


if __name__=='__main__':main()
