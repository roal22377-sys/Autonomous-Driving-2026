#!/usr/bin/env python3
"""Audit the changed IL expert on the full original/open validation cohorts."""
import argparse,json,multiprocessing as mp,os,sys,time,hashlib
from collections import deque
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def drive(job):
    import numpy as np,torch,random
    import train_il_s2r as IL
    from utils.env import F1Wrapper,normalize_speed
    from utils.rule_based import RacelinePlanner
    group,path,seed,cap,start=job;torch.set_num_threads(1);np.random.seed(seed);random.seed(seed)
    args=IL.get_args([]);args.seed=seed
    env=F1Wrapper(args,IL.MAPS,obs_noise=True,sim_mode='eval',tracks=[path])
    env.fixed_speed_cap=cap;env.start_mode=start
    try:
        env.reset(options={'track':path})
        planner=RacelinePlanner(env._env.unwrapped.track,args.length_f+args.length_r,v_min=env.min_speed,
                 v_max=env.speed_cap,max_steer=env.max_steer,max_steer_left=env.steer_left_max,
                 max_steer_right=env.steer_right_max,**IL.expert_kwargs(env))
        last=deque(maxlen=40)
        while True:
            sq,vq=env.pending_commands();v,d=planner.plan(env.vehicle_state(),sq,vq,env.timestep)
            action=np.clip([env.steer_command(d),normalize_speed(v,env.min_speed,env.max_speed)],-1,1)
            _,_,term,trunc,_=env.step(action)
            last.append([float(env.elapsed),float(action[0]),float(v)]+env.vehicle_state().tolist())
            if term or trunc:break
        return dict(family=group,track=Path(path).name,seed=seed,cap=cap,start=start,complete=bool(env.success),
                    time=float(env.elapsed),failure_tail=list(last) if not env.success else [])
    finally:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    cli=p.parse_args();os.chdir(S)
    import train_il_s2r as IL
    from eval_open_space import TEST_DIR
    config='configs/task/f1tenth_preview_1126.yaml';IL.get_args(['--task_config',config])
    regular=sorted(t for t in IL.list_variants(IL.MAPS) if IL.variant_map(t) in IL.EVAL_MAPS and
            (IL.HOLDOUT in t or Path(t).stem in [m+d for m in IL.EVAL_MAPS for d in ('','_rev')]))[:44]
    tracks=dict(regular=regular,open=sorted(str(p) for p in Path(TEST_DIR,'big').glob('*.npz')))
    settings=dict(regular=IL.EVAL_SETTINGS+(((2.,2.5,3.,3.5,4.),'center'),),open=((8.,'center'),(5.,'center')))
    jobs=[(g,str(Path(path).resolve()),7+i,cap[i%len(cap)] if isinstance(cap,tuple) else cap,start)
          for g,ps in tracks.items() for cap,start in settings[g] for i,path in enumerate(ps)]
    out=dict(qualification='Privileged expert feasibility only, not learned-policy performance',
             config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
             dynamics_sha256=hashlib.sha256(Path('configs/task/dynamic.yaml').read_bytes()).hexdigest(),
             total=len(jobs),runs=[],summary={})
    dest=Path('results/preflight_1126/expert_original_open_nominal.json');started=time.monotonic()
    with mp.get_context('spawn').Pool(cli.workers) as pool:
        for row in pool.imap_unordered(drive,jobs):
            out['runs'].append(row)
            if not row['complete']:print('EXPERT FAIL',json.dumps({k:v for k,v in row.items() if k!='failure_tail'}),flush=True)
            for g in tracks:
                rr=[r for r in out['runs'] if r['family']==g]
                out['summary'][g]=dict(complete=sum(r['complete'] for r in rr),total=len(rr))
            if len(out['runs'])%10==0 or not row['complete']:
                dest.write_text(json.dumps(out,indent=2)+'\n')
                print(len(out['runs']),'/',len(jobs),out['summary'],'elapsed',round(time.monotonic()-started),flush=True)
    out['status']='complete';dest.write_text(json.dumps(out,indent=2)+'\n')
    print('DONE',json.dumps(out['summary']),flush=True)


if __name__=='__main__':main()
