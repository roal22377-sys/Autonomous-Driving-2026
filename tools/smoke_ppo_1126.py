#!/usr/bin/env python3
"""Actual PPO updates/resume, synthetic scoring; NEVER a policy performance test."""
import os,sys,json,multiprocessing as mp
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def main():
    os.chdir(S)
    import torch
    from ruamel.yaml import YAML
    import train_ppo_s2r as PPO
    from utils.vectorize import RunningMeanStd
    source=Path('results/imitation/14/checkpoint/model_init.pt')
    ck=torch.load(source,map_location='cpu',weights_only=False)
    model=Path('results/imitation/code_smoke_1126')
    for folder in ['backup','checkpoint','sim2real']:(model/folder).mkdir(parents=True,exist_ok=True)
    yaml=YAML();cfg=yaml.load(Path('configs/task/f1tenth_preview_1126.yaml'))
    cfg['training_mix']={k:.2 for k in cfg['training_mix']}
    yaml.dump(cfg,model/'backup/f1tenth.yaml')
    torch.save({'actor':ck['actor']},model/'checkpoint/model.pt')
    rms=RunningMeanStd('agent_obs',132);rms.mean=ck['obs_mean'];rms.var=ck['obs_var'];rms.count=1200000
    rms.save(str(model/'sim2real'))
    calls=[]

    def fake_evaluate(args,policy,tracks,mode='eval',settings=PPO.IL.EVAL_SETTINGS):
        calls.append(len(tracks))
        time=100. if len(calls)<=5 else 99.
        rows=[]
        for caps,start in settings:
            for i,path in enumerate(sorted(tracks)):
                cap=caps[i%len(caps)] if isinstance(caps,(list,tuple)) else caps
                rows.append((f'{Path(path).stem}@{cap:g}/{start}',True,time,.01,{}))
        return 1.,time,rows

    PPO.IL.evaluate=fake_evaluate
    argv=[str(__file__),'--il','imitation/code_smoke_1126','--name','code_smoke_1126','--algo_idx','1',
          '--workers','5','--eval_workers','1','--n_steps','500','--total_steps','1000','--epochs','1',
          '--minibatch','250','--eval_every','1','--ckpt_every','1','--short_eval','1','--worst_eval','0',
          '--critic_warmup','0','--max_warmup','0','--warmup_ev=-1000000000','--cost_critic','1',
          '--crash_adapt','0','--collision_cost','5000','--log_std_steer','-3.4','--log_std_speed','-2.6',
          '--straight_heading_cost','.5','--straight_offset_cost','2','--lat_grip_share','.9']
    print('CODE SMOKE: synthetic evaluation scores; do not deploy outputs or use as performance evidence',flush=True)
    sys.argv=argv;PPO.main()
    root=Path('results/code_smoke_1126/1')
    st=torch.load(root/'checkpoint/resume.pt',map_location='cpu',weights_only=False)
    assert st['update']==2 and st['best']['name']=='update 1' and 'families' in st['best']
    assert st['il']['hall_res'][0][2]==100. and st['best']['families']['hall']['res'][0][2]==99.
    before=len(calls)
    sys.argv=argv.copy();sys.argv[sys.argv.index('--total_steps')+1]='1500';sys.argv.append('--resume')
    PPO.main()
    st=torch.load(root/'checkpoint/resume.pt',map_location='cpu',weights_only=False)
    assert st['update']==3 and len(calls)-before==5
    assert st['il']['hall_res'][0][2]==100. and st['il']['open_res'][0][2]==100.
    assert all(torch.isfinite(v).all() for net in ('actor','critic','ccritic') for v in st[net].values())
    result=dict(result='passed',real_rollout_steps=1500,updates=3,resumed=True,
                qualification='Synthetic scoring only; tested PPO updates, family selection, checkpoint promotion and original IL baseline preservation.')
    Path('results/preflight_1126/ppo_update_resume_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SMOKE PASSED',result,flush=True)


if __name__=='__main__':main()
