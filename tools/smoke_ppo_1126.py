#!/usr/bin/env python3
"""Actual PPO updates/resume, synthetic scoring; NEVER a policy performance test."""
import os,sys,json,multiprocessing as mp
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def main():
    os.chdir(S)
    import torch,numpy as np
    from ruamel.yaml import YAML
    import train_ppo_s2r as PPO
    from utils.vectorize import RunningMeanStd
    source=Path('results/imitation/14/checkpoint/model_init.pt')
    ck=torch.load(source,map_location='cpu',weights_only=False)
    model=Path('results/imitation/code_smoke_eval_archive_1126')
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
        time=100. if len(calls)<=6 else 99.
        rows=[]
        for caps,start in settings:
            for i,path in enumerate(sorted(tracks)):
                cap=caps[i%len(caps)] if isinstance(caps,(list,tuple)) else caps
                rows.append((f'{Path(path).stem}@{cap:g}/{start}',True,time,.01,{}))
        return 1.,time,rows

    PPO.IL.evaluate=fake_evaluate
    argv=[str(__file__),'--il','imitation/code_smoke_eval_archive_1126','--name','code_smoke_eval_archive_1126','--algo_idx','1',
          '--workers','5','--eval_workers','1','--n_steps','500','--total_steps','1000','--epochs','1',
          '--minibatch','250','--eval_every','1','--ckpt_every','1','--short_eval','1','--worst_eval','1','--lost_eval','1',
          '--critic_warmup','0','--max_warmup','0','--warmup_ev=-1000000000','--cost_critic','1',
          '--crash_adapt','0','--collision_cost','5000','--log_std_steer','-3.4','--log_std_speed','-2.6',
          '--straight_heading_cost','.5','--straight_offset_cost','2','--lat_grip_share','.9',
          '--lat_acc_mode','path_normal','--straight_back','1','--straight_entry_reset','1']
    print('CODE SMOKE: synthetic evaluation scores; do not deploy outputs or use as performance evidence',flush=True)
    sys.argv=argv;PPO.main()
    root=Path('results/code_smoke_eval_archive_1126/1')
    st=torch.load(root/'checkpoint/resume.pt',map_location='cpu',weights_only=False)
    assert st['update']==2 and st['best']['name']=='update 1' and 'families' in st['best']
    assert st['il']['hall_res'][0][2]==100. and st['best']['families']['hall']['res'][0][2]==99.
    before=len(calls)
    sys.argv=argv.copy();sys.argv[sys.argv.index('--total_steps')+1]='1500';sys.argv.append('--resume')
    PPO.main()
    st=torch.load(root/'checkpoint/resume.pt',map_location='cpu',weights_only=False)
    assert st['update']==3 and len(calls)-before==8
    assert st['il']['hall_res'][0][2]==100. and st['il']['open_res'][0][2]==100.
    assert all(torch.isfinite(v).all() for net in ('actor','critic','ccritic') for v in st[net].values())
    updates=[row for row in st['log'] if row['update']>0]
    assert len(updates)==3 and all(np.isfinite(row['slip_rms_rad']) and
                                  np.isfinite(row['slip_abs_p99_rad']) for row in updates)
    for row in updates:
        for key in ('res','worst_res','open_res','short_res','hall_res','real_res','lost_short_res','lost_eval_res'):
            assert row[key], key
            track_order=row['evaluation_track_order'][key]
            assert track_order==list(dict.fromkeys(r[0].split('@',1)[0] for r in row[key]))
            assert row['evaluation_seed_base']==7
    print('ALL EIGHT EVALUATION CASE COHORTS RETAINED THROUGH RESUME',flush=True)
    result=dict(result='passed',real_rollout_steps=1500,updates=3,resumed=True,
                lat_acc_mode='path_normal',stability_logging=True,evaluation_case_cohorts=8,
                qualification='Synthetic scoring only; tested PPO updates, family selection, checkpoint promotion and original IL baseline preservation.')
    Path('results/preflight_1126/ppo_eval_case_archive_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SMOKE PASSED',result,flush=True)


if __name__=='__main__':main()
