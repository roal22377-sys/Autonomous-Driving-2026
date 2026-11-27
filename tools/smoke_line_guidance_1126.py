from pathlib import Path
import os,sys,json,math
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth';sys.path[:0]=[str(S),str(S/'installation')]

def main():
 import argparse
 parser=argparse.ArgumentParser(description='Real-rollout PPO line-guidance schedule/resume smoke; evaluation scores are stubbed, not performance evidence.')
 parser.add_argument('--trial_id',type=int,required=True)
 parser.add_argument('--coupled',action='store_true')
 cli=parser.parse_args()
 os.chdir(S)
 import torch
 import train_ppo_s2r as P
 from utils.line_guidance import coefficient,validate,check_resume,imitation_multiplier
 assert [imitation_multiplier(10,v,True) for v in [10,5,0]]==[1.,.5,0.]
 assert imitation_multiplier(0,0,False)==1.
 try:check_resume({'line_cost':10},{'line_cost':10,'anneal_imitation_with_line':1})
 except ValueError:pass
 else:raise AssertionError('coupling changed on resume')
 for bad in [(10,11,40),(10,-1,40),(10,float('nan'),40),(10,0,0),(-1,None,40)]:
  try:validate(*bad)
  except ValueError:pass
  else:raise AssertionError('accepted invalid schedule')
 values=[coefficient(10,0,2,u,2 if u>=2 else None) for u in [0,1,2,3,4,50]]
 assert values==[10,10,5,0,0,0]
 assert all(coefficient(10,None,40,u,None)==10 for u in range(100))
 check_resume({'line_cost':10},{'line_cost':10,'line_cost_end':None,'line_cost_decay_updates':40})
 for key,value in [('line_cost',9),('line_cost_end',1),('line_cost_decay_updates',3)]:
  old=dict(line_cost=10,line_cost_end=0,line_cost_decay_updates=2);new=dict(old);new[key]=value
  try:check_resume(old,new)
  except ValueError:pass
  else:raise AssertionError('schedule change accepted')
 def synthetic(args,policy,tracks,mode='eval',settings=P.IL.EVAL_SETTINGS):
  rows=[]
  for cap,start in settings:
   for i,path in enumerate(sorted(tracks)):
    c=cap[i%len(cap)] if isinstance(cap,(tuple,list)) else cap
    rows.append((f'{Path(path).stem}@{c:g}/{start}',True,100.,.01,{}))
  return 1.,100.,rows
 P.IL.evaluate=synthetic
 folder=Path('results/code_smoke_line_guidance1126')/str(cli.trial_id)
 if folder.exists():raise FileExistsError('Preserve earlier smoke')
 argv=['smoke','--il','imitation/1607','--name','code_smoke_line_guidance1126','--algo_idx',str(cli.trial_id),'--sim_override','results/preflight_1126/line_guidance_smoke_override.yaml','--init','preflight_1126/ppo25_u70_actor_only.pt','--workers','5','--eval_workers','1','--n_steps','500','--total_steps','1000','--epochs','1','--minibatch','250','--eval_every','1000','--ckpt_every','1','--family_select','0','--short_eval','0','--lost_eval','0','--worst_eval','0','--open_eval','0','--critic_warmup','1','--max_warmup','1','--warmup_ev=-1000000000','--cost_critic','1','--crash_adapt','0','--collision_cost','5000','--crash_start','5000','--ret_norm','1','--log_std_steer','-3.4','--log_std_speed','-2.6','--bc_lost_off','1','--line_cost','10','--line_cost_end','0','--line_cost_decay_updates','2','--lat_acc_mode','path_normal','--lr','2e-5']
 if cli.coupled:
  argv+=['--anneal_imitation_with_line','1','--retention_file','results/ppo_retention_candidate_1126/replay_cache.npz','--retention_manifest','results/ppo_retention_candidate_1126/preparation.json','--retention_weight','.1','--retention_share','.125']
 sys.argv=argv;P.main()
 path=folder/'checkpoint/resume.pt';state=torch.load(path,map_location='cpu',weights_only=False)
 assert state['update']==2 and state['actor_start']==2
 assert [r['line_guidance_coefficient'] for r in state['log'] if r.get('update',0)>0]==[10.,10.]
 sys.argv=argv.copy()+['--resume'];sys.argv[sys.argv.index('--total_steps')+1]='2000';P.main()
 state=torch.load(path,map_location='cpu',weights_only=False)
 observed=[r['line_guidance_coefficient'] for r in state['log'] if r.get('update',0)>0]
 assert observed==[10.,10.,5.,0.],observed
 training=[r for r in state['log'] if r.get('update',0)>0]
 if cli.coupled:
  assert [r['imitation_guidance_multiplier'] for r in training]==[1.,1.,.5,0.]
  assert [r['retention_weight'] for r in training]==[.1,.1,.05,0.]
  assert training[-1]['bc_w']==0.
 assert state['args']['collision_cost']==5000 and state['args']['wall_cost']==5 and state['args']['bay_cost']==0
 q=Path('results/preflight_1126/line_guidance_coupled_checks.json' if cli.coupled else 'results/preflight_1126/line_guidance_schedule_checks.json')
 result=dict(status='passed',observed_rollout_coefficients=observed,actor_start=2,schedule_validation=True,changed_resume_rejected=True,legacy_constant_resume_accepted=True,coupled=cli.coupled,qualification='Two real500-step PPOrollouts,checkpointreload,two more real500-step rollouts with5spawnworkers. Evaluationsstubbed100%solelytoavoidspendingcomputeonsynthetic-modelqualification; thisiscode/schedulerintegrationonly,notdrivingperformance.No productionlearnerorphysicsmodified.')
 if cli.coupled:
  result.update(observed_imitation_multipliers=[r['imitation_guidance_multiplier'] for r in training],observed_retention_weights=[r['retention_weight'] for r in training],final_bc_weight=training[-1]['bc_w'])
 q.write_text(json.dumps(result,indent=2)+'\n');print('LINE_GUIDANCE_CHECKS_PASSED',observed,flush=True)
if __name__=='__main__':main()
