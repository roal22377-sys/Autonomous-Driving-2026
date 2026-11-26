#!/usr/bin/env python3
"""Small code-path verification; outputs do not replace any deployable model."""
import os,sys,json
from pathlib import Path
__file__=str(Path(__file__).resolve())
sys.argv[0]=__file__
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def main():
    os.chdir(S)
    import numpy as np,torch
    import train_il_s2r as IL
    from utils.training_mix import groups_for
    from utils.vectorize import RunningMeanStd
    from utils.policy_transfer import initialize
    torch.set_num_threads(2)
    a=IL.get_args(['--task_config','configs/task/f1tenth_preview_1126.yaml','--workers','2','--bc_threads','2',
                   '--hidden_dim','512','--batch','512'])
    groups=groups_for(a,[p for p in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in p])
    paths=[p for group,ps in groups.items() if a.training_mix.get(group,0)>0 for p in ps]
    O,Y,stats,N=IL.collect(a,8000,None,1.,.02,paths,0)
    assert len(O)==8000 and O.shape[1]==132 and np.isfinite(O).all() and not N[-1]
    actor=IL.make_actor(a);actor.initialize()
    rms=RunningMeanStd('agent_obs',a.obs_dim)
    report=initialize(actor,a,rms,'results/ppo_s2r/2130',O)
    val=IL.train_bc(a,actor,O,Y,rms,20,1e-5,N)
    assert np.isfinite(val)
    policy=({k:v.detach().clone() for k,v in actor.state_dict().items()},rms.mean.copy(),rms.var.copy())
    subset=groups['real']+[groups['hall'][0]]
    rate,lap,res=IL.evaluate(a,policy,subset,settings=((4.,'center'),))
    out=S/'results/preflight_1126/smoke_training.json'
    out.write_text(json.dumps(dict(samples=len(O),dim=O.shape[1],transfer=report,bc_val=val,smoke_eval=rate,
                                  qualification='20 gradient steps only; smoke score is not a model-selection result'),indent=2)+'\n')
    print('SMOKE PASSED',report,'val',val,flush=True)


if __name__=='__main__':main()
