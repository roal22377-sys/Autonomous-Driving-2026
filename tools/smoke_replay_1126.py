#!/usr/bin/env python3
"""Real cached data through mixed BC/CAPS; 20 steps, no policy qualification."""
import os,sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def main():
    os.chdir(S)
    import numpy as np,torch
    import train_il_s2r as IL
    from utils.vectorize import RunningMeanStd
    args=IL.get_args(['--task_config','configs/task/f1tenth_preview_1126.yaml','--batch','512','--bc_threads','2',
                      '--replay_file','results/retention_probe_1126/replay_cache.npz'])
    with np.load('results/imitation/14/data/round000.npz') as z:
        O,Y,N=z['O'],z['Y'],z['N']
    pieces=[np.arange(i,i+1600) for i in [0,480000,840000,1020000,1080000]]
    ix=np.concatenate(pieces);O,Y,N=O[ix],Y[ix],N[ix]
    N[np.arange(1599,8000,1600)]=False
    ck=torch.load('results/imitation/14/checkpoint/model_init.pt',map_location='cpu',weights_only=False)
    rms=RunningMeanStd('agent_obs',132);rms.mean=ck['obs_mean'];rms.var=ck['obs_var']
    cache=args.replay_file;checks={}
    for enabled in [False,True]:
        args.replay_file=cache if enabled else None
        actor=IL.actor_from_state(ck['actor']);before=actor.fc1.weight.detach().clone()
        loss=IL.train_bc(args,actor,O,Y,rms,20,1e-5,N)
        assert np.isfinite(loss) and all(torch.isfinite(p).all() for p in actor.parameters())
        assert not torch.equal(before,actor.fc1.weight)
        checks['replay_on' if enabled else 'replay_off']=dict(finite=True,parameters_updated=True)
    report=dict(result='passed',new_samples=8000,grad_steps=20,replay_share=.5,
                checks=checks,qualification='BC replay/CAPS numerical verification only; no deployment or PPO qualification')
    Path('results/preflight_1126/bc_replay_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':main()
