#!/usr/bin/env python3
"""Exercise the actual PPO vector workers, reward path and autoreset metadata."""
import os, sys, json, functools, multiprocessing as mp
from pathlib import Path
R=Path(__file__).resolve().parents[1]; S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def main():
    os.chdir(S)
    import numpy as np
    import gymnasium as gym
    import train_il_s2r as IL
    import train_ppo_s2r as PPO
    from utils.training_mix import family,step_budgets
    sys.argv=[str(__file__),'--il','ppo_s2r/2130','--workers','20']
    args=PPO.get_args()
    args.update(IL.get_args(['--task_config','configs/task/f1tenth_preview_1126.yaml','--workers','20']))
    args.lat_grip_share=.9; args.straight_heading_cost=.5; args.straight_offset_cost=2.
    cfg=PPO.env_config(args)
    allocation=step_budgets(args.training_mix,args.workers)
    families=[g for g,n in allocation.items() for _ in range(n)]
    fns=[functools.partial(PPO.make_env,1000+i,dict(cfg,track_family=g)) for i,g in enumerate(families)]
    terminal_count=0
    with gym.vector.AsyncVectorEnv(fns,context='spawn') as env:
        obs,info=env.reset(seed=2)
        assert obs.shape==(20,132) and np.isfinite(obs).all()
        assert [family(p) for p in env.get_attr('track_file')]==families
        weights=PPO.base_track_weights(args)
        env.call('set_track_weights',weights,1.)
        for _ in range(100):
            obs,reward,term,trunc,info=env.step(np.zeros((20,2),np.float32))
            assert np.isfinite(obs).all() and np.isfinite(reward).all()
            for i in np.where(term|trunc)[0]:
                assert np.isfinite(info['final_observation'][i]).all()
                assert np.isfinite(info['final_info'][i]['privileged']).all()
                terminal_count+=1
    out=dict(workers=20,steps=2000,allocation=allocation,terminal_count=terminal_count,
             result='passed: worker family, observations, reward and terminal-state metadata')
    (S/'results/preflight_1126/ppo_vector_smoke.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out,flush=True)


if __name__=='__main__':main()
