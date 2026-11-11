"""Parallel expert-data collection for main_il.py.

Collects with N processes and writes expert_data_obs{obs_dim}.pkl with the same
data_config main_il.py computes, so main_il.py loads it instead of collecting
single-process (~220 steps/s -> 300k steps took ~23 min; 16 workers: ~1.5 min).

Run from src/f1tenth with the same settings you pass to main_il.py:
    python collect_expert_parallel.py <total_steps> <workers> <max_episode_steps> <dart_std>
    python collect_expert_parallel.py 400000 16 4000 0.15
    python -u main_il.py --name imitation --algo_idx 6 --max_episode_steps 4000 --total_steps 400000 ...
"""
import sys, os, pickle, multiprocessing as mp
sys.path.insert(0, '.')
import numpy as np
from easydict import EasyDict
from ruamel.yaml import YAML
import main_il as M
from utils.env import F1Wrapper
from utils.rule_based import CurvaturePursuitPlanner

TOTAL, NW, MAX_EP, DART = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])

def make_args():
    args = EasyDict(vars(M.getParser().parse_args([])))
    args.pop('max_episode_steps')
    for f in ['configs/task/f1tenth.yaml', 'configs/task/dynamic.yaml', 'configs/algorithm/imitation.yaml']:
        args.update(EasyDict(YAML().load(open(f))))
    args.max_episode_steps = MAX_EP
    args.total_steps = TOTAL
    args.dart_std = DART
    return args

def worker(wid):
    args = make_args()
    args.seed = 1000 + wid
    np.random.seed(1000 + wid)
    env = F1Wrapper(args=args, maps=M.MAPS, obs_noise=True)
    obs_all, act_all, total = [], [], 0
    while total < TOTAL // NW:
        observation, info = env.reset()
        planner = CurvaturePursuitPlanner(env._env.unwrapped.track, M.WHEELBASE, v_min=env.min_speed,
                                          v_max=env.max_speed, max_steer=env.max_steer, **M.EXPERT)
        eo, ea, step, noise = [], [], 0, 0.0
        noise_std = np.random.uniform(0.0, DART)
        term = False
        while step < MAX_EP and not term:
            speed, steer = planner.plan(info['obs_dict'])
            sa = (speed - (env.max_speed + env.min_speed)/2.0) / ((env.max_speed - env.min_speed)/2.0)
            action = np.clip([steer/(env.max_steer+M.EPS), sa], -np.ones(2), np.ones(2))
            eo.append(observation); ea.append(action)
            noise = 0.9*noise + np.sqrt(1 - 0.9**2)*noise_std*np.random.randn()
            observation, r, term, trunc, info = env.step(np.clip(action + np.array([noise, 0.0]), -1, 1))
            step += 1
        if step > args.threshold:
            obs_all.append(eo); act_all.append(ea); total += step
            print(f'[w{wid}] {total}/{TOTAL//NW} ep {step} {env._env.map}', flush=True)
    cfg = env.obs_cfg
    env.close()
    return np.concatenate(obs_all), np.concatenate(act_all), cfg

if __name__ == '__main__':
    with mp.get_context('fork').Pool(NW) as p:
        res = p.map(worker, range(NW))
    args = make_args()
    data_config = {'obs_dim': args.obs_dim, **res[0][2],
                   'min_speed': args.min_speed, 'max_speed': args.max_speed, 'max_steer': args.max_steer,
                   'expert': {'planner': 'CurvaturePursuitPlanner', **M.EXPERT}, 'dart_std': DART,
                   'speed_label': 'inverse_unnormalize', 'max_episode_steps': MAX_EP, 'total_steps': TOTAL}
    obs = np.concatenate([r[0] for r in res]); act = np.concatenate([r[1] for r in res])
    with open(f'expert_data_obs{args.obs_dim}.pkl', 'wb') as f:
        pickle.dump({'observations': obs, 'actions': act, 'data_config': data_config}, f)
    print('DONE', obs.shape, act.shape, flush=True)
