#!/usr/bin/env python3
'''where do the TRAINING-condition crashes come from? policy = a PPO checkpoint (mean action + the training noise),
   env = exactly the run's training env (train_ppo_s2r.make_env with its env_config)
       RUN=results/ppo_s2r/18 python3 tools/train_crash_causes.py <checkpoint .pt> [noise scale]'''
import os, sys, multiprocessing as mp
import numpy as np
SRC = '/home/daehyeok/Autonomous-Driving-2026/src/f1tenth'
RUN = os.environ.get('RUN', 'results/ppo_s2r/18')      # the PPO run whose training env is reproduced

def worker(job):
    os.chdir(SRC); sys.path.insert(0, SRC)
    os.environ['F1_TASK_CONFIG'] = f'{RUN}/backup/f1tenth.yaml'
    import torch, train_ppo_s2r as P, train_il_s2r as IL
    from easydict import EasyDict
    seed, sd, mean, var, n_ep, noise = job
    torch.set_num_threads(1)
    pa = {}
    for line in open(f'{RUN}/backup/ppo_args.yaml'):
        if ': ' in line:
            k, v = line.rstrip('\n').split(': ', 1)
            try: v = float(v) if '.' in v or 'e' in v else int(v)
            except ValueError: v = None if v == 'None' else (v == 'True' if v in ('True', 'False') else v)
            pa[k] = v
    args = EasyDict(pa); args.real_tracks = ['installation/maps_real/Practice1124.npz', 'installation/maps_real/Practice1124_rev.npz']
    cfg = P.env_config(args)
    env = P.make_env(seed, cfg)
    actor = IL.actor_from_state(sd); actor.eval(); std = np.sqrt(var + 1e-8)
    rng = np.random.default_rng(seed)
    nstd = np.exp(np.array([pa['log_std_steer'], pa['log_std_speed']]))
    out = []
    for _ in range(n_ep):
        obs, info = env.reset()
        e = env
        tf = os.path.basename(e.track_file or '')
        kind = 'short' if ('_s' in tf and tf.startswith(('Rand', 'IMS_s'))) else ('open' if 'open_train' in (e.track_file or '') else ('real' if 'maps_real' in (e.track_file or '') else 'regular'))
        ln = e.loc_noise
        lost_ep = bool(ln is not None and (ln.lost_ep or ln.full_lost))
        cap = float(e.speed_cap); hard = bool(e.hard)
        done, lost_at_end, steps = False, False, 0
        while not done:
            with torch.no_grad():
                a = actor(torch.as_tensor((obs - mean) / std, dtype=torch.float32)).loc.numpy()
            if noise:
                a = np.clip(a + rng.normal(0, nstd * noise), -1, 1)
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            done = term or trunc
        lost_at_end = obs[-1] < 0.5
        out.append((kind, hard, lost_ep, cap, bool(e.success), bool(e.collision), bool(getattr(e, 'stuck', False)), lost_at_end, float(e.elapsed)))
    return out

if __name__ == '__main__':
    os.chdir(SRC); sys.path.insert(0, SRC)
    os.environ['F1_TASK_CONFIG'] = f'{RUN}/backup/f1tenth.yaml'
    import torch, train_il_s2r as IL
    from utils.vectorize import RunningMeanStd
    args = IL.get_args([])
    rms = RunningMeanStd('agent_obs', args.obs_dim); rms.load('results/imitation/13/sim2real', required=True)
    ck = sys.argv[1]; noise = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    sd = torch.load(ck, map_location='cpu')['actor']
    with mp.get_context('spawn').Pool(8) as p:
        res = sum(p.map(worker, [(1000 + w, sd, rms.mean, rms.var, 20, noise) for w in range(8)]), [])
    R = np.array([(k, h, l, c, s, co, st, la, t) for k, h, l, c, s, co, st, la, t in res], dtype=object)
    def show(lab, m):
        m = np.array(m, bool)
        if m.sum() == 0: return
        sub = R[m]
        print(f'  {lab:28s} {m.sum():4d} episodes: success {np.mean(sub[:,4].astype(bool))*100:5.1f}%  crash {np.mean(sub[:,5].astype(bool))*100:5.1f}%  '
              f'(crash while localization lost {np.mean((sub[:,5].astype(bool)) & (sub[:,7].astype(bool)))*100:4.1f}%)')
    print(f'{ck} in TRAINING conditions, exploration noise x{noise:g}, {len(R)} episodes')
    show('all', [True] * len(R))
    for k in ('regular', 'short', 'open', 'real'):
        show(f'track {k}', R[:, 0] == k)
    show('hard conditions', R[:, 1].astype(bool)); show('normal conditions', ~R[:, 1].astype(bool))
    show('episodes with localization losses', R[:, 2].astype(bool)); show('no losses', ~R[:, 2].astype(bool))
    caps = R[:, 3].astype(float)
    for lo, hi in ((0, 3.5), (3.5, 5.5), (5.5, 7.9), (7.9, 99)):
        show(f'cap {lo}-{hi}', (caps >= lo) & (caps < hi))
