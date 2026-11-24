# Simulator vs car check (11/24), run from src/f1tenth:
#   F1_TASK_CONFIG=configs/task/f1tenth_lidar88.yaml python check_sim_vs_car.py
# IL 11 in the CURRENT simulator (eval mode: delay 0.12, mu 0.65, a_max 3.3), measured like tools/steer_response.py:
# best command->yaw delay and actual / (v tan(steer) / 0.34) per side. Real car (11/24, IL 11, odom speed):
#   2.0 cap (1511): delay 0.15 s, left 0.79, right 0.64 | 4.0 cap (1521): delay 0.125 s, left 0.68, right 0.52
# Result 11/24 with IL 11's car steering ratio (1.30, 1.05): 2.0 cap 0.150 s / 0.75 / 0.67 / 1.95 m/s,
#   4.0 cap 0.150 s / 0.65 / 0.51 / 2.77 m/s (car 1.94 and 2.83 m/s) -> the simulator matches the car.
#   mu barely changes the response (0.52 at mu 0.5, 0.53 at 0.65).
import os, sys
sys.path[:0] = ['.', 'installation']
import numpy as np, torch
import train_il_s2r as IL
from evaluate_s2r import model_args
from utils.env import F1Wrapper
import multiprocessing as mp

def run(job):
    cap, scales, mu = job
    torch.set_num_threads(1)
    args = model_args('imitation/11'); args.seed = 7
    rms = __import__('utils.vectorize', fromlist=['RunningMeanStd']).RunningMeanStd('agent_obs', args.obs_dim)
    rms.load('results/imitation/11/sim2real', required=True)
    actor = torch.load('results/imitation/11/sim2real/actor.pt', map_location='cpu', weights_only=False); actor.eval()
    names = ('Sochi_w101', 'IMS_rev_w102', 'Catalunya_w103', 'Zandvoort_rev_w104', 'Spa_w106', 'Monza_rev_w107')
    tracks = sorted(t for t in IL.list_variants(IL.MAPS) if os.path.basename(t)[:-4] in names)
    T, Y, V, S = [], [], [], []
    for path in tracks:
        env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode='eval', tracks=[path])
        env.fixed_speed_cap = cap; env.start_mode = 'center'
        obs, _ = env.reset(options={'track': path})
        if scales: env.steer_scale = scales
        if mu: env.params['mu'] = mu; env._env.unwrapped.sim.update_params(env.params)
        t, done = 0.0, False
        while not done and t < 45:
            with torch.no_grad():
                a = actor(torch.as_tensor(rms.normalize(obs), dtype=torch.float32)).loc.numpy()
            obs, r, term, trunc, info = env.step(a)
            st = env.vehicle_state()
            Y.append(st[5]); V.append(st[3]); S.append(float(np.clip(a[0], -1, 1)) * env.max_steer)
            t += env.timestep; done = term or trunc
        Y.append(np.nan); V.append(np.nan); S.append(np.nan)
    Y, V, S = map(np.array, (Y, V, S))
    best = None
    for lag in range(0, 13):
        s_l = np.r_[np.full(lag, np.nan), S[:-lag]] if lag else S.copy()
        e = V * np.tan(s_l) / 0.34; ok = (V > 1.0) & np.isfinite(e) & np.isfinite(Y)
        cc = np.corrcoef(e[ok], Y[ok])[0, 1]
        if best is None or cc > best[1]: best = (lag, cc)
    lag = best[0]; s_l = np.r_[np.full(lag, np.nan), S[:-lag]] if lag else S.copy(); e = V * np.tan(s_l) / 0.34
    g = []
    for sel in (s_l > 0.05, s_l < -0.05):
        ok = sel & (V > 1.0) & (np.abs(s_l) < 0.35) & np.isfinite(e) & np.isfinite(Y)
        g.append(np.sum(Y[ok] * e[ok]) / np.sum(e[ok] ** 2))
    return f'cap {cap}, steer scale {scales or "nominal (1.05, 0.90)"}, mu {mu or 0.65}: delay {lag * 0.025:.3f} s, left {g[0]:.2f}, right {g[1]:.2f}, mean speed {np.nanmean(V[V > 0.5]):.2f}'

if __name__ == '__main__':
    jobs = [(2.0, (1.0, 1.0), None), (4.0, (1.0, 1.0), None), (2.0, (1.30, 1.05), None), (4.0, (1.30, 1.05), None),
            (4.0, (1.0, 1.0), 0.5)]
    with mp.get_context('spawn').Pool(len(jobs)) as p:
        for line in p.map(run, jobs):
            print(line)
