'''
    One report that compares models (IL, PPO, the expert) on everything used to judge them,
    run from src/f1tenth:

        python -u compare_models.py imitation/9 ppo_s2r/1                 # first model = reference
        python -u compare_models.py expert imitation/9 ppo_s2r/1 --quick  # 1 lap, fewer settings (smoke test)

    Each model drives with its own task config (lidar layout, steering calibration: results/<m>/backup)
    under the current car conditions (configs/task/f1tenth.yaml sim2real, dynamic.yaml), 2 laps:
      eval   : the IL evaluation tracks x IL.EVAL_SETTINGS (cap 8 centerline / cap 8 random / cap 5 random)
      worst  : the same tracks, cap 8 centerline, every sim2real gap at its harder end (sim_mode 'worst')
      open   : open-space test tracks (eval_open_space.py, level big) x caps 8 / 5 from the centerline
    Lap time is compared PAIRED: only runs that the model and the reference both finished, per run
    (time_ref - time) / time_ref, so tracks one of them crashed on do not distort the mean.
    Output: printed tables + results/reports/compare_<date>_<models>.md (with the per-track lap times).
'''
import argparse
import glob
import os
import sys
import time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'installation'))
import numpy as np


def run(job):
    model, path, cap, start, mode, laps = job
    import torch
    import train_il_s2r as IL
    from evaluate_s2r import model_args
    from utils.env import F1Wrapper, normalize_speed
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    seed = sum(map(ord, os.path.basename(path)))
    np.random.seed(seed)
    args = IL.get_args([]) if model == 'expert' else model_args(model)
    args.seed = seed
    args.laps = laps
    env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode=mode, tracks=[path])
    env.fixed_speed_cap = cap
    env.start_mode = start
    obs, _ = env.reset(options={'track': path})
    if model == 'expert':
        from utils.rule_based import RacelinePlanner
        pl = RacelinePlanner(env._env.unwrapped.track, args.length_f + args.length_r, v_min=env.min_speed,
                             v_max=env.speed_cap, max_steer=env.max_steer, max_steer_left=env.steer_left_max, max_steer_right=env.steer_right_max,
                             **IL.expert_kwargs(env))
    else:
        rms = RunningMeanStd('agent_obs', args.obs_dim)
        rms.load(f'results/{model}/sim2real', required=True)
        actor = torch.load(f'results/{model}/sim2real/actor.pt', map_location='cpu')
        actor.eval()
    done, v, st = False, [], []
    while not done:
        if model == 'expert':
            sq, vq = env.pending_commands()
            sp, s = pl.plan(env.vehicle_state(), sq, vq, env.timestep)
            a = np.clip([s / env.max_steer, normalize_speed(sp, env.min_speed, env.max_speed)], -1, 1)
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(rms.normalize(obs), dtype=torch.float32)).loc.numpy()
        obs, r, term, trunc, info = env.step(a)
        v.append(float(env.vehicle_state()[3]))
        st.append(float(np.clip(a[0], -1, 1)) * env.max_steer)
        done = term or trunc
    group = 'open' if 'maps_open_test' in path else ('worst' if mode == 'worst' else 'eval')
    return dict(model=model, group=group, key=f'{os.path.basename(path)[:-4]}@{cap:g}/{start}', ok=bool(env.success),
                t=float(env.elapsed), v=float(np.mean(v)), jitter=float(np.abs(np.diff(st)).mean()) if len(st) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('models', nargs='+', help="results/<model> names and/or 'expert'; the first is the reference")
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--quick', action='store_true', help='1 lap, eval cap 8 centerline + open cap 5 only, no worst')
    p.add_argument('--open_level', default='big')
    p.add_argument('--max_tracks', type=int, default=0, help='first N tracks of each set (smoke tests)')
    a = p.parse_args()
    import train_il_s2r as IL
    from eval_open_space import TEST_DIR
    allt = IL.list_variants(IL.MAPS)
    ev = sorted([t for t in allt if IL.variant_map(t) in IL.EVAL_MAPS and
                 (IL.HOLDOUT in t or os.path.basename(t)[:-4] in [m + s for m in IL.EVAL_MAPS for s in ('', '_rev')])])[:44]
    op = sorted(glob.glob(os.path.join(TEST_DIR, a.open_level, '*.npz')))
    if a.max_tracks:
        ev, op = ev[:a.max_tracks], op[:a.max_tracks]
    laps = 1 if a.quick else 2
    ev_set = [(8.0, 'center')] if a.quick else list(IL.EVAL_SETTINGS)
    op_set = [(5.0, 'center')] if a.quick else [(8.0, 'center'), (5.0, 'center')]
    jobs = []
    for m in a.models:
        jobs += [(m, t, c, s, 'eval', laps) for c, s in ev_set for t in ev]
        if not a.quick:
            jobs += [(m, t, 8.0, 'center', 'worst', laps) for t in ev]
        jobs += [(m, t, c, s, 'eval', laps) for c, s in op_set for t in op]
    t0 = time.time()
    print(f'{len(jobs)} runs ({len(a.models)} models), {laps} lap(s), {a.workers} workers ...', flush=True)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        res = pool.map(run, jobs, chunksize=1)
    ref = a.models[0]
    lines = [f'# model comparison ({time.strftime("%Y-%m-%d %H:%M")}, {laps} lap(s), reference = {ref})', '',
             f'tracks: eval {len(ev)} x {ev_set}' + ('' if a.quick else f', worst-case {len(ev)} x cap 8 centerline')
             + f', open-space ({a.open_level}) {len(op)} x {op_set}', '',
             '| model | eval success | worst-case success | open-space success | paired time vs ref (eval, uncapped) | mean speed | steering jitter |',
             '|---|---|---|---|---|---|---|']
    by = lambda m, g: [r for r in res if r['model'] == m and r['group'] == g]
    for m in a.models:
        e, w, o = by(m, 'eval'), by(m, 'worst'), by(m, 'open')
        sr = lambda rr: f'{np.mean([r["ok"] for r in rr])*100:.1f}% ({sum(r["ok"] for r in rr)}/{len(rr)})' if rr else '-'
        ref_ok = {r['key']: r['t'] for r in by(ref, 'eval') if r['ok'] and '@8/' in r['key']}
        mine = {r['key']: r['t'] for r in e if r['ok'] and '@8/' in r['key']}
        com = sorted(set(ref_ok) & set(mine))
        g = np.mean([(ref_ok[k] - mine[k]) / ref_ok[k] for k in com]) * 100 if com else 0.0
        gain = (f'{abs(g):.1f}% {"faster" if g >= 0 else "slower"} on {len(com)} runs'
                if com and m != ref else ('reference' if m == ref else '-'))
        lines.append(f'| {m} | {sr(e)} | {sr(w)} | {sr(o)} | {gain} | {np.mean([r["v"] for r in e]):.2f} m/s | '
                     f'{np.mean([r["jitter"] for r in e]):.4f} rad |')
    lines += ['', '## uncapped 2-lap time per track (centerline start; - = did not finish)', '',
              '| track | ' + ' | '.join(a.models) + ' |', '|---|' + '---|' * len(a.models)]
    keys = sorted({r['key'] for r in res if r['group'] == 'eval' and r['key'].endswith('@8/center')})
    tab = {(r['model'], r['key']): r for r in res if r['group'] == 'eval'}
    for k in keys:
        cells = []
        for m in a.models:
            r = tab.get((m, k))
            cells.append(f'{r["t"]:.1f}' if r and r['ok'] else '-')
        lines.append(f'| {k.split("@")[0]} | ' + ' | '.join(cells) + ' |')
    fails = [r for r in res if not r['ok']]
    lines += ['', f'## failures ({len(fails)})', ''] + [f'- {r["model"]} {r["group"]} {r["key"]} at {r["t"]:.1f} s' for r in fails]
    text = '\n'.join(lines)
    print(text[:text.index('## uncapped')])
    os.makedirs('results/reports', exist_ok=True)
    out = f'results/reports/compare_{time.strftime("%m%d_%H%M")}_{"_vs_".join(m.replace("/", "") for m in a.models)}.md'
    with open(out, 'w') as f:
        f.write(text + '\n')
    print(f'[saved] {out} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
