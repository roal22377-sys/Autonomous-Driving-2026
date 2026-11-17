'''
    Imitation learning for the real car (sim2real), run from src/f1tenth:

        python -u train_il_s2r.py --algo_idx 7 --workers 22

    vs. main_il.py (run 6):
      - expert: RacelinePlanner (utils/rule_based.py) on the minimum-curvature race line,
        with steering-delay compensation, instead of the centerline pursuit
      - simulator: sim_mode='train' (utils/env.py) -> per-episode steering delay 0.10-0.25 s,
        real acceleration limit, friction, lidar/odom noise measured on the car, generated
        tracks (random width, open-side pockets, both directions), 2-lap episodes
      - observation: lidar + speed + last 10 commands (configs/task/f1tenth.yaml, obs_dim 75)
      - data: DART (correlated steering noise while the expert drives, clean labels) then
        DAgger iterations (the student drives, the expert labels what it should have done)
      - parallel collection / evaluation on all CPU cores
      - held-out evaluation tracks (variant w105 never used for training)

    Output: results/imitation/<algo_idx>/ in the same layout as main_il.py
    (sim2real/actor.pt + agent_obs_scale.pkl, backup/*.yaml) so agent_node.py and
    main_rl.py / train_ppo_s2r.py can load it.
'''
import argparse
import os
import sys
import time
import pickle
import glob
import random
import multiprocessing as mp
from shutil import copyfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'installation'))
import numpy as np
import torch
from easydict import EasyDict
from ruamel.yaml import YAML

from utils.env import F1Wrapper, normalize_speed
from utils.rule_based import RacelinePlanner
from utils.track_gen import list_variants, variant_map
from utils.vectorize import RunningMeanStd
from utils.color import cprint

MAPS = ['Sochi', 'Spa', 'Nuerburgring', 'Monza', 'Melbourne', 'Austin',
        'Silverstone', 'Sakhir', 'IMS', 'Budapest', 'Montreal', 'Sepang',
        'Oschersleben', 'YasMarina', 'MoscowRaceway', 'Zandvoort', 'Catalunya',
        'BrandsHatch', 'Shanghai', 'Hockenheim', 'SaoPaulo', 'Spielberg', 'MexicoCity']
EVAL_MAPS = ['Sochi', 'IMS', 'Catalunya', 'Zandvoort', 'Silverstone', 'SaoPaulo',
             'YasMarina', 'MoscowRaceway', 'Shanghai', 'MexicoCity', 'Montreal']
HOLDOUT = '_w105'               # generated variant kept out of training
WHEELBASE = None                # from configs/task/dynamic.yaml (length_f + length_r)
EXPERT = {'a_lat': 4.0, 'lad_base': 0.6, 'lad_gain': 0.22, 'merge_dist': 0.6}
STEER_LOSS_WEIGHT = 2.0         # steering errors matter more than speed errors


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='imitation')
    p.add_argument('--algo_idx', type=int, default=7)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=22)
    p.add_argument('--expert_steps', type=float, default=1.2e6, help='DART expert rollout steps')
    p.add_argument('--dagger_iters', type=int, default=3)
    p.add_argument('--dagger_steps', type=float, default=6e5, help='steps per DAgger iteration')
    p.add_argument('--dart_std', type=float, default=0.15)
    p.add_argument('--grad_steps', type=int, default=15000, help='gradient steps after each data round')
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden_dim', type=int, default=512, help='actor width (2 hidden layers); run 7 used 256')
    p.add_argument('--eval_tracks', type=int, default=44)
    p.add_argument('--max_data', type=float, default=1.2e7,
                   help='keep at most this many samples (oldest DAgger rounds dropped, expert round kept)')
    p.add_argument('--min_beta', type=float, default=0.1, help='expert-driven episode fraction never below this')
    p.add_argument('--init', default=None, help='start from this model (e.g. imitation/7): its actor + normalizer, '
                                               'and go straight to DAgger with it driving')
    p.add_argument('--resume', action='store_true', help='reload the data saved in <save_dir>/data and continue')
    # CAPS (Mysore et al. 2021) smoothness regularization: removes high-frequency steering jitter without
    # adding delay. temporal: pi(s_t) ~ pi(s_t+1); spatial: pi(s) ~ pi(s + noise)
    p.add_argument('--caps_t', type=float, default=0.3, help='temporal smoothness weight (0 = off)')
    p.add_argument('--caps_s', type=float, default=0.3, help='spatial smoothness weight (0 = off)')
    p.add_argument('--caps_sigma', type=float, default=0.05, help='spatial noise std in normalized obs units')
    a = p.parse_args(argv)
    args = EasyDict(vars(a))
    for f in ['configs/task/f1tenth.yaml', 'configs/task/dynamic.yaml', 'configs/algorithm/imitation.yaml']:
        with open(f) as fh:
            for k, v in EasyDict(YAML().load(fh)).items():
                if k not in ('lr', 'hidden_dim'):
                    args[k] = v
    args.device = torch.device('cpu')
    args.save_dir = f'results/{args.name}/{args.algo_idx}'
    return args


def actor_from_state(state_dict):
    '''actor with the width of a saved state dict (256 for run 7, 512 for run 8+)'''
    a = get_args([])
    a.hidden_dim = int(state_dict['fc1.weight'].shape[0])
    actor = make_actor(a)
    actor.load_state_dict(state_dict)
    return actor


def make_actor(args):
    from algorithm.common.models import Actor
    return Actor(args)


# ------------------------------------------------------------------ rollouts (workers)
def rollout_worker(job):
    '''
        job: (wid, n_steps, policy, beta, dart_std, tracks, seed)
        policy: None or (actor state_dict, obs mean, obs var)
        beta: probability that the expert (with DART noise) drives an episode
    '''
    wid, n_steps, policy, beta, dart_std, tracks, seed = job
    torch.set_num_threads(1)
    args = get_args([])
    args.seed = seed
    np.random.seed(seed)
    env = F1Wrapper(args, MAPS, obs_noise=True, sim_mode='train', tracks=tracks)
    actor = None
    if policy is not None:
        actor = actor_from_state(policy[0])
        actor.eval()
        mean, std = policy[1], np.sqrt(policy[2] + 1e-8)

    O, Y, N, stats, total = [], [], [], [], 0      # N[i]: sample i+1 is the next step of the same episode
    while total < n_steps:
        obs, info = env.reset()
        # the expert respects this episode's speed cap (the policy is told the cap in its observation)
        pl = RacelinePlanner(env._env.unwrapped.track, args.length_f + args.length_r, v_min=env.min_speed, v_max=env.speed_cap,
                             a_acc=env.params['a_max'], a_brake=env.params['a_max'], max_steer=env.max_steer, **EXPERT)
        expert_drives = actor is None or np.random.rand() < beta
        noise_std = np.random.uniform(0.0, dart_std) if expert_drives else 0.0
        noise, done, steps = 0.0, False, 0
        while not done:
            sq, vq = env.pending_commands()
            v, s = pl.plan(env.vehicle_state(), sq, vq, env.timestep)
            label = np.clip([s / env.max_steer, normalize_speed(v, env.min_speed, env.max_speed)], -1, 1)
            O.append(obs)
            Y.append(label)
            N.append(True)
            if expert_drives:
                noise = 0.9 * noise + np.sqrt(1 - 0.9 ** 2) * noise_std * np.random.randn()
                act = np.clip(label + np.array([noise, 0.0]), -1, 1)
            else:
                with torch.no_grad():
                    act = actor(torch.as_tensor((obs - mean) / std, dtype=torch.float32)).loc.numpy()
            obs, r, term, trunc, info = env.step(act)
            steps += 1
            done = term or trunc
        N[-1] = False
        total += steps
        stats.append((expert_drives, int(env.success), steps))
    env.close()
    return np.asarray(O, np.float32), np.asarray(Y, np.float32), stats, np.asarray(N, bool)


def collect(args, n_steps, policy, beta, dart_std, tracks, round_idx):
    per = int(np.ceil(n_steps / args.workers))
    jobs = [(w, per, policy, beta, dart_std, tracks, args.seed * 100000 + round_idx * 1000 + w)
            for w in range(args.workers)]
    with mp.get_context('spawn').Pool(args.workers) as pool:
        res = pool.map(rollout_worker, jobs)
    O = np.concatenate([r[0] for r in res])
    Y = np.concatenate([r[1] for r in res])
    stats = [s for r in res for s in r[2]]
    Nn = np.concatenate([r[3] for r in res])
    return O, Y, stats, Nn


# ------------------------------------------------------------------ evaluation (workers)
def eval_worker(job):
    path, policy, mode, seed, cap, start = job
    torch.set_num_threads(1)
    np.random.seed(seed)          # same start pose / noise for every model -> comparable results
    random.seed(seed)
    args = get_args([])
    args.seed = seed
    env = F1Wrapper(args, MAPS, obs_noise=(mode != 'ideal'), sim_mode=mode, tracks=[path])
    env.fixed_speed_cap = cap
    env.start_mode = start
    actor = actor_from_state(policy[0])
    actor.eval()
    mean, std = policy[1], np.sqrt(policy[2] + 1e-8)
    obs, info = env.reset(options={'track': path})
    done, steers = False, []
    while not done:
        with torch.no_grad():
            act = actor(torch.as_tensor((obs - mean) / std, dtype=torch.float32)).loc.numpy()
        steers.append(act[0])
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
    ok = bool(env.success)
    st = np.asarray(steers) * env.max_steer
    return f'{os.path.basename(path)[:-4]}@{cap:g}/{start}', ok, float(env.elapsed), float(np.abs(np.diff(st)).mean())


# (speed cap [m/s], start): uncapped from the centerline (lap time), uncapped and car-practice cap
# from random starts anywhere across the track with a heading error (merging onto the race line)
EVAL_SETTINGS = ((8.0, 'center'), (8.0, 'random'), (3.0, 'random'))
EVAL_CAPS = tuple(c for c, _ in EVAL_SETTINGS)


def evaluate(args, policy, tracks, mode='eval', settings=EVAL_SETTINGS):
    '''success over all runs; lap = mean 2-lap time of the first setting's runs that finished'''
    jobs = [(p, policy, mode, 7 + i, cap, start) for cap, start in settings for i, p in enumerate(sorted(tracks))]
    with mp.get_context('spawn').Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(eval_worker, jobs)
    succ = np.mean([r[1] for r in res])
    held = [r[1] for r in res if HOLDOUT in r[0]]
    print(f'      held-out tracks ({HOLDOUT}, never trained on): success {np.mean(held)*100:.1f}% '
          f'| training geometries: {np.mean([r[1] for r in res if HOLDOUT not in r[0]])*100:.1f}%', flush=True)
    fast = [r for r in res if r[0].endswith(f'@{settings[0][0]:g}/{settings[0][1]}') and r[1]]
    lap = np.mean([r[2] for r in fast]) if fast else float('nan')
    return succ, lap, res


# ------------------------------------------------------------------ behavior cloning
def train_bc(args, actor, O, Y, rms, grad_steps, lr, Nn=None):
    torch.set_num_threads(args.workers)
    X = torch.as_tensor(rms.normalize(O), dtype=torch.float32)
    T = torch.as_tensor(Y, dtype=torch.float32)
    n = len(X)
    perm = torch.randperm(n)
    nval = max(2000, n // 50)
    val, tr = perm[:nval], perm[nval:]
    # samples whose next sample is the next step of the same episode (CAPS temporal term)
    nxt = torch.as_tensor(np.where(Nn[:-1])[0]) if (Nn is not None and args.caps_t > 0) else None
    w = torch.tensor([STEER_LOSS_WEIGHT, 1.0])
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=grad_steps, eta_min=lr * 0.05)
    actor.train()
    for it in range(1, grad_steps + 1):
        idx = tr[torch.randint(len(tr), (args.batch,))]
        out = actor(X[idx]).loc
        loss = ((out - T[idx]) ** 2 * w).mean()
        if args.caps_s > 0:
            noisy = actor(X[idx] + args.caps_sigma * torch.randn_like(X[idx])).loc
            loss = loss + args.caps_s * ((noisy - out) ** 2).mean()
        if nxt is not None:
            j = nxt[torch.randint(len(nxt), (args.batch // 2,))]
            loss = loss + args.caps_t * ((actor(X[j + 1]).loc - actor(X[j]).loc) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        opt.step()
        sched.step()
        if it % 3000 == 0 or it == grad_steps:
            with torch.no_grad():
                vl = ((actor(X[val]).loc - T[val]) ** 2 * w).mean().item()
            print(f'    [bc] step {it}/{grad_steps} train {loss.item():.5f} val {vl:.5f}', flush=True)
    actor.eval()
    return vl


def save(args, actor, rms, tag=None):
    sim2real = f'{args.save_dir}/sim2real'
    os.makedirs(sim2real, exist_ok=True)
    os.makedirs(f'{args.save_dir}/checkpoint', exist_ok=True)
    if tag is None:
        torch.save(actor, f'{sim2real}/actor.pt')
        rms.save(sim2real)
        rms.save(args.save_dir)
        RunningMeanStd('agent_reward', 1).save(sim2real)
        torch.save({'actor': actor.state_dict()}, f'{args.save_dir}/checkpoint/model.pt')
    else:
        torch.save({'actor': actor.state_dict(), 'obs_mean': rms.mean, 'obs_var': rms.var},
                   f'{args.save_dir}/checkpoint/model_{tag}.pt')


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(f'{args.save_dir}/backup', exist_ok=True)
    for f in ['configs/task/f1tenth.yaml', 'configs/task/dynamic.yaml', 'configs/algorithm/imitation.yaml']:
        copyfile(f, f'{args.save_dir}/backup/{os.path.basename(f)}')

    all_tracks = list_variants(MAPS)
    train_tracks = [t for t in all_tracks if HOLDOUT not in t]
    eval_tracks = sorted([t for t in all_tracks if variant_map(t) in EVAL_MAPS
                          and (HOLDOUT in t or os.path.basename(t)[:-4] in [m + s for m in EVAL_MAPS for s in ('', '_rev')])])
    eval_tracks = eval_tracks[:args.eval_tracks]
    cprint(f'[data] {len(train_tracks)} training tracks, {len(eval_tracks)} eval tracks '
           f'(original geometry + held-out {HOLDOUT}), both directions', bold=True)

    actor = make_actor(args)
    actor.initialize()
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    log = []
    data_dir = f'{args.save_dir}/data'
    os.makedirs(data_dir, exist_ok=True)

    def save_round(k, O, Y, Nn):
        np.savez(f'{data_dir}/round{k:03d}.npz', O=O, Y=Y, N=Nn)

    start_rnd = 0
    if args.resume and sorted(glob.glob(f'{data_dir}/round*.npz')):
        # continue: saved data + the newest round checkpoint (actor + normalizer)
        files = sorted(glob.glob(f'{data_dir}/round*.npz'))
        data_O, data_Y, data_N = [], [], []
        for f in files:
            z = np.load(f)
            data_O.append(z['O']); data_Y.append(z['Y']); data_N.append(z['N'])
        ck = torch.load(sorted(glob.glob(f'{args.save_dir}/checkpoint/model_round*.pt'))[-1])
        actor.load_state_dict(ck['actor']); rms.mean, rms.var = ck['obs_mean'], ck['obs_var']
        if os.path.isfile(f'{args.save_dir}/train_log.pkl'):
            log = pickle.load(open(f'{args.save_dir}/train_log.pkl', 'rb'))
        start_rnd = len(files) - 1
        while sum(len(o) for o in data_O) > args.max_data and len(data_O) > 2:
            del data_O[1], data_Y[1], data_N[1]
        cprint(f'[resume] {len(files)} data rounds ({sum(len(o) for o in data_O)} samples), continuing at round {start_rnd}', bold=True)
    else:
        if args.init:
            # start from an existing model: its normalizer stays fixed so the actor keeps working
            src = f'results/{args.init}'
            rms.load(f'{src}/sim2real', required=True)
            actor.load_state_dict(torch.load(f'{src}/checkpoint/model.pt')['actor'])
            cprint(f'[init] actor + normalizer from {src}', bold=True)
        # round 0: expert drives (DART)
        t0 = time.time()
        O, Y, stats, Nn = collect(args, args.expert_steps, None, 1.0, args.dart_std, train_tracks, 0)
        succ = np.mean([s[1] for s in stats])
        cprint(f'[round 0] expert+DART {len(O)} steps, {len(stats)} episodes, 2-lap success {succ*100:.0f}% '
               f'({time.time()-t0:.0f}s)', color='yellow')
        if not args.init:
            rms.update(O)
        data_O, data_Y, data_N = [O], [Y], [Nn]
        save_round(0, O, Y, Nn)

    s2r = args.sim2real
    eval_desc = (f"delay {s2r['steer_delay_nominal']} s, a_max {s2r['a_max_nominal']}, noise, "
                 f"start/cap {EVAL_SETTINGS}")
    for rnd in range(start_rnd, args.dagger_iters + 1):
        Oall, Yall, Nall = np.concatenate(data_O), np.concatenate(data_Y), np.concatenate(data_N)
        t0 = time.time()
        vl = train_bc(args, actor, Oall, Yall, rms, args.grad_steps,
                      args.lr if (rnd == 0 and not args.init) else args.lr * 0.5, Nall)
        policy = ({k: v.clone() for k, v in actor.state_dict().items()}, rms.mean.copy(), rms.var.copy())
        save(args, actor, rms, tag=f'round{rnd}')
        succ, lap, res = evaluate(args, policy, eval_tracks, 'eval')
        cprint(f'[round {rnd}] data {len(Oall)} | bc val {vl:.5f} ({time.time()-t0:.0f}s) | '
               f'EVAL ({eval_desc}): 2-lap success {succ*100:.1f}% '
               f'mean uncapped 2-lap time {lap:.1f}s',
               color='cyan', bold=True)
        for r in res:
            if not r[1]:
                print(f'      fail {r[0]:28s} {r[2]:6.1f}s')
        log.append({'round': rnd, 'data': len(Oall), 'val': vl, 'success': succ, 'lap': lap, 'res': res})
        if not log[:-1] or (succ, -np.nan_to_num(lap, nan=1e9)) >= max((l['success'], -np.nan_to_num(l['lap'], nan=1e9)) for l in log[:-1]):
            save(args, actor, rms)
            cprint(f'[round {rnd}] saved as best -> {args.save_dir}/sim2real', color='green')
        with open(f'{args.save_dir}/train_log.pkl', 'wb') as f:
            pickle.dump(log, f)
        if rnd == args.dagger_iters:
            break
        # DAgger: the student drives (expert drives a fraction beta of the episodes), expert labels
        beta = max(0.5 ** (rnd + 1), args.min_beta)
        t0 = time.time()
        O, Y, stats, Nn = collect(args, args.dagger_steps, policy, beta, args.dart_std, train_tracks, rnd + 1)
        save_round(rnd + 1, O, Y, Nn)
        stu = [s for s in stats if not s[0]]
        cprint(f'[dagger {rnd+1}] beta {beta:.2f}: {len(O)} steps, student 2-lap success '
               f'{np.mean([s[1] for s in stu])*100 if stu else float("nan"):.0f}% over {len(stu)} episodes '
               f'({time.time()-t0:.0f}s)', color='yellow')
        data_O.append(O)
        data_Y.append(Y)
        data_N.append(Nn)
        # memory cap (23 GB RAM): drop the oldest DAgger rounds, always keep the expert round 0
        # (the files stay on disk; --resume reloads them all, so keep max_data in mind there too)
        while sum(len(o) for o in data_O) > args.max_data and len(data_O) > 2:
            del data_O[1], data_Y[1], data_N[1]
            print(f'      [data] cap {args.max_data:.0f}: dropped the oldest DAgger round', flush=True)


if __name__ == '__main__':
    main()
