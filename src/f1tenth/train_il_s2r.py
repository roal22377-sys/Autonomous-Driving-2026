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
from utils.track_gen import list_variants, variant_map, require_fresh_variants
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
EXPERT = {'a_lat': 3.5, 'lad_base': 0.6, 'lad_gain': 0.22, 'merge_dist': 0.6}
# the expert corners at this share of the episode's grip (mu * g; sim tires saturate there since 11/23).
# A fixed 3.5 m/s^2 failed on low-grip episodes (mu 0.40-0.45: 62 % 2-lap success, 11/23 check).
EXPERT_GRIP_SHARE = 0.6
# ... and never drives faster than its commanded steering allows at this share of the grip (merging after a
# start or a disturbance). 11/23, expert only, 2 laps: 94.7 % -> 100 % on the eval tracks, open-space 87.5 % -> 97.7 %
EXPERT_STEER_GRIP_SHARE = 0.7
# 11/23 comparisons (expert alone, 2 laps; eval / open-space / worst / 2-lap time):
#   'cap'  100 / 97.7 %, but taught "slow down and steer hard" in DAgger states (IL run 9: slow, jittery)
#   'hold' 100 / 96.6 / 79.5 %, 156.4 s   <- used: no extra slow-down labels, not slower than no rule
#   'off'  94.7 / 87.5 %, 157.1 s;  'hold_off' (only off the line) 93.9 / 87.5 %;  'start' 98.5 / 88.6 %
#   longer look-ahead (lad_base 1.0/1.4, lad_gain 0.30/0.35, off-line +0.8 m/m) smoothed the labels but cut corners:
#   eval 67-98 %, open-space 52-91 %, 3-5 s slower -> lad_base 0.6, lad_gain 0.22 kept
EXPERT_STEER_SPEED_MODE = 'hold'    # utils/rule_based.py RacelinePlanner steer_speed_mode: 'cap' | 'hold' | 'hold_off' | 'start' | 'off'
EXPERT_LAD_OFF_GAIN = 0.0           # look-ahead + this x (distance from the race line, <= 1 m)
EXPERT_MU = None                    # None: the episode's true friction; a number: plan speeds with this mu always
                                    # (0.45 fixed: open-space 100 % but 2.5 s slower over 2 laps -> not used)


def expert_kwargs(env):
    '''
        expert planner settings for this episode from the true (randomized) friction: cornering and
        braking at EXPERT_GRIP_SHARE of mu * g (braking into a corner unloads the rear tires and spun
        the car with a fixed 3 m/s^2), acceleration up to the motor limit a_max
    '''
    grip = float(EXPERT_MU if EXPERT_MU else env.params['mu']) * 9.81
    a_max = float(env.params['a_max'])
    return dict(EXPERT, a_lat=EXPERT_GRIP_SHARE * grip, a_acc=a_max, a_brake=min(a_max, EXPERT_GRIP_SHARE * grip),
                a_lat_steer=EXPERT_STEER_GRIP_SHARE * grip, steer_speed_mode=EXPERT_STEER_SPEED_MODE,
                lad_off_gain=EXPERT_LAD_OFF_GAIN)
STEER_LOSS_WEIGHT = 2.0         # steering errors matter more than speed errors


def task_config():
    '''task yaml in use: --task_config (via F1_TASK_CONFIG, also seen by workers) or the default'''
    return os.environ.get('F1_TASK_CONFIG', 'configs/task/f1tenth.yaml')


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--task_config', default=None,
                   help='task yaml instead of configs/task/f1tenth.yaml (e.g. configs/task/f1tenth_lidar60.yaml '
                        'for the lidar A/B); saved as backup/f1tenth.yaml so agent_node reads it')
    p.add_argument('--name', default='imitation')
    p.add_argument('--algo_idx', type=int, default=7)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=20,
                   help='simulation processes (~0.4 GB each): 20 workers peaked at 9.1 GB + data 2.2 GB on the 15.5 GB WSL PC')
    p.add_argument('--bc_threads', type=int, default=16, help='torch threads for BC (WSL sees 16 cores x 2)')
    p.add_argument('--expert_steps', type=float, default=1.2e6, help='DART expert rollout steps')
    p.add_argument('--dagger_iters', type=int, default=20,
                   help='maximum DAgger rounds; the early stop below usually ends the run sooner')
    # early stop (11/23): stop when the best score has not improved noticeably for `patience` rounds in a row
    # 11/23: run 9 stopped after round 5 while the score still swung +-8 points between rounds -> more patience
    p.add_argument('--patience', type=int, default=3, help='rounds in a row without a noticeable improvement')
    p.add_argument('--min_rounds', type=int, default=5, help='never stop before this round')
    p.add_argument('--min_gain', type=float, default=0.01, help='score gain that counts as an improvement (0.01 = 1 %p)')
    p.add_argument('--jitter_gain', type=float, default=0.10,
                   help='at about the same score: this much less steering jitter counts as an improvement (0.10 = 10 %%)')
    # open-space check in the round score (eval_open_space.py test tracks: a wide dead-end bay at every corner)
    p.add_argument('--open_eval', type=int, default=1, help='score = mean of eval and open-space 2-lap success')
    p.add_argument('--open_level', default='big')
    p.add_argument('--open_caps', nargs='+', type=float, default=[8.0, 5.0], help='speed caps of the open-space runs')
    p.add_argument('--dagger_steps', type=float, default=6e5, help='steps per DAgger iteration')
    p.add_argument('--dart_std', type=float, default=0.15)
    p.add_argument('--grad_steps', type=int, default=15000, help='gradient steps after each data round')
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden_dim', type=int, default=512, help='actor width (2 hidden layers); run 7 used 256')
    p.add_argument('--eval_tracks', type=int, default=44)
    p.add_argument('--max_data', type=float, default=8e6,      # float16 observations (11/23): 8e6 fits the 15 GB WSL PC
                   help='keep at most this many samples (oldest DAgger rounds dropped, expert round kept)')
    p.add_argument('--min_beta', type=float, default=0.1, help='expert-driven episode fraction never below this')
    p.add_argument('--init', default=None, help='start from this model (e.g. imitation/7): its actor + normalizer, '
                                               'and go straight to DAgger with it driving')
    p.add_argument('--resume', action='store_true', help='reload the data saved in <save_dir>/data and continue')
    # CAPS (Mysore et al. 2021) smoothness regularization: removes high-frequency steering jitter without
    # adding delay. temporal: pi(s_t) ~ pi(s_t+1); spatial: pi(s) ~ pi(s + noise)
    p.add_argument('--caps_t', type=float, default=0.5, help='temporal smoothness weight (0 = off); 0.5 since 11/24 '
                                                              '(run 8-10: 0.3, straight-line weaving)')
    p.add_argument('--caps_s', type=float, default=0.3, help='spatial smoothness weight (0 = off)')
    p.add_argument('--caps_sigma', type=float, default=0.05, help='spatial noise std in normalized obs units')
    p.add_argument('--hist_noise', type=float, default=0.0,
                   help='gaussian noise (normalized obs units) on the action-history inputs during BC: the student\'s '
                        'own jittery commands are in its input, noise keeps it from copying them (11/23)')
    a = p.parse_args(argv)
    if a.task_config:
        # workers call get_args([]) again; the environment variable carries the choice to them
        os.environ['F1_TASK_CONFIG'] = a.task_config
    args = EasyDict(vars(a))
    for f in [task_config(), 'configs/task/dynamic.yaml', 'configs/algorithm/imitation.yaml']:
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
# ------------------------------------------------------------------ progress (shared counter)
_COUNTER = None
T_START = time.time()


def _init_counter(c):
    global _COUNTER
    _COUNTER = c


def _tick(n):
    if _COUNTER is not None:
        with _COUNTER.get_lock():
            _COUNTER.value += n


def _hms(sec):
    sec = int(max(sec, 0))
    return f'{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}'


def run_pool(fn, jobs, workers, total, label, unit, every=60):
    '''pool.map with a progress line every `every` s: done/total, elapsed, ETA, time since start'''
    ctx = mp.get_context('spawn')
    counter = ctx.Value('l', 0)
    t0 = time.time()
    with ctx.Pool(workers, initializer=_init_counter, initargs=(counter,)) as pool:
        ar = pool.map_async(fn, jobs)
        while not ar.ready():
            ar.wait(every)
            if ar.ready():
                break
            done, el = counter.value, time.time() - t0
            eta = el / done * (total - done) if done > 0 else float('nan')
            print(f'    [{label}] {min(done, total)}/{total} {unit} ({min(done / total, 1) * 100:.0f}%) | '
                  f'elapsed {_hms(el)} | ETA {_hms(eta) if done > 0 else "?"} | total {_hms(time.time() - T_START)}',
                  flush=True)
        return ar.get()


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
                             max_steer=env.max_steer, max_steer_left=env.steer_left_max, **expert_kwargs(env))
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
        _tick(steps)
        stats.append((expert_drives, int(env.success), steps))
    env.close()
    return np.asarray(O, np.float32), np.asarray(Y, np.float32), stats, np.asarray(N, bool)


def collect(args, n_steps, policy, beta, dart_std, tracks, round_idx):
    per = int(np.ceil(n_steps / args.workers))
    jobs = [(w, per, policy, beta, dart_std, tracks, args.seed * 100000 + round_idx * 1000 + w)
            for w in range(args.workers)]
    res = run_pool(rollout_worker, jobs, args.workers, per * args.workers, f'collect round {round_idx}', 'steps')
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
    _tick(1)
    st = np.asarray(steers) * env.max_steer
    return f'{os.path.basename(path)[:-4]}@{cap:g}/{start}', ok, float(env.elapsed), float(np.abs(np.diff(st)).mean())


# (speed cap [m/s], start): uncapped from the centerline (lap time), uncapped and car-practice cap
# from random starts anywhere across the track with a heading error (merging onto the race line)
# the cap is an upper limit: within it the policy picks its own speed (0.5-8 m/s) for every track and corner.
# 8 = no limit (lap time is compared there), 5 = the car's agent.yaml speed_limit these days (was 3 up to run 8).
# Starts like the car's (11/23): centerline, and one mildly disturbed start (utils/env.py START_MAX_*)
EVAL_SETTINGS = ((8.0, 'center'), (5.0, 'center'), (8.0, 'random'))
EVAL_CAPS = tuple(c for c, _ in EVAL_SETTINGS)


def evaluate(args, policy, tracks, mode='eval', settings=EVAL_SETTINGS):
    '''success over all runs; lap = mean 2-lap time of the first setting's runs that finished'''
    jobs = [(p, policy, mode, 7 + i, cap, start) for cap, start in settings for i, p in enumerate(sorted(tracks))]
    res = run_pool(eval_worker, jobs, min(args.workers, len(jobs)), len(jobs), 'eval', 'episodes')
    succ = np.mean([r[1] for r in res])
    held = [r[1] for r in res if HOLDOUT in r[0]]
    print(f'      held-out tracks ({HOLDOUT}, never trained on): success {np.mean(held)*100:.1f}% '
          f'| training geometries: {np.mean([r[1] for r in res if HOLDOUT not in r[0]])*100:.1f}%', flush=True)
    fast = [r for r in res if r[0].endswith(f'@{settings[0][0]:g}/{settings[0][1]}') and r[1]]
    lap = np.mean([r[2] for r in fast]) if fast else float('nan')
    return succ, lap, res


def paired_gain(res, ref_res, settings, relative=False):
    '''
        time saved (ref - candidate) over the runs of `settings` both finished, mean [s] or relative
        (fraction of the ref time), and their number. Means over different finished tracks are not comparable.
    '''
    ok_c = {r[0]: r[2] for r in res if r[1] and any(r[0].endswith(f'@{c:g}/{st}') for c, st in settings)}
    ok_r = {r[0]: r[2] for r in ref_res if r[1]}
    common = sorted(set(ok_c) & set(ok_r))
    if not common:
        return float('nan'), 0
    d = [(ok_r[k] - ok_c[k]) / ok_r[k] if relative else ok_r[k] - ok_c[k] for k in common]
    return float(np.mean(d)), len(common)


UNCAPPED = tuple(s for s in EVAL_SETTINGS if s[0] >= 8.0)


def round_jitter(res):
    '''mean |steering change| per step over the evaluation runs [rad] (eval_worker's 4th field)'''
    return float(np.mean([r[3] for r in res])) if res else float('nan')


SCORE_TIE = 0.005       # scores within 0.5 %p count as equal (evaluation noise)


def judge_round(entry, best, args):
    '''
        (better, noticeable, text) for a round against the best round so far. IL aims at stable driving
        (11/24): between rounds of about the same score the smoother one (less steering jitter) wins; lap time
        is left to PPO.
        better:     score higher by SCORE_TIE, or about the same score and less jitter -> saved as best
        noticeable: score + min_gain, or about the same score and jitter_gain less jitter -> resets the early stop
    '''
    if best is None:
        return True, True, 'first round'
    ds = entry['score'] - best['score']
    j, jb = entry.get('jitter', round_jitter(entry['res'])), best.get('jitter', round_jitter(best['res']))
    smoother = np.isfinite(j) and np.isfinite(jb) and j < jb
    better = ds > SCORE_TIE or (ds >= -SCORE_TIE and smoother)
    noticeable = ds >= args.min_gain - 1e-9 or (ds >= -SCORE_TIE and np.isfinite(j) and np.isfinite(jb)
                                                 and j <= (1.0 - args.jitter_gain) * jb)
    g, n = paired_gain(entry['res'], best['res'], UNCAPPED, relative=True)
    text = (f"score {ds*100:+.1f}%p vs best (round {best['round']}), steering jitter {j:.4f} vs {jb:.4f} rad"
            + (f", {g*100:+.1f}% faster on {n} uncapped runs both finished" if n else ''))
    return better, noticeable, text


# ------------------------------------------------------------------ behavior cloning
def train_bc(args, actor, O, Y, rms, grad_steps, lr, Nn=None):
    torch.set_num_threads(args.bc_threads)
    # normalize in chunks straight into float32: rms.normalize on everything at once makes a float64 copy
    # (6.8 GB at 10.7M samples; IL run 8 was OOM-killed at 20.8 of 23 GB)
    # float16 storage (11/23): half the memory, so twice the data fits (normalized values, ~1e-3 resolution);
    # every batch is cast back to float32
    X = torch.empty((len(O), O.shape[1]), dtype=torch.float16)
    for i in range(0, len(O), 500_000):
        X[i:i + 500_000] = torch.as_tensor(rms.normalize(O[i:i + 500_000].astype(np.float32)), dtype=torch.float16)
    T = torch.as_tensor(Y, dtype=torch.float32)
    n = len(X)
    perm = torch.randperm(n)
    nval = max(2000, n // 50)
    val, tr = perm[:nval], perm[nval:]
    # samples whose next sample is the next step of the same episode (CAPS temporal term)
    nxt = torch.as_tensor(np.where(Nn[:-1])[0]) if (Nn is not None and args.caps_t > 0) else None
    w = torch.tensor([STEER_LOSS_WEIGHT, 1.0])
    from utils.observation import get_obs_config
    oc = get_obs_config(args)
    h0 = oc['lidar']['num_beams'] + (1 if oc['speed']['enabled'] else 0)
    hist = slice(h0, h0 + 2 * oc['history']['length'])          # action-history columns of the observation
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=grad_steps, eta_min=lr * 0.05)
    actor.train()
    t0 = time.time()
    for it in range(1, grad_steps + 1):
        idx = tr[torch.randint(len(tr), (args.batch,))]
        xb = X[idx].float()
        if args.hist_noise > 0:
            xb[:, hist] += args.hist_noise * torch.randn_like(xb[:, hist])
        out = actor(xb).loc
        loss = ((out - T[idx]) ** 2 * w).mean()     # MSE: run 9 already steered too little where large steering
                                                    # was needed; a Huber loss would weaken exactly those labels
        if args.caps_s > 0:
            noisy = actor(xb + args.caps_sigma * torch.randn_like(xb)).loc
            loss = loss + args.caps_s * ((noisy - out) ** 2).mean()
        if nxt is not None:
            j = nxt[torch.randint(len(nxt), (args.batch // 2,))]
            loss = loss + args.caps_t * ((actor(X[j + 1].float()).loc - actor(X[j].float()).loc) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        opt.step()
        sched.step()
        if it % 2000 == 0 or it == grad_steps:
            with torch.no_grad():
                vl = ((actor(X[val].float()).loc - T[val]) ** 2 * w).mean().item()      # MSE, comparable to earlier runs
            el = time.time() - t0
            print(f'    [bc] step {it}/{grad_steps} train {loss.item():.5f} val {vl:.5f} | '
                  f'elapsed {_hms(el)} | ETA {_hms(el / it * (grad_steps - it))} | total {_hms(time.time() - T_START)}', flush=True)
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
    for f, name in [(task_config(), 'f1tenth.yaml'), ('configs/task/dynamic.yaml', 'dynamic.yaml'),
                    ('configs/algorithm/imitation.yaml', 'imitation.yaml')]:
        copyfile(f, f'{args.save_dir}/backup/{name}')

    all_tracks = list_variants(MAPS)
    require_fresh_variants(all_tracks)          # never train on tracks from an older generator
    train_tracks = [t for t in all_tracks if HOLDOUT not in t]
    eval_tracks = sorted([t for t in all_tracks if variant_map(t) in EVAL_MAPS
                          and (HOLDOUT in t or os.path.basename(t)[:-4] in [m + s for m in EVAL_MAPS for s in ('', '_rev')])])
    eval_tracks = eval_tracks[:args.eval_tracks]
    open_tracks, open_settings = [], ()
    if args.open_eval:
        from eval_open_space import TEST_DIR
        open_tracks = sorted(glob.glob(os.path.join(TEST_DIR, args.open_level, '*.npz')))
        open_settings = tuple((c, 'center') for c in args.open_caps)
        if not open_tracks:
            raise FileNotFoundError(f'no open-space test tracks in {TEST_DIR}/{args.open_level}: run '
                                    '`python -u eval_open_space.py --make` first (or --open_eval 0)')
        cprint(f'[data] open-space check: {len(open_tracks)} {args.open_level} tracks x caps {list(args.open_caps)}, 2 laps', bold=True)
    cprint(f'[data] {len(train_tracks)} training tracks, {len(eval_tracks)} eval tracks '
           f'(original geometry + held-out {HOLDOUT}), both directions', bold=True)

    actor = make_actor(args)
    actor.initialize()
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    log = []
    data_dir = f'{args.save_dir}/data'
    os.makedirs(data_dir, exist_ok=True)

    def save_round(k, O, Y, Nn):
        np.savez(f'{data_dir}/round{k:03d}.npz', O=O.astype(np.float16), Y=Y, N=Nn)

    start_rnd = 0
    if args.resume and sorted(glob.glob(f'{data_dir}/round*.npz')):
        # continue: saved data + the newest round checkpoint (actor + normalizer)
        files = sorted(glob.glob(f'{data_dir}/round*.npz'))
        data_O, data_Y, data_N = [], [], []
        for f in files:
            z = np.load(f)
            data_O.append(z['O'].astype(np.float16)); data_Y.append(z['Y']); data_N.append(z['N'])
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
        O = O.astype(np.float16)            # kept as float16 (memory), see train_bc
        data_O, data_Y, data_N = [O], [Y], [Nn]
        save_round(0, O, Y, Nn)

    # best round so far and rounds without a noticeable improvement (replayed from the log on --resume)
    best, stale = None, 0
    for e in log:
        e.setdefault('score', e.get('success', 0.0))
        b, n, _ = judge_round(e, best, args)
        stale = 0 if n else stale + 1
        if b:
            best = e

    s2r = args.sim2real
    eval_desc = (f"delay {s2r['steer_delay_nominal']} s, left steer {s2r.get('steer_left_max_nominal', args.max_steer)} rad, a_max {s2r['a_max_nominal']}, noise, "
                 f"start/cap {EVAL_SETTINGS}")
    for rnd in range(start_rnd, args.dagger_iters + 1):
        Oall, Yall, Nall = np.concatenate(data_O), np.concatenate(data_Y), np.concatenate(data_N)
        cprint(f'[round {rnd}] BC on {len(Oall)} samples -> eval ({time.strftime("%m-%d %H:%M")}, total {_hms(time.time() - T_START)})', bold=True)
        t0 = time.time()
        vl = train_bc(args, actor, Oall, Yall, rms, args.grad_steps,
                      args.lr if (rnd == 0 and not args.init) else args.lr * 0.5, Nall)
        n_data = len(Oall)
        del Oall, Yall, Nall            # the merged copy is only for BC; free it before evaluation (15 GB PC)
        policy = ({k: v.clone() for k, v in actor.state_dict().items()}, rms.mean.copy(), rms.var.copy())
        save(args, actor, rms, tag=f'round{rnd}')
        succ, lap, res = evaluate(args, policy, eval_tracks, 'eval')
        op = evaluate(args, policy, open_tracks, 'eval', settings=open_settings)[0] if args.open_eval else float('nan')
        score = 0.5 * (succ + op) if args.open_eval else succ
        cprint(f'[round {rnd}] data {n_data} | bc val {vl:.5f} ({time.time()-t0:.0f}s) | '
               f'EVAL ({eval_desc}): 2-lap success {succ*100:.1f}% '
               f'mean uncapped 2-lap time {lap:.1f}s'
               + (f' | open-space ({args.open_level}) 2-lap success {op*100:.1f}% | score {score*100:.1f}' if args.open_eval else ''),
               color='cyan', bold=True)
        for r in res:
            if not r[1]:
                print(f'      fail {r[0]:28s} {r[2]:6.1f}s')
        entry = {'round': rnd, 'data': n_data, 'val': vl, 'success': succ, 'lap': lap, 'open': op, 'score': score,
                 'jitter': round_jitter(res), 'res': res}
        log.append(entry)
        better, noticeable, why = judge_round(entry, best, args)
        stale = 0 if noticeable else stale + 1
        cprint(f'[round {rnd}] {why} -> {"noticeable improvement" if noticeable else f"no noticeable improvement ({stale}/{args.patience})"}',
               color='green' if noticeable else 'yellow')
        if better:
            best = entry
            save(args, actor, rms)
            cprint(f'[round {rnd}] saved as best -> {args.save_dir}/sim2real', color='green')
        with open(f'{args.save_dir}/train_log.pkl', 'wb') as f:
            pickle.dump(log, f)
        if rnd == args.dagger_iters:
            cprint(f'[stop] reached the maximum of {args.dagger_iters} DAgger rounds; best is round {best["round"]}', bold=True)
            break
        if rnd >= args.min_rounds and stale >= args.patience:
            cprint(f'[stop] early stop after round {rnd}: {args.patience} rounds in a row without a noticeable improvement '
                   f'(score +{args.min_gain*100:g}%p or {args.jitter_gain*100:g}% less jitter). Best is round {best["round"]} '
                   f'(score {best["score"]*100:.1f}) -> {args.save_dir}/sim2real', bold=True)
            break
        # DAgger: the student drives (expert drives a fraction beta of the episodes), expert labels
        beta = max(0.5 ** (rnd + 1), args.min_beta)
        t0 = time.time()
        O, Y, stats, Nn = collect(args, args.dagger_steps, policy, beta, args.dart_std, train_tracks, rnd + 1)
        O = O.astype(np.float16)
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
