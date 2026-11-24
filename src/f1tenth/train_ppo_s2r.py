'''
    PPO fine-tuning of the sim2real IL policy for lap time, run from src/f1tenth:

        python -u train_ppo_s2r.py --il imitation/9 --algo_idx 1 --workers 22             # recommended
        python -u train_ppo_s2r.py --il imitation/9 --algo_idx 0 --workers 22 --baseline  # old settings
        python -u train_ppo_s2r.py --il imitation/9 --algo_idx 1 --resume                  # continue a stopped run

    - starts from the IL actor (results/<il>/checkpoint/model.pt) and its obs normalizer (frozen),
      on the IL model's own task config (lidar layout, obs_dim, steering calibration)
    - same simulator as IL training: sim_mode='train' (delay, accel limit, friction, noise,
      generated tracks both directions, 2-lap episodes)
    - reward (utils/env.py calc_reward, NOTES_1119_sim2real_rl.md section 4):
          10 * progress [m] - time cost per step       -> with fixed-distance episodes = minimum lap time
          - 200 on a crash, - steering-rate penalty    -> weaving is slow and fragile on the car
          - wall-margin and grip-limit penalties       -> safety margin on the real car
          + optional small bonus for beating the expert-pace lap estimate
    - returns are normalized (running std of the discounted return), so the critic learns targets of
      order 1 instead of hundreds
    - asymmetric actor-critic: the critic also sees simulator-only state (utils/env.py
      F1Wrapper.privileged). The actor (what runs on the car) sees only lidar + speed + history + cap
    - critic warm-up with the actor frozen until the critic explains the returns (explained variance),
      then PPO (clip + KL early stop) with a behavior-cloning term towards the IL policy whose weight
      decays, and is raised again if the evaluation success drops below the IL policy's; the speed output is
      pulled only bc_speed (0.25) as hard as the steering (IL is moderately fast on purpose, PPO is for lap time)
    - completion as a constraint (--crash_adapt, PPO 3): after each evaluation below IL - 2 %p the collision
      cost goes x1.5 (up to 2000), at or above IL it eases x0.8 back towards --collision_cost; the BC term only
      decays (no boost), so safety comes from the reward and the policy is free to leave the IL line / speeds
    - --init: continue from an earlier PPO checkpoint (IL stays the baseline), --baseline_from: reuse its IL eval
    - failure-weighted training tracks (--track_focus): a track is drawn with weight 1 + focus x its recent
      failure rate, so the episodes go where the policy still crashes (dead-end bays, hairpins)
    - an episode that makes no progress (stuck) or drives the wrong way ends as a crash (--stuck_time,
      --backward_dist), e.g. after turning around in a bay; the evaluations keep the normal rules
    - separate, decaying exploration noise for steering (small: steering noise under delay = crashes)
      and speed; actor learning rate decays linearly
    - GAE that bootstraps from the true final observation on time-limit truncation
    - model selection: saved only if it finishes at least as many evaluation runs as the best model so
      far, is faster on the runs both finished (paired), and does not finish fewer worst-case runs
      (longest delay, lowest grip) than the IL policy, and does not finish fewer 2-lap runs than the IL
      policy on the open-space test tracks (eval_open_space.py: a wide dead-end bay at every corner,
      where the 11/23 car drove in and hit the end wall). Otherwise the IL policy stays the deployed one
    - --baseline reproduces the settings before 11/23 (first run of NOTES section 6 step 3), to compare
    Output: results/ppo_s2r/<algo_idx>/ (same layout as the other algorithms), train_log.pkl per update
'''
import argparse
import os
import sys
import time
import glob
import pickle
import functools
from shutil import copyfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'installation'))
import numpy as np
import torch
import gymnasium as gym
from easydict import EasyDict

import train_il_s2r as IL
from utils.track_gen import variant_map, require_fresh_variants
from utils.env import F1Wrapper
from utils.vectorize import RunningMeanStd
from utils.color import cprint
from eval_open_space import TEST_DIR as OPEN_TEST_DIR
from algorithm.common.models import Critic, LOG_STD_MIN

REWARD_KEYS = ('forward_reward', 'collision_cost', 'steer_cost', 'straight_cost', 'time_cost', 'wall_cost', 'lat_cost', 'lap_bonus')

# settings before 11/23, for the baseline run (--baseline)
BASELINE = dict(hard_prob=0.0, open_train_share=0.0, track_focus=0.0, stuck_time=0.0, bc_speed=1.0, ret_norm=0, time_cost=0.0, steer_rate_cost=5.0, wall_cost=0.0, lat_cost=0.0, lap_bonus=0.0,
                log_std_steer=-2.0, log_std_speed=-2.0, log_std_decay=0.0, uncapped_prob=0.25,
                lr_end=5e-5, max_kl=0.02, bc_end=0.05, warmup_ev=-1e9, bc_adapt=0, worst_eval=0)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--il', required=True, help='IL model to start from, e.g. imitation/9')
    p.add_argument('--name', default='ppo_s2r')
    p.add_argument('--algo_idx', type=int, default=1)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=16,
                   help='simulation processes for rollouts (~0.4 GB each, they stay alive during evaluation)')
    p.add_argument('--eval_workers', type=int, default=8,
                   help='processes for evaluation, started while the rollout workers are alive: '
                        '16 + 8 fits the 15.5 GB WSL training PC (20 + 20 would not)')
    p.add_argument('--resume', action='store_true',
                   help='continue from <save_dir>/checkpoint/resume.pt (same command + --resume)')
    p.add_argument('--ckpt_every', type=int, default=5, help='updates between resume checkpoints')
    p.add_argument('--total_steps', type=float, default=12e6)
    p.add_argument('--n_steps', type=int, default=44000, help='rollout steps per update (all envs)')
    p.add_argument('--minibatch', type=int, default=2048)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-5, help='actor learning rate at the start')
    p.add_argument('--lr_end', type=float, default=1e-5, help='... decayed linearly to this')
    p.add_argument('--critic_lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--gae', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.15)
    p.add_argument('--max_kl', type=float, default=0.015)
    # exploration noise (log std of the gaussian around the actor mean), per action, decayed linearly
    p.add_argument('--log_std_steer', type=float, default=-2.3)
    p.add_argument('--log_std_speed', type=float, default=-1.5)
    p.add_argument('--log_std_decay', type=float, default=0.5, help='both log stds go down by this by the end')
    # critic warm-up: actor frozen for at least critic_warmup updates and until explained variance >= warmup_ev
    p.add_argument('--critic_warmup', type=int, default=3)
    p.add_argument('--warmup_ev', type=float, default=0.6)
    p.add_argument('--max_warmup', type=int, default=20, help='start the actor after this many updates anyway')
    # behavior cloning towards the IL actor
    p.add_argument('--bc_start', type=float, default=0.5)
    p.add_argument('--bc_end', type=float, default=0.02)
    p.add_argument('--bc_steer_weight', type=float, default=0.0,
                   help='k > 0: the steering BC term of each state is weighted 1 + k x (teacher steering)^2, so the rare '
                        'states that need near full lock (hairpins) are not forgotten while the rest drifts freely. 11/25 '
                        'PPO 8: Shanghai_rev failed from update 20, Montreal_rev from 40 (unwinding mid-hairpin, at every '
                        'cap) although the IL teacher drives them; PPO 7 kept them. PPO 11: 8')
    p.add_argument('--bc_speed', type=float, default=0.25,
                   help='BC weight of the speed output relative to steering: IL drives deliberately moderately '
                        'fast, so pulling the speed back to it works against the lap time (1 = same as steering)')
    p.add_argument('--bc_adapt', type=int, default=0,
                   help='1: double the BC weight when eval success drops below IL (PPO 2; it kept the policy near IL '
                        'instead of letting it learn to be safe itself -> PPO 3 adapts the collision cost instead)')
    p.add_argument('--bc_max', type=float, default=8.0,
                   help='the boosted BC weight can reach this x bc_start (PPO 1 capped it at bc_start: no effect early on)')
    # reward (utils/env.py calc_reward)
    p.add_argument('--ret_norm', type=int, default=1, help='scale rewards by the running std of the discounted return')
    p.add_argument('--progress_coeff', type=float, default=10.0)
    p.add_argument('--collision_cost', type=float, default=500.0,
                   help='PPO 1 (200) traded completion for speed: 87.9 -> 50.8 %% success in 40 updates')
    p.add_argument('--steer_rate_cost', type=float, default=3.0)
    p.add_argument('--straight_steer_cost', type=float, default=0.0,
                   help='extra steering-rate cost where the track ahead is straight (radius > 1/--straight_curv): '
                        'the 11/24 car weaved +-20 cm on straights; corners keep --steer_rate_cost only')
    p.add_argument('--straight_curv', type=float, default=0.05)
    p.add_argument('--straight_v_ref', type=float, default=0.0,
                   help='[m/s] > 0: the straight-line cost is per meter (equal to per step at this speed), so slow '
                        'driving does not make corrections dearer (PPO 8: 4)')
    p.add_argument('--time_cost', type=float, default=0.1)
    p.add_argument('--wall_cost', type=float, default=5.0)
    p.add_argument('--wall_margin', type=float, default=0.30)
    p.add_argument('--lat_cost', type=float, default=2.0)
    p.add_argument('--lat_grip_share', type=float, default=0.8)
    p.add_argument('--lap_bonus', type=float, default=0.0, help='per second faster than the expert-pace lap (try 2)')
    p.add_argument('--uncapped_prob', type=float, default=0.5,
                   help='share of training episodes without a speed cap (a capped car gets no speed signal)')
    p.add_argument('--hard_prob', type=float, default=0.0,
                   help='share of training episodes with every sim2real gap drawn from its harder side at once '
                        '(delay, grip, accel, steering, noise; PPO 5: 0.25). The worst-case check needs these together')
    p.add_argument('--hard_frac', type=float, default=0.4, help='... the part of each range next to its harder end')
    p.add_argument('--open_train_share', type=float, default=0.0,
                   help='share of training episodes on open-space training tracks (a big dead-end bay at every corner, '
                        'eval_open_space.py --make_train; PPO 5: 0.2). The open-space test tracks stay unseen')
    p.add_argument('--track_focus', type=float, default=3.0,
                   help='training track weight = 1 + this x the track\'s recent failure rate (0 = uniform)')
    p.add_argument('--focus_by_group', type=int, default=0,
                   help='1: that failure rate per map and direction (all its variants) instead of per file. 11/25 PPO 10: '
                        'failures sat on hairpin maps (Montreal_rev, Shanghai_rev: every cap, from u40), while with ~1000 '
                        'files each came up once in ~40 updates and the hardest 20 %% got only 20-25 %% of the weight')
    p.add_argument('--stuck_time', type=float, default=3.0,
                   help='[s] end the episode as a crash after this long without --stuck_dist m of progress (0 = off)')
    p.add_argument('--stuck_dist', type=float, default=0.5)
    p.add_argument('--backward_dist', type=float, default=2.0, help='[m] wrong-way distance that ends it as a crash')
    # evaluation / model selection
    p.add_argument('--eval_every', type=int, default=10, help='updates between evaluations')
    p.add_argument('--eval_tracks', type=int, default=44)
    p.add_argument('--eval_low_caps', type=float, nargs='*', default=[2.0, 2.5, 3.0, 3.5, 4.0],
                   help='one more centerline run per eval track at these caps in turn (the car drives at 2-5 m/s). '
                        '11/25 PPO 7: 0/22 finished at cap 3 (drifted into the left wall on straights without '
                        'correcting) while caps 5 / 8, the only ones evaluated, still looked fine. Empty = off')
    p.add_argument('--worst_eval', type=int, default=1, help='also evaluate at the worst-case sim2real settings')
    p.add_argument('--open_eval', type=int, default=1,
                   help='also require no loss on the open-space test tracks (python -u eval_open_space.py --make)')
    p.add_argument('--full_eval', type=int, default=0,
                   help='1: run the worst-case / open-space checks at every evaluation, not only for a model that '
                        'beats the best on the eval tracks (needed to be saved anyway)')
    p.add_argument('--open_level', default='big', help='eval_open_space.py level used for model selection')
    p.add_argument('--open_caps', nargs='+', type=float, default=[5.0, 8.0], help='speed caps for that check')
    p.add_argument('--baseline', action='store_true', help='settings before 11/23 (see BASELINE)')
    p.add_argument('--sim_override', default=None,
                   help='yaml with `sim2real:` and/or `speed_limit:` entries that replace the IL model\'s ones for this PPO '
                        'run (e.g. a re-measured steering scale); the merged config is the run\'s backup/f1tenth.yaml')
    # start only from a solid IL policy (11/23 decision): PPO refines it, it should not have to repair it
    p.add_argument('--min_il_success', type=float, default=0.98, help='IL 2-lap success on the eval tracks')
    p.add_argument('--min_il_open', type=float, default=0.95, help='IL 2-lap success on the open-space tracks')
    p.add_argument('--force_start', action='store_true', help='start PPO even below those')
    # start / safety (PPO 3)
    p.add_argument('--init', default=None,
                   help='start the actor (and critic) from this PPO checkpoint under results/ instead of the IL policy, '
                        'e.g. ppo_s2r/2/checkpoint/model_1760000.pt; the IL policy stays the baseline of the save gates '
                        'and the BC term pulls towards this starting policy')
    p.add_argument('--baseline_from', default=None,
                   help='reuse the IL baseline evaluation of an earlier PPO run from the same --il (e.g. ppo_s2r/2) '
                        'instead of evaluating it again (~8 min)')
    p.add_argument('--crash_adapt', type=int, default=1,
                   help='after each evaluation: success below IL - 2 %%p -> collision cost x crash_up (up to crash_max); '
                        'at or above IL -> x crash_down (not below --collision_cost). Safety from the reward, not from IL')
    p.add_argument('--crash_target', default='il', choices=('il', 'best'),
                   help="success the collision cost aims at: 'il' = the IL policy's (fixed); 'best' = ratchet, the "
                        "best saved model's (starts at IL, only goes up, so completion keeps being pushed past IL)")
    p.add_argument('--crash_start', type=float, default=None,
                   help='collision cost at the start (default --collision_cost, which stays the floor), e.g. the cost '
                        'the --init checkpoint ended with, so its critic matches the rewards')
    p.add_argument('--cost_critic', type=int, default=0,
                   help='1: a second critic predicts the discounted crash probability; the reward critic learns the '
                        'rest, and the policy advantage is A_reward - crash_cost * A_crash. 11/25 PPO 8: one critic for '
                        'rewards of ~0.01 per step and crashes of -10000..-30000 explained the returns badly '
                        '(EV 0.2-0.66, twice -0.9), and success fell while the crash cost rose')
    p.add_argument('--crash_up', type=float, default=1.5)
    p.add_argument('--crash_down', type=float, default=0.8)
    p.add_argument('--crash_max', type=float, default=10000.0,
                   help='2000 was binding (PPO 5): with gamma 0.995 a crash then costs about as much as 10 %% more speed over '
                        '5 s at ~1 %% risk, which adds up to ~26 %% crashed episodes over 2 laps')
    a = p.parse_args()
    if a.baseline:
        vars(a).update(BASELINE)
    # use the IL model's own task config (lidar layout, obs_dim, steering calibration), not today's default
    il_task = f'results/{a.il}/backup/f1tenth.yaml'
    if not os.path.exists(il_task):
        raise FileNotFoundError(f'{il_task} missing: the IL model must come with its task config')
    os.environ['F1_TASK_CONFIG'] = merge_sim_override(il_task, a.sim_override, f'results/{a.name}/{a.algo_idx}') \
        if a.sim_override else il_task
    args = IL.get_args([])
    args.update(vars(a))
    args.save_dir = f'results/{args.name}/{args.algo_idx}'
    # time cost must stay below the progress reward at the slowest speed, or crashing pays (NOTES section 4)
    slowest = args.progress_coeff * args.min_speed * args.timestep
    if args.time_cost >= slowest:
        raise ValueError(f'--time_cost {args.time_cost} >= progress at min_speed ({slowest:.3f}/step): '
                         'driving slowly would cost more than crashing')
    return args


OVERRIDABLE = {'sim2real': None, 'speed_limit': {'range', 'uncapped_prob', 'nominal'}}


def merge_sim_override(il_task, override, save_dir):
    '''
        IL task config with the simulator conditions of `override` (sim2real, speed_limit range/caps) replaced,
        written to <save_dir>/backup/f1tenth.yaml. Everything that defines the observation or the meaning of
        the action (lidar, obs_dim, max_steer, steering gains, speed range, history) must stay the IL model's.
    '''
    from ruamel.yaml import YAML
    yaml = YAML()
    with open(il_task) as f:
        cfg = yaml.load(f)
    with open(override) as f:
        ov = yaml.load(f) or {}
    for k, v in ov.items():
        if k not in OVERRIDABLE:
            raise ValueError(f'--sim_override: `{k}` cannot be changed for PPO (it is part of the IL policy\'s '
                             f'observation/action); allowed: {sorted(OVERRIDABLE)}')
        allowed = OVERRIDABLE[k]
        for kk, vv in v.items():
            if allowed is not None and kk not in allowed:
                raise ValueError(f'--sim_override: {k}.{kk} cannot be changed (allowed: {sorted(allowed)})')
            if kk not in cfg[k]:
                print(f'[sim_override] note: {k}.{kk} is new (not in the IL config)')
            print(f'[sim_override] {k}.{kk}: {cfg[k].get(kk)} -> {vv}')
            cfg[k][kk] = vv
    os.makedirs(f'{save_dir}/backup', exist_ok=True)
    out = f'{save_dir}/backup/f1tenth.yaml'
    with open(out, 'w') as f:
        f.write(f'# IL config {il_task} with the PPO simulator override {override}\n')
        yaml.dump(cfg, f)
    return out


def env_config(args):
    '''the reward / training settings every worker applies to its env (spawned workers do not see args)'''
    return dict(PROGRESS_COEFF=args.progress_coeff, COLLISION_COST=args.crash_start or args.collision_cost,
                STEER_RATE_COST=args.steer_rate_cost, STRAIGHT_STEER_COST=args.straight_steer_cost,
                STRAIGHT_CURV=args.straight_curv, STRAIGHT_V_REF=args.straight_v_ref, TIME_COST=args.time_cost, WALL_COST=args.wall_cost,
                WALL_MARGIN=args.wall_margin, LAT_COST=args.lat_cost, LAT_GRIP_SHARE=args.lat_grip_share,
                LAP_BONUS=args.lap_bonus, uncapped_prob=args.uncapped_prob,
                STUCK_TIME=args.stuck_time, STUCK_DIST=args.stuck_dist, BACKWARD_DIST=args.backward_dist,
                HARD_PROB=args.hard_prob, HARD_FRAC=args.hard_frac, open_train=args.open_train_share > 0)


def open_train_tracks():
    from eval_open_space import TRAIN_DIR
    return sorted(glob.glob(os.path.join(TRAIN_DIR, '*.npz')))


def training_tracks(open_train):
    '''generated training tracks (held-out variant excluded) + the open-space training tracks if used'''
    tracks = [t for t in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in t]
    return tracks + (open_train_tracks() if open_train else [])


def base_track_weights(args):
    '''{track basename: weight} so that the open-space training tracks get --open_train_share of the episodes'''
    reg = [os.path.basename(t) for t in training_tracks(False)]
    base = {n: 1.0 for n in reg}
    if args.open_train_share > 0:
        op = [os.path.basename(t) for t in open_train_tracks()]
        s = args.open_train_share
        base.update({n: s / (1.0 - s) * len(reg) / len(op) for n in op})
    return base


def track_group(name):
    '''failure-statistics group of a track file: map, direction, dead-end (open-space, seeds >= 200) or not.
    Montreal_rev.npz, Montreal_rev_w103.npz -> Montreal_rev; Montreal_rev_w201.npz -> Montreal_rev_open'''
    parts = os.path.basename(name)[:-4].split('_')
    seed = int(parts[-1][1:]) if parts[-1][:1] == 'w' and parts[-1][1:].isdigit() else None
    return parts[0] + ('_rev' if 'rev' in parts else '') + ('_open' if seed is not None and seed >= 200 else '')


def track_weights(tstat, focus, prior=2.0, base=None, by_group=False):
    '''sampling weight per training track from its (decayed) episode / failure counts, shrunk to the overall rate;
    with `base` ({track: weight}, --open_train_share) every track is listed and its weight is base x that.
    by_group: the failure rate of the track's group (track_group: all variants of a map and direction), so a
    hard corner that every variant shares (a hairpin) gets practice although each file comes up rarely'''
    n = sum(v[0] for v in tstat.values())
    g = sum(v[1] for v in tstat.values()) / n if n > 0 else 0.0
    if by_group:
        agg = {}
        for k, v in tstat.items():
            a = agg.setdefault(track_group(k), [0.0, 0.0])
            a[0], a[1] = a[0] + v[0], a[1] + v[1]
        rate = lambda k: (agg[track_group(k)][1] + prior * g) / (agg[track_group(k)][0] + prior) if track_group(k) in agg else g
    else:
        rate = lambda k: (tstat[k][1] + prior * g) / (tstat[k][0] + prior) if k in tstat else g
    if base is not None:
        return {k: b * (1.0 + focus * rate(k)) for k, b in base.items()}, 1.0 + focus * g, g
    w = {k: 1.0 + focus * rate(k) for k in tstat}
    return w, 1.0 + focus * g, g


def make_env(seed, cfg):
    args = IL.get_args([])
    args.seed = seed
    env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode='train', tracks=training_tracks(cfg.get('open_train', False)))
    for k, v in cfg.items():
        if k == 'open_train':
            continue
        if k == 'uncapped_prob':
            env.sl_cfg['uncapped_prob'] = v
        else:
            setattr(env, k, v)
    return env


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, args.workers))
    os.makedirs(f'{args.save_dir}/backup', exist_ok=True)
    os.makedirs(f'{args.save_dir}/checkpoint', exist_ok=True)
    for f, name in [(IL.task_config(), 'f1tenth.yaml'), ('configs/task/dynamic.yaml', 'dynamic.yaml')]:
        if os.path.abspath(f) != os.path.abspath(f'{args.save_dir}/backup/{name}'):    # --sim_override writes it there
            copyfile(f, f'{args.save_dir}/backup/{name}')
    with open(f'{args.save_dir}/backup/ppo_args.yaml', 'w') as f:      # what this run used
        for k, v in sorted(args.items()):
            if isinstance(v, (int, float, str, bool)) or v is None:
                f.write(f'{k}: {v}\n')
    require_fresh_variants(IL.list_variants(IL.MAPS))
    if args.open_train_share > 0:
        if not open_train_tracks():
            raise FileNotFoundError('no open-space training tracks: run `python -u eval_open_space.py --make_train` first')
        from eval_open_space import require_fresh_open_train
        require_fresh_open_train(open_train_tracks())
    base_w = base_track_weights(args) if args.open_train_share > 0 else None

    # IL policy (actor + frozen normalizer) and a frozen copy as the BC teacher
    il_dir = f'results/{args.il}'
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load(f'{il_dir}/sim2real', required=True)
    actor = IL.actor_from_state(torch.load(f'{il_dir}/checkpoint/model.pt')['actor'])
    args.hidden_dim = actor.hidden1_units          # critic gets the same width
    std0 = np.array([args.log_std_steer, args.log_std_speed], dtype=np.float32)

    def set_log_std(frac):
        with torch.no_grad():
            actor.log_std.copy_(torch.as_tensor(np.maximum(std0 - args.log_std_decay * frac, LOG_STD_MIN + 0.1)))
    set_log_std(0.0)
    teacher = IL.actor_from_state(actor.state_dict())
    teacher.eval()

    # asymmetric critic: [normalized obs, normalized privileged state]
    P = F1Wrapper.PRIV_DIM
    critic = Critic(EasyDict({**args, 'obs_dim': args.obs_dim + P})).to(args.device)
    critic.initialize()
    priv_rms = RunningMeanStd('critic_priv', P)
    ret_rms = RunningMeanStd('return', 1)
    actor_params = [p for n, p in actor.named_parameters() if 'log_std' not in n]
    opt_actor = torch.optim.Adam(actor_params, lr=args.lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    ccritic, opt_ccritic = None, None
    if args.cost_critic:                           # crash critic: discounted crash probability (0..1)
        ccritic = Critic(EasyDict({**args, 'obs_dim': args.obs_dim + P})).to(args.device)
        ccritic.initialize()
        opt_ccritic = torch.optim.Adam(ccritic.parameters(), lr=args.critic_lr)

    def value(o, pr, net=None):
        x = np.concatenate([rms.normalize(o), priv_rms.normalize(pr)], axis=-1)
        shape = x.shape[:-1]
        v = (net or critic)(torch.as_tensor(x.reshape(-1, x.shape[-1]), dtype=torch.float32)).squeeze(-1)
        return v.reshape(shape)

    eval_tracks = sorted([t for t in IL.list_variants(IL.MAPS) if variant_map(t) in IL.EVAL_MAPS
                          and (IL.HOLDOUT in t or os.path.basename(t)[:-4] in [m + s for m in IL.EVAL_MAPS for s in ('', '_rev')])])[:args.eval_tracks]
    uncapped = tuple(s for s in IL.EVAL_SETTINGS if s[0] >= args.max_speed)       # lap time is compared on these
    eval_settings = tuple(IL.EVAL_SETTINGS) + ((tuple(args.eval_low_caps), 'center'),) if args.eval_low_caps else tuple(IL.EVAL_SETTINGS)
    policy = lambda: ({k: v.clone() for k, v in actor.state_dict().items()}, rms.mean.copy(), rms.var.copy())
    eval_args = EasyDict(args)
    eval_args.workers = args.eval_workers         # evaluation pool size (the rollout workers stay alive meanwhile)
    open_tracks = sorted(glob.glob(os.path.join(OPEN_TEST_DIR, args.open_level, '*.npz')))
    if args.open_eval and not open_tracks:
        raise FileNotFoundError(f'no open-space test tracks in {OPEN_TEST_DIR}/{args.open_level}: '
                                'run `python -u eval_open_space.py --make` first (or --open_eval 0)')
    open_settings = tuple((c, 'center') for c in args.open_caps)
    open_eval = lambda: IL.evaluate(eval_args, policy(), open_tracks, 'eval', settings=open_settings)[0]

    resume_path = f'{args.save_dir}/checkpoint/resume.pt'
    state = None
    if args.resume:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f'{resume_path} missing: nothing to resume (run without --resume to start)')
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        changed = {k: (state['args'].get(k), v) for k, v in args.items()
                   if k not in ('resume', 'device') and isinstance(v, (int, float, str, bool)) and state['args'].get(k) != v}
        if changed:
            cprint(f'[resume] settings differ from the stopped run (old, now): {changed}', color='yellow', bold=True)
        actor.load_state_dict(state['actor'])
        critic.load_state_dict(state['critic'])
        # BC teacher: saved since 11/24; older resume files of an --init run get it back from the --init checkpoint
        # (until then a resumed --init run pulled towards the IL policy instead: PPO 4 after update 30, PPO 5 after 40)
        if 'teacher' in state:
            teacher.load_state_dict(state['teacher'])
        elif state['args'].get('init'):
            teacher.load_state_dict(torch.load(f"results/{state['args']['init']}", map_location='cpu', weights_only=False)['actor'])
        opt_actor.load_state_dict(state['opt_actor'])
        opt_critic.load_state_dict(state['opt_critic'])
        if ccritic is not None and 'ccritic' in state:
            ccritic.load_state_dict(state['ccritic'])
            opt_ccritic.load_state_dict(state['opt_ccritic'])
        elif ccritic is not None:
            cprint('[resume] no crash critic in the stopped run: it starts untrained', color='yellow', bold=True)
        for r, key in ((priv_rms, 'priv_rms'), (ret_rms, 'ret_rms')):
            r.mean, r.var, r.count = state[key]
        il_succ, il_lap, il_res, il_worst, il_open = (state['il'][k] for k in ('succ', 'lap', 'res', 'worst', 'open'))
        best, log = state['best'], state['log']
        np.random.set_state(state['np_rng'])
        torch.set_rng_state(state['torch_rng'])
        cprint(f'[resume] continuing after update {state["update"]} ({state["total"]} steps), best so far: {best["name"]}',
               color='green', bold=True)
    elif args.baseline_from:
        src = torch.load(f'results/{args.baseline_from}/checkpoint/resume.pt', map_location='cpu', weights_only=False)
        if src['args'].get('il') != args.il:
            raise SystemExit(f'--baseline_from {args.baseline_from} started from {src["args"].get("il")}, not {args.il}')
        src_args = dict(src['args'])
        if 'eval_low_caps' not in src_args:
            # runs before 11/25 03:50 did not save list options: read the low caps off its baseline evaluation
            src_args['eval_low_caps'] = sorted({float(r[0].split('@')[1].split('/')[0]) for r in src['il']['res']
                                                if float(r[0].split('@')[1].split('/')[0]) < 5.0}) or None
        mine = lambda k: sorted(args[k]) if k == 'eval_low_caps' and args[k] else args[k]
        theirs = lambda k: sorted(src_args[k]) if k == 'eval_low_caps' and src_args.get(k) else src_args.get(k)
        diff = {k: (theirs(k), mine(k)) for k in ('worst_eval', 'open_eval', 'open_level', 'eval_tracks', 'eval_low_caps', 'sim_override')
                if theirs(k) != mine(k)}
        def body(path):                                     # yaml without comments / blank lines
            return [l.split('#')[0].rstrip() for l in open(path) if l.split('#')[0].strip()]
        for name, now in (('dynamic.yaml', 'configs/task/dynamic.yaml'), ('f1tenth.yaml', f'{args.save_dir}/backup/f1tenth.yaml')):
            if body(f'results/{args.baseline_from}/backup/{name}') != body(now):
                diff[name] = 'differs from the source run'
        if diff:
            raise SystemExit(f'--baseline_from {args.baseline_from}: not comparable ({diff}); evaluate the IL baseline again (drop the option)')
        il_succ, il_lap, il_res, il_worst, il_open = (src['il'][k] for k in ('succ', 'lap', 'res', 'worst', 'open'))
        if (args.worst_eval and not np.isfinite(il_worst)) or (args.open_eval and not np.isfinite(il_open)):
            raise SystemExit(f'--baseline_from {args.baseline_from}: its worst-case / open-space baseline is missing')
        cprint(f'[baseline] IL evaluation reused from {args.baseline_from}', color='cyan')
    else:
        il_succ, il_lap, il_res = IL.evaluate(eval_args, policy(), eval_tracks, 'eval', settings=eval_settings)
        il_worst = IL.evaluate(eval_args, policy(), eval_tracks, 'worst')[0] if args.worst_eval else float('nan')
        il_open = open_eval() if args.open_eval else float('nan')
    if state is None and args.init:
        # after the IL baseline (evaluated with the IL actor): continue from an earlier PPO policy
        ck = torch.load(f'results/{args.init}', map_location='cpu', weights_only=False)
        actor.load_state_dict(ck['actor'])
        if 'critic' in ck:
            critic.load_state_dict(ck['critic'])
        if ccritic is not None and 'ccritic' in ck:
            ccritic.load_state_dict(ck['ccritic'])
        teacher.load_state_dict(actor.state_dict())
        set_log_std(0.0)
        cprint(f'[init] actor{" and critic" if "critic" in ck else ""} from results/{args.init} '
               f'(BC pulls towards it; the save gates still compare with IL {args.il})', color='cyan', bold=True)
    cprint(f'[IL {args.il}] eval 2-lap success {il_succ*100:.1f}% mean uncapped 2-lap time {il_lap:.1f}s'
           + (f' | worst-case success {il_worst*100:.1f}%' if args.worst_eval else '')
           + (f' | open-space ({args.open_level}) 2-lap success {il_open*100:.1f}%' if args.open_eval else ''),
           color='cyan', bold=True)
    low = [f'eval {il_succ*100:.1f}% < {args.min_il_success*100:g}%'] if il_succ < args.min_il_success - 1e-9 else []
    if args.open_eval and il_open < args.min_il_open - 1e-9:
        low.append(f'open-space {il_open*100:.1f}% < {args.min_il_open*100:g}%')
    if low and not args.force_start and state is None:
        raise SystemExit(f'[stop] IL policy {args.il} is not solid enough for PPO ({", ".join(low)}). '
                         'Train IL further, or pass --force_start.')
    if state is None:
        best = dict(succ=il_succ, res=il_res, name='IL')
        log = [{'update': 0, 'step': 0, 'success': il_succ, 'lap': il_lap, 'worst': il_worst, 'open': il_open, 'res': il_res}]

    N = args.workers
    T = args.n_steps // N
    cfg = env_config(args)
    venv = gym.vector.AsyncVectorEnv([functools.partial(make_env, args.seed * 1000 + i, cfg) for i in range(N)],
                                     context='spawn')
    obs, info = venv.reset(seed=args.seed)
    priv = np.stack(list(info['privileged'])).astype(np.float32)
    ep_ret, ep_len, ep_speed, done_stats = np.zeros(N), np.zeros(N, int), np.zeros(N), []
    disc_ret = np.zeros(N)                         # running discounted return, for the reward scale
    total, update = 0, 0
    n_updates = int(args.total_steps // args.n_steps)
    actor_start = None                             # first update that trains the actor
    bc_boost = 1.0
    crash_cost = args.crash_start or args.collision_cost   # adapted after evaluations (--crash_adapt)
    bc_dim = torch.tensor([1.0, args.bc_speed])   # BC weight per action (steering, speed)
    tstat = {}                                     # track basename -> [decayed episodes, decayed failures]
    if state is not None:
        total, update, actor_start, bc_boost = state['total'], state['update'], state['actor_start'], state['bc_boost']
        done_stats = state['done_stats']
        tstat = state.get('tstat', {})
        crash_cost = min(state.get('crash_cost', crash_cost), args.crash_max)   # a lowered --crash_max applies at once
        venv.set_attr('COLLISION_COST', crash_cost)          # the workers started with --crash_start / --collision_cost
        if actor_start is not None:
            set_log_std(min(1.0, (update - actor_start) / max(1, n_updates - actor_start)))
    if (args.track_focus > 0 and tstat) or base_w is not None:   # resumed / open-space share from the start
        venv.call('set_track_weights', *track_weights(tstat, args.track_focus, base=base_w, by_group=args.focus_by_group)[:2])

    def save_state():
        '''everything needed to continue this run (--resume); written atomically'''
        scalar = lambda v: isinstance(v, (int, float, str, bool)) or v is None
        st = dict(args={k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in args.items()
                        if scalar(v) or (isinstance(v, (list, tuple)) and all(map(scalar, v)))},
                  actor=actor.state_dict(), critic=critic.state_dict(), teacher=teacher.state_dict(),
                  opt_actor=opt_actor.state_dict(), opt_critic=opt_critic.state_dict(),
                  **(dict(ccritic=ccritic.state_dict(), opt_ccritic=opt_ccritic.state_dict()) if ccritic is not None else {}),
                  priv_rms=(priv_rms.mean, priv_rms.var, priv_rms.count), ret_rms=(ret_rms.mean, ret_rms.var, ret_rms.count),
                  il=dict(succ=il_succ, lap=il_lap, res=il_res, worst=il_worst, open=il_open),
                  best=best, log=log, update=update, total=total, actor_start=actor_start, bc_boost=bc_boost,
                  crash_cost=crash_cost,
                  done_stats=done_stats[-200:], tstat=tstat, np_rng=np.random.get_state(), torch_rng=torch.get_rng_state())
        torch.save(st, resume_path + '.tmp')
        os.replace(resume_path + '.tmp', resume_path)
    T_START = time.time()
    while update < n_updates:
        t0 = time.time()
        B = {k: [] for k in ('obs', 'priv', 'act', 'logp', 'rew', 'done', 'fail', 'nobs', 'npriv')}
        comp = {k: 0.0 for k in REWARD_KEYS}
        for _ in range(T):
            with torch.no_grad():
                dist = actor(torch.as_tensor(rms.normalize(obs), dtype=torch.float32))
                a = dist.sample()
                logp = dist.log_prob(a).sum(-1).numpy()
            a = a.numpy()
            nobs, rew, term, trunc, info = venv.step(np.clip(a, -1, 1))
            npriv = np.stack(list(info['privileged'])).astype(np.float32)
            done = term | trunc
            # only a crash cuts the value; finishing the laps (success) is bootstrapped like a time limit
            succ = np.array([bool(info['final_info'][i].get('success', False)) if d else False
                             for i, d in enumerate(done)])
            fail = term & ~trunc & ~succ
            next_o, next_p = nobs.copy(), npriv.copy()
            for i in range(N):
                src = info['final_info'][i] if done[i] else {k: info[k][i] for k in REWARD_KEYS if k in info}
                for k in REWARD_KEYS:
                    comp[k] += float(src.get(k, 0.0))
            ep_speed += priv[:, 2]
            for i in np.where(done)[0]:                       # autoreset: use the real final state
                next_o[i] = info['final_observation'][i]
                fi = info['final_info'][i]
                next_p[i] = fi['privileged']
                done_stats.append(dict(ret=ep_ret[i] + rew[i], len=ep_len[i] + 1, laps=fi.get('lap_count', 0),
                                       crash=bool(fail[i]), success=bool(succ[i]), stuck=bool(fi.get('stuck', False)),
                                       hard=bool(fi.get('hard', False)), open='maps_open_train' in str(fi.get('track_file', '')),
                                       speed=(ep_speed[i]) / (ep_len[i] + 1)))
                if fi.get('track_file'):                     # failure-weighted track sampling
                    v = tstat.setdefault(os.path.basename(fi['track_file']), [0.0, 0.0])
                    v[0], v[1] = 0.9 * v[0] + 1.0, 0.9 * v[1] + float(not succ[i])
            disc_ret = disc_ret * args.gamma + (rew + crash_cost * fail if ccritic is not None else rew)
            ret_rms.update(disc_ret.reshape(-1, 1))
            disc_ret[done] = 0.0
            for k, v in (('obs', obs), ('priv', priv), ('act', a), ('logp', logp), ('rew', rew), ('done', done),
                         ('fail', fail), ('nobs', next_o), ('npriv', next_p)):
                B[k].append(np.asarray(v))
            ep_ret = np.where(done, 0.0, ep_ret + rew)
            ep_len = np.where(done, 0, ep_len + 1)
            ep_speed = np.where(done, 0.0, ep_speed)
            obs, priv = nobs, npriv
            total += N
        B = {k: np.stack(v) for k, v in B.items()}                 # (T, N, ...)
        focus = float('nan')
        if (args.track_focus > 0 and tstat) or base_w is not None:
            w, w0, _ = track_weights(tstat, args.track_focus, base=base_w, by_group=args.focus_by_group)
            venv.call('set_track_weights', w, w0)          # used from each env's next episode
            ws = np.sort(np.array([v / (base_w or {}).get(k, 1.0) for k, v in w.items()]))[::-1]
            focus = float(ws[:max(1, len(ws) // 5)].sum() / ws.sum())   # weight share of the hardest 20 % seen
        priv_rms.update(B['priv'].reshape(-1, P))
        t_roll = time.time() - t0
        r_scale = float(np.sqrt(ret_rms.var[0] + 1e-8)) if args.ret_norm else 1.0
        # with the crash critic the reward critic sees the rewards without the collision cost (fail <=> it was paid)
        rew_n = (B['rew'] + crash_cost * B['fail'] if ccritic is not None else B['rew']) / r_scale

        # ---- GAE with the asymmetric critic (on normalized rewards)
        with torch.no_grad():
            V = value(B['obs'], B['priv']).numpy()
            V_next = value(B['nobs'], B['npriv']).numpy()
        adv = np.zeros((T, N), dtype=np.float32)
        last = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            delta = rew_n[t] + args.gamma * (1.0 - B['fail'][t]) * V_next[t] - V[t]
            last = delta + args.gamma * args.gae * (1.0 - B['done'][t]) * last
            adv[t] = last
        ret = adv + V
        ev = float(1.0 - np.var(ret - V) / (np.var(ret) + 1e-8))      # before this update: how well it predicts
        evc, RC = float('nan'), None
        if ccritic is not None:
            with torch.no_grad():
                Vc = value(B['obs'], B['priv'], ccritic).numpy()
                Vc_next = value(B['nobs'], B['npriv'], ccritic).numpy()
            advc = np.zeros((T, N), dtype=np.float32)
            last = np.zeros(N, dtype=np.float32)
            for t in reversed(range(T)):
                delta = B['fail'][t] + args.gamma * (1.0 - B['fail'][t]) * Vc_next[t] - Vc[t]
                last = delta + args.gamma * args.gae * (1.0 - B['done'][t]) * last
                advc[t] = last
            retc = advc + Vc
            evc = float(1.0 - np.var(retc - Vc) / (np.var(retc) + 1e-8))
            RC = torch.as_tensor(retc.reshape(T * N), dtype=torch.float32)
            adv = adv - (crash_cost / r_scale) * advc         # same units as a collision cost inside the reward
        flat = lambda x: torch.as_tensor(x.reshape(T * N, *x.shape[2:]), dtype=torch.float32)
        S = flat(rms.normalize(B['obs']))
        C = flat(np.concatenate([rms.normalize(B['obs']), priv_rms.normalize(B['priv'])], axis=-1))
        A, LP, R = flat(B['act']), flat(B['logp']), flat(ret)
        ADV = flat(adv)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        # ---- schedules
        update += 1
        if actor_start is None and update > args.critic_warmup and (ev >= args.warmup_ev or update > args.max_warmup):
            actor_start = update
        warm = actor_start is None
        frac = 0.0 if warm else min(1.0, (update - actor_start) / max(1, n_updates - actor_start))
        bc_w = min(args.bc_start * args.bc_max, args.bc_start * (args.bc_end / args.bc_start) ** frac * bc_boost)
        lr = args.lr + (args.lr_end - args.lr) * frac
        for g in opt_actor.param_groups:
            g['lr'] = lr

        # ---- update (the noise schedule moves after it: the ratios must use the std the rollout used)
        with torch.no_grad():
            old_dist = actor(S)
        kl, clip_frac, n_mb = 0.0, 0.0, 0
        for ep in range(args.epochs):
            perm = torch.randperm(T * N)
            for i in range(0, T * N, args.minibatch):
                idx = perm[i:i + args.minibatch]
                closs = ((critic(C[idx]).squeeze(-1) - R[idx]) ** 2).mean()
                opt_critic.zero_grad()
                closs.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_critic.step()
                if ccritic is not None:
                    ccloss = ((ccritic(C[idx]).squeeze(-1) - RC[idx]) ** 2).mean()
                    opt_ccritic.zero_grad()
                    ccloss.backward()
                    torch.nn.utils.clip_grad_norm_(ccritic.parameters(), 1.0)
                    opt_ccritic.step()
                if warm:
                    continue
                d = actor(S[idx])
                ratio = torch.exp(d.log_prob(A[idx]).sum(-1) - LP[idx])
                pg = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * ADV[idx]).mean()
                with torch.no_grad():
                    t_mean = teacher(S[idx]).loc
                    clip_frac += float(((ratio - 1).abs() > args.clip).float().mean())
                    n_mb += 1
                if args.bc_steer_weight > 0:
                    # rare large-steering states (hairpins) keep the teacher's lock: weight 1 + k x teacher steer^2
                    w = 1.0 + args.bc_steer_weight * t_mean[:, 0].clamp(-1, 1) ** 2
                    bc = (((d.loc - t_mean) ** 2) * bc_dim * torch.stack([w, torch.ones_like(w)], -1)).mean()
                else:
                    bc = (((d.loc - t_mean) ** 2) * bc_dim).mean()
                opt_actor.zero_grad()
                (pg + bc_w * bc).backward()
                torch.nn.utils.clip_grad_norm_(actor_params, 0.5)
                opt_actor.step()
            if not warm:
                with torch.no_grad():
                    kl = torch.distributions.kl_divergence(old_dist, actor(S)).sum(-1).mean().item()
                if kl > args.max_kl:
                    break
        clip_frac = clip_frac / max(1, n_mb)
        set_log_std(frac)                          # exploration noise for the next rollout

        # ---- log
        recent = done_stats[-200:]
        mean = lambda k: float(np.mean([d.get(k, 0.0) for d in recent])) if recent else float('nan')
        steps_b = T * N
        rc = {k: comp[k] / steps_b for k in REWARD_KEYS}
        std = np.exp(actor.log_std.detach().numpy())
        row = dict(update=update, step=total, warm=warm, ev=ev, ev_crash=evc, kl=kl, clip_frac=clip_frac, bc_w=bc_w, lr=lr,
                   std_steer=float(std[0]), std_speed=float(std[1]), critic_loss=float(closs.item()), ret_scale=r_scale,
                   ep_return=mean('ret'), ep_laps=mean('laps'), crash_rate=mean('crash'), success_rate=mean('success'),
                   ep_speed=mean('speed'), stuck_rate=mean('stuck'), track_focus=focus, reward_per_step=rc)
        sub_crash = lambda key: (float(np.mean([d['crash'] for d in recent if d.get(key)]))
                                 if any(d.get(key) for d in recent) else float('nan'))
        row.update(hard_share=mean('hard'), hard_crash=sub_crash('hard'), open_share=mean('open'), open_crash=sub_crash('open'))
        log.append(row)
        print(f'[update {update}/{n_updates}] steps {total} | '
              + ('critic warm-up' if warm else f'kl {kl:.4f} clip {clip_frac:.2f} bc_w {bc_w:.3f} lr {lr:.1e}')
              + f' | EV {ev:.2f}' + (f' crash EV {evc:.2f}' if ccritic is not None else '') + f' | std steer {std[0]:.3f} speed {std[1]:.3f} | ret scale {r_scale:.1f}'
              + f' | episodes: crash {row["crash_rate"]*100:.0f}% success {row["success_rate"]*100:.0f}%'
              + f' stuck {row["stuck_rate"]*100:.0f}% laps {row["ep_laps"]:.2f} speed {row["ep_speed"]:.2f} m/s'
              + (f' | hardest 20% tracks get {focus*100:.0f}% of the weight' if np.isfinite(focus) else '')
              + (f' | hard-conditions episodes {row["hard_share"]*100:.0f}% (crash {row["hard_crash"]*100:.0f}%)' if args.hard_prob > 0 else '')
              + (f' | open-space tracks {row["open_share"]*100:.0f}% (crash {row["open_crash"]*100:.0f}%)' if args.open_train_share > 0 else '')
              + f' | per step: prog {rc["forward_reward"]:.2f} time -{rc["time_cost"]:.2f} steer -{rc["steer_cost"]:.3f}'
              + (f' straight -{rc["straight_cost"]:.3f}' if args.straight_steer_cost > 0 else '')
              + f' wall -{rc["wall_cost"]:.3f} lat -{rc["lat_cost"]:.3f} crash -{rc["collision_cost"]:.2f}'
              + f' | rollout {t_roll:.0f}s update {time.time()-t0-t_roll:.0f}s (total {IL._hms(time.time()-T_START)})',
              flush=True)

        if (update % args.eval_every == 0 or update == n_updates) and not warm:
            succ, lap, res = IL.evaluate(eval_args, policy(), eval_tracks, 'eval', settings=eval_settings)
            gain, n_common = IL.paired_gain(res, best['res'], uncapped)
            cand = succ > best['succ'] + 1e-9 or (succ >= best['succ'] - 1e-9 and np.isfinite(gain) and gain > 0)
            # the worst-case and open-space checks (~half of the evaluation time) only matter for a model that
            # could be saved; --full_eval 1 runs them every time (complete curves in plot_training.py)
            gates = cand or args.full_eval
            worst = IL.evaluate(eval_args, policy(), eval_tracks, 'worst')[0] if args.worst_eval and gates else float('nan')
            worst_ok = (not args.worst_eval) or worst >= il_worst - 1e-9
            op = open_eval() if args.open_eval and gates else float('nan')
            open_ok = (not args.open_eval) or op >= il_open - 1e-9
            better = cand and worst_ok and open_ok
            cprint(f'[eval @ update {update}] 2-lap success {succ*100:.1f}% (best {best["name"]} {best["succ"]*100:.1f}%) '
                   f'| vs best on {n_common} runs both finished: {gain:+.2f} s faster'
                   + (f' | worst-case success {worst*100:.1f}% (IL {il_worst*100:.1f}%)' if args.worst_eval and gates else '')
                   + (f' | open-space {op*100:.1f}% (IL {il_open*100:.1f}%)' if args.open_eval and gates else '')
                   + ('' if gates else ' | not better than the best: worst-case / open-space checks skipped')
                   + ('  -> saved' if better else ''), color='green' if better else 'cyan', bold=True)
            row.update(eval_success=succ, eval_lap=lap, paired_gain=gain, worst=worst, open=op, res=res, saved=better)
            if args.bc_adapt:
                # keep the policy near IL while it is worse than IL; relax again once it recovers
                bc_boost = min(bc_boost * 2.0, 64.0) if succ < il_succ - 0.02 else max(1.0, bc_boost / 2.0)
            if args.crash_adapt:
                # completion as a constraint: the policy itself must find it worth staying safe
                old = crash_cost
                target = max(il_succ, best['succ']) if args.crash_target == 'best' else il_succ
                # the worst-case / open-space checks count too when they ran (a model that reached the target on
                # the eval tracks but is below IL there is not safe enough yet: 11/24 PPO 4 update 30)
                gates_bad = gates and not (worst_ok and open_ok)
                if succ < target - 0.02 or gates_bad:
                    crash_cost = min(crash_cost * args.crash_up, args.crash_max)
                elif succ >= target - 1e-9 and gates:
                    crash_cost = max(crash_cost * args.crash_down, args.collision_cost)
                if crash_cost != old:
                    venv.set_attr('COLLISION_COST', crash_cost)
                    cprint(f'[crash cost] {old:.0f} -> {crash_cost:.0f} (eval {succ*100:.1f}% vs target {target*100:.1f}%'
                           f' = {"best saved" if args.crash_target == "best" else "IL"}'
                           + (' | worst-case / open-space below IL' if gates_bad else '') + ')', color='yellow', bold=True)
            row['collision_cost'] = crash_cost
            torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(),
                        **({'ccritic': ccritic.state_dict()} if ccritic is not None else {})},
                       f'{args.save_dir}/checkpoint/model_{total}.pt')
            if better:
                best = dict(succ=succ, res=res, name=f'update {update}')
                os.makedirs(f'{args.save_dir}/sim2real', exist_ok=True)
                torch.save(actor, f'{args.save_dir}/sim2real/actor.pt')
                rms.save(f'{args.save_dir}/sim2real')
                rms.save(args.save_dir)
                RunningMeanStd('agent_reward', 1).save(f'{args.save_dir}/sim2real')
                torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(),
                        **({'ccritic': ccritic.state_dict()} if ccritic is not None else {})}, f'{args.save_dir}/checkpoint/model.pt')
        with open(f'{args.save_dir}/train_log.pkl', 'wb') as f:
            pickle.dump(log, f)
        if update % args.ckpt_every == 0 or update == n_updates or 'eval_success' in row:
            save_state()
    venv.close()
    cprint(f'[done] IL {il_succ*100:.1f}% / {il_lap:.1f}s -> best: {best["name"]} {best["succ"]*100:.1f}% '
           f'({"PPO improved on IL" if os.path.isfile(f"{args.save_dir}/sim2real/actor.pt") else "no improvement: keep deploying the IL policy"})', bold=True)


if __name__ == '__main__':
    main()
