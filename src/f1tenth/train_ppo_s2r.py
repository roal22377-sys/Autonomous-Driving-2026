'''
    PPO fine-tuning of the sim2real IL policy for lap time, run from src/f1tenth:

        python -u train_ppo_s2r.py --il imitation/7 --algo_idx 1 --workers 22

    - starts from the IL actor (results/<il>/checkpoint/model.pt) and its obs normalizer (frozen)
    - same simulator as IL training: sim_mode='train' (delay, accel limit, friction, noise,
      generated tracks both directions, 2-lap episodes)
    - reward (utils/env.py): progress along the track (-> lap time), collision penalty,
      small steering-rate penalty (weaving is slow and fragile on the car)
    - asymmetric actor-critic: the critic also sees simulator-only state (utils/env.py
      F1Wrapper.privileged: lateral error, heading error, true speed/steer/yaw rate/slip, this
      episode's delay / a_max / friction / speed cap, upcoming curvature). The value estimate gets
      much better, the actor (what runs on the car) still sees only lidar + speed + history + cap
    - critic warm-up with the actor frozen, then PPO (clip + KL early stop) with a
      behavior-cloning term towards the IL policy whose weight decays (keeps it from collapsing)
    - GAE that bootstraps from the true final observation on time-limit truncation
    - the policy is saved only if it finishes at least as many evaluation laps as the IL policy
      AND is faster; otherwise the IL policy stays the deployed one
    Output: results/ppo_s2r/<algo_idx>/ (same layout as the other algorithms)
'''
import argparse
import os
import sys
import time
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
from utils.track_gen import variant_map
from utils.env import F1Wrapper
from utils.vectorize import RunningMeanStd
from utils.color import cprint
from algorithm.common.models import Critic

PROGRESS_COEFF = 10.0     # per meter of progress
COLLISION_COST = 200.0    # > value of the rest of the episode, so crashing never pays
STEER_RATE_COST = 5.0     # per (change of normalized steering)^2


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--il', default='imitation/7')
    p.add_argument('--name', default='ppo_s2r')
    p.add_argument('--algo_idx', type=int, default=1)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=22)
    p.add_argument('--total_steps', type=float, default=6e6)
    p.add_argument('--n_steps', type=int, default=44000, help='rollout steps per update (all envs)')
    p.add_argument('--minibatch', type=int, default=2048)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--gae', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.15)
    p.add_argument('--max_kl', type=float, default=0.02)
    p.add_argument('--log_std', type=float, default=-2.0)
    p.add_argument('--critic_warmup', type=int, default=3, help='updates with the actor frozen')
    p.add_argument('--bc_start', type=float, default=1.0)
    p.add_argument('--bc_end', type=float, default=0.05)
    p.add_argument('--eval_every', type=int, default=10, help='updates between evaluations')
    p.add_argument('--eval_tracks', type=int, default=44)
    a = p.parse_args()
    args = IL.get_args([])
    args.update(vars(a))
    args.save_dir = f'results/{args.name}/{args.algo_idx}'
    return args


def make_env(seed):
    args = IL.get_args([])
    args.seed = seed
    env = F1Wrapper(args, IL.MAPS, obs_noise=True, sim_mode='train',
                    tracks=[t for t in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in t])
    env.PROGRESS_COEFF, env.COLLISION_COST, env.STEER_RATE_COST = PROGRESS_COEFF, COLLISION_COST, STEER_RATE_COST
    return env


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, args.workers))
    os.makedirs(f'{args.save_dir}/backup', exist_ok=True)
    os.makedirs(f'{args.save_dir}/checkpoint', exist_ok=True)
    for f in ['configs/task/f1tenth.yaml', 'configs/task/dynamic.yaml']:
        copyfile(f, f'{args.save_dir}/backup/{os.path.basename(f)}')

    # IL policy (actor + frozen normalizer) and a frozen copy as the BC teacher
    il_dir = f'results/{args.il}'
    rms = RunningMeanStd('agent_obs', args.obs_dim)
    rms.load(f'{il_dir}/sim2real')
    actor = IL.make_actor(args)
    actor.load_state_dict(torch.load(f'{il_dir}/checkpoint/model.pt')['actor'])
    with torch.no_grad():
        actor.log_std.fill_(args.log_std)
    teacher = IL.make_actor(args)
    teacher.load_state_dict(actor.state_dict())
    teacher.eval()

    # asymmetric critic: [normalized obs, normalized privileged state]
    P = F1Wrapper.PRIV_DIM
    critic = Critic(EasyDict({**args, 'obs_dim': args.obs_dim + P})).to(args.device)
    critic.initialize()
    priv_rms = RunningMeanStd('critic_priv', P)
    actor_params = [p for n, p in actor.named_parameters() if 'log_std' not in n]
    opt_actor = torch.optim.Adam(actor_params, lr=args.lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)

    def value(o, pr):
        x = np.concatenate([rms.normalize(o), priv_rms.normalize(pr)], axis=-1)
        shape = x.shape[:-1]
        v = critic(torch.as_tensor(x.reshape(-1, x.shape[-1]), dtype=torch.float32)).squeeze(-1)
        return v.reshape(shape)

    eval_tracks = sorted([t for t in IL.list_variants(IL.MAPS) if variant_map(t) in IL.EVAL_MAPS
                          and (IL.HOLDOUT in t or os.path.basename(t)[:-4] in [m + s for m in IL.EVAL_MAPS for s in ('', '_rev')])])[:args.eval_tracks]
    policy = lambda: ({k: v.clone() for k, v in actor.state_dict().items()}, rms.mean.copy(), rms.var.copy())
    il_succ, il_lap, _ = IL.evaluate(args, policy(), eval_tracks, 'eval')
    cprint(f'[IL {args.il}] eval 2-lap success {il_succ*100:.1f}% mean uncapped 2-lap time {il_lap:.1f}s', color='cyan', bold=True)
    best = (il_succ, il_lap)
    log = [{'update': 0, 'step': 0, 'success': il_succ, 'lap': il_lap}]

    N = args.workers
    T = args.n_steps // N
    venv = gym.vector.AsyncVectorEnv([functools.partial(make_env, args.seed * 1000 + i) for i in range(N)],
                                     context='spawn')
    obs, info = venv.reset(seed=args.seed)
    priv = np.stack(list(info['privileged'])).astype(np.float32)
    ep_ret, ep_len, done_stats = np.zeros(N), np.zeros(N, int), []
    total, update = 0, 0
    n_updates = int(args.total_steps // args.n_steps)
    while update < n_updates:
        t0 = time.time()
        B = {k: [] for k in ('obs', 'priv', 'act', 'logp', 'rew', 'done', 'fail', 'nobs', 'npriv')}
        for _ in range(T):
            with torch.no_grad():
                dist = actor(torch.as_tensor(rms.normalize(obs), dtype=torch.float32))
                a = dist.sample()
                logp = dist.log_prob(a).sum(-1).numpy()
            a = a.numpy()
            nobs, rew, term, trunc, info = venv.step(np.clip(a, -1, 1))
            npriv = np.stack(list(info['privileged'])).astype(np.float32)
            done, fail = term | trunc, term & ~trunc
            next_o, next_p = nobs.copy(), npriv.copy()
            for i in np.where(done)[0]:                       # autoreset: use the real final state
                next_o[i] = info['final_observation'][i]
                fi = info['final_info'][i]
                next_p[i] = fi['privileged']
                done_stats.append((ep_ret[i] + rew[i], ep_len[i] + 1, fi.get('lap_count', 0)))
            for k, v in (('obs', obs), ('priv', priv), ('act', a), ('logp', logp), ('rew', rew), ('done', done),
                         ('fail', fail), ('nobs', next_o), ('npriv', next_p)):
                B[k].append(np.asarray(v))
            ep_ret = np.where(done, 0.0, ep_ret + rew)
            ep_len = np.where(done, 0, ep_len + 1)
            obs, priv = nobs, npriv
            total += N
        B = {k: np.stack(v) for k, v in B.items()}                 # (T, N, ...)
        priv_rms.update(B['priv'].reshape(-1, P))
        t_roll = time.time() - t0

        # ---- GAE with the asymmetric critic
        with torch.no_grad():
            V = value(B['obs'], B['priv']).numpy()
            V_next = value(B['nobs'], B['npriv']).numpy()
        adv = np.zeros((T, N), dtype=np.float32)
        last = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            delta = B['rew'][t] + args.gamma * (1.0 - B['fail'][t]) * V_next[t] - V[t]
            last = delta + args.gamma * args.gae * (1.0 - B['done'][t]) * last
            adv[t] = last
        ret = adv + V
        flat = lambda x: torch.as_tensor(x.reshape(T * N, *x.shape[2:]), dtype=torch.float32)
        S = flat(rms.normalize(B['obs']))
        C = flat(np.concatenate([rms.normalize(B['obs']), priv_rms.normalize(B['priv'])], axis=-1))
        A, LP, R = flat(B['act']), flat(B['logp']), flat(ret)
        ADV = flat(adv)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        # ---- update
        update += 1
        warm = update <= args.critic_warmup
        frac = min(1.0, max(0.0, (update - args.critic_warmup) / max(1, n_updates - args.critic_warmup)))
        bc_w = args.bc_start * (args.bc_end / args.bc_start) ** frac
        with torch.no_grad():
            old_dist = actor(S)
        kl = 0.0
        for ep in range(args.epochs):
            perm = torch.randperm(T * N)
            for i in range(0, T * N, args.minibatch):
                idx = perm[i:i + args.minibatch]
                closs = ((critic(C[idx]).squeeze(-1) - R[idx]) ** 2).mean()
                opt_critic.zero_grad()
                closs.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_critic.step()
                if warm:
                    continue
                d = actor(S[idx])
                ratio = torch.exp(d.log_prob(A[idx]).sum(-1) - LP[idx])
                pg = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * ADV[idx]).mean()
                with torch.no_grad():
                    t_mean = teacher(S[idx]).loc
                bc = ((d.loc - t_mean) ** 2).mean()
                opt_actor.zero_grad()
                (pg + bc_w * bc).backward()
                torch.nn.utils.clip_grad_norm_(actor_params, 0.5)
                opt_actor.step()
            if not warm:
                with torch.no_grad():
                    kl = torch.distributions.kl_divergence(old_dist, actor(S)).sum(-1).mean().item()
                if kl > args.max_kl:
                    break
        recent = done_stats[-200:]
        mr = np.mean([d[0] for d in recent]) if recent else float('nan')
        ml = np.mean([d[2] for d in recent]) if recent else float('nan')
        print(f'[update {update}/{n_updates}] steps {total} | {"critic warm-up" if warm else f"kl {kl:.4f} bc_w {bc_w:.3f}"} '
              f'| critic {closs.item():.3f} | recent episodes: return {mr:.0f}, laps {ml:.2f} '
              f'| rollout {t_roll:.0f}s update {time.time()-t0-t_roll:.0f}s', flush=True)

        if (update % args.eval_every == 0 or update == n_updates) and not warm:
            succ, lap, res = IL.evaluate(args, policy(), eval_tracks, 'eval')
            # strictly better: more evaluation runs finished, or as many and faster
            better = succ > best[0] + 1e-9 or (abs(succ - best[0]) < 1e-9 and np.isfinite(lap)
                                               and (not np.isfinite(best[1]) or lap < best[1]))
            cprint(f'[eval @ update {update}] 2-lap success {succ*100:.1f}% mean uncapped 2-lap time {lap:.1f}s '
                   f'(best so far {best[0]*100:.1f}% / {best[1]:.1f}s){"  -> saved" if better else ""}',
                   color='green' if better else 'cyan', bold=True)
            log.append({'update': update, 'step': total, 'success': succ, 'lap': lap, 'res': res})
            torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict()},
                       f'{args.save_dir}/checkpoint/model_{total}.pt')
            if better:
                best = (succ, lap)
                os.makedirs(f'{args.save_dir}/sim2real', exist_ok=True)
                torch.save(actor, f'{args.save_dir}/sim2real/actor.pt')
                rms.save(f'{args.save_dir}/sim2real')
                rms.save(args.save_dir)
                RunningMeanStd('agent_reward', 1).save(f'{args.save_dir}/sim2real')
                torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict()}, f'{args.save_dir}/checkpoint/model.pt')
            with open(f'{args.save_dir}/train_log.pkl', 'wb') as f:
                pickle.dump(log, f)
    venv.close()
    cprint(f'[done] IL {il_succ*100:.1f}% / {il_lap:.1f}s -> best PPO {best[0]*100:.1f}% / {best[1]:.1f}s '
           f'({"PPO improved on IL" if os.path.isfile(f"{args.save_dir}/sim2real/actor.pt") else "no improvement: keep deploying the IL policy"})', bold=True)


if __name__ == '__main__':
    main()
