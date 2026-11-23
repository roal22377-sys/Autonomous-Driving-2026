'''
    Training curves of IL (train_il_s2r.py) and PPO (train_ppo_s2r.py) runs, from their train_log.pkl.
    Run from src/f1tenth:

        python plot_training.py imitation/9                 # -> results/imitation/9/training.png
        python plot_training.py ppo_s2r/1 ppo_s2r/2          # one figure per run
        python plot_training.py imitation/9 --show           # also open a window

    IL: per round - eval / open-space 2-lap success and the round score, mean uncapped 2-lap time,
        BC validation loss, data size; the saved best round and early-stop rounds are marked.
    PPO: per update - explained variance, KL and clip fraction, exploration std, BC weight and lr,
         training-episode crash / success rate and mean speed, reward terms per step; at evaluations
         the eval / worst-case / open-space success and the paired time gain vs the best model.
    A short text summary is printed as well.
'''
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REWARD_KEYS = ('forward_reward', 'time_cost', 'steer_cost', 'wall_cost', 'lat_cost', 'collision_cost', 'lap_bonus')


def load(model):
    path = f'results/{model}/train_log.pkl'
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found (run from src/f1tenth; the run writes it after each round/update)')
    with open(path, 'rb') as f:
        return pickle.load(f)


def best_round(log):
    '''the round train_il_s2r saved: highest score, ties by the first one reaching it'''
    sc = [l.get('score', l.get('success', 0.0)) for l in log]
    return int(np.argmax(sc)) if sc else None


def plot_il(model, log, out):
    r = np.array([l['round'] for l in log])
    g = lambda k: np.array([l.get(k, np.nan) if l.get(k) is not None else np.nan for l in log], dtype=float)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    a = ax[0, 0]
    a.plot(r, g('success') * 100, 'o-', label='eval 2-lap success')
    if np.isfinite(g('open')).any():
        a.plot(r, g('open') * 100, 's-', label='open-space 2-lap success')
    a.plot(r, g('score') * 100, 'k--', label='round score')
    a.axhline(98, color='g', lw=0.8, ls=':', label='PPO start: eval 98 %')
    a.axhline(95, color='orange', lw=0.8, ls=':', label='PPO start: open-space 95 %')
    b = best_round(log)
    if b is not None:
        a.axvline(r[b], color='g', alpha=0.3, lw=6, label=f'saved best (round {r[b]})')
    a.set(xlabel='round', ylabel='%', title='success (2 laps)', ylim=(-2, 102))
    a.legend(fontsize=8)
    a = ax[0, 1]
    a.plot(r, g('lap'), 'o-')
    a.set(xlabel='round', ylabel='s', title='mean uncapped 2-lap time (finished runs; not paired)')
    a = ax[1, 0]
    a.plot(r, g('val'), 'o-')
    a.set(xlabel='round', ylabel='weighted MSE', title='BC validation loss', yscale='log')
    a = ax[1, 1]
    a.plot(r, g('data') / 1e6, 'o-')
    a.set(xlabel='round', ylabel='million samples', title='training data (after the memory cap)')
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle(f'IL {model}')
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f'IL {model}: {len(log)} rounds')
    for l in log:
        mark = '  <- best' if b is not None and l['round'] == r[b] else ''
        op = f" open {l['open']*100:5.1f}%" if l.get('open') is not None and np.isfinite(l.get('open', np.nan)) else ''
        print(f"  round {l['round']:2d}: eval {l['success']*100:5.1f}%{op} score {l.get('score', l['success'])*100:5.1f} "
              f"lap {l['lap']:6.1f}s data {l['data']/1e6:4.2f}M val {l['val']:.5f}{mark}")
    return out


def plot_ppo(model, log, out):
    il = log[0]
    rows = [l for l in log[1:] if 'update' in l and 'ev' in l]
    if not rows:
        raise ValueError(f'{model}: no PPO updates in the log yet')
    u = np.array([l['update'] for l in rows])
    g = lambda k: np.array([l.get(k, np.nan) for l in rows], dtype=float)
    ev_rows = [l for l in rows if 'eval_success' in l]
    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    a = ax[0, 0]
    a.plot(u, g('ev'), label='explained variance')
    a.axhline(0.6, color='gray', ls=':', lw=0.8)
    warm = [l['update'] for l in rows if l.get('warm')]
    if warm:
        a.axvspan(min(warm) - 0.5, max(warm) + 0.5, color='orange', alpha=0.15, label='critic warm-up')
    a.set(title='critic', xlabel='update', ylim=(-0.5, 1.05))
    a.legend(fontsize=8)
    a = ax[0, 1]
    a.plot(u, g('kl'), label='KL')
    a.plot(u, g('clip_frac'), label='clip fraction')
    a.set(title='policy change per update', xlabel='update')
    a.legend(fontsize=8)
    a = ax[1, 0]
    a.plot(u, g('std_steer'), label='steer std')
    a.plot(u, g('std_speed'), label='speed std')
    a2 = a.twinx()
    a2.plot(u, g('bc_w'), 'k--', lw=0.8, label='BC weight')
    a2.set_yscale('log')
    a.set(title='exploration noise / BC weight', xlabel='update')
    a.legend(fontsize=8, loc='upper left')
    a2.legend(fontsize=8, loc='upper right')
    a = ax[1, 1]
    a.plot(u, g('crash_rate') * 100, label='crash %')
    a.plot(u, g('success_rate') * 100, label='2-lap success %')
    a2 = a.twinx()
    a2.plot(u, g('ep_speed'), 'k--', lw=0.8, label='mean speed [m/s]')
    a.set(title='training episodes (last 200)', xlabel='update')
    a.legend(fontsize=8, loc='upper left')
    a2.legend(fontsize=8, loc='upper right')
    a = ax[2, 0]
    for k in REWARD_KEYS:
        v = np.array([l['reward_per_step'].get(k, 0.0) for l in rows])
        if np.any(np.abs(v) > 0):
            a.plot(u, v if k in ('forward_reward', 'lap_bonus') else -v, label=k.replace('_reward', '').replace('_cost', ''))
    a.set(title='reward terms per step (costs negative)', xlabel='update')
    a.legend(fontsize=8)
    a = ax[2, 1]
    if ev_rows:
        ue = [l['update'] for l in ev_rows]
        a.plot(ue, [l['eval_success'] * 100 for l in ev_rows], 'o-', label='eval success %')
        if any(np.isfinite(l.get('worst', np.nan)) for l in ev_rows):
            a.plot(ue, [l['worst'] * 100 for l in ev_rows], 's-', label='worst-case success %')
        if any(np.isfinite(l.get('open', np.nan)) for l in ev_rows):
            a.plot(ue, [l['open'] * 100 for l in ev_rows], '^-', label='open-space success %')
        a.axhline(il['success'] * 100, color='C0', ls=':', lw=0.8, label='IL eval')
        for l in ev_rows:
            if l.get('saved'):
                a.axvline(l['update'], color='g', alpha=0.25, lw=5)
        a2 = a.twinx()
        a2.plot(ue, [l.get('paired_gain', np.nan) for l in ev_rows], 'k--', marker='x', lw=0.8, label='s faster than best (paired)')
        a2.axhline(0, color='k', lw=0.5)
        a2.legend(fontsize=8, loc='lower right')
        a.legend(fontsize=8, loc='lower left')
    a.set(title='evaluations (green = saved)', xlabel='update')
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle(f'PPO {model} (from IL: eval {il["success"]*100:.1f} %, open {il.get("open", np.nan)*100:.1f} %)')
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    last = rows[-1]
    print(f'PPO {model}: {len(rows)} updates, {last["step"]} steps | last: EV {last["ev"]:.2f} KL {last["kl"]:.4f} '
          f'crash {last["crash_rate"]*100:.0f}% speed {last["ep_speed"]:.2f} m/s')
    for l in ev_rows:
        print(f"  eval @ update {l['update']:3d}: success {l['eval_success']*100:5.1f}% worst {l.get('worst', np.nan)*100:5.1f}% "
              f"open {l.get('open', np.nan)*100:5.1f}% | {l.get('paired_gain', np.nan):+.2f} s vs best{'  saved' if l.get('saved') else ''}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('models', nargs='+', help='e.g. imitation/9 ppo_s2r/1')
    p.add_argument('--out', default=None, help='png path (one model only); default results/<model>/training.png')
    p.add_argument('--show', action='store_true')
    a = p.parse_args()
    for m in a.models:
        log = load(m)
        out = a.out if (a.out and len(a.models) == 1) else f'results/{m}/training.png'
        is_ppo = any('update' in l for l in log)
        print(f'[saved] {(plot_ppo if is_ppo else plot_il)(m, log, out)}')
    if a.show:
        matplotlib.use('TkAgg', force=True)
        plt.show()


if __name__ == '__main__':
    main()
