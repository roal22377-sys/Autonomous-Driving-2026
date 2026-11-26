#!/usr/bin/env python3
'''
Models on the narrow hall courses (utils/track_hall.py, 2025-competition-like: 1.0-1.9 m corridors, one tube wall
between neighbouring corridors, hairpins; evaluation only): the held-out courses (block 3) and the 2025 course copy
(Course2025*), caps 8 and 4, 2 laps, centerline start. Same runner as compare_models.py.

    python3 tools/eval_hall.py expert imitation/13 ppo_s2r/2130 ppo_s2r/2150 ppo_s2r/12190 [--workers 8] [--train]

Writes src/f1tenth/results/hall_eval_<time>.md.
'''
import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'src/f1tenth')
sys.path.insert(0, SRC)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models', nargs='+')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--caps', default='8,4')
    ap.add_argument('--train', action='store_true', help='the training courses instead of the held-out ones')
    ap.add_argument('--max_tracks', type=int, default=0)
    a = ap.parse_args()
    os.chdir(SRC)
    import compare_models as CM
    from utils.track_hall import hall_tracks
    tracks = [os.path.abspath(t) for t in hall_tracks(holdout=not a.train)]
    if a.max_tracks:
        tracks = tracks[:a.max_tracks]
    caps = [float(c) for c in a.caps.split(',')]
    jobs = [(m, t, c, 'center', 'eval', 2, 'hall', 0) for m in a.models for c in caps for t in tracks]
    print(f'{len(jobs)} runs: {len(a.models)} models x {len(tracks)} courses x caps {caps}, {a.workers} workers', flush=True)
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.workers) as pool:
        res = pool.map(CM.run, jobs, chunksize=1)
    print(f'done in {time.time() - t0:.0f} s')
    ref = a.models[0]
    lines = [f'# narrow hall courses ({time.strftime("%Y-%m-%d %H:%M")}, {"training" if a.train else "held-out"} '
             f'{len(tracks)} courses incl. the 2025 copy, 2 laps, reference {ref})', '',
             '| model | ' + ' | '.join(f'success cap {c:g}' for c in caps) + ' | 2025 copy (fwd/rev, all caps) | '
             '2-lap time cap 8 vs ref (both finished) |', '|---|' + '---|' * (len(caps) + 2)]
    by = {}
    for r, j in zip(res, jobs):
        by.setdefault(j[0], []).append((r, j))

    def rate(rr):
        return f'{np.mean([r["ok"] for r in rr]) * 100:.1f}% ({sum(r["ok"] for r in rr)}/{len(rr)})' if rr else '-'
    ref8 = {r['key']: r['t'] for r, j in by[ref] if r['ok'] and j[2] == 8.0}
    for m in a.models:
        rows = by[m]
        cells = [rate([r for r, j in rows if j[2] == c]) for c in caps]
        c25 = [r for r, j in rows if 'Course2025' in j[1]]
        mine = {r['key']: r['t'] for r, j in rows if r['ok'] and j[2] == 8.0}
        com = sorted(set(ref8) & set(mine))
        dt = (f'{np.mean([mine[k] - ref8[k] for k in com]):+.2f} s on {len(com)} ({np.mean([mine[k] for k in com]):.1f} s)'
              if com and m != ref else ('reference' if m == ref else '-'))
        lines.append(f'| {m} | ' + ' | '.join(cells) + f' | {sum(r["ok"] for r in c25)}/{len(c25)} | {dt} |')
    lines += ['', '## failed runs', '']
    for m in a.models:
        bad = [f'{r["key"]} ({r["t"]:.1f} s)' for r, j in by[m] if not r['ok']]
        lines.append(f'- {m}: ' + (', '.join(bad) if bad else 'none'))
    out = os.path.join(SRC, 'results', f'hall_eval_{time.strftime("%m%d_%H%M")}.md')
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:len(a.models) + 4]))
    print(out)


if __name__ == '__main__':
    main()
