#!/usr/bin/env python3
'''expert on every short course of installation/maps_short (utils/track_short.py): caps 8 and 4, 2 seeds, 2 laps.
   Tracks it does not finish every time are no training/eval tracks: --remove_failed deletes them.
       python3 tools/check_short_tracks.py [WORKERS] [--remove_failed] [--dir=maps_hall]
   11/25: 8 of 114 failed (narrow tight corners), removed.'''
import glob, os, sys, multiprocessing as mp
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(REPO, 'src/f1tenth/installation', next((x.split('=', 1)[1] for x in sys.argv if x.startswith('--dir=')), 'maps_short'))
sys.path[:0] = [os.path.join(REPO, 'tools'), os.path.join(REPO, 'src/f1tenth')]

def run(job):
    import check_track as CT
    path, cap, seed = job
    d = CT.drive('expert', path, cap, seed)
    return os.path.basename(path), cap, seed, d['ok'], d['t']

if __name__ == '__main__':
    tracks = sorted(glob.glob(os.path.join(DIR, '*.npz')))
    jobs = [(t, c, s) for t in tracks for c in (8.0, 4.0) for s in (0, 1)]
    with mp.get_context('spawn').Pool(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8) as p:
        res = p.map(run, jobs)
    bad = [r for r in res if not r[3]]
    if "--remove_failed" in sys.argv:
        for n in sorted({r[0] for r in bad}):
            os.remove(os.path.join(DIR, n)); print("removed", n)
    print(f'expert: {len(res) - len(bad)}/{len(res)} finished')
    for r in bad:
        print('FAIL', r)
    import numpy as np
    t8 = [r[4] for r in res if r[1] == 8.0 and r[3]]
    print('2-lap time at cap 8: median %.1f s' % np.median(t8))
