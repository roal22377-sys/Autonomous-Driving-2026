#!/usr/bin/env python3
"""Expert feasibility and new reward calibration before IL collection."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
SRC=REPO/'src/f1tenth'
sys.path[:0]=[str(REPO/'tools'),str(SRC),str(SRC/'installation')]


def job(spec):
    from check_track import drive
    path,cap=spec
    result=drive('expert',path,cap,seed=0)
    result.pop('xy',None)
    return result


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--workers',type=int,default=10)
    a.add_argument('--practice_only',action='store_true')
    args=a.parse_args()
    os.chdir(SRC)
    os.environ['F1_TASK_CONFIG']='configs/task/f1tenth_preview_1126.yaml'
    from utils.training_mix import hall_split
    train,valid=hall_split()
    real=[str(p) for p in sorted((SRC/'installation/maps_real').glob('Practice1126b*.npz'))]
    specs=[(os.path.abspath(p),8.) for p in train+valid]+[(p,8.) for p in real]
    specs += [(os.path.abspath(p),2.5) for p in train if Path(p).name.startswith('Course2025')]
    specs += [(p,2.5) for p in real]
    if args.practice_only:
        specs=[(os.path.abspath(p),cap) for p in train+real if Path(p).name.startswith(('Course2025','Practice1126b'))
               for cap in (8.,2.5)]
    start=time.time();results=[]
    print(f'Checking {len(specs)} 2-lap expert drives; {len(train)} hall training / {len(valid)} hall validation tracks',flush=True)
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for result in pool.imap_unordered(job,specs):
            results.append(result)
            if not result['ok']:print('EXPERT FAILURE',result,flush=True)
            if len(results)%20==0:print(f'{len(results)}/{len(specs)} checked, {sum(r["ok"] for r in results)} complete, {time.time()-start:.0f}s',flush=True)
    output=SRC/'results/preflight_1126'
    output.mkdir(parents=True,exist_ok=True)
    filename='expert_feasibility_practice_final.json' if args.practice_only else 'expert_feasibility.json'
    (output/filename).write_text(json.dumps(dict(config=os.environ['F1_TASK_CONFIG'],
        elapsed_s=time.time()-start,results=results),indent=2)+'\n')
    print(f'DONE {sum(r["ok"] for r in results)}/{len(results)} complete, {time.time()-start:.0f}s',flush=True)


if __name__=='__main__':main()
