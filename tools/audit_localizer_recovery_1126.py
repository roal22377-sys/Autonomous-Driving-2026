#!/usr/bin/env python3
"""Matched offline ablations of existing localization settings on a recorded log."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/f1tenth'))
from loc_test import load_log
from utils.localizer import ScanLocalizer,course_candidates


def invalid_intervals(ts,valid):
    changes=np.diff(np.r_[False,~np.asarray(valid,bool),False].astype(int))
    rows=[]
    for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
        recovered=bool(b<len(ts))
        end=float(ts[b] if recovered else ts[-1])
        rows.append(dict(start=float(ts[a]),end=end,seconds=end-float(ts[a]),
                         recovered=recovered,right_censored=not recovered,scans=int(b-a)))
    return rows


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log',required=True);p.add_argument('--map',required=True)
    p.add_argument('--track',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--t0',type=float,default=38.);p.add_argument('--t1',type=float,default=84.)
    p.add_argument('--confirm-local',type=int,default=8,
                   help='Offline candidate confirmation length; production defaults are untouched')
    p.add_argument('--cases',nargs='+',choices=['current','no_skid','strict_refind','strict_refind_no_skid'],
                   default=['current','no_skid','strict_refind','strict_refind_no_skid'])
    cli=p.parse_args();dest=Path(cli.output_dir)
    if dest.exists():raise FileExistsError('Use a new audit directory')
    dest.mkdir(parents=True)
    R,ts,ang,od=load_log(cli.log,cli.t0,cli.t1)
    candidates=course_candidates(cli.track)
    inputs=[Path(cli.log)/'scan.npz',Path(cli.log)/'driving.csv',Path(cli.map)/'map.yaml',Path(cli.map)/'map.pgm',Path(cli.track)]
    report=dict(status='running',source_sha256={str(p.resolve()):sha(p) for p in inputs},
                localizer_source_sha256=sha(ROOT/'src/f1tenth/utils/localizer.py'),
                interval=[float(ts[0]),float(ts[-1])],scans=len(ts),cases=[],
                qualification='Offline replay with fixed recorded motion. Valid is estimator confidence, not position truth. No absolute position ground truth; no proof of false-positive rate, closed-loop collision safety or real-car recovery improvement.')
    for threshold,skid,label in [( .7,.6,'current'),(.7,0.,'no_skid'),(.8,.6,'strict_refind'),(.8,0.,'strict_refind_no_skid')]:
        if label not in cli.cases:continue
        loc=ScanLocalizer(cli.map,laser_x=.13,explained_refind=threshold,skid_ratio=skid,
                          confirm_local=cli.confirm_local)
        initial,res,explained,alternative=loc.global_init(R[0],ang,candidates,od[0])
        if initial is None or not loc.valid or alternative is not None:
            raise ValueError('Ambiguous/invalid global initialization: do not simulate accepted car startup')
        poses=[];valid=[];accepted=[];elapsed=[];skids=0
        for ranges,odom in zip(R,od):
            start=time.perf_counter();pose,ok,info=loc.step(ranges,ang,odom);elapsed.append(time.perf_counter()-start)
            poses.append(pose);valid.append(bool(ok));accepted.append(bool(info.get('accepted')))
            skids+=int(bool(info.get('skid')))
        intervals=invalid_intervals(ts,valid)
        np.savez_compressed(dest/(label+'.npz'),t=ts,pose=np.asarray(poses),valid=valid,accepted=accepted,
                            runtime_seconds=elapsed)
        case=dict(label=label,explained_refind=threshold,skid_ratio=skid,confirm_local=cli.confirm_local,
                  initial_pose=initial.tolist(),
                  initial_explained=explained,valid_fraction=float(np.mean(valid)),accepted_fraction=float(np.mean(accepted)),
                  invalid_intervals=intervals,longest_invalid_observed_seconds=max((r['seconds'] for r in intervals),default=0.),
                  skid_flagged_scans=skids,runtime_median_ms=float(np.median(elapsed)*1000),
                  runtime_p99_ms=float(np.quantile(elapsed,.99)*1000))
        report['cases'].append(case)
        (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(case),flush=True)
    report['status']='complete'
    report['interpretation']='Offline estimator-confidence measurements only; compare identical input hashes and report changed settings. More accepted poses alone is not proof of better localization. Preserve production settings pending true-pose and closed-loop tests.'
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
