#!/usr/bin/env python3
"""Bounded, resumable multi-family reference preparation; never invokes IL/PPO."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'src/f1tenth'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--max-tracks',type=int,default=5)
    p.add_argument('--max-seconds',type=float,default=900.)
    cli=p.parse_args()
    manifest=Path(cli.manifest).resolve();data=json.loads(manifest.read_text())
    rows=data.get('tracks',data.get('train',[]))
    assert all(row['role']=='train' for row in rows),'Preparation uses training-only development tracks'
    families={f:[r for r in rows if r['family']==f] for f in ['real','hall','short','regular','open']}
    # Include the known2025 example while guaranteeing all families first.
    families['hall'].sort(key=lambda r:(not Path(r['path']).name.startswith('Course2025'),r['length_m']))
    selected=[]
    while any(families.values()):
        for family in families:
            if families[family]:selected.append(families[family].pop(0))
    selected=selected[:cli.max_tracks]
    destination=Path(cli.output_dir).resolve();destination.mkdir(parents=True,exist_ok=True)
    state_file=destination/'preparation.json'
    optimizer=ROOT/'tools/mintime_joint_1127.py';code_hash=sha(optimizer)
    if state_file.exists():
        state=json.loads(state_file.read_text())
        assert state['manifest_sha256']==sha(manifest) and state['optimizer_sha256']==code_hash,'Use a new directory for changed inputs/code'
    else:
        state=dict(status='preparing',training_started=False,manifest_sha256=sha(manifest),
                   optimizer_sha256=code_hash,selected_tracks=selected,jobs=[])
    started=time.monotonic();child=None
    def save():
        state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        state_file.write_text(json.dumps(state,indent=2)+'\n')
    def stop_child():
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10)
    try:
        for index,row in enumerate(selected):
            if any(x['source']==row['path'] and x['status']=='complete' for x in state['jobs']):continue
            if sha(optimizer)!=code_hash:raise RuntimeError('Optimizer changed during a frozen batch')
            assert sha(row['path'])==row['sha256']
            if time.monotonic()-started>cli.max_seconds:
                state['status']='bounded_time_limit';save();return
            attempts=sum(x['source']==row['path'] for x in state['jobs'])
            folder=destination/f'{index:03d}_{Path(row["path"]).stem}_attempt{attempts}'
            job=dict(source=row['path'],family=row['family'],source_sha256=row['sha256'],
                     status='running',output_dir=str(folder))
            state['jobs'].append(job);save()
            argv=[sys.executable,'-u',str(optimizer),'--output-dir',str(folder),
                  '--tracks',row['path'],'--starts','joint_from_center','--maxiter','100']
            log=destination/f'{index:03d}_attempt{attempts}.log'
            with log.open('w') as stream:
                child=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                job['pid']=child.pid;save();checked=0.
                while child.poll() is None:
                    if time.monotonic()-started>cli.max_seconds:
                        stop_child();job['status']='stopped_time_limit';state['status']='bounded_time_limit';save();return
                    if time.monotonic()-checked>=30:
                        usage=subprocess.run([sys.executable,str(ROOT/'tools/check_usage_stop_1126.py')],cwd=ROOT,
                                             capture_output=True,text=True)
                        job['latest_quota_check']=usage.stdout.strip();checked=time.monotonic()
                        if usage.returncode!=0:
                            stop_child();job['status']='stopped_by_quota_guard';state['status']='quota_stop_or_unknown';save();return
                    time.sleep(1)
                job['exit_code']=child.returncode
            job['status']='complete' if child.returncode==0 else 'failed'
            report_file=folder/'report.json'
            if report_file.exists():
                result=json.loads(report_file.read_text());job['report']=str(report_file)
                job['dense_feasible_candidates']=sum(bool(c['eligible_for_closed_loop_review'])
                    for track in result.get('tracks',[]) for c in track['candidates'].values())
            save();print(json.dumps(job),flush=True)
        state['status']='bounded_suite_complete_not_training_ready';save()
    except KeyboardInterrupt:
        stop_child()
        if state['jobs'] and state['jobs'][-1]['status']=='running':state['jobs'][-1]['status']='interrupted'
        state['status']='interrupted';save();raise
    except Exception:
        stop_child();state['status']='error';save();raise


if __name__=='__main__':main()
