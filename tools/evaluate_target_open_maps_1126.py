#!/usr/bin/env python3
"""Matched controls and heldout openings; expert screens every candidate map."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src/f1tenth'
sys.path[:0]=[str(SRC), str(SRC/'installation')]
POLICIES={}


def initialize():
    import torch
    import train_il_s2r as IL
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    args=IL.get_args(['--task_config','results/ppo_s2r/25/backup/f1tenth.yaml'])
    rms=RunningMeanStd('agent_obs',args.obs_dim)
    rms.load('results/ppo_s2r/25/sim2real',required=True)
    for tag in ('440000','2200000'):
        ck=torch.load(f'results/ppo_s2r/25/checkpoint/model_{tag}.pt',map_location='cpu',weights_only=False)
        POLICIES[tag]=IL.portable_policy((ck['actor'],rms.mean,rms.var))


def worker(job):
    import train_il_s2r as IL
    from evaluate_expert_1126 import drive
    row=dict(job)
    if row['controller']=='expert':
        raw=drive(('target_open',row['path'],row['seed'],row['cap'],'center'))
        row.update(complete=raw['complete'],elapsed=raw['time'])
    else:
        raw=IL.eval_worker((row['path'],POLICIES[row['controller']],'eval',row['seed'],row['cap'],'center'))
        row.update(complete=bool(raw[1]),elapsed=float(raw[2]),phases=raw[4])
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=2)
    cli=p.parse_args();os.chdir(SRC)
    manifest=Path(cli.manifest); generation=json.loads(manifest.read_text())
    assert generation['status']=='generated_static_checks_passed_not_applied'
    sources=sorted(generation['sources'])
    for path,digest in generation['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    tracks=[dict(path=s,source=s,split='control') for s in sources]+generation['variants']
    jobs=[]
    for item in tracks:
        if 'sha256' in item:
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
        # Same initial/noise seed for each source and all of its variants.
        seed=7+sources.index(item['source'])
        controllers=('expert',) if item['split']=='train' else ('expert','440000','2200000')
        for cap in (8.,4.):
            for controller in controllers:
                jobs.append(dict(path=item['path'],source=item['source'],split=item['split'],
                                 seed=seed,cap=cap,controller=controller))
    dest=Path(cli.output)
    if dest.exists():raise FileExistsError('Do not overwrite a previous evaluation')
    result=dict(status='running',runs=[],total=len(jobs),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                checkpoint_sha256={tag:hashlib.sha256(Path(f'results/ppo_s2r/25/checkpoint/model_{tag}.pt').read_bytes()).hexdigest()
                                   for tag in ('440000','2200000')},
                qualification='Known geometries, unseen opening seeds in eval split. Expert is privileged; this single-seed screen is not final policy qualification.')
    with mp.get_context('spawn').Pool(cli.workers,initializer=initialize) as pool:
        for row in pool.imap_unordered(worker,jobs):
            result['runs'].append(row)
            dest.write_text(json.dumps(result,indent=2)+'\n')
            if len(result['runs'])%8==0 or not row['complete']:
                print(len(result['runs']), '/', len(jobs),row['controller'],Path(row['path']).name,
                      row['cap'],row['complete'],flush=True)
    result['status']='complete'
    result['summary']={}
    for controller in ('expert','440000','2200000'):
        result['summary'][controller]={}
        for split in ('control','train','eval'):
            rows=[r for r in result['runs'] if r['controller']==controller and r['split']==split]
            if rows:result['summary'][controller][split]=dict(complete=sum(r['complete'] for r in rows),total=len(rows))
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
