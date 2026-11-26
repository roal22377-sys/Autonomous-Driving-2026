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


def initialize(tags, model, normalizer_dir):
    import torch
    import train_il_s2r as IL
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    args=IL.get_args(['--task_config','results/'+model+'/backup/f1tenth.yaml'])
    rms=RunningMeanStd('agent_obs',args.obs_dim)
    rms.load(normalizer_dir,required=True)
    for tag in tags:
        if tag=='expert':continue
        ck=torch.load(f'results/{model}/checkpoint/model_{tag}.pt',map_location='cpu',weights_only=False)
        POLICIES[tag]=IL.portable_policy((ck['actor'],rms.mean,rms.var))


def worker(job):
    import train_il_s2r as IL
    from evaluate_expert_1126 import drive
    row=dict(job)
    if row['controller']=='expert':
        raw=drive(('target_open',row['path'],row['seed'],row['cap'],'center',row['mode']))
        row.update(complete=raw['complete'],elapsed=raw['time'])
    else:
        raw=IL.eval_worker((row['path'],POLICIES[row['controller']],row['mode'],row['seed'],row['cap'],'center'))
        row.update(complete=bool(raw[1]),elapsed=float(raw[2]),phases=raw[4])
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--mode',choices=['eval','worst'],default='eval')
    p.add_argument('--seeds',type=int,nargs='+',default=[7])
    p.add_argument('--controllers',nargs='+',default=['expert','440000','2200000'])
    p.add_argument('--model',default='ppo_s2r/25')
    p.add_argument('--base-split',choices=['all','train','eval'],default='all',
                   help='Filter disjoint-base manifests, including matched original controls')
    cli=p.parse_args();os.chdir(SRC)
    manifest=Path(cli.manifest); generation=json.loads(manifest.read_text())
    assert generation['status']=='generated_static_checks_passed_not_applied'
    if any(tag!='expert' and not tag.isdigit() for tag in cli.controllers):
        raise ValueError('Controllers must be expert or explicit numeric checkpoints')
    source_splits={path:{r['split'] for r in generation['variants'] if r['source']==path}
                   for path in generation['sources']}
    if cli.base_split!='all':
        if not generation.get('base_splits_disjoint') or any(len(s)!=1 for s in source_splits.values()):
            raise ValueError('Base filtering requires genuinely disjoint-base metadata')
        train={r['base_geometry_sha256'] for r in generation['variants'] if r['split']=='train'}
        valid={r['base_geometry_sha256'] for r in generation['variants'] if r['split']=='eval'}
        if not train or not valid or train&valid:raise ValueError('Base geometry overlap')
    sources=sorted(generation['sources'])
    for path,digest in generation['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    tracks=[dict(path=s,source=s,split='control') for s in sources]+generation['variants']
    jobs=[]
    for item in tracks:
        if cli.base_split!='all' and source_splits[item['source']]!={cli.base_split}:continue
        if 'sha256' in item:
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
        # Same initial/noise seed for each source and all of its variants.
        controllers=[c for c in cli.controllers if item['split']!='train' or c=='expert']
        for seed_base in cli.seeds:
            seed=seed_base+sources.index(item['source'])
            for cap in (8.,4.):
                for controller in controllers:
                    jobs.append(dict(path=item['path'],source=item['source'],split=item['split'],
                                     source_splits=sorted(source_splits[item['source']]),model=cli.model,
                                     seed=seed,seed_base=seed_base,cap=cap,controller=controller,mode=cli.mode))
    dest=Path(cli.output)
    if dest.exists():raise FileExistsError('Do not overwrite a previous evaluation')
    model_dir=Path('results')/cli.model
    metadata=json.loads((model_dir/'backup/ppo_args.json').read_text())
    normalizer_dir=model_dir/'sim2real'
    if not (normalizer_dir/'agent_obs_scale.pkl').exists():
        normalizer_dir=Path('results')/metadata['il']/'sim2real'
    result=dict(status='running',runs=[],total=len(jobs),mode=cli.mode,seeds=cli.seeds,
                model=cli.model,base_split_filter=cli.base_split,
                normalizer_directory=str(normalizer_dir),
                normalizer_sha256=hashlib.sha256((normalizer_dir/'agent_obs_scale.pkl').read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256((model_dir/'backup/f1tenth.yaml').read_bytes()).hexdigest(),
                dynamics_sha256=hashlib.sha256(Path('configs/task/dynamic.yaml').read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                checkpoint_sha256={tag:hashlib.sha256((model_dir/'checkpoint'/f'model_{tag}.pt').read_bytes()).hexdigest()
                                   for tag in cli.controllers if tag!='expert'},
                qualification=generation['qualification']+' Matched original/augmented source seeds; expert is privileged. No real-car qualification.')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with mp.get_context('spawn').Pool(cli.workers,initializer=initialize,
                                     initargs=(cli.controllers,cli.model,str(normalizer_dir))) as pool:
        for row in pool.imap_unordered(worker,jobs):
            result['runs'].append(row)
            dest.write_text(json.dumps(result,indent=2)+'\n')
            if len(result['runs'])%8==0 or not row['complete']:
                print(len(result['runs']), '/', len(jobs),row['controller'],Path(row['path']).name,
                      row['cap'],row['complete'],flush=True)
    result['status']='complete'
    result['summary']={}
    for controller in cli.controllers:
        result['summary'][controller]={}
        for split in ('control','train','eval'):
            rows=[r for r in result['runs'] if r['controller']==controller and r['split']==split]
            if rows:result['summary'][controller][split]=dict(complete=sum(r['complete'] for r in rows),total=len(rows))
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
