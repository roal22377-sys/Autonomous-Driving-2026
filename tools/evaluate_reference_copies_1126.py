#!/usr/bin/env python3
"""Compare isolated racelines using a frozen policy and a privileged expert."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'src/f1tenth'
sys.path[:0]=[str(SRC),str(SRC/'installation')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True)
    p.add_argument('--model',default='ppo_s2r/25')
    p.add_argument('--tag',default='440000')
    p.add_argument('--seeds',nargs='+',type=int,default=[7])
    p.add_argument('--variants',nargs='+',default=['stored','legacy','current','current_dense_best'])
    p.add_argument('--output',required=True)
    cli=p.parse_args();os.chdir(SRC)
    import numpy as np
    import torch
    import train_il_s2r as IL
    from utils.vectorize import RunningMeanStd
    from evaluate_expert_1126 import drive
    torch.set_num_threads(1)
    generation=json.loads(Path(cli.report).read_text());assert generation['status']=='complete'
    args=IL.get_args(['--task_config',f'results/{cli.model}/backup/f1tenth.yaml'])
    checkpoint=Path(f'results/{cli.model}/checkpoint/model_{cli.tag}.pt')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    rms=RunningMeanStd('agent_obs',args.obs_dim);rms.load(f'results/{cli.model}/sim2real',required=True)
    policy=IL.portable_policy((ck['actor'],rms.mean,rms.var))
    dest=Path(cli.output)
    if dest.exists():raise FileExistsError('Do not overwrite an existing evaluation')
    dest.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='running',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                source_report_sha256=hashlib.sha256(Path(cli.report).read_bytes()).hexdigest(),
                model=cli.model,tag=cli.tag,mode='eval',cap=8.,seeds=cli.seeds,runs=[],
                limitations=['Known training geometries; not a broad qualification.',
                             'Expert uses privileged state/episode parameters; policy does not.',
                             'Only raceline/derived metadata differ within a source course; original maps remain intact.'])
    for item in generation['tracks']:
        source=Path(item['source'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==item['source_sha256']
        for variant in cli.variants:
            path=source if variant=='stored' else Path(item['variants'][variant]['path'])
            geometry=item['stored'] if variant=='stored' else item['variants'][variant]
            assert geometry['footprint']['poses_touching_occupied_samples']==0, 'Static footprint screen failed'
            if variant!='stored':
                assert hashlib.sha256(path.read_bytes()).hexdigest()==item['variants'][variant]['sha256']
            for seed in cli.seeds:
                for controller in ('policy','expert'):
                    if controller=='policy':
                        raw=IL.eval_worker((str(path),policy,'eval',seed,8.,'center'))
                        row=dict(complete=bool(raw[1]),elapsed=float(raw[2]),phases=raw[4])
                    else:
                        raw=drive(('reference',str(path),seed,8.,'center'))
                        row=dict(complete=raw['complete'],elapsed=raw['time'])
                    row.update(track=source.name,variant=variant,controller=controller,seed=seed)
                    result['runs'].append(row)
                    dest.write_text(json.dumps(result,indent=2)+'\n')
                    print(json.dumps({k:v for k,v in row.items() if k!='phases'}),flush=True)
    result['status']='complete';dest.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
