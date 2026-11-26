#!/usr/bin/env python3
"""Matched evaluations of adapted checkpoints, including phase metrics and seed IDs."""
import argparse, json, os, sys, multiprocessing as mp, hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]; S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='imitation/14')
    p.add_argument('--tag',default='init')
    p.add_argument('--normalizer_model')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',help='separate report filename for a subset evaluation')
    p.add_argument('--seeds',type=int,nargs='+',default=[7])
    p.add_argument('--families',nargs='+',default=['regular','short','hall','real','2025','open'])
    cli=p.parse_args()
    os.chdir(S)
    import numpy as np, torch
    import train_il_s2r as IL
    from utils.training_mix import evaluation_sets
    from utils.vectorize import RunningMeanStd
    args=IL.get_args(['--task_config',f'results/{cli.model}/backup/f1tenth.yaml','--workers',str(cli.workers)])
    ckpath=Path(f'results/{cli.model}/checkpoint/model{("_"+cli.tag) if cli.tag != "best" else ""}.pt')
    if not ckpath.exists() and cli.tag=='best':
        ckpath=Path(f'results/{cli.model}/sim2real/actor.pt')
        ck={'actor':torch.load(ckpath,map_location='cpu',weights_only=False).state_dict()}
    else:
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    if 'obs_mean' in ck:
        mean,var=ck['obs_mean'],ck['obs_var']
    else:
        rms=RunningMeanStd('agent_obs',args.obs_dim)
        rms.load(f'results/{cli.normalizer_model or cli.model}/sim2real',required=True)
        mean,var=rms.mean,rms.var
    policy=IL.portable_policy((ck['actor'],mean,var))
    all_tracks=IL.list_variants(IL.MAPS)
    regular=sorted(t for t in all_tracks if IL.variant_map(t) in IL.EVAL_MAPS and
                   (IL.HOLDOUT in t or Path(t).stem in [m+d for m in IL.EVAL_MAPS for d in ('','_rev')]))[:44]
    from utils.track_short import short_tracks
    from eval_open_space import TEST_DIR
    extra=evaluation_sets(args)
    if not extra['real']:
        extra['real']=sorted(str(p.resolve()) for p in Path('installation/maps_real').glob('Practice1126b*.npz'))
    sets=dict(regular=(regular,IL.EVAL_SETTINGS),short=(short_tracks(True),IL.SHORT_SETTINGS),
              hall=(extra['hall'],((8.,'center'),(4.,'center'))),
              real=(extra['real'],((8.,'center'),(4.,'center'))),
              open=(sorted(str(p) for p in Path(TEST_DIR,'big').glob('*.npz')),((8.,'center'),(5.,'center'))),
              **{'2025':(sorted(str(p.resolve()) for p in Path('installation/maps_hall').glob('Course2025*.npz')),
                         ((8.,'center'),(4.,'center')))})
    out=dict(model=cli.model,tag=cli.tag,seeds=cli.seeds,checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),
             dynamics_sha256=hashlib.sha256(Path('configs/task/dynamic.yaml').read_bytes()).hexdigest(),
             config=str(args.task_config),families={},notes='Real and 2025 geometries are in training; hall validation excludes 2025.')
    dest=Path('results/analysis1126');dest.mkdir(exist_ok=True)
    target=dest/(cli.output or f'{cli.model.replace("/","_")}_{cli.tag}_seeds{"-".join(map(str,cli.seeds))}.json')
    for name in cli.families:
        tracks,settings=sets[name]
        jobs=[(str(Path(t).resolve()),policy,'eval',seed+i,cap,start)
              for seed in cli.seeds for cap,start in settings for i,t in enumerate(sorted(tracks))]
        res=IL.run_pool(IL.eval_worker,jobs,min(cli.workers,len(jobs)),len(jobs),name,'episodes')
        complete=[r for r in res if r[1]]
        row=dict(success=sum(r[1] for r in res)/len(res),complete=len(complete),total=len(res),
                 runs=[dict(seed=j[3],result=r) for j,r in zip(jobs,res)])
        out['families'][name]=row
        target.write_text(json.dumps(out,indent=2)+'\n')
        print(f'{name}: {len(complete)}/{len(res)} ({100*row["success"]:.2f}%) -> {target}',flush=True)
    print('DONE',target,flush=True)


if __name__=='__main__':main()
