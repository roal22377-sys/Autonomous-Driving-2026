#!/usr/bin/env python3
"""Matched evaluations of adapted checkpoints, including phase metrics and seed IDs."""
import argparse, json, os, sys, multiprocessing as mp, hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]; S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def regular_tracks(il, scope='standard'):
    """Keep the selection cohort stable; expose broader original-map audits."""
    tracks=sorted(il.list_variants(il.MAPS))
    bases={m+d for m in il.MAPS for d in ('','_rev')}
    if scope=='standard':
        eval_bases={m+d for m in il.EVAL_MAPS for d in ('','_rev')}
        return [t for t in tracks if il.variant_map(t) in il.EVAL_MAPS and
                (il.HOLDOUT in t or Path(t).stem in eval_bases)][:44]
    missing=bases-{Path(t).stem for t in tracks}
    if missing:
        raise ValueError('Missing original maps: '+', '.join(sorted(missing)))
    if scope=='all_base':
        return [t for t in tracks if Path(t).stem in bases]
    if scope=='all_variants':
        return tracks
    raise ValueError('Unknown regular scope: '+scope)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='imitation/14')
    p.add_argument('--tag',default='init')
    p.add_argument('--normalizer_model')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',help='separate report filename for a subset evaluation')
    caps=p.add_mutually_exclusive_group()
    caps.add_argument('--caps',type=float,nargs='+',help='diagnostic cap override, centerline starts only')
    caps.add_argument('--cycle_caps',type=float,nargs='+',help='one center start per track, cycling caps as in PPO low-cap validation')
    p.add_argument('--seeds',type=int,nargs='+',default=[7])
    p.add_argument('--families',nargs='+',default=['regular','short','hall','real','2025','open'])
    p.add_argument('--regular_scope',choices=['standard','all_base','all_variants'],default='standard',
                   help='broader scopes audit known original geometries; they are not held-out generalization tests')
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
        normalizer_path=None
    else:
        rms=RunningMeanStd('agent_obs',args.obs_dim)
        normalizer_dir=Path(f'results/{cli.normalizer_model or cli.model}/sim2real')
        rms.load(str(normalizer_dir),required=True)
        normalizer_path=normalizer_dir/'agent_obs_scale.pkl'
        mean,var=rms.mean,rms.var
    policy=IL.portable_policy((ck['actor'],mean,var))
    regular=regular_tracks(IL,cli.regular_scope)
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
    if cli.caps:
        sets={g:(tracks,tuple((cap,'center') for cap in cli.caps)) for g,(tracks,_) in sets.items()}
    elif cli.cycle_caps:
        sets={g:(tracks,((tuple(cli.cycle_caps),'center'),)) for g,(tracks,_) in sets.items()}
    out=dict(model=cli.model,tag=cli.tag,seeds=cli.seeds,checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),
             normalizer_source='embedded_checkpoint' if normalizer_path is None else str(normalizer_path),
             normalizer_file_sha256=None if normalizer_path is None else hashlib.sha256(normalizer_path.read_bytes()).hexdigest(),
             normalizer_arrays_sha256=hashlib.sha256(b''.join(np.asarray(x,dtype='<f8').tobytes(order='C') for x in (mean,var))).hexdigest(),
             normalizer_shape=list(np.asarray(mean).shape),
             dynamics_sha256=hashlib.sha256(Path('configs/task/dynamic.yaml').read_bytes()).hexdigest(),
             localization_forced_lost=os.environ.get('F1_LOC_LOST')=='1',
             diagnostic_caps=cli.caps,
             diagnostic_cycle_caps=cli.cycle_caps,
             regular_scope=cli.regular_scope,
             config_sha256=hashlib.sha256(Path(args.task_config).read_bytes()).hexdigest(),
             config=str(args.task_config),families={},notes='Real and 2025 geometries are in training; hall validation excludes 2025.')
    dest=Path('results/analysis1126');dest.mkdir(exist_ok=True)
    scope_suffix='' if cli.regular_scope=='standard' else '_'+cli.regular_scope
    target=dest/(cli.output or f'{cli.model.replace("/","_")}_{cli.tag}_seeds{"-".join(map(str,cli.seeds))}{scope_suffix}.json')
    for name in cli.families:
        tracks,settings=sets[name]
        jobs=[(str(Path(t).resolve()),policy,'eval',seed+i,
               cap[i%len(cap)] if isinstance(cap,(tuple,list)) else cap,start)
              for seed in cli.seeds for cap,start in settings for i,t in enumerate(sorted(tracks))]
        res=IL.run_pool(IL.eval_worker,jobs,min(cli.workers,len(jobs)),len(jobs),name,'episodes')
        complete=[r for r in res if r[1]]
        row=dict(success=sum(r[1] for r in res)/len(res),complete=len(complete),total=len(res),
                 track_names=[Path(t).name for t in sorted(tracks)],
                 runs=[dict(seed=j[3],result=r) for j,r in zip(jobs,res)])
        out['families'][name]=row
        target.write_text(json.dumps(out,indent=2)+'\n')
        print(f'{name}: {len(complete)}/{len(res)} ({100*row["success"]:.2f}%) -> {target}',flush=True)
    print('DONE',target,flush=True)


if __name__=='__main__':main()
