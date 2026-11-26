#!/usr/bin/env python3
"""Controlled IL retention probes on fixed collected data; never auto-start PPO."""
import os,sys,json,copy,gc,shutil,argparse,tempfile,multiprocessing as mp
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--modes',nargs='+',default=['current_stats','shared_stats','shared_stats_replay','current_stats_replay'],
                        choices=['current_stats','shared_stats','shared_stats_replay','current_stats_replay'])
    parser.add_argument('--output',default='comparison.json')
    parser.add_argument('--start_tag',default='init',help='IL14 snapshot to adapt; replay teacher stays the original PPO')
    cli=parser.parse_args()
    os.chdir(S)
    import numpy as np,torch
    from ruamel.yaml import YAML
    import train_il_s2r as IL
    from utils.vectorize import RunningMeanStd
    from utils.policy_transfer import columns,expand_state
    from utils.training_mix import hall_split
    torch.set_num_threads(4)
    args=IL.get_args(['--task_config','configs/task/f1tenth_preview_1126.yaml','--workers','4','--bc_threads','4',
                      '--hidden_dim','512','--batch','2048'])
    chunks=[]
    for name in ['round000.npz','round001.npz']:
        with np.load(Path('results/imitation/14/data')/name) as z:chunks.append(tuple(z[k] for k in ['O','Y','N']))
    O,Y,N=[np.concatenate([x[j] for x in chunks]) for j in range(3)]
    init=torch.load('results/imitation/14/checkpoint/model_init.pt',map_location='cpu',weights_only=False)
    start=init if cli.start_tag=='init' else torch.load(f'results/imitation/14/checkpoint/model_{cli.start_tag}.pt',map_location='cpu',weights_only=False)
    assert np.array_equal(start['obs_mean'],init['obs_mean']) and np.array_equal(start['obs_var'],init['obs_var'])
    teacher=IL.actor_from_state(init['actor']);teacher.eval()
    new_rms=RunningMeanStd('agent_obs',132);new_rms.mean=init['obs_mean'];new_rms.var=init['obs_var'];new_rms.count=1200000
    old_cfg=YAML().load(Path('results/ppo_s2r/2130/backup/f1tenth.yaml'))
    old_rms=RunningMeanStd('agent_obs',129);old_rms.load('results/ppo_s2r/2130/sim2real',required=True)
    names=columns(args);shared=[names.index(k) for k in columns(old_cfg)]
    kept=copy.deepcopy(new_rms);kept.mean[shared]=old_rms.mean;kept.var[shared]=old_rms.var
    rebased=expand_state(start['actor'],names,names,new_rms.mean,new_rms.var,kept.mean,kept.var)
    preserved=IL.actor_from_state(rebased);preserved.eval()
    start_actor=IL.actor_from_state(start['actor']);start_actor.eval()
    with torch.no_grad():
        a=start_actor(torch.as_tensor(new_rms.normalize(O[:512].astype(np.float32)),dtype=torch.float32)).loc
        b=preserved(torch.as_tensor(kept.normalize(O[:512].astype(np.float32)),dtype=torch.float32)).loc
        error=float((a-b).abs().max());assert error<5e-5,error
    # Current regular/open observations: exact step-quota order from round 0.
    assert len(chunks[0][0])==1200000
    assert list(args.training_mix)==['hall','short','regular','open','real']
    assert [args.training_mix[k] for k in args.training_mix]==[.4,.3,.15,.05,.1]
    current=chunks[0][0][840000:1080000]
    anchor_O=[current.copy()];anchor_Y=[]
    with torch.no_grad():
        anchor_Y.append(np.concatenate([teacher(torch.as_tensor(new_rms.normalize(current[i:i+4096].astype(np.float32)),dtype=torch.float32)).loc.numpy()
                                        for i in range(0,len(current),4096)]))
    # Broad historical training observations; labels come from the validated PPO, not the older expert.
    archive=Path('/home/daehyeok/Autonomous-Driving-2026/src/f1tenth/results/imitation/13/data/round000.npz')
    rng=np.random.default_rng(17)
    with np.load(archive) as z:
        raw=z['O'];old_O=raw[rng.choice(len(raw),200000,replace=False)].copy()
    del raw
    old_teacher=torch.load('results/ppo_s2r/2130/sim2real/actor.pt',map_location='cpu',weights_only=False);old_teacher.eval()
    with torch.no_grad():
        old_Y=np.concatenate([old_teacher(torch.as_tensor(old_rms.normalize(old_O[i:i+4096].astype(np.float32)),dtype=torch.float32)).loc.numpy()
                              for i in range(0,len(old_O),4096)])
    expanded=np.tile(kept.mean.astype(np.float16),(len(old_O),1));expanded[:,shared]=old_O
    anchor_O.append(expanded);anchor_Y.append(old_Y)
    anchor_O=np.concatenate(anchor_O);anchor_Y=np.concatenate(anchor_Y)
    del old_O,expanded,old_teacher,chunks,current;gc.collect()
    all_tracks=IL.list_variants(IL.MAPS)
    regular=sorted(t for t in all_tracks if IL.variant_map(t) in IL.EVAL_MAPS and IL.HOLDOUT in t)[::3][:8]
    from utils.track_short import short_tracks
    hall=hall_split()[1][::6][:8]
    cases=dict(regular=(regular,((8.,'center'),(4.,'center'))),short=(short_tracks(True)[::4],IL.SHORT_SETTINGS),
               hall=(hall,((8.,'center'),(4.,'center'))),
               known=(sorted(str(p) for p in Path('installation/maps_hall').glob('Course2025*.npz'))+
                      sorted(str(p) for p in Path('installation/maps_real').glob('Practice1126b*.npz')),((8.,'center'),(4.,'center'))))
    root=Path('results/retention_probe_1126');root.mkdir(exist_ok=True)
    cache=root/'replay_cache.npz'
    if not cache.exists():
        missing=np.r_[np.zeros(len(anchor_O)-len(old_Y),bool),np.ones(len(old_Y),bool)]
        with tempfile.NamedTemporaryFile(dir=root,prefix='replay_',suffix='.npz',delete=False) as f:
            np.savez(f,O=anchor_O,Y=anchor_Y,columns=np.asarray(names),old_columns=np.asarray(columns(old_cfg)),
                     missing_new_features=missing,teacher=np.asarray('ppo_s2r/2130'))
            cache_tmp=f.name
        os.replace(cache_tmp,cache)
    out=dict(qualification='Fixed small validation cohort for tuning only; no PPO/deployment qualification',
             samples=len(O),steps=2000,lr=.0001,start_tag=cli.start_tag,initial_equivalence_max_error=error,cohort={g:[Path(p).name for p in ps] for g,(ps,_) in cases.items()},trials={})
    for mode in cli.modes:
        np.random.seed(1);torch.manual_seed(1)
        rms=copy.deepcopy(new_rms if mode.startswith('current_stats') else kept)
        actor=IL.actor_from_state(start['actor'] if mode.startswith('current_stats') else rebased)
        train_O,train_Y,train_N=O,Y,N
        if mode.endswith('replay'):
            pick=rng.integers(len(anchor_O),size=len(O))
            train_O=np.concatenate([O,anchor_O[pick]]);train_Y=np.concatenate([Y,anchor_Y[pick]])
            train_N=np.concatenate([N,np.zeros(len(pick),bool)])
        print('PROBE',mode,'samples',len(train_O),'2000 gradient steps; anchor labels from frozen PPO',flush=True)
        val=IL.train_bc(args,actor,train_O,train_Y,rms,2000,.0001,train_N)
        if mode.endswith('replay'):del train_O,train_Y,train_N;gc.collect()
        label=mode if cli.start_tag=='init' else f'{mode}_from_{cli.start_tag}'
        dest=root/label;(dest/'checkpoint').mkdir(parents=True,exist_ok=True);(dest/'backup').mkdir(exist_ok=True)
        shutil.copyfile('configs/task/f1tenth_preview_1126.yaml',dest/'backup/f1tenth.yaml')
        torch.save(dict(actor=actor.state_dict(),obs_mean=rms.mean,obs_var=rms.var),dest/'checkpoint/model_probe.pt')
        policy=(actor.state_dict(),rms.mean,rms.var);scores={}
        for group,(paths,settings) in cases.items():
            rate,lap,results=IL.evaluate(args,policy,paths,settings=settings)
            scores[group]=dict(success=rate,lap=lap,res=results)
            print(mode,group,rate,flush=True)
        out['trials'][label]=dict(val=val,scores=scores)
        (root/cli.output).write_text(json.dumps(out,indent=2)+'\n')
        del actor,policy;gc.collect()
    print('DONE retention probe',root/cli.output,flush=True)


if __name__=='__main__':main()
