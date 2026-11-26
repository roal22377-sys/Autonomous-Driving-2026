#!/usr/bin/env python3
"""Rehearse successful frozen-PPO drives over selected TRAINING geometries.

This writes a separate candidate replay cache; running learners keep their
existing cache. No evaluation geometry or new-policy performance is used.
"""
import argparse,hashlib,json,multiprocessing as mp,os,sys,time,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def init_worker(policy,config):
    global ACTOR,MEAN,STD,CONFIG
    import torch,numpy as np
    import train_il_s2r as IL
    torch.set_num_threads(1);CONFIG=config
    IL.get_args(['--task_config',config])
    ACTOR=IL.actor_from_state(policy[0]);ACTOR.eval()
    MEAN,STD=policy[1],np.sqrt(policy[2]+1e-8)


def drive(job):
    import numpy as np,torch,random
    import train_il_s2r as IL
    from utils.env import F1Wrapper
    path,seed,dest=job;args=IL.get_args(['--task_config',CONFIG]);args.seed=seed
    np.random.seed(seed);random.seed(seed)
    env=F1Wrapper(args,IL.MAPS,obs_noise=True,sim_mode='eval',tracks=[path])
    env.fixed_speed_cap=8.;env.start_mode='center'
    observations,labels=[],[]
    try:
        obs,_=env.reset(options={'track':path})
        while True:
            with torch.no_grad():act=ACTOR(torch.as_tensor((obs-MEAN)/STD,dtype=torch.float32)).loc.numpy()
            act=np.clip(act,-1,1)
            observations.append(obs);labels.append(act)
            obs,_,term,trunc,_=env.step(act)
            if term or trunc:break
        O=np.asarray(observations,np.float16);Y=np.asarray(labels,np.float32)
        assert O.shape[1]==132 and Y.shape==(len(O),2)
        assert np.isfinite(O).all() and np.isfinite(Y).all() and abs(Y).max()<=1.000001
        row=dict(track=Path(path).name,seed=seed,complete=bool(env.success),steps=len(O),time=float(env.elapsed))
        np.savez_compressed(dest,O=O,Y=Y)
        return row
    finally:env.close()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--family',choices=('open','regular','regular_variants'),default='open')
    cli=parser.parse_args();os.chdir(S)
    import torch,numpy as np
    import train_il_s2r as IL
    from utils.policy_transfer import columns
    from eval_open_space import TRAIN_DIR,TEST_DIR,require_fresh_open_train
    root=Path(f'results/{cli.family}_replay_1126');root.mkdir(exist_ok=True)
    shards=root/'shards';shards.mkdir(exist_ok=True)
    if any(shards.glob('*.npz')):raise RuntimeError('Collection already started; inspect its live session before resuming')
    config=str(Path('results/imitation/14/backup/f1tenth.yaml').resolve())
    args=IL.get_args(['--task_config',config]);assert args.obs_dim==132
    source=Path('results/imitation/14/checkpoint/model_init.pt')
    ck=torch.load(source,map_location='cpu',weights_only=False)
    policy=IL.portable_policy((ck['actor'],ck['obs_mean'],ck['obs_var']))
    if cli.family=='open':
        tracks=sorted(Path(TRAIN_DIR).glob('*.npz'));require_fresh_open_train([str(p) for p in tracks])
        assert len(tracks)==184 and all(Path(TEST_DIR).resolve() not in p.resolve().parents for p in tracks)
    elif cli.family=='regular':
        original_names={name+direction for name in IL.MAPS for direction in ('','_rev')}
        tracks=sorted(Path(p) for p in IL.list_variants(IL.MAPS) if Path(p).stem in original_names)
        IL.require_fresh_variants([str(p) for p in tracks])
        assert len(tracks)==46 and all(IL.HOLDOUT not in str(p) for p in tracks)
    else:
        # Predeclared training augmentations across every original direction;
        # never select the held-out w105 cases used in the failure analysis.
        eligible=[Path(p) for p in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in p]
        tracks=sorted(p for p in eligible if p.stem.endswith(('_w101','_w103')))
        IL.require_fresh_variants([str(p) for p in tracks])
        expected={m+d+'_w'+str(seed)+'.npz' for m in IL.MAPS for d in ('','_rev') for seed in (101,103)}
        assert {p.name for p in tracks}==expected and len(tracks)==92
        assert all(IL.HOLDOUT not in str(p) for p in tracks)
    base=Path('results/retention_probe_1126/replay_cache.npz')
    metadata=dict(family=cli.family,source='PPO2130 function, affine-transferred IL14 init; new preview reference6',
                  source_sha256=sha(source),config_sha256=sha(config),dynamics_sha256=sha('configs/task/dynamic.yaml'),
                  base_cache_sha256=sha(base),selection='Add successful source episodes only',
                  qualification='Training-only frozen-policy rehearsal collection, not new-model validation',
                  tracks=[str(p) for p in tracks],runs=[])
    if cli.family=='regular_variants':
        metadata.update(selection_rule='Every base/direction at training seeds101 and103; exclude heldout105',
                        collection_caps=[8.0],live_training_modified=False,
                        intended_use='Separate future replay-coverage experiment only; current PPO cache is immutable')
    target=root/'collection.json';target.write_text(json.dumps(metadata,indent=2)+'\n')
    jobs=[(str(p.resolve()),50000+i,str((shards/(p.stem+'.npz')).resolve())) for i,p in enumerate(tracks)]
    started=time.monotonic()
    with mp.get_context('spawn').Pool(cli.workers,initializer=init_worker,initargs=(policy,config)) as pool:
        for row in pool.imap_unordered(drive,jobs):
            metadata['runs'].append(row)
            target.write_text(json.dumps(metadata,indent=2)+'\n')
            if len(metadata['runs'])%10==0 or not row['complete']:
                print('COLLECT',len(metadata['runs']),len(jobs),'complete',sum(r['complete'] for r in metadata['runs']),
                      'elapsed',round(time.monotonic()-started),'s',flush=True)
    with np.load(base) as z:cache={key:z[key] for key in z.files}
    assert list(cache['columns'])==columns(args)
    extra=[]
    for row in sorted(metadata['runs'],key=lambda r:r['track']):
        if row['complete']:
            with np.load(shards/(Path(row['track']).stem+'.npz')) as z:extra.append((z['O'],z['Y']))
    if not extra:raise RuntimeError('No successful source episodes; do not construct a replay cache')
    count=sum(len(o) for o,_ in extra);base_n=len(cache['O'])
    cache['O']=np.concatenate([cache['O']]+[o for o,_ in extra])
    cache['Y']=np.concatenate([cache['Y']]+[y for _,y in extra])
    cache['missing_new_features']=np.r_[cache['missing_new_features'],np.zeros(count,bool)]
    dest=root/'replay_cache.npz'
    with tempfile.NamedTemporaryFile(dir=root,prefix='replay_',suffix='.npz',delete=False) as f:
        np.savez(f,**cache);temporary=f.name
    os.replace(temporary,dest)
    metadata.update(status='complete',base_samples=base_n,extra_samples=count,total_samples=base_n+count,
                    successful_tracks=sum(r['complete'] for r in metadata['runs']),cache_sha256=sha(dest),
                    elapsed_seconds=time.monotonic()-started)
    target.write_text(json.dumps(metadata,indent=2)+'\n')
    print('DONE',json.dumps({k:metadata[k] for k in ('successful_tracks','extra_samples','total_samples','cache_sha256')}),flush=True)


if __name__=='__main__':main()
