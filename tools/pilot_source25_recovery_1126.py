#!/usr/bin/env python3
"""Bounded source25 recovery IL with broad, adverse-state and actual-loss anchors."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/f1tenth'
sys.path[:0]=[str(SRC),str(SRC/'installation'),str(ROOT/'tools')]
from collect_expert_recovery_probe_1126 import sha


def initialize():
    import collect_expert_recovery_probe_1126 as C
    C.initialize('/tmp/source25_pilot_evaluation',True)


def evaluate(job):
    import pilot_recovery_adaptation_1126 as P
    dest,row,tag=job
    P.DEST=Path(dest)
    return P.evaluate_case((row,tag))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--prepare_only',action='store_true')
    cli=parser.parse_args();os.chdir(SRC)
    import numpy as np
    import torch
    import train_il_s2r as IL
    from utils.policy_transfer import columns
    from utils.training_mix import groups_for,step_budgets
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    dest=Path(cli.destination)
    if dest.exists():raise FileExistsError('Preserve previous pilot')
    source=Path('results/ppo_s2r/25/checkpoint/model_3080000.pt')
    config=Path('results/ppo_s2r/29/backup/f1tenth.yaml')
    paths={k:Path('results')/v/'collection.json' for k,v in dict(
        recovery='source25_targeted_recovery_1126',lost='source25u70_lost_rehearsal_1126',stress='source29u20_stress_rehearsal_1126').items()}
    docs={k:json.loads(p.read_text()) for k,p in paths.items()}
    audits={k:json.loads(p.with_name('audit.json').read_text()) for k,p in paths.items()}
    for key,doc in docs.items():
        assert doc['status']=='complete_not_applied' and audits[key]['status']=='passed_not_applied'
        assert audits[key]['collection_sha256']==sha(paths[key])
        if key!='recovery':
            for item in doc['artifacts'].values():assert sha(item['path'])==item['sha256']
    assert docs['recovery']['student_sha256']==docs['lost']['artifacts']['source']['sha256']==sha(source)
    assert docs['recovery']['config_sha256']==docs['lost']['artifacts']['config']['sha256']==docs['stress']['artifacts']['config']['sha256']==sha(config)
    assert docs['recovery']['dynamics_sha256']==docs['lost']['artifacts']['dynamics']['sha256']==docs['stress']['artifacts']['dynamics']['sha256']==sha('configs/task/dynamic.yaml')
    args=IL.get_args(['--task_config',str(config)])
    names=columns(args)
    rms=RunningMeanStd('agent_obs',132);rms.load('results/imitation/1607/sim2real',required=True)
    norm=Path('results/imitation/1607/sim2real/agent_obs_scale.pkl')
    assert docs['recovery']['normalizer_sha256']==docs['lost']['artifacts']['normalizer']['sha256']==sha(norm)
    groups=groups_for(args,[p for p in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in p])
    training={str(Path(p).resolve()) for group in groups.values() for p in group}
    manifest_path=Path('results/imitation/14/backup/track_manifest.json')
    manifest=json.loads(manifest_path.read_text())
    for family,tracks in manifest['training'].items():
        assert {str(Path(p).resolve()) for p in tracks}=={str(Path(p).resolve()) for p in groups[family]}
    rng=np.random.default_rng(112680)
    archive=Path('results/imitation/14/data/round000.npz')
    quotas=step_budgets(dict(args.training_mix),16000)
    with np.load(archive,allow_pickle=False) as z:
        raw=z['O'];assert raw.shape==(1200000,132)
        chunks=[];offset=0
        for family,budget in step_budgets(manifest['step_shares'],len(raw)).items():
            chunks.append(raw[rng.choice(budget,quotas[family],replace=False)+offset].astype(np.float32));offset+=budget
        broad=np.concatenate(chunks)
    # Same source function on synthetic loss inputs remains a broad anchor;
    # the separate lost bank below contains actual closed-loop loss trajectories.
    missing=rng.choice(len(broad),len(broad)//4,replace=False)
    broad[np.ix_(missing,[i for i,n in enumerate(names) if n.startswith('preview/')])]=0
    provenance={}
    def sample_successes(key,count):
        successes=sorted((r for r in docs[key]['runs'] if r['admissible_for_rehearsal']),key=lambda r:r['seed'])
        assert len(successes)==audits[key]['completed'] and successes
        q,rem=divmod(count,len(successes));parts=[];records=[]
        for i,row in enumerate(successes):
            assert row['track'] in training and sha(row['track'])==row['track_sha256']
            assert row['complete'] and not row['collision'] and len(row['lap_times'])==2
            assert sha(row['shard'])==row['shard_sha256']
            with np.load(row['shard'],allow_pickle=False) as z:
                assert list(z['columns'])==names
                n=q+int(i<rem);assert n<=len(z['O'])
                selected=rng.choice(len(z['O']),n,replace=False)
                O=z['O'][selected].copy();assert np.isfinite(O).all()
                if key=='lost':assert np.all(O[:,names.index('preview/valid')]==0)
                parts.append(O)
                records.append(dict(track=row['track'],seed=row['seed'],rows=n,shard=row['shard'],shard_sha256=row['shard_sha256'],indices=selected.tolist()))
        provenance[key]=records
        return np.concatenate(parts)
    raw_banks=dict(broad=broad,stress=sample_successes('stress',8000),lost=sample_successes('lost',8000))
    ck=torch.load(source,map_location='cpu',weights_only=False)
    teacher=IL.actor_from_state(ck['actor']);teacher.eval()
    actor=IL.actor_from_state(ck['actor']);actor.train()
    banks={}
    for name,O in raw_banks.items():
        X=torch.as_tensor((O-rms.mean)/np.sqrt(rms.var+1e-8),dtype=torch.float32)
        with torch.no_grad():T=torch.cat([teacher(X[i:i+2048]).loc for i in range(0,len(X),2048)])
        assert torch.isfinite(X).all() and torch.isfinite(T).all()
        banks[name]=(X,T)
    recovery=[]
    rows=sorted((r for r in docs['recovery']['runs'] if r.get('outcome')=='rescued'),key=lambda r:r['track'])
    assert len(rows)==audits['recovery']['rescued']==3
    for row in rows:
        assert row['track'] in training and sha(row['shard'])==row['shard_sha256']
        with np.load(row['shard'],allow_pickle=False) as z:
            assert list(z['columns'])==names
            X=torch.as_tensor((z['O']-rms.mean)/np.sqrt(rms.var+1e-8),dtype=torch.float32)
            Y=torch.as_tensor(z['Y'],dtype=torch.float32)
            early=z['seconds_since_handoff']<=2
            assert early.any() and (~early).any()
            recovery.append((X,Y,np.flatnonzero(early),np.flatnonzero(~early)))
    corpus_path=Path('results/recovery_corpus_round2_1126/manifest.json')
    corpus=json.loads(corpus_path.read_text());assert corpus['total_cases']==len(corpus['cases'])==13
    if cli.prepare_only:
        print(json.dumps(dict(status='prepared_no_fit',source_sha256=sha(source),banks={k:len(v[0]) for k,v in banks.items()},recovery_cases=3,recovery_rows=sum(len(x[0]) for x in recovery),driving_cases=13)));return
    (dest/'checkpoint').mkdir(parents=True);(dest/'backup').mkdir()
    shutil.copyfile(config,dest/'backup/f1tenth.yaml')
    report=dict(status='fitting',source_checkpoint=str(source),source_sha256=sha(source),config_sha256=sha(config),normalizer_sha256=sha(norm),
                input_collections={k:dict(path=str(p),sha256=sha(p),audit_sha256=sha(p.with_name('audit.json'))) for k,p in paths.items()},
                archive_sha256=sha(archive),manifest_sha256=sha(manifest_path),corpus_sha256=sha(corpus_path),
                anchor_banks={k:len(v[0]) for k,v in banks.items()},broad_family_quotas=quotas,synthetic_broad_loss_rows=len(missing),provenance=provenance,
                recovery_cases=3,steps=100,lr=1e-5,seed=112680,batch=dict(broad=192,stress=96,lost=96,recovery=128),
                loss='0.75*(0.5*broadMSE+0.25*stressMSE+0.25*lostMSE)+0.25*expertRecoveryMSE; steer/speedweights1/.25',
                checkpoints={},fit=[],driving=[],
                qualification='Source25u70 isolatedoffline recoveryfit, notPPO30. Three source25failures useexactprefix teachercontinuations; oneisfromreset. Anchorsalllabelledbyfrozen25u70. Loststatesarefromsource25successfulactualmissingpreviewrollouts; stressstatesarefromsource29successfuladverserollouts,relabelledbysource25functionandNOTclaimedassuccessfulsource25trajectories. Broadarchiveusesrecordedtrainingfamilyorder. Newreference/linepenaltyexperimentsremainseparate; currentreferenceunchanged. Noheldoutgradients,cacheorvehiclechangesorautomaticpromotion.')
    def save():
        temp=dest/'pilot.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(dest/'pilot.json')
    weights=torch.tensor([1.,.25]);optimizer=torch.optim.Adam(actor.parameters(),lr=1e-5)
    torch.manual_seed(112680)
    def metrics(step):
        actor.eval()
        with torch.no_grad():
            anchor={k:torch.cat([(actor(X[i:i+2048]).loc-T[i:i+2048]).square() for i in range(0,len(X),2048)]).mean(0).tolist() for k,(X,T) in banks.items()}
            values=[]
            for X,Y,early,late in recovery:
                e=torch.cat([(actor(X[i:i+2048]).loc-Y[i:i+2048]).square() for i in range(0,len(X),2048)])
                values.append(dict(early_mse=e[early].mean(0).tolist(),late_mse=e[late].mean(0).tolist()))
        actor.train();return dict(step=step,anchor_mse=anchor,recovery=values)
    report['fit'].append(metrics(0));assert all(max(v)==0 for v in report['fit'][0]['anchor_mse'].values());save()
    for step in range(1,101):
        anchor_loss=0.
        for name,count,weight in [('broad',192,.5),('stress',96,.25),('lost',96,.25)]:
            X,T=banks[name];ix=rng.integers(len(X),size=count)
            anchor_loss=anchor_loss+weight*((actor(X[ix]).loc-T[ix]).square()*weights).mean()
        xp,yp=[],[]
        for early_part in [True,False]:
            ids=rng.integers(len(recovery),size=64)
            for i,(X,Y,early,late) in enumerate(recovery):
                ix=rng.choice(early if early_part else late,int((ids==i).sum()),replace=True);xp.append(X[ix]);yp.append(Y[ix])
        loss=.75*anchor_loss+.25*((actor(torch.cat(xp)).loc-torch.cat(yp)).square()*weights).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
        if step in [50,100]:
            assert all(torch.isfinite(v).all() for v in actor.state_dict().values())
            path=dest/'checkpoint'/('model_%s.pt'%step);torch.save(dict(actor=actor.state_dict(),obs_mean=rms.mean,obs_var=rms.var),path)
            report['checkpoints'][str(step)]=dict(path=str(path),sha256=sha(path));report['fit'].append(metrics(step));save();print('FIT',step,report['fit'][-1]['anchor_mse'],flush=True)
    report['status']='evaluating_training_corpus';save()
    jobs=[(str(dest),row,str(tag)) for tag in [50,100] for row in corpus['cases']]
    with mp.get_context('spawn').Pool(2,initializer=initialize) as pool:
        for result in pool.imap_unordered(evaluate,jobs):
            report['driving'].append(result);save();print('CASE',result,flush=True)
    assert len(report['driving'])==26 and sha(source)==report['source_sha256']
    report['status']='complete_not_promoted';save()


if __name__=='__main__':main()
