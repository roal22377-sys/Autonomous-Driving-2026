#!/usr/bin/env python3
"""Equal-budget source25 adaptation to stored versus recomputed reference lines."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'src/f1tenth'
sys.path[:0]=[str(SRC),str(SRC/'installation'),str(ROOT/'tools')]
from collect_expert_recovery_probe_1126 import sha


def initialize():
    import collect_expert_recovery_probe_1126 as C
    C.initialize('/tmp/reference_adaptation_eval',True)


def evaluate(job):
    import collect_expert_recovery_probe_1126 as C
    from utils.env import F1Wrapper
    from utils.driving_metrics import DrivingMetrics
    import random
    np,torch=C.np,C.torch
    branch,checkpoint,track,seed,cap=job
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if 'obs_mean' in ck:
        np.testing.assert_array_equal(ck['obs_mean'],C.MEAN)
        np.testing.assert_array_equal(np.sqrt(ck['obs_var']+1e-8),C.STD)
    actor=C.IL.actor_from_state(ck['actor']);actor.eval()
    args=C.IL.get_args(['--task_config',C.CONFIG]);args.seed=seed
    np.random.seed(seed);random.seed(seed)
    env=F1Wrapper(args,C.IL.MAPS,obs_noise=True,sim_mode='eval',tracks=[track])
    env.fixed_speed_cap=cap;env.start_mode='center';metrics=DrivingMetrics()
    try:
        obs,_=env.reset(options={'track':track})
        while True:
            with torch.no_grad():action=np.clip(actor(torch.as_tensor((obs-C.MEAN)/C.STD,dtype=torch.float32)).loc.numpy(),-1,1)
            obs,_,term,trunc,_=env.step(action);metrics.update(env,action)
            if term or trunc:break
        return dict(branch=branch,checkpoint=checkpoint,checkpoint_sha256=sha(checkpoint),track=track,track_sha256=sha(track),
                    seed=seed,cap=cap,complete=bool(env.success),collision=bool(env.collision),seconds=float(env.elapsed),
                    lap_times=[float(t) for t in env.lap_times],phases=metrics.result())
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--prepare_only',action='store_true')
    cli=parser.parse_args();os.chdir(SRC)
    os.environ['F1_LOC_LOST']='0'
    dest=Path(cli.destination)
    if dest.exists():raise FileExistsError('Preserve prior experiment')
    import numpy as np
    import torch
    import train_il_s2r as IL
    from utils.policy_transfer import columns
    from utils.training_mix import step_budgets,groups_for
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    source=Path('results/ppo_s2r/25/checkpoint/model_3080000.pt')
    config=Path('results/ppo_s2r/29/backup/f1tenth.yaml')
    collection_path=Path('results/reference_adaptation_pairs_1126/collection.json')
    audit_path=collection_path.with_name('audit.json')
    collection,audit=[json.loads(p.read_text()) for p in [collection_path,audit_path]]
    assert collection['status']=='complete_not_applied' and audit['status']=='passed_not_applied'
    assert audit['collection_sha256']==sha(collection_path) and audit['cases']==32
    assert collection['config_sha256']==sha(config) and collection['dynamics_sha256']==sha('configs/task/dynamic.yaml')
    assert collection['expert_config_sha256']==sha('configs/task/f1tenth_preview_1126.yaml')
    assert collection['planner_sha256']==sha('utils/rule_based.py')
    joint_path=Path('results/source25_recovery_adaptation_1126/pilot.json')
    joint=json.loads(joint_path.read_text());assert joint['status']=='complete_not_promoted' and joint['source_sha256']==sha(source)
    for entry in joint['input_collections'].values():
        p=Path(entry['path']);assert sha(p)==entry['sha256'] and sha(p.with_name('audit.json'))==entry['audit_sha256']
    args=IL.get_args(['--task_config',str(config)]);names=columns(args)
    rms=RunningMeanStd('agent_obs',132);rms.load('results/imitation/1607/sim2real',required=True)
    assert sha('results/imitation/1607/sim2real/agent_obs_scale.pkl')==joint['normalizer_sha256']
    groups=groups_for(args,[p for p in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in p])
    allowed={str(Path(p).resolve()) for g in groups.values() for p in g}
    manifest_path=Path('results/imitation/14/backup/track_manifest.json')
    manifest=json.loads(manifest_path.read_text());assert sha(manifest_path)==joint['manifest_sha256']
    for family,tracks in manifest['training'].items():
        assert {str(Path(p).resolve()) for p in tracks}=={str(Path(p).resolve()) for p in groups[family]}
    archive=Path('results/imitation/14/data/round000.npz');assert sha(archive)==joint['archive_sha256']
    # Reconstruct exactly the same32k source25 anchor observations as the recovery
    # pilot, including actual loss and adverse states, without fitting that pilot.
    rng=np.random.default_rng(112680)
    with np.load(archive,allow_pickle=False) as z:
        raw=z['O'];assert raw.shape==(1200000,132)
        chunks=[];offset=0
        for family,budget in step_budgets(manifest['step_shares'],len(raw)).items():
            chunks.append(raw[rng.choice(budget,joint['broad_family_quotas'][family],replace=False)+offset].astype(np.float32));offset+=budget
        broad=np.concatenate(chunks)
    missing=rng.choice(len(broad),len(broad)//4,replace=False)
    broad[np.ix_(missing,[i for i,n in enumerate(names) if n.startswith('preview/')])]=0
    banks=[broad]
    for key in ['stress','lost']:
        chunks=[]
        for row in joint['provenance'][key]:
            assert row['track'] in allowed and sha(row['shard'])==row['shard_sha256']
            with np.load(row['shard'],allow_pickle=False) as z:
                O=z['O'][row['indices']];assert len(O)==row['rows'] and list(z['columns'])==names
                if key=='lost':assert np.all(O[:,names.index('preview/valid')]==0)
                chunks.append(O)
        banks.append(np.concatenate(chunks))
    O=np.concatenate(banks);assert O.shape==(32000,132)
    AX=torch.as_tensor((O-rms.mean)/np.sqrt(rms.var+1e-8),dtype=torch.float32)
    ck=torch.load(source,map_location='cpu',weights_only=False)
    teacher=IL.actor_from_state(ck['actor']);teacher.eval()
    with torch.no_grad():AT=torch.cat([teacher(AX[i:i+2048]).loc for i in range(0,len(AX),2048)])
    demonstrations={}
    for branch in ['stored','recomputed']:
        rows=sorted((r for r in collection['runs'] if r['variant']==branch),key=lambda r:(Path(r['track']).name,r['seed'],r['cap']))
        assert len(rows)==16 and all(r['complete'] and r['admissible_for_future_review'] for r in rows)
        pieces=[]
        for row in rows:
            assert sha(row['track'])==row['track_sha256'] and sha(row['shard'])==row['shard_sha256']
            with np.load(row['shard'],allow_pickle=False) as z:
                assert list(z['columns'])==names and z['O'].shape==(row['rows'],132)
                X=torch.as_tensor((z['O']-rms.mean)/np.sqrt(rms.var+1e-8),dtype=torch.float32)
                Y=torch.as_tensor(z['Y'],dtype=torch.float32)
                assert torch.isfinite(X).all() and torch.isfinite(Y).all()
                pieces.append((X,Y))
        demonstrations[branch]=pieces
    identities={v:[(Path(r['track']).name,r['seed'],r['cap']) for r in sorted((x for x in collection['runs'] if x['variant']==v),key=lambda r:(Path(r['track']).name,r['seed'],r['cap']))] for v in demonstrations}
    assert identities['stored']==identities['recomputed']
    if cli.prepare_only:
        print(json.dumps(dict(status='prepared_no_fit',anchor_rows=len(AX),demonstrations_per_branch=16,source_sha256=sha(source))));return
    dest.mkdir(parents=True)
    report=dict(status='fitting',source_checkpoint=str(source),source_sha256=sha(source),config_sha256=sha(config),
                collection_sha256=sha(collection_path),audit_sha256=sha(audit_path),anchor_provenance_sha256=sha(joint_path),
                normalizer_sha256=joint['normalizer_sha256'],anchor_rows=len(AX),anchor_observation_sha256=__import__('hashlib').sha256(O.tobytes()).hexdigest(),
                steps=100,lr=1e-5,seed=112690,batch=dict(anchor=384,expert=128),
                loss='0.75*frozen_source_anchor_MSE+0.25*equal_episode_expert_MSE;steer/speedweights1/.25',branches={},driving=[],
                qualification='Samefrozen25u70actor,32kanchors,optimizer,batchbudgetsandseedforeachreferencebranch. Eachbranchusesitsownconsistentpreviewandspeed/steerexpertlabels. Knowntrainingexamplesonly; evaluationusesdifferentseeds7/17/27butisNOTunseengeometry. Referencecopiesareisolated;currentmapsandphysicsunchanged. ThisisILreferenceadaptation,notline-penaltyPPOablationoroptimalracelineproof.')
    def save():
        temp=dest/'pilot.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(dest/'pilot.json')
    weights=torch.tensor([1.,.25])
    for branch,data in demonstrations.items():
        folder=dest/branch;(folder/'checkpoint').mkdir(parents=True);(folder/'backup').mkdir()
        shutil.copyfile(config,folder/'backup/f1tenth.yaml')
        actor=IL.actor_from_state(ck['actor']);actor.train()
        optimizer=torch.optim.Adam(actor.parameters(),lr=1e-5)
        rng=np.random.default_rng(112690);torch.manual_seed(112690)
        entry=dict(checkpoints={},fit=[]);report['branches'][branch]=entry
        for step in range(1,101):
            ix=rng.integers(len(AX),size=384);ids=rng.integers(len(data),size=128)
            xp,yp=[],[]
            for i,(X,Y) in enumerate(data):
                indices=rng.integers(len(X),size=int((ids==i).sum()));xp.append(X[indices]);yp.append(Y[indices])
            loss=.75*((actor(AX[ix]).loc-AT[ix]).square()*weights).mean()+.25*((actor(torch.cat(xp)).loc-torch.cat(yp)).square()*weights).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
            if step in [50,100]:
                assert all(torch.isfinite(v).all() for v in actor.state_dict().values())
                path=folder/'checkpoint'/('model_%s.pt'%step);torch.save(dict(actor=actor.state_dict(),obs_mean=rms.mean,obs_var=rms.var),path)
                entry['checkpoints'][str(step)]=dict(path=str(path),sha256=sha(path))
                with torch.no_grad():error=torch.cat([(actor(AX[i:i+2048]).loc-AT[i:i+2048]).square() for i in range(0,len(AX),2048)]).mean(0).tolist()
                entry['fit'].append(dict(step=step,anchor_mse=error));save();print('FIT',branch,step,error,flush=True)
    report['status']='evaluating_known_courses';save()
    jobs=[]
    for branch,field in [('stored','stored'),('recomputed','recomputed')]:
        for pair in collection['pairs']:
            for seed in [7,17,27]:
                for cap in [4.,8.]:
                    jobs.append((branch,str(source),pair[field],seed,cap))
                    for tag in ['50','100']:
                        jobs.append((branch,report['branches'][branch]['checkpoints'][tag]['path'],pair[field],seed,cap))
    assert len(jobs)==144
    with mp.get_context('spawn').Pool(4,initializer=initialize) as pool:
        for result in pool.imap_unordered(evaluate,jobs):
            report['driving'].append(result);save()
            if len(report['driving'])%12==0 or not result['complete']:print('EVAL',len(report['driving']),'/144',result['branch'],result['complete'],flush=True)
    assert len(report['driving'])==144 and sha(source)==report['source_sha256']
    report['status']='complete_not_promoted';save()


if __name__=='__main__':main()
