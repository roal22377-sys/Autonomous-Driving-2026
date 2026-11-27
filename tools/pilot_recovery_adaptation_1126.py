#!/usr/bin/env python3
"""Bounded offline recovery adaptation with frozen-function rehearsal.

Creates isolated100/400-step candidates from29u20, never mutates live PPO.
Training-case recovery tests are diagnostics, not model qualification.
"""
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC/'installation'), str(ROOT/'tools')]
DEST = Path('results/recovery_adaptation_1126')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_init():
    import collect_expert_recovery_probe_1126 as collector
    collector.initialize('results/expert_recovery_failures_1126', True)


def evaluate_case(job):
    import collect_expert_recovery_probe_1126 as C
    row, tag = job
    np, torch = C.np, C.torch
    ck = torch.load(DEST/'checkpoint'/('model_%s.pt'%tag),map_location='cpu',weights_only=False)
    actor = C.IL.actor_from_state(ck['actor']); actor.eval()
    np.testing.assert_array_equal(ck['obs_mean'], C.MEAN)
    np.testing.assert_array_equal(np.sqrt(ck['obs_var']+1e-8), C.STD)
    assert sha(row['track']) == row['track_sha256']
    _, env, obs = C.environment(row['track'],row['seed'])
    try:
        while True:
            with torch.no_grad():
                action = np.clip(actor(torch.as_tensor((obs-C.MEAN)/C.STD,dtype=torch.float32)).loc.numpy(),-1,1)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        result = dict(tag=tag, track=Path(row['track']).name, seed=row['seed'],
            handoff_at_reset_in_data=row['handoff_at_reset'], success=bool(env.success),
            collision=bool(env.collision), seconds=float(env.elapsed), lap_times=[float(t) for t in env.lap_times])
        if tag == 'source':
            assert result['success'] == row['baseline']['success']
            assert result['collision'] == row['baseline']['collision']
            assert abs(result['seconds']-row['baseline']['seconds']) < 1e-8
        return result
    finally:
        env.close()


def main():
    os.chdir(SRC)
    import numpy as np
    import torch
    import train_il_s2r as IL
    from utils.policy_transfer import columns
    from utils.training_mix import groups_for, step_budgets
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    if DEST.exists():
        raise FileExistsError('Preserve earlier pilot; inspect before rerunning')
    config = Path('results/ppo_s2r/29/backup/f1tenth.yaml')
    source = Path('results/ppo_s2r/29/checkpoint/model_880000.pt')
    collection_path = Path('results/expert_recovery_failures_1126/collection.json')
    audit_path = collection_path.with_name('audit_classified.json')
    collection, audit = [json.loads(p.read_text()) for p in (collection_path,audit_path)]
    assert collection['status'] == 'complete_not_applied' and audit['status'] == 'passed_not_applied'
    assert audit['collection_sha256'] == sha(collection_path)
    assert collection['student_sha256'] == sha(source) and collection['config_sha256'] == sha(config)
    assert collection['dynamics_sha256'] == sha('configs/task/dynamic.yaml')
    args = IL.get_args(['--task_config',str(config)])
    rms = RunningMeanStd('agent_obs',132); rms.load('results/imitation/1607/sim2real',required=True)
    assert sha('results/imitation/1607/sim2real/agent_obs_scale.pkl') == collection['normalizer_sha256']
    groups = groups_for(args,[p for p in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in p])
    manifest_path = Path('results/imitation/14/backup/track_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    for family, tracks in manifest['training'].items():
        assert {str(Path(p).resolve()) for p in tracks} == {str(Path(p).resolve()) for p in groups[family]}
    original_args = IL.get_args(['--task_config','results/imitation/14/backup/f1tenth.yaml'])
    assert columns(original_args) == columns(args)
    args = IL.get_args(['--task_config',str(config)])
    ck = torch.load(source,map_location='cpu',weights_only=False)
    teacher = IL.actor_from_state(ck['actor']); teacher.eval()
    actor = IL.actor_from_state(ck['actor']); actor.train()
    rng = np.random.default_rng(112671)
    archive = Path('results/imitation/14/data/round000.npz')
    with np.load(archive,allow_pickle=False) as z:
        raw = z['O']; assert raw.shape == (1200000,132)
        budgets = step_budgets(manifest['step_shares'],len(raw))
        target = step_budgets(dict(args.training_mix),32000)
        chunks, anchor_families = [], []
        offset = 0
        for family, budget in budgets.items():
            selected = rng.choice(budget,target[family],replace=False)+offset
            chunks.append(raw[selected].astype(np.float32))
            anchor_families.extend([family]*len(selected)); offset += budget
        O = np.concatenate(chunks)
    names = columns(args)
    # Preserve the source's response to absent preview as well as valid preview.
    # Synthetic dropout inputs are anchors, never claimed successful rollout labels.
    missing = rng.choice(len(O),len(O)//4,replace=False)
    O[np.ix_(missing,[i for i,n in enumerate(names) if n.startswith('preview/')])] = 0
    X = torch.as_tensor((O-rms.mean)/np.sqrt(rms.var+1e-8),dtype=torch.float32)
    with torch.no_grad():
        T = torch.cat([teacher(X[i:i+2048]).loc for i in range(0,len(X),2048)])
    assert torch.isfinite(X).all() and torch.isfinite(T).all()
    recovery = []
    rows = sorted((r for r in collection['runs'] if r.get('outcome')=='rescued'),key=lambda r:r['track'])
    assert len(rows)==6 and sum(r['handoff_at_reset'] for r in rows)==1
    for r in rows:
        assert sha(r['shard'])==r['shard_sha256']
        with np.load(r['shard'],allow_pickle=False) as z:
            assert list(z['columns'])==names
            rx=torch.as_tensor((z['O']-rms.mean)/np.sqrt(rms.var+1e-8),dtype=torch.float32)
            ry=torch.as_tensor(z['Y'],dtype=torch.float32)
            early=z['seconds_since_handoff']<=2
            assert early.any() and (~early).any()
            recovery.append((rx,ry,np.flatnonzero(early),np.flatnonzero(~early)))
    (DEST/'checkpoint').mkdir(parents=True)
    (DEST/'backup').mkdir()
    shutil.copyfile(config,DEST/'backup/f1tenth.yaml')
    report=dict(status='training',source_checkpoint=str(source),source_sha256=sha(source),
        config_sha256=sha(config),normalizer_sha256=collection['normalizer_sha256'],
        collection_sha256=sha(collection_path),audit_sha256=sha(audit_path),
        anchor_archive=str(archive),anchor_archive_sha256=sha(archive),
        anchor_manifest_sha256=sha(manifest_path),anchor_rows=len(X),anchor_family_counts=target,
        synthetic_missing_preview_rows=len(missing),recovery_cases=6,post_prefix_cases=5,startup_cases=1,
        seed=112671,steps=400,lr=1e-5,batch=512,anchor_batch=384,recovery_batch=128,
        loss='0.75*frozen_source_MSE + 0.25*expert_recovery_MSE; steering/speed weights1/.25',
        recovery_sampling='Equal case probability; half rows from first2seconds and half from remainder; joint actions',
        checkpoints={},fit=[],driving=[],
        qualification='Isolated offline adaptation of29u20, not PPO resume or model promotion. Broad anchors are from the archived training-only IL14 initial collection in recorded family order, labelled by frozen29u20 (not expert actions);25% receive synthetic absent preview. Six recovery cases span five previously trained base geometries; no random-row unseen-geometry claim. All recovery data is in-sample and requires closed-loop and independent cohort evaluation. No live learner, cache, observation normalization or vehicle changes.')
    def save():
        temp=DEST/'pilot.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(DEST/'pilot.json')
    def checkpoint(tag):
        path=DEST/'checkpoint'/('model_%s.pt'%tag)
        torch.save(dict(actor=actor.state_dict(),obs_mean=rms.mean,obs_var=rms.var),path)
        report['checkpoints'][tag]=dict(path=str(path),sha256=sha(path))
    checkpoint('source');save()
    optimizer=torch.optim.Adam(actor.parameters(),lr=1e-5)
    weights=torch.tensor([1.,.25])
    def fit_metrics(step):
        actor.eval()
        with torch.no_grad():
            anchor_error=torch.cat([(actor(X[i:i+2048]).loc-T[i:i+2048]).square() for i in range(0,len(X),2048)])
            values=[]
            for rx,ry,early,late in recovery:
                error=torch.cat([(actor(rx[i:i+2048]).loc-ry[i:i+2048]).square() for i in range(0,len(rx),2048)])
                values.append(dict(early_mse=error[early].mean(0).tolist(),late_mse=error[late].mean(0).tolist()))
        actor.train()
        return dict(step=step,anchor_mse=anchor_error.mean(0).tolist(),recovery=values)
    report['fit'].append(fit_metrics(0))
    assert max(report['fit'][0]['anchor_mse'])==0
    torch.manual_seed(112671);random.seed(112671)
    for step in range(1,401):
        ai=rng.integers(len(X),size=384)
        rx_parts,ry_parts=[],[]
        for early_part in (True,False):
            case_ids=rng.integers(len(recovery),size=64)
            for index,(rx,ry,early,late) in enumerate(recovery):
                count=int((case_ids==index).sum())
                indices=rng.choice(early if early_part else late,count,replace=True)
                rx_parts.append(rx[indices]);ry_parts.append(ry[indices])
        rx=torch.cat(rx_parts);ry=torch.cat(ry_parts)
        anchor_loss=((actor(X[ai]).loc-T[ai]).square()*weights).mean()
        recovery_loss=((actor(rx).loc-ry).square()*weights).mean()
        loss=.75*anchor_loss+.25*recovery_loss
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
        if step in (100,400):
            assert all(torch.isfinite(v).all() for v in actor.state_dict().values())
            checkpoint(str(step));report['fit'].append(fit_metrics(step));save()
            print('FIT',step,report['fit'][-1]['anchor_mse'],flush=True)
    report['status']='evaluating_training_cases';save()
    with mp.get_context('spawn').Pool(2,initializer=worker_init) as pool:
        for result in pool.imap_unordered(evaluate_case,[(r,tag) for tag in ('source','100','400') for r in rows]):
            report['driving'].append(result);save();print('CASE',result,flush=True)
    assert len(report['driving'])==18
    assert sha(source)==report['source_sha256']
    report['status']='complete_not_promoted';save()


if __name__=='__main__':
    main()
