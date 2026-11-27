#!/usr/bin/env python3
"""Collect matched expert demonstrations for stored and recomputed references."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'src/f1tenth'
sys.path[:0]=[str(SRC),str(SRC/'installation'),str(ROOT/'tools')]
from collect_expert_recovery_probe_1126 import sha
CONFIG='results/ppo_s2r/29/backup/f1tenth.yaml'


def drive(job):
    import numpy as np
    import random
    import torch
    import train_il_s2r as IL
    from ruamel.yaml import YAML
    from utils.env import F1Wrapper,normalize_speed
    from utils.policy_transfer import columns
    from utils.rule_based import RacelinePlanner
    torch.set_num_threads(1)
    variant,path,seed,cap,shard=job
    args=IL.get_args(['--task_config',CONFIG]);args.seed=seed
    np.random.seed(seed);random.seed(seed)
    env=F1Wrapper(args,IL.MAPS,obs_noise=True,sim_mode='eval',tracks=[path])
    env.expert_cfg=dict(YAML().load(Path('configs/task/f1tenth_preview_1126.yaml'))['expert_plan'])
    env.fixed_speed_cap=cap;env.start_mode='center'
    O,Y,W,V=[],[],[],[]
    try:
        obs,_=env.reset(options={'track':path})
        planner=RacelinePlanner(env._env.unwrapped.track,args.length_f+args.length_r,v_min=env.min_speed,
            v_max=env.speed_cap,max_steer=env.max_steer,max_steer_left=env.steer_left_max,
            max_steer_right=env.steer_right_max,**IL.expert_kwargs(env))
        while True:
            sq,vq=env.pending_commands()
            speed,wheel=planner.plan(env.vehicle_state(),sq,vq,env.timestep)
            action=np.asarray([env.steer_command(wheel),normalize_speed(speed,env.min_speed,env.max_speed)],dtype=np.float32)
            assert np.isfinite(action).all() and abs(action).max()<=1.000001
            O.append(obs.copy());Y.append(action.copy());W.append(wheel);V.append(speed)
            obs,_,term,trunc,_=env.step(action)
            if term or trunc:break
        raw=np.asarray(O,np.float32);target=np.asarray(Y,np.float32);names=columns(args)
        assert raw.shape==(len(raw),132) and target.shape==(len(raw),2)
        assert np.isfinite(raw).all() and np.isfinite(target).all()
        assert np.all(raw[:,names.index('speed_cap')]==cap)
        if env.success:assert not env.collision and len(env.lap_times)==2
        np.savez_compressed(shard,O=raw,Y=target,columns=np.asarray(names),teacher_wheel=np.asarray(W),teacher_speed=np.asarray(V))
        return dict(variant=variant,track=path,track_sha256=sha(path),seed=seed,cap=cap,complete=bool(env.success),
                    collision=bool(env.collision),seconds=float(env.elapsed),lap_times=[float(t) for t in env.lap_times],rows=len(raw),
                    preview_valid_rows=int((raw[:,names.index('preview/valid')]>.5).sum()),shard=shard,shard_sha256=sha(shard),
                    admissible_for_future_review=bool(env.success))
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',required=True)
    cli=parser.parse_args();os.chdir(SRC)
    os.environ['F1_LOC_LOST']='0'
    dest=Path(cli.destination)
    if dest.exists():raise FileExistsError('Preserve previous experiment')
    import numpy as np
    import train_il_s2r as IL
    from utils.training_mix import groups_for
    args=IL.get_args(['--task_config',CONFIG])
    groups=groups_for(args,[p for p in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in p])
    allowed={str(Path(p).resolve()) for group in groups.values() for p in group}
    generation_path=Path('results/reference_lines_1126/limits_v3_seeded/report.json')
    generation=json.loads(generation_path.read_text());assert generation['status']=='complete' and len(generation['tracks'])==4
    assert generation['limits']['current_dense_best']==dict(v_min=.5,v_max=8.,a_lat=6.,a_acc=2.9,a_brake=3.7)
    pairs=[];jobs=[]
    for i,item in enumerate(generation['tracks']):
        old=Path(item['source']);new=Path(item['variants']['current_dense_best']['path'])
        assert str(old.resolve()) in allowed and sha(old)==item['source_sha256']
        assert sha(new)==item['variants']['current_dense_best']['sha256']
        assert item['stored']['footprint']['poses_touching_occupied_samples']==0
        assert item['variants']['current_dense_best']['footprint']['poses_touching_occupied_samples']==0
        with np.load(old,allow_pickle=False) as a,np.load(new,allow_pickle=False) as b:
            for key in a.files:
                if key not in ['raceline','gen_sig','line_rmin']:np.testing.assert_array_equal(a[key],b[key])
        pairs.append(dict(stored=str(old),stored_sha256=sha(old),recomputed=str(new),recomputed_sha256=sha(new)))
        for repeat in range(2):
            seed=68000+2*i+repeat
            for cap in [4.,8.]:
                for variant,path in [('stored',old),('recomputed',new)]:
                    name=old.stem+'_'+str(seed)+'_cap'+str(int(cap))+'.npz'
                    jobs.append((variant,str(path),seed,cap,str((dest/variant/name).resolve())))
    assert len(jobs)==32
    for variant in ['stored','recomputed']:(dest/variant).mkdir(parents=True)
    report=dict(status='running',generation_sha256=sha(generation_path),pairs=pairs,
                config_sha256=sha(CONFIG),dynamics_sha256=sha('configs/task/dynamic.yaml'),
                expert_config_sha256=sha('configs/task/f1tenth_preview_1126.yaml'),planner_sha256=sha('utils/rule_based.py'),
                jobs=[dict(variant=v,track=p,seed=s,cap=c,shard=z) for v,p,s,c,z in jobs],runs=[],
                qualification='Pairedtraining-only knownPractice/2025examples,unchangedoccupancyandphysicaldynamics. Storedand6.0/2.9/3.7dense-recomputedreferences eachgenerateownpreview/speedprofile/privilegedexpertlabels. Sameperpairseeds,capsandstartmode. Demonstrationfeasibilityandstaticbodyclearancearenotmodelperformanceorhardwareproof. No automaticmapreplacement,heldoutgradientdataorPPOpromotion. Isolatedgeneratoromitsfullactuatordynamics;closedloopadaptationneeded.')
    def save():
        temp=dest/'collection.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(dest/'collection.json')
    save()
    with mp.get_context('spawn').Pool(4) as pool:
        for result in pool.imap_unordered(drive,jobs):
            report['runs'].append(result);save();print('CASE',result['variant'],Path(result['track']).name,result['seed'],result['cap'],result['complete'],flush=True)
    assert len(report['runs'])==32
    for pair in pairs:
        assert sha(pair['stored'])==pair['stored_sha256'] and sha(pair['recomputed'])==pair['recomputed_sha256']
    report['status']='complete_not_applied';save();print('DONE',flush=True)


if __name__=='__main__':main()
