#!/usr/bin/env python3
"""Prepare balanced successful old-policy rehearsal; does not start a learner."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'src/f1tenth'
sys.path[:0]=[str(SRC),str(SRC/'installation')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.chdir(SRC)
    import numpy as np
    import torch
    import train_il_s2r as IL
    from eval_open_space import TRAIN_DIR,TEST_DIR
    from utils.policy_transfer import columns
    from utils.vectorize import RunningMeanStd
    from utils.replay import load_replay
    torch.set_num_threads(1)
    args=IL.get_args(['--task_config','results/imitation/1607/backup/f1tenth.yaml','--batch','2048'])
    output=Path('results/ppo_retention_candidate_1126')
    if output.exists():raise FileExistsError('Inspect existing candidate rather than overwriting it')
    source=Path('results/imitation/14/checkpoint/model_init.pt');source_hash=sha(source)
    names=columns(args)
    with np.load('results/retention_probe_1126/replay_cache.npz') as z:
        old_names=z['old_columns'].copy();teacher=z['teacher'].copy()
    original_names={m+d+'.npz' for m in IL.MAPS for d in ('','_rev')}
    rng=np.random.default_rng(112650);obs=[];labels=[];groups=[];track_ids=[];provenance=[]
    for group in ('regular','open'):
        root=Path(f'results/{group}_replay_1126');meta=json.loads((root/'collection.json').read_text())
        assert meta['status']=='complete' and meta['source_sha256']==source_hash
        track_paths={Path(p).name:Path(p).resolve() for p in meta['tracks']}
        assert len(track_paths)==len(meta['tracks']), 'Ambiguous source track basename'
        successful=sorted((r for r in meta['runs'] if r['complete']),key=lambda r:r['track'])
        assert successful
        quota=max(1,32768//len(successful))
        for row in successful:
            track=track_paths[row['track']]
            if group=='regular':
                assert track.name in original_names and IL.HOLDOUT not in str(track)
                assert track.parent==Path('installation/maps_gen').resolve()
            else:
                assert track.parent==Path(TRAIN_DIR).resolve()
                assert Path(TEST_DIR).resolve() not in track.parents
            shard=root/'shards'/(track.stem+'.npz')
            with np.load(shard) as z:
                assert len(z['O'])==row['steps']
                ix=rng.choice(len(z['O']),min(quota,len(z['O'])),replace=False)
                obs.append(z['O'][ix]);labels.append(z['Y'][ix])
            groups.extend([group]*len(ix));track_ids.extend([len(provenance)]*len(ix))
            provenance.append(dict(track=str(track),family=group,rows=len(ix),shard_sha256=sha(shard),
                                   source_episode_complete=True))
    O,Y=np.concatenate(obs),np.concatenate(labels)
    assert O.shape==(len(Y),args.obs_dim) and Y.shape[1]==2
    assert np.isfinite(O).all() and np.isfinite(Y).all() and np.abs(Y).max()<=1.00001
    invalid=O[:,names.index('preview/valid')]<.5
    assert np.all(O[invalid][:,[i for i,n in enumerate(names) if n.startswith('preview/')]]==0)
    output.mkdir(parents=True);dest=output/'replay_cache.npz'
    np.savez(dest,O=O,Y=Y,columns=np.array(names),old_columns=old_names,teacher=teacher,
             missing_new_features=np.zeros(len(O),bool),family=np.array(groups),track_id=np.array(track_ids,dtype=np.int32))
    rms=RunningMeanStd('agent_obs',args.obs_dim);rms.load('results/imitation/1607/sim2real',required=True)
    args.replay_file=str(dest);args.replay_share=.125
    replay=load_replay(args,rms)
    ck=torch.load(source,map_location='cpu',weights_only=False);actor=IL.actor_from_state(ck['actor']);actor.eval()
    indices=rng.choice(len(O),min(4096,len(O)),replace=False)
    normalized=rms.normalize(O[indices].astype(np.float32))
    expected=normalized.astype(np.float16)
    actual=replay.X[indices].numpy()
    # NumPy and torch can round halfway float64->float16 values one ULP apart.
    # Require exact agreement with torch conversion and at most one NumPy ULP.
    np.testing.assert_array_equal(actual,torch.as_tensor(normalized,dtype=torch.float16).numpy())
    assert np.all(actual>=np.nextafter(expected,np.float16(-np.inf)))
    assert np.all(actual<=np.nextafter(expected,np.float16(np.inf)))
    own=((O[indices].astype(np.float32)-ck['obs_mean'])/np.sqrt(ck['obs_var']+1e-8)).astype(np.float32)
    with torch.no_grad():pred=actor(torch.as_tensor(own)).loc.clamp(-1,1).numpy()
    p99=float(np.quantile(np.abs(pred-Y[indices]),.99));assert p99<.01
    report=dict(status='prepared_not_applied',rows=len(O),families={g:groups.count(g) for g in set(groups)},
                tracks=provenance,source_checkpoint_sha256=source_hash,cache_sha256=sha(dest),
                source_action_error_p99=p99,invalid_preview_rows=int(invalid.sum()),
                numpy_torch_float16_rounding_differences=int(np.count_nonzero(actual!=expected)),
                normalization_max_numpy_difference=float(np.max(np.abs(actual.astype(float)-expected.astype(float)))),
                validation='production replay loader exact normalized inputs and reconstructed source labels',
                loader_smoke_share=.125,live_training_modified=False,
                collection_caps=[8.0],
                purpose='Candidate auxiliary old-policy steering rehearsal during a later controlled PPO ablation.',
                restrictions=['Training geometries only; successfulsourceepisodes only.',
                              'Does not include legacyhall/short expert targets, which may contradict newhairpin behavior.',
                              'No learner started, no PPO hyperparameters selected, no performance claim.'])
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tracks'}),flush=True)


if __name__=='__main__':main()
