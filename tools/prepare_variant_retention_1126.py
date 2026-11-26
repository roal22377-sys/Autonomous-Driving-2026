#!/usr/bin/env python3
"""Prepare a same-size replay-coverage candidate; never alter a running cache."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC/'installation')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.chdir(SRC)
    import numpy as np
    import torch
    import train_il_s2r as IL
    from utils.training_mix import groups_for
    from utils.vectorize import RunningMeanStd
    from utils.ppo_retention import RetentionCache
    torch.set_num_threads(1)
    original = Path('results/ppo_retention_candidate_1126')
    collection = Path('results/regular_variants_replay_1126')
    output = Path('results/ppo_variant_retention_candidate_1126')
    if output.exists():
        raise FileExistsError('Preserve previous candidate')
    meta = json.loads((collection/'collection.json').read_text())
    source = Path('results/imitation/14/checkpoint/model_init.pt')
    manifest = json.loads((original/'preparation.json').read_text())
    assert meta.get('status')=='complete' and meta['source_sha256']==sha(source)==manifest['source_checkpoint_sha256']
    assert meta['collection_caps']==[8.] and len(meta['tracks'])==len(meta['runs'])==92
    assert meta['dynamics_sha256']==sha('configs/task/dynamic.yaml')
    assert meta['config_sha256']==sha('results/imitation/14/backup/f1tenth.yaml')
    before = sha(original/'replay_cache.npz')
    assert before==manifest['cache_sha256']
    with np.load(original/'replay_cache.npz',allow_pickle=False) as z:
        old = {k:z[k].copy() for k in z.files}
    task_config = Path('results/ppo_s2r/29/backup/f1tenth.yaml')
    args = IL.get_args(['--task_config',str(task_config)])
    regular = [p for p in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in p]
    training = groups_for(args,regular)
    allowed = {str(Path(p).resolve()) for paths in training.values() for p in paths}
    targets = {Path(p).name:Path(p).resolve() for p in meta['tracks']}
    expected = {m+d+'_w'+str(seed)+'.npz' for m in IL.MAPS for d in ('','_rev') for seed in (101,103)}
    assert set(targets)==expected and len({r['track'] for r in meta['runs']})==92
    assert all(str(p) in allowed and IL.HOLDOUT not in str(p) for p in targets.values())
    successes = sorted((r for r in meta['runs'] if r['complete']),key=lambda r:r['track'])
    assert successes
    n_regular = int((old['family']=='regular').sum())
    assert n_regular==32752 and n_regular%2==0
    budget = n_regular//2
    rng = np.random.default_rng(112651)
    observations, labels, families, ids, provenance = [], [], [], [], []

    def append(O,Y,entry):
        index = len(provenance)
        observations.append(O);labels.append(Y)
        families.extend([entry['family']]*len(O));ids.extend([index]*len(O))
        provenance.append(dict(entry,rows=len(O),track_sha256=sha(entry['track'])))

    for i,entry in enumerate(manifest['tracks']):
        if entry['family']!='regular':
            continue
        indices = np.flatnonzero(old['track_id']==i)
        assert len(indices)==712
        chosen = rng.choice(indices,len(indices)//2,replace=False)
        append(old['O'][chosen],old['Y'][chosen],dict(entry,origin='retained_original_subset'))
    assert sum(len(x) for x in observations)==budget
    quotient,remainder = divmod(budget,len(successes))
    for i,row in enumerate(successes):
        track = targets[row['track']]
        shard = collection/'shards'/(track.stem+'.npz')
        with np.load(shard,allow_pickle=False) as z:
            O,Y = z['O'],z['Y']
            assert len(O)==row['steps'] and O.shape==(len(O),132) and Y.shape==(len(O),2)
            n = quotient+int(i<remainder)
            assert 0<n<=len(O)
            indices = rng.choice(len(O),n,replace=False)
            append(O[indices],Y[indices],dict(track=str(track),family='regular',shard_sha256=sha(shard),
                source_episode_complete=True,source_seed=row['seed'],origin='training_variant'))
    assert sum(len(x) for x in observations)==n_regular
    for i,entry in enumerate(manifest['tracks']):
        if entry['family']=='open':
            indices = np.flatnonzero(old['track_id']==i)
            append(old['O'][indices],old['Y'][indices],dict(entry,origin='unchanged_open_rows'))
    O,Y = np.concatenate(observations),np.concatenate(labels)
    family = np.array(families)
    assert O.shape==old['O'].shape and Y.shape==old['Y'].shape
    assert np.array_equal(O[family=='open'],old['O'][old['family']=='open'])
    assert np.array_equal(Y[family=='open'],old['Y'][old['family']=='open'])
    assert np.isfinite(O).all() and np.isfinite(Y).all() and np.abs(Y).max()<=1.00001
    names = list(old['columns'])
    assert np.all(O[:,names.index('speed_cap')]==8) and np.all(O[:,names.index('preview/valid')]>0.5)
    ck = torch.load(source,map_location='cpu',weights_only=False)
    actor = IL.actor_from_state(ck['actor']);actor.eval()
    audit = {}
    origins = np.array([provenance[i]['origin'] for i in ids])
    for origin in ('retained_original_subset','training_variant','unchanged_open_rows'):
        pool = np.flatnonzero(origins==origin)
        selected = rng.choice(pool,min(4096,len(pool)),replace=False)
        normalized = ((O[selected].astype(np.float32)-ck['obs_mean'])/np.sqrt(ck['obs_var']+1e-8)).astype(np.float32)
        with torch.no_grad():prediction = actor(torch.as_tensor(normalized)).loc.clamp(-1,1).numpy()
        error = np.abs(prediction-Y[selected])
        audit[origin] = dict(samples=len(selected),source_action_error_p99=float(np.quantile(error,.99)),maximum=float(error.max()))
        assert audit[origin]['source_action_error_p99']<.01,(origin,audit[origin])
    output.mkdir()
    cache = output/'replay_cache.npz'
    np.savez(cache,O=O,Y=Y,columns=old['columns'],old_columns=old['old_columns'],teacher=old['teacher'],
        missing_new_features=np.zeros(len(O),bool),family=family,track_id=np.asarray(ids,dtype=np.int32))
    report = dict(status='prepared_not_applied',rows=len(O),families={g:int((family==g).sum()) for g in ('regular','open')},
        tracks=provenance,source_checkpoint_sha256=sha(source),cache_sha256=sha(cache),
        control_cache_sha256=before,collection_report_sha256=sha(collection/'collection.json'),
        collection_caps=[8.],invalid_preview_rows=0,action_label_audit=audit,
        original_regular_rows=budget,variant_regular_rows=budget,open_rows_unchanged=True,
        successful_variant_tracks=len(successes),failed_variant_tracks=92-len(successes),
        purpose='Separate future same-size replay geometry-coverage experiment; no live cache replacement.',
        qualification='Same65432 total rows and32752/32680 regular/open split as control. Original regular rows halved within every base/direction, replaced by successful predeclared training augmentations. Known geometry augmentation does not establish novel-layout generalization. No held-out105 data, no missing-preview or lower-cap targets, no PPO setting chosen.')
    target = output/'preparation.json';target.write_text(json.dumps(report,indent=2)+'\n')
    rms = RunningMeanStd('agent_obs',args.obs_dim);rms.load('results/imitation/1607/sim2real',required=True)
    cache_object = RetentionCache(args,rms,cache,target,[p for paths in training.values() for p in paths],
                                  minibatch=2048,share=.125,seed=1127)
    indices = rng.choice(len(O),4096,replace=False)
    expected = torch.as_tensor(rms.normalize(O[indices].astype(np.float32)),dtype=torch.float16).numpy()
    np.testing.assert_array_equal(cache_object.data.X[indices].numpy(),expected)
    with torch.no_grad():loss,components = cache_object.loss(actor)
    assert torch.isfinite(loss).item()
    assert sha(original/'replay_cache.npz')==before
    (output/'loader_audit.json').write_text(json.dumps(dict(status='passed',metadata=cache_object.metadata,
        loader_task_config=str(task_config),loader_task_config_sha256=sha(task_config),
        normalized_loader_and_sampler_passed=True,normalization_samples_checked=4096,current_cache_unchanged=True,
        qualification='Production-loader structural check only; sampled loss is not a performance score.'),indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tracks'}),flush=True)


if __name__=='__main__':
    main()
