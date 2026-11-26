#!/usr/bin/env python3
"""Cap each new rehearsal family; keep broad coverage without long-map dominance."""
import argparse,hashlib,json,os,sys,tempfile,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'src/f1tenth'
sys.path[:0]=[str(S),str(S/'installation')]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true')
    p.add_argument('--rows_per_family',type=int,default=240000);cli=p.parse_args();os.chdir(S)
    roots={g:Path(f'results/{g}_replay_1126') for g in ('regular','open')}
    deadline=time.monotonic()+3600;last=0
    while True:
        reports={}
        for g,root in roots.items():
            try:reports[g]=json.loads((root/'collection.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):reports[g]={}
        if all(r.get('status')=='complete' for r in reports.values()):break
        if not cli.wait:raise RuntimeError('Both collectors must complete first')
        if time.monotonic()>deadline:raise TimeoutError('Collectors incomplete after one hour; no learner restarted')
        if time.monotonic()-last>60:
            print('WAIT', {g:len(r.get('runs',[])) for g,r in reports.items()},flush=True);last=time.monotonic()
        time.sleep(10)
    import numpy as np,torch
    import train_il_s2r as IL
    from utils.training_mix import family
    from utils.vectorize import RunningMeanStd
    from utils.replay import load_replay
    base=Path('results/retention_probe_1126/replay_cache.npz');base_hash=sha(base)
    with np.load(base) as z:cache={key:z[key] for key in z.files}
    base_rows=len(cache['O']);Os=[cache['O']];Ys=[cache['Y']];provenance={};rng=np.random.default_rng(1706)
    source_hashes={r['source_sha256'] for r in reports.values()};assert len(source_hashes)==1
    for g,report in reports.items():
        assert report['base_cache_sha256']==base_hash
        assert report['dynamics_sha256']==sha('configs/task/dynamic.yaml')
        assert all(family(track)==g for track in report['tracks'])
        successful=sorted((r for r in report['runs'] if r['complete']),key=lambda r:r['track'])
        assert successful
        quota=max(1,cli.rows_per_family//len(successful));selected=0
        for row in successful:
            with np.load(roots[g]/'shards'/(Path(row['track']).stem+'.npz')) as z:
                assert len(z['O'])==row['steps']
                indices=rng.choice(len(z['O']),min(quota,len(z['O'])),replace=False)
                Os.append(z['O'][indices]);Ys.append(z['Y'][indices]);selected+=len(indices)
        provenance[g]=dict(attempted=len(report['runs']),successful_tracks=len(successful),selected_rows=selected,
                           rows_per_successful_track_max=quota,source_report_sha256=sha(roots[g]/'collection.json'))
    cache['O']=np.concatenate(Os);cache['Y']=np.concatenate(Ys)
    cache['missing_new_features']=np.r_[cache['missing_new_features'],np.zeros(len(cache['O'])-base_rows,bool)]
    del Os,Ys
    assert np.isfinite(cache['O']).all() and np.isfinite(cache['Y']).all()
    assert abs(cache['Y']).max()<=1.000001
    names=list(cache['columns']);invalid=cache['O'][:,names.index('preview/valid')]<.5
    preview=[i for i,name in enumerate(names) if name.startswith('preview/')]
    assert np.all(cache['O'][invalid][:,preview]==0),'Invalid localization must have raw-zero preview'
    root=Path('results/balanced_replay_1126');root.mkdir(exist_ok=True);dest=root/'replay_cache.npz'
    if dest.exists():raise RuntimeError('Balanced cache already exists; inspect its report instead of overwriting')
    with tempfile.NamedTemporaryFile(dir=root,prefix='replay_',suffix='.npz',delete=False) as f:
        np.savez(f,**cache);temporary=f.name
    os.replace(temporary,dest)
    source=Path('results/imitation/14/checkpoint/model_init.pt');assert sha(source) in source_hashes
    ck=torch.load(source,map_location='cpu',weights_only=False)
    args=IL.get_args(['--task_config','results/imitation/14/backup/f1tenth.yaml','--batch','2048'])
    args.replay_file=str(dest);args.replay_share=.25
    rms=RunningMeanStd('agent_obs',132);rms.mean=ck['obs_mean'];rms.var=ck['obs_var']
    # Exercise the production loader, including historical missing-feature semantics.
    replay=load_replay(args,rms);actor=IL.actor_from_state(ck['actor']);actor.eval();torch.set_num_threads(1)
    indices=rng.choice(np.arange(base_rows,len(cache['O'])),min(2048,len(cache['O'])-base_rows),replace=False)
    with torch.no_grad():pred=actor(replay.X[indices].float()).loc.numpy()
    error=np.abs(pred-cache['Y'][indices]);p99=float(np.quantile(error,.99));maximum=float(error.max())
    assert p99<.01 and maximum<.05,(p99,maximum)
    out=dict(status='complete',base_rows=base_rows,total_rows=len(cache['O']),families=provenance,
             cache_sha256=sha(dest),base_cache_sha256=base_hash,source_sha256=sha(source),
             invalid_preview_rows=int(invalid.sum()),production_loader='passed, replay_share .25',
             quantized_frozen_label_error=dict(p99=p99,maximum=maximum),
             qualification='Training-only rehearsal candidate; no new-model performance claim or automatic PPO start')
    (root/'build.json').write_text(json.dumps(out,indent=2)+'\n');print('DONE',json.dumps(out),flush=True)


if __name__=='__main__':main()
