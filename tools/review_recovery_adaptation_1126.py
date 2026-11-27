#!/usr/bin/env python3
"""Compare completed recovery-pilot evaluations to their exact source policy."""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'src/f1tenth'
sys.path.insert(0,str(ROOT/'tools'))
from review_ppo_retention_1126 import per_cap_comparison
from review_ppo_line_ablation_1126 import matched


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',choices=['100','400'],required=True)
    parser.add_argument('--output',required=True)
    cli=parser.parse_args()
    out=Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve earlier review')
    base=SRC/'results'
    paths={kind:base/('analysis1126/recovery_adaptation_'+cli.tag+'_'+kind+'_seed7.json') for kind in ['broad','known']}
    docs={k:json.loads(p.read_text()) for k,p in paths.items()}
    assert set(docs['broad']['families'])=={'regular','open','hall','short'}
    assert set(docs['known']['families'])=={'real','2025'}
    source_known_path=base/'analysis1126/recovery_adaptation_source_known_seed7.json'
    source_known=json.loads(source_known_path.read_text())
    assert set(source_known['families'])=={'real','2025'}
    pilot_path=base/'recovery_adaptation_1126/pilot.json'
    pilot=json.loads(pilot_path.read_text())
    assert pilot['status']=='complete_not_promoted'
    source_ck=base/'ppo_s2r/29/checkpoint/model_880000.pt'
    assert sha(source_ck)==pilot['source_sha256']==source_known['checkpoint_sha256']
    ck=SRC/pilot['checkpoints'][cli.tag]['path']
    assert sha(ck)==pilot['checkpoints'][cli.tag]['sha256']
    assert all(doc['checkpoint_sha256']==sha(ck) for doc in docs.values())
    assert sha(base/'ppo_s2r/29/backup/f1tenth.yaml')==pilot['config_sha256']
    assert source_known['dynamics_sha256']==sha(SRC/'configs/task/dynamic.yaml')
    # The live learner rewrites this history non-atomically; a short partial
    # read is not a missing checkpoint or a reason to restart training.
    for attempt in range(10):
        try:
            with (base/'ppo_s2r/29/train_log.pkl').open('rb') as f:
                history=pickle.load(f)
            break
        except (EOFError,pickle.UnpicklingError):
            if attempt==9:
                raise
            time.sleep(.1)
    candidates=[r for r in history if r.get('update')==20 and 'eval_success' in r]
    assert len(candidates)==1
    source=candidates[0]
    assert source['evaluation_seed_base']==7
    for doc in list(docs.values())+[source_known]:
        assert doc['seeds']==[7] and doc['simulation_mode']=='eval' and not doc['localization_forced_lost']
        assert doc['diagnostic_caps'] is None and doc['diagnostic_cycle_caps'] is None
        assert doc['config_sha256']==pilot['config_sha256']
        assert doc['normalizer_arrays_sha256']==source_known['normalizer_arrays_sha256']
        assert doc['dynamics_sha256']==source_known['dynamics_sha256']
    rows={}
    for group,key,total in [('regular','res',132),('open','open_res',88),('hall','hall_res',102),('short','short_res',32)]:
        data=docs['broad']['families'][group]
        assert data['total']==len(data['runs'])==total
        order=source['evaluation_track_order'][key]
        reference={r[0]:r for r in source[key]}
        current=[];old=[]
        for entry in data['runs']:
            r=entry['result'];assert entry['seed']==7+order.index(r[0].split('@')[0])
            assert r[0] in reference
            current.append(r);old.append(reference[r[0]])
        assert len({r[0] for r in current})==total
        rows[group]=dict(candidate_complete=sum(bool(r[1]) for r in current),source_complete=sum(bool(r[1]) for r in old),
            total=total,comparison=matched(current,old),per_cap=per_cap_comparison(current,old))
    for group in ['real','2025']:
        data=docs['known']['families'][group]['runs'];reference=source_known['families'][group]['runs']
        assert [(r['seed'],r['result'][0]) for r in data]==[(r['seed'],r['result'][0]) for r in reference]
        assert len(data)==4
        current=[r['result'] for r in data];old=[r['result'] for r in reference]
        rows[group]=dict(candidate_complete=sum(bool(r[1]) for r in current),source_complete=sum(bool(r[1]) for r in old),
            total=4,comparison=matched(current,old),per_cap=per_cap_comparison(current,old))
    regression=[g for g,r in rows.items() if r['candidate_complete']<r['source_complete']]
    report=dict(status='complete_read_only_not_promoted',tag=cli.tag,checkpoint_sha256=sha(ck),source_sha256=sha(source_ck),
        report_sha256={k:sha(p) for k,p in paths.items()},source_known_sha256=sha(source_known_path),cohorts=rows,
        completion_regressions=regression,requires=['44 additional lowcap cases','132 worst-dynamics cases','missing-preview short/general cases','expanded geometry/start/seed checks before global or real-car claims'],
        qualification='Exact source29u20 intervention comparison on ordered nominal cases and per-track seed schedule. Regular132 excludes the44 additional low-cap checks; those remain outstanding. Known-course source was rerun because the separate real/2025 evaluator resets seed indexing per family. Pace only includes paired completions; training-case6/6 and lower supervised error do not establish generalization or hardware readiness. No automatic model promotion.')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({g:{k:r[k] for k in ['candidate_complete','source_complete','total']} for g,r in rows.items()},indent=2))


if __name__=='__main__':
    main()
