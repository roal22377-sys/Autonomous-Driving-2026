#!/usr/bin/env python3
"""Mine exact source25 prefixes on its failed cases from the training corpus."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/f1tenth'
sys.path[:0]=[str(SRC),str(SRC/'installation'),str(ROOT/'tools')]
from collect_expert_recovery_probe_1126 import sha
SOURCE='results/ppo_s2r/25/checkpoint/model_3080000.pt'


def initialize(destination):
    import collect_expert_recovery_probe_1126 as C
    C.initialize(destination,True,SOURCE)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',required=True)
    cli=parser.parse_args();os.chdir(SRC)
    dest=Path(cli.destination)
    if dest.exists():raise FileExistsError('Preserve prior collection')
    selection=Path('results/analysis1126/recovery_corpus_starting_policy_review.json')
    corpus_path=Path('results/recovery_corpus_round2_1126/manifest.json')
    sel=json.loads(selection.read_text());corpus=json.loads(corpus_path.read_text())
    assert sel['status']=='complete_read_only_not_selected' and sel['corpus_sha256']==sha(corpus_path)
    assert sel['policies']['source25u70']['sha256']==sha(SOURCE)
    controls=[r for r in sel['runs'] if r['policy']=='source25u70']
    assert len(controls)==13 and len({r['case_id'] for r in controls})==13
    cases={r['case_id']:r for r in corpus['cases']}
    failures=[r for r in controls if not r['complete']]
    assert failures
    allowed=sorted(str(p.resolve()) for p in Path('installation/maps_open_train').glob('*.npz'))
    jobs=[]
    for row in sorted(failures,key=lambda r:r['track']):
        case=cases[row['case_id']]
        assert (case['track'],case['seed'])==(row['track'],row['seed'])
        assert row['track'] in allowed and row['seed']==64000+allowed.index(row['track'])
        jobs.append(('open',row['track'],row['seed']))
    initialize(str(dest))
    import collect_expert_recovery_probe_1126 as C
    report=dict(status='running',protocol='selected_training_corpus_failures',
                student_checkpoint=str(Path(SOURCE).resolve()),student_sha256=sha(SOURCE),
                config_sha256=sha(C.CONFIG),normalizer_sha256=sha('results/imitation/1607/sim2real/agent_obs_scale.pkl'),
                dynamics_sha256=sha('configs/task/dynamic.yaml'),expert_config_sha256=sha('configs/task/f1tenth_preview_1126.yaml'),
                planner_code_sha256=sha('utils/rule_based.py'),selection_report=str(selection),selection_sha256=sha(selection),
                corpus_sha256=sha(corpus_path),jobs=[dict(family=f,track=p,seed=s,track_sha256=sha(p)) for f,p,s in jobs],runs=[],
                qualification='Source25failedcases selectedonlyfrom13auditedtrainingcases. Exactsource25prefix2sbeforetermination,orresetforshortstartup. Fullteachercontinuationsareprivilegedfeasibilitydata,notdeployedcontrollerorindependentqualification. No heldoutgradients orautomaticadoption.')
    (dest/'shards').mkdir(parents=True)
    def save():
        temp=dest/'collection.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(dest/'collection.json')
    save()
    with mp.get_context('spawn').Pool(2,initializer=initialize,initargs=(str(dest),)) as pool:
        for result in pool.imap_unordered(C.drive,jobs):
            expected=next(r for r in failures if (r['track'],r['seed'])==(result['track'],result['seed']))
            assert result['baseline']['success']==expected['complete']
            assert abs(result['baseline']['seconds']-expected['seconds'])<1e-8
            report['runs'].append(result);save();print('CASE',Path(result['track']).name,result.get('outcome'),flush=True)
    assert len(report['runs'])==len(jobs) and sha(SOURCE)==report['student_sha256']
    report['status']='complete_not_applied';save();print('DONE',len(jobs),flush=True)


if __name__=='__main__':main()
