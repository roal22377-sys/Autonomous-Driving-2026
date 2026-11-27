#!/usr/bin/env python3
"""Review all complete source25 recovery50 cohorts against the exact source."""
import argparse
import json
from pathlib import Path

from review_recovery_adaptation_1126 import SRC,sha
from review_ppo_line_ablation_1126 import matched
from review_ppo_retention_1126 import per_cap_comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    cli=parser.parse_args();out=Path(cli.output)
    if out.exists():raise FileExistsError('Preserve prior review')
    base=SRC/'results';analysis=base/'analysis1126'
    pilot_path=base/'source25_recovery_adaptation_1126/pilot.json'
    pilot=json.loads(pilot_path.read_text());assert pilot['status']=='complete_not_promoted'
    ck=pilot['checkpoints']['50'];assert sha(SRC/ck['path'])==ck['sha256']
    assert sha(SRC/pilot['source_checkpoint'])==pilot['source_sha256']
    full_path=analysis/'ppo25_u70_full_family_reference.json'
    full=json.loads(full_path.read_text());assert full['checkpoint_sha256']==pilot['source_sha256']
    names=['broad','worst','lowcap','lost_short','lost_general']
    paths={k:analysis/('source25_recovery_50_'+k+'_seed7.json') for k in names}
    docs={k:json.loads(p.read_text()) for k,p in paths.items()}
    source_paths={
        'worst':analysis/'ppo25_u70_worst_regular_reference_seed7.json',
        'lost_short':analysis/'ppo25_u70_lost_short_reference_seed7.json',
        'lost_general':analysis/'ppo25_u70_lost_general_reference_seed7.json',
        'known':analysis/'source25_u70_known_separate_seed7.json'}
    sources={k:json.loads(p.read_text()) for k,p in source_paths.items()}
    norm=sources['known']
    assert norm['checkpoint_sha256']==pilot['source_sha256']
    assert sha(SRC/'configs/task/dynamic.yaml')==norm['dynamics_sha256']
    assert sha(base/'source25_recovery_adaptation_1126/backup/f1tenth.yaml')==pilot['config_sha256']
    for label,doc in docs.items():
        assert doc['checkpoint_sha256']==ck['sha256'] and doc['seeds']==[7]
        assert doc['simulation_mode']==('worst' if label=='worst' else 'eval')
        assert doc['localization_forced_lost']==label.startswith('lost_')
        assert doc['diagnostic_caps']==([5.] if label=='lost_general' else None)
        assert doc['diagnostic_cycle_caps']==([2.,2.5,3.,3.5,4.] if label=='lowcap' else None)
        assert doc['config_sha256']==pilot['config_sha256']
        assert doc['dynamics_sha256']==norm['dynamics_sha256']
        assert doc['normalizer_arrays_sha256']==norm['normalizer_arrays_sha256']
    assert set(docs['broad']['families'])=={'regular','short','hall','real','2025','open'}
    cohorts={};complete_sets={}
    def compare_group(label,doc,family,reference,total,source_entries=None):
        data=doc['families'][family];assert data['total']==len(data['runs'])==total
        entries={r['result'][0]:r for r in data['runs']};old={r[0]:r for r in reference}
        assert len(entries)==len(old)==total and set(entries)==set(old)
        # Historical full references record the shared seed schedule and retain
        # track order. Per-setting repeats reuse that track's original seed.
        order=list(dict.fromkeys(r[0].split('@')[0] for r in reference))
        exact={r['result'][0]:r for r in source_entries} if source_entries is not None else None
        current=[];previous=[]
        for name in old:
            entry=entries[name]
            assert entry['seed']==(exact[name]['seed'] if exact is not None else 7+order.index(name.split('@')[0]))
            current.append(entry['result']);previous.append(old[name])
        cohorts[label]=dict(candidate_complete=sum(bool(r[1]) for r in current),source_complete=sum(bool(r[1]) for r in previous),total=total,
                            comparison=matched(current,previous),per_cap=per_cap_comparison(current,previous))
        complete_sets[label]=(current,previous)
    for family,total in [('regular',132),('short',32),('hall',102),('open',88)]:
        reference=full['families'][family]['res']
        if family=='regular':reference=[r for r in reference if float(r[0].split('@')[1].split('/')[0])>=5]
        compare_group(family,docs['broad'],family,reference,total)
    for family in ['real','2025']:
        entries=sources['known']['families'][family]['runs']
        compare_group(family,docs['broad'],family,[r['result'] for r in entries],4,entries)
    low=[r for r in full['families']['regular']['res'] if float(r[0].split('@')[1].split('/')[0])<5]
    compare_group('lowcap',docs['lowcap'],'regular',low,44)
    for label,family,total in [('worst','regular',132),('lost_short','short',32),('lost_general','regular',44)]:
        source=sources[label];assert source['checkpoint_sha256']==pilot['source_sha256']
        assert source['dynamics_sha256']==norm['dynamics_sha256'] and source['seeds']==[7]
        entries=source['families'][family]['runs']
        compare_group(label,docs[label],family,[r['result'] for r in entries],total,entries)
    cur=complete_sets['regular'][0]+complete_sets['lowcap'][0]
    ref=complete_sets['regular'][1]+complete_sets['lowcap'][1]
    assert len({r[0] for r in cur})==176
    cohorts['regular176']=dict(candidate_complete=sum(bool(r[1]) for r in cur),source_complete=sum(bool(r[1]) for r in ref),total=176,comparison=matched(cur,ref))
    regression=[k for k,v in cohorts.items() if v['candidate_complete']<v['source_complete']]
    report=dict(status='complete_read_only_not_promoted',checkpoint_sha256=ck['sha256'],source_sha256=pilot['source_sha256'],
                input_report_sha256={k:sha(p) for k,p in paths.items()},source_report_sha256={k:sha(p) for k,p in source_paths.items()},
                historical_full_reference_sha256=sha(full_path),cohorts=cohorts,completion_regressions=regression,
                qualification='Exactsource25u70casesandtrack-seeds;knownreal/2025sourcewasrerunwithfamily-localindexing.All614candidatecasescompletebeforethisreview.General176reconstructedfrom132+44withoutduplicates.Paceonlypairedcompletions.Same-countcohortscanexchangefailures.Training13/13isnotindependentqualification;noautomaticPPOinitializationorhardwarepromotion.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{x:v[x] for x in ['candidate_complete','source_complete','total']} for k,v in cohorts.items()},indent=2))


if __name__=='__main__':main()
