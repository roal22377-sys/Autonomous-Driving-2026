#!/usr/bin/env python3
"""Review complete recovery-pilot guard cohorts against the exact frozen source."""
import argparse
import json
import pickle
from pathlib import Path

from review_recovery_adaptation_1126 import SRC, sha
from review_ppo_retention_1126 import per_cap_comparison
from review_ppo_line_ablation_1126 import matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', choices=['100', '400'], required=True)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    out = Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve previous review')
    base = SRC / 'results'
    analysis = base / 'analysis1126'
    pilot = json.loads((base / 'recovery_adaptation_1126/pilot.json').read_text())
    known = json.loads((analysis / 'recovery_adaptation_source_known_seed7.json').read_text())
    ck = SRC / pilot['checkpoints'][cli.tag]['path']
    assert sha(ck) == pilot['checkpoints'][cli.tag]['sha256']
    assert sha(base / 'ppo_s2r/29/checkpoint/model_880000.pt') == pilot['source_sha256'] == known['checkpoint_sha256']
    assert sha(base / 'ppo_s2r/29/backup/f1tenth.yaml') == pilot['config_sha256']
    assert sha(SRC / 'configs/task/dynamic.yaml') == known['dynamics_sha256']
    history_path = base / 'ppo_s2r/29/train_log.pkl'
    history = pickle.loads(history_path.read_bytes())
    sources = [r for r in history if r.get('update') == 20 and 'eval_success' in r]
    assert len(sources) == 1
    source = sources[0]
    assert source['evaluation_seed_base'] == 7
    rows, report_hashes = {}, {}
    specifications = [
        ('lowcap', 'regular', 'res', 44, 'eval', False, None, [2., 2.5, 3., 3.5, 4.]),
        ('worst', 'regular', 'worst_res', 132, 'worst', False, None, None),
        ('lost_short', 'short', 'lost_short_res', 32, 'eval', True, None, None),
        ('lost_general', 'regular', 'lost_eval_res', 44, 'eval', True, [5.], None),
    ]
    for label, family, key, total, mode, lost, caps, cycle in specifications:
        path = analysis / ('recovery_adaptation_' + cli.tag + '_' + label + '_seed7.json')
        doc = json.loads(path.read_text())
        report_hashes[label] = sha(path)
        assert doc['checkpoint_sha256'] == sha(ck)
        assert doc['seeds'] == [7] and doc['simulation_mode'] == mode
        assert doc['localization_forced_lost'] == lost
        assert doc['diagnostic_caps'] == caps and doc['diagnostic_cycle_caps'] == cycle
        assert doc['regular_scope'] == 'standard'
        assert doc['config_sha256'] == pilot['config_sha256']
        assert doc['normalizer_arrays_sha256'] == known['normalizer_arrays_sha256']
        assert doc['dynamics_sha256'] == known['dynamics_sha256']
        assert set(doc['families']) == {family}
        data = doc['families'][family]
        assert data['total'] == len(data['runs']) == total
        reference = {r[0]: r for r in source[key]}
        if label == 'lowcap':
            reference = {k: v for k, v in reference.items() if float(k.split('@')[1].split('/')[0]) < 5}
        assert len(reference) == total
        order = source['evaluation_track_order'][key]
        current, old = [], []
        for entry in data['runs']:
            r = entry['result']
            assert entry['seed'] == 7 + order.index(r[0].split('@')[0])
            assert r[0] in reference
            current.append(r)
            old.append(reference[r[0]])
        assert {r[0] for r in current} == set(reference)
        rows[label] = dict(candidate_complete=sum(bool(r[1]) for r in current),
                           source_complete=sum(bool(r[1]) for r in old), total=total,
                           comparison=matched(current, old), per_cap=per_cap_comparison(current, old))
    report = dict(status='complete_read_only_not_promoted', tag=cli.tag,
                  checkpoint_sha256=sha(ck), source_sha256=pilot['source_sha256'],
                  source_history_sha256=sha(history_path), report_sha256=report_hashes,
                  cohorts=rows,
                  completion_regressions=[k for k, v in rows.items() if v['candidate_complete'] < v['source_complete']],
                  qualification='Matched source29u20 cases and per-track seeds, including low-cap44, worst132, missing-preview short32/general44. Missing preview retains LiDAR and is not relocalization. Same-count cohorts can exchange failure identities. Pace includes paired completions only. Broad nominal, source25u70 floors and expanded geometry/start/seed checks remain necessary; no automatic promotion.')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
