#!/usr/bin/env python3
"""Review paired recovery-retention branches with exact cohort and seed checks."""
import argparse
import json
import pickle
from pathlib import Path

from review_recovery_adaptation_1126 import SRC, sha
from review_ppo_line_ablation_1126 import matched
from review_ppo_retention_1126 import per_cap_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', choices=['50', '100'], default='100')
    parser.add_argument('--kind', choices=['worst', 'broad'], required=True)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    out = Path(cli.output)
    if out.exists():
        raise FileExistsError('Preserve previous review')
    base = SRC/'results'; analysis = base/'analysis1126'
    pilot_path = base/'stress_recovery_adaptation_1126/pilot.json'
    pilot = json.loads(pilot_path.read_text())
    assert pilot['status'] == 'complete_not_promoted'
    source_ck = SRC/pilot['source_checkpoint']
    assert sha(source_ck) == pilot['source_sha256']
    assert sha(base/'ppo_s2r/29/backup/f1tenth.yaml') == pilot['config_sha256']
    normalizer = json.loads((analysis/'recovery_adaptation_source_known_seed7.json').read_text())
    assert normalizer['checkpoint_sha256'] == pilot['source_sha256']
    with (base/'ppo_s2r/29/train_log.pkl').open('rb') as f:
        history = pickle.load(f)
    sources = [r for r in history if r.get('update') == 20 and 'eval_success' in r]
    assert len(sources) == 1
    source = sources[0]; assert source['evaluation_seed_base'] == 7
    specs = [('regular', 'worst_res', 132)] if cli.kind == 'worst' else [
        ('regular', 'res', 132), ('open', 'open_res', 88), ('hall', 'hall_res', 102), ('short', 'short_res', 32)]
    paths = {branch: analysis/('stress_recovery_'+branch+'_'+cli.tag+'_'+cli.kind+'_seed7.json') for branch in pilot['branches']}
    paths['original_recovery100'] = analysis/('recovery_adaptation_100_'+cli.kind+'_seed7.json')
    docs = {k: json.loads(p.read_text()) for k, p in paths.items()}
    original = json.loads((base/'recovery_adaptation_1126/pilot.json').read_text())
    assert original['source_sha256'] == pilot['source_sha256']
    ck_hashes = {}
    for branch, doc in docs.items():
        if branch == 'original_recovery100':
            ck = original['checkpoints']['100']
        else:
            ck = pilot['branches'][branch]['checkpoints'][cli.tag]
        assert sha(SRC/ck['path']) == ck['sha256'] == doc['checkpoint_sha256']
        ck_hashes[branch] = ck['sha256']
        assert doc['seeds'] == [7] and doc['simulation_mode'] == ('worst' if cli.kind == 'worst' else 'eval')
        assert not doc['localization_forced_lost'] and doc['diagnostic_caps'] is None and doc['diagnostic_cycle_caps'] is None
        assert doc['regular_scope'] == 'standard'
        assert doc['config_sha256'] == pilot['config_sha256']
        assert doc['normalizer_arrays_sha256'] == normalizer['normalizer_arrays_sha256']
        assert doc['dynamics_sha256'] == normalizer['dynamics_sha256'] == sha(SRC/'configs/task/dynamic.yaml')
        assert set(doc['families']) == {x[0] for x in specs}
    cohorts = {}
    for family, key, total in specs:
        reference = {r[0]: r for r in source[key]}
        if key == 'res':
            reference = {k: v for k, v in reference.items() if float(k.split('@')[1].split('/')[0]) >= 5}
        assert len(reference) == total
        aligned = {}
        order = source['evaluation_track_order'][key]
        for branch, doc in docs.items():
            data = doc['families'][family]
            assert data['total'] == len(data['runs']) == total
            entries = {r['result'][0]: r for r in data['runs']}
            assert len(entries) == total and set(entries) == set(reference)
            aligned[branch] = []
            for name in reference:
                entry = entries[name]
                assert entry['seed'] == 7+order.index(name.split('@')[0])
                aligned[branch].append(entry['result'])
        old = list(reference.values())
        cohorts[family] = dict(total=total, source_complete=sum(bool(r[1]) for r in old), branches={})
        for branch in pilot['branches']:
            current = aligned[branch]
            cohorts[family]['branches'][branch] = dict(
                complete=sum(bool(r[1]) for r in current),
                vs_source=matched(current, old), per_cap_source=per_cap_comparison(current, old),
                vs_original_recovery100=matched(current, aligned['original_recovery100']))
        cohorts[family]['stress_vs_generic'] = matched(aligned['stress_rehearsal'], aligned['generic_control'])
        cohorts[family]['stress_vs_generic_per_cap'] = per_cap_comparison(aligned['stress_rehearsal'], aligned['generic_control'])
    report = dict(status='complete_read_only_not_promoted', kind=cli.kind, tag=cli.tag,
                  pilot_sha256=sha(pilot_path), source_sha256=pilot['source_sha256'], checkpoint_sha256=ck_hashes,
                  evaluation_sha256={k: sha(p) for k, p in paths.items()}, cohorts=cohorts,
                  qualification='Both branches restart source29u20 with identical recovery labels, batches and weights; only the second retention bank changes. Comparison with originalrecovery100 also changes retention strength/batch composition and is not a single-variable ablation. Exact evaluation cases, seeds, configuration, dynamics and normalization verified. Pairedcompletion pace excludes failures. Lowcaps,missingpreview,knownpractice/2025 and expanded geometry checks remain necessary before global selection; no automatic promotion or real-car qualification.')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(cohorts, indent=2))


if __name__ == '__main__':
    main()
