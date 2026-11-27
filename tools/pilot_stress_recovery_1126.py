#!/usr/bin/env python3
"""Compare stronger generic retention with adverse-state retention in recovery IL.

Both branches restart frozen29u20 and use the SAME original six recoveries.
Only the second anchor bank differs. Newly mined failures are kept for a later
DAgger round, avoiding simultaneous label and retention interventions here.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC/'installation'), str(ROOT/'tools')]
from collect_expert_recovery_probe_1126 import sha


def worker_init():
    import collect_expert_recovery_probe_1126 as C
    C.initialize('results/expert_recovery_failures_1126', True)


def evaluate(job):
    import pilot_recovery_adaptation_1126 as P
    destination, branch, row, tag = job
    P.DEST = Path(destination)/branch
    result = P.evaluate_case((row, tag))
    result['branch'] = branch
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--stress_collection', default='results/source29u20_stress_rehearsal_1126/collection.json')
    parser.add_argument('--stress_audit', default='results/source29u20_stress_rehearsal_1126/audit.json')
    parser.add_argument('--prepare_only', action='store_true', help='Validate inputs and assemble banks, without writing or fitting')
    cli = parser.parse_args()
    os.chdir(SRC)
    import numpy as np
    import torch
    import train_il_s2r as IL
    from utils.policy_transfer import columns
    from utils.training_mix import groups_for, step_budgets
    from utils.vectorize import RunningMeanStd
    torch.set_num_threads(1)
    dest = Path(cli.destination)
    if dest.exists():
        raise FileExistsError('Preserve previous experiment')
    source = Path('results/ppo_s2r/29/checkpoint/model_880000.pt')
    config = Path('results/ppo_s2r/29/backup/f1tenth.yaml')
    cp = Path('results/expert_recovery_failures_1126/collection.json')
    ap = cp.with_name('audit_classified.json')
    sp, sap = Path(cli.stress_collection), Path(cli.stress_audit)
    collection, audit, stress, stress_audit = [json.loads(p.read_text()) for p in (cp, ap, sp, sap)]
    assert collection['status'] == stress['status'] == 'complete_not_applied'
    assert audit['status'] == stress_audit['status'] == 'passed_not_applied'
    assert audit['collection_sha256'] == sha(cp) and stress_audit['collection_sha256'] == sha(sp)
    assert collection['student_sha256'] == stress['artifacts']['source']['sha256'] == stress_audit['source_sha256'] == sha(source)
    assert collection['config_sha256'] == stress['artifacts']['config']['sha256'] == sha(config)
    assert collection['dynamics_sha256'] == stress['artifacts']['dynamics']['sha256'] == sha('configs/task/dynamic.yaml')
    for item in stress['artifacts'].values():
        assert sha(item['path']) == item['sha256']
    assert stress_audit['cases'] == len(stress['runs']) == 184
    args = IL.get_args(['--task_config', str(config)])
    rms = RunningMeanStd('agent_obs', 132); rms.load('results/imitation/1607/sim2real', required=True)
    assert sha('results/imitation/1607/sim2real/agent_obs_scale.pkl') == collection['normalizer_sha256'] == stress['artifacts']['normalizer']['sha256']
    names = columns(args)
    groups = groups_for(args, [p for p in IL.list_variants(IL.MAPS) if IL.HOLDOUT not in p])
    mpth = Path('results/imitation/14/backup/track_manifest.json')
    manifest = json.loads(mpth.read_text())
    for family, tracks in manifest['training'].items():
        assert {str(Path(p).resolve()) for p in tracks} == {str(Path(p).resolve()) for p in groups[family]}
    assert columns(IL.get_args(['--task_config', 'results/imitation/14/backup/f1tenth.yaml'])) == names
    # get_args also changes F1_TASK_CONFIG, which actor_from_state reads.
    args = IL.get_args(['--task_config', str(config)])
    ck = torch.load(source, map_location='cpu', weights_only=False)
    teacher = IL.actor_from_state(ck['actor']); teacher.eval()
    archive = Path('results/imitation/14/data/round000.npz')
    rng = np.random.default_rng(112671)
    with np.load(archive, allow_pickle=False) as z:
        raw = z['O']; assert raw.shape == (1200000, 132)
        budgets = step_budgets(manifest['step_shares'], len(raw))
        target = step_budgets(dict(args.training_mix), 32000)
        chunks = []; offset = 0
        for family, budget in budgets.items():
            indices = rng.choice(budget, target[family], replace=False)+offset
            chunks.append(raw[indices].astype(np.float32)); offset += budget
        O = np.concatenate(chunks)
    missing = rng.choice(len(O), len(O)//4, replace=False)
    O[np.ix_(missing, [i for i, name in enumerate(names) if name.startswith('preview/')])] = 0
    X = torch.as_tensor((O-rms.mean)/np.sqrt(rms.var+1e-8), dtype=torch.float32)
    with torch.no_grad():
        T = torch.cat([teacher(X[i:i+2048]).loc for i in range(0, len(X), 2048)])
    successes = sorted((r for r in stress['runs'] if r['admissible_for_rehearsal']), key=lambda r: r['seed'])
    assert len(successes) == stress_audit['completed'] and successes
    stress_rng = np.random.default_rng(112673)
    q, remainder = divmod(32000, len(successes))
    stress_obs, stress_targets, provenance = [], [], []
    allowed = {str(Path(p).resolve()) for p in groups['regular']}
    for i, row in enumerate(successes):
        assert row['track'] in allowed and row['complete'] and not row['collision'] and len(row['lap_times']) == 2
        assert sha(row['shard']) == row['shard_sha256'] and sha(row['track']) == row['track_sha256']
        with np.load(row['shard'], allow_pickle=False) as z:
            count = q+int(i < remainder)
            assert count <= len(z['O']) and list(z['columns']) == names
            indices = stress_rng.choice(len(z['O']), count, replace=False)
            stress_obs.append(z['O'][indices]); stress_targets.append(z['T'][indices])
            provenance.append(dict(track=row['track'], seed=row['seed'], start=row['start'], rows=count,
                                   shard=row['shard'], shard_sha256=row['shard_sha256'], indices=indices.tolist()))
    SX = torch.as_tensor((np.concatenate(stress_obs)-rms.mean)/np.sqrt(rms.var+1e-8), dtype=torch.float32)
    ST = torch.as_tensor(np.concatenate(stress_targets), dtype=torch.float32)
    assert SX.shape == X.shape == (32000, 132) and ST.shape == T.shape == (32000, 2)
    assert all(torch.isfinite(v).all() for v in (X, T, SX, ST))
    rows = sorted((r for r in collection['runs'] if r.get('outcome') == 'rescued'), key=lambda r: r['track'])
    assert len(rows) == 6 and sum(r['handoff_at_reset'] for r in rows) == 1
    recovery = []
    for row in rows:
        assert sha(row['shard']) == row['shard_sha256']
        with np.load(row['shard'], allow_pickle=False) as z:
            assert list(z['columns']) == names
            rx = torch.as_tensor((z['O']-rms.mean)/np.sqrt(rms.var+1e-8), dtype=torch.float32)
            ry = torch.as_tensor(z['Y'], dtype=torch.float32)
            early = z['seconds_since_handoff'] <= 2
            assert early.any() and (~early).any()
            recovery.append((rx, ry, np.flatnonzero(early), np.flatnonzero(~early)))
    with torch.no_grad():
        start_stress_error = torch.cat([(teacher(SX[i:i+2048]).loc-ST[i:i+2048]).square() for i in range(0, len(SX), 2048)]).mean(0)
    assert start_stress_error.max() < 1e-10
    if cli.prepare_only:
        print(json.dumps(dict(status='prepared_in_memory_no_fit', broad_rows=len(X), stress_rows=len(SX), successful_stress_cases=len(successes), recoveries=len(recovery), stress_source_mse=start_stress_error.tolist())))
        return
    dest.mkdir(parents=True)
    report = dict(status='fitting', source_checkpoint=str(source), source_sha256=sha(source),
                  config_sha256=sha(config), normalizer_sha256=collection['normalizer_sha256'],
                  recovery_collection_sha256=sha(cp), recovery_audit_sha256=sha(ap),
                  stress_collection_sha256=sha(sp), stress_audit_sha256=sha(sap),
                  anchor_archive_sha256=sha(archive), anchor_manifest_sha256=sha(mpth),
                  broad_rows=32000, broad_family_counts=target, synthetic_missing_preview_rows=len(missing),
                  stress_rows=32000, successful_stress_cases=len(successes), stress_provenance=provenance,
                  recovery_cases=6, seed=112674, steps=100, snapshots=[50, 100], lr=1e-5,
                  batch=dict(broad=256, second_anchor=128, recovery=128),
                  loss='0.75*broad_anchor + 0.75*second_anchor + 0.25*recovery, joint steer/speed weights1/.25; second_anchor is same broadbank for generic_control and stressbank for stress_rehearsal',
                  branches={}, driving=[],
                  qualification='Paired offline branches restart29u20, identical batches/seeds except secondanchor observations and targets. Fixed original6recovery cases only, including1from-reset startup. Newcandidate100failure labels withheld to isolate retention intervention. Broad syntheticmissing anchors are not successful-rollout demonstrations. Successful stress episodes use training augmentations and do not establish unseen-layout or real-hardware performance. No PPO promotion or vehicle changes.')
    def save():
        temp = dest/'pilot.tmp'; temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); temp.replace(dest/'pilot.json')
    weights = torch.tensor([1., .25])
    for branch, BX, BT in [('generic_control', X, T), ('stress_rehearsal', SX, ST)]:
        branch_dir = dest/branch
        (branch_dir/'checkpoint').mkdir(parents=True); (branch_dir/'backup').mkdir()
        shutil.copyfile(config, branch_dir/'backup/f1tenth.yaml')
        actor = IL.actor_from_state(ck['actor']); actor.train()
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
        fit_rng = np.random.default_rng(112674); torch.manual_seed(112674); random.seed(112674)
        entry = dict(checkpoints={}, fit=[]); report['branches'][branch] = entry
        def metrics(step):
            actor.eval()
            with torch.no_grad():
                errors = {}
                for label, ax, at in [('broad', X, T), ('stress', SX, ST)]:
                    errors[label] = torch.cat([(actor(ax[i:i+2048]).loc-at[i:i+2048]).square() for i in range(0, len(ax), 2048)]).mean(0).tolist()
                recoveries = []
                for rx, ry, early, late in recovery:
                    e = torch.cat([(actor(rx[i:i+2048]).loc-ry[i:i+2048]).square() for i in range(0, len(rx), 2048)])
                    recoveries.append(dict(early_mse=e[early].mean(0).tolist(), late_mse=e[late].mean(0).tolist()))
            actor.train()
            return dict(step=step, anchor_mse=errors, recovery=recoveries)
        entry['fit'].append(metrics(0)); assert max(entry['fit'][0]['anchor_mse']['broad']) == 0
        for step in range(1, 101):
            ai = fit_rng.integers(len(X), size=256); bi = fit_rng.integers(len(BX), size=128)
            xp, yp = [], []
            for early_part in (True, False):
                ids = fit_rng.integers(len(recovery), size=64)
                for index, (rx, ry, early, late) in enumerate(recovery):
                    indices = fit_rng.choice(early if early_part else late, int((ids == index).sum()), replace=True)
                    xp.append(rx[indices]); yp.append(ry[indices])
            rx, ry = torch.cat(xp), torch.cat(yp)
            loss = (.75*((actor(X[ai]).loc-T[ai]).square()*weights).mean()
                    + .75*((actor(BX[bi]).loc-BT[bi]).square()*weights).mean()
                    + .25*((actor(rx).loc-ry).square()*weights).mean())
            assert torch.isfinite(loss)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.); optimizer.step()
            if step in (50, 100):
                assert all(torch.isfinite(v).all() for v in actor.state_dict().values())
                path = branch_dir/'checkpoint'/('model_%s.pt'%step)
                torch.save(dict(actor=actor.state_dict(), obs_mean=rms.mean, obs_var=rms.var), path)
                entry['checkpoints'][str(step)] = dict(path=str(path), sha256=sha(path))
                entry['fit'].append(metrics(step)); save(); print('FIT', branch, step, entry['fit'][-1]['anchor_mse'], flush=True)
    report['status'] = 'evaluating_training_cases'; save()
    jobs = [(str(dest), branch, row, str(tag)) for branch in report['branches'] for tag in (50, 100) for row in rows]
    with mp.get_context('spawn').Pool(2, initializer=worker_init) as pool:
        for result in pool.imap_unordered(evaluate, jobs):
            report['driving'].append(result); save(); print('CASE', result, flush=True)
    assert len(report['driving']) == 24 and sha(source) == report['source_sha256']
    report['status'] = 'complete_not_promoted'; save()


if __name__ == '__main__':
    main()
