#!/usr/bin/env python3
"""Inventory every production track family and maintain explicit pre-training gates.

This prepares/diagnoses only. It cannot invoke IL, PPO or a real-car process.
Held-out courses are indexed separately and never emitted as training inputs.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/f1tenth'
sys.path[:0]=[str(SRC),str(SRC/'installation')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    cli=p.parse_args();os.chdir(SRC)
    import numpy as np
    from easydict import EasyDict
    import train_il_s2r as il
    from utils.training_mix import groups_for,hall_split
    from utils.track_short import short_tracks
    from utils.track_gen import list_variants
    from eval_open_space import TEST_DIR
    destination=Path(cli.output_dir).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    real=['installation/maps_real/Practice1126b.npz','installation/maps_real/Practice1126b_rev.npz']
    args=EasyDict(training_real_tracks=real,
                 training_mix=dict(hall=.4,short=.3,regular=.15,open=.1,real=.05))
    regular=list_variants(il.MAPS)
    groups=groups_for(args,[x for x in regular if il.HOLDOUT not in x])
    held=dict(regular=[x for x in regular if il.HOLDOUT in x],hall=hall_split()[1],
              short=short_tracks(True),open=sorted(str(x) for x in Path(TEST_DIR).rglob('*.npz')))
    train={str(Path(x).resolve()) for paths in groups.values() for x in paths}
    evaluation={str(Path(x).resolve()) for paths in held.values() for x in paths}
    assert not train & evaluation,'Train/evaluation path overlap'
    def records(sets,role):
        rows=[]
        for family,paths in sets.items():
            for path in paths:
                path=Path(path).resolve()
                with np.load(path,allow_pickle=False) as z:
                    c=z['centerline'].astype(float)
                    length=float(np.linalg.norm(np.roll(c,-1,axis=0)-c,axis=1).sum())
                rows.append(dict(path=str(path),relative_path=str(path.relative_to(SRC)),
                                 family=family,role=role,sha256=sha(path),length_m=length,
                                 direction='reverse' if '_rev' in path.stem else 'forward'))
        return rows
    manifest=dict(schema=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  purpose='Generic per-map planning across every production family; 2025 is only one known training example.',
                  train=records(groups,'train'),evaluation=records(held,'evaluation_only'),
                  shares=dict(args.training_mix),
                  caveat='Existing regular variant splits share base layouts; do not call these unseen-base tests. Hall/random-short base layouts and later frozen independent scenarios need separate reporting.')
    manifest['counts']={'train':{k:len(v) for k,v in groups.items()},
                        'evaluation':{k:len(v) for k,v in held.items()}}
    (destination/'track_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Representatives exercise the generic code; they do not certify all tracks.
    representatives=[]
    for family in groups:
        rows=sorted((r for r in manifest['train'] if r['family']==family),key=lambda r:r['length_m'])
        for row in [rows[0],rows[len(rows)//2],rows[-1]]:
            if row['path'] not in [x['path'] for x in representatives]:representatives.append(row)
    for row in manifest['train']:
        if Path(row['path']).name.startswith('Course2025') and row['path'] not in [x['path'] for x in representatives]:
            representatives.append(row)
    (destination/'development_suite.json').write_text(json.dumps(dict(
        status='prepared_not_qualified',source_manifest_sha256=sha(destination/'track_manifest.json'),
        tracks=representatives),indent=2)+'\n')
    checks=[('generic_reference_optimizer','in_progress','Same solver supports arbitrary npz path; dense feasibility remains under diagnosis.'),
            ('all_training_reference_coverage','pending','Inventory all train paths; each replacement needs explicit valid/fallback decision.'),
            ('dense_body_and_actuator_audit','pending','Reject sampled-only success; verify dense force/rate/body constraints and static raster footprint.'),
            ('joint_preview_teacher_consistency','pending','Exact validated optimized speed and geometry must reach both consumers; no legacy recomputation.'),
            ('nominal_and_worst_closed_loop','pending','Compare source and candidate package with paired physics/seeds, two laps, straight/hairpin diagnostics.'),
            ('stale_map_geometry','in_progress','15 known-example variants prepared; generalize preparation/evaluation to all families with nominal map fixed.'),
            ('shared_course_two_car','pending','Separated start positions, actual occlusion and collisions; same scans to actor and scan localizer.'),
            ('wrong_valid_and_recovery','pending','Actual scan-localizer errors, false valid, dropouts and reacquisition; synthetic noise alone is insufficient.'),
            ('dataset_and_split_audit','pending','Matched teacher labels/recovery and held-out isolation; paths, hashes and normalizer frozen.'),
            ('launch_parse_resume_smoke','pending','Prepare a new run/config/argv; verify without launching PPO or changing preserved30u10.'),
            ('user_explicit_training_start','required','User has allowed preparation only. Never start IL/PPO from this tool.')]
    gate=dict(status='not_ready',IL_PPO_started=False,user_start_required=True,
              pipeline=['generic trajectory optimization','dense/static feasibility','profile integration',
                        'closed-loop nominal/worst/traffic/map mismatch diagnosis','data preparation and audit',
                        'frozen launch preflight','wait for explicit user training start'],
              checks=[dict(name=n,status=s,requirement=r) for n,s,r in checks],
              optimizer_must_not_change='Physical wall variants retain original reference geometry. Never give the actor a reoptimized true physical map.',
              manifest_sha256=sha(destination/'track_manifest.json'))
    (destination/'readiness.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps(dict(status=gate['status'],counts=manifest['counts'],development_tracks=len(representatives),
                          training_started=False),indent=2),flush=True)


if __name__=='__main__':main()
