#!/usr/bin/env python3
"""Prepare narrow/long/open combinations, preserving the existing base split.

Files stay outside live training directories. They are candidates for a later
data-only experiment, not a change to the current retention-only trial.
"""
import argparse
import json
import os
from pathlib import Path

from prepare_target_open_maps_1126 import ROOT, SRC, sha, make_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    cli = parser.parse_args()
    os.chdir(SRC)
    import numpy as np
    source_manifest = Path(cli.base_manifest).resolve()
    generation = json.loads(source_manifest.read_text())
    if (generation['status'] != 'generated_static_checks_recorded_not_applied'
            or not generation['base_splits_disjoint']):
        raise ValueError('Expected audited extended-layout base split')
    train = {r['base_geometry_sha256'] for r in generation['variants'] if r['split'] == 'train'}
    valid = {r['base_geometry_sha256'] for r in generation['variants'] if r['split'] == 'eval'}
    if not train or not valid or train & valid:
        raise ValueError('Base geometries must be nonempty and disjoint')
    for path, digest in generation['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Base generator source changed: ' + path)
    dest = Path(cli.output_dir).resolve()
    if dest.exists() or SRC / 'installation' in dest.parents or dest == SRC / 'installation':
        raise ValueError('Use a new isolated directory outside live maps')
    sources = {r['path']: r['sha256'] for r in generation['variants']}
    if len(sources) != len(generation['variants']):
        raise ValueError('Duplicate base variant paths')
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('An audited base variant changed')
    dest.mkdir(parents=True)
    for split in ('train', 'eval'):
        (dest / split).mkdir()
    report = dict(status='generating_not_used_in_training', sources=sources, variants=[],
        base_manifest=str(source_manifest), base_manifest_sha256=sha(source_manifest),
        base_splits_disjoint=True,
        generator_sha256=sha(Path(__file__).resolve()),
        opening_generator_sha256=sha(ROOT / 'tools/prepare_target_open_maps_1126.py'),
        qualification=('Inherits six train and six diagnostic-eval base geometries before '
                       'direction and opening augmentation. Eval bases have already been used '
                       'for diagnostics, so they are not a sealed final test. Static free-space '
                       'checks do not establish closed-loop feasibility or hardware performance.'),
        geometry_role='Future separate data experiment; current PPO28 training membership unchanged.')

    def save():
        temp = dest / 'manifest.tmp.json'
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(dest / 'manifest.json')

    save()
    for item in generation['variants']:
        source = Path(item['path'])
        with np.load(source, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        # Split follows base identity, never opening placement or policy success.
        seed = 112627 + int(item['seed'])
        candidate, checks = make_variant(arrays, seed)
        name = source.stem + '_open' + str(seed)
        candidate['name'] = np.asarray(name)
        candidate['augmentation_source_sha256'] = np.asarray(sources[str(source)])
        target = dest / item['split'] / (name + '.npz')
        for key in ('centerline', 'raceline', 'w_left', 'w_right', 'length'):
            if not np.array_equal(candidate[key], arrays[key]):
                raise ValueError('Opening changed the reference geometry: ' + key)
        np.savez_compressed(target, **candidate)
        report['variants'].append(dict(path=str(target), source=str(source), split=item['split'],
            seed=seed, base_seed=item['seed'], base_geometry_sha256=item['base_geometry_sha256'],
            reverse=item['reverse'], sha256=sha(target), checks=checks))
        save()
        print(item['split'], name, checks['added_free_area_m2'], flush=True)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Source variant changed during generation')
    report.update(status='generated_static_checks_passed_not_applied', source_hashes_unchanged=True)
    save()
    print('DONE', len(report['variants']), dest, flush=True)


if __name__ == '__main__':
    # Python3.8 can retain a relative __file__ across chdir.
    __file__ = str(Path(__file__).resolve())
    main()
