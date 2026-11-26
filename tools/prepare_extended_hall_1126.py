#!/usr/bin/env python3
"""Build longer narrow hall layouts in isolation; split base layouts first.

More grid cells extend the course. Pitch, rounding, wall and width distributions
remain those of the existing hall generator. These are synthetic coverage probes,
not predictions of the unpublished competition course. No live map is changed.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC / 'installation')]
BANDS = ((110., 150.), (150., 210.), (210., 270.))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extended_centerline(seed, target, tries=1000):
    import numpy as np
    from utils import track_gen as tg, track_hall as hall
    from utils.raceline import resample_closed
    rng = np.random.default_rng(seed)
    for attempt in range(tries):
        pitch = float(rng.uniform(*hall.PITCH))
        a, b = int(rng.integers(2, 11)), int(rng.integers(2, 11))
        if max(a, b) > 3 * min(a, b):
            continue
        if not target[0] <= 4 * a * b * pitch <= target[1] * 1.15:
            continue
        straight = float(rng.uniform(.35, .95))
        loop = hall._tree_loop(a, b, hall._spanning_tree(a, b, rng, straight))
        if loop is None:
            continue
        rounding = float(rng.uniform(.8, 1.3))
        vertices = hall._corners(np.asarray(loop, float) * pitch)
        dense = hall._fillet(vertices, rounding, .05)
        center, length = resample_closed(hall._warp(dense, rng), tg.DENSE_DS)
        if target[0] <= length <= target[1]:
            return center, pitch, dict(grid=[a, b], pitch_m=pitch,
                rounding_limit_m=rounding, straight_preference=straight,
                grid_corners=len(vertices), geometry_attempts=attempt + 1,
                center_length_m=float(length))
    return None, None, dict(geometry_attempts=tries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--bases-per-band-per-split', type=int, default=2)
    cli = p.parse_args()
    os.chdir(SRC)
    import numpy as np
    from utils import track_gen as tg, track_hall as hall
    from compare_reference_limits_1126 import footprint_screen
    dest = Path(cli.output_dir).resolve()
    installation = SRC / 'installation'
    if dest == installation or installation in dest.parents or dest.exists():
        raise ValueError('Use a new directory outside live installation maps')
    if not 1 <= cli.bases_per_band_per_split <= 10:
        raise ValueError('Between 1 and 10 bases per band and split')
    sources = [SRC / 'utils' / (n + '.py') for n in ('track_gen', 'track_hall', 'raceline')]
    hashes = {str(s): sha(s) for s in sources}
    live_maps = {str(s): sha(s) for d in ('maps_hall', 'maps_short', 'maps_real')
                 for s in (installation / d).glob('*.npz')}
    signature = tg.generator_signature()
    report = dict(status='generating_not_used_in_training', variants=[], bases=[],
        rejected_candidates=[], source_sha256=hashes, live_maps_sha256=live_maps,
        generator_signature=signature, generator_script_sha256=sha(ROOT / 'tools' / Path(__file__).name),
        bands_m=BANDS, split_unit='base geometry, before direction and augmentation',
        qualification='Novel synthetic layouts, not a known competition course; static checks do not establish dynamic feasibility.',
        geometry_changes='More grid cells; aspect <=3; straight preference .35-.95; existing pitch/rounding/width distributions retained.',
        reference_line_limits=dict(tg.LAP_TIME_LIMITS))
    dest.mkdir(parents=True)
    def save():
        (dest / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    seen_geometry = set()
    for split, block in (('train', 410000), ('eval', 510000)):
        (dest / split).mkdir()
        for band_index, band in enumerate(BANDS):
            for slot in range(cli.bases_per_band_per_split):
                for retry in range(100):
                    seed = block + band_index * 1000 + slot * 100 + retry
                    center, pitch, metadata = extended_centerline(seed, band)
                    if center is None:
                        report['rejected_candidates'].append(dict(seed=seed, reason='no_length_matched_geometry'))
                        continue
                    geometry_hash = hashlib.sha256(center.tobytes()).hexdigest()
                    if geometry_hash in seen_geometry:
                        raise AssertionError('Duplicate base geometry across slots or splits')
                    pair = [hall.hall_course(center, pitch, np.random.default_rng(seed + 7919),
                            'HallExtended' + str(seed), seed, reverse) for reverse in (False, True)]
                    if any(v is None for v in pair):
                        report['rejected_candidates'].append(dict(seed=seed, reason='corridor_wall_geometry'))
                        continue
                    checks = [footprint_screen(v['raceline'], v) for v in pair]
                    # Record all valid corridor layouts, even when the approximate
                    # reference line clips a wall. Do not select on policy success.
                    seen_geometry.add(geometry_hash)
                    base = dict(seed=seed, split=split, band_m=band,
                                geometry_sha256=geometry_hash, **metadata)
                    report['bases'].append(base)
                    for v, check in zip(pair, checks):
                        v['base_geometry_sha256'] = geometry_hash
                        v['base_split'] = split
                        path = Path(tg.save_variant(v, str(dest / split))).resolve()
                        width = v['w_left'] + v['w_right']
                        row = dict(path=str(path), sha256=sha(path), split=split, seed=seed,
                            base_geometry_sha256=geometry_hash, reverse=bool(v['reverse']),
                            length_m=float(v['length']), width_min_m=float(width.min()),
                            width_median_m=float(np.median(width)), wall_min_m=float(v['wall_min']),
                            reference_min_radius_m=float(v['line_rmin']), footprint=check)
                        report['variants'].append(row)
                        print(split, v['name'], round(row['length_m'], 2),
                              'width', round(row['width_median_m'], 2), 'body_contacts',
                              check['poses_touching_occupied_samples'], flush=True)
                    save()
                    break
                else:
                    save()
                    raise RuntimeError('Could not generate base for ' + str((split, band, slot)))
    train = {b['geometry_sha256'] for b in report['bases'] if b['split'] == 'train'}
    heldout = {b['geometry_sha256'] for b in report['bases'] if b['split'] == 'eval'}
    assert train.isdisjoint(heldout)
    assert all(sha(path) == digest for path, digest in {**hashes, **live_maps}.items())
    assert tg.generator_signature() == signature
    assert len(report['variants']) == 12 * cli.bases_per_band_per_split
    report.update(status='generated_static_checks_recorded_not_applied',
                  base_splits_disjoint=True, live_maps_and_generator_unchanged=True)
    save()
    print('DONE', len(report['variants']), 'variants', dest, flush=True)


if __name__ == '__main__':
    main()
