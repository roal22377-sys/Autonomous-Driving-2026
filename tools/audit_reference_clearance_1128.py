#!/usr/bin/env python3
"""Compare optimized-line body clearance with its stored reference on each map.

This is a conservative pretraining gate, not a dynamic tracking certificate.
It handles narrow-course exceptions by measuring the original line on the same
occupancy grid instead of imposing a universal clearance on every map.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(ROOT / 'tools')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_path(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return None


def run(report_paths):
    import numpy as np
    from compare_reference_limits_1126 import footprint_screen
    records = []
    for report_path in report_paths:
        report = json.loads(Path(report_path).read_text())
        if report['status'] not in ('complete', 'complete_not_applied'):
            raise ValueError('Incomplete optimizer report: ' + str(report_path))
        for track in report['tracks']:
            source = Path(track['source'])
            if sha(source) != track['source_sha256']:
                raise ValueError('Changed source track: ' + str(source))
            with np.load(source, allow_pickle=False) as z:
                source_arrays = {k: z[k].copy() for k in z.files}
            source_screen = footprint_screen(source_arrays['raceline'], source_arrays)
            source_clearance = source_screen['minimum_body_grid_clearance_m']
            resolution = float(source_arrays['resolution'])
            for name, candidate in track['candidates'].items():
                path = Path(candidate['path'])
                if sha(path) != candidate['sha256']:
                    raise ValueError('Changed candidate track: ' + str(path))
                with np.load(path, allow_pickle=False) as z:
                    arrays = {k: z[k].copy() for k in z.files}
                for key in ('occupancy', 'origin', 'resolution', 'centerline'):
                    if not np.array_equal(source_arrays[key], arrays[key]):
                        raise ValueError('Nominal map changed: ' + key)
                screen = footprint_screen(arrays['raceline'], arrays)
                clearance = screen['minimum_body_grid_clearance_m']
                # Half a raster cell tolerates sampling/quantization differences.
                # It is not a claim that this loss of physical margin is safe.
                tolerance = resolution / 2
                records.append(dict(report=str(report_path), source=str(source),
                                    source_repo_path=repo_path(source),
                                    candidate=name, candidate_path=str(path),
                                    candidate_repo_path=repo_path(path),
                                    source_sha256=track['source_sha256'],
                                    candidate_sha256=candidate['sha256'],
                                    resolution_m=resolution,
                                    source_minimum_body_grid_clearance_m=source_clearance,
                                    candidate_minimum_body_grid_clearance_m=clearance,
                                    clearance_loss_m=source_clearance-clearance,
                                    raster_tolerance_m=tolerance,
                                    source_static_collision_poses=source_screen['poses_touching_occupied_samples'],
                                    candidate_static_collision_poses=screen['poses_touching_occupied_samples'],
                                    relative_clearance_gate_passed=bool(
                                        screen['poses_touching_occupied_samples'] == 0
                                        and clearance + tolerance >= source_clearance)))
    return dict(schema=1, status='complete', training_started=False,
                qualification='Conservative sampled static comparison only. Closed-loop and local uncertainty checks remain required.',
                records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run([Path(p) for p in args.reports])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(candidates=len(report['records']),
                          passed=sum(r['relative_clearance_gate_passed'] for r in report['records']),
                          output=str(target))))


if __name__ == '__main__':
    main()
