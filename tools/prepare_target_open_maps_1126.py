#!/usr/bin/env python3
"""Make isolated open-area variants of known courses; never modify live maps."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC/'installation')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def topology(free):
    """Check both digital connectivities; openings must not create new loops."""
    import numpy as np
    from scipy.ndimage import label
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    return tuple(label(x, structure=s)[1] for x in (free, ~free)
                 for s in (cross, np.ones((3, 3))))


def make_variant(arrays, seed):
    import numpy as np
    from scipy.ndimage import binary_dilation, distance_transform_edt, label
    from scipy.spatial import cKDTree
    from utils.raceline import tangents_normals, curvature_smooth
    rng = np.random.default_rng(seed)
    resolution = float(arrays['resolution'])
    pad = int(np.ceil(6. / resolution))
    base = np.pad(arrays['occupancy'] > 128, pad, constant_values=False)
    origin = arrays['origin'].copy(); origin[:2] -= pad*resolution
    y, x = np.indices(base.shape)
    points = np.stack([origin[0]+(x.ravel()+.5)*resolution,
                       origin[1]+(y.ravel()+.5)*resolution], -1)
    center = arrays['centerline'].astype(float)
    tangent, normal = tangents_normals(center)
    curvature = curvature_smooth(center, span=.6)
    _, nearest = cKDTree(center).query(points)
    length = float(arrays['length']); ds = length/len(center)
    original_topology = topology(base)
    free = base.copy(); accepted = []; used = []
    wl, wr = arrays['w_left'], arrays['w_right']
    for attempt in range(250):
        index = int(rng.integers(len(center)))
        if any(min(abs(index-j), len(center)-abs(index-j))*ds < 5. for j in used):
            continue
        side = int(rng.choice([-1, 1]))
        width = float(wl[index] if side > 0 else wr[index])
        kind = 'side_pocket' if attempt % 2 == 0 else 'forward_bay'
        if kind == 'side_pocket':
            if abs(curvature[index]) > .3:
                continue
            mouth = float(rng.uniform(1.5, 3.0)); depth = float(rng.uniform(1.0, 2.5))
            start = center[index]-tangent[index]*mouth/2
            lo, hi = sorted((side*(width-.20), side*(width+depth)))
            reach = mouth
            local_arc = mouth/2+1.5
        else:
            # The next 3 m must actually bend: a forward bay at an arbitrary
            # straight would simply duplicate the existing corridor.
            ahead = (index+int(3./ds)) % len(center)
            if np.dot(tangent[index], tangent[ahead]) > .8:
                continue
            start = center[index]
            reach = float(rng.uniform(2., 4.))
            lo, hi = -float(wr[index]), float(wl[index])
            if side < 0: lo -= float(rng.uniform(.3, 1.2))
            else: hi += float(rng.uniform(.3, 1.2))
            local_arc = reach+1.5
        rel = points-start
        u, v = rel@tangent[index], rel@normal[index]
        mask = ((u >= 0)&(u <= reach)&(v >= lo)&(v <= hi)).reshape(base.shape)
        added = mask & ~free
        area = float(added.sum()*resolution**2)
        if area < .8 or label(added)[1] != 1:
            continue
        di = np.abs(nearest-index); arc = np.minimum(di, len(center)-di)*ds
        foreign = base & (arc.reshape(base.shape) > local_arc)
        # Keep an occupied strip to every nonlocal portion of the course.
        clearance = distance_transform_edt(~foreign)*resolution
        if np.any(clearance[added] < .30):
            continue
        contact = binary_dilation(added) & free
        if contact.sum()*resolution < .5:
            continue
        candidate = free | mask
        if np.any(candidate[[0, -1]]) or np.any(candidate[:, [0, -1]]):
            continue
        if topology(candidate) != original_topology:
            continue
        free = candidate; used.append(index)
        yaw = float(np.arctan2(tangent[index, 1], tangent[index, 0]))
        accepted.append(dict(kind=kind, centerline_index=index, added_area_m2=area,
                             rectangle=[float(start[0]), float(start[1]), yaw, reach, lo, hi]))
        if len(accepted) == 3:
            break
    if not accepted:
        raise ValueError('No valid open-area variant for seed '+str(seed))
    result = {k: v.copy() for k, v in arrays.items()}
    result['occupancy'] = (free.astype(np.uint8)*255)
    result['origin'] = origin
    previous = np.asarray(arrays.get('runoffs', np.empty((0, 6)))).reshape(-1, 6)
    result['runoffs'] = np.concatenate([previous, np.asarray([r['rectangle'] for r in accepted])]).astype(np.float32)
    result['augmentation_seed'] = np.asarray(seed)
    result['augmentation_kind'] = np.asarray('target_open_1126_v1')
    assert np.all(free[base])
    assert topology(free) == original_topology
    return result, dict(openings=accepted, added_free_area_m2=float((free&~base).sum()*resolution**2),
                        original_free_cells_preserved=True, topology_before=list(original_topology),
                        topology_after=list(topology(free)), foreign_corridor_clearance_m=.30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    cli = p.parse_args(); os.chdir(SRC)
    import numpy as np
    dest = Path(cli.output_dir).resolve()
    if SRC/'installation' in dest.parents or dest.exists():
        raise ValueError('Use a new isolated directory outside live installation maps')
    sources = sorted(Path('installation/maps_real').glob('Practice1126b*.npz'))
    sources += sorted(Path('installation/maps_hall').glob('Course2025*.npz'))
    assert len(sources) == 4
    hashes = {str(p.resolve()): sha(p) for p in sources}
    report = dict(status='generating_not_used_in_training', sources=hashes, variants=[],
                  qualification='Known geometry augmentation; evaluation seeds unseen, but base geometry is trained. Static grid checks do not replace closed-loop evaluation.',
                  proposed_training_mix=dict(target=.20, hall=.35, short=.20, regular=.15, open=.10))
    dest.mkdir(parents=True)
    for split, seeds in [('train', [112601, 112602, 112603, 112604]), ('eval', [112691, 112692])]:
        (dest/split).mkdir()
        for source in sources:
            with np.load(source, allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in z.files}
            for seed in seeds:
                candidate, checks = make_variant(arrays, seed)
                name = source.stem+'_open'+str(seed)
                candidate['name'] = np.asarray(name)
                candidate['augmentation_source_sha256'] = np.asarray(hashes[str(source.resolve())])
                target = dest/split/(name+'.npz')
                np.savez_compressed(target, **candidate)
                for key in ('centerline', 'raceline', 'w_left', 'w_right', 'length'):
                    assert np.array_equal(candidate[key], arrays[key]), key
                report['variants'].append(dict(path=str(target), source=str(source.resolve()), split=split,
                                               seed=seed, sha256=sha(target), checks=checks))
                (dest/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
                print(split, name, checks['added_free_area_m2'], flush=True)
    assert all(sha(p) == h for p, h in hashes.items())
    report['status'] = 'generated_static_checks_passed_not_applied'
    report['source_hashes_unchanged'] = True
    (dest/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print('DONE', len(report['variants']), dest, flush=True)


if __name__ == '__main__':
    main()
