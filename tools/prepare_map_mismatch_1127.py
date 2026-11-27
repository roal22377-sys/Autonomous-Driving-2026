#!/usr/bin/env python3
"""Create offline physical-wall variants while preserving the nominal map/path.

Only the actual occupancy is warped. All pre-existing reference geometry and
widths remain nominal. A scan-localizer benchmark must explicitly load
nominal_occupancy rather than the physical occupancy as its reference map.
This is data preparation, not trained robustness or a continuous safety proof.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import distance_transform_edt, label, map_coordinates

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/f1tenth'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def warp(image,resolution,amplitude,seed,correlation=2.):
    h,w=image.shape
    yy,xx=np.indices((h,w),dtype=float)
    # Smooth displacement at a physical length scale, not independent pixel noise.
    gy=np.linspace(0,h-1,max(4,int(np.ceil(h*resolution/correlation))+1))
    gx=np.linspace(0,w-1,max(4,int(np.ceil(w*resolution/correlation))+1))
    rng=np.random.default_rng(seed)
    fields=[]
    for _ in range(2):
        controls=rng.normal(size=(len(gy),len(gx)))
        controls[[0,-1],:]=0.;controls[:,[0,-1]]=0.
        fields.append(RectBivariateSpline(gy,gx,controls)(np.arange(h),np.arange(w)))
    dy,dx=fields
    scale=amplitude/max(float(np.hypot(dx,dy).max()),1e-12)
    dx*=scale;dy*=scale
    # Bound deformation gradients: a modest smooth geometric warp, no folding.
    dy_y,dy_x=np.gradient(dy,resolution);dx_y,dx_x=np.gradient(dx,resolution)
    jac=(1+dx_x)*(1+dy_y)-dx_y*dy_x
    if jac.min()<.5:raise ValueError('Deformation too large for correlation length')
    actual=map_coordinates(image,[yy+dy/resolution,xx+dx/resolution],
                           order=0,mode='nearest').astype(image.dtype)
    return actual,dict(max_sampling_displacement_m=float(np.hypot(dx,dy).max()),
                       minimum_warp_jacobian_determinant=float(jac.min()),
                       changed_cells=int((actual!=image).sum()))


def screen(actual,arrays):
    free=actual>128;nominal=arrays['occupancy']>128
    resolution=float(arrays['resolution']);origin=arrays['origin'][:2]
    ij=np.floor((arrays['centerline']-origin)/resolution).astype(int)
    in_bounds=(ij[:,0]>=0)&(ij[:,0]<free.shape[1])&(ij[:,1]>=0)&(ij[:,1]<free.shape[0])
    if not in_bounds.all():return dict(pass_route_screen=False,reason='Nominal route out of grid')
    component,ncomponents=label(free)
    nominal_component,nominal_count=label(nominal)
    labels=component[ij[:,1],ij[:,0]]
    dist=distance_transform_edt(free)*resolution
    clearance=dist[ij[:,1],ij[:,0]]
    ok=bool((labels>0).all() and len(np.unique(labels))==1 and clearance.min()>=.18
            and ncomponents==nominal_count)
    return dict(pass_route_screen=ok,free_components=int(ncomponents),
                nominal_free_components=int(nominal_count),
                minimum_nominal_centerpoint_clearance_m=float(clearance.min()),
                note='Route connectivity/point-clearance raster screen only; no swept body or closed-loop certificate.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--amplitudes',type=float,nargs='+',default=[.05,.10])
    p.add_argument('--seeds',type=int,nargs='+',default=[731,947])
    p.add_argument('--manifest',help='Preparation inventory or development suite; training rows only')
    p.add_argument('--per-family',type=int,default=1,help='With manifest: shortest maps per family;0 selects every training map')
    p.add_argument('--physical-resolution',type=float,help='Upsample only physical grid to at most this resolution; retain original nominal map resolution')
    args=p.parse_args()
    if any(a<=0 or a>.2 for a in args.amplitudes):raise ValueError('Expected explicit small 0<a<=.2m hypotheses')
    dest=Path(args.output_dir).resolve()
    if SRC/'installation' in dest.parents:raise ValueError('Offline outputs only')
    dest.mkdir(parents=True,exist_ok=False)
    generation_path=Path(args.manifest).resolve() if args.manifest else SRC/'results/reference_lines_1126/limits_v3_seeded/report.json'
    if not generation_path.exists():
        generation_path=ROOT/'planning/1127_raceline/input_maps.json'
    generation=json.loads(generation_path.read_text())
    if args.manifest:
        rows=generation.get('train',generation.get('tracks',[]))
        assert rows and all(row['role']=='train' for row in rows),'Training-only deformation preparation'
        selected=[]
        for family in sorted({row['family'] for row in rows}):
            subset=sorted([r for r in rows if r['family']==family],key=lambda r:r['length_m'])
            selected.extend(subset[:args.per_family] if args.per_family else subset)
        generation={'tracks':[dict(source=row['path'],source_sha256=row['sha256']) for row in selected]}
    else:assert generation['status']=='complete'
    report=dict(status='running',base_split='production training families only' if args.manifest else 'four known training examples only',
                generation_sha256=sha(generation_path),variants=[],
                purpose='Stale nominal map/line versus changed physical walls, not ordinary sensor noise.',
                amplitude_note='Inverse warp displacement hypotheses, not measured barrier motion or guaranteed raster boundary displacement.',
                integration_requirements=['Policy preview must retain original centerline/raceline and plan.',
                    'Physical raycasting and wall collision use variant occupancy.',
                    'Actual scan localizer must use nominal_occupancy; existing benchmark must not export variant occupancy as its reference map.',
                    'Do not optimize a new line on each altered occupancy: that leaks the physical map into the nominal reference.',
                    'Independent validation seeds and unmodified reference evaluation remain required.'],
                no_policy_training=True)
    for item in generation['tracks']:
        source=Path(item['source'])
        if not source.is_absolute():source=ROOT/source
        before=sha(source);assert before==item['source_sha256']
        with np.load(source,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        nominal_res=float(arrays['resolution'])
        if args.physical_resolution is not None and args.physical_resolution<=0:raise ValueError('Positive physical resolution required')
        factor=max(1,int(np.ceil(nominal_res/args.physical_resolution))) if args.physical_resolution else 1
        physical_res=nominal_res/factor
        physical_base=np.repeat(np.repeat(arrays['occupancy'],factor,axis=0),factor,axis=1)
        for amplitude in args.amplitudes:
            for seed in args.seeds:
                actual,stats=warp(physical_base,physical_res,amplitude,seed)
                validation=screen(actual,dict(arrays,resolution=np.array(physical_res)))
                row=dict(source=str(source),source_sha256=before,amplitude_m=amplitude,
                         seed=seed,warp=stats,screen=validation,
                         physical_resolution_m=physical_res,nominal_resolution_m=nominal_res)
                if validation['pass_route_screen'] and stats['changed_cells']>0:
                    variant=dict(arrays,occupancy=actual,nominal_occupancy=arrays['occupancy'],
                                 resolution=np.array(physical_res),nominal_resolution=np.array(nominal_res),
                                 mismatch_amplitude_m=np.array(amplitude),mismatch_seed=np.array(seed),
                                 nominal_source_sha256=np.array(before))
                    target=dest/source.stem/('warp_%03d_s%d.npz'%(round(amplitude*1000),seed))
                    target.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(target,**variant)
                    with np.load(target,allow_pickle=False) as z:
                        for k in arrays:
                            if k not in ('occupancy','resolution'):assert np.array_equal(z[k],arrays[k]),k
                        assert np.array_equal(z['nominal_occupancy'],arrays['occupancy'])
                        assert float(z['nominal_resolution'])==nominal_res
                    row.update(status='prepared_not_trained',path=str(target),sha256=sha(target))
                else:row['status']='rejected_by_screen'
                report['variants'].append(row)
                (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        assert sha(source)==before
    report['status']='complete_not_trained'
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],prepared=sum(v['status']=='prepared_not_trained' for v in report['variants']),total=len(report['variants']))))


if __name__=='__main__':main()
