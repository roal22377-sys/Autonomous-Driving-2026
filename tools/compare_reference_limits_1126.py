#!/usr/bin/env python3
"""Reoptimize isolated raceline copies with matched bounds and old/new speed limits."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC/'installation')]


def refine_dense(*args, **kwargs):
    """Isolated single-thread probe: score the same dense line later saved/used.

    Reuse the exact existing optimizer and change only objective sampling.
    Restore module functions even on failure; never edit generator source files.
    """
    from utils import raceline as rl
    profile, lap = rl.speed_profile, rl.lap_time
    try:
        rl.speed_profile = lambda xy,*a,**kw: profile(rl.resample_closed(xy,.1)[0],*a,**kw)
        rl.lap_time = lambda xy,v: lap(rl.resample_closed(xy,.1)[0],v)
        return rl.refine_min_time(*args,**kwargs)
    finally:
        rl.speed_profile,rl.lap_time = profile,lap


def footprint_screen(xy, arrays):
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from utils.raceline import resample_closed, tangents_normals
    # Dense sampled body coverage, not a continuous swept-body certificate.
    xy,_ = resample_closed(xy,.02)
    tangent, normal = tangents_normals(xy)
    free = arrays['occupancy'] > 128
    res = float(arrays['resolution']); org = arrays['origin'][:2]
    u = np.linspace(-.31,.29,int(np.ceil(.60/(res/2)))+1)
    v = np.linspace(-.15,.15,int(np.ceil(.30/(res/2)))+1)
    uu,vv = np.meshgrid(u,v,indexing='ij')
    pts = xy[:,None,:]+tangent[:,None,:]*uu.ravel()[None,:,None]+normal[:,None,:]*vv.ravel()[None,:,None]
    ij = np.floor((pts-org)/res).astype(int)
    inside = (ij[...,0]>=0)&(ij[...,0]<free.shape[1])&(ij[...,1]>=0)&(ij[...,1]<free.shape[0])
    ok = np.zeros(inside.shape,bool)
    ok[inside] = free[ij[...,1][inside],ij[...,0][inside]]
    distance = distance_transform_edt(free)*res
    clear = np.zeros(inside.shape)
    clear[inside] = distance[ij[...,1][inside],ij[...,0][inside]]
    return dict(sampled_poses=len(xy),body_samples_per_pose=len(uu.ravel()),
                poses_touching_occupied_samples=int((~ok.all(axis=1)).sum()),
                minimum_body_grid_clearance_m=float(clear.min()),
                note='Sampled static body, no localization/tracking error or dynamic-slip envelope')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks',nargs='+',required=True)
    p.add_argument('--output-dir',required=True)
    cli=p.parse_args();os.chdir(SRC)
    import numpy as np
    from scipy.spatial import cKDTree
    from utils.raceline import resample_closed,tangents_normals,min_curvature_line,refine_min_time
    from utils import track_gen as tg
    from utils import track_hall as hall
    from map_to_track import cast
    from audit_reference_profile_1126 import screen
    dest=Path(cli.output_dir).resolve()
    if SRC/'installation' in dest.parents:
        raise ValueError('Write candidates outside live installation/maps directories')
    if dest.exists():
        raise FileExistsError('Use a new isolated output directory; never overwrite an experiment')
    dest.mkdir(parents=True)
    limits={'legacy':dict(tg.LAP_TIME_LIMITS),
            'current':dict(v_min=.5,v_max=8.,a_lat=6.,a_acc=2.9,a_brake=3.7)}
    limits['current_dense']=dict(limits['current'])
    limits['current_dense_best']=dict(limits['current'])
    report=dict(status='running',limits=limits,tracks=[],limitations=[
        'Same reconstructed bounds and starting offsets for old/new limits; no live map mutations.',
        'Stored maps do not retain original rough wall-width arrays. Bounds are reconstructed conservatively from occupancy and stored corridor widths.',
        'Reconstructed legacy line may differ from stored line; evaluate both controls.',
        'Static feasibility and computed ideal lap time are not closed-loop or real-car qualification.'])
    kwargs=dict(wheelbase=.34,max_steer=.5,max_rate=2.65)
    for item in cli.tracks:
        source=Path(item).resolve();source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        with np.load(source,allow_pickle=False) as z:
            arrays={k:z[k].copy() for k in z.files}
        center=arrays['centerline'].astype(float)
        copt,_=resample_closed(center,tg.OPT_DS)
        _,j=cKDTree(center).query(copt)
        _,normal=tangents_normals(center)
        free=arrays['occupancy']>128;res=float(arrays['resolution']);org=arrays['origin'][:2]
        wl=np.minimum(arrays['w_left'],cast(free,res,org,center,normal,10.))
        wr=np.minimum(arrays['w_right'],cast(free,res,org,center,-normal,10.))
        bl,br=wl[j].copy(),wr[j].copy()
        margin=tg.RACELINE_MARGIN
        if 'pitch' in arrays:
            # Same extra inner-corner/tracking allowance as the hall generator.
            k=hall._curvature(center)
            tight=hall._circ_smooth((np.abs(k)>1/1.5).astype(float),12)>.05
            inner=hall.INNER_EXTRA*tight[j];margin=hall.HALL_MARGIN
            bl=np.maximum(bl-hall.TRACK_ERR-np.where(k[j]>0,inner,0.),margin+.02)
            br=np.maximum(br-hall.TRACK_ERR-np.where(k[j]<0,inner,0.),margin+.02)
        _,initial=min_curvature_line(copt,bl,br,margin)
        row=dict(source=str(source),source_sha256=source_hash,margin=float(margin),
                 bounds_sha256=hashlib.sha256(np.stack([bl,br]).tobytes()).hexdigest(),
                 stored=dict(profile=screen(arrays['raceline'],limits['current'],**kwargs),
                             footprint=footprint_screen(arrays['raceline'],arrays)),variants={})
        initial_line=copt+initial[:,None]*tangents_normals(copt)[1]
        row['initial_current_profile']=dict(coarse=screen(initial_line,limits['current'],**kwargs),
            dense=screen(resample_closed(initial_line,tg.DENSE_DS)[0],limits['current'],**kwargs))
        solutions={}
        for name,lim in limits.items():
            dense_objective=name.startswith('current_dense')
            refine=refine_dense if dense_objective else refine_min_time
            seed=solutions['legacy'][1] if name=='current_dense_best' else initial
            line,offset,objective_time=refine(copt,bl,br,margin,seed.copy(),**lim)
            chosen_seed='legacy_solution' if name=='current_dense_best' else 'minimum_curvature'
            if name=='current_dense_best' and solutions['current_dense'][2]<objective_time:
                line,offset,objective_time=solutions['current_dense']
                chosen_seed='minimum_curvature'
            solutions[name]=(line,offset,objective_time)
            dense,_=resample_closed(line,tg.DENSE_DS)
            candidate=dict(arrays);candidate['raceline']=dense.astype(np.float32)
            candidate['gen_sig']=np.array('isolated_limits_audit:'+source_hash+':'+name)
            candidate['line_audit_source_sha256']=np.array(source_hash)
            if 'line_rmin' in arrays:
                candidate['line_rmin']=np.array(hall.min_radius(dense))
            folder=dest/name;folder.mkdir(exist_ok=True);target=folder/source.name
            np.savez_compressed(target,**candidate)
            with np.load(target) as z:
                for k in arrays:
                    if k not in ('raceline','gen_sig','line_rmin'):
                        assert np.array_equal(z[k],arrays[k]),k
                saved=z['raceline'].copy()
            result=dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                        objective_seconds=float(objective_time),
                        objective_spacing_m=.1 if dense_objective else tg.OPT_DS,
                        chosen_seed=chosen_seed,
                        coarse_current_profile=screen(line,limits['current'],**kwargs),
                        current_profile=screen(saved,limits['current'],**kwargs),
                        footprint=footprint_screen(saved,arrays))
            row['variants'][name]=result
            if dense_objective:
                assert abs(result['current_profile']['lap_time_s']-objective_time)<1e-4
            if name=='current_dense_best':
                assert result['current_profile']['lap_time_s']<=row['variants']['legacy']['current_profile']['lap_time_s']+1e-4
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
        report['tracks'].append(row);(dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(source.name,{k:v['current_profile']['lap_time_s'] for k,v in row['variants'].items()},flush=True)
    report['status']='complete';(dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',dest/'report.json',flush=True)


if __name__=='__main__':main()
