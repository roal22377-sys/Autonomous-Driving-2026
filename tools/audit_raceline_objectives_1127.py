#!/usr/bin/env python3
"""Isolated comparison of path objectives and minimum-time initializations.

No production maps, reward settings or model weights are modified. Computed
times are simplified planning estimates, never closed-loop lap-time claims.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'src/f1tenth'
sys.path[:0]=[str(SRC),str(SRC/'installation'),str(ROOT/'tools')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--refine',action='store_true')
    cli=parser.parse_args();os.chdir(SRC)
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    from utils.raceline import resample_closed,tangents_normals,min_curvature_line,curvature
    from utils import track_gen as tg,track_hall as hall
    from map_to_track import cast
    from compare_reference_limits_1126 import refine_dense,footprint_screen
    from audit_reference_profile_1126 import screen
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=Path(cli.output_dir)
    if dest.exists():raise FileExistsError('Preserve earlier audit')
    dest.mkdir(parents=True)
    generation_path=Path('results/reference_lines_1126/limits_v3_seeded/report.json')
    generation=json.loads(generation_path.read_text());assert generation['status']=='complete'
    limits=dict(v_min=.5,v_max=8.,a_lat=6.,a_acc=2.9,a_brake=3.7)
    report=dict(status='running',limits=limits,source_generation_sha256=hashlib.sha256(generation_path.read_bytes()).hexdigest(),tracks=[],
                qualification='KnownPractice/2025trainingexamplesonly. Truepolyline-shortestobjective,minimum-curvatureproxy,storedlineandoptionaldensermulti-starttime-refinementcomparedwithsamebounds/occupancy. Simplifiedtimeprofileuses1msmoothedcurvatureandindependentaccel/brakelimits;kinematicandstaticfootprintscreensareNOTsufficientdynamicfeasibility. No policytraining,live-mapreplacement,realcaroperationorglobaloptimalityclaim.')
    for item in generation['tracks']:
        path=Path(item['source']);source_hash=hashlib.sha256(path.read_bytes()).hexdigest();assert source_hash==item['source_sha256']
        with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        center=a['centerline'].astype(float);copt,_=resample_closed(center,.5)
        _,indices=cKDTree(center).query(copt);_,normal=tangents_normals(center)
        free=a['occupancy']>128;res=float(a['resolution']);org=a['origin'][:2]
        wl=np.minimum(a['w_left'],cast(free,res,org,center,normal,10.))
        wr=np.minimum(a['w_right'],cast(free,res,org,center,-normal,10.))
        left,right=wl[indices].copy(),wr[indices].copy();margin=tg.RACELINE_MARGIN
        if 'pitch'in a:
            k=hall._curvature(center);tight=hall._circ_smooth((abs(k)>1/1.5).astype(float),12)>.05
            inner=hall.INNER_EXTRA*tight[indices];margin=hall.HALL_MARGIN
            left=np.maximum(left-hall.TRACK_ERR-np.where(k[indices]>0,inner,0.),margin+.02)
            right=np.maximum(right-hall.TRACK_ERR-np.where(k[indices]<0,inner,0.),margin+.02)
        lo,hi=-(right-margin),left-margin;bad=lo>hi;lo[bad]=hi[bad]=0
        _,n=tangents_normals(copt)
        def shortest_cost(offset):
            p=copt+offset[:,None]*n;d=np.roll(p,-1,axis=0)-p
            lengths=np.sqrt((d*d).sum(1)+1e-14);u=d/lengths[:,None]
            gradient=((np.roll(u,1,axis=0)-u)*n).sum(1)
            return float(lengths.sum()),gradient
        zero=np.clip(np.zeros(len(copt)),lo,hi)
        _,mc=min_curvature_line(copt,left,right,margin)
        result=minimize(shortest_cost,zero,jac=True,method='L-BFGS-B',bounds=list(zip(lo,hi)),options=dict(maxiter=2000,ftol=1e-12,gtol=1e-9))
        shortest=np.clip(result.x,lo,hi)
        _,j=cKDTree(a['raceline']).query(copt)
        stored_offset=np.clip(((a['raceline'][j]-copt)*n).sum(1),lo,hi)
        offsets={'centerline':zero,'min_curvature':mc,'true_shortest_polyline':shortest}
        paths={'stored':a['raceline'].astype(float)}
        for name,off in offsets.items():paths[name]=copt+off[:,None]*n
        if cli.refine:
            for name,off in dict(centerline=zero,min_curvature=mc,shortest=shortest,stored=stored_offset).items():
                line,_,_=refine_dense(copt,left,right,margin,off.copy(),bump_spacing=1.5,bump_width=1.25,iters=100,**limits)
                paths['time_from_'+name]=line
        row=dict(source=str(path),source_sha256=source_hash,shortest_solver_success=bool(result.success),shortest_solver_message=str(result.message),
                 bounds_sha256=hashlib.sha256(np.stack([lo,hi]).tobytes()).hexdigest(),margin=float(margin),candidates={})
        for name,line in paths.items():
            dense,_=resample_closed(line,.1);profile=screen(dense,limits,wheelbase=.34,max_steer=.5,max_rate=2.65)
            length=float(np.linalg.norm(np.roll(dense,-1,axis=0)-dense,axis=1).sum())
            raw_k=curvature(dense);foot=footprint_screen(dense,a)
            output=dest/path.stem/name;output.mkdir(parents=True)
            arrays=dict(a,raceline=dense.astype(np.float32),gen_sig=np.array('objective_audit:'+source_hash+':'+name))
            np.savez_compressed(output/'track.npz',**arrays)
            row['candidates'][name]=dict(length_m=length,current_profile=profile,raw_max_kinematic_steer_rad=float(abs(np.arctan(.34*raw_k)).max()),
                                        raw_steer_limit_fraction=float((abs(np.arctan(.34*raw_k))>.5).mean()),footprint=foot,path=str(output/'track.npz'))
        fig,ax=plt.subplots(figsize=(10,6));extent=[org[0],org[0]+free.shape[1]*res,org[1],org[1]+free.shape[0]*res]
        ax.imshow(a['occupancy'],origin='lower',extent=extent,cmap='gray',vmin=0,vmax=255)
        show=['centerline','stored','min_curvature','true_shortest_polyline']
        if cli.refine:
            best=min((key for key in paths if key.startswith('time_from_')),key=lambda key:row['candidates'][key]['current_profile']['lap_time_s']);show.append(best)
        for name in show:
            line=paths[name];metric=row['candidates'][name];closed=np.vstack([line,line[0]])
            ax.plot(closed[:,0],closed[:,1],label=f"{name}: {metric['length_m']:.2f}m, proxy {metric['current_profile']['lap_time_s']:.2f}s",lw=1.4)
        ax.set_aspect('equal');ax.set_title(path.stem+' | path objectives (planning estimates)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(dest/(path.stem+'.png'),dpi=160);plt.close(fig)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source_hash
        report['tracks'].append(row);(dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(path.stem,{name:dict(length=round(x['length_m'],3),proxy_time=round(x['current_profile']['lap_time_s'],3),static_collisions=x['footprint']['poses_touching_occupied_samples']) for name,x in row['candidates'].items()},flush=True)
    report['status']='complete_not_applied';(dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
