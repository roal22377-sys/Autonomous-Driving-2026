#!/usr/bin/env python3
"""Offline joint path/squared-speed NLP; experimental, never a deployment tool.

Periodic C4 quintic geometry, analytic curvature/steering derivatives, longitudinal energy
balance, steering/rate bounds and a provisional combined acceleration ellipse.
This quasi-steady bicycle approximation omits slip and actuator delay. Exported
speed/steering are NOT yet consumed by the production preview or teacher.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
SRC = ROOT / 'src/f1tenth'
sys.path[:0] = [str(SRC), str(SRC/'installation'), str(ROOT/'tools')]

import numpy as np
import torch
from scipy.interpolate import make_interp_spline
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from scipy.spatial import cKDTree

torch.set_num_threads(1)
DTYPE = torch.float64


def tensor(x):
    return torch.as_tensor(x, dtype=DTYPE)


class JointProblem:
    def __init__(self, center, normals, lower, upper, limits, arrays=None,
                 samples=3, clearance=.03, reserve=.98):
        self.center, self.normals = center, normals
        self.lower, self.upper, self.limits = lower, upper, limits
        self.n, self.samples = len(center), samples
        self.m = self.n * samples
        self.clearance = clearance
        self.reserve = reserve
        self.constraint_names=['ellipse_start','ellipse_end','steering','steering_rate',
                               'path_derivative','forward_progress']
        if arrays is not None:self.constraint_names.append('body_clearance')
        self.u = np.arange(self.m)/samples
        basis = make_interp_spline(np.arange(self.n+1),
                                   np.vstack([np.eye(self.n), np.eye(self.n)[:1]]),
                                   bc_type='periodic',k=5)
        self.matrices = [tensor(basis(self.u, nu=d)) for d in range(4)]
        direction = self.matrices[1] @ tensor(center)
        self.reference_direction = direction / torch.sqrt((direction*direction).sum(1))[:,None]
        self.arrays = arrays
        if arrays is not None:
            free = arrays['occupancy'] > 128
            resolution = float(arrays['resolution'])
            sdf = (distance_transform_edt(free)-distance_transform_edt(~free))*resolution
            self.sdf = tensor(sdf)
            self.origin = tensor(arrays['origin'][:2])
            self.resolution = resolution
        # Cover the body including its interior; a dense independent screen follows.
        xx, yy = np.meshgrid(np.linspace(-.31,.29,7), np.linspace(-.15,.15,5))
        self.body = tensor(np.stack([xx.ravel(), yy.ravel()],axis=1))

    def geometry(self, offset, matrices=None):
        matrices = self.matrices if matrices is None else matrices
        knots = tensor(self.center)+offset[:,None]*tensor(self.normals)
        p, d, dd = [b @ knots for b in matrices[:3]]
        norm = torch.sqrt((d*d).sum(1)+1e-16)
        tangent = d/norm[:,None]
        k = (d[:,0]*dd[:,1]-d[:,1]*dd[:,0])/norm**3
        return p, tangent, k, norm

    def body_clearance(self, p, tangent):
        normal = torch.stack([-tangent[:,1], tangent[:,0]],1)
        pts = (p[:,None,:]+tangent[:,None,:]*self.body[None,:,0,None]
               +normal[:,None,:]*self.body[None,:,1,None])
        uv = (pts-self.origin)/self.resolution-.5
        h,w = self.sdf.shape
        # Clamp before interpolation; explicit outside penalty prevents edge escape.
        x,y = uv[...,0].clamp(0,w-1.000001),uv[...,1].clamp(0,h-1.000001)
        ix,iy = torch.floor(x).long(),torch.floor(y).long()
        fx,fy = x-ix,y-iy
        val = ((1-fx)*(1-fy)*self.sdf[iy,ix]+fx*(1-fy)*self.sdf[iy,ix+1]
               +(1-fx)*fy*self.sdf[iy+1,ix]+fx*fy*self.sdf[iy+1,ix+1])
        outside = torch.relu(-uv[...,0])+torch.relu(uv[...,0]-(w-1))
        outside += torch.relu(-uv[...,1])+torch.relu(uv[...,1]-(h-1))
        val = val-outside*self.resolution
        # A smooth lower bound avoids switching the active body grid sample.
        return -.003*torch.logsumexp(-val/.003,dim=1)

    def values(self, z):
        off, q = z[:self.n], z[self.n:]
        p,tangent,k,norm = self.geometry(off)
        # Trapezoidal arc length and exact constant-acceleration segment travel time.
        ds = .5*(norm+torch.roll(norm,-1))/self.samples
        v = torch.sqrt(q)
        dt = 2*ds/(v+torch.roll(v,-1))
        # q is piecewise linear in the knot parameter. Check longitudinal
        # acceleration at BOTH segment endpoints; using only mean ds hid peaks.
        dqdu=(torch.roll(q,-1)-q)*self.samples
        ax = dqdu/(2*norm)
        ax_end = dqdu/(2*torch.roll(norm,-1))
        ay = q*k
        steer = torch.atan(self.limits['wheelbase']*k)
        knots=tensor(self.center)+off[:,None]*tensor(self.normals)
        d,dd,ddd=[b @ knots for b in self.matrices[1:]]
        cross=d[:,0]*dd[:,1]-d[:,1]*dd[:,0]
        kp=(d[:,0]*ddd[:,1]-d[:,1]*ddd[:,0])/norm**3
        kp-=3*cross*(d*dd).sum(1)/norm**5
        rate=self.limits['wheelbase']*kp/(1+(self.limits['wheelbase']*k)**2)*v/norm
        # Conservative ellipse uses asymmetric longitudinal drive/brake bounds.
        ax_scale = torch.where(ax>=0,self.limits['a_acc'],self.limits['a_brake'])
        ellipse = (ax/ax_scale)**2+(ay/self.limits['a_lat'])**2
        ellipse_end=(ax_end/ax_scale)**2+(torch.roll(ay,-1)/self.limits['a_lat'])**2
        forward = (tangent*self.reference_direction).sum(1)
        constraints = [self.reserve**2-ellipse,self.reserve**2-ellipse_end,
                       (self.reserve**2*self.limits['steer']**2-steer**2)/self.limits['steer']**2,
                       (self.reserve**2*self.limits['rate']**2-rate**2)/self.limits['rate']**2,
                       norm-.02,forward-.1]
        if self.arrays is not None:
            body_reserve=.015 if self.reserve<1 else 0.
            constraints.append((self.body_clearance(p,tangent)-self.clearance-body_reserve)/.1)
        return dt.sum(), torch.cat(constraints)

    def solve(self, initial_offset, maxiter=120, fixed=False):
        start = time.time()
        initial_offset = np.clip(initial_offset,self.lower,self.upper)
        with torch.no_grad():
            _,_,k,_ = self.geometry(tensor(initial_offset))
        q = np.minimum(self.limits['v_max']**2,
                       .7*self.limits['a_lat']/(np.abs(k.numpy())+1e-6))
        # Constant initial speed is feasible longitudinally, even in tight corners.
        q[:] = max(self.limits['v_min']**2,float(q.min()))
        x0 = np.r_[initial_offset,q]
        if fixed:
            with torch.no_grad():
                _,_,initial_k,_=self.geometry(tensor(initial_offset))
                maximum=float(torch.abs(torch.atan(self.limits['wheelbase']*initial_k)).max())
            if maximum>self.reserve*self.limits['steer']+1e-6:
                f,g=self.values(tensor(x0))
                return x0,dict(solver_success=False,
                    message='Fixed spline reconstruction exceeds steering bound; speed cannot repair geometry.',
                    iterations=0,seconds=time.time()-start,
                    maximum_kinematic_steering_rad=maximum,
                    model_lap_time_s=float(f),minimum_scaled_constraint=float(g.min()),
                    collocation_feasible=False)
        if fixed:
            free = np.arange(self.n,len(x0))
        else:
            free = np.arange(len(x0))
        base = tensor(x0)
        scales = np.r_[np.ones(self.n),np.full(self.m,16.)][free]
        scale_t = tensor(scales)
        cached = {}
        best = dict(x=None,time=float('inf'))
        def remember(x,y):
            if y[1:].min()>=-1e-6 and y[0]<best['time']:
                best.update(x=x.copy(),time=float(y[0]))
        def evaluate(x):
            if 'x' not in cached or not np.array_equal(x,cached['x']):
                xx = tensor(x).requires_grad_(True)
                def all_values(y):
                    z = base.index_copy(0,torch.as_tensor(free),y*scale_t)
                    f,g = self.values(z)
                    return torch.cat([f[None],g])
                y = all_values(xx)
                jac = torch.autograd.functional.jacobian(all_values,xx,vectorize=True,strategy='forward-mode')
                cached.update(x=x.copy(),y=y.detach().numpy(),jac=jac.detach().numpy())
                remember(x,cached['y'])
            return cached['y'],cached['jac']
        bounds = list(zip(self.lower,self.upper))+[(self.limits['v_min']**2,self.limits['v_max']**2)]*self.m
        free_bounds=[(bounds[i][0]/scales[j],bounds[i][1]/scales[j]) for j,i in enumerate(free)]
        # Restore feasibility before reducing time. Infeasible starts otherwise
        # consume the lap-time solver's iterations trying to repair sharp geometry.
        def feasibility_fg(x):
            xx=tensor(x).requires_grad_(True)
            z=base.index_copy(0,torch.as_tensor(free),xx*scale_t)
            _,g=self.values(z)
            loss=torch.relu(-g).square().sum()
            gradient=torch.autograd.grad(loss,xx)[0]
            return float(loss.detach()),gradient.detach().numpy()
        pre=minimize(feasibility_fg,x0[free]/scales,jac=True,method='L-BFGS-B',
                     bounds=free_bounds,
                     options=dict(maxiter=600,ftol=1e-13,gtol=1e-7))
        print('feasibility_stage',float(pre.fun),'elapsed',round(time.time()-start,2),flush=True)
        evaluate(pre.x)
        warm=pre.x
        for weight in (100.,1000.,10000.):
            def penalty_fg(x):
                xx=tensor(x).requires_grad_(True)
                z=base.index_copy(0,torch.as_tensor(free),xx*scale_t)
                f,g=self.values(z)
                loss=f+weight*torch.relu(-g).square().sum()
                gradient=torch.autograd.grad(loss,xx)[0]
                remember(x,torch.cat([f[None],g]).detach().numpy())
                return float(loss.detach()),gradient.detach().numpy()
            warm_result=minimize(penalty_fg,warm,jac=True,method='L-BFGS-B',bounds=free_bounds,
                                 options=dict(maxiter=250,ftol=1e-11,gtol=1e-6,maxls=40))
            warm=warm_result.x
            print('penalty_stage',weight,'elapsed',round(time.time()-start,2),flush=True)
        result = minimize(lambda x:(evaluate(x)[0][0],evaluate(x)[1][0]),warm,jac=True,
                          method='SLSQP',bounds=free_bounds,
                          constraints=[dict(type='ineq',fun=lambda x:evaluate(x)[0][1:],
                                            jac=lambda x:evaluate(x)[1][1:])],
                          options=dict(maxiter=maxiter,ftol=1e-8,disp=False))
        y,_=evaluate(result.x)
        chosen=best['x'] if best['x'] is not None and (y[1:].min() < -1e-6 or best['time']<y[0]) else result.x
        z = x0.copy();z[free]=chosen*scales
        f,g = self.values(tensor(z))
        info = dict(solver_success=bool(result.success),message=str(result.message),
                    iterations=int(result.nit),seconds=time.time()-start,
                    feasibility_warmstart_squared_violation=float(pre.fun),
                    returned_best_feasible=bool(not np.array_equal(chosen,result.x)),
                    model_lap_time_s=float(f),minimum_scaled_constraint=float(g.min()),
                    collocation_feasible=bool(g.min()>=-1e-5))
        info['constraint_minima']=dict(zip(self.constraint_names,g.detach().numpy().reshape(-1,self.m).min(1).tolist()))
        return z,info

    def dense_audit(self,z,factor=6):
        dense = JointProblem(self.center,self.normals,self.lower,self.upper,self.limits,
                             self.arrays,self.samples*factor,self.clearance,reserve=1.)
        q = np.interp(dense.u,np.r_[self.u,self.n],np.r_[z[self.n:],z[self.n]])
        zz = np.r_[z[:self.n],q]
        with torch.no_grad():
            f,g = dense.values(tensor(zz))
            p,t,k,_ = dense.geometry(tensor(z[:self.n]))
            clearance = float(dense.body_clearance(p,t).min()) if self.arrays is not None else None
        return dict(model_lap_time_s=float(f),minimum_scaled_constraint=float(g.min()),
                    constraint_minima=dict(zip(dense.constraint_names,g.numpy().reshape(-1,dense.m).min(1).tolist())),
                    feasible=bool(g.min()>=-1e-4),minimum_sampled_body_clearance_m=clearance), p.numpy(), q, k.numpy()


def circle_check():
    n=40; angle=np.arange(n)*2*np.pi/n; radius=3.
    c=radius*np.stack([np.cos(angle),np.sin(angle)],1)
    normal=-c/radius
    limits=dict(v_min=.5,v_max=8.,a_lat=6.,a_acc=2.9,a_brake=3.7,wheelbase=.34,steer=.5,rate=2.65)
    problem=JointProblem(c,normal,np.zeros(n),np.zeros(n),limits,samples=3)
    z,info=problem.solve(np.zeros(n),maxiter=80,fixed=True)
    exact=2*np.pi*radius/np.sqrt(problem.reserve*limits['a_lat']*radius)
    assert info['solver_success'] and info['collocation_feasible'],info
    assert abs(info['model_lap_time_s']/exact-1)<.003,(info,exact)
    # Central finite differences check automatic derivatives independently.
    z=tensor(z);z.requires_grad_(True)
    f,_=problem.values(z); grad=torch.autograd.grad(f,z)[0].detach().numpy()
    x=z.detach().numpy(); errors=[]
    for j in [3,n+7,n+31]:
        a=x.copy();b=x.copy();a[j]+=1e-5;b[j]-=1e-5
        fd=float((problem.values(tensor(a))[0]-problem.values(tensor(b))[0])/2e-5)
        errors.append(abs(fd-grad[j]))
    assert max(errors)<1e-6,errors
    return dict(status='pass',analytic_circle_time_s=exact,solve=info,
                objective_gradient_max_absolute_error=max(errors))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--tracks',nargs='*',default=[])
    p.add_argument('--spacing',type=float,default=.75)
    p.add_argument('--samples',type=int,default=6)
    p.add_argument('--maxiter',type=int,default=120)
    p.add_argument('--clearance',type=float,default=.03)
    args=p.parse_args();os.chdir(SRC)
    dest=Path(args.output_dir).resolve()
    if (SRC/'installation') in dest.parents:raise ValueError('Offline results only')
    dest.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',circle_check=circle_check(),tracks=[],
                optimizer_source_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                model='quasi-steady C4 quintic spline / longitudinal energy balance / analytic steering rate',
                limitations=['No global optimum certificate. No tire-slip or actuator-delay dynamics.',
                             'Combined acceleration ellipse and a_lat=6 are planning hypotheses, not measured hardware limits.',
                             'Sampled signed-distance body constraints need independent dense/static and closed-loop validation.',
                             'The fixed stored control is a spline reconstruction, not an exact original path. Infeasibility does not prove the original path infeasible.',
                             '2 percent force/steering reserve and 15mm body reserve are tested at 6x denser samples.',
                             'Production preview and teacher do not yet consume exported speed/steering. No policy trained here.'])
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    from utils.raceline import resample_closed,tangents_normals
    from map_to_track import cast
    for item in args.tracks:
        path=Path(item).resolve();source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        c,_=resample_closed(arrays['centerline'].astype(float),args.spacing)
        _,normal=tangents_normals(c)
        _,j=cKDTree(arrays['centerline']).query(c)
        free=arrays['occupancy']>128
        wl=np.minimum(arrays['w_left'][j],cast(free,float(arrays['resolution']),arrays['origin'][:2],c,normal,10.))
        wr=np.minimum(arrays['w_right'][j],cast(free,float(arrays['resolution']),arrays['origin'][:2],c,-normal,10.))
        # Body clearance enforced separately, these are broad center-position bounds.
        lo,hi=-(wr-.18),wl-.18
        if np.any(lo>hi):raise ValueError('Empty corridor')
        limits=dict(v_min=.5,v_max=8.,a_lat=6.,a_acc=2.9,a_brake=3.7,wheelbase=.34,steer=.5,rate=2.65)
        problem=JointProblem(c,normal,lo,hi,limits,arrays,args.samples,args.clearance)
        _,idx=cKDTree(arrays['raceline']).query(c)
        initial=((arrays['raceline'][idx]-c)*normal).sum(1)
        row=dict(source=str(path),source_sha256=source_hash,limits=limits,candidates={})
        # Matched fixed-geometry control, then full joint optimization; initial paths
        # do not restrict possible final shapes inside corridor/vehicle constraints.
        for name,offset,fixed in [('stored_geometry_control',initial,True),
                                  ('joint_from_stored',initial,False),
                                  ('joint_from_center',np.zeros(len(c)),False)]:
            z,info=problem.solve(offset,maxiter=args.maxiter,fixed=fixed)
            audit,xy,q,k=problem.dense_audit(z)
            info['dense_audit']=audit
            info['eligible_for_closed_loop_review']=bool(info['collocation_feasible'] and audit['feasible'])
            folder=dest/path.stem/name;folder.mkdir(parents=True)
            candidate=dict(arrays,raceline=xy.astype(np.float32),
                           joint_speed=np.sqrt(q).astype(np.float32),
                           joint_steering=np.arctan(limits['wheelbase']*k).astype(np.float32),
                           joint_offset=z[:problem.n],joint_center=c,
                           joint_q_collocation=z[problem.n:],joint_collocation_samples=np.array(problem.samples),
                           joint_spline_degree=np.array(5),
                           gen_sig=np.array('joint_mintime_experiment:'+source_hash+':'+name))
            np.savez_compressed(folder/'track.npz',**candidate)
            info['path']=str(folder/'track.npz');row['candidates'][name]=info
            print(path.stem,name,json.dumps(info),flush=True)
            report['in_progress_track']=row
            (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source_hash
        report['tracks'].append(row);report.pop('in_progress_track',None)
        (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['status']='complete_not_applied'
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',dest/'report.json',flush=True)


if __name__=='__main__':main()
