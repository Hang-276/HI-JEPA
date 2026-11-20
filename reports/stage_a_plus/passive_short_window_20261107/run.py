"""Passive unlabeled calibration for the frozen strong 192-step VAR task."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from stats import design_cutoff, summarize

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'hidden_boundary_cpu_20261107'
SOURCES = ('CONFIG.json','PROTOCOL.md','run.py','stats.py',
           '../hidden_boundary_cpu_20261107/run.py','../hidden_boundary_cpu_20261107/rule.py',
           '../hidden_boundary_cpu_20261107/score.py','../short_window_ceiling_20261107/RESULTS.md')
sys.path.insert(0,str(BASE))
spec = importlib.util.spec_from_file_location('_passive_frozen_var',BASE/'run.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
METHODS = ('oracle','passive','known_distance')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((HERE/name).read_text())


def sha(name):
    return hashlib.sha256((HERE/name).read_bytes()).hexdigest()


def save(name,obj):
    with (HERE/name).open('x') as out:
        json.dump(obj,out,separators=(',',':'),allow_nan=False,ensure_ascii=False)
        out.write('\n')


def rng(seed,ix,start,role):
    return np.random.default_rng(np.random.SeedSequence([seed,ix,start,role]))


def signs(seed,ix,start,shape,role):
    return 2*rng(seed,ix,start,role).integers(0,2,shape,dtype=np.int8)-1


def alignment(residual,u,beta):
    c,s = np.cos(beta),np.sin(beta)
    z1,z2 = c*residual[...,1]+s*residual[...,2], -s*residual[...,1]+c*residual[...,2]
    w2 = (u==3).astype(np.int8)-(u==4)
    w3 = (u==5).astype(np.int8)-(u==6)
    return z1,z2,w2*z2-w3*z1,np.abs(w2)+np.abs(w3)


def fit_panel(x,u,multiplier):
    """Both inputs are observed and contain an explicit trajectory axis."""
    panels,paths,frames,dim = x.shape
    assert dim==4 and u.shape==(panels,paths,frames-1)
    prev = x[:,:,:-1].reshape(panels,-1,4)
    after = x[:,:,1:].reshape(panels,-1,4)
    transposed = prev.swapaxes(1,2)
    gram = transposed@prev
    coef = np.linalg.solve(gram,transposed@after)
    residual = (after-prev@coef).reshape(panels,paths,frames-1,4)
    direction = coef[:,0,1:3]*np.array([-1.,1.])
    length2 = (direction**2).sum(1)
    assert (length2>0).all()
    beta = math.pi/4-np.arctan2(direction[:,1],direction[:,0])

    # Cluster scores for the two drift coordinates that determine the boundary.
    basis = np.broadcast_to(np.eye(4)[:,0,None],(panels,4,1))
    leverage = (prev@np.linalg.solve(gram,basis)).reshape(panels,paths,frames-1)
    cluster = (leverage[...,None]*residual[...,1:3]).sum(2)
    jac = -direction[:,::-1]/length2[:,None]
    angle_effect = np.einsum('bei,bi->be',cluster,jac)
    angle_radius = multiplier*np.sqrt((angle_effect**2).sum(1)*paths/(paths-1))

    neutral = u==0
    neutral_count = neutral.sum((1,2))
    sigma = np.sqrt(((residual**2).sum(3)*neutral).sum((1,2))/(4*neutral_count))
    w1 = (u==1).astype(np.int8)-(u==2)
    w4 = (u==7).astype(np.int8)-(u==8)
    unaffected = (np.abs(w1)+np.abs(w4)).sum((1,2))
    tau = (w1*residual[...,0]+w4*residual[...,3]).sum((1,2))/(sigma*unaffected)
    assert (sigma>0).all() and np.isfinite(tau).all()
    _,_,signed,active = alignment(residual/sigma[:,None,None,None],u,beta[:,None,None])
    m = active.sum(2)
    sq_amplitude = ((signed.sum(2)**2/np.maximum(m,1))-(m>0)).sum(1)/m.sum(1)
    sine = np.sqrt(np.maximum(sq_amplitude,0))/np.maximum(tau,1e-12)
    valid = (tau>0)&(sine>0)&(sine<math.sin(math.pi/4))
    distance = np.arcsin(np.where(valid,sine,0))
    return dict(a=coef.swapaxes(1,2),beta=beta,sigma=sigma,tau=tau,distance=distance,
                angle_radius=angle_radius,valid=valid,neutral_count=neutral_count)


def likelihood_ratio(x,u,a,beta,sigma,tau,distance):
    residual = (x[1:]-np.einsum('tbi,bji->tbj',x[:-1],a))/sigma[:,None]
    z1,z2,signed,_ = alignment(residual,u,beta)
    extra = ((u==10)*z1*z2-(u==11)*z1*z2).sum(0)
    return 2*tau*np.sin(distance)*signed.sum(0)+tau*np.sin(2*distance)/(1+tau)*extra


def eval_side(seed,ix,start,count):
    return signs(seed,ix,start,count,7)


def one_batch(cfg,seed,ix,start,count,paths):
    a,beta = generator.hidden(cfg,seed,ix,-1,start,count)
    strong = cfg['strong']
    sigma = math.sqrt(1-cfg['rho']**2)
    tau = strong['tau']
    distance = math.radians(strong['distance_deg'])
    side_cal = signs(seed,ix,start,(count,paths),5)
    panel_cell = dict(**strong)
    panel_cell['n'] = cfg['calibration_transitions_per_path']
    xx,uu = generator.trajectory(rng(seed,ix,start,6),np.repeat(a,paths,axis=0),np.repeat(beta,paths),
                                 panel_cell,side_cal.ravel(),sigma)
    xx = xx.reshape(len(xx),count,paths,4).transpose(1,2,0,3)
    uu = uu.reshape(len(uu),count,paths).transpose(1,2,0)
    fit = fit_panel(xx,uu,cfg['cluster_normal_multiplier'])
    side = eval_side(seed,ix,start,count)
    x,u = generator.trajectory(rng(seed,ix,start,8),a,beta,strong,side,sigma)
    oracle = likelihood_ratio(x,u,a,beta,np.full(count,sigma),np.full(count,tau),distance)
    magnitude = np.maximum(fit['tau'],0)
    common = (x,u,fit['a'],fit['beta'],fit['sigma'],magnitude)
    passive = likelihood_ratio(*common,fit['distance'])
    known = likelihood_ratio(*common,distance)
    values = np.column_stack((oracle,passive,known))
    params = np.column_stack((fit['beta'],fit['angle_radius'],fit['sigma'],fit['tau'],
                              fit['distance'],fit['neutral_count'],fit['valid']))
    assert np.isfinite(values).all() and np.isfinite(params).all()
    return values,params


def pilot(cfg):
    seed = cfg['pilot_seed']
    count = 8
    sigma = math.sqrt(1-cfg['rho']**2)
    d,tau = math.radians(cfg['strong']['distance_deg']),cfg['strong']['tau']
    errors = []
    for ix in range(2):
        values,params = one_batch(cfg,seed,ix,0,count,16)
        assert values.shape==(count,3) and params.shape==(count,7)
        a,beta = generator.hidden(cfg,seed,ix,-1,0,count)
        side = eval_side(seed,ix,0,count)
        x,u = generator.trajectory(rng(seed,ix,0,8),a,beta,cfg['strong'],side,sigma)
        selected,gap = generator.population_set(a,beta+side*d,cfg['lags'])
        assert (selected==np.where(side<0,12,10)).all() and (gap>0).all()
        residual = (x[1:]-np.einsum('tbi,bji->tbj',x[:-1],a))/sigma
        mean,scale = np.zeros((13,4)),np.ones((13,4))
        mean[1:9:2],mean[2:9:2] = tau*np.eye(4),-tau*np.eye(4)
        scale[9+np.arange(4),np.arange(4)] = math.sqrt(1+tau)
        laws = []
        for sign in (1,-1):
            rot = generator.rotations(1,2,beta+sign*d)
            eps = np.einsum('tbj,bij->tbi',residual,rot)
            laws.append(-.5*(((eps-mean[u])/scale[u])**2).sum((0,2)))
        error = float(np.max(abs(values[:,0]-(laws[0]-laws[1]))))
        assert error<1e-12
        errors.append(error)
    save('PILOT.json',dict(utc=utc(),passed=True,lock_sha256=sha('LOCK.json'),generic_llr_errors=errors,
                           checked=['passive fit shape','separate seeded side streams','population set','two Gaussian likelihoods']))


def simulate(cfg,seed,ix,n):
    batch = cfg['batch']
    values,params = [],[]
    for start in range(0,n,batch):
        y,p = one_batch(cfg,seed,ix,start,min(n-start,batch),cfg['calibration_paths'])
        values.append(y)
        params.append(p)
    return np.concatenate(values),np.concatenate(params)


def design(cfg,ix):
    assert read('PILOT.json')['passed']
    started = time.monotonic()
    values,_ = simulate(cfg,cfg['design_seed'],ix,cfg['design_reps_per_dynamics'])
    scores = {m:design_cutoff(values[:,j].tolist(),values[:,0].tolist(),cfg['design_wrong_target'])
              for j,m in enumerate(METHODS)}
    save(f'DESIGN_{ix}.json',dict(utc=utc(),lock_sha256=sha('LOCK.json'),thresholds=scores,
                                 seconds=time.monotonic()-started,n=len(values)))


def predict(cfg,ix):
    d = read(f'DESIGN_{ix}.json')
    assert d['lock_sha256']==sha('LOCK.json')
    started = time.monotonic()
    scores,params = simulate(cfg,cfg['validation_seed'],ix,cfg['validation_reps_per_dynamics'])
    filename = f'PREDICT_{ix}.npz'
    with (HERE/filename).open('xb') as out:
        np.savez_compressed(out,scores=scores,params=params)
    save(f'PREDICT_{ix}.json',dict(utc=utc(),lock_sha256=sha('LOCK.json'),design_sha256=sha(f'DESIGN_{ix}.json'),
                                  predict_sha256=sha(filename),seconds=time.monotonic()-started))


def parameter_summary(x):
    return dict(median=float(np.median(x)),q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)))


def reveal(cfg,ix):
    name = f'PREDICT_{ix}'
    p = read(name+'.json')
    assert p['predict_sha256']==sha(name+'.npz') and p['design_sha256']==sha(f'DESIGN_{ix}.json')
    with np.load(HERE/(name+'.npz'),allow_pickle=False) as ar:
        values,params = ar['scores'],ar['params']
    n,batch,seed = cfg['validation_reps_per_dynamics'],cfg['batch'],cfg['validation_seed']
    panels = [(start,min(n-start,batch)) for start in range(0,n,batch)]
    side = np.concatenate([eval_side(seed,ix,start,count) for start,count in panels])
    beta = np.concatenate([generator.hidden(cfg,seed,ix,-1,start,count)[1] for start,count in panels])
    rows = [dict(side=int(s),**dict(zip(METHODS,map(float,v)))) for s,v in zip(side,values,strict=True)]
    out = summarize(rows,read(f'DESIGN_{ix}.json')['thresholds'],family_alpha=cfg['family_alpha'])
    bhat,radius,sigma,tau,distance,neutral,valid = params.T
    out['calibration'] = dict(beta_abs_error_deg=parameter_summary(np.degrees(abs(beta-bhat))),
                              beta_interval_radius_deg=parameter_summary(np.degrees(radius)),
                              beta_cluster_interval_empirical_coverage=float((abs(beta-bhat)<=radius).mean()),
                              sigma=parameter_summary(sigma),tau=parameter_summary(tau),
                              distance_deg=parameter_summary(np.degrees(distance[valid>0])),
                              distance_invalid=int((valid==0).sum()),neutral_transitions=parameter_summary(neutral))
    save(f'SUMMARY_{ix}.json',dict(**out,utc=utc(),prediction_sha256=sha(name+'.npz'),
                                   design_sha256=sha(f'DESIGN_{ix}.json'),dynamics=cfg['psi_deg'][ix]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('lock','pilot','design','predict','reveal'))
    parser.add_argument('dynamics',type=int,nargs='?',default=0)
    args = parser.parse_args()
    hashes = {s:sha(s) for s in SOURCES}
    if args.phase=='lock':
        save('LOCK.json',dict(utc=utc(),sources=hashes,numpy=np.__version__,cpu_threads=1))
        return
    assert read('LOCK.json')['sources']==hashes
    cfg = read('CONFIG.json')
    assert len({cfg[k] for k in ('pilot_seed','design_seed','validation_seed')})==3
    assert cfg['validation_wrong_budget']==.05 and args.dynamics in (0,1)
    if args.phase=='pilot':
        pilot(cfg)
    else:
        {'design':design,'predict':predict,'reveal':reveal}[args.phase](cfg,args.dynamics)


if __name__=='__main__':
    main()
