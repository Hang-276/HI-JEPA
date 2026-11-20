"""CPU-only paired test of observational and known selection boundaries."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from rule import calibrate, chi8_radius, decide, scores
from score import summarize

HERE = Path(__file__).resolve().parent
SOURCES = ('CONFIG.json','PROTOCOL.md','run.py','rule.py','score.py',
           '../phase2_cpu_20261107/CONFIG.json','../phase2_boundary_20261107/BOUNDARY.md')


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(name):
    return hashlib.sha256((HERE/name).read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE/name).read_text())


def save(name,value):
    with (HERE/name).open('x') as out:
        json.dump(value,out,separators=(',',':'),ensure_ascii=False,allow_nan=False)
        out.write('\n')


def rng_for(seed,ix,side,start,stream):
    return np.random.default_rng(np.random.SeedSequence([seed,ix,(side+1)//2,start,stream]))


def rotations(i,j,angle):
    angle = np.asarray(angle)
    r = np.broadcast_to(np.eye(4),angle.shape+(4,4)).copy()
    c,s = np.cos(angle),np.sin(angle)
    r[...,i,i] = r[...,j,j] = c
    r[...,i,j],r[...,j,i] = s,-s
    return r


def hidden(cfg,seed,ix,side,start,n):
    phi = np.radians(rng_for(seed,ix,side,start,0).uniform(*cfg['hidden_phi_deg'],n))
    w = rotations(1,2,phi)
    psi = math.radians(cfg['psi_deg'][ix])
    a = cfg['rho']*w@rotations(0,1,psi)@w.swapaxes(1,2)
    beta = math.pi/4-phi
    return a,beta


def calibration_data(rng,a,count,sigma):
    initial = rng.standard_normal((len(a),count,4))
    output = initial@a[:,1:3].swapaxes(1,2)+sigma*rng.standard_normal((len(a),count,2))
    return initial,output


def trajectory(rng,a,beta,cell,side,sigma):
    n,tau = cell['n'],cell['tau']
    theta = beta+side*math.radians(cell['distance_deg'])
    r = rotations(1,2,theta)
    u = rng.integers(0,13,(n,len(a)),dtype=np.int8)
    mean = np.zeros((13,4))
    mean[1:9:2] = sigma*tau*np.eye(4)
    mean[2:9:2] = -sigma*tau*np.eye(4)
    sd = np.full((13,4),sigma)
    sd[9+np.arange(4),np.arange(4)] *= math.sqrt(1+tau)
    noise = rng.standard_normal((n,len(a),4))*sd[u]+mean[u]
    noise[:,:,1:3] = np.einsum('tbi,bij->tbj',noise[:,:,1:3],r[:,1:3,1:3])
    x = np.empty((n+1,len(a),4))
    x[0] = math.sqrt(1+(2*tau*tau+tau)/13)*rng.standard_normal((len(a),4))
    for t in range(n):
        x[t+1] = np.einsum('bi,bji->bj',x[t],a)+noise[t]
    return x,u


def population_set(a,theta,lags):
    physical = rotations(1,2,theta)
    energy = np.mean([2*np.eye(4)-p-p.swapaxes(1,2)
                      for p in (np.linalg.matrix_power(a,h) for h in lags)],axis=0)
    slow = np.diagonal(physical@energy@physical.swapaxes(1,2),axis1=1,axis2=2)
    ranking = np.argsort(slow,axis=1)
    mask = (1<<ranking[:,:2]).sum(1)
    sorted_scores = np.take_along_axis(slow,ranking,axis=1)
    return mask,sorted_scores[:,2]-sorted_scores[:,1]


def fit_observed(cfg,seed,ix,side,start,count,a,sigma):
    initial,output = calibration_data(rng_for(seed,ix,side,start,1),a,count,sigma)
    return calibrate(initial,output,sigma,cfg['alpha_beta'])


def compare(cfg,x,u,fit,sigma,beta):
    drift,ginv,center,width = fit
    direction,penalty,sample,active = scores(x,u,drift,ginv,sigma,cfg['alpha_drift'],cfg['alpha_score'])
    shared_radius = penalty+sample
    estimated,est_projection,est_radius = decide(direction,center,width,shared_radius,active)
    oracle,ref_projection,ref_radius = decide(direction,beta,np.zeros_like(beta),shared_radius,active)
    return (estimated,oracle,active,penalty,est_projection,est_radius,ref_projection,ref_radius)


def pilot(cfg):
    sigma = math.sqrt(1-cfg['rho']**2)
    seed = cfg['pilot_seed']
    checks = []
    for ix in range(2):
        for side in (-1,1):
            a,beta = hidden(cfg,seed,ix,side,0,8)
            direction = a[:,1:3,0]*np.array([-1.,1.])
            np.testing.assert_allclose(math.pi/4-np.arctan2(direction[:,1],direction[:,0]),beta,atol=1e-14)
            target,gap = population_set(a,beta+side*math.radians(7),cfg['lags'])
            assert (target==(12 if side<0 else 10)).all() and (gap>0).all()
            x,u = trajectory(rng_for(seed,ix,side,0,2),a,beta,
                             dict(n=96,tau=.8,distance_deg=7.),side,sigma)
            fit = fit_observed(cfg,seed,ix,side,0,256,a,sigma)
            drift,ginv,center,width = fit
            fast,p,_,m = scores(x,u,drift,ginv,sigma,cfg['alpha_drift'],cfg['alpha_score'])
            observed = []
            for t,context in enumerate(u[:,0]):
                if context not in (3,4,5,6):
                    continue
                value = x[t+1,0,1:3]-drift[0]@x[t,0]
                if context in (5,6):
                    value = value[::-1]*np.array([1.,-1.])
                observed.append((-1 if context in (4,6) else 1)*value/sigma)
            np.testing.assert_allclose(fast[0],np.mean(observed,axis=0),atol=1e-13,rtol=0)
            assert m[0]==len(observed)
            truth,_,_,_ = scores(x,u,a[:,1:3],ginv,sigma,cfg['alpha_drift'],cfg['alpha_score'])
            delta = drift-a[:,1:3]
            stat = np.einsum('bki,bij,bkj->b',delta,np.linalg.inv(ginv),delta)/sigma**2
            inside = stat<=chi8_radius(cfg['alpha_drift'])**2
            assert (np.linalg.norm(fast-truth,axis=1)[inside]<=p[inside]+1e-12).all()
            decisions = compare(cfg,x,u,fit,sigma,beta)
            assert len(decisions[0])==8 and np.isin(decisions[0],(-1,0,1)).all()
            checks.append(dict(dynamics=ix,side=side,known_boundary_identity=True,
                               projection_identity=True,covered_drift_bound=True))
    save('PILOT.json',dict(utc=utc(),passed=True,lock_sha256=digest('LOCK.json'),checks=checks))


def maybe(value):
    return float(value) if np.isfinite(value) else None


def predict(cfg,ix):
    started = time.monotonic()
    pg = read('PILOT.json')
    assert pg['passed'] and pg['lock_sha256']==digest('LOCK.json')
    if ix:
        assert all(read(f'SUMMARY_{ix-1}.json')['passed_by_dynamics'].values())
    rows = []
    sigma = math.sqrt(1-cfg['rho']**2)
    seed = cfg['seed']
    for side in (-1,1):
        for start in range(0,cfg['replications_per_side'],cfg['batch']):
            count = min(cfg['batch'],cfg['replications_per_side']-start)
            a,beta = hidden(cfg,seed,ix,side,start,count)
            fit = fit_observed(cfg,seed,ix,side,start,cfg['calibration_pairs_per_replication'],a,sigma)
            for ci,cell in enumerate(cfg['cells']):
                x,u = trajectory(rng_for(seed,ix,side,start,2+ci),a,beta,cell,side,sigma)
                estimated,oracle,active,p,ep,er,op,rr = compare(cfg,x,u,fit,sigma,beta)
                for j in range(count):
                    rows.append(dict(panel=(side+1)//2,rep=start+j,cell=cell['name'],
                                     estimated=int(estimated[j]),oracle=int(oracle[j]),active=int(active[j]),
                                     beta_hat_deg=math.degrees(fit[2][j]),beta_radius_deg=maybe(np.degrees(fit[3][j])),
                                     drift_penalty=float(p[j]),estimated_projection=float(ep[j]),
                                     estimated_radius=maybe(er[j]),oracle_projection=float(op[j]),oracle_radius=float(rr[j])))
    save(f'PREDICT_{ix}.json',dict(utc=utc(),lock_sha256=digest('LOCK.json'),rows=rows,
                                  elapsed_seconds=time.monotonic()-started))


def reveal(cfg,ix):
    p = read(f'PREDICT_{ix}.json')
    assert p['lock_sha256']==digest('LOCK.json')
    truth = {}
    settings = cfg['psi_deg'][ix]
    name = f'psi_{settings:g}'
    for side in (-1,1):
        for start in range(0,cfg['replications_per_side'],cfg['batch']):
            count = min(cfg['batch'],cfg['replications_per_side']-start)
            a,beta = hidden(cfg,cfg['seed'],ix,side,start,count)
            theta = beta+side*math.radians(7)
            target,gap = population_set(a,theta,cfg['lags'])
            assert (target==(12 if side<0 else 10)).all() and (gap>0).all()
            for j in range(count):
                truth[((side+1)//2,start+j)] = dict(side=side,dynamics=name,beta_true_deg=math.degrees(beta[j]),
                                                   physical_top2_mask=int(target[j]),physical_gap=float(gap[j]))
    labels = [dict(panel=k[0],rep=k[1],**v) for k,v in truth.items()]
    assert len(truth)==2*cfg['replications_per_side'] and len(p['rows'])==len(truth)*len(cfg['cells'])
    save(f'TRUTH_{ix}.json',dict(utc=utc(),prediction_sha256=digest(f'PREDICT_{ix}.json'),labels=labels))
    rows = [{**r,**truth[(r['panel'],r['rep'])]} for r in p['rows']]
    result = summarize(rows,method_alpha=.05,family_alpha=cfg['familywise_scoring_alpha'])
    save(f'SUMMARY_{ix}.json',dict(**result,utc=utc(),prediction_sha256=digest(f'PREDICT_{ix}.json'),
                                   truth_sha256=digest(f'TRUTH_{ix}.json')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('lock','pilot','predict','reveal'))
    parser.add_argument('dynamics',type=int,default=0,nargs='?')
    args = parser.parse_args()
    sources = {n:digest(n) for n in SOURCES}
    if args.phase=='lock':
        save('LOCK.json',dict(utc=utc(),sources=sources,cpu_threads=1,numpy=np.__version__))
        return
    assert read('LOCK.json')['sources']==sources
    cfg = read('CONFIG.json')
    assert math.isclose(sum(cfg[k] for k in ('alpha_beta','alpha_drift','alpha_score')),.05)
    assert all(c['distance_deg']==7 for c in cfg['cells'])
    if args.phase=='pilot':
        pilot(cfg)
    else:
        assert args.dynamics in range(len(cfg['psi_deg']))
        (predict if args.phase=='predict' else reveal)(cfg,args.dynamics)


if __name__=='__main__':
    main()
