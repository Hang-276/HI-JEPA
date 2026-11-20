"""Single-threaded selective likelihood benchmark with a frozen design split."""

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
from stats import choose_threshold, score_cells

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'hidden_boundary_cpu_20261107'
SOURCES = ('CONFIG.json','PROTOCOL.md','run.py','stats.py',
           '../hidden_boundary_cpu_20261107/run.py','../hidden_boundary_cpu_20261107/rule.py',
           '../hidden_boundary_cpu_20261107/score.py','../hidden_boundary_cpu_20261107/CONFIG.json',
           '../phase2_boundary_20261107/BOUNDARY.md')
sys.path.insert(0,str(BASE))
spec = importlib.util.spec_from_file_location('_frozen_hidden_var',BASE/'run.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


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


def rng(seed,dyn,cell,start,role):
    return np.random.default_rng(np.random.SeedSequence([seed,dyn,cell,start,role]))


def moments_for(cell):
    tau,d = cell['tau'],math.radians(cell['distance_deg'])
    mean = 2*(tau*math.sin(d))**2
    var = (tau*math.sin(2*d))**2/(2*(1+tau))
    return mean,var


def design_sample(gen,cell,n):
    """Magnitude law of both LLRs from independent normalized innovations."""
    mean,var = moments_for(cell)
    count_mean,count_var,_ = gen.multinomial(cell['n'],[4/13,2/13,7/13],size=n).T
    signed = count_mean*mean+np.sqrt(2*count_mean*mean)*gen.standard_normal(n)
    present = count_var>0
    df = count_var[present]
    delta = math.sqrt(var*var+2*var)
    full = signed.copy()
    full[present] += .5*((var+delta)*gen.chisquare(df)+(var-delta)*gen.chisquare(df))
    return full,signed


def compute_design(cfg):
    thresholds,detail = {},{}
    for j,cell in enumerate(cfg['cells']):
        scores = design_sample(rng(cfg['seed'],0,j,0,1),cell,cfg['design_reps_per_cell'])
        thresholds[cell['name']] = {method:choose_threshold(vals.tolist(),cfg['target_wrong_design'])
                                    for method,vals in zip(('full','projection'),scores)}
        summary = {}
        for method,vals in zip(('full','projection'),scores):
            t = thresholds[cell['name']][method]
            m = abs(vals)
            selected = (m>0)&(m>=t)
            v = np.exp(-m[selected])
            summary[method] = dict(threshold=t,coverage=float(selected.mean()),
                                   posterior_wrong=float((v/(1+v)).sum()/len(m)))
            assert summary[method]['posterior_wrong']<=cfg['target_wrong_design']+1e-13
        detail[cell['name']] = summary
    save('DESIGN.json',dict(utc=utc(),lock_sha256=sha('LOCK.json'),thresholds=thresholds,
                           design_reps_per_cell=cfg['design_reps_per_cell'],diagnostics=detail))


def innovations(x,a,sigma):
    return (x[1:]-np.einsum('tbi,bji->tbj',x[:-1],a))/sigma


def ratios(e,u,beta,cell):
    d,tau = math.radians(cell['distance_deg']),cell['tau']
    c,s = np.cos(beta),np.sin(beta)
    z1,z2 = c*e[:,:,1]+s*e[:,:,2], -s*e[:,:,1]+c*e[:,:,2]
    w2 = (u==3).astype(np.int8)-(u==4)
    w3 = (u==5).astype(np.int8)-(u==6)
    active = (np.abs(w2)+np.abs(w3)).sum(0)
    t = (w2*z2-w3*z1).sum(0)
    part = 2*tau*math.sin(d)*t
    variance = tau*math.sin(2*d)/(1+tau)*((u==10)*z1*z2-(u==11)*z1*z2).sum(0)
    cutoff = math.sqrt(2*math.log(2/.05))*np.sqrt(np.maximum(active,1))
    old = ((active>0)&(t>cutoff)).astype(np.int8)-((active>0)&(t< -cutoff)).astype(np.int8)
    return part+variance,part,old


def generic_terms(e,u,beta,cell):
    d,tau = math.radians(cell['distance_deg']),cell['tau']
    mean = np.zeros((13,4))
    mean[1:9:2] = tau*np.eye(4)
    mean[2:9:2] = -tau*np.eye(4)
    sd = np.ones((13,4))
    sd[9+np.arange(4),np.arange(4)] *= math.sqrt(1+tau)
    results = []
    for sign in (1,-1):
        r = generator.rotations(1,2,beta+sign*d)
        physical = np.einsum('tbj,bij->tbi',e,r)
        results.append(-.5*(((physical-mean[u])/sd[u])**2).sum(2))
    return results[0]-results[1]


def pilot(cfg):
    seed = cfg['pilot_seed']
    sigma = math.sqrt(1-cfg['rho']**2)
    max_error = 0.
    all_contexts = set()
    for ix in range(2):
        for j,cell in enumerate(cfg['cells']):
            count = 16
            side = np.tile([-1,1],count//2)
            a,beta = generator.hidden(cfg,seed,ix,-1,j,count)
            x,u = generator.trajectory(rng(seed,ix,j,0,2),a,beta,cell,side,sigma)
            all_contexts.update(np.unique(u).tolist())
            theta = beta+side*math.radians(cell['distance_deg'])
            target,gap = generator.population_set(a,theta,cfg['lags'])
            assert (target==np.where(side<0,12,10)).all() and (gap>0).all()
            for sign in (-1,1):
                assert np.ptp(gap[side==sign])<1e-12
            assert abs(gap[side<0].mean()-gap[side>0].mean())<1e-12
            e = innovations(x,a,sigma)
            full,projected,old = ratios(e,u,beta,cell)
            terms = generic_terms(e,u,beta,cell)
            max_error = max(max_error,float(abs(full-terms.sum(0)).max()))
            np.testing.assert_allclose(full,terms.sum(0),rtol=0,atol=1e-12)
            np.testing.assert_allclose(projected,(terms*((3<=u)&(u<=6))).sum(0),rtol=0,atol=1e-12)
            assert np.isin(old,[-1,0,1]).all()
    assert all_contexts==set(range(13))
    save('PILOT.json',dict(utc=utc(),passed=True,lock_sha256=sha('LOCK.json'),
                           max_path_llr_difference=max_error,observed_contexts=sorted(all_contexts),
                           checks=['equal physical gap and opposite set','generic Gaussian versus path LLR',
                                   'generic Gaussian mean-only LLR versus projection']))


def side_sample(cfg,ix,j,start,count):
    return 2*rng(cfg['seed'],ix,j,start,3).integers(0,2,count,dtype=np.int8)-1


def predict(cfg,ix):
    start_time = time.monotonic()
    pg,design = read('PILOT.json'),read('DESIGN.json')
    assert pg['passed'] and pg['lock_sha256']==design['lock_sha256']==sha('LOCK.json')
    seed,sigma = cfg['seed'],math.sqrt(1-cfg['rho']**2)
    n,batch = cfg['validation_reps_per_cell'],cfg['batch']
    payload = {}
    for j,cell in enumerate(cfg['cells']):
        out = []
        for start in range(0,n,batch):
            count = min(batch,n-start)
            side = side_sample(cfg,ix,j,start,count)
            a,beta = generator.hidden(cfg,seed,ix,-1,start+j*n,count)
            x,u = generator.trajectory(rng(seed,ix,j,start,4),a,beta,cell,side,sigma)
            full,projected,old = ratios(innovations(x,a,sigma),u,beta,cell)
            out.append(np.column_stack((full,projected,old)))
        payload[cell['name']] = np.concatenate(out)
    path = f'PREDICT_{ix}.npz'
    with (HERE/path).open('xb') as out:
        np.savez_compressed(out,**payload)
    save(f'PREDICT_{ix}.json',dict(utc=utc(),lock_sha256=sha('LOCK.json'),design_sha256=sha('DESIGN.json'),
                                   scores_sha256=sha(path),seconds=time.monotonic()-start_time))


def reveal(cfg,ix):
    p = read(f'PREDICT_{ix}.json')
    path = f'PREDICT_{ix}.npz'
    assert p['lock_sha256']==sha('LOCK.json') and p['design_sha256']==sha('DESIGN.json')
    assert p['scores_sha256']==sha(path)
    n,batch = cfg['validation_reps_per_cell'],cfg['batch']
    name = f'psi_{cfg["psi_deg"][ix]:g}'
    rows = []
    with np.load(HERE/path,allow_pickle=False) as predictions:
        for j,cell in enumerate(cfg['cells']):
            side = np.concatenate([side_sample(cfg,ix,j,start,min(batch,n-start)) for start in range(0,n,batch)])
            score = predictions[cell['name']]
            assert score.shape==(n,3)
            rows.extend(dict(dynamics=name,cell=cell['name'],side=int(sign),
                             ll_full=float(a),ll_projection=float(b),old=int(c))
                        for sign,(a,b,c) in zip(side,score,strict=True))
    result = score_cells(rows,read('DESIGN.json')['thresholds'],alpha=cfg['family_alpha'])
    save(f'SUMMARY_{ix}.json',dict(**result,utc=utc(),prediction_sha256=sha(path),
                                  design_sha256=sha('DESIGN.json'),replications=len(rows)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('lock','pilot','design','predict','reveal'))
    parser.add_argument('dynamics',type=int,nargs='?',default=0)
    args = parser.parse_args()
    hashes = {name:sha(name) for name in SOURCES}
    if args.phase=='lock':
        save('LOCK.json',dict(utc=utc(),sources=hashes,cpu_threads=1,numpy=np.__version__))
        return
    assert read('LOCK.json')['sources']==hashes
    cfg = read('CONFIG.json')
    assert cfg['wrong_budget']==.05 and args.dynamics in (0,1)
    if args.phase=='pilot':
        pilot(cfg)
    elif args.phase=='design':
        assert read('PILOT.json')['passed']
        compute_design(cfg)
    else:
        (predict if args.phase=='predict' else reveal)(cfg,args.dynamics)


if __name__=='__main__':
    main()
