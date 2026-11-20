"""Frozen finite-reference normalization and population-target certification."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from scipy.signal import lfilter
from cert import reference_bands, numerator_bounds, certify

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'dose_joint_window_20261110'/'run.py'
spec=importlib.util.spec_from_file_location('joint',PRIOR)
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
LENGTHS=(1025,16385)
BUDGETS=(64,1024,16384,65536)
RHO=(.995,.99,.97)
ELL=.35
METRICS=('identity_error','warp_error','physical_disagreement','empirical_disagreement','hidden_disagreement','false_consensus','safe','safe_error','known_reference_safe')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,**values):
    with (HERE/name).open('x') as stream:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),**values),stream,indent=2,allow_nan=False)


def read(name):
    return json.loads((HERE/name).read_text())


def sources():
    files=[HERE/f for f in ('run.py','cert.py','PROTOCOL.md','THEORY.md')]
    files.extend((PRIOR,old.prior.HERE/'run.py'))
    return {str(p):sha(p) for p in files}


def verify():
    assert sources()==read('LOCK.json')['sources']
    return sha(HERE/'LOCK.json')


def paths(seed,group,length,count):
    rng=np.random.default_rng(np.random.SeedSequence([seed,group]))
    innovation=rng.standard_normal((4,count,length))
    def ar(axis,rho):
        data=innovation[axis]
        tail,_=lfilter([np.sqrt(1-rho*rho)],[1,-rho],data[:,1:],axis=1,zi=rho*data[:,:1])
        return np.concatenate([data[:,:1],tail],axis=1)
    return ar(0,.995),[ar(1,r) for r in RHO],ar(2,.999),ar(3,.90),sha_array(innovation)


def sha_array(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def panel(index,lengths,budgets,count,pilot=False):
    seed=2026341001 if pilot else 2026341002
    x,competitors,slow,fast,trajectory_hash=paths(seed,index,max(lengths),count)
    rng=np.random.default_rng(np.random.SeedSequence([2026341001 if pilot else 2026341003,index,99]))
    reference=rng.standard_normal((max(budgets),4))
    base=old.path_scores(x,lengths)
    warp_num=old.path_scores(old.warp(x,ELL),lengths)
    scores_other=[old.path_scores(a,lengths) for a in (*competitors,slow,fast)]
    ref_records=[]
    counts=np.zeros((len(lengths),len(budgets),len(RHO),len(METRICS)),dtype=int)
    true_var=np.ones(4)
    warp_var=old.variance_bounds(ELL)[0]
    bands=[]
    for n in budgets:
        lower,upper,audit=reference_bands(reference[:n],ELL)
        empirical_var=reference[:n].var(axis=0)
        empirical_warp=old.warp(reference[:n,0],ELL).var()
        bands.append((lower,upper,empirical_var,empirical_warp))
        ref_records.append(dict(n=n,lower=lower.tolist(),upper=upper.tolist(),
            covers_fixed_truths=bool(np.all(lower<=true_var) and np.all(upper>=true_var)
                                    and lower[0]<=warp_var<=upper[0]),audit=audit))
    for t,length in enumerate(lengths):
        observed=np.stack((x[:,:length],competitors[0][:,:length],slow[:,:length],fast[:,:length]),axis=-1)
        low_num,high_num=numerator_bounds(observed,ELL)
        assert np.all(low_num[:,0]<=base[t]+1e-10) and np.all(high_num[:,0]>=base[t]-1e-10)
        assert np.all(low_num[:,0]<=warp_num[t]+1e-10) and np.all(high_num[:,0]>=warp_num[t]-1e-10)
        weights=old.prior.cross_weights(x[:,:length])
        cmin,cmax=old.prior.cross_extrema(x[:,:length],weights,ELL)
        known_lo,known_hi=old.prior.envelope(base[t],cmin,cmax,ELL)
        for condition in range(len(RHO)):
            numerators=np.column_stack((base[t],scores_other[condition][t],scores_other[3][t],scores_other[4][t]))
            physical_warp=numerators.copy()
            physical_warp[:,0]=warp_num[t]/warp_var
            truth_id=old.top2(numerators)
            truth_warp=old.top2(physical_warp)
            physical_disagree=np.any(truth_id!=truth_warp,axis=1)
            low_num[:,1:]=high_num[:,1:]=numerators[:,1:]
            known_cutoff=np.sort(numerators[:,1:],axis=1)[:,1]
            known_safe=(known_hi<known_cutoff)|(known_lo>known_cutoff)
            for j,(lower,upper,vhat,vwarp) in enumerate(bands):
                point=old.top2(numerators/vhat)
                estimated_warp=numerators/vhat
                estimated_warp[:,0]=warp_num[t]/vwarp
                warped_point=old.top2(estimated_warp)
                error=np.any(point!=truth_id,axis=1)
                warp_error=np.any(warped_point!=truth_warp,axis=1)
                disagree=np.any(point!=warped_point,axis=1)
                false_consensus=(~disagree)&(error|warp_error)
                safe=certify(low_num,high_num,lower,upper,point)
                safe_error=safe&(error|np.any(point!=truth_warp,axis=1))
                values=(error,warp_error,physical_disagree,disagree,(~disagree)&physical_disagree,false_consensus,safe,safe_error,known_safe)
                counts[t,j,condition]=[int(v.sum()) for v in values]
    return dict(counts=counts.tolist(),trajectory_hash=trajectory_hash,reference_hash=sha_array(reference),reference=ref_records)


def interval(values,draws):
    return dict(estimate=float(values.mean()),ci95=np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('lock','pilot','formal'))
    phase=parser.parse_args().phase
    if phase=='lock':
        save('LOCK.json',sources=sources(),lengths=LENGTHS,budgets=BUDGETS,panels=32,paths_per_panel=64)
        return
    lock=verify()
    if phase=='pilot':
        from cert import self_check
        checks=self_check()
        result=panel(0,(257,),(512,),8,True)
        assert np.asarray(result['counts'])[... ,7].sum()==0
        save('PILOT.json',lock=lock,passed=True,checks=checks,panel=result)
        return
    assert read('PILOT.json')['passed']
    rows=[]
    for index in range(32):
        rows.append(panel(index,LENGTHS,BUDGETS,64))
        if (index+1)%8==0:
            print(f'finished {index+1}/32 independent panels',flush=True)
    rates=np.asarray([r['counts'] for r in rows])/64
    draws=np.random.default_rng(2026341004).integers(32,size=(4096,32))
    contrasts=(('reference_identity_error',rates[:,-1,0,0,0]-rates[:,-1,-1,0,0],.10),
               ('hidden_physical_disagreement',rates[:,-1,0,0,4],.05),
               ('large_gap_coverage',rates[:,-1,-1,2,6],.50),
               ('large_gap_coverage_gain',rates[:,-1,-1,2,6]-rates[:,-1,0,2,6],.25),
               ('empirical_disagreement_increases',rates[:,-1,-1,0,3]-rates[:,-1,0,0,3],.10))
    gates={}
    for name,values,threshold in contrasts:
        stat=interval(values,draws)
        gates[name]=stat|dict(threshold=threshold,passed=stat['ci95'][0]>threshold)
    results=[]
    for t,length in enumerate(LENGTHS):
        for j,n in enumerate(BUDGETS):
            for c,rho in enumerate(RHO):
                result={name:interval(rates[:,t,j,c,m],draws) for m,name in enumerate(METRICS)}
                results.append(dict(length=length,n=n,rho=rho,metrics=result))
    failures=sum(not ref['covers_fixed_truths'] for row in rows for ref in row['reference'])
    safe_errors=int(np.asarray([r['counts'] for r in rows])[...,7].sum())
    save('SUMMARY.json',lock=lock,pilot_sha=sha(HERE/'PILOT.json'),results=results,gates=gates,
         all_gates=all(g['passed'] for g in gates.values()),reference_failures=failures,safe_errors=safe_errors,
         panels=rows,metrics=METRICS,independent_panels=32,trajectories=2048)
    print(json.dumps(gates,indent=2),flush=True)

if __name__=='__main__':
    main()
