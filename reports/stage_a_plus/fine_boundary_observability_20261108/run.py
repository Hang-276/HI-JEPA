"""Frozen TRAIN-only contrast, trajectory placebo, and gap sensitivity check."""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT/'outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2'
OWNER = BASE/'data/DATA_PREPARED.json'
SLOTS = {'ar_rich_fresh1':(128,129,(0,2)), 'fhn_fresh_eps008':(96,193,(0,3))}
LAGS = (4,16,32)
CODES = (0,5,6,7,8)
SEED = 2026110801
NPLACEBO = 127
NBOOT = 512
SOURCES = (HERE/'PROTOCOL.md', Path(__file__).resolve(), OWNER,
           ROOT/'work/hcf_method_finalize_20261027/v1/explicit_selector.py')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(a):
    digest = hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode())
    digest.update(a.tobytes())
    return dict(sha256=digest.hexdigest(),shape=list(a.shape),dtype=str(a.dtype))


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(name,obj):
    with (HERE/name).open('x') as out:
        json.dump(obj,out,default=lambda a:a.tolist(),separators=(',',':'),ensure_ascii=False,allow_nan=False)
        out.write('\n')


def context_arrays(slot,n,t):
    ref = next(r for r in read(OWNER)['records'] if r['slot_id']==slot)
    file = BASE/'data/TRAIN_CAL'/f'{slot}.npz'
    assert ref['observations']['path']==str(file)
    with np.load(file,allow_pickle=False) as f:
        actions,u = f['train__actions'],f['train__regimes']
    train = ref['diagnostics']['train']['arrays']
    assert identity(actions)==train['actions'] and identity(u)==train['regimes']
    assert actions.shape==(n,t-1,2) and u.shape==(n,t-1)
    assert np.isfinite(actions).all() and np.isin(u,np.arange(9)).all()
    return actions.astype(np.float64),u,ref


def fine_train(slot,rep,n,t,owner):
    folder = BASE/'execution/jobs/representation/OursFine4'/slot/f'seed{rep}'
    file = folder/'FEATURES.npz'
    comp = read(folder/'COMPLETED.json')
    g0_path = folder/'G0.json'
    assert (comp['slot_id'],comp['logical_rep'],comp['valid_t_start'])==(slot,rep,0)
    assert comp['G0']==dict(path=str(g0_path),sha256=sha(g0_path))
    assert comp['features']['path']==str(file)
    job_path = BASE/'execution/registration/representation/OursFine4'/slot/f'seed{rep}'/'JOB.json'
    assert comp['job']==dict(path=str(job_path),sha256=sha(job_path))
    job = read(job_path)
    assert job['data_prepared']==dict(path=str(OWNER),sha256=sha(OWNER))
    assert job['data_manifest']['source']==owner['observations']
    assert job['data_identity']['train']==owner['diagnostics']['train']['arrays']
    g0 = read(g0_path)
    with np.load(file,allow_pickle=False) as f:
        raw = f['train']
    r = identity(raw)
    assert r['sha256']==g0['TRAIN_raw_Fine_sha256']
    assert r['dtype']=='float32' and r['shape']==g0['TRAIN_shape']==[n,t,4]
    assert np.isfinite(raw).all() and g0['horizons']==list(LAGS)
    y = (raw-g0['mean_FP64'])/g0['std_FP64_ddof0']
    return y,r


def residual(y,a,u,ref):
    prev = y[:,:-1]
    delta = y[:,1:]-prev
    x = np.concatenate((np.ones(prev.shape[:2]+(1,)),prev,prev**2,prev**3,a),axis=-1)
    mask = np.isin(u[ref],CODES)
    coef,_,rank,_ = np.linalg.lstsq(x[ref][mask],delta[ref][mask],rcond=1e-12)
    assert rank==x.shape[-1]
    return delta-x@coef


def make_schedule(groups,seed):
    rng = np.random.default_rng(seed)
    shuffle,boot = [],[]
    for g in groups:
        n = len(g)
        reference = np.arange(n)
        sampled = []
        for _ in range(NPLACEBO):
            p = rng.permutation(n)
            while (p==reference).any():
                p = rng.permutation(n)
            sampled.append(p)
        shuffle.append(sampled)
        indexes = rng.integers(n,size=(NBOOT,n))
        boot.append(np.stack([np.bincount(row,minlength=n) for row in indexes]))
    return np.asarray(shuffle),np.concatenate(boot,axis=1)


def episode_stats(rr,u):
    mask = u[...,None]==np.array(CODES)
    return np.einsum('etc,etij->ecij',mask,rr),mask.sum(1)


def axes(sums,counts):
    assert (counts>0).all()
    means = sums/counts[...,None,None]
    contrast = means[...,1:,:,:]-means[...,:1,:,:]
    values,vectors = np.linalg.eigh(contrast)
    columns = vectors[...,-1].swapaxes(-1,-2)
    dominance = values[...,-1]/np.maximum(abs(values[...,:-1]).max(-1),np.finfo(float).tiny)
    condition = np.linalg.cond(columns)
    good = (values[...,-1]>0).all(-1)&(condition<=10)
    return dict(columns=columns,eigenvalues=values,dominance=dominance,condition=condition,good=good)


def export_axes(result):
    c = result['condition']
    return dict(columns=result['columns'],eigenvalues=result['eigenvalues'],
                dominance=result['dominance'],condition=float(c) if np.isfinite(c) else None,
                invertible_for_diagnostic=bool(result['good']))


def similarity(bases,pair):
    return np.min([abs((bases[i][:,pair]*bases[j][:,pair]).sum(0))
                   for i,j in itertools.combinations(range(3),2)],axis=0).clip(0,1)


def dynamic_matrices(y):
    flat = y.reshape(-1,4)
    z = flat-flat.mean(0)
    s = z.T@z/len(z)
    assert np.linalg.eigvalsh(s).min()>0
    mm = []
    for h in LAGS:
        d = y[:,h:]-y[:,:-h]
        mm.append(d.swapaxes(1,2)@d/d.shape[1])
    return np.mean(mm,axis=0),s


def slowness(bases,mm,s):
    w = np.linalg.inv(bases)
    norm = np.einsum('bji,ik,bjk->bj',w,s,w)
    assert (norm>0).all()
    return np.einsum('bji,eik,bjk->bej',w,mm,w)/norm[:,None,:]


def top_sets(scores):
    return np.sort(np.argsort(scores,axis=-1)[...,:2],axis=-1)


def gap_check(yref,episodes,pair,pooled,groups,boot):
    count = int(boot['good'].sum())
    if not pooled['good'] or not count:
        return dict(passed=False,available=False,valid_bootstrap=count,near_count=0)
    mm,s = dynamic_matrices(yref)
    center = slowness(pooled['columns'][None],mm,s)[0]
    pool_top = top_sets(center)
    a,b = pair
    g = center[:,b]-center[:,a]
    crosses = ((pool_top==a).any(1)!=(pool_top==b).any(1))
    eligible = np.flatnonzero(crosses)
    near = eligible[np.argsort(abs(g[eligible]),kind='stable')[:math.ceil(len(eligible)/4)]]
    bootstrap = slowness(boot['columns'][boot['good']],mm,s)
    gaps = bootstrap[...,b]-bootstrap[...,a]
    low,high = np.quantile(gaps,[.025,.975],axis=0)
    top_change = (top_sets(bootstrap)!=pool_top[None]).any(2)
    if all(k['good'] for k in groups):
        group_scores = slowness(np.stack([k['columns'] for k in groups]),mm,s)
        fold = group_scores[...,b]-group_scores[...,a]
    else:
        fold = None
    observed = []
    for k,ep in enumerate(episodes):
        boot_signs = bool(gaps[:,k].min()<0<gaps[:,k].max())
        f = None if fold is None else fold[:,k]
        same = f is not None and ((g[k]>0 and low[k]>0 and f.min()>0)
                                 or (g[k]<0 and high[k]<0 and f.max()<0))
        observed.append(dict(episode=int(ep),near=bool(k in near),straddles=bool(crosses[k]),
            pooled_gap=float(g[k]),pooled_top2=pool_top[k],interval=[low[k],high[k]],
            observed_range=[gaps[:,k].min(),gaps[:,k].max()],group_gaps=f,
            opposite_bootstrap_signs=boot_signs,
            opposite_group_signs=bool(f is not None and f.min()<0<f.max()),
            top2_change_count=int(top_change[:,k].sum()),stable_sign=bool(same)))
    windows = [r for r in observed if r['near']]
    stable = sum(r['stable_sign'] for r in windows)
    return dict(available=True,valid_bootstrap=count,near_count=len(windows),straddling=len(eligible),
                near_stable=stable,near_bootstrap_reversal=sum(r['opposite_bootstrap_signs'] for r in windows),
                near_group_reversal=sum(r['opposite_group_signs'] for r in windows),
                near_interval_both_signs=sum(r['interval'][0]<0<r['interval'][1] for r in windows),
                near_top2_changes=sum(r['top2_change_count']>0 for r in windows),
                windows=observed,passed=bool(len(windows)>=4 and count/NBOOT>=.95 and 2*stable>=len(windows)))


def evaluate(y,a,u,groups,pair,shuffle,weights):
    ref,*rest = groups
    err = residual(y,a,u,ref)
    rr = err[...,None]*err[...,None,:]
    sums,counts = episode_stats(rr,u)
    support = [dict(frames=counts[g].sum(0),trajectories=(counts[g]>0).sum(0)) for g in rest]
    has_support = all((s['frames']>=64).all() and (s['trajectories']>=12).all() for s in support)
    group_axes = [axes(sums[g].sum(0),counts[g].sum(0)) for g in rest]
    index = np.concatenate(rest)
    pool = axes(sums[index].sum(0),counts[index].sum(0))
    align = similarity([r['columns'] for r in group_axes],list(pair))
    null = []
    for j in range(NPLACEBO):
        b = []
        for k,g in enumerate(rest):
            p = shuffle[k,j]
            mis,c = episode_stats(rr[g],u[g[p]])
            assert np.array_equal(c.sum(0),counts[g].sum(0))
            b.append(axes(mis.sum(0),c.sum(0))['columns'])
        null.append(float(similarity(b,list(pair)).min()))
    null95 = float(np.quantile(null,.95))
    dominance = np.stack([x['dominance'] for x in group_axes])[:,list(pair)]
    checks = dict(support=bool(has_support),
                  positive_dominant=bool((pool['dominance'][list(pair)]>=2).all()
                                         and ((dominance>=2).sum(0)>=2).all()),
                  group_angles=bool((align>=math.cos(math.radians(25))).all()),
                  exceeds_placebo=bool(align.min()>null95),
                  invertible=bool(pool['good'] and all(x['good'] for x in group_axes)))
    resampled = axes(np.einsum('be,ecij->bcij',weights,sums[index]),weights@counts[index])
    return dict(direction_passed=all(checks.values()),direction_checks=checks,group_support=support,
                pooled=export_axes(pool),folds=[export_axes(x) for x in group_axes],
                pair_max_angles_deg=np.degrees(np.arccos(align)),stability_score=float(align.min()),
                placebo_q95=null95,placebo_scores=null,
                gap=gap_check(y[ref],ref,pair,pool,group_axes,resampled))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase',choices=('lock','run'))
    phase = p.parse_args().phase
    sources = {str(f.relative_to(ROOT)):sha(f) for f in SOURCES}
    plan = {s:dict(groups=[list(range(i,n,4)) for i in range(4)],pair=pair,shape=[n,t,4],seed=SEED+i)
            for i,(s,(n,t,pair)) in enumerate(SLOTS.items())}
    if phase=='lock':
        save('LOCK.json',dict(utc=utc(),sources=sources,plan=plan,placebo=NPLACEBO,bootstrap=NBOOT,cpu_threads=1))
        return
    lock = read(HERE/'LOCK.json')
    assert lock['sources']==sources and lock['plan']==json.loads(json.dumps(plan))
    results = {}
    for slot,(n,t,pair) in SLOTS.items():
        a,u,owner = context_arrays(slot,n,t)
        groups = [np.array(g) for g in lock['plan'][slot]['groups']]
        shuffled,weights = make_schedule(groups[1:],lock['plan'][slot]['seed'])
        models = []
        for rep in range(3):
            y,ident = fine_train(slot,rep,n,t,owner)
            out = evaluate(y,a,u,groups,pair,shuffled,weights)
            models.append(dict(rep=rep,train_identity=ident,**out))
        n_pass = sum(m['direction_passed'] and m['gap']['passed'] for m in models)
        results[slot] = dict(pair=pair,trajectories=n,direction_pass=sum(m['direction_passed'] for m in models),
                             combined_pass=n_pass,passed=n_pass>=2,models=models)
    save('SUMMARY.json',dict(utc=utc(),lock_sha256=sha(HERE/'LOCK.json'),passed=all(r['passed'] for r in results.values()),
                             results=results,access=['train__actions','train__regimes','train Fine'],
                             gap_scope='fixed observable windows; perturb directions only, no physical certificate'))


if __name__=='__main__':
    main()
