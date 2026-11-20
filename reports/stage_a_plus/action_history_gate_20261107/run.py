"""Single-threaded action-history screen and gated Fine cache pilot."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parent / 'phase1_cache_20261107'
P3 = HERE.parent / 'phase3_regime_pilot_20261107'
STEP6 = HERE.parent / 'step6_fine_20261105'
sys.path.insert(0, str(P1))
from run_phase1 import BASE, ROOT, LAGS, as_json, dynamic, model_dir, order, phys_file, train_calibrator
import summarize_phase1 as aggregation

SLOT = 'fhn_fresh_eps008'
OWNER = BASE / 'data' / 'DATA_PREPARED.json'
OBS = BASE / 'data' / 'TRAIN_CAL' / (SLOT + '.npz')
METHODS = ('pooled', 'energy', 'action', 'mismatched')
SOURCES = (HERE / 'PROTOCOL.md', Path(__file__).resolve(), OWNER, P1 / 'SHA256SUMS',
           P3 / 'SHA256SUMS', STEP6 / 'SHA256SUMS', P1 / 'run_phase1.py',
           P1 / 'summarize_phase1.py', STEP6 / 'transfer.py', STEP6 / 'summarize.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved(path):
    manifest = dict((name, digest) for digest, name in
                    (line.split(maxsplit=1) for line in (path.parent/'SHA256SUMS').read_text().splitlines()))
    assert sha(path) == manifest[path.name], path
    return read(path)


def save(name, value):
    with (HERE/name).open('x') as out:
        json.dump(value, out, default=as_json, ensure_ascii=False, allow_nan=False,
                  separators=(',', ':'))
        out.write('\n')


def close(x, y):
    np.testing.assert_allclose(x, y, atol=1e-12, rtol=0)


def gap(scores):
    return scores[..., 3] - scores[..., 0]


def archived_cases():
    phase1 = saved(P1/f'reveal_{SLOT}.json')
    freeze = saved(P1/f'freeze_{SLOT}.json')
    phase3 = saved(P3/f'PREDICT_{SLOT}.json')
    records = []
    for row in (r for r in phase1['rows'] if r['episode'] == 1):
        rep = row['rep']
        p = next(x for x in phase3['rows'] if (x['rep'], x['episode']) == (rep, 1))
        diag = row['full_diagnostic']
        cal = freeze['parents'][rep]['cal']
        actual = float(gap(np.array(row['z'])))
        cross = float(gap(np.array(diag['residual_cross'])))
        train_cross = float(gap(np.array(cal['train_residual_cross']).mean(0)))
        energy = float(gap(np.array(diag['residual_energy_shift'])))
        predicted = {m:float(gap(np.array(v))) for m,v in p['predictions'].items()}
        close(predicted['pooled']+(cross-train_cross)+energy, actual)
        records.append(dict(episode=1, rep=rep, fine_top2=diag['fine_top2'],
                            physical_top2=diag['physical_top2'], true_axis3_minus_0=actual,
                            predicted_axis3_minus_0=predicted, raw_cal_cross=cross,
                            train_cross=train_cross, cross_drift=cross-train_cross,
                            energy_drift=energy, remaining_pooled_correction=actual-predicted['pooled']))
    assert [r['rep'] for r in records] == [0,1,2]
    save('CASE_DIAGNOSIS.json', dict(utc=utc(), rows=records, scope='post-hoc existing scores only'))


def source_member(path, name, forecast):
    assert (path, name) == (phys_file(SLOT), 'train__physical') or any(
        path == model_dir(SLOT, r)/'FEATURES.npz' and name in ('train','cal') for r in range(3))
    reference = next(x for x in forecast['access']
                     if x['path'] == str(path.relative_to(ROOT)) and x['member'] == name)
    with np.load(path, allow_pickle=False) as file:
        x = file[name]
    assert hashlib.sha256(x.tobytes()).hexdigest() == reference['full_member_sha256']
    assert x.dtype == np.float32 and x.ndim == 3 and x.shape[-1] == 4 and np.isfinite(x).all()
    return x.astype(np.float64)


def load_actions(split):
    owner = next(r for r in read(OWNER)['records'] if r['slot_id'] == SLOT)
    assert owner['observations']['path'] == str(OBS)
    with np.load(OBS, allow_pickle=False) as file:
        x = file[split+'__actions']
    check = hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
    assert dict(sha256=check,dtype=str(x.dtype),shape=list(x.shape)) == owner['diagnostics'][split]['arrays']['actions']
    assert x.dtype == np.float32 and x.shape[-1] == 2 and np.isin(x, (-1.,-.5,0.,.5,1.)).all()
    assert np.array_equal(x, np.repeat(x[:, ::8], 8, axis=1))
    return x.astype(np.float64)


def load_inputs(cal=False):
    ref = saved(P1/f'freeze_{SLOT}.json')
    old = {p['rep']:p for p in saved(STEP6/'forecast_lock.json')['parents'] if p['slot'] == SLOT}
    physical = source_member(phys_file(SLOT), 'train__physical', ref)
    z = (physical-old[0]['train_mu_z'])/old[0]['train_sd_z']
    x, ev = [], []
    for rep, p in sorted(old.items()):
        close(old[0]['train_mu_z'],p['train_mu_z'])
        close(old[0]['train_sd_z'],p['train_sd_z'])
        for split, dest in (('train',x),('cal',ev)):
            if split=='cal' and not cal:
                continue
            y = source_member(model_dir(SLOT,rep)/'FEATURES.npz',split,ref)
            dest.append(((y-p['train_mu_y'])/p['train_sd_y'])[...,p['perm']]*p['sign'])
    a = load_actions('train')
    assert z.shape == (96,193,4) and a.shape == (96,192,2) and all(y.shape == z.shape for y in x)
    if cal:
        ac = load_actions('cal')
        assert ac.shape == (24,192,2) and all(y.shape == (24,193,4) for y in ev)
        return z, x, a, ref, ev, ac
    return z,x,a


def history(actions,h):
    m = h//2
    prefix = np.pad(actions.cumsum(1), ((0,0),(1,0),(0,0)))
    early = (prefix[:,m:-m] - prefix[:,:-h])/m
    late = (prefix[:,h:] - prefix[:,m:-m])/m
    result = np.concatenate((early,late),axis=-1)
    close(result[0,0],np.r_[actions[0,:m].mean(0),actions[0,m:h].mean(0)])
    return result


def fit_affine(x,y):
    u,v = x.reshape(-1,x.shape[-1]), y.reshape(-1,y.shape[-1])
    assert len(u)==len(v)
    center = u.mean(0)
    coef,_,rank,singular = np.linalg.lstsq(u-center,v-v.mean(0),rcond=1e-12)
    assert rank==u.shape[-1], (rank,singular)
    return coef, v.mean(0)-center@coef


def lag_delta(x,h):
    return x[:,h:]-x[:,:-h]


def predict(z,y,a,yt,at):
    base = train_calibrator(y,z)
    assert base['available'] and base['rank']==4
    b = base['B'].T
    by_lag = {m:[] for m in METHODS}
    for h in LAGS:
        dz,dy,dyt = lag_delta(z,h),lag_delta(y,h),lag_delta(yt,h)
        l,lt = dy@b,dyt@b
        physical = (dz**2).mean((0,1))
        center = physical-(l**2).mean((0,1))
        pooled = (lt**2).mean(1)+center
        by_lag['pooled'].append(pooled)
        for name,aa,bb in (('action',a,at),('mismatched',np.roll(a,1,axis=0),np.roll(at,1,axis=0))):
            q,qt = history(aa,h),history(bb,h)
            coef,intercept = fit_affine(dy,q)
            q = q-(dy@coef+intercept)
            qt = qt-(dyt@coef+intercept)
            close(q.mean((0,1)),np.zeros(4))
            k,_ = fit_affine(q,dz-l)
            w,wt = q@k,qt@k
            energy = (wt**2).mean(1)-(w**2).mean((0,1))
            cross = (2*lt*wt).mean(1)-(2*l*w).mean((0,1))
            by_lag[name].append(pooled+energy+cross)
            offset = physical-((l+w)**2).mean((0,1))
            close(by_lag[name][-1],((lt+wt)**2).mean(1)+offset)
            if name=='action':
                by_lag['energy'].append(pooled+energy)
    return {m:np.stack(v,axis=0).mean(0) for m,v in by_lag.items()}


def assess(records):
    masks = np.array([r['near'] for r in records])
    actual = np.array([r['truth_gap'] for r in records])
    values = np.array([[r['predictions'][m] for m in METHODS] for r in records])
    return {scope:{m:dict(mae=np.abs(values[mask,i]-actual[mask]).mean(),
                          correct_signs=int((values[mask,i]*actual[mask]>0).sum()),
                          ties=int((values[mask,i]==0).sum()),model_trajectory_rows=int(mask.sum()))
                   for i,m in enumerate(METHODS)}
            for scope,mask in (('all',np.ones(len(records),dtype=bool)),('near',masks))}


def train_screen():
    started = time.monotonic()
    z,ys,a = load_inputs()
    physical = gap(dynamic(z).mean(1))
    folds, records = [], []
    for fold in range(4):
        ev = np.arange(96)%4==fold
        fit = ~ev
        q25 = float(np.quantile(np.abs(physical[fit]),.25))
        near = np.abs(physical[ev])<=q25
        rows = []
        for rep,y in enumerate(ys):
            scores = predict(z[fit],y[fit],a[fit],y[ev],a[ev])
            for j,ep in enumerate(np.flatnonzero(ev)):
                rows.append(dict(episode=int(ep),rep=rep,near=bool(near[j]),truth_gap=physical[ep],
                                 predictions={m:gap(v)[j] for m,v in scores.items()}))
        folds.append(dict(fold=fold,near_episodes=int(near.sum()),fit_q25=q25,
                          metrics=assess(rows),rows=rows))
        records.extend(rows)
    assert len({(r['episode'],r['rep']) for r in records}) == len(records) == 288
    full = assess(records)
    g = full['near']
    close(len([r for r in records if r['near']]),3*sum(x['near_episodes'] for x in folds))
    checks = dict(near_support=sum(x['near_episodes'] for x in folds)>=16 and
                              all(x['near_episodes']>=3 for x in folds),
                  versus_pooled=g['action']['mae']<=.90*g['pooled']['mae'],
                  versus_mismatched=g['action']['mae']<=.90*g['mismatched']['mae'],
                  signed_over_energy=g['action']['mae']<g['energy']['mae'] and sum(
                      f['metrics']['near']['action']['mae']<f['metrics']['near']['energy']['mae'] for f in folds)>=3,
                  sign_improved=g['action']['correct_signs']>g['pooled']['correct_signs'])
    save('TRAIN_SCREEN.json',dict(utc=utc(),lock_sha256=sha(HERE/'LOCK.json'),checks=checks,
                                 passed=all(checks.values()),folds=folds,metrics=full,
                                 elapsed_seconds=time.monotonic()-started))


def cal_predict():
    screen = read(HERE/'TRAIN_SCREEN.json')
    assert screen['passed'] and screen['lock_sha256']==sha(HERE/'LOCK.json')
    z,ys,a,ref,evals,at = load_inputs(cal=True)
    p3 = saved(P3/f'PREDICT_{SLOT}.json')
    reference = {(r['rep'],r['episode']):r for r in p3['rows']}
    rows = []
    for rep,(y,yt) in enumerate(zip(ys,evals,strict=True)):
        scores = predict(z,y,a,yt,at)
        fine = dynamic(yt).mean(1)
        for ep,s in enumerate(fine):
            row = reference[(rep,ep)]
            close(s,row['y'])
            close(scores['pooled'][ep],row['predictions']['pooled'])
            rows.append(dict(rep=rep,episode=ep,y=s,selected=order(s),slot=SLOT,
                             train_r2=ref['parents'][rep]['train_r2'],
                             predictions={m:v[ep] for m,v in scores.items()}))
    save('CAL_PREDICTIONS.json',dict(utc=utc(),train_screen_sha256=sha(HERE/'TRAIN_SCREEN.json'),
                                     lock_sha256=sha(HERE/'LOCK.json'),rows=rows))


def cal_score():
    p = read(HERE/'CAL_PREDICTIONS.json')
    assert p['lock_sha256']==sha(HERE/'LOCK.json')
    archived = saved(P1/f'reveal_{SLOT}.json')
    truth = {(r['rep'],r['episode']):r for r in archived['rows']}
    rows = []
    for row in p['rows']:
        t = truth[(row['rep'],row['episode'])]
        close(t['y'],row['y'])
        close(t['train_r2'],row['train_r2'])
        rows.append({**t,'predictions':row['predictions']})
    aggregation.METHODS = aggregation.GATE_METHODS = METHODS
    metrics = aggregation.load_metrics()
    pairs = aggregation.pair_records(rows,metrics)
    save('CAL_RESULTS.json',dict(utc=utc(),prediction_sha256=sha(HERE/'CAL_PREDICTIONS.json'),
                                 metrics=aggregation.metric_bundle(rows,metrics),
                                 matched=aggregation.matched_bundle(pairs,(SLOT,),metrics),
                                 cases=[r for r in rows if r['episode']==1],pairs=pairs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('diagnose','lock','screen','cal_predict','cal_score'))
    phase = parser.parse_args().phase
    if phase=='diagnose':
        archived_cases()
        return
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    if phase=='lock':
        save('LOCK.json',dict(utc=utc(),sources=hashes,slot=SLOT,methods=METHODS,numeric_threads=1))
        return
    assert read(HERE/'LOCK.json')['sources']==hashes
    {'screen':train_screen,'cal_predict':cal_predict,'cal_score':cal_score}[phase]()


if __name__=='__main__':
    main()
