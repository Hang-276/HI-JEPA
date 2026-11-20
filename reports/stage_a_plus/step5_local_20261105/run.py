"""Run one local, prospective experiment phase. Stages refuse to overwrite products."""
import os
for setting in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[setting] = '1'
import argparse
import hashlib
import json
import math
import platform
import resource
import signal
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from shared import PHIS, LAGS, AMPLITUDES, candidates, scores, tcrit

OUT = Path(__file__).resolve().parent
PLAN = OUT.parent/'core_plan_20261105'
T = 2048
SEED = 20261105
EXPERIMENT_ID = 5
PHI_INDEX = {p: i for i, p in enumerate(PHIS)}
SPLIT_ID = {'cal': 0, 'val': 1}
LIMIT_SECONDS = 900
MAX_BYTES = 1_000_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(name, data):
    path = OUT/name
    if path.exists():
        raise RuntimeError(f'Output already exists: {name}')
    tmp = path.with_suffix('.part')
    tmp.write_text(json.dumps(data, default=encode, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    tmp.replace(path)


def read_json(name):
    return json.loads((OUT/name).read_text())


def memory_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == 'darwin' else rss*1024


def budget():
    if memory_bytes() >= MAX_BYTES:
        raise RuntimeError('Reached the frozen memory limit')


def seed_key(phi, split, episode):
    return (SEED, EXPERIMENT_ID, PHI_INDEX[phi], SPLIT_ID[split], episode)


def profile_risk(q, variances):
    risk = []
    for row in variances:
        cov = (q*row) @ q.T
        risk.append(.5*(np.log(np.diag(cov)).sum()-np.linalg.slogdet(cov)[1]))
    return float(np.mean(risk))


def generator(phi):
    eye = np.eye(4)
    alpha, tau, v = .02, .2, .1
    means = np.zeros((9, 4))
    means[1:5] = alpha*eye
    variances = np.full((9,4), v)
    variances[5:] += v*tau*eye
    offsets = means-means.mean(axis=0)
    K = np.diag(variances.mean(axis=0)) + offsets.T@offsets/9
    K_expected = v*(1+tau/9)*eye+alpha**2*eye/9-alpha**2*np.ones((4,4))/81
    co, si = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    def equation(eta):
        return sum((.6+eta)**h-(.6-eta)**h for h in LAGS)/3 - .04/(2*co)
    left, right = 0., .2
    assert equation(left) < 0 < equation(right)
    for _ in range(64):
        mid = (left+right)/2
        if equation(mid) < 0:
            left = mid
        else:
            right = mid
    eta = (left+right)/2
    A = np.diag([.6,.6,.9,0])
    A[:2,:2] += eta*np.array([[co,si],[si,-co]])
    BBT = eye - A@A.T - K
    B = np.linalg.cholesky(BBT)
    Mh = []
    lag_errors = []
    for h in LAGS:
        power = np.linalg.matrix_power(A, h)
        m_h = 2*eye - power - power.T
        # Independent construction from propagated noise and the initial state.
        cov = (power-eye)@(power-eye).T
        for k in range(h):
            ak = np.linalg.matrix_power(A,k)
            cov += ak@(B@B.T+K)@ak.T
        lag_errors.append(np.max(abs(cov-m_h)))
        Mh.append(m_h)
    M = np.mean(Mh, axis=0)
    base = np.array([-.7,.2,.55,.9])
    natural = np.concatenate(((base+offsets)/variances, -.5/variances), axis=1)
    contrast = natural[1:]-natural[:1]
    block1, block2 = np.eye(4)*alpha/v, np.eye(4)*tau/(2*v*(1+tau))
    eig = np.linalg.eigvalsh(M[:2,:2])
    risk_errors, gap_errors = [], []
    for name,q in candidates(phi):
        s = scores(q,M)
        if name.startswith('r'):
            radians = math.radians(int(name[1:]))
            f = math.log1p(math.sin(2*radians)**2/120)/9
            g = .04*(math.cos(2*radians)+math.tan(math.radians(phi))*math.sin(2*radians))
        elif name.startswith('block'):
            f = math.log1p(math.sin(math.radians(18))**2/120)/9
            cs = math.cos(math.radians(9))**2
            g = M[1,1]-(cs*M[0,0]+(1-cs)*M[2,2])
            assert set(np.argsort(s)[:2]) == {0,2}
        else:
            t = float(name[5:])
            f = .5*np.log1p(t*t*variances[:,1]/variances[:,0]).mean()
            g = .04*(1+math.tan(math.radians(phi))*t)/(1+t*t)
        risk_errors.append(abs(f-profile_risk(q,variances)))
        gap_errors.append(abs(g-(s[1]-s[0])))
    diagnostics = dict(K_error=np.max(abs(K-K_expected)), root_error=abs(equation(eta)),
        lyapunov_error=np.max(abs(A@A.T+B@B.T+K-eye)), lag_moment_error=max(lag_errors),
        gamma_error=abs(M[1,1]-M[0,0]-.04), chi_error=abs(2*M[0,1]+.04*si/co),
        contrast_block_error=max(np.max(abs(contrast[:4,:4]-block1)),np.max(abs(contrast[:4,4:])),
                                np.max(abs(contrast[4:,4:]-block2))),
        max_profile_risk_error=max(risk_errors), max_candidate_gap_error=max(gap_errors))
    assert diagnostics['root_error'] <= 1e-12 and max(diagnostics.values()) <= 1e-10
    assert np.linalg.matrix_rank(contrast) == 8 and M[2,2] < eig[0] <= eig[1] < M[3,3]
    assert np.min(np.linalg.eigvalsh(BBT))>0 and np.linalg.norm(A,2)<1
    return dict(A=A,B=B,D=variances,offsets=offsets,M=M,M_by_h=Mh,eta=eta,
                checks=diagnostics,rank=int(np.linalg.matrix_rank(contrast)),
                min_BBT_eigenvalue=np.linalg.eigvalsh(BBT)[0],
                guard_clearances=[eig[0]-M[2,2],M[3,3]-eig[1]])


def preflight():
    for line in (PLAN/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        assert digest(PLAN/name) == expected, name
    assert abs(tcrit(.975,31) - 2.0395134464) < 1e-9
    models = {str(phi):generator(phi) for phi in PHIS}
    save_json('generator.json',dict(models=models, passed=True, scientific_start_utc=now(),
        scientific_start_epoch=time.time(), protocol_sha256=digest(PLAN/'LOCAL_VALIDATION_PROTOCOL.md'),
        script_sha256=digest(Path(__file__)), shared_sha256=digest(OUT/'shared.py'), python=sys.version,
        numpy=np.__version__, platform=platform.platform(), rng='explicit numpy PCG64(SeedSequence(seed_key))',
        seed_key='[20261105,5,phi_index in (0,60,80),split_id cal=0 val=1,episode_index from 0]',
        draw_order='initial standard normal 4; 2048 regime IDs int64; 2048x4 actions; 2048x4 innovation normals',
        square_root='lower Cholesky',batch_size=16,threads=1,preflight_random_draws=0))
    print('Generator checks passed for all three configurations.')


def simulate(phi, split, n, model):
    dest=OUT/f'{split}_{phi}.npz'
    if dest.exists():
        raise RuntimeError(f'Refusing to regenerate {dest.name}')
    A,B,D,offsets = (np.array(model[key]) for key in ('A','B','D','offsets'))
    result = dict(episode_index=np.arange(n),
                  seed_keys=np.array([seed_key(phi,split,i) for i in range(n)]),
                  mean=np.zeros((n,4)),second=np.zeros((n,4,4)),M_by_h=np.zeros((n,3,4,4)),
                  counts=np.zeros((n,9),dtype='int64'),innov=np.zeros((n,9,4,4)))
    started=time.monotonic()
    for begin in range(0,n,16):
        end=min(begin+16,n)
        block=slice(begin,end)
        states=np.empty((end-begin,T+1,4))
        drives=np.empty((end-begin,T,4))
        for j,i in enumerate(range(begin,end)):
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_key(phi,split,i))))
            states[j,0]=rng.standard_normal(4)
            u=rng.integers(0,9,size=T,dtype='int64')
            action=rng.standard_normal((T,4))
            innovation=rng.standard_normal((T,4))*np.sqrt(D[u])
            drives[j]=action@B.T+offsets[u]+innovation
            result['counts'][i]=np.bincount(u,minlength=9)
            for k in range(9):
                noise=innovation[u==k]
                result['innov'][i,k]=noise.T@noise
        for t in range(T):
            states[:,t+1]=states[:,t]@A.T + drives[:,t]
        result['mean'][block]=states.mean(axis=1)
        result['second'][block]=states.transpose(0,2,1)@states/(T+1)
        for ih,h in enumerate(LAGS):
            increments=states[:,h:]-states[:,:-h]
            result['M_by_h'][block,ih]=increments.transpose(0,2,1)@increments/(T+1-h)
        if begin==0 and phi==80:
            # Verify reduction and sign against actual transformed time series using the same episode.
            for name,q in candidates(phi):
                if name not in ('r-9','r+9','block+9','shear-0.22','shear+0.22'): continue
                y=states[0]@q.T
                direct=np.mean([np.square(y[h:]-y[:-h]).mean(axis=0) for h in LAGS],axis=0)
                assert np.max(abs(direct-scores(q,result['M_by_h'][0].mean(axis=0)))) < 1e-10
        budget()
    result['M']=result['M_by_h'].mean(axis=1)
    assert np.all(result['counts'].sum(axis=1)==T)
    assert all(np.isfinite(arr).all() for arr in result.values())
    tmp=dest.with_suffix('.part')
    with tmp.open('xb') as stream:
        np.savez_compressed(stream,**result)
    tmp.replace(dest)
    return dict(phi=phi,split=split,episodes=n,file=dest.name,sha256=digest(dest),
                elapsed_seconds=time.monotonic()-started,peak_rss_bytes=memory_bytes())


def predictions(phi, batch):
    M=batch['M'].mean(axis=0)
    mu=batch['mean'].mean(axis=0)
    cov=batch['second'].mean(axis=0)-np.outer(mu,mu)
    gamma=M[1,1]-M[0,0]
    chi=2*M[0,1]
    rows={}
    for name,q in candidates(phi):
        s=scores(q,M)
        scaled=s/scores(q,cov)
        g=float(s[1]-s[0])
        if name.startswith('r'):
            theta=math.radians(int(name[1:]))
            ablated=gamma*math.cos(2*theta)
            assert abs(g-(ablated-chi*math.sin(2*theta))) < 1e-10
        elif name.startswith('shear'):
            t=float(name[5:])
            ablated=gamma/(1+t*t)
            assert abs(g-(gamma-chi*t)/(1+t*t)) < 1e-10
        else:
            diag_only=scores(q,np.diag(np.diag(M)))
            ablated=diag_only[1]-diag_only[0]
        rows[name]=dict(gap=g,scores=s,scaled_scores=scaled,ablated_gap=ablated)
    return rows


def calibration(models):
    logs=[]
    phi=80
    logs.append(simulate(phi,'cal',64,models[str(phi)]))
    gate={}
    qs=dict(candidates(phi))
    with np.load(OUT/'cal_80.npz') as batch:
        preds={str(phi):predictions(phi,batch)}
        for name in ('r-9','r+9'):
            ep=scores(qs[name],batch['M'])
            sd=float((ep[:,1]-ep[:,0]).std(ddof=1))
            pop=scores(qs[name],np.array(models[str(phi)]['M']))
            oracle=float(pop[1]-pop[0])
            half=tcrit(.9875,255)*sd/16
            gate[name]=dict(calibration_sd=sd, projected_half_width=half,
                            allowed_half_width=.8*abs(oracle), oracle_gap=oracle,
                            frozen_calibration_gap=preds['80'][name]['gap'],passed=bool(half<=.8*abs(oracle)))
    passed=all(r['passed'] for r in gate.values())
    if passed:
        for phi in (0,60):
            logs.append(simulate(phi,'cal',64,models[str(phi)]))
            with np.load(OUT/f'cal_{phi}.npz') as batch:
                preds[str(phi)]=predictions(phi,batch)
    save_json('calibration_lock.json',dict(frozen_utc=now(), gate_passed=passed, gate=gate,
        predictions=preds,episode_files=logs,generator_sha256=digest(OUT/'generator.json'),
        validation_episodes_generated_at_lock=0, t_quantile=tcrit(.9875,255)))
    print(json.dumps(dict(gate_passed=passed,gate=gate),default=encode))


def validation(models):
    lock=read_json('calibration_lock.json')
    if not lock['gate_passed']:
        raise RuntimeError('Calibration gate closed')
    assert lock['generator_sha256']==digest(OUT/'generator.json')
    for row in lock['episode_files']:
        assert digest(OUT/row['file'])==row['sha256']
    assert all(set(lock['predictions'][str(p)])==set(dict(candidates(p))) for p in PHIS)
    frozen_hash=digest(OUT/'calibration_lock.json')
    logs=[simulate(p,'val',256,models[str(p)]) for p in PHIS]
    assert digest(OUT/'calibration_lock.json')==frozen_hash
    save_json('validation_record.json',dict(ended_utc=now(),calibration_lock_sha256=frozen_hash,
        episode_files=logs, total_validation_episodes=sum(r['episodes'] for r in logs),
        elapsed_from_preflight_seconds=time.time()-read_json('generator.json')['scientific_start_epoch']))
    print('Validation complete. 256 independent episodes per configuration.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight','calibration','validation'))
    phase=parser.parse_args().phase
    if phase=='preflight':
        preflight()
        return
    config=read_json('generator.json')
    assert config['script_sha256']==digest(Path(__file__)) and config['shared_sha256']==digest(OUT/'shared.py')
    time_left=LIMIT_SECONDS - (time.time()-config['scientific_start_epoch'])
    if time_left<=0:
        raise RuntimeError('Reached the frozen wall time limit')
    signal.alarm(math.ceil(time_left))
    if phase=='calibration':
        calibration(config['models'])
    else:
        validation(config['models'])


if __name__=='__main__':
    main()
