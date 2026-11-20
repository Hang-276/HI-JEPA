"""All directed one-shear strict-top-k boundaries from frozen physical moments.

No fitting, neural forward, new physical samples, randomness, or FINAL reads.
Finite-array score moments and analytic free-profile risk are different objects.
"""
from pathlib import Path
from decimal import Decimal, localcontext
import hashlib
import json
import math
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
PREV=OUT.parent/'round2'
BASE=ROOT/'outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2'
SCOPE=OUT/'SHEAR_AUDIT_SCOPE.md'
H=(4,16,32)
ACCESS=[]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path,member=None):
    path=Path(path).resolve()
    assert 'FINAL' not in str(path)
    ACCESS.append(dict(path=str(path),member=member,sha256=sha(path)))
    if member is None:
        return json.loads(path.read_text())
    assert member in ('train__physical','cal__physical')
    with np.load(path,allow_pickle=False)as data:
        return np.asarray(data[member],np.float64)

def nearest_root(g,beta,alpha):
    assert g>0
    with localcontext() as ctx:
        ctx.prec=80
        g,b,a=map(Decimal.from_float,(float(g),float(beta),float(alpha)))
        disc=b*b-a*g
        if a==0 and b==0:
            return dict(kind='no_finite_root_limiting_tie',radius=None,root=None)
        if a>0 and disc<0:
            return dict(kind='no_finite_root_positive_limit',radius=None,root=None)
        assert disc>=0
        denom=abs(b)+disc.sqrt()
        assert denom>0
        radius=g/denom
        # The first root has sign opposite beta; either sign if beta=0.
        t=-radius if b>0 else radius
        residual=g+2*b*t+a*t*t
        assert abs(residual)<Decimal('1e-60')*max(Decimal(1),abs(g))
        return dict(kind='tangent_tie' if disc==0 else 'crossing_tie',
                    radius=float(radius),root=float(t),
                    radius_decimal=str(radius),root_decimal=str(t),
                    discriminant_decimal=str(disc),polynomial_decimal_residual=str(residual))

def score_at(Sigma,M,i,j,t):
    A=np.eye(4);A[i,j]=t
    cov=A@Sigma@A.T;inc=A@M@A.T
    return np.diag(inc)/np.diag(cov)

def main():
    began=time.perf_counter();scope_sha=sha(SCOPE)
    prior=read(PREV/'NONLINEAR_REMAINDER_AUDIT.json')
    law=read(PREV/'INNOVATION_BUDGET_AUDIT.json')
    rows=[];matrix_error=0.;root_gap_error=0.;base_rows=[]
    for oracle in prior['oracles']:
        slot=oracle['slot'];target=oracle['target'];other=[k for k in range(4)if k not in target]
        mean=np.asarray(oracle['physical_TRAIN_mean']);std=np.asarray(oracle['physical_TRAIN_std'])
        data={split:(read(BASE/'private/TRAIN_CAL'/f'{slot}.npz',split+'__physical')-mean)/std
              for split in ('train','cal')}
        flat=data['train'].reshape(-1,4);center=flat-flat.mean(0)
        Sigma=center.T@center/len(center)
        assert np.linalg.eigvalsh(Sigma).min()>0
        D=np.asarray(next(x for x in law['noise_laws_and_state_gauge']if x['slot']==slot)
                     ['innovation_variance_by_regime_state_gauge'])
        for split,v in data.items():
            moments=[]
            for h in H:
                delta=(v[:,h:]-v[:,:-h]).reshape(-1,4)
                moments.append(delta.T@delta/len(delta))
            M=np.mean(moments,axis=0);S=np.diag(M)/np.diag(Sigma)
            margin=float(min(S[b]-S[a]for a in target for b in other))
            assert abs(margin-oracle['scores_by_split'][split]['target_margin'])<1e-12
            base_rows.append(dict(slot=slot,split=split,target=target,physical_scores=S.tolist(),
                target_margin=margin,Sigma_Q0_TRAIN=Sigma.tolist(),M_Qh=M.tolist(),
                state_count=list(v.shape),CAL_is_not_independent_confirmation=(split=='cal')))
            for i in range(4):
                for j in range(4):
                    if i==j:continue
                    base=dict(slot=slot,split=split,affected_axis=i,added_axis=j,target=target,
                              physical_target_margin=margin,variance_ratios=(D[:,j]/D[:,i]).tolist())
                    if margin<=0:
                        rows.append(dict(**base,status='BASE_TARGET_NOT_STRICT',
                                         strict_risk_barrier=0.,strict_parameter_radius=0.,constraints=[]))
                        continue
                    competitors=other if i in target else target
                    constraints=[]
                    for k in competitors:
                        # q / variance(row_i) = target-order gap; orientation by i's target role.
                        sign=-1 if i in target else 1
                        g=sign*(M[i,i]-S[k]*Sigma[i,i])
                        beta=sign*(M[i,j]-S[k]*Sigma[i,j])
                        alpha=sign*(M[j,j]-S[k]*Sigma[j,j])
                        root=nearest_root(g,beta,alpha)
                        constraints.append(dict(competing_axis=k,g=float(g),beta=float(beta),alpha=float(alpha),**root))
                        for t in (-.37,.19):
                            ss=score_at(Sigma,M,i,j,t)
                            direct=sign*(ss[i]-ss[k])
                            formula=(g+2*beta*t+alpha*t*t)/(Sigma[i,i]+2*t*Sigma[i,j]+t*t*Sigma[j,j])
                            matrix_error=max(matrix_error,abs(direct-formula))
                    finite=[c for c in constraints if c['radius'] is not None]
                    if not finite:
                        rows.append(dict(**base,status='NO_FINITE_STRICT_TIE',strict_risk_barrier=None,
                                         strict_parameter_radius=None,constraints=constraints))
                        continue
                    first=min(finite,key=lambda c:c['radius']);radius=first['radius'];t=first['root']
                    critical=score_at(Sigma,M,i,j,t)
                    gcrit=min(critical[b]-critical[a]for a in target for b in other)
                    root_gap_error=max(root_gap_error,abs(gcrit))
                    assert abs(gcrit)<1e-10
                    for tt in (-radius*(1-1e-6),radius*(1-1e-6)):
                        ss=score_at(Sigma,M,i,j,tt)
                        assert min(ss[b]-ss[a]for a in target for b in other)>0
                    F=float(.5*np.log1p(radius*radius*D[:,j]/D[:,i]).mean())
                    A=np.eye(4);A[i,j]=t
                    directF=float(np.mean([.5*(np.log(np.diag((A*d[None,:])@A.T)).sum()-np.log(d).sum())for d in D]))
                    assert abs(F-directF)<1e-12
                    rows.append(dict(**base,status='FINITE_STRICT_TIE',strict_parameter_radius=radius,
                        strict_risk_barrier=F,critical_signed_t=t,first_competing_axis=first['competing_axis'],
                        first_event_kind=first['kind'],critical_scores=critical.tolist(),critical_min_pair_gap=float(gcrit),
                        constraints=constraints,guarantee='strict unique top-k in this one-shear family only'))
    summary=[]
    for base in base_rows:
        group=[r for r in rows if r['slot']==base['slot']and r['split']==base['split']]
        finite=[r for r in group if r['status']=='FINITE_STRICT_TIE']
        best=min(finite,key=lambda r:r['strict_risk_barrier'])if finite else None
        summary.append(dict(slot=base['slot'],split=base['split'],base_target_margin=base['target_margin'],
            directed_shears=len(group),finite_ties=len(finite),
            no_finite_ties=sum(r['status']=='NO_FINITE_STRICT_TIE'for r in group),
            baseline_invalid=sum(r['status']=='BASE_TARGET_NOT_STRICT'for r in group),
            minimum_single_shear_barrier=best['strict_risk_barrier']if best else (0. if base['target_margin']<=0 else None),
            minimum_direction=[best['affected_axis'],best['added_axis']]if best else None,
            minimum_signed_t=best['critical_signed_t']if best else None,
            is_upper_bound_on_all_affine_failure_radius=True,
            is_lower_bound_safety_certificate_for_all_affine=False))
    assert len(rows)==96 and len(base_rows)==8 and sha(SCOPE)==scope_sha
    result=dict(scope='finite physical TRAIN/CAL moment one-shear strict-top-k boundaries; ideal analytic profile risk; no actual Fine risk measurement',
        script_sha256=sha(__file__),scope_sha256=scope_sha,horizons=list(H),regime_weight='uniform1/9',
        base_moments=base_rows,rows=rows,summary=summary,
        checks=dict(max_score_formula_error=matrix_error,max_root_gap_error=root_gap_error,
                    all_inside_radius_samples_strict=True,all_risk_identities=True),accesses=ACCESS,
        cost=dict(physical_array_members=8,new_fits=0,neural_training=0,model_forward=0,PRNG=0,new_physical_data=0,
                  TEST_FINAL_array_reads=0,server_GPU_operations=0,elapsed_seconds=time.perf_counter()-began))
    return result

if __name__=='__main__':
    destination=OUT/'SHEAR_BARRIER_AUDIT.json'
    if destination.exists():raise SystemExit('Preserve existing audit; refusing overwrite.')
    result=main();destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'summary':result['summary'],'checks':result['checks'],'cost':result['cost']},ensure_ascii=False,indent=2))
