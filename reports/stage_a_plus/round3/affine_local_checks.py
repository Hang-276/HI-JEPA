"""Deterministic finite-matrix checks. No samples, trajectories, NN or fitting."""
from pathlib import Path
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parent
DIM=4
PAIRS=[(i,j) for i in range(DIM) for j in range(i+1,DIM)]


def offdiag_vector(b):
    return np.array([value for i,j in PAIRS for value in (b[i,j],b[j,i])])


def score(a,sigma,moment):
    return np.diag(a@moment@a.T)/np.diag(a@sigma@a.T)


diagonals=np.ones((9,DIM))
for axis in range(DIM):
    diagonals[axis+5,axis]=9.0
R=float(max(np.max(x[:,None]/x[None,:]) for x in diagonals))
hessian=np.zeros((DIM*(DIM-1),DIM*(DIM-1)))
for index,(i,j) in enumerate(PAIRS):
    hessian[2*index:2*index+2,2*index:2*index+2]=[
        [np.mean(diagonals[:,j]/diagonals[:,i]),1.0],
        [1.0,np.mean(diagonals[:,i]/diagonals[:,j])]]
kappa=float(np.linalg.eigvalsh(hessian).min())
r0=min(0.5,3*kappa/16,math.sqrt(kappa/(2*R*R)))

sigma=np.array([[1,.2,.05,-.1],[.2,1,.15,.02],
                [.05,.15,1,.1],[-.1,.02,.1,1.]])
moment=np.array([[.3,.08,.01,.03],[.08,.6,.12,.02],
                 [.01,.12,1.3,.09],[.03,.02,.09,1.7]])
assert np.linalg.eigvalsh(sigma).min()>0
assert np.linalg.eigvalsh(moment).min()>0
assert np.linalg.eigvalsh(4*sigma-moment).min()>0
ks=float(np.linalg.norm(sigma,ord=2))
sj=np.diag(moment)
base=np.array([[0,.3,-.1,.2],[-.2,0,.1,-.3],
               [.1,.2,0,-.1],[-.05,.1,-.2,0.]])
base=base/np.linalg.norm(base)


def risk_and_budgets(b):
    # Exact log-diagonal minus log-det, in the fixed diag(A)=1 chart.
    sign,logdet=np.linalg.slogdet(np.eye(DIM)+b)
    assert sign>0
    q=np.array([[sum(b[i,j]**2*d[j]/d[i] for j in range(DIM) if i!=j)
                 for i in range(DIM)] for d in diagonals])
    risk=float(.5*np.log1p(q).sum(axis=1).mean()-logdet)
    vector=offdiag_vector(b)
    quadratic=float(vector@hessian@vector/2)
    fro=float(np.linalg.norm(b))
    op=float(np.linalg.norm(b,ord=2))
    c=fro/(3*(1-fro))+R*R*fro*fro/4
    delta_log=float(.5*(q-np.log1p(q)).sum(axis=1).mean())
    delta_det=fro*fro*op/(3*(1-op))
    return risk,quadratic,fro,op,c,delta_log,delta_det


fixtures=[]
for magnitude in (r0*.25,r0*.5,r0*.9,0.15):
    b=base*magnitude
    risk,quadratic,fro,op,c,delta_log,delta_det=risk_and_budgets(b)
    actual=score(np.eye(DIM)+b,sigma,moment)
    linear=[]
    remainder=[]
    constants=[]
    for j in range(DIM):
        tj=moment-sj[j]*sigma
        linear.append(float(2*tj[j,:]@b[j,:]))
        astar=float(np.linalg.norm(tj[:,j]))
        tstar=float(np.linalg.norm(tj,ord=2))
        constants.append((tstar+4*astar*math.sqrt(ks)+2*astar*ks*fro)
                         /(1-math.sqrt(ks)*fro)**2)
        remainder.append(float(actual[j]-sj[j]-linear[j]))
    linear=np.array(linear)
    remainder=np.array(remainder)
    applicable=c<=kappa/4
    target_gaps=[]
    for target in (0,1):
        for outsider in (2,3):
            grad=np.zeros((DIM,DIM))
            grad[outsider,:]=2*(moment[outsider,:]-sj[outsider]*sigma[outsider,:])
            grad[target,:]=-2*(moment[target,:]-sj[target]*sigma[target,:])
            np.fill_diagonal(grad,0)
            ell=offdiag_vector(grad)
            psi=float(ell@np.linalg.solve(hessian,ell))
            gamma=float(sj[outsider]-sj[target])
            cab=max(constants[target],constants[outsider])
            lower=gamma-2*math.sqrt(max(0,psi*risk))-4*cab*risk/kappa
            exact=float(actual[outsider]-actual[target])
            target_gaps.append({
                "target_outsider":[target,outsider],"gamma":gamma,
                "psi":psi,"exact_gap":exact,"risk_lower_bound":lower,
                "linear_coefficient_error":abs(float(ell@offdiag_vector(b))
                                               -(linear[outsider]-linear[target])),
                "bound_holds_when_applicable":bool(not applicable or exact>=lower-1e-12),
            })
    fixtures.append({
        "fro_radius":fro,"operator_norm":op,"uniform_c":c,
        "risk":risk,"quadratic_term":quadratic,
        "absolute_taylor_remainder":abs(risk-quadratic),
        "uniform_remainder_budget":c*fro*fro,
        "pointwise_delta_log":delta_log,"pointwise_delta_det_bound":delta_det,
        "uniform_neighborhood_condition":applicable,
        "uniform_remainder_holds":bool(abs(risk-quadratic)<=c*fro*fro+1e-12),
        "pointwise_remainder_holds":bool(abs(risk-quadratic)<=delta_log+delta_det+1e-12),
        "coercivity_when_applicable":bool(not applicable or risk>=quadratic/2-1e-12),
        "score_remainders":remainder.tolist(),
        "score_remainder_budgets":[float(constants[j]*sum(b[j,:]**2)) for j in range(DIM)],
        "score_remainder_bounds_hold":all(abs(remainder[j])<=constants[j]*sum(b[j,:]**2)+1e-12
                                            for j in range(DIM)),
        "target_gap_checks":target_gaps,
    })

# A zero-gradient fixture must retain its nonzero second-order score shift.
independent_b=np.zeros((DIM,DIM))
independent_b[0,1]=0.1
independent_scores=score(np.eye(DIM)+independent_b,np.eye(DIM),np.diag(sj))
independent_exact=sj.copy()
independent_exact[0]=(sj[0]+0.1**2*sj[1])/(1+0.1**2)

# Single shear at t=.4 lies beyond the conservative general-affine r0.
single_b=np.zeros((DIM,DIM))
single_b[0,1]=0.4
single_risk,*_=risk_and_budgets(single_b)
single_closed=float(.5*np.log1p(0.4**2*diagonals[:,1]/diagonals[:,0]).mean())

checks={
    "positive_kappa_affine":kappa>0,
    "explicit_radius_satisfies_condition":r0/(3*(1-r0))+R*R*r0*r0/4<=kappa/4+1e-12,
    "uniform_taylor_remainders":all(x["uniform_remainder_holds"] for x in fixtures),
    "pointwise_taylor_remainders":all(x["pointwise_remainder_holds"] for x in fixtures),
    "finite_neighborhood_coercivity":all(x["coercivity_when_applicable"] for x in fixtures),
    "normalized_score_remainders":all(x["score_remainder_bounds_hold"] for x in fixtures),
    "target_dual_linear_coefficients":all(y["linear_coefficient_error"]<1e-12
                                          for x in fixtures for y in x["target_gap_checks"]),
    "target_risk_gap_bounds":all(y["bound_holds_when_applicable"]
                                 for x in fixtures for y in x["target_gap_checks"]),
    "independent_ar_second_order_formula":float(np.max(abs(
        independent_scores-independent_exact)))<1e-12,
    "independent_ar_nonzero_shift":abs(independent_scores[0]-sj[0])>1e-4,
    "single_shear_exact_formula":abs(single_risk-single_closed)<1e-12,
}
checks={key:bool(value) for key,value in checks.items()}
report={
    "kind":"deterministic_matrix_formula_checks_only",
    "no_samples_or_trajectories_generated":True,"no_model_fitting_training_or_forward":True,
    "does_not_validate_actual_fine_population_assumptions":True,
    "dimension":DIM,"variance_ratio_bound":R,"kappa_affine":kappa,
    "sufficient_fro_radius":r0,"fixtures":fixtures,
    "independent_ar_scores":independent_scores.tolist(),
    "single_shear_risk":single_risk,"single_shear_closed_form":single_closed,
    "checks":checks,"passed":all(checks.values()),
}
out=ROOT/"AFFINE_LOCAL_CHECKS.json"
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"passed":report["passed"],"checks":checks,"output":str(out)},
                 ensure_ascii=False,indent=2))
assert report["passed"]
