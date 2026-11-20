"""Analytic distributions/matrices only; no samples, trajectories or trained models."""
from pathlib import Path
import itertools
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parent
DIM=4
theta=math.pi/4
rot=np.eye(DIM)
rot[:2,:2]=[[math.cos(theta),-math.sin(theta)],
            [math.sin(theta),math.cos(theta)]]


def gaussian_kl(mu_p,cov_p,mu_q,cov_q):
    delta=mu_q-mu_p
    return float(.5*(np.trace(np.linalg.solve(cov_q,cov_p))
                    +delta@np.linalg.solve(cov_q,delta)-len(mu_p)
                    +np.linalg.slogdet(cov_q)[1]-np.linalg.slogdet(cov_p)[1]))


rows=[]
for tau in (.2,.03,.003,.0003):
    means=np.zeros((9,DIM))
    variances=np.ones((9,DIM))
    for j in range(DIM):
        means[j+1,j]=tau
        variances[j+5,j]+=tau
    natural=np.concatenate([means/variances,-.5/variances],axis=1)
    contrasts=natural[1:]-natural[0]
    rank=int(np.linalg.matrix_rank(contrasts))
    direct_kl=np.mean([gaussian_kl(mu,np.diag(d),rot@mu,
                                  rot@np.diag(d)@rot.T)
                       for mu,d in zip(means,variances)])
    exact_kl=tau*tau/9*(2*(1-math.cos(theta))+math.sin(theta)**2/(1+tau))
    direct_reference=np.mean([gaussian_kl(mu,np.diag(d),np.zeros(DIM),np.eye(DIM))
                              for mu,d in zip(means,variances)])
    k0=2/9*(tau*tau+tau-math.log1p(tau))
    mean=means.mean(axis=0)
    omega=np.diag(variances.mean(axis=0))+means.T@means/9-np.outer(mean,mean)
    omega_formula=((1+tau/9+tau*tau/9)*np.eye(DIM)
                   -tau*tau/81*np.ones((DIM,DIM)))
    observed_cov=rot@omega@rot.T
    cross=rot@omega
    r2=cross*cross/(np.diag(observed_cov)[:,None]*np.diag(omega)[None,:])
    best=max(itertools.permutations(range(DIM)),
             key=lambda p:sum(r2[i,p[i]] for i in range(DIM)))
    matched=[float(r2[i,best[i]]) for i in range(DIM)]
    rows.append({
        "tau":tau,"natural_parameter_rank":rank,
        "natural_parameter_min_singular_value":float(np.linalg.svd(contrasts,compute_uv=False).min()),
        "innovation_variance_range":[float(variances.min()),float(variances.max())],
        "rotation_direct_kl":float(direct_kl),"rotation_analytic_kl":exact_kl,
        "rotation_kl_error":abs(direct_kl-exact_kl),
        "reference_direct_kl":float(direct_reference),"reference_analytic_kl":k0,
        "reference_kl_error":abs(direct_reference-k0),
        "reference_kl_upper":tau*tau/3,
        "unconditional_covariance_formula_error":float(np.max(abs(omega-omega_formula))),
        "wrong_identity_encoder_best_affine_axis_r2":matched,
        "wrong_identity_encoder_axis_mean_r2":sum(matched)/DIM,
    })


def twist_projection_complex(alpha,terms):
    return sum((1j*alpha)**k/(math.factorial(k)*(1+2*k)**2) for k in range(terms))


alpha=1.0
m80=twist_projection_complex(alpha,80)
m100=twist_projection_complex(alpha,100)
delta2=2*(1-abs(m100)**2)
aa=np.array([[m100.real,-m100.imag],[m100.imag,m100.real]])
projection_energy=2-float(np.trace(aa@aa.T))
jacobian_checks=[]
for z in (np.array([0.,0.]),np.array([.2,.4]),np.array([1.,-.3]),np.array([2.,.5])):
    radius2=float(z@z)
    angle=alpha*math.exp(-radius2)
    rr=np.array([[math.cos(angle),-math.sin(angle)],
                 [math.sin(angle),math.cos(angle)]])
    grad=-2*alpha*math.exp(-radius2)*z
    jz=np.array([-z[1],z[0]])
    jac=rr@(np.eye(2)+np.outer(jz,grad))
    limit=1+2*abs(alpha)/math.e
    jacobian_checks.append({
        "analytic_radius_squared":radius2,
        "determinant":float(np.linalg.det(jac)),
        "operator_norm":float(np.linalg.norm(jac,ord=2)),
        "inverse_operator_norm":float(np.linalg.norm(np.linalg.inv(jac),ord=2)),
        "uniform_norm_limit":limit,
    })

finite_lower_bounds=[]
for n in (16,256,4096,65536):
    tau=math.sqrt(3)/(16*math.sqrt(n))
    k0=2/9*(tau*tau+tau-math.log1p(tau))
    tv_upper=min(1.,2*math.sqrt(n*k0))
    finite_lower_bounds.append({
        "independent_observation_count_in_construction":n,"tau":tau,
        "nonlinear_twist_tv_upper":tv_upper,
        "soundness_error_plus_lack_of_power_lower":1-tv_upper,
        "maximum_binary_error_lower":(1-tv_upper)/2,
        "scope":"distance_bound_only; positive residual threshold also requires sufficiently small tau",
    })

checks={
    "all_positive_tau_models_rank8":all(x["natural_parameter_rank"]==8 for x in rows),
    "innovation_variances_uniformly_nondegenerate":all(
        x["innovation_variance_range"][0]>=1 for x in rows),
    "rotation_kl_formula":all(x["rotation_kl_error"]<1e-12 for x in rows),
    "reference_kl_formula":all(x["reference_kl_error"]<1e-12 for x in rows),
    "reference_kl_quadratic_bound":all(
        x["reference_analytic_kl"]<=x["reference_kl_upper"]+1e-15 for x in rows),
    "latent_covariance_formula":all(x["unconditional_covariance_formula_error"]<1e-12 for x in rows),
    "identity_encoder_not_axis_recovered_in_rotated_model":all(
        x["wrong_identity_encoder_axis_mean_r2"]<.8 for x in rows),
    "twist_series_converged":abs(m80-m100)<1e-14,
    "twist_positive_best_affine_remainder":delta2>1e-3,
    "twist_projection_energy_formula":abs(delta2-projection_energy)<1e-12,
    "analytic_twist_determinant_one":all(abs(x["determinant"]-1)<1e-12 for x in jacobian_checks),
    "analytic_twist_jacobian_uniform_bound":all(
        max(x["operator_norm"],x["inverse_operator_norm"])<=x["uniform_norm_limit"]+1e-12
        for x in jacobian_checks),
    "finite_data_nontrivial_lower_bounds":all(
        x["maximum_binary_error_lower"]>=.375 for x in finite_lower_bounds),
}
checks={k:bool(v) for k,v in checks.items()}
report={
    "kind":"deterministic_gaussian_distribution_and_series_checks",
    "no_random_samples_or_physical_trajectories":True,
    "no_trained_model_load_forward_or_training":True,
    "no_existing_scientific_arrays_read":True,
    "fullrank_rotation_models":rows,
    "nonlinear_twist_best_affine_remainder":{
        "alpha":alpha,"projection_complex_mean_real":m100.real,
        "projection_complex_mean_imag":m100.imag,"best_affine_residual_energy_limit":delta2},
    "analytic_jacobian_fixtures":jacobian_checks,
    "finite_data_distance_bounds":finite_lower_bounds,
    "checks":checks,"passed":all(checks.values())}
out=ROOT/"UNLABELED_LIMITS_CHECKS.json"
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"passed":report["passed"],"checks":checks,
                  "twist_residual_energy_limit":delta2,"path":str(out)},
                 ensure_ascii=False,indent=2))
assert report["passed"]
