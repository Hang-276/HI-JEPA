"""Fixed covariance/analytic fixtures only; no samples, fitting or trajectories."""
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent


def rotation(theta):
    return np.array([[math.cos(theta), math.sin(theta)],
                     [-math.sin(theta), math.cos(theta)]])


def contrast(c):
    sign, logdet = np.linalg.slogdet(c)
    assert sign > 0
    return float(0.5*(np.log(np.diag(c)).sum()-logdet))


def correlation(c):
    return float(c[0,1]/math.sqrt(c[0,0]*c[1,1]))


def l_function(z, aa, weights):
    return float(sum(w*a*z/(1+a*z) for w,a in zip(weights,aa)))


def inverse_l(value, aa, weights):
    if value >= l_function(1.0,aa,weights):
        return 1.0
    left,right = 0.0,1.0
    for _ in range(70):
        middle=(left+right)/2
        if l_function(middle,aa,weights) < value:
            left=middle
        else:
            right=middle
    return (left+right)/2


weights = [0.4,0.6]
regime_diagonals = [[0.08,0.12],[0.07,0.16]]
aa = [(x-y)**2/(4*x*y) for x,y in regime_diagonals]
theta = 0.2
z = math.sin(2*theta)**2
q = rotation(theta)
fixtures = []

for case in ("linear_partial_unrotation","nonlinear_invertible_shear"):
    per_regime = []
    for diagonal in regime_diagonals:
        d = np.diag(diagonal)
        core = q @ d @ q.T
        if case == "linear_partial_unrotation":
            transform = rotation(-0.06)
            residual_map = transform-np.eye(2)
            remainder_cov = residual_map @ core @ residual_map.T
            actual = transform @ core @ transform.T
            core_actual_cross = core @ transform.T
            actual_risk = contrast(actual)
        else:
            # Y1=Z1, Y2=Z2+k Z1^2. Joint Gaussian odd moments vanish.
            # The map is globally invertible and has determinant one.
            k = 0.4
            remainder_cov = np.diag([0.0,2*k*k*core[0,0]**2])
            actual = core+remainder_cov
            core_actual_cross = core.copy()
            actual_risk = float(0.5*np.log(
                actual[0,0]*actual[1,1]/np.linalg.det(core)))
        tau = np.sqrt(np.diag(remainder_cov)/np.diag(core))
        vector_distance_sq = 2-2*np.diag(core_actual_cross)/np.sqrt(
            np.diag(core)*np.diag(actual))
        corr_core, corr_actual = correlation(core),correlation(actual)
        pair_error_budget = min(2.0,float(2*tau.sum()))
        nu_sq = float(np.trace(np.linalg.solve(core,remainder_cov)))
        a = (diagonal[0]-diagonal[1])**2/(4*diagonal[0]*diagonal[1])
        per_regime.append({
            "tau":tau.tolist(), "nu_sq":nu_sq,
            "correlation_core":corr_core,
            "correlation_actual":corr_actual,
            "error_budget":pair_error_budget,
            "actual_risk":actual_risk,
            "g_actual":contrast(actual),
            "vector_bound":bool(np.all(vector_distance_sq <= 4*tau*tau+1e-12)),
            "correlation_difference_bound":bool(abs(corr_core-corr_actual)
                                                <= pair_error_budget+1e-12),
            "pair_information_bound":bool(-0.5*math.log(1-corr_actual**2)
                                          <= contrast(actual)+1e-12),
            "core_correlation_formula_error":abs(corr_core**2-a*z/(1+a*z)),
        })
    j = sum(w*x["actual_risk"] for w,x in zip(weights,per_regime))
    corr_core_l2 = math.sqrt(sum(w*x["correlation_core"]**2
                                 for w,x in zip(weights,per_regime)))
    corr_actual_l2 = math.sqrt(sum(w*x["correlation_actual"]**2
                                   for w,x in zip(weights,per_regime)))
    error_l2 = math.sqrt(sum(w*x["error_budget"]**2
                            for w,x in zip(weights,per_regime)))
    rho_from_risk = math.sqrt(-math.expm1(-2*j))
    angle_budget = inverse_l((rho_from_risk+error_l2)**2,aa,weights)
    fixtures.append({
        "case":case,"per_regime":per_regime,
        "actual_population_risk":j,
        "core_correlation_l2":corr_core_l2,
        "actual_correlation_l2":corr_actual_l2,
        "error_l2":error_l2,
        "jensen_risk_correlation_bound":rho_from_risk,
        "actual_z":z,"certified_z_upper":angle_budget,
        "jensen_bound_holds":bool(corr_actual_l2 <= rho_from_risk+1e-12),
        "risk_bridge_holds":bool(corr_core_l2 <= rho_from_risk+error_l2+1e-12),
        "angle_bridge_holds":bool(z <= angle_budget+1e-12),
    })

# Check full-dimensional Gaussian total correlation >= each pair information.
full = np.array([[1.0,0.25,0.12],[0.25,1.4,-0.08],[0.12,-0.08,0.8]])
assert np.linalg.eigvalsh(full).min() > 0
full_g = contrast(full)
all_pairs = []
for i in range(3):
    for k in range(i+1,3):
        rho = full[i,k]/math.sqrt(full[i,i]*full[k,k])
        pair_g = -0.5*math.log(1-rho*rho)
        all_pairs.append({"pair":[i,k],"pair_g":pair_g,
                          "full_g":full_g,"bound":pair_g <= full_g+1e-12})

# Algebraic inversion of C7; no actual model risk is supplied here.
z_star = 0.35
l_star = l_function(z_star,aa,weights)
error_assumed = 0.03
t = math.sqrt(l_star)-error_assumed
assert 0 < t < 1
risk_budget = -0.5*math.log1p(-t*t)
threshold_recovered = (math.sqrt(-math.expm1(-2*risk_budget))+error_assumed)**2

# Persistent AR shear counterexample: analytic moments only, no trajectories.
horizons=(4,16,32)
q_rate=0.9
s3=sum(2*(1-q_rate**h) for h in horizons)/len(horizons)
persistent_shear=[]
for eps in (0.1,0.03,0.01,0.003):
    slow_rate=math.sqrt(1-eps*eps)
    s2=sum(2*(1-slow_rate**h) for h in horizons)/len(horizons)
    true_scores=[2.0,s2,s3]
    actual_scores=[1+s2/2,s2,s3]
    expected_delta=(s2-2)/2
    direct_delta=actual_scores[0]-true_scores[0]
    true_set=sorted(sorted(range(3),key=lambda k:true_scores[k])[:2])
    actual_set=sorted(sorted(range(3),key=lambda k:actual_scores[k])[:2])
    innovation=np.diag([1.0,eps*eps,1-q_rate*q_rate])
    shear=np.array([[1.0,1.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    transformed= shear @ innovation @ shear.T
    j=0.5*math.log1p(eps*eps)
    persistent_shear.append({
        "epsilon":eps,"conditional_innovation_nu":eps,
        "profiled_observation_nll_excess":j,
        "profiled_nll_formula_error":abs(contrast(transformed)-j),
        "state_remainder_variance":1.0,
        "true_scores":true_scores,"actual_scores":actual_scores,
        "true_target_zero_based":true_set,"actual_selection_zero_based":actual_set,
        "first_axis_score_error":direct_delta,
        "n1_variance_correction_error":abs(direct_delta-expected_delta),
    })

# Nine-regime full-rank enhancement; contexts/parameters are enumerated, not sampled.
full_rank_shear=[]
affine_curvature=[]
for eps in (0.1,0.03,0.01,0.003):
    rates=np.array([0.1,math.sqrt(1-eps*eps),0.9,0.2])
    innovation_variance=1-rates*rates
    # The second component is defined analytically as epsilon^2; avoid
    # cancellation from re-squaring sqrt(1-epsilon^2) in its scale.
    innovation_variance[1]=eps*eps
    scales=np.sqrt(9*innovation_variance/19)
    means=np.zeros((9,4))
    stds=np.ones((9,4))
    for axis in range(4):
        means[axis+1,axis]=1
        means[axis+5,axis]=-1
        stds[axis+5,axis]=3
    means=means*scales
    d=(stds*scales)**2
    current=np.array([0.2,-0.3,0.1,0.4])
    conditional_means=rates*current+means
    natural=np.concatenate([conditional_means/d,-0.5/d],axis=1)
    contrasts=natural[1:]-natural[0]
    rank=int(np.linalg.matrix_rank(contrasts))
    innovation_second=np.diag(d.mean(axis=0))+means.T@means/9
    state_covariance=innovation_second/(1-rates[:,None]*rates[None,:])
    true_scores=np.array([sum(2*(1-rate**h) for h in horizons)/len(horizons)
                          for rate in rates])
    actual_scores=true_scores.copy()
    actual_scores[0]=(true_scores[0]+true_scores[1])/2
    ratios=d[:,1]/d[:,0]
    risk=float(np.log1p(ratios).mean()/2)
    nu2=float(ratios.mean())
    candidate_diag=d.copy()
    candidate_diag[:,0]=d[:,0]+d[:,1]
    all_std=np.concatenate([np.sqrt(d).ravel(),np.sqrt(candidate_diag).ravel()])
    true_set=sorted(np.argsort(true_scores)[:2].tolist())
    actual_set=sorted(np.argsort(actual_scores)[:2].tolist())
    full_rank_shear.append({
        "epsilon":eps,"natural_parameter_contrast_rank":rank,
        "state_covariance_identity_error":float(np.max(np.abs(state_covariance-np.eye(4)))),
        "minimum_conditional_innovation_variance":float(d.min()),
        "true_and_profile_candidate_std_range":[float(all_std.min()),float(all_std.max())],
        "std_within_logstd_minus6_plus2":bool(
            all_std.min()>=math.exp(-6) and all_std.max()<=math.exp(2)),
        "population_profiled_risk":risk,"mean_nu_sq":nu2,
        "nu_formula_error":abs(nu2-eps*eps/0.99*145/81),
        "true_scores":true_scores.tolist(),"actual_scores":actual_scores.tolist(),
        "true_target_zero_based":true_set,"actual_selection_zero_based":actual_set,
        "target_margin":float(min(true_scores[[0,3]])-max(true_scores[[1,2]])),
        "state_remainder_variance":1.0,"first_axis_scalar_affine_r2":0.5,
    })
    r_ratio=float(ratios.mean())
    s_ratio=float((1/ratios).mean())
    trace=r_ratio+s_ratio
    discriminant=math.sqrt((r_ratio-s_ratio)**2+4)
    lambda_min=2*(r_ratio*s_ratio-1)/(trace+discriminant)
    c_eps=eps*eps/0.99
    k_ratio=145/81
    # Exact 2x2 log-det contrast, evaluated with log1p for stable second differences.
    t=1e-6
    directions=[]
    for x,y in ((1.0,0.0),(1.0,-1.0),(0.3,0.7)):
        twice_f=float(np.mean(np.log1p(t*t*x*x*ratios)
                              +np.log1p(t*t*y*y/ratios)
                              -2*math.log1p(-t*t*x*y)))
        numerical_curvature=twice_f/(t*t)
        expected_curvature=x*x*r_ratio+2*x*y+y*y*s_ratio
        directions.append({
            "offdiagonal_direction":[x,y],
            "expected_second_derivative":expected_curvature,
            "log1p_second_difference":numerical_curvature,
            "relative_error":abs(numerical_curvature-expected_curvature)
                             /max(1e-20,abs(expected_curvature)),
        })
    affine_curvature.append({
        "epsilon":eps,"R":r_ratio,"S":s_ratio,
        "ratio_formula_error":max(abs(r_ratio-k_ratio*c_eps),abs(s_ratio-k_ratio/c_eps))
                              /max(1,s_ratio),
        "rotation_generator_curvature":r_ratio+s_ratio-2,
        "unit_frobenius_rotation_curvature":(r_ratio+s_ratio-2)/2,
        "unit_shear_curvature":r_ratio,"minimum_pair_affine_curvature":lambda_min,
        "lambda_asymptotic_coefficient":lambda_min/c_eps,
        "predicted_asymptotic_coefficient":k_ratio-1/k_ratio,
        "direction_checks":directions,
    })

checks = {
    "all_normalized_vector_bounds":all(
        y["vector_bound"] for x in fixtures for y in x["per_regime"]),
    "all_pair_correlation_bounds":all(
        y["correlation_difference_bound"] for x in fixtures for y in x["per_regime"]),
    "all_core_formulas":all(
        y["core_correlation_formula_error"]<1e-12
        for x in fixtures for y in x["per_regime"]),
    "all_jensen_risk_bounds":all(x["jensen_bound_holds"] for x in fixtures),
    "all_risk_bridges":all(x["risk_bridge_holds"] for x in fixtures),
    "all_angle_bridges":all(x["angle_bridge_holds"] for x in fixtures),
    "all_full_pair_information_bounds":all(x["bound"] for x in all_pairs),
    "c7_threshold_inversion":abs(threshold_recovered-l_star)<1e-12,
    "persistent_shear_wrong_selection":all(
        x["true_target_zero_based"]==[1,2]
        and x["actual_selection_zero_based"]==[0,1] for x in persistent_shear),
    "persistent_shear_nll_formula":all(
        x["profiled_nll_formula_error"]<1e-12 for x in persistent_shear),
    "persistent_shear_n1_formula":all(
        x["n1_variance_correction_error"]<1e-12 for x in persistent_shear),
    "persistent_shear_nonvanishing_score_error":abs(
        persistent_shear[-1]["first_axis_score_error"]+1)<0.0001,
    "nine_regime_fullrank":all(x["natural_parameter_contrast_rank"]==8
                               for x in full_rank_shear),
    "nine_regime_stationary_covariance":all(
        x["state_covariance_identity_error"]<1e-10 for x in full_rank_shear),
    "nine_regime_nu_formula":all(x["nu_formula_error"]<1e-12 for x in full_rank_shear),
    "nine_regime_wrong_selection_positive_margin":all(
        x["true_target_zero_based"]==[1,2] and x["actual_selection_zero_based"]==[0,1]
        and x["target_margin"]>0.58 for x in full_rank_shear),
    "nine_regime_epsilon_001_std_within_clamp":next(
        x["std_within_logstd_minus6_plus2"] for x in full_rank_shear
        if x["epsilon"]==0.01),
    "affine_hessian_ratio_formula":all(x["ratio_formula_error"]<1e-12
                                      for x in affine_curvature),
    "affine_hessian_factor_two":all(y["relative_error"]<2e-6
                                    for x in affine_curvature for y in x["direction_checks"]),
    "affine_hessian_positive_minimum":all(x["minimum_pair_affine_curvature"]>0
                                          for x in affine_curvature),
    "affine_hessian_asymptotic":abs(
        affine_curvature[-1]["lambda_asymptotic_coefficient"]
        -affine_curvature[-1]["predicted_asymptotic_coefficient"])<1e-8,
}
report = {
    "kind":"deterministic_matrix_and_analytic_checks_only",
    "no_random_data_or_physical_trajectories":True,
    "no_fitting_or_model_training":True,
    "does_not_validate_actual_fine_population_assumptions":True,
    "fixtures":fixtures,"full_pair_information":all_pairs,
    "conditional_risk_budget_inversion":{
        "z_star":z_star,"l_star":l_star,"assumed_error_l2":error_assumed,
        "required_true_risk_below":risk_budget,
        "recovered_l_threshold":threshold_recovered,
        "scope":"pure_core_ordering_only_unless_score_remainder_and_sampling_errors_zero",
    },
    "persistent_ar_shear_analytic_counterexample":persistent_shear,
    "nine_regime_fullrank_shear_analytic_counterexample":full_rank_shear,
    "general_affine_curvature_checks":affine_curvature,
    "checks":checks,"passed":all(checks.values()),
}
out=ROOT/"CORRELATION_RISK_CHECKS.json"
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"passed":report["passed"],"checks":checks,"path":str(out)},
                 ensure_ascii=False,indent=2))
assert report["passed"]
