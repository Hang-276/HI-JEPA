"""Deterministic matrix and analytic checks. No samples, trajectories or training."""
from pathlib import Path
import json
import math
import numpy as np


ROOT = Path(__file__).resolve().parent


def spectral_power(c, exponent):
    vals, vecs = np.linalg.eigh(c)
    assert vals.min() > 0
    return (vecs * vals**exponent) @ vecs.T


def g(c):
    sign, logdet = np.linalg.slogdet(c)
    assert sign > 0
    return float(0.5 * (np.log(np.diag(c)).sum() - logdet))


def scores(covariance, increment_second_moment):
    return np.diag(increment_second_moment) / np.diag(covariance)


def exact_score_discrepancy(c, m, cross_state, residual_state,
                            cross_increment, residual_increment):
    v = np.diag(c)
    s = np.diag(m) / v
    c0 = np.diag(cross_state)
    vr = np.diag(residual_state)
    cr = np.diag(cross_increment)
    nr = np.diag(residual_increment)
    return (2 * cr + nr - s * (2 * c0 + vr)) / (v + 2 * c0 + vr)


# Fixed covariance/increment matrices only. No state vectors are generated.
sigma = np.array([[1.0, 0.25], [0.25, 1.0]])
moment = np.array([[0.60, 0.12], [0.12, 1.30]])
a = np.array([[1.05, 0.15], [-0.08, 0.92]])
u, _, vt = np.linalg.svd(a)
q = u @ vt
n = a - q
c = a @ sigma @ a.T
m = a @ moment @ a.T
cq = q @ sigma @ q.T
mq = q @ moment @ q.T

# A generic admissible joint second-moment example for X and remainder R.
# R=B X+an uncorrelated second-moment component; this is not asserted to be
# a fitted nonlinear encoder or a new physical process.
b = np.array([[0.03, -0.02], [0.015, 0.025]])
w = np.diag([0.0004, 0.0007])
cross_state = c @ b.T
residual_state = b @ c @ b.T + w
cross_increment = m @ b.T
residual_increment = b @ m @ b.T + 2 * w
cy = c + cross_state + cross_state.T + residual_state
my = m + cross_increment + cross_increment.T + residual_increment
delta_formula = exact_score_discrepancy(
    c, m, cross_state, residual_state, cross_increment, residual_increment)
delta_direct = scores(cy, my) - scores(c, m)

# Check Q -> A (nonorthogonal) -> Y telescoping using the same identity.
nonorth_cross_state = q @ sigma @ n.T
nonorth_residual_state = n @ sigma @ n.T
nonorth_cross_increment = q @ moment @ n.T
nonorth_residual_increment = n @ moment @ n.T
nonorth_delta = exact_score_discrepancy(
    cq, mq, nonorth_cross_state, nonorth_residual_state,
    nonorth_cross_increment, nonorth_residual_increment)
telescope = nonorth_delta + delta_formula
total_direct = scores(cy, my) - scores(cq, mq)

score_checks = {
    "exact_discrepancy_max_error": float(np.max(np.abs(delta_formula-delta_direct))),
    "polar_nonorthogonal_max_error": float(np.max(np.abs(
        nonorth_delta - (scores(c, m)-scores(cq, mq))))),
    "telescope_max_error": float(np.max(np.abs(telescope-total_direct))),
    "actual_discrepancies": delta_direct.tolist(),
    "q_orthogonality_error": float(np.linalg.norm(q @ q.T-np.eye(2))),
}

bound_cases = []
for diagonal in ([0.08, 0.12], [0.002, 0.07], [0.7, 1.3]):
    d = np.diag(diagonal)
    core = a @ d @ a.T
    # Fixed second-moment perturbations at two sizes; no random draws.
    for scale in (0.05, 0.15):
        b_res = scale * np.array([[0.15, -0.2], [0.1, 0.12]])
        w_res = scale**2 * np.diag([0.0003, 0.0004])
        residual_cov = b_res @ core @ b_res.T + w_res
        actual = (np.eye(2)+b_res) @ core @ (np.eye(2)+b_res).T+w_res
        root_inv = spectral_power(core, -0.5)
        nu2 = float(np.trace(root_inv @ residual_cov @ root_inv))
        nu = math.sqrt(nu2)
        relative = root_inv @ actual @ root_inv
        beta = float(np.linalg.norm(relative-np.eye(2), ord=2))
        nu_bound = 2 * math.log((1+nu)/(1-nu))
        beta_bound = math.log((1+beta)/(1-beta))
        gap = abs(g(actual)-g(core))
        relative_eigs = np.linalg.eigvalsh(relative)
        loewner_ok = bool(relative_eigs.min() >= (1-nu)**2-1e-12
                          and relative_eigs.max() <= (1+nu)**2+1e-12)
        bound_cases.append({
            "innovation_diagonal": list(diagonal),
            "scale": scale,
            "nu": nu,
            "beta_exact": beta,
            "contrast_difference": gap,
            "relative_nu_bound": nu_bound,
            "relative_beta_bound": beta_bound,
            "loewner_bounds_hold": loewner_ok,
            "contrast_bounds_hold": bool(gap <= min(nu_bound,beta_bound)+1e-12),
        })


def oscillatory_gaussian(n, amplitude=0.8):
    # Analytic Gaussian moments and an absolutely convergent Fourier series.
    # No quadrature over simulated states or random data is used.
    root = math.sqrt(1-amplitude**2)
    const = (1+root)/2
    fourier_ratio = amplitude/(1+root)
    log_jac = math.log(const)
    for k in range(1, 200):
        log_jac += (2 * ((-1)**(k+1)) * fourier_ratio**k
                    * math.exp(-0.5*(k*n)**2) / k)
    variance = (1 + 2*amplitude*math.exp(-n*n/2)
                + amplitude**2/(2*n*n)*(1-math.exp(-2*n*n)))
    remainder_l2 = amplitude/n*math.sqrt((1-math.exp(-2*n*n))/2)
    risk = 0.5*math.log(variance)-log_jac
    return {
        "n": n, "remainder_l2": remainder_l2,
        "variance": variance, "expected_log_jacobian": log_jac,
        "profiled_risk": risk,
    }


oscillation = [oscillatory_gaussian(n) for n in (1,2,4,8,16,32)]
expected_limit = -math.log((1+math.sqrt(1-0.8**2))/2)
checks = {
    "score_identity": score_checks["exact_discrepancy_max_error"] < 1e-12,
    "polar_identity": score_checks["polar_nonorthogonal_max_error"] < 1e-12,
    "telescope_identity": score_checks["telescope_max_error"] < 1e-12,
    "all_covariance_bounds": all(x["loewner_bounds_hold"]
                                  and x["contrast_bounds_hold"] for x in bound_cases),
    "oscillatory_remainder_decreases": all(
        y["remainder_l2"] < x["remainder_l2"] for x,y in zip(oscillation,oscillation[1:])),
    "oscillatory_nonzero_risk_limit": abs(
        oscillation[-1]["profiled_risk"]-expected_limit) < 0.001,
}
report = {
    "kind": "deterministic_formula_checks_only",
    "no_random_data_or_physical_trajectories_generated": True,
    "no_model_training_or_fitting": True,
    "does_not_verify_actual_fine_population_assumptions": True,
    "score_checks": score_checks,
    "conditional_covariance_checks": bound_cases,
    "high_frequency_gaussian_analytic_checks": oscillation,
    "high_frequency_positive_risk_limit": expected_limit,
    "checks": checks,
    "passed": all(checks.values()),
}
path = ROOT / "NONLINEAR_FORMULA_CHECKS.json"
path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({"passed": report["passed"], "checks": checks,
                  "max_score_formula_error": score_checks["exact_discrepancy_max_error"],
                  "output": str(path)}, ensure_ascii=False, indent=2))
assert report["passed"]
