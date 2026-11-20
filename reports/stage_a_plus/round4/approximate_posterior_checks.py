#!/usr/bin/env python3
"""Fixed probability-vector arithmetic only; no PRNG/model/data input."""
import hashlib
import json
import math
from pathlib import Path


def check_case(name, p, q, eps, diag):
    m = len(p)
    assert m == 9 and len(q) == m and len(diag) == m - 1
    assert abs(math.fsum(p) - 1) < 1e-14
    assert abs(math.fsum(q) - 1) < 1e-14
    assert min(p + q) >= eps - 1e-15
    kl = math.fsum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
    assert kl >= -1e-14
    kl = max(kl, 0.0)
    logdiff = [math.log(pi) - math.log(qi) for pi, qi in zip(p, q)]
    odds = [logdiff[u] - logdiff[0] for u in range(1, m)]
    odds_sq = math.fsum(x * x for x in odds)
    rhs_sq = 4 * (m - 1) * kl / (eps * eps)
    s = min(abs(x) for x in diag)
    residual_sq = math.fsum((x / d) ** 2 for x, d in zip(odds, diag))
    residual_rhs_sq = rhs_sq / (s * s)
    tol = 1e-12 * max(1.0, odds_sq, residual_sq)
    assert odds_sq <= rhs_sq + tol
    assert residual_sq <= odds_sq / (s * s) + tol
    assert residual_sq <= residual_rhs_sq + tol
    return {
        "name": name, "m": m, "epsilon": eps, "p": p, "q": q,
        "kl_p_q": kl, "log_odds_error_squared": odds_sq,
        "requested_log_odds_bound_squared": rhs_sq,
        "requested_bound_ratio": odds_sq / rhs_sq if rhs_sq else None,
        "head_contrast_diagonal": diag, "sigma_min": s,
        "statistic_span_residual_squared": residual_sq,
        "requested_span_bound_squared": residual_rhs_sq,
        "all_arithmetic_checks_pass": True,
    }


def main():
    diag = [0.25, 0.4, 0.6, 0.9, 1.1, 1.5, 2.0, 3.0]
    uni = [1 / 9] * 9
    cases = [
        ("identical_uniform", uni, uni, 0.1),
        ("small_pivot_transfer", uni,
         [1 / 9 + 0.001, 1 / 9 - 0.001] + [1 / 9] * 7, 0.1),
        ("boundary_pivot", [0.01] + [0.99 / 8] * 8,
         [0.02] + [0.98 / 8] * 8, 0.01),
        ("peaked_opposite", [0.6, 0.05] + [0.05] * 7,
         [0.05, 0.6] + [0.05] * 7, 0.05),
        ("alternating", [0.16, 0.06, 0.16, 0.06, 0.16, 0.06, 0.16, 0.06, 0.12],
         [0.06, 0.16, 0.06, 0.16, 0.06, 0.16, 0.06, 0.16, 0.12], 0.06),
        ("rare_class", [0.001, 0.199] + [0.8 / 7] * 7,
         [0.005, 0.195] + [0.8 / 7] * 7, 0.001),
    ]
    results = [check_case(name, list(p), list(q), eps, diag)
               for name, p, q, eps in cases]

    # Shared epsilon/s condition: integrate two fixed contexts, no sampling.
    context_weights = [0.3, 0.7]
    selected = [results[1], results[4]]
    eps_common = min(x["epsilon"] for x in selected)
    s_common = min(x["sigma_min"] for x in selected)
    a_avg = math.fsum(w * x["kl_p_q"] for w, x in zip(context_weights, selected))
    r_avg_sq = math.fsum(w * x["log_odds_error_squared"]
                        for w, x in zip(context_weights, selected))
    e_avg_sq = math.fsum(w * x["statistic_span_residual_squared"]
                        for w, x in zip(context_weights, selected))
    joint_bound = 4 * 8 * a_avg / eps_common ** 2
    assert r_avg_sq <= joint_bound + 1e-12
    assert e_avg_sq <= joint_bound / s_common ** 2 + 1e-12

    # Omega selects the first atom in this fixed finite probability space.
    omega_mass = context_weights[0]
    atom = selected[0]
    restricted_a = omega_mass * atom["kl_p_q"]
    restricted_e_sq = omega_mass * atom["statistic_span_residual_squared"]
    restricted_bound_sq = 4 * 8 * restricted_a / atom["epsilon"] ** 2 / s_common ** 2
    conditional_bound_sq = restricted_bound_sq / omega_mass
    assert restricted_e_sq <= restricted_bound_sq + 1e-12
    assert abs(conditional_bound_sq - atom["requested_span_bound_squared"]) < 1e-9

    script = Path(__file__).resolve()
    payload = {
        "kind": "deterministic fixed-probability-vector checks; not physical or model data",
        "m": 9, "natural_log": True, "case_count": len(results), "cases": results,
        "averaged_context_check": {
            "weights": context_weights, "shared_epsilon": eps_common,
            "shared_sigma_min": s_common, "average_kl": a_avg,
            "average_odds_error_squared": r_avg_sq,
            "odds_error_bound_squared": joint_bound,
            "average_span_error_squared": e_avg_sq,
            "span_error_bound_squared": joint_bound / s_common ** 2,
        },
        "omega_normalization_check": {
            "omega_mass": omega_mass, "unnormalized_restricted_kl": restricted_a,
            "unnormalized_span_error_squared": restricted_e_sq,
            "unnormalized_span_bound_squared": restricted_bound_sq,
            "conditional_span_bound_squared": conditional_bound_sq,
            "conditional_rhs_requires_dividing_unnormalized_rhs_squared_by_omega_mass": True,
        },
        "all_checks_pass": True,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "execution_counts": {"random_draws": 0, "physical_samples_generated": 0,
                             "model_loads": 0, "data_array_files_read": 0, "training_jobs": 0},
        "limitations": ["Does not check Gaussian/global epsilon compatibility",
                        "Does not establish posterior estimation",
                        "Does not establish approximate polynomial closure or axis-affine error"],
    }
    destination = script.with_name("APPROXIMATE_POSTERIOR_CHECKS.json")
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "case_count": len(results),
                      "all_checks_pass": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
