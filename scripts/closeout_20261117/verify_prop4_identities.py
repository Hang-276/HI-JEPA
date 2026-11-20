#!/usr/bin/env python3
"""Deterministic algebraic and numerical checks for Proposition 4 (Two-Regime Fast-Axis
Degradation Dichotomy & LTT Binomial Sample-Complexity Floor) in L4 closeout."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/closeout_20261117/l4_theory_checks"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lags = np.array([4, 16, 32], dtype=np.float64)

    # Exact population slowness of visual_sprites formal_v1 rho = (0.964, 0.961, 0.780, 0.760)
    rho = np.array([0.964, 0.961, 0.780, 0.760], dtype=np.float64)
    mu = np.array([2.0 * np.mean(1.0 - r**lags) for r in rho])
    mu_slow_max = float(max(mu[0], mu[1]))
    mu_fast_min = float(min(mu[2], mu[3]))
    mu_white = 2.0

    # Check Regime II-A (pure attenuation + white noise, lambda_leak = 0, sigma > 0):
    # s_bar_f(s, 0, sigma) = (s * mu_f + 2 * sigma^2) / (s + sigma^2) = mu_f + sigma^2 * (2 - mu_f) / (s + sigma^2) > mu_f!
    s_vals = np.linspace(0.0, 1.0, 21)
    sigma = 0.3
    s_bar_iia = (s_vals * mu[2] + sigma**2 * mu_white) / (s_vals + sigma**2)
    assert np.all(s_bar_iia >= mu[2] - 1e-15), "Regime II-A must shift population score UPWARD toward 2.0"

    # Check Regime II-B (collapsed fast signal s -> 0 with slow leakage lambda_leak > 0):
    # Critical slow-leakage relative variance share alpha_slow = lambda^2 / (s + lambda^2 + sigma^2)
    # Population inversion s_bar_f < b occurs iff alpha_slow * (b - mu_d) > alpha_fast * (mu_f - b) + alpha_noise * (mu_xi - b)
    # Equivalently, when s=0, alpha_slow > (mu_xi - b) / (mu_xi - mu_d).
    alpha_slow_pop_crit_s0 = (mu_white - mu_slow_max) / (mu_white - mu[0])

    # Compare against real visual_sprites L1 measurements
    l1 = json.loads((ROOT / "results/closeout_20261117/l1_visual_sprites/l1_visual_sprites_diagnostics.json").read_text())
    real_checks = {}
    for cell in l1["cells"]:
        if cell["arm"] == "long_horizon_pred":
            dy = cell["fast_axis_orthogonal_decomposition"]["distractor_y"]
            pred_s = cell["mean_normalized_scores_test"]["predicted"]["distractor_y"]
            true_s = cell["mean_normalized_scores_test"]["true"]["distractor_y"]
            real_checks[cell["cell"]] = {
                "distractor_y_true_mean_score": true_s,
                "distractor_y_predicted_mean_score": pred_s,
                "score_shift_pred_minus_true": pred_s - true_s,
                "slow_leakage_share_ref": dy["ref_slow_leakage_share_total"],
                "fast_signal_share_ref": dy["ref_fast_signal_share_total"],
                "regime": "II-B_pseudo_slow_hijack" if pred_s < true_s else "II-A_upward_noise_attenuation",
            }

    # Verify seed101 & seed103 shifted UPWARD (Regime II-A: 1.790 -> 1.88/1.89) while seed102 shifted DOWNWARD (Regime II-B: 1.790 -> 1.369)!
    assert real_checks["long_horizon_pred_seed101"]["score_shift_pred_minus_true"] > 0.05
    assert real_checks["long_horizon_pred_seed103"]["score_shift_pred_minus_true"] > 0.05
    assert real_checks["long_horizon_pred_seed102"]["score_shift_pred_minus_true"] < -0.40

    # Binomial sample complexity checks
    sample_floors = {}
    for label, alpha in [("single_005", 0.05), ("sprites_bonf6", 0.05 / 6), ("fourroom_bonf12", 0.05 / 12), ("lowdim_bonf128", 0.05 / 128)]:
        n_req = math.ceil(math.log(alpha) / math.log(0.99))
        # Verify n_req passes <= 0.01 and n_req - 1 fails > 0.01
        ub_pass = 1.0 - alpha ** (1.0 / n_req)
        ub_fail = 1.0 - alpha ** (1.0 / (n_req - 1))
        assert ub_pass <= 0.01 < ub_fail
        sample_floors[label] = {"alpha": alpha, "min_zero_error_accepted_windows": n_req, "cp_upper_at_n_min": ub_pass}

    receipt = {
        "evidence_tier": "deterministic_algebraic_and_real_anchor_check",
        "status": "PASSED",
        "visual_sprites_population_slowness": {
            "rho": rho.tolist(),
            "mu": mu.tolist(),
            "mu_slow_max": mu_slow_max,
            "mu_fast_min": mu_fast_min,
            "alpha_slow_population_inversion_threshold_at_s0_white_residual": alpha_slow_pop_crit_s0,
        },
        "real_long_horizon_pred_regime_verification": real_checks,
        "binomial_1pct_certification_sample_floors": sample_floors,
    }
    (OUT / "prop4_and_ltt_floor_checks.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
