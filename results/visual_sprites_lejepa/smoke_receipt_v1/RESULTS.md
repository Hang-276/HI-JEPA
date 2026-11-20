# Two-Sprite visual comparison

Evidence tier: `development_smoke`
Device: cpu
Elapsed seconds: 7.2
Per-cell Bonferroni alpha: 0.025000000
Forecast files verified: 12

| Arm and seed | agent_x R² | agent_y R² | distractor_x R² | distractor_y R² | Eligible | Coverage | Errors / accepted | CP upper | Result |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| visual_lejepa_seed1901 | -0.006279 | 0.102210 | -0.008276 | 0.008488 | False | 12.5000% | 1/1 | 100.0000% | SMOKE_ONLY |
| long_horizon_pred_seed1901 | 0.013186 | 0.114323 | -0.010261 | 0.004301 | False | 12.5000% | 0/1 | 97.5000% | SMOKE_ONLY |

| Arm and seed | Subset | All-answer errors / windows (risk) | own_cal errors / accepted (risk) |
|---|---|---:|---:|
| visual_lejepa_seed1901 | regular | 3/4 (75.0000%) | 1/1 (100.0000%) |
| visual_lejepa_seed1901 | flip | 3/4 (75.0000%) | 0/0 (undefined) |
| long_horizon_pred_seed1901 | regular | 2/4 (50.0000%) | 0/1 (0.0000%) |
| long_horizon_pred_seed1901 | flip | 3/4 (75.0000%) | 0/0 (undefined) |

All figures here come from a tiny engineering fixture on disjoint development streams.
