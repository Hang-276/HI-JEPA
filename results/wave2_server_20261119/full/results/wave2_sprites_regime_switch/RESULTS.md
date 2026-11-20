# Wave-2 S3: episode-level regime switch

data_seed 2026111903; 80% Mode A / 20% Mode B. family_size 15, alpha 0.00333333.

| Arm | Models | Audit min mean | PASS | Mode-B coverage | Mode-B selective error | Regimes |
|---|---:|---:|---:|---:|---|---|
| symmetric_lejepa_lag1 | 5 | 0.983 | 5 | 0.459 | 0.000 | {'Normal Recovery': 5, 'Regime II-A': 0, 'Regime II-B': 0, 'Mixed': 0} |
| predictor_lag1 | 5 | 0.989 | 5 | 0.466 | 0.000 | {'Normal Recovery': 5, 'Regime II-A': 0, 'Regime II-B': 0, 'Mixed': 0} |
| predictor_lag10 | 5 | 0.678 | 3 | 0.325 | 0.053 | {'Normal Recovery': 3, 'Regime II-A': 2, 'Regime II-B': 0, 'Mixed': 0} |

Mode-B coverage > 0 confirms active switching (not the 0/0 abstention case). Constant-{0,1} baseline is in each cell's stratified.json.

