# Wave-2 S2: two-sprite predictor factorial

data_seed 2026111902, rho [0.964, 0.961, 0.78, 0.76]. family_size 50, alpha 0.001.

| Arm | Lag | w_sig | Models | Audit min (mean/min) | Coverage | Errors/Accepted | PASS | Regimes |
|---|---:|---:|---:|---|---:|---:|---:|---|
| predictor | 1 | 0.0005 | 5 | 0.532/0.448 | 0.497 | 2270/5089 | 0 | {'Regime II-A': 2, 'Mixed': 3} |
| predictor | 1 | 0.004 | 5 | -0.010/-0.016 | 0.503 | 1989/5146 | 0 | {'Regime II-B': 3, 'Regime II-A': 2} |
| predictor | 1 | 0.02 | 5 | 0.977/0.969 | 0.485 | 1/4970 | 5 | {'Normal Recovery': 5} |
| predictor | 4 | 0.0005 | 5 | 0.519/0.444 | 0.503 | 2324/5146 | 0 | {'Mixed': 4, 'Regime II-A': 1} |
| predictor | 4 | 0.004 | 5 | -0.007/-0.011 | 0.501 | 2363/5126 | 0 | {'Regime II-B': 5} |
| predictor | 4 | 0.02 | 5 | 0.935/0.925 | 0.486 | 0/4978 | 5 | {'Normal Recovery': 5} |
| predictor | 10 | 0.0005 | 5 | 0.486/0.410 | 0.493 | 2544/5046 | 0 | {'Regime II-A': 2, 'Mixed': 3} |
| predictor | 10 | 0.004 | 5 | 0.163/-0.004 | 0.508 | 908/5198 | 0 | {'Regime II-B': 1, 'Regime II-A': 4} |
| predictor | 10 | 0.02 | 5 | 0.867/0.822 | 0.489 | 5/5008 | 2 | {'Normal Recovery': 4, 'Regime II-A': 1} |
| symmetric | 1 | 0.05 | 5 | 0.980/0.979 | 0.491 | 0/5024 | 5 | {'Normal Recovery': 5} |

PASS requires AUDIT min R2 >= 0.85, coverage >= 0.35, and the Bonferroni-corrected 1% error bound.

