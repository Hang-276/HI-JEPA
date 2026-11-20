# Wave-2 combined summary

## S1: FourRoom Level-1 192-D projections

family_size 36, alpha 0.00138889.

| Cell | Readout | Audit min | Coverage | Errors/Accepted | Upper | Status |
|---|---|---:|---:|---:|---:|---|
| fourroom_lewm_pca192_mlp_seed42 | mlp | 0.993 | 0.494 | 0/1012 | 0.0065 | PASS |
| fourroom_lewm_pca192_ridge_seed42 | ridge | 0.921 | 0.486 | 0/995 | 0.0066 | PASS |
| fourroom_lewm_rand192_mlp_seed42 | mlp | 0.973 | 0.489 | 0/1001 | 0.0066 | PASS |
| fourroom_lewm_rand192_ridge_seed42 | ridge | 0.847 | 0.496 | 3/1016 | 0.0124 | RECOVERY_NOT_VERIFIED |
| fourroom_lewm_pca192_mlp_seed43 | mlp | 0.994 | 0.488 | 0/999 | 0.0066 | PASS |
| fourroom_lewm_pca192_ridge_seed43 | ridge | 0.940 | 0.497 | 0/1017 | 0.0064 | PASS |
| fourroom_lewm_rand192_mlp_seed43 | mlp | 0.955 | 0.500 | 0/1024 | 0.0064 | PASS |
| fourroom_lewm_rand192_ridge_seed43 | ridge | 0.850 | 0.496 | 0/1015 | 0.0065 | RECOVERY_NOT_VERIFIED |
| fourroom_lewm_pca192_mlp_seed44 | mlp | 0.994 | 0.491 | 0/1006 | 0.0065 | PASS |
| fourroom_lewm_pca192_ridge_seed44 | ridge | 0.921 | 0.486 | 0/996 | 0.0066 | PASS |
| fourroom_lewm_rand192_mlp_seed44 | mlp | 0.964 | 0.494 | 0/1011 | 0.0065 | PASS |
| fourroom_lewm_rand192_ridge_seed44 | ridge | 0.848 | 0.492 | 1/1008 | 0.0088 | RECOVERY_NOT_VERIFIED |
| fourroom_hjepa_l2_pca192_mlp_seed42 | mlp | 0.995 | 0.494 | 0/1012 | 0.0065 | PASS |
| fourroom_hjepa_l2_pca192_ridge_seed42 | ridge | 0.924 | 0.497 | 0/1018 | 0.0064 | PASS |
| fourroom_hjepa_l2_rand192_mlp_seed42 | mlp | 0.912 | 0.492 | 0/1008 | 0.0065 | PASS |
| fourroom_hjepa_l2_rand192_ridge_seed42 | ridge | 0.792 | 0.539 | 5/1104 | 0.0144 | RECOVERY_NOT_VERIFIED |
| fourroom_hjepa_l2_pca192_mlp_seed43 | mlp | 0.995 | 0.494 | 0/1011 | 0.0065 | PASS |
| fourroom_hjepa_l2_pca192_ridge_seed43 | ridge | 0.922 | 0.477 | 0/977 | 0.0067 | PASS |
| fourroom_hjepa_l2_rand192_mlp_seed43 | mlp | 0.918 | 0.494 | 0/1012 | 0.0065 | PASS |
| fourroom_hjepa_l2_rand192_ridge_seed43 | ridge | 0.739 | 0.518 | 3/1060 | 0.0119 | RECOVERY_NOT_VERIFIED |
| fourroom_hjepa_l2_pca192_mlp_seed44 | mlp | 0.993 | 0.498 | 0/1019 | 0.0064 | PASS |
| fourroom_hjepa_l2_pca192_ridge_seed44 | ridge | 0.912 | 0.487 | 0/998 | 0.0066 | PASS |
| fourroom_hjepa_l2_rand192_mlp_seed44 | mlp | 0.921 | 0.495 | 0/1014 | 0.0065 | PASS |
| fourroom_hjepa_l2_rand192_ridge_seed44 | ridge | 0.766 | 0.514 | 3/1052 | 0.0120 | RECOVERY_NOT_VERIFIED |
| fourroom_hjepa_l4_pca192_mlp_seed42 | mlp | 0.990 | 0.494 | 0/1012 | 0.0065 | PASS |
| fourroom_hjepa_l4_pca192_ridge_seed42 | ridge | 0.914 | 0.504 | 1/1033 | 0.0086 | PASS |
| fourroom_hjepa_l4_rand192_mlp_seed42 | mlp | 0.933 | 0.492 | 0/1008 | 0.0065 | PASS |
| fourroom_hjepa_l4_rand192_ridge_seed42 | ridge | 0.806 | 0.496 | 2/1015 | 0.0106 | RECOVERY_NOT_VERIFIED |
| fourroom_hjepa_l4_pca192_mlp_seed43 | mlp | 0.994 | 0.491 | 0/1006 | 0.0065 | PASS |
| fourroom_hjepa_l4_pca192_ridge_seed43 | ridge | 0.948 | 0.483 | 0/990 | 0.0066 | PASS |
| fourroom_hjepa_l4_rand192_mlp_seed43 | mlp | 0.891 | 0.505 | 1/1034 | 0.0085 | PASS |
| fourroom_hjepa_l4_rand192_ridge_seed43 | ridge | 0.775 | 0.514 | 2/1053 | 0.0102 | RECOVERY_NOT_VERIFIED |
| fourroom_hjepa_l4_pca192_mlp_seed44 | mlp | 0.998 | 0.485 | 0/993 | 0.0066 | PASS |
| fourroom_hjepa_l4_pca192_ridge_seed44 | ridge | 0.953 | 0.492 | 0/1008 | 0.0065 | PASS |
| fourroom_hjepa_l4_rand192_mlp_seed44 | mlp | 0.978 | 0.503 | 0/1030 | 0.0064 | PASS |
| fourroom_hjepa_l4_rand192_ridge_seed44 | ridge | 0.872 | 0.482 | 0/988 | 0.0066 | PASS |

## S2: two-sprite predictor factorial

family_size 50, alpha 0.001.

| Arm | Lag | w_sig | Models | Audit mean/min | Coverage | Errors/Accepted | PASS | Regimes |
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

## S3: episode-level regime switch

family_size 15, alpha 0.00333333.

| Arm | Models | Audit mean | PASS | Mode-B coverage | Regimes |
|---|---:|---:|---:|---:|---|
| symmetric_lejepa_lag1 | 5 | 0.983 | 5 | 0.459 | {'Normal Recovery': 5, 'Regime II-A': 0, 'Regime II-B': 0, 'Mixed': 0} |
| predictor_lag1 | 5 | 0.989 | 5 | 0.466 | {'Normal Recovery': 5, 'Regime II-A': 0, 'Regime II-B': 0, 'Mixed': 0} |
| predictor_lag10 | 5 | 0.678 | 3 | 0.325 | {'Normal Recovery': 3, 'Regime II-A': 2, 'Regime II-B': 0, 'Mixed': 0} |

## Integrity

Per-cell forecast receipts and every .npz payload are checked by verify_wave2.py. Binaries are excluded from git.
