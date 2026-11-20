# Wave-1 FourRoom H-JEPA / LeWM extension

Experiment: `fourroom_probe_v2`. Family size 24, per-claim alpha 0.00208333.

| Cell | Rule | Min R2 | Coverage | Errors/Accepted | Error upper | Status |
|---|---|---:|---:|---:|---:|---|
| cls_8k_small_seed42_level1 | own_cal | 0.511 | 51.6% | 36/1057 | 5.32% | RECOVERY_NOT_VERIFIED |
| cls_8k_small_seed43_level1 | own_cal | -0.084 | 53.8% | 107/1102 | 12.53% | RECOVERY_NOT_VERIFIED |
| cls_8k_small_seed44_level1 | own_cal | -0.065 | 49.7% | 116/1018 | 14.52% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed42_level1 | own_cal | 0.695 | 49.3% | 9/1010 | 2.11% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed43_level1 | own_cal | 0.428 | 49.1% | 23/1005 | 3.99% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed44_level1 | own_cal | 0.377 | 51.3% | 36/1050 | 5.36% | RECOVERY_NOT_VERIFIED |
| grid4x4_160k_official_seed42_level1 | own_cal | 0.978 | 50.1% | 0/1026 | 0.60% | PASS |
| grid4x4_160k_official_seed43_level1 | own_cal | 0.972 | 49.9% | 0/1022 | 0.60% | PASS |
| grid4x4_160k_official_seed44_level1 | own_cal | 0.973 | 49.7% | 0/1017 | 0.61% | PASS |
| grid4x4_160k_ridge_seed42_level1 | own_cal | 0.996 | 49.4% | 0/1012 | 0.61% | PASS |
| grid4x4_160k_ridge_seed43_level1 | own_cal | 0.995 | 49.6% | 0/1015 | 0.61% | PASS |
| grid4x4_160k_ridge_seed44_level1 | own_cal | 0.994 | 49.8% | 0/1019 | 0.60% | PASS |
| cls_8k_small_seed42_level2 | own_cal | 0.325 | 51.6% | 81/1056 | 10.30% | RECOVERY_NOT_VERIFIED |
| cls_8k_small_seed42_level2 | level1_cal | 0.325 | 47.5% | 74/972 | 10.36% | RECOVERY_NOT_VERIFIED |
| cls_8k_small_seed43_level2 | own_cal | -0.072 | 51.8% | 123/1061 | 14.67% | RECOVERY_NOT_VERIFIED |
| cls_8k_small_seed43_level2 | level1_cal | -0.072 | 11.4% | 29/234 | 19.72% | RECOVERY_NOT_VERIFIED |
| cls_8k_small_seed44_level2 | own_cal | -0.067 | 51.0% | 136/1044 | 16.27% | RECOVERY_NOT_VERIFIED |
| cls_8k_small_seed44_level2 | level1_cal | -0.067 | 50.5% | 136/1035 | 16.41% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed42_level2 | own_cal | 0.518 | 48.3% | 38/990 | 5.93% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed42_level2 | level1_cal | 0.518 | 39.9% | 30/818 | 5.96% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed43_level2 | own_cal | 0.135 | 51.5% | 38/1054 | 5.57% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed43_level2 | level1_cal | 0.135 | 47.4% | 34/971 | 5.54% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed44_level2 | own_cal | 0.060 | 51.2% | 81/1049 | 10.37% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed44_level2 | level1_cal | 0.060 | 46.1% | 72/944 | 10.42% | RECOVERY_NOT_VERIFIED |

RECOVERY_NOT_VERIFIED: this readout did not recover all four factors at AUDIT; it does not prove the representation lost the information.
CRITERIA_NOT_MET: recovery passed but this frozen threshold missed the coverage/error criteria.
PASS: recovery passed, coverage >= 35%, and the Bonferroni-corrected one-sided Clopper-Pearson error bound is <= 1%.
Coverage is an observed test fraction, not a confidence lower bound.
Level-1 cells report own_cal only: level1_cal is numerically identical there.
grid4x4_* readouts operate on the projected 4x4 patch grid and exist at Level 1 only.
The TEST truth was read only after forecast_receipt.json was sealed.
