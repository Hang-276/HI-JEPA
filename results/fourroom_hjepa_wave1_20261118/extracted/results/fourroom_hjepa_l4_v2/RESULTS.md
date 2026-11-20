# Wave-1 FourRoom H-JEPA / LeWM extension

Experiment: `fourroom_hjepa_l4_v2`. Family size 51, per-claim alpha 0.000980392.

| Cell | Rule | Min R2 | Coverage | Errors/Accepted | Error upper | Status |
|---|---|---:|---:|---:|---:|---|
| cls_160k_official_seed42_level1 | own_cal | 0.911 | 47.7% | 0/976 | 0.71% | PASS |
| cls_160k_official_seed43_level1 | own_cal | 0.834 | 48.8% | 6/1000 | 1.80% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed44_level1 | own_cal | 0.958 | 49.8% | 0/1020 | 0.68% | PASS |
| grid4x4_160k_official_seed42_level1 | own_cal | 0.954 | 49.1% | 0/1006 | 0.69% | PASS |
| grid4x4_160k_official_seed43_level1 | own_cal | 0.967 | 49.5% | 0/1013 | 0.68% | PASS |
| grid4x4_160k_official_seed44_level1 | own_cal | 0.992 | 49.0% | 0/1004 | 0.69% | PASS |
| grid4x4_160k_ridge_seed42_level1 | own_cal | 0.992 | 49.2% | 0/1008 | 0.68% | PASS |
| grid4x4_160k_ridge_seed43_level1 | own_cal | 0.996 | 49.4% | 0/1011 | 0.68% | PASS |
| grid4x4_160k_ridge_seed44_level1 | own_cal | 0.996 | 49.0% | 0/1003 | 0.69% | PASS |
| cls_160k_official_seed42_level2 | own_cal | 0.730 | 52.3% | 10/1071 | 2.24% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed42_level2 | level1_cal | 0.730 | 50.9% | 10/1042 | 2.30% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed43_level2 | own_cal | 0.589 | 48.3% | 19/990 | 3.68% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed43_level2 | level1_cal | 0.589 | 48.3% | 19/990 | 3.68% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed44_level2 | own_cal | 0.828 | 51.7% | 4/1058 | 1.39% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed44_level2 | level1_cal | 0.828 | 49.6% | 4/1015 | 1.45% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed42_level3 | own_cal | 0.487 | 53.6% | 45/1097 | 6.30% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed42_level3 | level1_cal | 0.487 | 43.9% | 35/899 | 6.31% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed43_level3 | own_cal | 0.347 | 50.4% | 53/1032 | 7.62% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed43_level3 | level1_cal | 0.347 | 38.0% | 31/779 | 6.63% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed44_level3 | own_cal | 0.649 | 50.8% | 27/1040 | 4.50% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed44_level3 | level1_cal | 0.649 | 38.7% | 16/792 | 4.08% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed42_level4 | own_cal | 0.216 | 50.2% | 89/1029 | 11.69% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed42_level4 | level1_cal | 0.216 | 41.4% | 72/847 | 11.86% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed43_level4 | own_cal | 0.188 | 51.0% | 86/1045 | 11.18% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed43_level4 | level1_cal | 0.188 | 37.7% | 50/772 | 9.68% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed44_level4 | own_cal | 0.424 | 49.4% | 47/1012 | 7.06% | RECOVERY_NOT_VERIFIED |
| cls_160k_official_seed44_level4 | level1_cal | 0.424 | 39.8% | 32/815 | 6.50% | RECOVERY_NOT_VERIFIED |

RECOVERY_NOT_VERIFIED: this readout did not recover all four factors at AUDIT; it does not prove the representation lost the information.
CRITERIA_NOT_MET: recovery passed but this frozen threshold missed the coverage/error criteria.
PASS: recovery passed, coverage >= 35%, and the Bonferroni-corrected one-sided Clopper-Pearson error bound is <= 1%.
Coverage is an observed test fraction, not a confidence lower bound.
Level-1 cells report own_cal only: level1_cal is numerically identical there.
grid4x4_* readouts operate on the projected 4x4 patch grid and exist at Level 1 only.
The TEST truth was read only after forecast_receipt.json was sealed.
