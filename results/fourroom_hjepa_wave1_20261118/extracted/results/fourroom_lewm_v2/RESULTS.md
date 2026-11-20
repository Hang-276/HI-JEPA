# Wave-1 FourRoom H-JEPA / LeWM extension

Experiment: `fourroom_lewm_v2`. Family size 51, per-claim alpha 0.000980392.

| Cell | Rule | Min R2 | Coverage | Errors/Accepted | Error upper | Status |
|---|---|---:|---:|---:|---:|---|
| cls_160k_official_seed42_level1 | own_cal | 0.978 | 50.1% | 0/1027 | 0.67% | PASS |
| cls_160k_official_seed43_level1 | own_cal | 0.978 | 49.4% | 0/1011 | 0.68% | PASS |
| cls_160k_official_seed44_level1 | own_cal | 0.976 | 48.8% | 0/1000 | 0.69% | PASS |
| grid4x4_160k_official_seed42_level1 | own_cal | 0.994 | 50.3% | 0/1030 | 0.67% | PASS |
| grid4x4_160k_official_seed43_level1 | own_cal | 0.988 | 49.9% | 0/1022 | 0.68% | PASS |
| grid4x4_160k_official_seed44_level1 | own_cal | 0.990 | 48.8% | 0/1000 | 0.69% | PASS |
| grid4x4_160k_ridge_seed42_level1 | own_cal | 0.991 | 48.9% | 0/1001 | 0.69% | PASS |
| grid4x4_160k_ridge_seed43_level1 | own_cal | 0.994 | 48.6% | 0/995 | 0.69% | PASS |
| grid4x4_160k_ridge_seed44_level1 | own_cal | 0.987 | 49.0% | 0/1003 | 0.69% | PASS |

RECOVERY_NOT_VERIFIED: this readout did not recover all four factors at AUDIT; it does not prove the representation lost the information.
CRITERIA_NOT_MET: recovery passed but this frozen threshold missed the coverage/error criteria.
PASS: recovery passed, coverage >= 35%, and the Bonferroni-corrected one-sided Clopper-Pearson error bound is <= 1%.
Coverage is an observed test fraction, not a confidence lower bound.
Level-1 cells report own_cal only: level1_cal is numerically identical there.
grid4x4_* readouts operate on the projected 4x4 patch grid and exist at Level 1 only.
The TEST truth was read only after forecast_receipt.json was sealed.
