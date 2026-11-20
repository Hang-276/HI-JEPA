# B1-3 (P3-B): Parameter-Free Temporal Margin Audit Gate vs. Static `min R² >= 0.85`

Compares admission rules computed **strictly before accessing TEST** using independent `AUDIT` windows (`128 x 193` = 128 windows):
- `static_r2_085`: $\min_i R^2_{\mathrm{AUDIT},i} \ge 0.85$
- `temporal_audit_zero_err`: 0 selective errors on accepted `AUDIT` windows and `AUDIT` coverage $\ge 0.35$
- `temporal_margin_strict` (**parameter-free triangle-inequality gate**): `temporal_audit_zero_err` AND $Q_{0.95,\mathrm{AUDIT}}(2\|\hat{s}-s\|_\infty) < \tau_{\mathrm{cal}}$
- `combined_r2_and_temporal`: both `static_r2_085` and `temporal_margin_strict`

## Confusion Matrix Comparison (`target = TEST passes` at Bonferroni CP upper <= 1% & cov >= 35%)

| Evaluation Pool | Total (Safe / Unsafe) | Gate Rule | Admitted (TP+FP) | TP | **False-Pass (FP)** | TN | FN | FPR (`FP/Unsafe`) | FDR (`FP/Adm`) | True-Pass Recall |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Pool 1: Low-Dim Sweep (176)` | 176 (97/79) | `static_r2_085` | 168 | 97 | **71** | 8 | 0 | 89.9% | 42.3% | 100.0% |
| `Pool 1: Low-Dim Sweep (176)` | 176 (97/79) | `temporal_audit_zero_err` | 110 | 95 | **15** | 64 | 2 | 19.0% | 13.6% | 97.9% |
| `Pool 1: Low-Dim Sweep (176)` | 176 (97/79) | `temporal_margin_strict` | 14 | 14 | **0** | 79 | 83 | 0.0% | 0.0% | 14.4% |
| `Pool 1: Low-Dim Sweep (176)` | 176 (97/79) | `combined_r2_and_temporal` | 14 | 14 | **0** | 79 | 83 | 0.0% | 0.0% | 14.4% |
| `Pool 2: Visual 32x32 (33)` | 33 (18/15) | `static_r2_085` | 23 | 18 | **5** | 10 | 0 | 33.3% | 21.7% | 100.0% |
| `Pool 2: Visual 32x32 (33)` | 33 (18/15) | `temporal_audit_zero_err` | 21 | 14 | **7** | 8 | 4 | 46.7% | 33.3% | 77.8% |
| `Pool 2: Visual 32x32 (33)` | 33 (18/15) | `temporal_margin_strict` | 2 | 2 | **0** | 15 | 16 | 0.0% | 0.0% | 11.1% |
| `Pool 2: Visual 32x32 (33)` | 33 (18/15) | `combined_r2_and_temporal` | 2 | 2 | **0** | 15 | 16 | 0.0% | 0.0% | 11.1% |
| `Pool 3A: FourRoom Family A (24)` | 24 (6/18) | `static_r2_085` | 6 | 6 | **0** | 18 | 0 | 0.0% | 0.0% | 100.0% |
| `Pool 3A: FourRoom Family A (24)` | 24 (6/18) | `temporal_audit_zero_err` | 7 | 6 | **1** | 17 | 0 | 5.6% | 14.3% | 100.0% |
| `Pool 3A: FourRoom Family A (24)` | 24 (6/18) | `temporal_margin_strict` | 6 | 6 | **0** | 18 | 0 | 0.0% | 0.0% | 100.0% |
| `Pool 3A: FourRoom Family A (24)` | 24 (6/18) | `combined_r2_and_temporal` | 6 | 6 | **0** | 18 | 0 | 0.0% | 0.0% | 100.0% |
| `Pool 3B: FourRoom Family B (51)` | 51 (23/28) | `static_r2_085` | 23 | 23 | **0** | 28 | 0 | 0.0% | 0.0% | 100.0% |
| `Pool 3B: FourRoom Family B (51)` | 51 (23/28) | `temporal_audit_zero_err` | 30 | 23 | **7** | 21 | 0 | 25.0% | 23.3% | 100.0% |
| `Pool 3B: FourRoom Family B (51)` | 51 (23/28) | `temporal_margin_strict` | 18 | 18 | **0** | 28 | 5 | 0.0% | 0.0% | 78.3% |
| `Pool 3B: FourRoom Family B (51)` | 51 (23/28) | `combined_r2_and_temporal` | 18 | 18 | **0** | 28 | 5 | 0.0% | 0.0% | 78.3% |
| `Pool 3-U: FourRoom Unique own_cal (45)` | 45 (23/22) | `static_r2_085` | 23 | 23 | **0** | 22 | 0 | 0.0% | 0.0% | 100.0% |
| `Pool 3-U: FourRoom Unique own_cal (45)` | 45 (23/22) | `temporal_audit_zero_err` | 27 | 23 | **4** | 18 | 0 | 18.2% | 14.8% | 100.0% |
| `Pool 3-U: FourRoom Unique own_cal (45)` | 45 (23/22) | `temporal_margin_strict` | 18 | 18 | **0** | 22 | 5 | 0.0% | 0.0% | 78.3% |
| `Pool 3-U: FourRoom Unique own_cal (45)` | 45 (23/22) | `combined_r2_and_temporal` | 18 | 18 | **0** | 22 | 5 | 0.0% | 0.0% | 78.3% |
| `ALL UNIQUE POOLS COMBINED (254)` | 254 (138/116) | `static_r2_085` | 214 | 138 | **76** | 40 | 0 | 65.5% | 35.5% | 100.0% |
| `ALL UNIQUE POOLS COMBINED (254)` | 254 (138/116) | `temporal_audit_zero_err` | 158 | 132 | **26** | 90 | 6 | 22.4% | 16.5% | 95.7% |
| `ALL UNIQUE POOLS COMBINED (254)` | 254 (138/116) | `temporal_margin_strict` | 34 | 34 | **0** | 116 | 104 | 0.0% | 0.0% | 24.6% |
| `ALL UNIQUE POOLS COMBINED (254)` | 254 (138/116) | `combined_r2_and_temporal` | 34 | 34 | **0** | 116 | 104 | 0.0% | 0.0% | 24.6% |

## Empirical-Safety Confusion Matrix Comparison (`target = TEST cov >= 35% & emp_sel_err <= 1%`)

| Evaluation Pool | Total (EmpSafe / Unsafe) | Gate Rule | Admitted | TP | **False-Pass (FP)** | Recall |
|---|---:|---|---:|---:|---:|---:|
| `Pool 1: Low-Dim Sweep (176)` | 176 (116/60) | `static_r2_085` | 168 | 116 | **52** | 100.0% |
| `Pool 1: Low-Dim Sweep (176)` | 176 (116/60) | `temporal_audit_zero_err` | 110 | 105 | **5** | 90.5% |
| `Pool 1: Low-Dim Sweep (176)` | 176 (116/60) | `temporal_margin_strict` | 14 | 14 | **0** | 12.1% |
| `Pool 1: Low-Dim Sweep (176)` | 176 (116/60) | `combined_r2_and_temporal` | 14 | 14 | **0** | 12.1% |
| `Pool 2: Visual 32x32 (33)` | 33 (24/9) | `static_r2_085` | 23 | 23 | **0** | 95.8% |
| `Pool 2: Visual 32x32 (33)` | 33 (24/9) | `temporal_audit_zero_err` | 21 | 19 | **2** | 79.2% |
| `Pool 2: Visual 32x32 (33)` | 33 (24/9) | `temporal_margin_strict` | 2 | 2 | **0** | 8.3% |
| `Pool 2: Visual 32x32 (33)` | 33 (24/9) | `combined_r2_and_temporal` | 2 | 2 | **0** | 8.3% |
| `Pool 3A: FourRoom Family A (24)` | 24 (7/17) | `static_r2_085` | 6 | 6 | **0** | 85.7% |
| `Pool 3A: FourRoom Family A (24)` | 24 (7/17) | `temporal_audit_zero_err` | 7 | 7 | **0** | 100.0% |
| `Pool 3A: FourRoom Family A (24)` | 24 (7/17) | `temporal_margin_strict` | 6 | 6 | **0** | 85.7% |
| `Pool 3A: FourRoom Family A (24)` | 24 (7/17) | `combined_r2_and_temporal` | 6 | 6 | **0** | 85.7% |
| `Pool 3B: FourRoom Family B (51)` | 51 (29/22) | `static_r2_085` | 23 | 23 | **0** | 79.3% |
| `Pool 3B: FourRoom Family B (51)` | 51 (29/22) | `temporal_audit_zero_err` | 30 | 27 | **3** | 93.1% |
| `Pool 3B: FourRoom Family B (51)` | 51 (29/22) | `temporal_margin_strict` | 18 | 18 | **0** | 62.1% |
| `Pool 3B: FourRoom Family B (51)` | 51 (29/22) | `combined_r2_and_temporal` | 18 | 18 | **0** | 62.1% |
| `Pool 3-U: FourRoom Unique own_cal (45)` | 45 (27/18) | `static_r2_085` | 23 | 23 | **0** | 85.2% |
| `Pool 3-U: FourRoom Unique own_cal (45)` | 45 (27/18) | `temporal_audit_zero_err` | 27 | 26 | **1** | 96.3% |
| `Pool 3-U: FourRoom Unique own_cal (45)` | 45 (27/18) | `temporal_margin_strict` | 18 | 18 | **0** | 66.7% |
| `Pool 3-U: FourRoom Unique own_cal (45)` | 45 (27/18) | `combined_r2_and_temporal` | 18 | 18 | **0** | 66.7% |
| `ALL UNIQUE POOLS COMBINED (254)` | 254 (167/87) | `static_r2_085` | 214 | 162 | **52** | 97.0% |
| `ALL UNIQUE POOLS COMBINED (254)` | 254 (167/87) | `temporal_audit_zero_err` | 158 | 150 | **8** | 89.8% |
| `ALL UNIQUE POOLS COMBINED (254)` | 254 (167/87) | `temporal_margin_strict` | 34 | 34 | **0** | 20.4% |
| `ALL UNIQUE POOLS COMBINED (254)` | 254 (167/87) | `combined_r2_and_temporal` | 34 | 34 | **0** | 20.4% |

## Notable False-Pass Examples Under Static `R² >= 0.85` Caught by Temporal Margin Gate

| Pool | Unit | Audit min R² | AUDIT sel err / cov | `Q95(2||s_hat-s||_inf)` vs `tau_cal` | `TEST` sel err (CP upper) |
|---|---|---:|---:|---:|---:|
| `pool1_lowdim_sweep` | `crossed_wide_20154__separated_wide__n12` | 0.8598 | 5 / 43.8% | 0.9676 vs 0.1814 | 8.11% (10.71%) |
| `pool1_lowdim_sweep` | `crossed_wide_20153__separated_wide__n16` | 0.8983 | 1 / 46.1% | 1.0032 vs 0.1811 | 7.38% (9.86%) |
| `pool1_lowdim_sweep` | `crossed_wide_20153__separated_wide__n12` | 0.8623 | 2 / 44.5% | 0.9520 vs 0.1782 | 7.08% (9.51%) |
| `pool1_lowdim_sweep` | `crossed_wide_20154__separated_narrow__n12` | 0.8653 | 6 / 45.3% | 1.1742 vs 0.1906 | 6.63% (9.14%) |
| `pool1_lowdim_sweep` | `crossed_wide_20153__separated_narrow__n16` | 0.8886 | 7 / 46.9% | 1.1749 vs 0.1908 | 6.50% (8.97%) |
| `pool1_lowdim_sweep` | `separated_wide_20152__separated_wide__n16` | 0.9472 | 5 / 55.5% | 0.8702 vs 0.1708 | 6.55% (8.90%) |
| `pool1_lowdim_sweep` | `separated_wide_20153__separated_narrow__n12` | 0.9218 | 0 / 46.1% | 0.9644 vs 0.1725 | 6.41% (8.78%) |
| `pool1_lowdim_sweep` | `crossed_wide_20153__separated_narrow__n24` | 0.9058 | 7 / 46.1% | 1.0873 vs 0.1849 | 6.14% (8.52%) |
| `pool1_lowdim_sweep` | `crossed_wide_20154__separated_wide__n16` | 0.9015 | 2 / 41.4% | 0.7680 vs 0.1747 | 6.18% (8.47%) |
| `pool1_lowdim_sweep` | `separated_wide_20154__separated_wide__n16` | 0.9373 | 3 / 54.7% | 0.8521 vs 0.1732 | 6.10% (8.38%) |
| `pool1_lowdim_sweep` | `separated_wide_20154__separated_narrow__n12` | 0.9127 | 4 / 45.3% | 0.8660 vs 0.1863 | 5.86% (8.23%) |
| `pool1_lowdim_sweep` | `separated_wide_20151__separated_wide__n16` | 0.9400 | 3 / 61.7% | 0.7571 vs 0.1674 | 5.75% (8.01%) |
| `pool1_lowdim_sweep` | `separated_wide_20154__separated_wide__n12` | 0.9029 | 4 / 51.6% | 0.8386 vs 0.1792 | 5.61% (7.88%) |
| `pool1_lowdim_sweep` | `separated_wide_20153__separated_wide__n12` | 0.9273 | 4 / 50.8% | 0.8727 vs 0.1710 | 5.47% (7.70%) |
| `pool1_lowdim_sweep` | `crossed_wide_20153__separated_wide__n24` | 0.9135 | 2 / 43.8% | 0.9262 vs 0.1873 | 5.36% (7.60%) |
| `pool1_lowdim_sweep` | `separated_wide_20152__separated_wide__n12` | 0.9350 | 4 / 50.8% | 0.8313 vs 0.1682 | 5.29% (7.43%) |
| `pool1_lowdim_sweep` | `separated_wide_20152__separated_narrow__n16` | 0.9472 | 2 / 38.3% | 0.9365 vs 0.1876 | 5.02% (7.27%) |
| `pool1_lowdim_sweep` | `separated_wide_20152__separated_narrow__n12` | 0.9376 | 2 / 39.8% | 0.7084 vs 0.1770 | 5.04% (7.22%) |
| `pool1_lowdim_sweep` | `crossed_wide_20153__separated_narrow__n12` | 0.8525 | 6 / 46.9% | 1.0068 vs 0.1923 | 4.91% (7.19%) |
| `pool1_lowdim_sweep` | `crossed_wide_20154__separated_wide__n24` | 0.9279 | 3 / 46.1% | 0.6305 vs 0.1751 | 4.94% (7.06%) |
