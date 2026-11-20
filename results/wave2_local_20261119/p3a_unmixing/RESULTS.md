# B1-2 (P3-A): Unsupervised Lag Unmixing (`SOBI` & `eSOBI` vs. Single-Lag SFA)

JAD verified (Cardoso–Souloumiac Givens rotations, `tol=1e-9`). Recomputed B2 baselines match sealed `SUMMARY.json` with max diff `0.00e+00`.

## Sub-Experiment 1: Low-Dim `B2` Exact Head-to-Head (`CAL` = 1,024 trajectories, `TEST` = 4,096 windows x 8 encoders)

| Process | Arm | TEST Accuracy | Gain vs `lag1` | Gain vs `multi` | Boundary Axes `corr²` | Off-Diag Loading RMS | Bonf. LB vs `lag1` | Bonf. LB vs `multi` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `separated_wide` | `lag1` | 82.43% | +0.00pp | +2.13pp | 0.9837 | 0.0599 | — | — |
| `separated_wide` | `lag4` | 82.02% | -0.42pp | +1.71pp | 0.9823 | 0.0705 | — | — |
| `separated_wide` | `lag16` | 79.98% | -2.46pp | -0.33pp | 0.9758 | 0.1047 | — | — |
| `separated_wide` | `lag32` | 71.29% | -11.15pp | -9.02pp | 0.9162 | 0.2524 | — | — |
| `separated_wide` | `multi` | 80.31% | -2.13pp | +0.00pp | 0.9769 | 0.0990 | — | — |
| `separated_wide` | `sobi_equal` | 80.63% | -1.81pp | +0.32pp | 0.9780 | 0.0948 | -2.11pp | +0.18pp |
| `separated_wide` | `esobi_inv` | 82.05% | -0.38pp | +1.74pp | 0.9827 | 0.0675 | -0.54pp | +1.47pp |
| `separated_wide` | `esobi_snr` | 81.80% | -0.63pp | +1.50pp | 0.9821 | 0.0723 | -0.83pp | +1.24pp |
| `separated_narrow` | `lag1` | 79.01% | +0.00pp | +12.99pp | 0.9563 | 0.1558 | — | — |
| `separated_narrow` | `lag4` | 73.04% | -5.97pp | +7.02pp | 0.9009 | 0.2844 | — | — |
| `separated_narrow` | `lag16` | 70.81% | -8.20pp | +4.79pp | 0.8632 | 0.3376 | — | — |
| `separated_narrow` | `lag32` | 59.44% | -19.57pp | -6.59pp | 0.7383 | 0.5654 | — | — |
| `separated_narrow` | `multi` | 66.02% | -12.99pp | +0.00pp | 0.7794 | 0.4797 | — | — |
| `separated_narrow` | `sobi_equal` | 74.17% | -4.84pp | +8.15pp | 0.8836 | 0.2732 | -5.48pp | +7.36pp |
| `separated_narrow` | `esobi_inv` | 73.03% | -5.98pp | +7.01pp | 0.8734 | 0.3048 | -6.52pp | +6.24pp |
| `separated_narrow` | `esobi_snr` | 73.78% | -5.23pp | +7.76pp | 0.8829 | 0.2827 | -5.80pp | +6.97pp |

## Sub-Experiment 2: `32x32` Frozen 8-D Visual Representations (`latent_dim = 8 > 4`, zero labels on `CAL`)

| Model Arm | Method | Slow Subspace `R²` (`z0,z1`) | Principal Angle `sin Θ_max` | Modes `(2,3)` Fast `R²` | Modes `(2,3)` Slow-Quad `R²` | Peak Fast-Axis Mode Index (`R²`) |
|---|---|---:|---:|---:|---:|---|
| `visual_lejepa` | `sfa_lag1` | 0.4259 | 0.0297 | 0.0004 | 0.2780 | `[6, 7, 7]` (0.9512) |
| `visual_lejepa` | `sfa_lag4` | 0.4279 | 0.0357 | 0.0007 | 0.2838 | `[6, 7, 7]` (0.9513) |
| `visual_lejepa` | `sfa_lag16` | 0.4394 | 0.0634 | 0.0042 | 0.3142 | `[6, 7, 7]` (0.9508) |
| `visual_lejepa` | `sobi_equal` | 0.4324 | 0.0390 | 0.0016 | 0.2981 | `[6, 7, 7]` (0.9519) |
| `visual_lejepa` | `esobi_inv` | 0.4337 | 0.0432 | 0.0020 | 0.3015 | `[6, 7, 7]` (0.9520) |
| `long_horizon_pred` | `sfa_lag1` | 0.9095 | 0.0219 | 0.0006 | 0.8411 | `[5, 1, 7]` (0.0025) |
| `long_horizon_pred` | `sfa_lag4` | 0.9198 | 0.0222 | 0.0005 | 0.8551 | `[5, 7, 7]` (0.0023) |
| `long_horizon_pred` | `sfa_lag16` | 0.9284 | 0.0227 | 0.0005 | 0.8681 | `[5, 4, 6]` (0.0022) |
| `long_horizon_pred` | `sobi_equal` | 0.9272 | 0.0228 | 0.0005 | 0.8670 | `[5, 4, 7]` (0.0025) |
| `long_horizon_pred` | `esobi_inv` | 0.9275 | 0.0230 | 0.0005 | 0.8673 | `[5, 4, 6]` (0.0024) |
