# B1-1 (P0-ZeroRetrain): Zero-Retrain Episode-Level Regime Switching

Each split mixes **80% Mode A** (slow {0,1}) and **20% Mode B** (slow {2,3}) episodes with identical $N(0, I_4)$ single-frame marginals. Frozen encoders and static probes are reused with zero retraining; thresholds are frozen on unlabeled CAL before touching TEST.

## Panel A: 32x32 Visual Two-Sprite (`formal_v1`, 6 frozen models, `M=6`, `alpha=0.05/6`)

Rho A = `[0.964, 0.961, 0.78, 0.76]`, Rho B = `[0.78, 0.76, 0.964, 0.961]`, TEST windows = `2048`.

| Cell | Audit min R² | Overall all_err | Const-{0,1} err | Overall own_cal cov | Overall own_cal err (CP upper) | Mode-B all_err | **Mode-B own_cal cov** | **Mode-B err / acc** | Mode-B pred {2,3} | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `visual_lejepa_seed101` | 0.9833 | 4.05% | 28.86% | 50.73% | 0.10% (0.66%) | 5.48% | **50.71%** | **0/213** | 99.5% | `PASS` |
| `visual_lejepa_seed102` | 0.9815 | 4.64% | 28.86% | 50.59% | 0.19% (0.83%) | 6.19% | **50.95%** | **0/214** | 99.5% | `PASS` |
| `visual_lejepa_seed103` | 0.9818 | 4.10% | 28.86% | 51.22% | 0.10% (0.65%) | 5.00% | **51.43%** | **0/216** | 99.5% | `PASS` |
| `long_horizon_pred_seed101` | 0.1257 | 34.57% | 28.86% | 52.39% | 16.96% (19.88%) | 63.33% | **33.57%** | **76/141** | 46.8% | `RECOVERY_NOT_VERIFIED` |
| `long_horizon_pred_seed102` | -0.0045 | 75.83% | 28.86% | 51.07% | 67.78% (71.21%) | 74.29% | **61.90%** | **188/260** | 26.2% | `RECOVERY_NOT_VERIFIED` |
| `long_horizon_pred_seed103` | 0.2621 | 22.56% | 28.86% | 53.91% | 6.43% (8.41%) | 56.43% | **20.48%** | **27/86** | 66.3% | `RECOVERY_NOT_VERIFIED` |

## Panel B: Low-Dim Frozen A4 Strong vs. A2 Weak Readouts (`M=16` per process, `alpha=0.05/16`)

| Process | Arm | Audit min R² | PASS | Const-{0,1} err | Overall all_err | Overall own_cal cov | Overall own_cal err (worst CP) | **Mode-B own_cal cov** | **Mode-B err / acc** | Mode-B pred {2,3} |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clear_switch` | `strong` | 0.9957 | 8/8 | 38.23% | 2.42% | 49.55% | 0.000% (0.58%) | **50.95%** | **0/1606** | 88.1% |
| `clear_switch` | `weak` | 0.9820 | 8/8 | 38.23% | 4.94% | 49.35% | 0.025% (0.78%) | **50.22%** | **0/1583** | 88.1% |
| `narrow_switch` | `strong` | 0.9957 | 8/8 | 73.39% | 3.96% | 49.79% | 0.000% (0.57%) | **50.29%** | **0/1541** | 35.9% |
| `narrow_switch` | `weak` | 0.9820 | 5/8 | 73.39% | 8.43% | 49.28% | 0.360% (2.59%) | **49.71%** | **6/1523** | 36.1% |
