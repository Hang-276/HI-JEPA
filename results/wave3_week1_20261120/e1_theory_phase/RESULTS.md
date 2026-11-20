# E1: Quantitative Theory-vs-Empirical Verification (Theorems 1 & 2)

## 1. E1-A: Exact `L0-elem` Hermite Identities & Universal Order Inversion

| Transform | Param | Theo $\mu_{\mathcal{L}}$ | Quad Diff | MC Emp $\mu_{\mathcal{L}}$ | MC Diff | Distortion Share $\eta_1$ |
|---|---:|---:|---:|---:|---:|---:|
| `sinh` | 1.0 | 1.29869 | 1.04e-14 | 1.29396 | 0.00473 | 0.1491 |
| `sinh` | 1.5 | 1.49054 | 5.30e-13 | 1.44622 | 0.04432 | 0.5204 |
| `cubic` | 1.0 | 1.35608 | 9.33e-15 | 1.35136 | 0.00472 | 0.2727 |
| `cubic` | 3.0 | 1.39292 | 8.66e-15 | 1.38641 | 0.00650 | 0.3506 |
| `tanh` | 2.0 | 1.31615 | 2.75e-14 | 1.32461 | 0.00846 | 0.1623 |
| `tanh` | 6.0 | 1.40005 | 1.48e-12 | 1.40802 | 0.00797 | 0.2833 |

| Grid (boundary pair `0.93` vs `0.93 - gap`) | Exact Theo Flip vs Emp >50% Agreement | Cells Satisfying Suff. Bound (1.1) | Max Boundary Flip Under Suff. Bound |
|---|---:|---:|---:|
| `sinh_T193` | 66.7% | 34/63 | 0.4688 |
| `sinh_T769` | 66.7% | 34/63 | 0.4453 |
| `cubic_T193` | 75.5% | 33/49 | 0.5093 |
| `tanh_T193` | 90.5% | 33/42 | 0.5493 |

## 2. E1-B: Single-Frame Indistinguishability (`L1` Impossibility)

| Process | Target $R^2$ | Arm | Emp min $R^2$ | Emp min MCC | `all_err` | `own_cal` cov | `own_cal` sel_err | PASS |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `wide_0.04` | 0.85 | `Safe_MatchedAC` | 0.8435 | 0.9213 | 18.02% | 51.2% | 3.24% | False |
| `wide_0.04` | 0.85 | `Uniform_HighAC_0.75` | 0.8431 | 0.9212 | 27.98% | 51.1% | 10.70% | False |
| `wide_0.04` | 0.85 | `Adversarial_AsymAC` | 0.8422 | 0.9205 | 29.35% | 47.4% | 11.02% | False |
| `wide_0.04` | 0.90 | `Safe_MatchedAC` | 0.8970 | 0.9482 | 15.48% | 51.2% | 0.95% | False |
| `wide_0.04` | 0.90 | `Uniform_HighAC_0.75` | 0.8970 | 0.9487 | 20.90% | 50.6% | 6.75% | False |
| `wide_0.04` | 0.90 | `Adversarial_AsymAC` | 0.8964 | 0.9482 | 19.09% | 51.1% | 4.11% | False |
| `wide_0.04` | 0.95 | `Safe_MatchedAC` | 0.9490 | 0.9744 | 10.01% | 50.7% | 0.00% | True |
| `wide_0.04` | 0.95 | `Uniform_HighAC_0.75` | 0.9492 | 0.9744 | 14.79% | 51.0% | 1.15% | False |
| `wide_0.04` | 0.95 | `Adversarial_AsymAC` | 0.9480 | 0.9741 | 11.57% | 49.5% | 0.59% | False |
| `wide_0.04` | 0.98 | `Safe_MatchedAC` | 0.9798 | 0.9899 | 6.88% | 51.4% | 0.00% | True |
| `wide_0.04` | 0.98 | `Uniform_HighAC_0.75` | 0.9798 | 0.9898 | 10.11% | 49.7% | 0.10% | True |
| `wide_0.04` | 0.98 | `Adversarial_AsymAC` | 0.9798 | 0.9898 | 7.03% | 51.4% | 0.00% | True |
| `wide_0.04` | 0.99 | `Safe_MatchedAC` | 0.9899 | 0.9950 | 5.52% | 48.3% | 0.00% | True |
| `wide_0.04` | 0.99 | `Uniform_HighAC_0.75` | 0.9899 | 0.9950 | 7.37% | 50.3% | 0.00% | True |
| `wide_0.04` | 0.99 | `Adversarial_AsymAC` | 0.9900 | 0.9950 | 5.57% | 50.2% | 0.00% | True |
| `narrow_0.002` | 0.85 | `Safe_MatchedAC` | 0.8438 | 0.9216 | 18.65% | 50.8% | 3.46% | False |
| `narrow_0.002` | 0.85 | `Uniform_HighAC_0.75` | 0.8433 | 0.9212 | 26.56% | 48.0% | 10.27% | False |
| `narrow_0.002` | 0.85 | `Adversarial_AsymAC` | 0.8439 | 0.9214 | 31.74% | 49.6% | 14.27% | False |
| `narrow_0.002` | 0.90 | `Safe_MatchedAC` | 0.8971 | 0.9482 | 13.87% | 49.8% | 1.67% | False |
| `narrow_0.002` | 0.90 | `Uniform_HighAC_0.75` | 0.8967 | 0.9478 | 21.78% | 51.5% | 5.69% | False |
| `narrow_0.002` | 0.90 | `Adversarial_AsymAC` | 0.8972 | 0.9484 | 21.44% | 47.9% | 5.50% | False |
| `narrow_0.002` | 0.95 | `Safe_MatchedAC` | 0.9492 | 0.9745 | 9.96% | 48.0% | 0.20% | True |
| `narrow_0.002` | 0.95 | `Uniform_HighAC_0.75` | 0.9493 | 0.9744 | 14.84% | 50.9% | 1.82% | False |
| `narrow_0.002` | 0.95 | `Adversarial_AsymAC` | 0.9491 | 0.9745 | 14.40% | 49.6% | 0.79% | False |
| `narrow_0.002` | 0.98 | `Safe_MatchedAC` | 0.9799 | 0.9899 | 6.25% | 51.0% | 0.00% | True |
| `narrow_0.002` | 0.98 | `Uniform_HighAC_0.75` | 0.9797 | 0.9898 | 9.42% | 50.4% | 0.10% | True |
| `narrow_0.002` | 0.98 | `Adversarial_AsymAC` | 0.9798 | 0.9898 | 7.71% | 51.0% | 0.00% | True |
| `narrow_0.002` | 0.99 | `Safe_MatchedAC` | 0.9899 | 0.9949 | 4.88% | 50.1% | 0.00% | True |
| `narrow_0.002` | 0.99 | `Uniform_HighAC_0.75` | 0.9900 | 0.9949 | 7.08% | 48.5% | 0.00% | True |
| `narrow_0.002` | 0.99 | `Adversarial_AsymAC` | 0.9899 | 0.9949 | 4.93% | 51.2% | 0.00% | True |

## 3. E1-C: Theorem 2 Phase Overlay (`R^2 x rho_r` & `Delta_rho x T`) + `K_pop` Baseline

- **Grid 1 (`separated_narrow`)**: Spearman(theo vs emp `all_err`) = **0.9976** (MAE=0.1143), Spearman($\sigma_{\mathrm{err}}/\tau$ vs emp `sel_err`) = **0.9782** (MAE=0.0369), 1% safe-boundary agreement = **91.8%**, stationary `K_pop` population errors = **3/49**.
- **Grid 1 (`separated_wide`)**: Spearman(theo vs emp `all_err`) = **0.9965** (MAE=0.1067), Spearman($\sigma_{\mathrm{err}}/\tau$ vs emp `sel_err`) = **0.9772** (MAE=0.0382), 1% safe-boundary agreement = **95.9%**, stationary `K_pop` population errors = **0/49**.
- **Grid 2 (`Delta_rho x T`)**: Spearman(theo vs emp `all_err`) = **0.4101**, Spearman($\sigma_{\mathrm{err}}/\tau$ vs emp `sel_err`) = **0.9248**, 1% safe-boundary agreement = **96.7%**.
