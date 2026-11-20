# E2-pilot (SMOKE): Overcomplete 32x32 Sprites Disentanglement & Slow-Subspace Crowding

- Mode: `smoke`, Device: `cpu`, Seeds: `[101]`, Pretrain steps: `120`, Probe steps: `150`, Elapsed: `67.3s`

## 1. Per-Arm Mean CRL & Linear/MLP Recovery (`L1`)

| Arm | Final Loss | `Lin min R2` | `Lin MCC` | `Raw Perm MCC` | `ZCA Perm MCC` | `MLP min R2` | `Lin PASS` | `MLP PASS` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `autoencoder_d4` | 0.0125 | 0.0551 | 0.5812 | 0.5223 | 0.4754 | 0.2698 | 0/1 | 0/1 |
| `autoencoder_d8` | 0.0125 | 0.1202 | 0.5107 | 0.4535 | 0.4587 | 0.1876 | 0/1 | 0/1 |
| `lejepa_d4` | 0.5417 | 0.2197 | 0.6005 | 0.5641 | 0.5034 | 0.4769 | 0/1 | 0/1 |
| `lejepa_d8` | 0.8446 | 0.1492 | 0.6105 | 0.6066 | 0.4761 | 0.5980 | 0/1 | 0/1 |
| `lejepa_d16` | 0.9541 | 0.3645 | 0.7449 | 0.6767 | 0.4482 | 0.7057 | 0/1 | 0/1 |

## 2. Unsupervised Zero-Label Slow Subspace (`SFA_lag1` & `SOBI_4lags` on `CAL`) & Harmonic Crowding

| Arm | Method | `slow_subspace_lin_R2` | `top2_lin_slow_frac` | `top2_quad_harm_frac` | `top4_harm_slots` |
|---|---|---:|---:|---:|---:|
| `autoencoder_d4` | `SFA` | 0.3822 | 0.5555 | 0.1847 | 1.00 |
| `autoencoder_d4` | `SOBI` | 0.3827 | 0.5791 | 0.1725 | 1.00 |
| `autoencoder_d8` | `SFA` | -0.1841 | 0.3185 | 0.3894 | 3.00 |
| `autoencoder_d8` | `SOBI` | -0.1271 | 0.3555 | 0.2618 | 3.00 |
| `lejepa_d4` | `SFA` | 0.3504 | 0.4366 | 0.0959 | 2.00 |
| `lejepa_d4` | `SOBI` | 0.3267 | 0.4750 | 0.0996 | 2.00 |
| `lejepa_d8` | `SFA` | 0.4807 | 0.5881 | 0.0887 | 2.00 |
| `lejepa_d8` | `SOBI` | 0.3026 | 0.5824 | 0.1409 | 2.00 |
| `lejepa_d16` | `SFA` | 0.5250 | 0.5533 | 0.0644 | 2.00 |
| `lejepa_d16` | `SOBI` | 0.1084 | 0.4917 | 0.2065 | 3.00 |

## 3. Downstream `K_W + own_cal g50` (`Linear Probe` vs `MLP Probe` on `TEST`)

| Arm | Probe | `all_err` | `cov` | `sel_err` | `cp_upper` | `res_lag1_AC` | `harm_fast` |
|---|---|---:|---:|---:|---:|---:|---:|
| `autoencoder_d4` | `Linear` | 87.50% | 25.0% | 75.00% | 99.87% | 0.7500 | 0.0756 |
| `autoencoder_d4` | `MLP` | 56.25% | 37.5% | 33.33% | 85.64% | 0.7354 | 0.0790 |
| `autoencoder_d8` | `Linear` | 81.25% | 56.2% | 77.78% | 98.79% | 0.6860 | 0.2213 |
| `autoencoder_d8` | `MLP` | 62.50% | 50.0% | 50.00% | 90.01% | 0.6504 | 0.1461 |
| `lejepa_d4` | `Linear` | 81.25% | 62.5% | 80.00% | 98.91% | 0.7782 | 0.1398 |
| `lejepa_d4` | `MLP` | 43.75% | 68.8% | 27.27% | 69.33% | 0.6832 | 0.0640 |
| `lejepa_d8` | `Linear` | 81.25% | 43.8% | 71.43% | 98.42% | 0.7768 | 0.2249 |
| `lejepa_d8` | `MLP` | 25.00% | 43.8% | 28.57% | 79.70% | 0.6656 | 0.0498 |
| `lejepa_d16` | `Linear` | 37.50% | 50.0% | 37.50% | 83.03% | 0.6747 | 0.0714 |
| `lejepa_d16` | `MLP` | 37.50% | 50.0% | 25.00% | 74.22% | 0.6393 | 0.0258 |
