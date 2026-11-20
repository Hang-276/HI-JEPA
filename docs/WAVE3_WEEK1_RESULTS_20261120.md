# Wave-3 Week 1 Verification Summary (`2026-11-20`–`2026-11-21`)

> **Evidence Status.** This document records strictly verified outputs from Week 1 of Wave 3 under the unified research framing ***When Does Factor Identifiability Yield Reliable Slow Representations?*** (`L0 / L1 / L2` hierarchy). All protected repository baselines remain untouched, all Python runs used `-B` / `sys.dont_write_bytecode = True`, and `E2-pilot` is explicitly separated into verified CPU smoke engineering validation (`120` steps) vs. the formal CUDA protocol (`6000` steps) ready for GPU launch.

---

## 1. Unified Theoretical Foundation (`L0 / L1 / L2`)

Document: [`docs/THEORY_L0_L1_L2_UNIFIED_20261120.md`](THEORY_L0_L1_L2_UNIFIED_20261120.md)

1. **`L0 / L1 / L2` Identifiability-to-Reliability Hierarchy (`Definition 1`)**: separates population identifiability classes (`L0-elem` $\Pi \circ \mathcal{H}_{\text{elem}}$, `L0-orth` $O(d)$ / overcomplete $d_h > d$, `L0-lin` $\Pi\Lambda$), single-frame static readout accuracy (`L1`: $\min_i R^2_i \ge 1 - \varepsilon^2$), and temporal/reference fidelity (`L2`: controlled residual autocorrelation $\rho_{r,i}$, matched `REF` normalization, and absence of variance collapse).
2. **`Theorem 1` (Mehler Slowness Inflation & Universal `L0-elem` Order Non-Invariance)**: proves multi-lag slowness inflation $\mu_{\mathcal{L}}(h_i(Z_i)) = \sum_{n\ge 1} w_{i,n} \mu_{\mathcal{L}}(\rho_i^n) \ge \mu_{\mathcal{L}}(\rho_i)$, derives sufficient order-preservation bound $\eta_1 < \frac{\mu_{\mathcal{L}}(\rho_2)-\mu_{\mathcal{L}}(\rho_1)}{2-\mu_{\mathcal{L}}(\rho_1)}$, establishes universal order inversion for *any* spectral gap $0 < \rho_2 < \rho_1 < 1$ via $h_\gamma(z)=\sinh(\gamma z)$ ($\operatorname{Corr}=\sinh(\gamma^2\rho^\ell)/\sinh(\gamma^2)$, $w_1=\gamma^2/\sinh(\gamma^2)$), plus exact closed forms for cubic $z+\beta z^3$ ($\eta_1\to 2/5$) and $\tanh(\gamma z)$ ($\to \frac{2}{\pi}\arcsin(\rho^\ell)$). **`Corollary 1.1`** unifies `Regime II-B` quadratic leakage and overcomplete `SIGReg` harmonic crowding ($He_2(z_s)$ scaling as $\rho_s^2$).
3. **`Theorem 2` (`L1` Single-Frame Indistinguishability & Exact Finite-Window Phase Boundary)**:
   - *Part A*: models sharing identical single-frame joint laws $(Z_{t,i}, \hat Z_{t,i})$—hence identical $R^2$, linear MCC, and single-frame mutual information—can either pass reliability safely or fail globally (even inverting population order under asymmetric autocorrelation).
   - *Part B*: exact finite-window score error $d_i = -\varepsilon_i^2 z_i^\top M z_i + \varepsilon_i^2 r_i^\top M r_i + 2\varepsilon_i\sqrt{1-\varepsilon_i^2} z_i^\top M r_i$ has pairwise uncorrelated terms, with variance dominated at small/moderate $\varepsilon$ by $4\varepsilon_i^2(1-\varepsilon_i^2)\operatorname{tr}(M\Sigma_{z,i}M\Sigma_{r,i})$, peaking when residual autocorrelation $\rho_r$ overlaps the slow boundary spectrum and governed jointly by noise-to-margin ratio $\sigma_{\mathrm{err}}/\tau$.
4. **`Proposition 1`**: formalizes the closed-form `Regime II-A` ($\Delta\mu_f > 0$) vs. `Regime II-B` ($\Delta\mu_f < 0$) sign bifurcation.

---

## 2. Verified Numerical Experiments

### 2.1 `E1`: Theory-vs-Empirical Phase Verification (`Theorems 1 & 2`)

Script: [`scripts/wave3_week1_20261120/e1_theory_phase_and_l0_flip.py`](../scripts/wave3_week1_20261120/e1_theory_phase_and_l0_flip.py)  
Results: [`results/wave3_week1_20261120/e1_theory_phase/RESULTS.md`](../results/wave3_week1_20261120/e1_theory_phase/RESULTS.md) (`45.0s`)

- **`E1-A` (`L0-elem` Mehler Identities & Boundary Inversion)**:
  - 2D spectral trapezoidal quadrature matches the analytical Mehler formulas to **`9.33e-15 – 1.48e-12`** across `sinh(1.0)`, `sinh(1.5)`, `cubic(1.0)`, `cubic(3.0)`, `tanh(2.0)`, and `tanh(6.0)`.
  - On 2D (`Delta_rho x distortion`) grids (`[0.965, 0.930, 0.930-gap, 0.750]`): analytical `Theo Flip vs Emp >50%` agreement reaches **`90.5%`** on bounded `tanh_T193` and **`75.5%`** on `cubic_T193`, dropping to **`66.7%`** on heavy-tailed `sinh_T193` / `sinh_T769`. *Caveat*: for super-exponential `sinh` tails (`sinh(1.5)` MC diff = `0.04432`), normalizing single finite windows by global `REF` variance creates extreme right-skewness—mean population scores flip as predicted by Theorem 1, whereas median single-window scores stay below the threshold until an outlier spike occurs. Also max empirical boundary flip under the Theorem 1 sufficient bound `(1.1)` stayed in `0.4453 – 0.5493` across zero-margin extreme noise cells rather than exceeding 0.5 systematically.
- **`E1-B` (`L1` Single-Frame Impossibility)**:
  - At virtually identical static fidelity (**`min R² = 0.9480–0.9493`**, **`MCC = 0.9741–0.9745`**), `Safe_MatchedAC` passes **`own_cal g50`** on both wide (`0.00%` sel_err) and narrow (`0.20%` sel_err) processes, whereas `Uniform_HighAC_0.75` fails both (`1.15%` and `1.82%` sel_err) and `Adversarial_AsymAC` fails (`pass=False`, `0.59%` / `0.79%` point error breaching CP upper bound at `alpha=0.05/15`). All three arms pass only once **`R² >= 0.98`** (`0.00%–0.10%` sel_err).
- **`E1-C` (`Theorem 2` Phase Overlay & Stationary `K_pop`)**:
  - **Grid 1 (`R² x rho_r`)**: Spearman correlation between Theorem 2 analytical vs empirical `all_err` is **`0.9976`** (`separated_narrow`, `MAE=0.1143`) and **`0.9965`** (`separated_wide`, `MAE=0.1067`); Spearman(`sigma_err / tau` vs empirical `sel_err`) is **`0.9782`** and **`0.9772`** (`MAE=0.0369` / `0.0382`); `1%` safe-boundary agreement is **`91.8%`** and **`95.9%`**. Global `K_pop` commits `0/49` population errors on `separated_wide` and `3/49` on `separated_narrow` when stationary.
  - **Grid 2 (`Delta_rho x T`)**: Spearman(`sigma_err / tau` vs `sel_err`) = **`0.9248`** and `1%` safe-boundary agreement = **`96.7%`**. *Caveat*: Spearman(`ana_all`, `emp_all`) on Grid 2 is **`0.4101`** because single-pair Gaussian quadrature across small fixed `T in {65, 129}` hits a `x_true = 0` integrand sign cusp while omitting off-boundary pairs (`2 vs 3`).

### 2.2 `E3` & `E4`: High-Dim Scaling (`d=4..32`), `L0-orth`, `K_pop` Switching, Non-Gaussian & `AR(2)`

Script: [`scripts/wave3_week1_20261120/e3_e4_highdim_kpop_robustness.py`](../scripts/wave3_week1_20261120/e3_e4_highdim_kpop_robustness.py)  
Results: [`results/wave3_week1_20261120/e3_e4_scaling_robustness/RESULTS.md`](../results/wave3_week1_20261120/e3_e4_scaling_robustness/RESULTS.md) (`23.3s`, `score_windows_nd` verified `<1e-13` identical to `src/slowrepr/metrics.py` at `d=4, k=2`)

- **`E3-A1` Continuous Spectrum (`d in {4,8,16,32}, k=d/2, rho linspace 0.96..0.75, T=193`)**:
  - `d=4` (`Delta_rho=0.0700`): `g50` sel_err = `0.00%–0.79%` across `R² >= 0.95`.
  - `d=8` (`Delta_rho=0.0300`): `g50` sel_err = `1.15%` at `R²=0.995, rho_r=0.65`, tightening to `0.16%` (`cov=31.1%`) under `g30`; at `R²=0.95, rho_r=0.00`, `g30` = `0.93%` (`cov=31.4%`).
  - `d=16, 32` (`Delta_rho=0.0140, 0.0068`): exact full-set error degrades sharply (`g30 = 22.74%` even at `d=32, R²=0.995, rho_r=0.65`; `K_pop vs K_W = 99.46%–100.00%`), whereas per-axis `topk_rec` stays **`91.31%–97.58%`**.
- **`E3-A2` Block-Separated Control (`Delta_rho=0.12` fixed across `d=4..32, T=193`)**:
  - Supervised `L1+L2`: at `d=8, R²=0.98, rho_r=0.65`, `g50` passes at **`0.63%`** (`cov=46.1%`, `g30=0.00%`); yet at `d=16` (`g50=17.73%`, `g30=7.87%`) and `d=32` (`g50=43.41%`, `g30=32.98%`) exact-set error STILL rises despite constant pairwise gap `0.12`, while accepted `g50_rec` stays **`95.88%–99.74%`** (`topk_rec` all = `93.82%–97.12%`).
  - Unsupervised `L0-orth` (`SFA_lag1` & `SOBI_4lags`, `CAL=1024`): slow-subspace recovery scales cleanly to `d=32` (**`slow_subspace_lin_R2 = 0.9999`** on `exact_L0_orth`, **`0.9799`** on `R²=0.98, rr=0.65`), even when within-block mixing lowers `d=32` Hungarian `MCC` to `0.8424` (`SFA`) / `0.7755` (`SOBI`) and pushes exact-set error above `94%`.
- **`E3-A3` Window-Length Scaling (`d=16, k=8, Delta_rho=0.12, R²=0.98, rho_r=0.65`)**:
  - Explains and resolves the exact-set extreme-value erosion over $k(d-k)=64$ boundary pairs ($\approx\sigma\sqrt{2\ln(d/2)}$): scaling `T = 193 → 385 → 769` drives `all_err` **`29.79% → 17.53% → 6.69%`**, `g50 sel_err` **`8.79% → 2.10% → 0.00%`** (`g50_rec: 98.90% → 99.74% → 100.00%`), and `g30 sel_err` **`3.20% → 0.00% → 0.00%`**.
- **`E3-B` 80/20 Regime-Switching (`R²=0.98, rho_r=0.65`)**:
  - Global `K_pop` suffers **`100.0%` Mode-B error** across `d=4..32` (`20.00%` oracle error at `d=4`, rising to **`100.00%` chimera error** at `d>=8` where pooled rates reverse `0.8*0.87 + 0.2*0.75 = 0.846 < 0.864 = 0.8*0.84 + 0.2*0.96`). Adaptive `K_W + own_cal g50` holds `d=4` at **`0.00%`** (`Mode-B = 0.00%`).
- **`E4-A` Non-Gaussian Stationary AR(1) (`gaussian`, `laplace`, `student_t5`, `student_t3`)**:
  - Bounded/sublinear `L0-elem` diffeomorphisms (`tanh(4z)`, `asinh(4z)`) still induce **`36.33%–55.13%`** boundary flip rates across all four families.
  - Under `L1+L2 + own_cal g50`, calibration automatically adapts to heavier tails: even on `student_t3` (excess kurtosis `3.83`), `R²=0.95` holds `g50 sel_err` at **`0.29%`** (`rho_r=0.00`) and **`0.78%`** (`rho_r=0.75`); `R²=0.98, rho_r=0.75` holds at **`0.20%`** (all `<= 0.87%` across all 12 cells).
- **`E4-B` Damped Oscillatory `AR(2)` & Toeplitz `Theorem 2`**:
  - Plugging exact stationary `AR(2)` Toeplitz ACF into `Theorem 2`'s trace formula matches empirical `sd(d1-d2)` across all frequencies (`slow_osc_pi64`: `0.1297 vs 0.1305`; `mid_osc_pi32`: `0.1466 vs 0.1448`; `alias_osc_pi16`: `0.1530 vs 0.1573`).
  - *Honest Limitation (`Lag Aliasing`)*: when oscillation half-period resonates with `LAGS=(4,16,32)` (`pi/32`, `pi/16`), slow-envelope factors `{0,1}` no longer stay minimal in multi-lag difference variance (`env_is_K_W` drops `73.6% → 34.7% → 15.9%`).

### 2.3 `E2-pilot` (32×32 Overcomplete Sprites — CPU Smoke Check)

Script: [`scripts/wave3_week1_20261120/e2_pilot_sprites_overcomplete.py`](../scripts/wave3_week1_20261120/e2_pilot_sprites_overcomplete.py)  
Smoke Results (`120` pretrain / `150` probe steps, `seed=[101]`, `cpu`, `67.3s`): [`results/wave3_week1_20261120/e2_pilot_smoke/RESULTS.md`](../results/wave3_week1_20261120/e2_pilot_smoke/RESULTS.md)

- Pipeline check passed cleanly (`exit code 0`, zero NaNs, finite reconstruction/SIGReg losses and all CRL, `SFA/SOBI`, harmonic crowding, and Linear/MLP probe `K_W + own_cal` metrics recorded). Note that `120` steps on `16` episodes (`T=48..49`) serves strictly as an engineering verification (`MLP min R2` reached `0.7057` on `lejepa_d16`, `0/1` audit passes at `0.85`); scientific comparison of `autoencoder_d4/d8` vs `lejepa_d4/d8/d16` requires executing the full `--formal --device cuda` run (`6000` pretrain steps, `3000` probe steps, `seeds=[101,102,103]`).

---

## 3. Turnkey GPU Launch Command (Week 2 Formal `E2-pilot`)

Run on a CUDA machine in the workspace root (write-safe new output directory `results/wave3_week1_20261120/e2_pilot_formal/`, zero bytecode):

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/wave3_week1_20261120/e2_pilot_sprites_overcomplete.py --formal --device cuda
```

---

## 4. Week 1 Go / No-Go Evaluation

- **Theory & Identifiability Foundation (`Go`)**: `L0/L1/L2` hierarchy, `Theorem 1` (`L0-elem` Hermite/Mehler identities verified to `<2e-12`), and `Theorem 2` (`0.997` `all_err` Spearman, `0.978` `sigma_err/tau` Spearman, `>91%` safe-boundary accuracy) are theoretically tight and empirically validated.
- **High-Dim & Robustness Boundaries (`Go with Explicit Scope Conditions`)**: All three stress tests exposed real, publishable scope boundaries rather than hiding flaws—(i) exact `top-k` set error erodes via extreme-value margin shrinkage at `d >= 16, T=193` unless `T` grows (`E3-A3`: `T=769 -> 0.00%`) or the target relaxes to slow-subspace recovery (`subspace_R2 = 0.98–1.00`); (ii) multi-lag `LAGS=(4,16,32)` assumes non-aliased dynamics (`AR(2)` `pi/16` drops envelope agreement to `15.9%`); and (iii) super-exponential `L0-elem` tails skew global-`REF` window normalization.
- **Next Gate (Week 2)**: Run formal `--formal --device cuda` `E2-pilot` on GPU to measure `SIGReg` quadratic-harmonic crowding (`top2_quad_harm_frac` & `SFA/SOBI` subspace drop from `d_h=4` to `d_h=8, 16`) vs Autoencoder baselines and confirm empirical support for `Corollary 1.1`.
