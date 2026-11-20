# Unified Theory: $L_0 / L_1 / L_2$ Identifiability Hierarchy, Mehler/Hermite Order Non-Invariance, and Finite-Window Slowness Reliability

> **Scope & Attribution Note.** This specification consolidates the formal theory for the `L0 / L1 / L2` hierarchy under the paper's core question (*When Does Factor Identifiability Yield Reliable Slow Representations?*). Existing external mathematical tools are explicitly attributed throughout: Mehler's Hermite expansion and harmonic slowness decay $\rho^{n\ell}$ (Sprekeler et al., *JMLR* 2014; Klindt et al., *NeurIPS* 2026 Appendix F), Gaussian quadratic-form concentration (Hanson–Wright), Davis–Kahan subspace perturbation (Yu et al., *Biometrika* 2015), selective risk control and Clopper–Pearson / LTT bounds (Geifman & El-Yaniv, *NeurIPS* 2017; Angelopoulos et al., *AoAS* 2025). The original contributions formalized here are: (i) the $L_0 / L_1 / L_2$ reliability hierarchy; (ii) **Theorem 1** (order non-invariance across standard CRL $L_0$ equivalence classes and exact Hermite distortion thresholds); (iii) **Theorem 2** ($L_1$ single-frame indistinguishability impossibility and exact finite-window quadratic-form phase boundary); and (iv) **Proposition 1** (closed-form Regime II-A / II-B sign bifurcation).

---

## 1. Setup, Notation, and the $L_0 / L_1 / L_2$ Hierarchy

Consider trajectories $z = (z_t)_{t=0}^{T-1} \in \mathbb{R}^{T \times d}$ from a stationary process over $d$ physical factors with unit marginal variance ($\operatorname{Var}(Z_{t,i}) = 1$) and finite multi-lag scoring operator $\mathcal{L} \subset \{1, \dots, T-1\}$ ($|\mathcal{L}| = m$; default $T=193, \mathcal{L}=\{4,16,32\}$). Define the $T \times T$ symmetric positive semidefinite slowness matrix
$$M := \frac{1}{m}\sum_{\ell \in \mathcal{L}} \frac{1}{T-\ell} D_\ell^\top D_\ell, \qquad (D_\ell u)_t := u_{t+\ell} - u_t,$$
so the unnormalized finite-window slowness of any trajectory $u \in \mathbb{R}^T$ is $\widehat S(u) := u^\top M u$. Under exact population reference variance normalization (and $\widehat V_i$ under finite `REF` panel), the normalized window score of coordinate $i$ is $s_i(u) = \widehat S(u_i) / V_i$, with population mean $\mu_{\mathcal{L}}(u_i) = \mathbb{E}[\widehat S(u_i)] / \operatorname{Var}(u_{0,i})$. For stationary Gaussian AR(1) with parameter $\rho_i \in (0, 1)$,
$$\mu_{\mathcal{L}}(\rho_i) := \frac{2}{m}\sum_{\ell \in \mathcal{L}} (1 - \rho_i^\ell),$$
so smaller $\rho_i$ implies larger $\mu_{\mathcal{L}}(\rho_i)$ (faster variation). The window top-$k$ slow factor set is $K_W(s) = \operatorname{arg\,top}_k^{\min}(s)$ (default $k=d/2=2$, ties broken stably), with margin $\Delta(s) = s_{(k+1)} - s_{(k)}$.

### Definition 1 ($L_0 / L_1 / L_2$ Reliability Hierarchy)

Given an observation encoder $h(x_t)$ (with optional readout $f(h(x_t)) = \hat z_t$ aligned to factors):

1. **$L_0$ (Population Factor Identifiability)** holds up to equivalence class $\mathcal{G}$ if $\hat z = g(z)$ for some $g \in \mathcal{G}$ on the data support:
   - **`L0-elem`** ($\mathcal{G} = \Pi \circ \mathcal{H}_{\mathrm{elem}}$: permutation + smooth strictly monotone elementwise bijections $h_i: \mathbb{R}\to\mathbb{R}$, covering time-contrastive/causal CRL: PCL, GCL, iVAE, LEAP/TDRL);
   - **`L0-orth`** ($\mathcal{G} = O(d)$ or $GL(d)$, covering Gaussian LeJEPA + `SIGReg` when dimension-matched $d_h = d$, vs. overcomplete $d_h > d$);
   - **`L0-lin`** ($\mathcal{G} = \Pi\Lambda$: permutation + nonzero coordinate scaling, covering SlowVAE under Laplace transitions $\alpha < 2$).
2. **$L_1$ (Static Single-Frame $\varepsilon$-Recovery)** holds at tolerance $\varepsilon \in (0, 1)$ if an aligned readout satisfies
   $$\min_{i \in \{1,\dots,d\}} R^2_i \ge 1 - \varepsilon^2 \qquad \Bigl(\text{equivalently, } R^2_{\mathrm{AUDIT},i} \ge 0.85 \text{ at } \varepsilon^2 \le 0.15\Bigr).$$
3. **$L_2$ (Temporal & Reference Fidelity)** holds when (i) readout residuals $r_i = \hat z_i - z_i$ have controlled temporal autocorrelation ($\rho_{r,i}$ bounded relative to spectral gap and window horizon $T_{\mathrm{eff}}$), (ii) reference variance uses matched `REF` empirical normalization (`shared REF`), and (iii) no fast axis undergoes variance collapse paired with slow-harmonic amplification.

---

## 2. Theorem 1: Elementwise Identifiability (`L0-elem`) Does Not Preserve Slowness Ordering

Let $Z_i$ be a stationary Gaussian AR(1) factor with correlation $\rho_i \in (0,1)$, and let $h_i: \mathbb{R} \to \mathbb{R}$ be any nonconstant measurable transformation with finite variance ($\operatorname{Var}(h_i(Z_i)) \in (0, \infty)$). Standardize the centered representation $\tilde h_i(z) := (h_i(z) - \mathbb{E}[h_i(Z_i)])/\sqrt{\operatorname{Var}(h_i(Z_i))}$ into the normalized probabilists' Hermite basis $\{He_n/\sqrt{n!}\}_{n\ge 1}$:
$$\tilde h_i(z) = \sum_{n=1}^{\infty} c_{i,n} \frac{He_n(z)}{\sqrt{n!}}, \qquad w_{i,n} := c_{i,n}^2 \ge 0, \qquad \sum_{n=1}^{\infty} w_{i,n} = 1.$$
Define the **higher-order Hermite distortion energy fraction** $\eta_i := 1 - w_{i,1} = \sum_{n\ge 2} w_{i,n} \in [0, 1]$.

### Theorem 1 (Mehler Slowness Inflation, Exact Order-Preservation Threshold, and Universal `L0-elem` Inversion)

1. **(Exact Multi-Lag Mehler Slowness Formula)** For every lag $\ell \ge 1$, $\operatorname{Corr}(h_i(Z_{i,t}), h_i(Z_{i,t+\ell})) = \sum_{n=1}^\infty w_{i,n} \rho_i^{n\ell}$, hence the normalized multi-lag population slowness of $h_i(Z_i)$ is strictly non-decreasing in nonlinear distortion:
   $$\mu_{\mathcal{L}}(h_i(Z_i)) = \sum_{n=1}^{\infty} w_{i,n}\,\mu_{\mathcal{L}}(\rho_i^n) = \mu_{\mathcal{L}}(\rho_i) + \frac{2}{m}\sum_{\ell\in\mathcal{L}}\sum_{n\ge 2} w_{i,n}(\rho_i^\ell - \rho_i^{n\ell}) \;\ge\; \mu_{\mathcal{L}}(\rho_i),$$
   with strict inequality whenever $\eta_i > 0$ and $\rho_i \in (0, 1)$.

2. **(Sufficient Condition for Order Preservation)** Consider two factors with $0 < \rho_2 < \rho_1 < 1$ ($\mu_{\mathcal{L}}(\rho_1) < \mu_{\mathcal{L}}(\rho_2)$). For any measurable transformation $h_2$ of the faster axis, a sufficient condition for $\mu_{\mathcal{L}}(h_1(Z_1)) < \mu_{\mathcal{L}}(h_2(Z_2))$ is
   $$\eta_1 < \frac{\mu_{\mathcal{L}}(\rho_2) - \mu_{\mathcal{L}}(\rho_1)}{2 - \mu_{\mathcal{L}}(\rho_1)}. \tag{1.1}$$
   Conversely, when $h_2$ is linear ($\eta_2 = 0$), exact inversion $\mu_{\mathcal{L}}(h_1(Z_1)) > \mu_{\mathcal{L}}(Z_2)$ occurs if and only if
   $$\sum_{n\ge 2} w_{1,n}\bigl(\mu_{\mathcal{L}}(\rho_1^n) - \mu_{\mathcal{L}}(\rho_1)\bigr) > \mu_{\mathcal{L}}(\rho_2) - \mu_{\mathcal{L}}(\rho_1), \tag{1.2}$$
   which is guaranteed whenever $\eta_1 > \frac{\mu_{\mathcal{L}}(\rho_2)-\mu_{\mathcal{L}}(\rho_1)}{\mu_{\mathcal{L}}(\rho_1^{n_0})-\mu_{\mathcal{L}}(\rho_1)}$ and $h_1$ places all nonlinear mass on modes $n \ge n_0$ with $\rho_1^{n_0} < \rho_2$ (for any odd transformation, $n_0 = 3$).

3. **(Universal Inversion Under Exact `L0-elem`)** For **every** spectral gap $0 < \rho_2 < \rho_1 < 1$, there exists a strictly increasing $C^\infty$ diffeomorphism $h_1(z) = \sinh(\gamma z)$ ($\gamma > 0$) lying strictly in $\mathcal{H}_{\mathrm{elem}}$ that inverts the population slowness order ($\mu_{\mathcal{L}}(h_1(Z_1)) > \mu_{\mathcal{L}}(Z_2)$). Specifically:
   - **Exponential diffeomorphism** $h_\gamma(z) = \sinh(\gamma z)$ yields exact closed-form lag correlation and linear energy:
     $$\operatorname{Corr}\bigl(\sinh(\gamma Z_t), \sinh(\gamma Z_{t+\ell})\bigr) = \frac{\sinh(\gamma^2 \rho^\ell)}{\sinh(\gamma^2)}, \qquad w_1(\gamma) = \frac{\gamma^2}{\sinh(\gamma^2)} \xrightarrow{\gamma\to\infty} 0, \qquad \eta_1(\gamma) = 1 - \frac{\gamma^2}{\sinh(\gamma^2)} \xrightarrow{\gamma\to\infty} 1. \tag{1.3}$$
   - **Cubic diffeomorphism** $h_\beta(z) = z + \beta z^3$ ($\beta \ge 0$, $h'_\beta \ge 1$) gives exact rank-2 Hermite weights
     $$w_1(\beta) = \frac{(1+3\beta)^2}{1 + 6\beta + 15\beta^2}, \qquad w_3(\beta) = \eta_1(\beta) = \frac{6\beta^2}{1 + 6\beta + 15\beta^2} \xrightarrow{\beta\to\infty} \frac{2}{5}, \tag{1.4}$$
     inverting every gap satisfying $\mu_{\mathcal{L}}(\rho_2)-\mu_{\mathcal{L}}(\rho_1) < \frac{2}{5}\bigl(\mu_{\mathcal{L}}(\rho_1^3)-\mu_{\mathcal{L}}(\rho_1)\bigr)$.
   - **Saturating diffeomorphism** $\tanh(\gamma z)$ ($\gamma > 0$) converges to $\operatorname{sgn}(z)$ with Grothendieck arcsine correlation $\frac{2}{\pi}\arcsin(\rho^\ell)$ and distortion ceiling $\eta_1(\gamma) \to 1 - \frac{2}{\pi} \approx 0.3634$.

*Proof Sketch.* Part 1 applies Mehler's formula $\mathbb{E}[He_n(Z_t) He_m(Z_{t+\ell})]/(n!)=\delta_{nm}\rho_i^{n\ell}$ directly inside $\mathbb{E}[(\tilde h_i(Z_{t+\ell})-\tilde h_i(Z_t))^2] = 2(1-\operatorname{Corr}(\tilde h_i(Z_t),\tilde h_i(Z_{t+\ell})))$. Since $\rho_i \in (0,1)$ implies $\rho_i^n < \rho_i$ for $n\ge 2$, $\mu_{\mathcal{L}}(\rho_i) \le \mu_{\mathcal{L}}(\rho_i^n) < 2$. Part 2 follows immediately because $\mu_{\mathcal{L}}(h_2(Z_2)) \ge \mu_{\mathcal{L}}(\rho_2)$ and $\mu_{\mathcal{L}}(h_1(Z_1)) \le (1-\eta_1)\mu_{\mathcal{L}}(\rho_1) + 2\eta_1$, yielding (1.1). For Part 3, writing $\sinh(\gamma z) = \frac{1}{2}(e^{\gamma z} - e^{-\gamma z})$ and using the Gaussian bivariate MGF $\mathbb{E}[e^{a Z_t + b Z_{t+\ell}}] = \exp((a^2+b^2+2ab r)/2)$ with $r=\rho^\ell$ yields $\operatorname{Var}(\sinh(\gamma Z)) = e^{\gamma^2}\sinh(\gamma^2)$ and $\operatorname{Cov}(\sinh(\gamma Z_t), \sinh(\gamma Z_{t+\ell})) = e^{\gamma^2}\sinh(\gamma^2 \rho^\ell)$; dividing cancels $e^{\gamma^2}$, while Stein's identity $\mathbb{E}[Z \sinh(\gamma Z)] = \gamma e^{\gamma^2/2}$ gives $w_1(\gamma) = \gamma^2/\sinh(\gamma^2)$. As $\gamma \to \infty$, $\sinh(\gamma^2 \rho^\ell)/\sinh(\gamma^2) \sim e^{-\gamma^2(1-\rho^\ell)} \to 0$, so $\mu_{\mathcal{L}}(\sinh(\gamma Z_1)) \to 2 > \mu_{\mathcal{L}}(\rho_2)$ for any $\rho_1, \rho_2 \in (0,1)$. For $z + \beta z^3 = (1+3\beta)He_1(z) + \beta He_3(z)$, orthogonality gives variance $(1+3\beta)^2 + 6\beta^2 = 1+6\beta+15\beta^2$ and (1.4). $\square$

### Corollary 1.1 (Unification of Regime II-B & Overcomplete `SIGReg` Harmonic Crowding)
Let $z_s$ be a slow factor ($\rho_s$) and $z_f$ be a fast factor ($\rho_f < \rho_s$). Under Gaussian `SIGReg` (isotropic Gaussian marginal regularization), if either (a) a fast axis collapses in variance leaving quadratic slow leakage $He_2(z_s)$ (`Regime II-B`), or (b) latent dimension is overcomplete ($d_h > d$) so predictors match the slow subspace and `SIGReg` fills excess dimensions with uncorrelated degree-2 slow harmonics $He_2(z_s)$, then those quadratic modes carry lag correlation $\rho_s^{2\ell}$ and slowness $\mu_{\mathcal{L}}(\rho_s^2)$. Whenever $\rho_s^2 > \rho_f$, unsupervised linear SFA / SOBI / eSOBI ranks the nonlinear slow harmonic ahead of the true fast factor $z_f$, crowding linear slow-subspace recovery (observed as linear $R^2 \approx 0.43$ despite `MLP R² = 0.983` at $d_h = 8 > d = 4$).

---

## 3. Theorem 2: $L_1$ Single-Frame Indistinguishability & Exact Finite-Window Phase Boundary

Consider a stationary Gaussian AR(1) state $z_i \sim \mathcal{N}(0, \Sigma_{z,i})$ ($(\Sigma_{z,i})_{ts} = \rho_i^{|t-s|}$) and aligned representation $\hat z_i = \sqrt{1-\varepsilon_i^2}\, z_i + \varepsilon_i r_i$, where $r_i \sim \mathcal{N}(0, \Sigma_{r,i})$ is independent of $z_i$ with unit marginal variance and $(\Sigma_{r,i})_{ts} = \rho_{r,i}^{|t-s|}$. Then **for all $\rho_{r,i} \in (-1,1)$**:
1. Single-frame marginal joint law is invariant: $(Z_{t,i}, \hat Z_{t,i}) \sim \mathcal{N}\!\left(0, \begin{pmatrix} 1 & \sqrt{1-\varepsilon_i^2} \\ \sqrt{1-\varepsilon_i^2} & 1 \end{pmatrix}\right)$;
2. Exact single-frame recovery $R^2_i = 1 - \varepsilon_i^2$, linear MCC $= \sqrt{1-\varepsilon_i^2}$, and mutual information $I(Z_{t,i}; \hat Z_{t,i}) = -\tfrac{1}{2}\log(\varepsilon_i^2)$ are **strictly independent** of $(\rho_{r,1},\dots,\rho_{r,d})$.

### Theorem 2 ($L_1$ Impossibility & Exact Finite-Window Relative-Error Distribution)

1. **(Part A — Single-Frame Impossibility)** Fix any static tolerance $\varepsilon^2 = 1 - R^2 \in (0,1)$ and any gap $0 < \rho_2 < \rho_1 < 1$. There exist pairs of models $\mathcal{M}_{\mathrm{safe}}$ ($\rho_{r,1} = \rho_{r,2} = \rho_{\mathrm{match}}$ or white/matched-ordering) and $\mathcal{M}_{\mathrm{adv}}$ ($\rho_{r,1} = 0$, $\rho_{r,2} = \rho_{\mathrm{hi}} \to 1$) with **identical single-frame joint distributions $P(Z_t, \hat Z_t)$**—hence indistinguishable by $R^2$, NMSE, linear MCC, DCI, or single-frame mutual information—such that:
   - $\mathcal{M}_{\mathrm{safe}}$ preserves population slowness gap $(\mu_2 - \mu_1)(1-\varepsilon^2) > 0$; whereas
   - whenever $\varepsilon^2 > \frac{\mu_{\mathcal{L}}(\rho_2) - \mu_{\mathcal{L}}(\rho_1)}{(2 - \mu_{\mathcal{L}}(\rho_1)) + \mu_{\mathcal{L}}(\rho_2) - \mu_{\mathcal{L}}(\rho_{\mathrm{hi}})}$, $\mathcal{M}_{\mathrm{adv}}$ **inverts the population ordering** ($\mathbb{E}[\hat s_1] > \mathbb{E}[\hat s_2]$), driving finite-window error above $50\%$ as $T$ grows (even at $R^2 = 0.90$ when $\Delta\rho \approx 0.02–0.05$).
   Furthermore, even under **symmetric** residual autocorrelation ($\rho_{r,1} = \rho_{r,2} = \rho_r$), increasing $\rho_r$ from $0$ toward $\rho_{\mathrm{boundary}}$ amplifies the relative score-error variance (Part B), producing $>10\times$ selective-error inflation at identical $R^2$.

2. **(Part B — Exact Finite-Window Score-Error Moments & Phase Boundary)** Under unit population reference normalization, let $d_i := \hat z_i^\top M \hat z_i - z_i^\top M z_i$ be the window score error:
   $$d_i = -\varepsilon_i^2 z_i^\top M z_i + \varepsilon_i^2 r_i^\top M r_i + 2\varepsilon_i\sqrt{1-\varepsilon_i^2}\, z_i^\top M r_i.$$
   Its three quadratic terms are **pairwise uncorrelated**, giving exact finite-window mean and variance:
   $$\mathbb{E}[d_i] = \varepsilon_i^2\bigl(\mu_{\mathcal{L}}(\rho_{r,i}) - \mu_{\mathcal{L}}(\rho_i)\bigr), \tag{2.1}$$
   $$\operatorname{Var}(d_i) = 4\varepsilon_i^2(1-\varepsilon_i^2)\operatorname{tr}(M\Sigma_{z,i} M\Sigma_{r,i}) + 2\varepsilon_i^4\Bigl(\|M^{1/2}\Sigma_{z,i}M^{1/2}\|_F^2 + \|M^{1/2}\Sigma_{r,i}M^{1/2}\|_F^2\Bigr). \tag{2.2}$$
   Across independent boundary axes $(1,2)$, the relative score error $\delta_{12} := d_1 - d_2$ has exact variance $\sigma_{\mathrm{err}}^2 := \operatorname{Var}(d_1) + \operatorname{Var}(d_2)$ (plus $O(N_{\mathrm{ref}}^{-1})$ under shared `REF`), which is dominated at small/moderate $\varepsilon$ by the $O(\varepsilon)$ **first-order cross-term variance** $4\varepsilon^2(1-\varepsilon^2)\operatorname{tr}(M\Sigma_z M\Sigma_r)$—peaking precisely when residual autocorrelation $\rho_r$ overlaps the slowness spectrum near $\rho_{\mathrm{boundary}}$ (shrinking $T_{\mathrm{eff}}$ by $>6\times$ relative to white noise). Consequently, selective risk under fixed margin $\tau$ is governed by the dimensionless **noise-to-margin ratio** $\sigma_{\mathrm{err}} / \tau$ (and bias shift $\mathbb{E}[\delta_{12}]$), rather than $\varepsilon^2$ alone.

*Proof.* Pairwise uncorrelatedness of $z^\top M z$, $r^\top M r$, and $z^\top M r$ holds because $z \perp r$ and $\mathbb{E}[z] = \mathbb{E}[r] = 0$, so $\mathbb{E}[(z^\top M z)(z^\top M r)] = \mathbb{E}[z^\top M z\, z^\top M]\mathbb{E}[r] = 0$ and symmetrically for $r^\top M r$. For the cross-term variance, $\mathbb{E}[(z^\top M r)^2] = \mathbb{E}_r[r^\top M \mathbb{E}_z[z z^\top] M r] = \mathbb{E}_r[r^\top M \Sigma_z M r] = \operatorname{tr}(M\Sigma_z M\Sigma_r)$. The quadratic variances $\operatorname{Var}(u^\top M u) = 2\operatorname{tr}((M\Sigma_u)^2) = 2\|M^{1/2}\Sigma_u M^{1/2}\|_F^2$ follow from standard Gaussian trace identities (Lemma 1 in `THEORY_FORMALIZATION_20261116.md`). Finally, for $\mathcal{M}_{\mathrm{adv}}$ ($\rho_{r,1}=0 \implies \mu(\rho_{r,1})=2$, $\rho_{r,2}=\rho_{\mathrm{hi}}$), $\mathbb{E}[\hat s_1] - \mathbb{E}[\hat s_2] = (1-\varepsilon^2)(\mu(\rho_1)-\mu(\rho_2)) + \varepsilon^2(2 - \mu(\rho_{\mathrm{hi}}))$, which exceeds $0$ precisely under the stated $\varepsilon^2$ threshold. $\square$

---

## 4. Proposition 1: Closed-Form Regime II-A / II-B Sign Bifurcation

Write a fast axis prediction ($f$) orthogonally decomposed on `REF` as $\hat z_{t,f} = \alpha z_{t,f} + g(z_{\mathrm{slow},t}) + \sigma \xi_{t,f}$ with true-fast variance share $\pi_{\mathrm{true}} = \alpha^2/\nu_f$, slow linear/harmonic leakage share $\pi_{\mathrm{slow}} = \operatorname{Var}(g)/\nu_f$, and unexplained high-frequency residual share $\pi_{\mathrm{res}} = \sigma^2/\nu_f$ ($\pi_{\mathrm{true}} + \pi_{\mathrm{slow}} + \pi_{\mathrm{res}} = 1$, $\nu_f = \operatorname{Var}(\hat z_f)$).

### Proposition 1 (Variance-Weighted Sign Dichotomy & Harmonic Threshold)
Assuming evaluation-lag orthogonality across the three components on `REF`, the normalized population score shift $\Delta\mu_f := \mu_{\mathcal{L}}(\hat z_f) - \mu_{\mathcal{L}}(z_f)$ equals
$$\Delta\mu_f = \pi_{\mathrm{slow}}\bigl(\mu_{\mathcal{L}}(g) - \mu_{\mathcal{L}}(z_f)\bigr) + \pi_{\mathrm{res}}\bigl(\mu_{\mathcal{L}}(\xi) - \mu_{\mathcal{L}}(z_f)\bigr), \tag{3.1}$$
with exact Hermite form for $n$-th order slow harmonic leakage $He_n(z_s)$ ($\mu_{\mathcal{L}}(He_n(z_s)) = \mu_{\mathcal{L}}(\rho_s^n)$). Two opposite failure branches emerge:
- **Regime II-A ($\pi_{\mathrm{res}}(\mu_{\mathcal{L}}(\xi)-\mu_{\mathcal{L}}(z_f)) > \pi_{\mathrm{slow}}(\mu_{\mathcal{L}}(z_f)-\mu_{\mathcal{L}}(g))$, $\Delta\mu_f > 0$):** Fast scores shift upward; models appear normal on regular windows yet systematically miss local-flip windows (overconfident blind failure).
- **Regime II-B (fast-axis collapse $\nu_f \ll 1$ with $\rho_s^n > \rho_f$, amplifying $\pi_{\mathrm{slow}}(\mu_{\mathcal{L}}(z_f)-\mu_{\mathcal{L}}(g)) > \pi_{\mathrm{res}}(\mu_{\mathcal{L}}(\xi)-\mu_{\mathcal{L}}(z_f))$, $\Delta\mu_f < 0$):** Predicted fast scores shift below the slow threshold, hijacking regular windows (`34%–72%` all-answer error).

*(Additional supporting lemmas—matching-`REF` variance debiasing, temporal scaling limits, Davis–Kahan multi-lag unmixing bounds, and Clopper–Pearson / LTT binomial floors—are retained intact in [THEORY_FORMALIZATION_20261116.md](THEORY_FORMALIZATION_20261116.md) and [THEORY_AND_LITERATURE_SUPPLEMENT_20261117.md](THEORY_AND_LITERATURE_SUPPLEMENT_20261117.md) as appendix material.)*
