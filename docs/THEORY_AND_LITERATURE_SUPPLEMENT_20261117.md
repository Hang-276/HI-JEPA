# 理论补强与原文文献归属增补（2026-11-17）

本文档是对 [THEORY_FORMALIZATION_20261116.md](THEORY_FORMALIZATION_20261116.md) 与 [LITERATURE_GAPS_AND_POSITIONING_20261116.md](LITERATURE_GAPS_AND_POSITIONING_20261116.md) 的只读增量补强（原封存文档一字不改）。主要完成三件事：

1. **形式化两类读出／瓶颈失效机制（Proposition 4）**：将低维样本欠拟合（Type-I）、ViT 世界模型空间汇聚零空间折叠（Type-II，`[CLS]` vs `Patch Grid`）以及快轴衰减后的**“上移盲判（Regime II-A）vs 下移劫持（Regime II-B）”双分支二分定理**纳入统一数学框架；
2. **明确有标签风险校准（LTT / SGR）的二项样本复杂度下界与不可替代性边界（Proposition 5）**：严格证明为何门一（`AUDIT` 全因素恢复）无法被纯下游置信校准取代；
3. **补齐前轮标记“访问受限”或新近预印本的原文全文核读记录**（含 `SlowVAE` ICLR 2021 Theorem 1、`Predictive Closure` 2026、`MotionJEPA` 2026、`H-JEPA` 2026、`Learn Then Test` AoAS 2025 等）。

配套确定性核验脚本与收据：[verify_prop4_identities.py](../scripts/closeout_20261117/verify_prop4_identities.py) · [prop4_and_ltt_floor_checks.json](../results/closeout_20261117/l4_theory_checks/prop4_and_ltt_floor_checks.json)。

---

## 1. 两类读出／汇聚瓶颈与快轴退化的双分支二分法（Proposition 4）

沿用 [THEORY_FORMALIZATION_20261116.md](THEORY_FORMALIZATION_20261116.md) §1.1 的记号：物理因素 $z_t = (z_{\mathrm{slow},t}, z_{\mathrm{fast},t}) \in \mathbb{R}^d$ 各轴单位方差、二阶平稳，多步归一化算子 $M = \frac{1}{m}\sum_{\ell \in \mathcal{L}} \frac{1}{T-\ell} D_\ell^\top D_\ell$，总体慢度 $\mu_i = \mathbb{E}[z_i^\top M z_i]$ 满足 $\max_{s \in K_{\mathrm{pop}}} \mu_s =: b < \mu_f$（$\forall f \notin K_{\mathrm{pop}}$）。

### 1.1 两类不同的“读出未过门”机制

对于从观测 $x_t$ 到具名物理坐标预测 $\widehat{z}_t = (\psi \circ \Pi \circ \Phi)(x_t)$ 的复合映射（其中 $\Phi(x_t)$ 为冻结编码器的完整空间／局部特征，$\Pi$ 为汇聚或层级瓶颈算子，$\psi$ 为拟合于 $N_{\mathrm{probe}}$ 个带标签样本上的读出头）：

- **Type-I：统计读出欠拟合（Sample / Capacity Readout Bottleneck，对应低维 `A2 weak`）**
  汇聚后表征 $h_t = (\Pi \circ \Phi)(x_t)$ 本身保留全因素信息（总体最优读出 $\psi_\star$ 满足 $\min_j R^2(\psi_{\star,j}(h_t), z_{t,j}) \ge 1 - \varepsilon_0^2$），但小样本拟合头 $\widehat{\psi}_{N_{\mathrm{probe}}}$ 引入估计误差 $r_t = (\widehat{\psi} - \psi_\star)(h_t) + r_{\star,t}$。因为 $h_t$ 是持久物理状态 $z_t$ 的确定性光滑变换，$r_t$ **直接继承 $z_t$ 的强时间自相关**（实测 `A2 weak` 残差 `lag-1` 自相关达 `0.68–0.71`，显著慢于白噪声），从而放大有限窗口动态交叉项 $K_j = 2\langle z_j, r_j\rangle_\Delta$ 的波动方差 $\operatorname{Var}(K_j) = 4\operatorname{tr}(M \Sigma_{z_j} M \Sigma_{r_j})$。

- **Type-II：结构汇聚与预测闭包零空间瓶颈（Architectural Pooling / Hierarchy Bottleneck，对应 `LeWM / H-JEPA` 的 `[CLS]` vs `grid4x4` 及 `L2–L4`）**
  空间特征 $\Phi_{\mathrm{patch}}(x_t)$ 对全体因素 $(z_{\mathrm{slow}}, z_{\mathrm{fast}})$ 保持近似线性可逆（FourRoom Level-1 `grid4x4` 闭式线性 Ridge 实测四轴 $R^2 \in [0.994, 1.000]$），但送入动作条件世界模型预测器 $F^{(1)}(z_t^{\mathrm{CLS}}, a_t)$ 及上层编码器 $E^{(2..L)}$ 的全局汇聚算子 $z_t^{\mathrm{CLS}} = \Pi_{\mathrm{CLS}}(\Phi_{\mathrm{patch}}(x_t))$ 在预测目标驱动下，将与动作 $a_t$ 独立、含随机跳变的外生快因素 $z_{\mathrm{fast}}$ 压缩至低信噪子空间：
  \[
  \sup_{\psi \in \mathcal{F}_{\mathrm{probe}}} R^2\bigl(\psi_f(z_t^{\mathrm{CLS}}),\, z_{t,f}\bigr) \le 1 - \delta_{\mathrm{pool}} < \tau_{\mathrm{audit}}.
  \]
  因此，在同一冻结编码器上，不检查中间空间特征而仅在 $\Pi_{\mathrm{CLS}}$ 输出端评测，必然混淆“编码器丢弃物理因素”与“汇聚瓶颈过滤外生快因素”。

### 1.2 Proposition 4（快轴退化下的总体分数偏移符号二分与同窗失效模式）

设慢因素读出近似准确（$\widetilde{S}_s \approx D_s$，$\mathbb{E}[\widetilde{S}_s] \approx \mu_s \le b$），而某一快轴 $f \notin K_{\mathrm{pop}}$ 的读出可正交分解为：
\[
\widehat{z}_{t,f} = m_f + \sqrt{s}\, z_{t,f} + \lambda\, g_t + \sigma\, \xi_{t,f},
\]
其中 $z_{t,f}, g_t, \xi_{t,f}$ 零均值、单位方差、二阶平稳，且满足**评估滞后集合上的互不相关性** $\operatorname{Cov}(u_t, v_{t+\ell}) = 0$（对任意两不同分量 $u \ne v$ 及 $\ell \in \{0\} \cup (\pm\mathcal{L})$；当外生快实体 $z_{\mathrm{fast}}$ 与慢实体 $z_{\mathrm{slow}}$ 独立，且 $\xi_{t,f}$ 为独立高频噪声或高斯 AR 过程关于线性项更高阶的 Wiener–Itô Hermite 混沌分量时，该全滞后正交性自动精确成立）：$z_{t,f}$ 为真实快信号（总体慢度 $\mu_f > b$），$g_t = g(z_{\mathrm{slow},t})$ 为慢因素线性／高阶谐波泄漏项（总体慢度 $\mu_g < \mu_f$，且主一阶项满足 $\mu_g \le b$），$\xi_{t,f}$ 为快残差／高频渲染噪声（总体慢度 $\mu_\xi \in (\mu_f, 2]$，白噪声特例为 $\mu_\xi = 2$）。（注：若有限样本实证采用同刻 OLS 正交分解而 $\xi_{t,f}$ 仍含微弱滞后互协方差，则式 (P4.1) 分子另加精确滞后交叉余项 $\delta_{\mathrm{cross}}/\nu_f$；在 `L1` 视觉实测中两分支符号完全由主项主导。）记预测总体方差 $\nu_f = s + \lambda^2 + \sigma^2 > 0$，三项相对方差份额为
\[
\alpha_{\mathrm{fast}} = \frac{s}{\nu_f},\qquad
\alpha_{\mathrm{slow}} = \frac{\lambda^2}{\nu_f},\qquad
\alpha_{\mathrm{noise}} = \frac{\sigma^2}{\nu_f},\qquad
\alpha_{\mathrm{fast}} + \alpha_{\mathrm{slow}} + \alpha_{\mathrm{noise}} = 1.
\]
在无限参考样本（$\widehat{V}_f = \nu_f$）下，快轴归一化分数的总体均值精确满足：
\[
\overline{s}_f := \frac{\mathbb{E}[\widehat{S}(\widehat{z}_f)]}{\nu_f}
= \alpha_{\mathrm{fast}}\mu_f + \alpha_{\mathrm{slow}}\mu_g + \alpha_{\mathrm{noise}}\mu_\xi
= \mu_f + \underbrace{\alpha_{\mathrm{noise}}(\mu_\xi - \mu_f)}_{\ge\, 0\text{（高频抬升）}} - \underbrace{\alpha_{\mathrm{slow}}(\mu_f - \mu_g)}_{>\, 0\text{（慢泄漏拉低）}}. \tag{P4.1}
\]

由此严格导出两种截然相反的失效分支：

1. **分支 II-A：快轴衰减且高频残差占优（Upward Noise Attenuation + Local Flip Blindness）**
   当 $\alpha_{\mathrm{noise}}(\mu_\xi - \mu_f) > \alpha_{\mathrm{slow}}(\mu_f - \mu_g)$ 时，总体归一化慢度**严格上移**（$\overline{s}_f > \mu_f > b$）。
   - **同窗后果**：在常规窗口（$\max_{s\in K_{\mathrm{pop}}} D_s < \min_{f\notin K_{\mathrm{pop}}} D_f$）上，$\widetilde{S}_f$ 维持高位，预测稳定默认输出总体慢集 $K_{\mathrm{pop}}$（表面常规窗准确率可接近 $99\%$）；但在**局部速度翻转窗** $\mathcal{E}_{\mathrm{flip}} = \{D_f < \max_{s\in K_{\mathrm{pop}}} D_s\}$ 上，由于真实快轴波动衰减（$s \ll 1$），当 $s \to 0, \lambda \to 0$ 且高频噪声窗口集中于 $\mu_\xi$ 附近时，$\widetilde{S}_f \stackrel{p}{\to} \mu_\xi > \max_s D_s$，模型在翻转窗上发生系统性漏检（盲判默认选 $K_{\mathrm{pop}}$）。
   - **真实模型对应**：`visual_sprites` 的 `long_horizon_pred_seed101`（$\alpha_{\mathrm{slow}}=6.2\%$, 分数偏移 $+0.085$，接受翻转窗错率 `83.3%`）、`seed103`（$\alpha_{\mathrm{slow}}=2.1\%$, 分数偏移 $+0.101$，接受翻转窗错率 `76.9%`），以及官方 `FourRoom fourroom_01`（接受翻转窗 `481/482` 错）。

2. **分支 II-B：快轴坍缩且相对慢泄漏占优（Downward Slow-Harmonic Hijacking）**
   当快信号与高频地板同时坍缩（$s + \sigma^2 \to 0$），即使慢因素绝对泄漏能量 $\lambda^2$ 极小，分母 $\nu_f = s + \lambda^2 + \sigma^2$ 的同步缩小也会将相对份额 $\alpha_{\mathrm{slow}} = \lambda^2/\nu_f$ 放大至主导地位。只要 $\alpha_{\mathrm{slow}}(\mu_f - \mu_g) > \alpha_{\mathrm{noise}}(\mu_\xi - \mu_f)$，总体分数**严格下移**（$\overline{s}_f < \mu_f$）；且当
   \[
   \alpha_{\mathrm{slow}}(b - \mu_g) > \alpha_{\mathrm{fast}}(\mu_f - b) + \alpha_{\mathrm{noise}}(\mu_\xi - b) \tag{P4.2}
   \]
   时发生总体慢集替换（$\overline{s}_f < b$）。即便尚未跨过总体替换界（$\overline{s}_f > b$），只要差值 $\overline{s}_f - b$ 被压缩至有限窗口波动量级 $\sqrt{\operatorname{Var}(\widetilde{S}_f) + \operatorname{Var}(D_s)}$ 以内，该伪快轴就会在**常规窗口**上大规模侵入 $\widehat{K}_W$，且因无标签 `own_cal` 只取退化后自身的分差中位数，约 $50\%$ 的高置信接受窗口仍被劫持。
   - **真实模型对应**：`visual_sprites` 的 `long_horizon_pred_seed102`（预测 `REF` 方差坍缩至物理方差的 `0.32% / 0.77%`；绝对慢泄漏方差仅 `0.0015 / 0.0050`，但相对慢泄漏份额 $\alpha_{\mathrm{slow}}$ 飙升至 **`46.8% / 65.5%`**，总体分数大幅下移 `-0.422`，导致常规窗全答错率 `71.9%`、`own_cal` 选择性错率 `61.2%`）。

**证明说明**：将 $\widehat{z}_{t,f} - m_f$ 的三项正交分量代入线性差分算子 $D_\ell$，利用三者在全体时间步上的二阶正交性（或条件期望正交性：对高斯平稳独立实体，$z_{\mathrm{fast}}$ 与 $g(z_{\mathrm{slow}})$ 独立，零均值高频独立残差 $\xi$ 与前两者正交），取期望得 $\mathbb{E}[\widehat{S}(\widehat{z}_f)] = s\mu_f + \lambda^2\mu_g + \sigma^2\mu_\xi$，除以 $\nu_f = s+\lambda^2+\sigma^2$ 即得 (P4.1)；令 $\overline{s}_f < b$ 并代入 $\alpha_{\mathrm{fast}}+\alpha_{\mathrm{slow}}+\alpha_{\mathrm{noise}}=1$ 整理即得 (P4.2)。$\square$

---

## 2. 有标签二项风险校准（LTT / SGR）的样本复杂度下界与门一不可替代性（Proposition 5）

审稿中常见的一个替代设想是：“既然有现成的选择性分类／Learn-Then-Test（Geifman & El-Yaniv, 2017; Angelopoulos et al., 2025），为何不省去 `AUDIT` 恢复门，直接在有标签校准轨迹上对预测分差 $\widehat{\Delta}$ 做精确二项上界筛选？”下面给出精确数学界与实证对照。

### 2.1 Proposition 5（精确二项风险认证的最小有标签接受窗口数与盲判不可救性）

设目标选择性错选率上界为 $r_{\max} \in (0,1)$，校准显著性水平为 $\alpha_{\mathrm{cal}} \in (0,1)$（单模型为 $\alpha$，对大小 $M$ 的检验族做 Bonferroni 同时认证时为 $\alpha/M$）。

1. **零错最小样本量地板**：在有标签校准集上，即使候选阈值 $\tau$ 接受的 $n_{\mathrm{acc}}$ 条独立窗口实现**零错选**（$k=0$），其单侧精确 Clopper–Pearson 上界 $U_{\mathrm{CP}}(0, n_{\mathrm{acc}}; \alpha_{\mathrm{cal}}) = 1 - \alpha_{\mathrm{cal}}^{1/n_{\mathrm{acc}}}$ 满足 $\le r_{\max}$ **当且仅当**
   \[
   n_{\mathrm{acc}} \ge n_{\min}(\alpha_{\mathrm{cal}}, r_{\max}) := \left\lceil \frac{\log \alpha_{\mathrm{cal}}}{\log(1 - r_{\max})} \right\rceil. \tag{P5.1}
   \]
   当 $r_{\max} = 0.01$ 时：
   - 单次检验 $\alpha_{\mathrm{cal}} = 0.05 \implies n_{\min} = \mathbf{299}$（若目标覆盖率为 $50\%$，至少需 $\approx 600$ 条完整带标签校准轨迹，即 $600 \times 193 = 115{,}800$ 帧带标签物理状态）；
   - 多模型族同时控制 $M=6$（Sprites）$\implies n_{\min} = \mathbf{477}$；$M=12$（FourRoom）$\implies n_{\min} = \mathbf{546}$；$M=128$（低维迁移）$\implies n_{\min} = \mathbf{781}$。
   对任意总校准轨迹数 $N_{\mathrm{cal}} < n_{\min}$ 的实验预算，精确 LTT / SGR 的认证覆盖率恒等于 **$0\%$**。

2. **门一失效时的拒答坍缩**：若快轴退化落入 Proposition 4 的分支 II-A 或 II-B，使得在最高预测分差的 $n_{\min}$ 条窗口中局部翻转窗（或伪快轴劫持窗）占比仍超过 $r_{\max}$（例如当真实翻转事件与仅由慢轴决定的预测分差近似独立、且翻转率 $p_{\mathrm{flip}} \gg r_{\max}$ 时），则固定序列 LTT 在起始锚点阈值处即无法拒绝原假设，输出空认证集（覆盖率 $0\%$）。

**实证对齐（见 [l5_selective_baseline_comparison.json](../results/closeout_20261117/l5_selective_baseline_comparison/l5_selective_baseline_comparison.json)）**：
- 在 `visual_sprites` 的三个未过 `AUDIT` 模型（`long_horizon_pred` seeds 101–103）以及 `FourRoom fourroom_01` 的全部 6 个未过 `AUDIT` 模型层（12 折双折交叉）上，即便向 LTT 提供 **`1,024` 条完整有标签校准轨迹**，LTT 的认证覆盖率在 **全部 `3/3` 与 `12/12` 个单元上均为 `0.0%`**（首锚点直接拒绝）；
- 而在过 `AUDIT` 门的正面模型上，`N_cal = 128, 256` 条带标签轨迹下 LTT 覆盖率同样因 (P5.1) 锁死在 `0.0%`，而本文 **`AUDIT`（仅需少量单帧状态审计标签）+ 无标签 `own_cal`（0 条带标签校准轨迹）** 在低维 `baseline / coupled / faster` 上实现 `16/16` 通过、在 `visual_lejepa` 上实现 `3/3` 通过。

---

## 3. 原文全文核读增补表（补齐前轮未尽条目与最新 JEPA 理论）

本轮对下列关键文献均下载并核读了完整原文／LaTeX 源码，补齐 [LITERATURE_GAPS_AND_POSITIONING_20261116.md](LITERATURE_GAPS_AND_POSITIONING_20261116.md) §5 的待核项：

| 论文 | 发表状态与核读范围 | 原文核心定理／机制 | 与本研究的精确边界 |
|---|---|---|---|
| **Klindt, Schott, Sharma, Ustyuzhaninov, Brendel, Bethge, Paiton (2021)**, *Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding (SlowVAE)*, **ICLR 2021** | 已正式发表；本轮核读完整 LaTeX 正文 §3 与 Appendix 证明 | **Theorem 1**：当潜变量先验 $p(z_{t-1})=\mathcal{N}(0,I)$，且时间转移噪声服从形状参数 **$\alpha < 2$ 的广义 Laplace 稀疏分布** $p(z_t\mid z_{t-1}) \propto \prod_i \exp(-\lambda |z_{t,i}-z_{t-1,i}|^\alpha)$、无模型失配且生成映射单射可微时，单凭相邻观测对分布即可将潜变量辨识至**排列与符号翻转**。当 $\alpha = 2$（高斯转移）时旋转对称未破缺，该定理明确失效。 | 正面补齐了前轮 §5 的核查提醒：不能笼统写“所有既有时序非线性 ICA 都只辨识到非线性变换或旋转”。准确表述应为：**在高斯平稳 AR/OU 转移（$\alpha=2$）下，边缘加单步高斯先验留有 $O(d)$ 正交自由度（需靠异质自相关谱与滞后解混破缺）；且即便坐标已辨识到排列与符号，有限窗口同窗慢集排序与快轴衰减盲判仍不在 SlowVAE 的研究范围内。** |
| **Levy 等 (2026)**, *Predictive Closure: Which Features Does JEPA Learn?*, arXiv:2610.27364 | 预印本；核读全文 `formalization.tex` §2–4 及附录 | **Eq. (1)–(2)**：固定表征熵容量下，理想 JEPA 选择使预测互信息 $I(Y_t(S); Y_{t+1}(S))$ 最大（等价于最小化偶然不确定性 $H(Y_{t+1}(S)\mid X_t)$ 加预测泄漏 $I(X_t; Y_{t+1}(S)\mid Y_t(S))$）的特征集 $S^\star$。**§4.2（Side Information Steering）**：向预测器输入动作 $U_t=A_t$ 会降低受控因素的条件偶然不确定性，从而强化受控特征并排挤不可控外生特征。 | 直接从理论上解释了我们 Stage-1 在官方 `H-JEPA / LeWM` 上观察到的现象：动作条件预测器在 `[CLS]` 容量瓶颈上优先保留受控慢实体 `agent` 并压缩外生 `distractor`，而未受 `[CLS]` 瓶颈直接裁剪的 ViT 空间 `patch token` 仍线性保留全因素。 |
| **Karmann 等 (2026)**, *MotionJEPA: Preventing Temporal Feature Suppression in Joint-Embedding Predictive Architectures*, arXiv:2609.23881 | 预印本；核读全文 §1–6 | 实证展示标准视频 JEPA 倾向编码慢变化背景／计分板而抑制快运动球体（temporal feature suppression），提出运动引导损失保留快动态。 | 与我们形成互补：他们从训练端改进快运动保留；我们从**诊断与认证理论**刻画快因素保留／衰减何时允许或破坏有限窗口物理慢集决策（含局部速度翻转窗与分母劫持二分法）。 |
| **Zhang 等 (2026)**, *H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning*, arXiv:2610.06805v2 | 预印本；核读完整 `main_release.tex` §3–4 及 Appendix C–F | 端到端多层动作条件 JEPA + `SIGReg`（层间无 `stop-gradient`）；Appendix C.4 给出官方逐实体单隐层 `512` 宽 `LayerNorm+GELU` 探针协议；正文 Fig. 3 汇报 4 层模型跨层 `NMSE` 递增。 | 原文未区分 Level-1 `[CLS]` 汇聚损失与空间 `patch token` 保留度，也未测试有限窗口动态慢集选择或局部速度翻转窗。我们在不改动其官方冻结权重的前提下揭示了 `CLS` vs `grid4x4` 的读出可辨识性分水岭。 |
| **Maes, Le Lidec, Scieur, LeCun, Balestriero (2026)**, *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*, arXiv:2603.19312 | 预印本；核读全文 | 单层像素端到端 `ViT-Tiny ([CLS]) + Predictor + SIGReg` 世界模型（即 H-JEPA 的 `L=1` 基线）。 | 作为我们层级深度谱系（`L=1` vs `L=2` vs `L=4`）的单层对照基线，严格归属原作者。 |
