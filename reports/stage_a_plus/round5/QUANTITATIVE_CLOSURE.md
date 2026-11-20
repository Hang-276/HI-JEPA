# Approximate Gaussian quadratic closure → near axis-affine

2026-11-04。范围见 [scope](QUANTITATIVE_CLOSURE_SCOPE.md)。本文件仅给条件数学结论与固定多项式公式核验；未读取科学数组、checkpoint，未执行模型或生成样本。前轮 [approximate posterior bridge](../round4/APPROXIMATE_POSTERIOR_BRIDGE.md) 只给 sufficient-statistic span 误差，本轮补其后一个环节；**没有补成当前 actual Fine 的端到端无标签证书**。

**主结果：** 标准 Gaussian anchor、逐行中心/方差规范及一个定量非塌缩门下，L² quadratic-statistic span 误差 δ 可推出 L² axis-affine 误差 O(δ)，带显式有限常数。只用平方/四次系数会得到较松的 O(√δ)；本证明再提取三次系数，将二次项压到 O(δ)。所有系数提取均由有界的 Gaussian tilt × Hermite witness 完成，不调用未经定量的 polynomial norm equivalence。

## 1. Anchor、规范与精确假设

令 `Q` 为固定总体参考 law，

\[
V\sim N(0,I_d),\qquad Y=H(V),\qquad E_QY_j=0,\quad E_QY_j^2=1.
\tag{A1}
\]

H 可为任意可测函数；本定理甚至不需要 H 可逆。定义 additive quadratic 空间

\[
\mathcal A_2=\operatorname{span}\{1,V_i,V_i^2:1\le i\le d\}.
\]

对每个 j，假设存在 `p̃_j,q_j∈A₂`，使

\[
\|Y_j-\widetilde p_j\|_2\le\delta_{1j},\qquad
\|Y_j^2-q_j\|_2\le\delta_{2j}.
\tag{A2}
\]

全部 norm 均用同一个完整 Gaussian Q。总向量假设

\[
\|[Y,Y^2]-B[V,V^2]-b\|_{L^2(Q;\mathbb R^{2d})}\le\delta
\tag{A3}
\]

特别推出 A2 中 `δ₁j,δ₂j≤δ`。不要求 B 可逆，也不将其 condition number 当成已知。

平方统计量的 L² 假设已蕴含 `Y_j∈L⁴(Q)`，因为 q_j 为 Gaussian 多项式，属于 L²；不必再添加独立、任意有利的四阶矩上界。证明中将平方误差降到 L¹只需 Y_j、p_j 属于 L²。

真实 baseline conditional Gaussian 可以逐轴标准化为 A1；这是证明用的坐标规范，**不是算法获得了真实 mean/D 或实际已知道该 anchor**。全域 posterior 概率正下界在 rich Gaussian 情形不可与 rank 共存，前轮 finite-domain span 界仍需尾矩补齐才能提供这里的全域 A2。

## 2. 投影先消去不必要的常数损失

令 `p_j=Proj_A₂ Y_j`，即 Q 下的正交投影。由 Gaussian Hermite 正交性及 `EY_j=0`，唯一写成

\[
p_j(V)=\ell_j^\top V+\sum_i a_{ji}(V_i^2-1).
\tag{A4}
\]

记 `e_j=Y_j−p_j`、`α_j=||e_j||₂≤δ₁j`、`β_j=||Y_j²−q_j||₂≤δ₂j`。投影不增加原候选的近似误差，且

\[
e_j\perp\mathcal A_2,\quad
\|p_j\|_2^2=\|\ell_j\|^2+2\|a_j\|^2=1-\alpha_j^2\le1.
\tag{A5}
\]

因此 `P_j=p_j²−q_j` 是至多四次多项式，满足

\[
\begin{aligned}
\|P_j\|_1
&\le\|Y_j^2-q_j\|_1+\|(p_j-Y_j)(p_j+Y_j)\|_1\\
&\le\beta_j+\alpha_j(\|p_j\|_2+\|Y_j\|_2)
\le\underbrace{\beta_j+2\alpha_j}_{\varepsilon_j}.
\end{aligned}
\tag{A6}
\]

可只使用给定预算 `ε_j=δ₂j+2δ₁j`；以下所有式子对更大的有效预算仍成立。p_j 是证明中的最佳投影，不是新增可训练模型，也没有要求部署时计算这个物理坐标投影。

## 3. 有界 dual witnesses：四次、二次和三次系数

本节对任意

\[
p=c+\sum_i\ell_iV_i+\sum_i a_iV_i^2,\quad q\in\mathcal A_2,
\quad P=p^2-q,\quad\|P\|_1\le\varepsilon
\]

成立。使用 probabilists' Hermite 多项式

\[
He_1(t)=t,\ He_2(t)=t^2-1,\ He_3(t)=t^3-3t,
\ He_4(t)=t^4-6t^2+3.
\]

### 3.1 只 tilt 被测坐标

对集合 I 中的坐标，将标准 Gaussian 方差改成 `s=1/2`，其余坐标不变，得到 Q_I。likelihood ratio 为

\[
r_I(v)=\frac{dQ_I}{dQ}(v)
=2^{|I|/2}\exp\{-\tfrac12\sum_{i\in I}v_i^2\}.
\tag{A7}
\]

仅依赖被 tilt 坐标 I 的有限 Hermite 多项式乘 r_I 后均有界；本节全部 witness 都满足这一限制，故

\[
|E_{Q_I}[P\Psi]|\le\|r_I\Psi\|_\infty\,\varepsilon.
\tag{A8}
\]

这一步有效覆盖 Gaussian 全部尾部；没有用有限点值推断多项式系数。

### 3.2 四次项先给一个 bootstrap

仅 tilt i，正交性给

\[
E_{Q_{\{i\}}}[P\,He_4(\sqrt2V_i)]=6a_i^2.
\tag{A9}
\]

理由是 p² 的 `V_i⁴` 系数恰为 a_i²，其他项在积分掉未 tilt 坐标后关于 V_i 的次数至多三。令

\[
C_4^*=\frac{\sqrt2}{6}\max\left\{
3,
8(\sqrt{22}-4)e^{-(7-\sqrt{22})/4},
8(4+\sqrt{22})e^{-(7+\sqrt{22})/4}
\right\}<1.
\tag{A10}
\]

则

\[
a_i^2\le C_4^*\varepsilon\le\varepsilon.
\tag{A11}
\]

A10 来自 `sup_{t≥0}|4t²−12t+3|e^(−t/2)` 的显式极值：t=0 及 `(7±√22)/2`。数值约 `.8815`，但主定理取上界 `C₄=1` 使常数简洁；不是整体最优常数声称。

### 3.3 交叉一次系数

对 i≠k，仅 tilt i,k。奇偶性与 Gaussian 正交性给

\[
E_{Q_{\{i,k\}}}[P\,He_1(\sqrt2V_i)He_1(\sqrt2V_k)]
=\ell_i\ell_k.
\tag{A12}
\]

p² 中只有 `2ℓ_iℓ_k V_iV_k` 留下贡献；cubic、quartic 项分别因奇偶性消去，q 的 additive 项也消去。利用 `sup_t |t|e^(−t²/4)=√(2/e)`，

\[
|\ell_i\ell_k|\le K_{\rm cross}\varepsilon,
\qquad K_{\rm cross}=4/e.
\tag{A13}
\]

### 3.4 三次项使二次系数也成为一阶小量

同一坐标的 cubic witness 给

\[
E_{Q_{\{i\}}}[P\,He_3(\sqrt2V_i)]
=3\sqrt2\,\ell_i a_i.
\tag{A14}
\]

不同坐标则有

\[
E_{Q_{\{i,k\}}}[P\,He_1(\sqrt2V_i)He_2(\sqrt2V_k)]
=\sqrt2\,\ell_i a_k,\quad i\ne k.
\tag{A15}
\]

中心化 A4 带来的常数只产生低次项，不污染 A14–A15。quartic 项为 even，不能污染这些 odd-degree 提取；其他 cubic 项由 Hermite 正交性消去。A8 给出的两个常数分别不超过

\[
\frac{(6/e)^{3/2}+3\sqrt{2/e}}3<2,
\qquad
\frac{2}{\sqrt e}\max\{1,4e^{-5/4}\}<2.
\]

因而可统一取

\[
|\ell_i a_k|\le K_3\varepsilon\quad\forall i,k,
\qquad K_3=2.
\tag{A16}
\]

它是得到 O(δ) 而不只 O(√δ) 的关键一步。以上所有 witness 恒等式和常数由另一代理独立核过。

## 4. 显式每行有限误差定理（Proved）

对每个 j，选择有效 `α_j≥||Y_j−Proj_A₂Y_j||₂`、`β_j≥||Y_j²−q_j||₂`，定义

\[
\varepsilon_j=\beta_j+2\alpha_j,
\quad L_j=1-\alpha_j^2-2d\varepsilon_j,
\quad C_d=8d^2+\frac{16d(d-1)}{e^2}.
\tag{A17}
\]

假设 `L_j>0`。令 `i_j∈argmax_i |ℓ_ji|`，`m_j=ℓ_{j i_j}`。则 m_j≠0，且

\[
\boxed{
\|Y_j-m_jV_{i_j}\|_2^2
=1-m_j^2
\le E_j:=\alpha_j^2+\frac{C_d\varepsilon_j^2}{L_j}.}
\tag{A18}
\]

可以再把有效上界与 1 取最小，但 signed-axis 的非平凡结论使用 `E_j<1`。

**证明。** A5、A11 给

\[
\|\ell_j\|^2\ge1-\alpha_j^2-2d\varepsilon_j=L_j,
\quad m_j^2\ge L_j/d.
\]

A16 将每个 quadratic 系数除以 dominant linear coefficient：

\[
\|a_j\|^2\le\frac{d K_3^2\varepsilon_j^2}{m_j^2}
\le\frac{d^2K_3^2\varepsilon_j^2}{L_j}.
\]

A13 同理控制 off-axis linear energy：

\[
\sum_{i\ne i_j}\ell_{ji}^2
\le\frac{(d-1)K_{\rm cross}^2\varepsilon_j^2}{m_j^2}
\le\frac{d(d-1)K_{\rm cross}^2\varepsilon_j^2}{L_j}.
\]

投影残差、Gaussian linear chaos 与 centered additive quadratic chaos 彼此正交。因此平方误差恰好相加：

\[
\|Y_j-m_jV_{i_j}\|_2^2
=\|Y_j-p_j\|_2^2+2\|a_j\|^2
+\sum_{i\ne i_j}\ell_{ji}^2.
\]

代入 `K₃=2`、`Kcross=4/e` 即得 A18。又由 `E(Y_jV_i)=ℓ_ji` 与 unit variance，左侧等于 `1−m_j²`。证毕。

### 同规范的 signed unit-axis 界

令 `s_j=sign(m_j)`。这时不是粗的三角不等式，而有精确关系

\[
\boxed{
\|Y_j-s_jV_{i_j}\|_2^2
=2(1-|m_j|)
\le 2(1-\sqrt{1-E_j})\le2E_j,
\quad E_j<1.}
\tag{A19}
\]

它将 normalized-output 误差直接写成 signed-axis 误差。对于未归一化输出 Z_j，结论还原为 `Z_j≈b_j+a_jS_i`，其尺度与中心须按照第7节处理。

## 5. 排除多个输出行占同一源轴（Proved）

只用函数可逆性不能定量排除近塌缩。增加总体、同 anchor 的门

\[
\operatorname{Cov}_Q(Y)\succeq\lambda I_d,
\quad\lambda>0,
\qquad
\max_{j\ne k}(E_j+E_k)<\lambda.
\tag{A20}
\]

则所有 i_j 互异，故 j↦i_j 是置换，A18–A19 同时成立为 near axis-affine / near signed-permutation 保证。

**证明。** 若 i_j=i_k=i，写 `Y_j=m_jV_i+r_j`、`Y_k=m_kV_i+r_k`。取单位二维系数 `c=(m_k,−m_j)/√(m_j²+m_k²)`。共同 core 精确抵消；由 Cauchy–Schwarz，

\[
E(c_jY_j+c_kY_k)^2
\le( |c_j|\|r_j\|_2+|c_k|\|r_k\|_2)^2
\le E_j+E_k<\lambda,
\]

与 covariance 门矛盾。证毕。统一预算 E 下只需 `2E<λ`，无需更松的 `dE<λ`。

`Cov_Q(Y)` 原则上是给定可观测 anchor law 后的输出 covariance，不依赖物理坐标匹配；但当前没有记录这一固定合法 conditional anchor 的总体 covariance 下界或有效有限样本界。经验正定不自动给出 A20。

### 为什么 smooth invertible alone 不足够

在 d=4 取

\[
Y_1=V_1,\quad Y_2=(V_1+tV_2)/\sqrt{1+t^2},
\quad Y_3=V_3,\quad Y_4=V_4,\quad t>0.
\tag{A21}
\]

它对每个 t 都是光滑可逆 linear map，每行 unit variance。一阶 span 误差为0；平方统计量总 span 误差仅 `2t/(1+t²)→0`。但前两行都以 V₁ 为 dominant source，任何逐行 signed permutation 至少有一行误差不趋0。covariance 最小特征值为 `1−1/√(1+t²)→0`，恰好被 A20 排除。此例不否定带固定 λ 的定理。

## 6. d=4 的可直接代数版本、非空域与率

在 A3 下用统一预算 `α_j,β_j≤δ`，则

\[
L=1-\delta^2-24\delta,
\quad C_4^{\rm row}=128+192/e^2\approx153.9844,
\]

\[
\boxed{
E(\delta)=\delta^2+
\frac{9(128+192/e^2)\delta^2}{1-\delta^2-24\delta}.}
\tag{A22}
\]

若 `L>0`、`E(δ)<1` 且 `2E(δ)<λ`，则存在 signed permutation P，使

\[
\|Y_j-(PV)_j\|_2
\le\sqrt{2(1-\sqrt{1-E(\delta)})},
\quad
\|Y-PV\|_{L^2(\mathbb R^4)}\le2\sqrt{2E(\delta)}.
\tag{A23}
\]

这些门对任何固定 λ>0 都在足够小 δ 下非空；例如 `δ=.005, λ=.5` 可满足。该数值只是定理常数的固定量级例，不来自 Fine 或科学数组。常数明显保守，未称最优 finite threshold。

当 δ→0、λ 固定时，A22 是 `E=O(δ²)`，所以 axis L² norm 为 `O(δ)`。平方统计量误差与表示误差在小 dense rotation 中均为角度一阶，说明一般不能期待 `o(δ)`：二维旋转嵌入四维时，总 span 误差为 `√2|sin2θ|`，最佳 signed-permutation L² 误差为 `2√(1−cosθ)`，二者比值趋1/2。

## 7. 哪些量不是当前 actual Fine 的无标签输入

### 7.1 输出中心与方差不能随手换

若当前实际输出是 Z，Q 下均值 μ_j、标准差 σ_j>0，定义 `Y_j=(Z_j−μ_j)/σ_j`。若已有 Z 与 Z² 的 span 候选及误差 `α_j^Z,β_j^Z`，则可构造 Y 的候选，预算为

\[
\alpha_j^Y\le\alpha_j^Z/\sigma_j,
\quad
\beta_j^Y\le(\beta_j^Z+2|\mu_j|\alpha_j^Z)/\sigma_j^2.
\tag{A24}
\]

因此 output scale、特别是 variance 下界不能省略。使用估计中心/标准差时，需另计总体规范偏差；不能把 empirical TRAIN 标准化当成 A1 的总体 conditional 标准化。

### 7.2 前轮 posterior bridge 到本 Gaussian anchor

本轮输入 δ 是总体 sufficient-statistic span 误差，既不是普通 ECE，也不是直接储存的 CAL NLL。前轮 posterior bridge 先产生 log-odds 误差并除以同一规范下 contrast 的最小奇异值。其 finite-domain 版本只控 Ω，仍需域外四阶残差预算；单独小 tail probability 不够。

若已获得同一合法 c 的**全域** mixture-law span 界，而 baseline regime propensity 为 π₀(c)>0，则 baseline conditional law Q 满足 `Q≤P_c/π₀(c)`，所以同一未重规范残差的 L² 界至多扩大 `1/√π₀(c)`。之后仍须 A24 的输出规范。baseline 真实因素由其真实 mean、diagonal variance 标准化为 N(0,I) 只改变 `A₂` 的 basis，不改变它作为函数子空间；并未让这些 latent 参数成为算法已知。

同轮 [TAIL_POSTERIOR_BOUND.md 的 T7](TAIL_POSTERIOR_BOUND.md) 已直接给出 Gaussian anchor Q 下的全域 span 界，且其 A→Q 转换已经计入 π₀。若从该 T7 接入本定理，**不能再套上一段重复除以 √π₀**；只需继续处理同一输出规范、basis 与 A24 所需的中心/尺度。

### 7.3 Anchor 到 Q0 / 多时距支持仍未闭合

证明只控制一个固定 Gaussian anchor 上的函数误差。G0 的 TRAIN/state 混合参考及多时距端点是其他 law。若另外证明 `dQ0/dQ≤C0`，才可直接把**同一函数、同一尺度**的 L² 预算扩大 `√C0`；端点 law 类似。这个 domination 对不同 Gaussian means/variances 或 state mixture 常会没有有限全域常数；不能默认为1。

还原到实际 Z 与物理 S 后，要把 affine core 和余项一同搬到 Q0，再重新处理 Q0 normalization 分母。Round2 的 exact normalized-score discrepancy 或有效范数界可用于下一步，但 conditional innovation remainder、小 anchor 误差与多时距 state-score remainder 仍是不同预算。

因此 A23 的 η 不能直接填入以 TRAIN/Q0 标准化为前提的 G0 gap 公式。需要可信的 support/transport、tail moment、再规范和有限样本误差界；本轮没有用旧 truth-assisted A/D、物理数组或有利数据值替它们填空。

## 8. 地位与边界

| 环节 | 本轮状态 |
|---|---|
| Gaussian 全域 approximate quadratic closure → 每行 near single-axis | **Proved**，A18；cubic witness 得 O(δ) |
| Covariance 非塌缩门 → one-to-one axis assignment | **Proved**，A20；invertibility alone 不足 |
| 同 anchor、同规范下 signed-permutation 误差 | **Proved**，A19/A23 |
| 当前 Fine 满足 anchor 全域 span 误差、λ 门 | **Unresolved**，未读模型或数据 |
| 可观测有限 posterior 条件矩 → 足够小的全域 δ | **Conditional / Unresolved**，强度、tails、统计覆盖未补齐 |
| Gaussian anchor → 实际 Q0 和多时距 endpoint score | **Unresolved**，不能跨规范代入 |
| 论文级新颖性 | **Unresolved**；是 Gaussian/Hermite 的显式稳定性推导，未完成新一轮 primary novelty 比对 |

这里补的是一个此前缺失的条件性数学环节，不是 learned model 的实证验证。新推导、已知技术的专门化和文献新颖性分开记录；没有将显式常数或较快率本身当作研究首创。

固定公式脚本和结果见 [quantitative_closure_checks.py](quantitative_closure_checks.py)、[QUANTITATIVE_CLOSURE_CHECKS.json](QUANTITATIVE_CLOSURE_CHECKS.json)。它们只核手工多项式的 Gaussian 解析矩、witness 系数与非退化反例，没有 PRNG 或实际表示评估。

后续条件链见同轮 [尾部后验 T7](TAIL_POSTERIOR_BOUND.md) 与 [transport / selection S1–S5](TRANSPORT_AND_SELECTION.md)：T7 的误差已经是 anchor Q 口径，不重复计 π₀；迁移与重新归一化则仍须另外满足后文前提。
