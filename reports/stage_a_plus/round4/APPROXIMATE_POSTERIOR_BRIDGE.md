# 近似 regime-posterior 到充分统计量 span：只到这一层的条件桥

2026-11-04。承接 [PRIMARY_NEIGHBORS §10](PRIMARY_NEIGHBORS.md#10-exact-regime-posterior-接口的直接原理来源)。本文件没有新文献、模型/数组读取、采样或训练；附带检查只对手工固定概率向量验证确定性不等式，不生成物理数据，不验证实际 Fine。

## 1. 核验结论

给定同一合法 context，若真实与 head posterior 的每个概率分量都至少 ε>0，请求的两个界成立：

\[
E\|r\|_2^2\le \frac{4(m-1)}{\epsilon^2}a,
\qquad
\|\Delta T\|_{L^2}\le
\frac{2\sqrt{(m-1)a}}{\epsilon s}.
\]

这里 m=9，a 是**真实 posterior KL 的期望**，s 是同一 context 的 head 自然参数 contrast 矩阵最小奇异值下界，ΔT 是充分统计量 span 残差；它不是表示相对逐轴 affine 核心的残差。常数是有效但非最紧的。

主要限制比常数更重要：在 R⁴ full-support Gaussian＋全域双射下，所有 head posterior 分量的全域 ε 下界与非零（更不用说满秩）自然参数 contrast 不能同时成立。实际可用版本应限制到 Ω 并另控尾部，或使用另有证明的加权 log-odds 误差条件。当前没有完成近似平方闭包到 axis-affine 的量化证明。

## 2. 固定 context 下的概率不等式

固定合法的 c=(x_t,a_t)，令 V 表示 future 物理状态，分布为真实混合 \(P_c\)。定义 posterior 向量

\[
p_u(v)=P(U=u\mid C=c,V=v),\quad q_u(v)=\hat P(U=u\mid C=c,V=v),
\quad u=0,\ldots,m-1.
\]

假定 p、q 都归一化，且 p_u,q_u≥ε 对 \(P_c\)-a.e. v 成立。必有 ε≤1/m。取自然对数，

\[
K(v)=D_{KL}(p(v)\|q(v)),\quad a_c=E_{P_c}K(V),
\]

\[
r_u(v)=\log\frac{p_u(v)}{p_0(v)}-
\log\frac{q_u(v)}{q_0(v)},\quad u=1,\ldots,m-1.
\]

令 δ_u=p_u−q_u，\(\ell_u=\log p_u-\log q_u\)。log 在 [ε,1] 上为 1/ε-Lipschitz，因此

\[
\begin{aligned}
\|r\|_2^2
&=\sum_{u=1}^{m-1}(\ell_u-\ell_0)^2\\
&\le2\sum_{u=1}^{m-1}\ell_u^2+2(m-1)\ell_0^2\\
&\le2(m-1)\|\ell\|_2^2\\
&\le \frac{2(m-1)}{\epsilon^2}\|p-q\|_2^2\\
&\le \frac{2(m-1)}{\epsilon^2}\|p-q\|_1^2\\
&\le \frac{4(m-1)}{\epsilon^2}K(v).
\end{aligned}
\]

最后一步使用自然对数版本的 Pinsker，\(\|p-q\|_1^2\le2D_{KL}(p\|q)\)。对 v 取期望即得首式。可以用更紧的概率向量几何改善常数，但不改变 ε 与下一步 s 的病态依赖，也不解决适用域缺口。

## 3. 接到当前 Gaussian head 的代数层

另要求真实 future 条件 law 与 head 都具有此前 exact 接口的对角 Gaussian 形式，regime propensity 正确处理，且同一个 c 上的 odds 可以写为

\[
\ell_p(v)=L_{true}(c)T(v)+b_{true}(c),
\quad
\ell_q(v)=\hat L(c)T(H(v))+\hat b(c),
\]

其中 H 是真实观测混合与 Fine 的复合，\(T(z)=(z_1,\ldots,z_4,z_1^2,\ldots,z_4^2)\)，m−1=8。这里 b 吸收 Gaussian normalizer、均值常数和 log propensity；共同的 future Jacobian 已从 regime odds 中消去。若把模型 propensity 与真实 propensity 设成不同但只依赖 c 的正数，它们的差也应吸收到 b，不能忽略掉。

若 \(\sigma_{min}(\hat L(c))\ge s_c>0\)，定义

\[
B(c)=\hat L(c)^{-1}L_{true}(c),\quad
b(c)=\hat L(c)^{-1}(b_{true}(c)-\hat b(c)).
\]

则有**精确代数恒等式**

\[
\Delta T_c(v):=T(H(v))-B(c)T(v)-b(c)
=-\hat L(c)^{-1}r(v).
\]

因而

\[
E_{P_c}\|\Delta T_c\|^2\le
\frac{4(m-1)a_c}{\epsilon_c^2s_c^2},
\qquad
\|\Delta T_c\|_{L^2(P_c)}\le
\frac{2\sqrt{(m-1)a_c}}{\epsilon_cs_c}.
\]

这是请求的第二式。m=9 时系数为 \(4\sqrt2/(\epsilon_cs_c)\)。若取另一种 contrast 排序，必须同时置换 L、r 和 T；s 始终是这一个矩阵在同一坐标与 norm 下的最小奇异值，不是跨 context 平均 rank、训练 Hessian 或旋转曲率。

**Gauge 提醒。** Exact rank 在逐轴可逆 affine 重参数化下保持，但 \(\sigma_{min}(\hat L)\) 的数值不保持。若 \(\tilde y=cy\)，则 \(T(y)=\operatorname{diag}(c^{-1}I,c^{-2}I)T(\tilde y)\)，contrast 的列尺度随之改变；平移还会混合一次/二次统计量并改变常数项。因此不能任意 rescale 把 s 做大后宣称物理辨识更强。比较实际 Fine 前，应预先固定 Q0 的逐轴中心/方差尺度，以及统计量 basis 的排序、尺度权重与 Euclidean norm；若 Q0 只能经验估计，估计误差也须另计。残差 norm 与矩阵必须在同一规范下一起变换，不能只挑有利的 s。

## 4. 条件风险与平均 context 风险不能混写

固定 c 的结论需要该 c 的 a_c、ε_c 和 s_c，以及 regular conditional probability 的合法版本。联合 a.e. posterior 条件只能对几乎处处合法 c 使用；不能在某个零测 context 上任意定义一个 posterior，再用其 rank 作保证。

若只有 \(a=E_C a_C\)，且对几乎处处 c 有共同 \(\epsilon_c\ge\epsilon\)、\(s_c\ge s\)，可推出

\[
E_CE_{P_C}\|\Delta T_C\|^2\le
\frac{4(m-1)a}{\epsilon^2s^2}.
\]

这是**联合平均风险下、允许 B(C),b(C) 随 context 变化**的 span 结论。它没有证明每个 c 具有相同误差上界，也没有产生一个 context 无关的 axis-affine 核心。若 ε_C、s_C 不统一，下界必须留在期望内部：

\[
E\|\Delta T_C\|^2\le4(m-1)
E_C\left[\frac{a_C}{\epsilon_C^2s_C^2}\right].
\]

不能把平均 ε 或平均 s 放进分母。误差可能集中于 propensity 很小或 contrast 很弱的 context。

## 5. 为什么全域 ε 在当前 Gaussian 情形有根本限制

若所有 posterior 分量≥ε，则任意 u,w 的 odds 同时有上下界，因此 \(\log(q_u/q_w)\) 有界。对于固定 c 的 Gaussian head，它是 H(v) 的二次多项式加常数。若 H 是 R⁴ 到 R⁴ 的全域双射，v 走遍 R⁴ 时 H(v) 也走遍 R⁴。任何非恒定实二次多项式在 R⁴ 上不可能有界，故所有一阶/二次自然参数 contrast 必为零，\(\hat L(c)=0\)。这与 s_c>0 冲突。

真实 posterior 同理：若两 regime 的正定 Gaussian future law 不同，log-density ratio 是非恒定二次多项式，至少沿某尾部无界，某类 posterior 将趋 0。只有各 regime future Gaussian 相同时，所有 posterior 分量才可能具有全域正下界；这正不具备本接口需要的 variation。

所以这里不是“ε 很难估准”而已。在全域 Gaussian/full-support/full-rank 版本中，前提不能共同满足。全域正 ε 的一般概率不等式仍正确，但不能将该空适用域当作实际 Fine 证书。

## 6. 合法的有限域 Ω 版本与未控尾部

可令 Ω_c 是仅由 (c,v) 决定的区域，\(\pi_c=P_c(\Omega_c)>0\)，并仅在该区域要求 p、q≥ε_c。连续正 posterior 在固定紧域上可有正最小值，但该值还需真实/模型与域的证据；“有限观测样本上最小值为正”不证明整个域的下界。

定义不归一的 restricted KL

\[
a_{\Omega,c}=E_{P_c}[1_{\Omega_c}K(V)].
\]

同一证明给出

\[
\left(E[1_{\Omega_c}\|\Delta T_c\|^2]\right)^{1/2}
\le\frac{2\sqrt{(m-1)a_{\Omega,c}}}{\epsilon_cs_c}.
\]

若使用条件于 Ω_c 的 L2 norm，则右侧必须除以 \(\sqrt{\pi_c}\)，或直接用 \(a^{cond}_{\Omega,c}=a_{\Omega,c}/\pi_c\)。若只知道全域 a_c，可用 \(a_{\Omega,c}\le a_c\) 取得更松的 restricted 界；不能丢掉归一化或将该界扩展到域外。

要控制全域残差，还必须另给

\[
E\|\Delta T_c\|^2
=E[1_{\Omega_c}\|\Delta T_c\|^2]
+E[1_{\Omega_c^c}\|\Delta T_c\|^2].
\]

第二项没有被域内 KL 界控制。尾部概率小本身也不够，因为 T(H(V)) 含平方项，L2 预算涉及相应四阶量；需要幅度或尾矩控制。可以构造在 Ω 上等于 identity、域外施加任意大平滑 shear 的可逆图，保持所有域内诊断却放大域外误差。只在有限点上评估更不能排除点间行为。

对 posterior 做 clipping 可以使数值 log 有界，但 clipped q 的 odds 一般不再等于原 Gaussian head 的二次式；若代入上式，必须另计 clipped/原始 log-odds 差。clip p 也改变真实 posterior/KL 对象。不能靠 clip 修复 ε 后继续无误差使用原 L 代数。

## 7. 仍不能推出 axis-affine remainder 或慢选集证书

精确端点依赖多项式恒等式：“H_j 是至多二次多项式，H_j² 同时为至多二次”，从而强制 affine 与单轴支持。小 L2 span 残差并不是恒等式；把精确闭包的结论直接连续化，缺少分布/基函数条件数、函数幅度/尾部、规范固定及 quantitative closure 证明。尤其两份近似式的平方操作会引入额外矩与交叉误差。

因此目前只得到 \(T(H)\) 对 \(T(V)\) 的 context 条件线性 span 误差，**没有**得到
\(\|H(V)-(DPV+b)\|\) 的可用界，没有证出 Round 3 所需 innovation/多时距余项，也没有证明 G0 指定选集稳定。

a_c 是总体真实 posterior KL，不是普通 ECE、分类 accuracy 或直接存储的 empirical cross-entropy。proper log-loss 的总体 Bayes excess 可等于该 KL，但把有限验证值转换为此 excess 仍要独立数据/依赖处理、泛化误差与 Bayes 基准等条件。上述问题均未由固定向量算术检查解决。

这不表示没有 Bayes entropy 就不可能给任何上界。因为
\(a_c=CE_c(p,q)-H(U\mid C=c,V)\le CE_c(p,q)\)，可用观测 regime 标签 u 的 multiclass cross-entropy 构造不依赖物理源标签的粗总体上界，再另做合法统计控制；Ω 条件版本同理。该粗界包含不可约后验熵，在 overlapping Gaussian regimes 中即使 q=p 也通常不归零。本轮没有证明它能给当前实际 Fine 一个非空余项/选集预算，也没有计算当前数据。

## 8. 确定性检查的范围

[approximate_posterior_checks.py](approximate_posterior_checks.py) 与 [APPROXIMATE_POSTERIOR_CHECKS.json](APPROXIMATE_POSTERIOR_CHECKS.json) 检查 m=9 的手工固定概率向量、固定混合权重、指定对角 contrast 矩阵的 operator bound，以及 Ω 条件/不归一预算关系。全部无 PRNG、无物理样本、无模型或数组文件输入。它们是常数和归一化算术检查，不能当作 Gaussian 前提可满足、posterior 学习成功或定量闭包的实证验证。
