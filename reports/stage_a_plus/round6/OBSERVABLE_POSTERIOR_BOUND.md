# Round 6：后验 KL 的条件性可观测有限样本上界

2026-11-05，由主代理直接接管并完成。承接[原定范围](POSTERIOR_BOUND_SCOPE.md)。本文件给出完整证明，未启动新的研究分支、子任务、模型执行或数据实验。新有限样本部分经作者逐式自查及固定算术检查，**没有经过新的独立复核**；此前独立审查只覆盖 Brier 主恒等式、完整 q 的注入性和已有选择接口。

**结论：有限样本不等式可以证明，但其成为小误差证书仍依赖真实后验的局部变化控制与合法审计样本。当前记录不满足这些输入，固定 context、尾部和动态残差预算也未补齐，所以实际 Fine 的新证书仍为 0。** 下文不把条件定理写成纯数据、无结构假设的保证。

## P1. 对象、可观测范围与审计设计

令 W=(C,X′)，U∈{1,…,m} 是观测 regime 标签，p_u(w)=P(U=u|W=w)，q(w) 是固定、归一化的模型后验。无需物理因素标签。对 Fine，q 须使用正确设计下的 regime propensity 和全部 regime 的条件似然，不能把仅实际 U 下的 transition NLL 当作 posterior log-loss。

固定可观测集合 Ω，且在 Ω 内 min_u q_u≥ε，0<ε≤1/m。有限个可测 cells B₁,…,B_B **恰好划分 Ω**；不要求 q 在 cell 内常数。域外样本不进入 cell 计数，但分母始终为全体审计样本数 n。目标是未归一的 restricted KL：

\[
A_\Omega=E[1_\Omega KL(p(W)\|q(W))]. \tag{P1}
\]

审计样本 (W_i,U_i), i=1,…,n 独立同分布，q、Ω、cells、ε及下述正则性预算均预先固定，或只依赖独立训练资料；证明可条件于这些训练资料。允许 U 与 W 相关。多条轨迹中的所有 pairs 不能自动作为独立样本。CAL 已参与选择时，也不能直接把最终 q 当作独立于 CAL。

统计结论只要求下面的已知有效余项预算：

\[
V:=\sum_b\mu_b E[\|p(W)-p_b\|_2^2\mid W\in B_b]
\le v,\quad \mu_b=P(W\in B_b),\quad p_b=E[p(W)\mid B_b]. \tag{P2}
\]

零质量 cell 的项定义为 0。一个充分条件是在所声明的**完整 W 空间度量**中，各 cell 直径≤h，真实 p 在 cell 内为 L-Lipschitz。取两个独立 cell 条件副本 W,W̃，方差恒等式给

\[
E_b\|p-p_b\|^2=\tfrac12E_b\|p(W)-p(\widetilde W)\|^2
\le\tfrac12L^2h^2.
\]

因此 V≤κμ_Ω，其中 κ=L²h²/2，μ_Ω=P(Ω)。**L 是真实 p 的条件，不是 head q 的导数界。** 这一步不需真实 p 的正下界；但没有 L 或其他有效 v 时，有限分箱本身不给小余项。

## P2. 精确分解与正确比较方向

定义 Brier 风险 R=E[1_Ω||e_U−q(W)||²]。条件于 W：

\[
E[\|e_U-q\|^2\mid W]=1-\|p\|^2+\|p-q\|^2.
\]

又 E_b||p||²=||p_b||²+E_b||p−p_b||²，故

\[
\boxed{E[1_\Omega\|p-q\|^2]=R-N+V,\qquad
N=\sum_b\mu_b(1-\|p_b\|^2).} \tag{P3}
\]

对 q_u>0，使用 log x≤x−1（p_u=0 项按连续延拓）：

\[
KL(p\|q)\le\sum_u p_u^2/q_u-1
=\sum_u(p_u-q_u)^2/q_u\le\epsilon^{-1}\|p-q\|^2. \tag{P4}
\]

于是 A_Ω≤(R−N+V)/ε。要给上界，必须**上界 R、下界 N、上界 V**，方向不能反。

相应 log-risk 恒等式为 A_Ω=R_log,Ω(q)−H_Ω(U|W)。普通参考分类器 r 满足 R_log,Ω(r)≥H_Ω，所以 R_log,Ω(q)−R_log,Ω(r)≤A_Ω：它给下界而非上界。只有另证 r 的 excess≤a_ref，才有 A_Ω≤R_log,Ω(q)−R_log,Ω(r)+a_ref。未知的 Bayes 熵、逼近误差或全局优化误差不能默认为 0。

## P3. 显式有限样本定理

取 δ_R,δ_N>0，δ=δ_R+δ_N<1，K=Bm。令

\[
\widehat\mu_{bu}=\frac1n\sum_i1\{W_i\in B_b,U_i=u\},\quad
\widehat\mu_b=\sum_u\widehat\mu_{bu},\quad
\widehat N=\sum_{b:\widehat\mu_b>0}
\left(\widehat\mu_b-\frac{\sum_u\widehat\mu_{bu}^2}{\widehat\mu_b}\right),
\]

\[
\widehat R=\frac1n\sum_i1_\Omega(W_i)\|e_{U_i}-q(W_i)\|^2,
\quad t=\sqrt{\frac{\log(2K/\delta_N)}{2n}},
\quad r=\sqrt{\frac{2\log(1/\delta_R)}n}.
\]

定义完全可计算的统计端点

\[
R^+=\min(2,\widehat R+r),\quad
N^-=\max(0,\widehat N-2Kt),\quad
\mu^+=\min(1,\sum_b\widehat\mu_b+Kt). \tag{P5}
\]

若 P2 中有给定 v，则取 v⁺=v；若使用 Lipschitz 条件，则取 v⁺=κμ⁺。以至少 1−δ 的概率，

\[
\boxed{A_\Omega\le A^+
:=\min\left\{\mu^+\log(1/\epsilon),
\frac{[R^+-N^-+v^+]_+}{\epsilon}\right\}.} \tag{P6}
\]

这里第一项只是普适上界，不意味着足够小；[·]_+=max(·,0)。不能把 Ω 内条件平均、全样本 restricted 平均和固定 context 平均混用。

**证明（包括统计常数）。** 对独立 Z_i∈[a,b]，有 P(n⁻¹ΣZ_i−EZ≥s)≤exp(−2ns²/(b−a)²)，下尾同理。可由以下简单推导得到：令 g(λ)=log E exp(λ(Z−EZ))，则 g(0)=g′(0)=0，g″(λ) 是倾斜分布下方差，≤(b−a)²/4；积分得 g(λ)≤λ²(b−a)²/8，再对独立乘积用指数 Markov 并优化 λ。取 Z 为 K 个 joint-cell 指示变量，对每个变量的双尾及全部 K 项作 union bound，得以至少 1−δ_N 的概率

\[
\max_{b,u}|\widehat\mu_{bu}-\mu_{bu}|\le t. \tag{P7}
\]

无需各 cell 计数之间独立，也无需要求每个 cell 被访问。对 v⃗≥0，s=Σv_u，定义 f(v⃗)=s−||v⃗||²/s（s=0 时为 0）。当 s>0，令 a_u=v_u/s，

\[
\partial_u f=1-2a_u+\|a\|^2
=(1-a_u)^2+\sum_{j\ne u}a_j^2\in[0,2].
\]

沿非负正交域内的线段积分，并以连续性处理零向量，得到 |f(v⃗)−f(w⃗)|≤2||v⃗−w⃗||₁。对 cells 求和即 |N−N̂|≤2Kt；又 N≥0，因此 N≥N⁻。同一事件给 μ_Ω≤μ⁺。

Brier 单样本项处于 [0,2]，上面的单尾不等式给 P(R>R̂+r)≤δ_R，再用 R≤2 得 R≤R⁺。不需要这一事件与计数事件独立；union bound 后总失败概率≤δ。代入 P3/P4 与 V≤v⁺，给出第二个上界。最后 KL(p||q)=Σp_u log p_u−Σp_u log q_u≤log(1/ε)，积分得 A_Ω≤μ_Ω log(1/ε)≤μ⁺ log(1/ε)。两个上界取小，证毕。

## P4. 非空性、样本代价与当前 n

P6 是保守的显式充分界，不声称最优样本复杂度。一个易核对的充分接受式是

\[
\widehat R-\widehat N+r+2Kt+v^+<\epsilon a_*, \tag{P8}
\]

它推出 A⁺<a_*，因为 R⁺−N⁻≤R̂−N̂+r+2Kt。给定相应有效预算，P8 确有有限 n 的解；例如正的余量 d=εa_*−(R̂−N̂)−v⁺ 满足

\[
n>\left[\frac{\sqrt{2\log(1/\delta_R)}+
2K\sqrt{\log(2K/\delta_N)/2}}d\right]^2
\]

时成立。d 含审计统计量，因此这是检查已给预算的代数条件，**不是在采样前已知的样本量承诺**。更一般地，在 q=p、固定 ε、已知有限 L 下，若合法细分使 h_n→0 且 K_n√(log(2K_n/δ_N)/n)→0，则 P3/P7 给 R̂−N̂→0、v⁺→0、A⁺→0（以相应高概率意义）；因此条件上界的非空性并未被有限样本形式排除。这个结论不提供实际 Fine 的 L、样本或尾部预算。

当前记录的 24/32 条 CAL 轨迹不能直接充当 P1 的独立同分布 tuples；2048 个 pairs 更不能当作独立 n。即使**乐观地另行拥有** n=24 或 32 个合法独立单位，取 m=9、δ_R=δ_N=.025、最少 B=1，仍有：

| 假设的独立 n | Brier 半径 r | 最小噪声扣减 2Kt | ε=1/9 时 r/ε |
|---|---:|---:|---:|
| 24 | 约 .5544 | 约 6.6641 | 约 4.9900 |
| 32 | 约 .4802 | 约 5.7713 | 约 4.3215 |

精确数值以[固定检查 JSON](POSTERIOR_BOUND_CHECKS.json)为准。N̂≤1−1/m=8/9，所以该半径令 N⁻=0；即使 R̂=0、v⁺=0，第二项也不优于 log 9≈2.1972 的平凡 cap。对任何 ε≤1/9 仍如此，因为 ε log(1/ε)≤(log 9)/9<r；增加 B 也不会改善。**本构造在该样本数量级无法给小于平凡 cap 的界。** 这是此保守 Hoeffding/分箱构造的局限，不是所有统计方法的样本复杂度下界，更不是实际模型错误的证据。

仅为衡量所需量级，在上一轮 T6 中乐观地令 t_tail=0，再取示意常数 π₀=1/9、ε=1/9、α=1/2，其严格吸收要求

\[
A_\Omega<\pi_0\epsilon\varphi(1/2)e^{-4}\approx3.47\times10^{-5}.
\]

这还不是轴恢复和排序全部通过的门槛，实际 ε、π₀、tails 也未获预算。尤其 ε=1/m 会迫使 Ω 内 q 均匀，不能把这个尺度比较当作满秩 Fine 的可实现联合例。其用途仅是说明 2.1972 这种上界无法连接此吸收证明；没有把这些有利常数赋给任何实际父模型。

## P5. 有限校准不能省掉的余项，以及 Fine 的特殊正面结论

固定二类例：同一个 bin 内 W 等概率取两点，p 分别为 (.9,.1)、(.1,.9)，q 恒为 (.5,.5)。bin 完全校准，R=N=.5，但 V=.32，A_Ω=.9 log1.8+.1 log.2>0。忽略 V 会误报后验零误差。这是一般 coarsening 反例；**不以此否定下面实际 Fine 结构的满秩特例**。

对 d=4、m=9 的 Gaussian head，固定合法 c，若

\[
\log(q_u/q_0)=b_u(c)+[\widehat L(c)T(E(x′))]_u,
\quad T(y)=[y,y^2],
\]

且 L̂(c) 的 8×8 矩阵可逆，q 严格正，E 具有可测逆，则从完整 (c,q) 可恢复 T(E(x′))=L̂⁻¹(log-odds−b)，再取前四个坐标并用 E⁻¹恢复 x′。因此 σ(C,q)=σ(C,X′)，完整条件校准 P(U|C,q)=q 等价于真实后验正确。这项注入性本身无需真实 Gaussian kernel；物理逐轴恢复则仍需另一个真实生成结构前提。

这意味着不能笼统说“任何校准都遗漏信息”。困难在于**有限样本能验证的是多粗的校准，以及从粗到完整需要多少正则性**。有限 bins、普通 ECE、舍去 C 或有限精度量化并不继承精确注入性。actual learned rank/强度尚未核验。若改在 (C,q) 空间分箱，所需的是该空间中真实条件 posterior 的变化界；q 的值域光滑并不证明这个界。

## P6. 固定 context、尾部和选择接口：哪些已闭合，哪些没有

**固定 context。** P6 可以用于一个事先合法指定 c 的条件审计 law P(·|C=c)，此时需要从该条件 law 获得独立审计单位，或另有有效条件估计理论。本数据是连续 C，未提供这种重复条件设计。对联合 W 的 A⁺，最多用 Markov 推出：对随机 C，P_C(A_Ω(C)>A⁺/γ)≤γ；它不保证某个预先指定 c，也不保证同一 c 的 rank、anchor、tail、迁移门同时通过。连续 C 的单点条件 law 还只能在合法版本/结构前提下谈论，不能靠边际数据识别任意零质量点。邻域条件化需另加偏差/覆盖控制，本轮不以平均风险强行填固定 c。

**尾部。** P6 只估 Ω 内未归一 KL；不宣称控制 Ω 外 KL。上一轮 [T6–T7](../round5/TAIL_POSTERIOR_BOUND.md)恰可接 restricted A，却还要求同一 c 的真实 Gaussian anchor、t_tail=Q(Ωᶜ)、head odds 二阶尾能量及总矩、π₀和 contrast 强度。小尾概率不等于小尾能量。例如若另有 r₀>2 阶矩有效上界 M≥E_Q||ℓ_q||^{r₀}，Hölder 才给 H_tail²≤M^{2/r₀}t_tail^{1−2/r₀}。当前没有这个 M 的有效总体上界；有限样本未见极端值不能代替它。也不能令 Ω 过小，以小 restricted A 掩盖 t_tail≈1。

**闭链。** 有了同一 c 的 A⁺及上述 tail/head 门，才能接 T7 的 span 误差；再接 [Gaussian closure](../round5/QUANTITATIVE_CLOSURE.md) 的总体规范与非塌缩门，及 [迁移和重新归一化](../round5/TRANSPORT_AND_SELECTION.md)，得到 Q₀ 下各轴 η。T7 已计入 π₀，不能重复除以 √π₀。最后接[主报告 §4](ROUND6_REPORT.md#4-完整的接受条件现在可以写清但实际输入仍为空缺)的动态残差界或完整候选相关界。所有统计门可以 union bound 合并，前提是每个门本身有合法覆盖。

**数据复用。** 若 q 是独立于审计资料生成的预声明 M 个候选之一，可以对所有候选同时应用 P6（各失败概率 δ/M），再允许审计选择。若候选训练或 bins 使用同一审计资料且无统一覆盖，这个有限列表修正也不自动有效。对独立轨迹，只有预先定义的每轨迹有界统计量及对应相同 estimand 才可套同类集中界；不能仅把 n 改成轨迹数而沿用任意 pair 加权。当前这两项均未建立。

## P7. 最终停点

| 项目 | 本次结论 |
|---|---|
| Brier−bin noise＋within-bin variation 精确分解 | 已证明，不要求 q 在 bin 内常数 |
| 不依赖真实 pmin/Bayes 熵数值的 finite-n KL 上界 | P6 已证明；需有效 v 或真实 posterior 正则性及合法独立/统一审计 |
| 低维/细分、充分样本下条件非空性 | 可以成立；无当前实际 Fine 数值接受事件 |
| 当前 24/32 量级下本保守构造 | 即使给予独立单位，仍退化为平凡 cap |
| 从当前记录验证真实 posterior 的 bin 内变化 | 未解决；q 平滑、ECE、NLL、seed 共识均不提供该上界 |
| 平均 context→固定合法 context | 未解决；缺条件设计/覆盖与偏差预算 |
| tails、head 强度、anchor 非塌缩、Q₀ 迁移、动态残差 | 数学接口已写明，当前总体置信输入未建立 |
| 实际 Fine 非空保证 / 相对 raw 独立方法价值 | 本轮未建立；实际新证书 0 |

因此有限样本**证明任务已收尾**；“仅凭当前数据给实际 Fine 非空证书”的研究目标未达成，缺口已定位而未被假设掩盖。保留条件稳定性与失败边界、受控机制诊断；停止把当前结果包装为实际认证或 Fine 必要性。固定检查只验证恒等式、边界方向和小样本算术，不验证任何实际分布假设，也不代替独立审稿。这里的集中不等式和 proper-score 分解为标准工具，本文件不给新颖性声明。
