# Stage A+ Round 2：实际非线性 Fine 的余项与单向风险桥

日期：2026-11-04。依据：上一轮 `../THEORY.md` 的受限旋转理论；本轮只作数学分析与确定性矩阵/解析公式检查，不读取标签改变模型，不训练、不生成物理轨迹、不改历史文件。

## 1. 本轮最强结论

**已经得到一条不要求 Jacobian 接近线性核心的单向总体风险桥。** 在真实一步条件创新是 Gaussian、条件 covariance 仅依赖 regime 的前提下，最大熵与 log-det 凹性给出：实际非线性可逆表示的条件 Gaussian 观测 NLL excess，不小于其条件残差 covariance 的 Gaussian joint-diagonalization contrast。只要非线性余项的**条件创新尺度**二阶矩小，就能把该 contrast 连回任意可逆仿射核心 A。

这关闭的是一条明确的 **conditional population connection**：

\[
F(A)\le J_{\rm actual}(T)+\delta_{\rm nl}.
\]

它不证明当前学到的 Fine 的 A、余项和总体 NLL excess 已被认证。特别是“小 fitted residual”“高 dense R²”“小经验 CAL NLL”不能直接填入该式当作总体已知量。连接到上一轮 κ/rotation barrier 还需要固定同一个物理尺度 gauge，并另行控制 A 的非正交部分。

同时得到了包含经验/总体重新标准化分母的 exact score discrepancy；这使余项诊断可以直接对齐真实 G0，而不用假设 Fine 已经是一个旋转。

## 2. 记号与两个不同条件分布

以固定物理尺度 H 将状态逐轴标准化为 \(v\)。标准化支持为 \(Q_0\)，多时距合法端点为 \(Q_h\)。它们服务于 G0 score。另一方面，似然理论使用真实一步转移联合律及外部 context

\[
c=(v_t,a_t,u_t),\qquad
V'=v_{t+1}=m(c)+\xi,
\quad \xi\mid c\sim N(0,D_u),\quad D_u\succ0.
\]

这里 D 只依赖 u，均值允许耦合、非线性和动作驱动。Fine 与真观测混合复合后的映射记为

\[
T(v)=Av+b+r(v),\qquad A\text{可逆}.
\]

余项 r 是一个函数，而不仅是某批训练点的拟合残差。A 可以来自规定好的标签辅助事后拟合，但此时需区分“定义一个候选核心”和“证明总体误差小”。若 A 近奇异，下面的相对创新界会恶化；不能因拟合矩阵存在就假定 A 可逆且良态。

## 3. Exact score discrepancy：必须保留重新标准化

### 命题 N1（Proved；总体与有限数组均成立）

对一条输出轴，令 \(X=a^Tv\)，\(Y=X+R\)，其中 R 是该轴余项。以同一 Q0 定义

\[
V=\operatorname{Var}(X)>0,\quad c_0=\operatorname{Cov}(X,R),
\quad v_R=\operatorname{Var}(R),\quad
V_Y=V+2c_0+v_R>0.
\]

各 h 的合法端点上定义并平均

\[
A_\Delta=\overline{E(\Delta_hX)^2},\quad
C_\Delta=\overline{E(\Delta_hX\Delta_hR)},\quad
R_\Delta=\overline{E(\Delta_hR)^2},\quad S_X=A_\Delta/V.
\]

则真实 G0 对该轴重新标准化后的差恰为

\[
\boxed{
S_Y-S_X=\frac{2C_\Delta+R_\Delta-S_X(2c_0+v_R)}
{V+2c_0+v_R}.}
\tag{N1}
\]

**证明。** Y 的增量二阶矩是 \(A_\Delta+2C_\Delta+R_\Delta\)，方差是 VY；相减两个比值并通分即可。均值在差分分子里消失，但通过方差分母进入。h 的端点支持可以不同，不需要平稳性。

把所有期望换成同一实际数组上的规定平均，N1 仍是逐样本恒等式。它不受编码器是否使用同一 TRAIN 训练的影响。当前 G0 使用标准差退化 guard，不是持续作用的标准差 floor；退化状态应拒绝，不能将其静默当作正方差情形。FP64/FP32 舍入需单独计入，不能在接近同分时称为机器精度下绝对同选。

若含 intercept 的 OLS 在**同一 Q0 支持**上拟合 A，则余项均值和对全部 v 的协方差为零，从而 c0=0；该正交性不自动延伸到 CAL、TEST、其他 Qh 或总体。

### 命题 N2（Proved；范数充分界）

令 \(e=\sqrt{v_R/V}<1\)、\(\eta=\sqrt{R_\Delta/V}\)。由 Cauchy–Schwarz、方差反三角与 N1：

\[
\boxed{|S_Y-S_X|\le
\frac{2\sqrt{S_X}\eta+\eta^2+S_X(2e+e^2)}{(1-e)^2}.}
\tag{N2}
\]

它是安全但可能很松的条件性界。若 c0=0，可用实际分母 V+vR，并去掉 2e 预算。若可读取带真值的数组，优先记录 N1 的 signed numerator 与实际正分母，而不是把每个项全部替换成最坏正号。

若两个端点边缘相对 Q0 的密度比分别受 Ch0、Ch1 控制，则对 Q0 中心化的 R 有

\[
\|\Delta_hR\|_{L^2(Q_h)}
\le(\sqrt{C_{h0}}+\sqrt{C_{h1}})\sqrt{v_R}.
\]

有限等长轨迹、全部 L 个时点标准化、每 h 使用全部 L−h 合法起点时，得到 \(\eta_h\le2\sqrt{L/(L-h)}e\)。burn-in 与不等长轨迹须重算覆盖常数。**Q0 小 L² 不控制未覆盖的未来支持。**

## 4. Q＋非正交＋非线性：分解和条件性拼接

对 A 作左 polar decomposition

\[
A=P Q,\quad P=(AA^T)^{1/2}\succ0,\quad Q\in O(d),
\quad N=A-Q=(P-I)Q.
\]

于是 \(T(v)=Qv+Nv+b+r(v)\)。Q 是当前固定欧氏物理尺度下的 polar 因子；若先按行缩放、进行 state whitening、使用数据加权最小二乘，得到的 Q 和角度会改变。必须固定 gauge，不能将不同规范下的 κ、γ 和角度混用。

一般仿射核心的 score 是

\[
S_{A,j}=\frac{a_j^T M a_j}{a_j^T\Sigma a_j},\quad
\Sigma=\operatorname{Cov}_{Q_0}(v),\quad
M=\overline{E_{Q_h}(\Delta_hv\Delta_hv^T)}.
\]

非正交差 \(S_A-S_Q\) 可以直接算该比值，或把 Nv 当作 N1 的余项。对任意目标 a 和非目标 b，

\[
g^T_{ab}=g^Q_{ab}
 +(\delta^{A-Q}_b-\delta^{A-Q}_a)
 +(\delta^{T-A}_b-\delta^{T-A}_a).
\tag{N3}
\]

这是精确差分，不是新的学习保证。若上一轮只证明某二维 Q 的 gQ，而实际 Q 是四维多轴旋转，仍需全部 cross-target pair 的论证；polar decomposition 本身不会把四维问题化成一条二维角度。

## 5. 主结果：最大熵单向总体风险桥

### 5.1 对条件 Gaussian head 作自由 profile

本节假设有限二阶矩、下述 Ru 正定，所用条件 entropy、log-det 与 log-Jacobian 可积；T 为 context-free C¹ diffeomorphism（Jacobian 处处非奇异）。若风险或预算无穷，相应式子至多是扩展实数意义的空泛不等式，不能称为有限有效证书。

候选高斯 head 的均值允许是 c 的任意函数，方差为仅依赖 u 的对角正矩阵。对 Y=T(V')，记

\[
\mu_T(c)=E[Y\mid c],\quad
K_T(c)=\operatorname{Cov}(Y\mid c),\quad
R_u=E[K_T(c)\mid u].
\]

free-profile 的最优均值为 μT(c)，最优逐轴 regime 方差为 Rujj。设 T 为 context-free 可逆 C¹ 映射，条件 entropy 与 log-Jacobian 可积。相对于真实观测条件 entropy 的 NLL excess 为

\[
J_{\rm free}(T)=
\frac12E_u\sum_j\log(2\pi e R_{u,jj})
-E_c h(Y\mid c).
\tag{N4}
\]

**证明。** 最优 Gaussian cross-entropy 只依赖各轴条件残差二阶矩。原观测 change-of-variables 的 Jacobian 与真实条件 entropy 合并后，恰好形成 h(Y|c)；因此不是把 Jacobian 删除，而是在 entropy 表达式中完整计入。

对任意当前有限 mean head、regime 方差 head 或 clamp，

\[
J_{\rm actual}(T)\ge J_{\rm free}(T),
\]

因为它们属于自由 profile 类的子类。这条不等式不要求当前 head 成功达到 free optimum。Jactual 必须是**同一完整 conditional observation NLL、减去同一真实条件 entropy baseline 的总体 excess**；未扣正确 baseline 的观测 CAL NLL 数字、latent Gaussian NLL 或某个参数正则目标不能直接代替。

### 定理 N3：条件 covariance contrast 下界（Proved）

定义经典 Gaussian diagonality contrast

\[
G(C)=\tfrac12\log\frac{\prod_j C_{jj}}{\det C},\qquad C\succ0.
\]

在上述条件下，

\[
\boxed{J_{\rm actual}(T)\ge J_{\rm free}(T)
\ge E_u G(R_u).}
\tag{N5}
\]

**证明。** Gaussian 在给定 covariance 下具有最大 differential entropy，故

\[
h(Y\mid c)\le\tfrac12\log[(2\pi e)^d\det K_T(c)].
\]

对 c|u 平均后，log-det 凹性给 \(E[\log\det K_T(c)\mid u]\le\log\det R_u\)。代回 N4 即得。也可用与同均值、同 covariance Gaussian 的非负 KL 直接证明最大熵这一步。

N5 本身不需要真 V' 是 Gaussian；但要将线性核心 contrast 等同于它的确切 profiled risk，下面需要 Gaussian 条件创新与 covariance 只依赖 u。

### 定理 N4：一般可逆 A 的单向余项桥（Proved；应用 Conditional）

令

\[
C_u=A D_u A^T,\quad
\zeta=A\xi,\quad
r^\circ=r(V')-E[r(V')\mid c],
\quad\varepsilon_u^2=E[\|r^\circ\|^2\mid u].
\]

线性核心 Av' 的条件 covariance 正是 Cu，因此

\[
F(A):=J_{\rm free}(A)=E_uG(C_u).
\]

同时

\[
R_u=C_u+E[\zeta(r^\circ)^T+r^\circ\zeta^T+r^\circ(r^\circ)^T\mid u].
\tag{N6}
\]

若 \(m_uI\preceq C_u\preceq M_uI\)，则

\[
\|R_u-C_u\|_{\rm op}\le
\eta_u:=2\sqrt{M_u}\varepsilon_u+\varepsilon_u^2.
\]

若 ηu<mu，且下式可积，则

\[
\boxed{F(A)\le J_{\rm actual}(T)
+E_u\frac{d\eta_u}{m_u-\eta_u}.}
\tag{N7}
\]

**证明。** 对任意单位向量 a,b，cross-covariance 的双线性形式由 Cauchy–Schwarz 控制为 \(\sqrt{M_u}\varepsilon_u\)，余项 covariance op norm≤εu²，得 η。线段 C+t(R−C) 的最小特征值≥mu−ηu，因此 log-det 差绝对值≤dηu/(mu−ηu)；逐轴 log diagonal 的总差也不超过此量。G 中两项各有系数 1/2，故 |G(R)−G(C)|≤dη/(m−η)。再用 N5。证毕。

### 定理 N5：相对创新尺度界（Proved；通常更自然）

定义

\[
\nu_u^2=E[\|C_u^{-1/2}r^\circ\|^2\mid u].
\]

若 νu<1，L² 正/反三角不等式对任意白化方向给

\[
(1-\nu_u)^2C_u\preceq R_u\preceq(1+\nu_u)^2C_u.
\]

对 diagonal 和 determinant 分别使用该 Loewner 夹界，得到

\[
\boxed{F(A)\le J_{\rm actual}(T)
+E_u d\log\frac{1+\nu_u}{1-\nu_u}.}
\tag{N8}
\]

**证明细节。** 白化后的 core innovation 任意单位方向 L² norm 为 1，余项方向 L² norm≤νu；相加后的 L² norm位于[1−νu,1+νu]。因此 R/C 的相对特征值和逐轴方差比分别在[(1−νu)²,(1+νu)²]，G 的两项最坏相向变化给 N8。这里没有使用 \(D r\) 的大小。

还可以定义实际相对 covariance 扰动

\[
\beta_u^{\rm exact}=
\|C_u^{-1/2}(R_u-C_u)C_u^{-1/2}\|_{\rm op}.
\]

当 βexact<1 时，另一合法界为

\[
|G(R_u)-G(C_u)|\le\frac d2\log\frac{1+\beta_u^{\rm exact}}{1-\beta_u^{\rm exact}}.
\tag{N9}
\]

范数-only 预算 β≤2ν+ν² 也成立，但要求该预算<1时更严格（ν<√2−1）。N8 只要求 ν<1；有限使用另需 log 预算可积，单有 ν<1 并不保证期望有限。哪条更紧由实际 covariance 决定。βexact 仍依赖物理核心与条件 covariance，不是一个已获得的无标签量。

### 5.2 从一般 A 接回 Q/κ 的两条方式

若直接研究 F(A)，N7–N9 已适用，但上一轮仅正交 gauge 的 κ/angle theorem 不能原封不动套给 A。可任选下述明确路线：

1. 用 \(C^Q_u=QD_uQ^T\) 和 \(C^A_u=AD_uA^T\) 的确定性扰动界，控制 |F(Q)−F(A)|，然后接 N8。
2. 把 \(R_{\rm total}(v)=(A-Q)v+r(v)\) 当作相对 Q 的全部余项，直接用 N8。其条件中心化为 \((A-Q)\xi+r^\circ\)，相对 L² 预算满足 Minkowski：

\[
\nu^{\rm total}_u\le
\left\{\operatorname{tr}[(C^Q_u)^{-1}(A-Q)D_u(A-Q)^T]\right\}^{1/2}
+\left\{E[\|(C^Q_u)^{-1/2}r^\circ\|^2\mid u]\right\}^{1/2}.
\]

只有当这一相对余项受控，才能写 \(F(Q)\le J_{\rm actual}(T)+\delta_{\rm total}\)，再进入上一轮的受限角度/排序结论。不能把 OLS residual 数值小当作 nonorthogonal 部分也小。

### 5.3 量级、真实可算性与弱噪声问题

N8 的 δ 约为 2d Eν（小 ν 时）；d=4、ν=.01 的统一预算约 .0800 nats/transition，ν=.001 约 .0080。若要该保守项小于 10⁻⁴，需 ν 约小于 1.25×10⁻⁵。**这不是当前数据的测量结论**，只是说明总体桥可能合法却仍不够紧。若 R 的实际 signed perturbation使 G 变化小，N9 或直接 |G(R)−G(C)| 可显著更好，但需要更强的 oracle 条件矩信息。

ν 是相对于 innovation covariance 的误差，不是 state 方差归一化的 reconstruction residual。真实 innovation 很小时，state residual 看起来微小也可能 ν≥1；A 近奇异也会放大它。只有指定下界、条件矩或相对预算后，才能讨论非空量级。

若条件均值难以计算，可用嵌套条件投影给安全总体上界：

\[
\nu_u^2\le
E[\|C_u^{-1/2}(r(V')-E[r(V')\mid u])\|^2\mid u]
\le E[\|C_u^{-1/2}r(V')\|^2\mid u].
\tag{N10}
\]

有限 regime、正概率下这些量可规划为标签辅助审计；单次有限数组估计不自动成为总体上界。u 的时间索引须是驱动 V' 的 transition regime，不能误取未来观测对应的下一 regime。

## 6. L²、小 Jacobian 与双向风险连续性：不能混为一谈

### 6.1 高频严格反例（Proved）

令 X 是 IID 标准 Gaussian 状态，真 inverse 为 identity，

\[
T_n(x)=x+\frac a n\sin(nx),\qquad0<a<1.
\]

每个 Tn 都是光滑全局双射，导数 \(1+a\cos(nx)\) 介于 1−a 和 1+a；余项 sup norm≤a/n、L²→0。其最优 Gaussian head 的 observation NLL excess为

\[
J(T_n)=\tfrac12\log\operatorname{Var}(T_n(X))
-E\log[1+a\cos(nX)].
\]

Gaussian 矩给

\[
\operatorname{Var}(T_n(X))=
1+2ae^{-n^2/2}+\frac{a^2}{2n^2}(1-e^{-2n^2})\to1,
\]

而周期平均给

\[
E\log[1+a\cos(nX)]\to
\log\frac{1+\sqrt{1-a^2}}2.
\]

因此 \(J(T_n)\to-\log[(1+\sqrt{1-a^2})/2]>0\)，例如 a=.8 时极限 .22314355。score 在 IID 情形下始终为 2，依然不能推导 Jacobian 或完整 NLL 连续。

**准确否定对象。** 本例否定 \(|J(T)-F(A)|\le\omega(\|r\|_2)\)、\(\omega(\epsilon)\to0\) 这种双向连续性，以及单由 L² 得到 log-Jacobian 小。它**没有否定本轮所需的单向式** \(F(A)\le J(T)+\delta\)：此处 F(identity)=0，N8 完全成立。不能用该反例错误宣称“任何 NLL-to-core 桥都必须 C¹”。

### 6.2 一个 C¹ 充分条件（Proved，主要用于双向控制）

若 \(\sup_v\|A^{-1}D r(v)\|_{\rm op}\le q<1\)，则

\[
|\log|\det DT(v)|-\log|\det A||\le d[-\log(1-q)].
\]

这来自 I+A⁻¹Dr 的奇异值界[1−q,1+q]。若定义每轴条件余项比例
\(\tau_j(u)^2=E[(r_j^\circ)^2\mid u]/(C_u)_{jj}<1\)，那么 free-profile covariance 比在[(1−τj)²,(1+τj)²]，所以

\[
|J_{\rm free}(T)-F(A)|
\le E_u\sum_j[-\log(1-\tau_j(u))]+d[-\log(1-q)].
\tag{N11}
\]

这里的方差比指 regime 平均的 Ru,jj/Cu,jj，不是每个 c 下的逐条件方差比；有限有效结论另需 N11 右侧 log 预算可积。它是一个简单充分条件，通常很松。若讨论实际 finite head 的双向 risk，还需额外控制两个表示各自的 mean/variance approximation penalty；任意糟糕的 head 都可使实际 risk 很高，不能仅由表示 C¹ 接近得出实际 NLL 接近。

## 7. 本轮对实际 Fine 的结论边界

| 结论 | 状态 | 当前不能自动替代的量 |
|---|---|---|
| 任意仿射核心＋余项的 normalized score exact discrepancy | Proved | 总体事件需误差泛化；有限数组仅样本内 |
| nonlinear risk ≥ conditional covariance contrast | Proved | 正确完整 NLL、entropy baseline、可积性与 covariance 非退化 |
| conditional L²余项给单向 F(A)≤Jactual(T)+δ | Proved under assumptions | 当前 fitted residual 不等于总体条件创新误差 |
| 回到正交 Q 的 κ/角度模型 | Conditional | 固定 gauge、nonorthogonal remainder、全部 cross-pair |
| L²小推出 Jacobian/NLL 双向接近 | False | 高频 diffeomorphism 已给严格反例 |
| 当前Fine总体 excess已足够小 | Unresolved | CAL NLL不是总体 excess的认证 |
| 新定理已足够ICML创新 | Unresolved | 最大熵、log-det与covariance扰动均是经典工具 |

本轮数学上最有用的改进是：**不再把“Jacobian 双向连续性不可得”误判为“单向核心风险桥不可得”**。最大熵路线允许 actual Fine 的 Jacobian 振荡使 risk 增大，却禁止在条件 covariance 接近核心时隐藏核心 Gaussian diagonality cost。这是后续解释的有效接口。

经典性判断：Gaussian 最大熵、entropy与determinant关系和log-det凹性是标准工具，可参见 [Dembo、Cover、Thomas，Information Theoretic Inequalities，1991](https://isl.stanford.edu/~cover/papers/dembo_cover_thomas_91.pdf)。本报告给出针对当前条件头与余项的完整适配证明；未宣称该数学工具或一般连续性原理首次出现。能否形成新贡献，取决于相对创新余项的可检验量级、与前轮κ/γ/χ选择边界的有效连接，以及独立的失败预测，而不是再增加条件性公式数量。

## 8. 确定性核验与建议给主报告的表述

`nonlinear_formula_checks.py` 与 `NONLINEAR_FORMULA_CHECKS.json` 只计算固定小矩阵和高频例的 Gaussian 解析矩/Fourier级数；不抽取随机样本、不生成轨迹、不拟合模型。检查包括 N1 恒等式、polar 两阶段 telescope、相对 Loewner/contrast 界、Gaussian 高频反例的非零极限。

建议主报告使用：**“在明确条件 Gaussian 创新及条件余项矩界下，已将受限线性核心风险桥扩展到一般非线性可逆表示的单向总体下界；实际 Fine 的余项与总体 NLL excess 尚未认证，因此未建立训练器端到端保证。”**

