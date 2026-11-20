# Round 5：不假设真实 pmin 的 posterior 尾部桥

2026-11-04。执行范围见 [TAIL_POSTERIOR_SCOPE.md](TAIL_POSTERIOR_SCOPE.md)。不读科学数据/模型，不采样、不训练；固定检查只是概率向量和 Gaussian 解析矩算术。承接 [Round 4](../round4/APPROXIMATE_POSTERIOR_BRIDGE.md)，本文件最多推进到充分统计量 span；同轮 [quantitative closure 附录](QUANTITATIVE_CLOSURE.md) 已补后续条件性数学环节，但不能把该条件定理当作实际 Fine/G0 保证。

## 1. 判断

**可以去掉真实 posterior 的统一 pmin，得到一个合法且可条件性非空的吸收界。** 只要求 head posterior 在可观测区域 Ω 内远离0；真实 p 偏离 q 的坏集合由真实 posterior KL 控制。再用真实 Gaussian anchor 下二次 log-odds 的小集合收缩，将未知真实 odds 的系数 norm 吸收到左侧；最终不需要把未知真实系数/尾矩直接填入公式。

这还不是当前实际 Fine 的证书。剩下的关键输入是能趋零的**真实 posterior KL 上界**、真实 Gaussian anchor 结构、固定 context 的合法统计估计、head 的总体 score/tail 矩与 head contrast 强度。单独把 KL 上界换成观测 cross-entropy 的粗界，会使本文坏集合预算恒饱和，无法进入吸收区。本轮未读取当前数据，因此没有声称实际预算非空。

## 2. 对象、measure 与固定规范

固定合法 context c=(x_t,a_t)。令 P 是真实 future 物理状态 V 给定 c 的混合 law，U 有 m=9 个 regime，propensity πu(c)>0。p_u(v) 是真实 posterior，q_u(v) 是由固定 Fine/head/正 propensity 定义的模型 posterior。两者均归一化；p 不假设统一正下界。

取仅由观测 future x′ 与固定 head 决定的区域

\[
\Omega=\{x':\min_u q_u(x')\ge\epsilon\},\quad0<\epsilon\le1/m,
\]

也允许另加预先固定的可观测约束。设 \(A\ge E_P[1_\Omega K(V)]\)，其中 \(K(v)=D_{KL}(p(v)\|q(v))\)，自然对数。A 是**未归一 restricted KL 上界**；若输入为条件于 Ω 的 KL 均值，必须先乘 P(Ω)。

选一个真实 Gaussian anchor \(Q=P(V\mid c,U=0)=N(\mu_0,D_0)\)，D0正定，π0=π0(c)>0。观测上的 Q 就是给定同一 c、U=0 的 x′ 分布，故固定 head score 的 Q 矩原则上可用观测量定义，无须物理源标签。P≥π0Q，所以

\[
E_Q[1_\Omega K]\le A/\pi_0.
\]

真实每个 regime 的 future law 还须为正定 Gaussian，且在同一个 physical V/共同混合下表达。这样真实 pivot log-odds \(\ell_p=(\log(p_u/p_0))_{u=1}^{m-1}\) 在 anchor 标准坐标 \(Z=D_0^{-1/2}(V-\mu_0)\sim N(0,I)\) 中是至多二次多项式。定义 head odds ℓq、误差 r=ℓp−ℓq，及

\[
R=\|r\|_{L^2(Q)},\qquad N=\|\ell_q\|_{L^2(Q)}<\infty.
\]

这是固定 context/anchor 的定理，不是跨所有 context 的自动一致保证。π0、A、尾矩与 s 若依赖 c，就不能拿平均值当统一下界。下面坐标/统计量 basis 全部固定；s 的数值仍随逐轴 affine gauge 变化，须沿用 Round 4 的 Q0/basis 规范。

## 3. 不假设真实下界的 good/bad 分解

任选固定 α∈(0,1)，设

\[
\varphi(z)=z\log z-z+1,\quad
\phi_\alpha=\varphi(\alpha)>0,\quad
C_\alpha=\frac{\log^2\alpha}{\phi_\alpha},\quad
\beta=\epsilon\phi_\alpha.
\]

令 G=Ω∩{p_u≥αq_u 对所有u}，B=Gᶜ。因为

\[
K(v)=\sum_u q_u(v)\varphi(p_u(v)/q_u(v)),
\]

每项非负，若 v∈Ω∖G，则某 p_u/q_u<α，因而 K(v)≥εφα=β。故

\[
Q(\Omega\setminus G)\le d,
\quad d:=\min\{1-t,A/(\pi_0\beta)\},
\quad t:=Q(\Omega^c),
\quad Q(B)\le\bar\delta:=t+d.
\tag{T1}
\]

这里 d 是坏集合质量的上界，没有把 p 的未知下界当作输入。也可用精确 binary-KL 阈值取得略紧常数；本文用逐项非负式的显式 β，不据此声称常数最优。

在 z≥α 时有

\[
\log^2z\le C_\alpha\varphi(z).
\tag{T2}
\]

证明：z≥1 时 \(\varphi(z)\ge\tfrac12\log^2z\)。α≤z≤1 时置 w=−logz，\(\varphi(e^{-w})=\int_0^w se^{-s}ds\)；该值除以 w² 为 \(\int_0^1 se^{-ws}ds\)，随 w 递减，故最坏常数在 z=α，且 Cα≥2。

G 上利用 qu≥ε，有 \(\sum_u[\log(p_u/q_u)]^2\le(C_\alpha/\epsilon)K\)。pivot 差矩阵 J=[−1,I] 的 operator norm squared=m。因此

\[
\|r1_G\|_{L^2(Q)}\le g,
\qquad g^2:=\frac{mC_\alpha A}{\pi_0\epsilon}.
\tag{T3}
\]

相较直接 log-Lipschitz＋Pinsker，此 good-set 预算只呈 ε⁻¹，且没有假设 p≥ε。代价明确留在 B 的控制中。

## 4. Gaussian 二次多项式的小集合收缩

对 standard Gaussian 下 degree≤2 的 scalar 多项式 h，Gaussian hypercontractivity 给 \(\|h\|_p\le(p-1)\|h\|_2\)，p≥2。该常数见 [O’Donnell 作者版 Theorem 11.23](https://www.cs.cmu.edu/~odonnell/papers/Analysis-of-Boolean-Functions-by-Ryan-ODonnell.pdf) 的 Gaussian semigroup 形式；直接 polynomial 表述见 [Diakonikolas–Kane，Theorem 12](https://cseweb.ucsd.edu/~dakane/CoversPolynomials.pdf)。这是经典输入，不是本项目新结果。

对向量 ℓp，先以 Minkowski 作用于平方和，再逐分量应用上式，得到同样的 \(\|\ell_p\|_p\le(p-1)\|\ell_p\|_2\)。Hölder 因而给

\[
\|\ell_p1_B\|_2\le k(\bar\delta)\|\ell_p\|_2,
\quad
k(\delta):=\inf_{p\ge2}(p-1)\delta^{1/2-1/p}.
\tag{T4}
\]

定义 k(0)=0；0<δ≤1 时，令 L=log(1/δ)。精确最优 exponent 是

\[
p_*=
\begin{cases}
2,&L\le4,\\
\tfrac12[L+\sqrt{L^2-4L}],&L>4.
\end{cases}
\]

因此该 hypercontractive bound 给出的严格吸收区为 \(\delta<e^{-4}\approx.0183156\)。固定 p=4 则给较易读的充分条件 δ<1/81。这里不是声称该阈值是所有 Gaussian 多项式最优小集合阈值；它只是本次 hypercontractivity/Hölder 计算的阈值。小 δ 时 k(δ)=O(√δ log(1/δ))。

## 5. 吸收未知真实 odds norm

head score 在 Ω 上有确定界

\[
\|\ell_q(v)\|^2\le K_q^2,
\quad K_q^2=(m-1)\log^2\frac{1-(m-1)\epsilon}{\epsilon}.
\]

令 \(H_{tail}^2=E_Q[1_{\Omega^c}\|\ell_q\|^2]\)。虽然真实坏集合未知，head 坏集合能量仍可用

\[
\|\ell_q1_B\|_2\le h,
\qquad h^2:=H_{tail}^2+K_q^2d
\tag{T5}
\]

控制。N、t 和 Htail 均是固定 head 与可观测 anchor law 的 functional。有限样本置信上界仍需依赖/尾矩/复杂度条件，不能以有限缓存上的数值替换总体输入；本轮没有估计它们。

真实 ℓp 是 Gaussian 多项式，故有限 L2；结合 N<∞，R 也有限。由 triangle、T3/T4/T5，

\[
\begin{aligned}
R&\le\|r1_G\|_2+\|\ell_p1_B\|_2+\|\ell_q1_B\|_2\\
&\le g+k(\bar\delta)\|\ell_p\|_2+h\\
&\le g+k(\bar\delta)(R+N)+h.
\end{aligned}
\]

只要 k=k(δ̄)<1，就得到

\[
\boxed{R\le\frac{g+kN+h}{1-k}.}
\tag{T6}
\]

**未知真实 odds 的 coefficient norm/尾矩已从右侧消除。** 这依赖真实 Gaussian degree2 结构及 anchor measure，而非宣称真实结构已经由校准诊断验证。head 的坏集合/tail 能量没有只用概率代替，也没有把自然参数强度当可任意缩放的常数。

在当前四维九regime Gaussian head 的固定同一 context 上，若真实/模型 odds 具有 Round 4 的充分统计量形式，且 \(\sigma_{min}(\hat L)\ge s>0\)，则精确代数恒等式 ΔT=−Lhat⁻¹r 给

\[
\boxed{\|T(H(V))-BT(V)-b\|_{L^2(Q)}
\le\frac{g+kN+h}{s(1-k)}.}
\tag{T7}
\]

这是 anchor Q 下的全域充分统计量 span 界；不是混合 P 下自动同常数的界。若要混合 norm，可逐 component 各设 Gaussian anchor 并加权；只对一个 anchor 成立的结果不能直接由 P≥π0Q 反向推广到 P。后续 G0 若使用另一 Q0/Qh，还必须另证 measure/endpoint coverage。

## 6. 为什么真实 KL 与 CE 的差别在这里决定能否进入吸收区

真实 A 可以由正确模型覆盖/逼近与学习理论导出条件上界；本轮没有证明当前算法的这类上界。可观测 regime 标签的 proper cross-entropy 满足

\[
E_P[1_\Omega K]\le CE_\Omega
:=E_P[1_\Omega(-\log q_U)].
\]

所以不是“没有 Bayes entropy 就任何上界都不可能”。但若**仅**将 A 设为 CEΩ，则本文 route 的坏集合预算必饱和：Ω 上每个 qu≤1−(m−1)ε，故

\[
CE_\Omega\ge P(\Omega)\{-\log[1-(m-1)\epsilon]\}
\ge\pi_0Q(\Omega)\{-\log[1-(m-1)\epsilon]\}.
\]

当 Q(Ω)>0 时，对 m≥2、α∈(0,1)，\( -\log[1-(m-1)\epsilon]>\epsilon\phi_\alpha=\beta\)。于是 \(CE_\Omega/(\pi_0\beta)>Q(\Omega)\)，T1 得 d=Q(Ω)、δ̄=1、k=1，T6 **不能吸收**。若 Q(Ω)=0，则 t=1，本来就直接落在 k=1，不能使用严格的“0>0”推理。这比“CE界可能很松”更具体。

该结论只针对单独使用 CE 的这一套阈值/吸收证法；不是所有无标签方法不可能的结论。交叉熵差、覆盖假设、独立的 posterior 学习误差上界或更有信息的验证方法可能给更紧 A，均须另行证明。将 empirical CE 估成总体 CE 也还需要合法统计控制。

## 7. 非空性、实际局限与停止位置

T6 有条件性非空适用域：固定坐标与 π0、s、N 及 α 下，若可选区域满足 t→0、Htail→0，同时 A/ε→0 **且 Kq²d→0**，则 d→0、g→0、k→0、h→0，右侧可趋0。若 ε→0，一个显式充分条件是 \((A/\epsilon)\log^2(1/\epsilon)\to0\)；单独 A/ε→0 并不够，因为 Kq²可按 log²(1/ε) 增大。无需不可能的全域真实 pmin，也不需显式知道真实 odds 系数大小。这个联合条件不是仅凭 A→0 自动满足；区域越大通常 ε越小，真实误差上界必须足够快缩小。

[固定解析检查](TAIL_POSTERIOR_CHECKS.json) 另给四维九regime、自然 rank8 的已知 Gaussian 参数例：anchor 为 N(0,I4)，4个 mean context 与4个 variance context，head 只作预先指定的微小 mean 偏差；a 用已知 generative joint-KL 的解析上界，cube 的质量/score尾矩用 Gaussian erfc 与截断二/四阶矩精确计算。

| 预先指定的 cube radius / mean相对偏差 | 吸收后 span 上界 | 是否达到预先固定 .01 预算 |
|---|---:|---|
| 4 / 10⁻³ | 3.40130227 | 否 |
| 6 / 10⁻⁶ | .0266894613 | 否 |
| 8 / 10⁻⁹ | .0000450015 | 是 |

三例均进入 k<1，但不等于都给实用误差；前两例失败保留。第三例说明条件界在固定规范下确可非空，所需误差极小也说明不能把“存在非空例”当成现实可行性。这只是常数可行性展示，不是当前 Fine 或可观测 A 的实验证据，也没有训练或采样。

若仅有 q 下界与小平均 KL，且没有 Gaussian/其他尾部结构，unweighted odds 控制不成立：在质量 η 的事件上令 p0=e^(−1/η)、其余p均分，q恒为uniform；其余事件p=q。此时 A=O(η)→0，但 \(E\|r\|^2\sim(m-1)/\eta\to\infty\)。固定向量检查保留这个解析序列。吸收法正是用真实 quadratic 结构排除这种任意稀有极端行为。

本文未覆盖近似 sufficient-statistic span 到逐轴 affine 的后续步骤；同轮 [QUANTITATIVE_CLOSURE.md](QUANTITATIVE_CLOSURE.md) 已在完整 Gaussian anchor、总体输出中心/方差规范及 covariance 非塌缩门下补出 O(δ) 条件定理，[TRANSPORT_AND_SELECTION.md](TRANSPORT_AND_SELECTION.md) 则另给迁移、重规范与 G0 score 的条件接口。实际固定 c 的可趋零 KL 上界、真实 Gaussian/anchor 与 finite-model 兼容、总体 head 尾矩置信预算、σmin/gauge、closure 的实际规范/非塌缩前提，以及实际多时距 transport/score 输入预算仍须建立。**T7 与同轮 closure/transport 可以继续作为条件理论链，不能据它们宣布实际无真值证书已经获得。** 若使用本 T7，所得误差已在 anchor Q 下且 A 中已作 π0 转换；不能再重复按 mixture→anchor 多除一次 √π0。

## 8. 可选 weighted log-score 界：避免真实 pmin，但不等于轴误差

Ω 上 z=p_u/q_u≤1/ε。定义

\[
D_\epsilon=\frac{M\log^2M}{M\log M-M+1},\qquad M=1/\epsilon.
\]

有 \(z\log^2z\le D_\epsilon\varphi(z)\) 对 0≤z≤M 成立（z≤1 时常数2足够，z≥1 时该 ratio 单调增加）。所以

\[
\sum_u p_u[\log(p_u/q_u)]^2\le D_\epsilon K.
\]

若 h_u=log(p_u/q_u)，则 \(\sum_{u>0}p_0p_u(h_u-h_0)^2\le\operatorname{Var}_p(h)\le\sum_up_uh_u^2\)。因此相应 true-posterior-weighted odds 风险可由 DεA 控制，无须真实 pmin 或 Gaussian 系数。它将稀有真实类的误差降权；这些权重本身含真实 p，而且没有给固定坐标的 unweighted ΔT/G0 风险，不能把 weighted 与 unweighted 证书混同。

## 9. 来源及检查

新查 primary 限于上文 Gaussian hypercontractivity 的作者文本与具体 theorem。有限样本 nonlinear ICA 的对照沿用 [Round 4 已核 Lyu–Fu 2022](../round4/PRIMARY_NEIGHBORS.md#1-已有定量-nonlinear-ica不能称从来只有总体定理)：不能以本次 tail 接口声称首次有限样本辨识或首次定量稳定性。本文 good/bad、Hölder 与 absorption 是标准工具的专门接法，未作论文级首创声明。

[tail_posterior_checks.py](tail_posterior_checks.py) 固定概率向量、预先指定 Gaussian 参数与解析矩，验证 scalar 常数、bad-set 阈值、CE饱和、吸收常数与三组 Gaussian 预算。无 PRNG、模型、科学数组或物理轨迹输入。脚本结果只是算术/可行性检查，不能验证本文实际科学假设或当前模型能力。
