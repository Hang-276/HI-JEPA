# Round 2 追加：相关系数与平方根风险桥

日期：2026-11-04。与 `NONLINEAR_THEORY_REVIEW.md` 的 N8 并列，不替换或重新解释原式。全部结论是指定总体分布和受限核心下的命题；没有训练、物理轨迹生成或无标签实际认证。

## 1. 为什么追加这一条

N8 的预算 \(d\log[(1+\nu)/(1-\nu)]\) 在小 ν 时是一阶量，直接加进风险可能很松。本节不先控制两个完整 diagonality contrasts 的差，而是控制指定 pair 的 residual correlation，再把相关程度接到风险。这使余项对允许旋转角预算的影响通过平方进入。

## 2. 条件与记号

沿用 context \(c=(v_t,a_t,u_t)\)、\(V'=m(c)+\xi\)、\(\xi\mid c\sim N(0,D_u)\)。T 是 context-free C¹ diffeomorphism，有限二阶矩、条件 entropy/log-Jacobian 可积，实际 residual covariance 正定；Jactual≥0 是完整总体 observation NLL 减真实 conditional entropy 的 excess。

取可逆仿射核心 A，令

\[
\zeta=A\xi,\quad C_u=AD_uA^T,\quad
e=r(V')-E[r(V')\mid c],\quad
R_u=E[(\zeta+e)(\zeta+e)^T\mid u].
\]

对固定输出 pair i,j，定义

\[
\tau_{i,u}^2=\frac{E[e_i^2\mid u]}{C_{u,ii}},\qquad
\rho^C_{ij,u}=\frac{C_{u,ij}}{\sqrt{C_{u,ii}C_{u,jj}}},\qquad
\rho^R_{ij,u}=\frac{R_{u,ij}}{\sqrt{R_{u,ii}R_{u,jj}}}.
\]

以下只要求实际 R 的两个 variance 正，不要求 τ<1。余项必须是**条件均值已消除的 innovation residual**。未条件中心化的标签辅助 residual 二阶矩可以给其总体上界，但有限数组 plug-in 不自动给总体上界。

## 3. 归一化向量与 pair correlation

### 命题 C1（Proved）

对任意 Hilbert 空间中的 \(x\ne0,y\ne0\)，若 \(\|y-x\|\le\tau\|x\|\)，则

\[
\left\|\frac y{\|y\|}-\frac x{\|x\|}\right\|\le2\tau.
\]

**证明。** 先令 ||x||=1，用
\(\|y/\|y\|-x\|\le\|y/\|y\|-y\|+\|y-x\|\)，第一项为 |1−||y|||≤||y−x||。不需要 τ<1，但 y 必须非零。

在每个 regime 的 L² 空间中，core innovation 与实际 innovation 均为零均值；将 correlation 写为 normalized inner product，然后两次三角不等式给

\[
\boxed{|\rho^C_{ij,u}-\rho^R_{ij,u}|
\le b_{ij,u}:=2(\tau_{i,u}+\tau_{j,u}).}
\tag{C1}
\]

可把 \(b_{ij,u}\) 替换为 \(\min\{2,b_{ij,u}\}\)，因为相关系数差至多 2。记 \(b_{ij}=\|b_{ij,u}\|_{L^2(P_u)}\)。本节其后使用任何已证的逐 regime 上界均成立。

若 \(\nu_u^2=E[e^TC_u^{-1}e\mid u]\)，则每个 τi,u≤νu，因为对任意向量 e 有 \(e_i^2/C_{u,ii}\le e^TC_u^{-1}e\)。因此

\[
b_{ij}\le4\sqrt{E_u\nu_u^2}.
\tag{C2}
\]

pair-specific τ 通常比全维 ν 预算更有解释力；使用 C2 是保守替代，不必强行改成全局最大误差。

## 4. 风险控制实际 correlation

### 命题 C2（Proved）

N5 已证明 \(J_{\rm actual}(T)\ge E_uG(R_u)\)。任意正定 R 满足

\[
G(R)\ge-\tfrac12\log(1-\rho_{ij}^2).
\tag{C3}
\]

**证明。** 将 R 标准化成 correlation matrix，并把 i,j 置于左上角。Schur complement 的对角元至多为 1，其 determinant 由 Hadamard 界至多为 1。因此完整 correlation determinant≤该 pair 的 determinant=1−ρij²。取 −½log 得结论。它也等价于 Gaussian total correlation 不小于任一 pair 的 mutual information。

于是 \(E_u(\rho^R_{ij,u})^2\le2J_{\rm actual}(T)\)。更紧地，由 −log(1−x) 的凸性，

\[
J_{\rm actual}(T)
\ge-\tfrac12\log\{1-E_u(\rho^R_{ij,u})^2\},
\]

故

\[
\boxed{\|\rho^R_{ij,u}\|_{L^2(P_u)}
\le\sqrt{1-e^{-2J_{\rm actual}(T)}}
\le\sqrt{2J_{\rm actual}(T)}.}
\tag{C4}
\]

用 C1 和 L² 三角不等式得到主桥

\[
\boxed{\|\rho^C_{ij,u}\|_{L^2(P_u)}
\le\sqrt{1-e^{-2J_{\rm actual}(T)}}+b_{ij}.}
\tag{C5}
\]

这是对任意可逆 A 的 population statement。它不要求 Jacobian 接近 A，也不要求当前 finite head 达到最优；有限 head 的 approximation error只会使实际 risk 更大。若 J 是经验负数，不得裁为零后冒充 C5 的总体输入。

## 5. 纯二维旋转核心与角度控制

进一步限制 A=Qθ 只旋转两个真实因素，且 D_u 在同一固定 physical gauge 下对角。沿用

\[
z=\sin^2 2\theta,\quad
a_u=\frac{(d_i(u)-d_j(u))^2}{4d_i(u)d_j(u)}.
\]

直接展开有

\[
(\rho^C_{ij,u})^2=\frac{a_u z}{1+a_u z}.
\]

定义

\[
L(z)=E_u\frac{a_u z}{1+a_u z},\qquad
\lambda_{\rm corr}=L(1)=E_u\frac{a_u}{1+a_u}.
\]

只要 P(a>0)>0，L 在[0,1]连续、严格递增且凹，λcorr>0。它与上一轮 κcurv、F(z)、barrier B 均不是同一个量。

### 定理 C3（Proved under restricted core）

令

\[
B_{\rm corr}=
\left[\sqrt{1-e^{-2J_{\rm actual}(T)}}+b_{ij}\right]^2.
\]

则

\[
\boxed{L(z)\le B_{\rm corr},\qquad
z\le Z(B_{\rm corr})\le\min\{1,B_{\rm corr}/\lambda_{\rm corr}\},}
\tag{C6}
\]

其中当 Bcorr<L(1) 时 Z(Bcorr)=L⁻¹(Bcorr)，否则 Z=1。证明：C5 平方后得第一式；逐 u 有 \(a z/(1+a z)\ge[a/(1+a)]z\)，平均得到 L(z)≥λcorr z。Z 是此 particular correlation bound 的精确反演；**不称整体风险—选择保证 sharp**，因为前面的相关系数和 entropy 步可能有松弛。

小 J、b 时，右侧近似 \((\sqrt{2J}+b)^2/\lambda_{\rm corr}\)。相较 N8 先将 O(ν) 加进 risk，此式在 J=0 的余项影响是 O(b²)，因而可能明显更小。但若 b≥√λcorr，则即使 J=0 该角度界也可能已空泛；“形式阶数更好”不是实际已认证。

### 与 T4/T5 的衔接：核心与实际输出必须分开

**版本一，pure core 的排序。** C6 只控制 Q/core 的角度。对 Qv 的分数，其评分误差若逐轴≤εcore，可将 zbar 代入上一轮 T5。无 core score 统计误差时，临界 \(z_*=\gamma^2/(\gamma^2+\chi^2)\)。若 b<√L(z*)，可报告该 particular bound 保持 **pure core 排序**所需的总体 risk 预算

\[
\boxed{J_{\rm actual}(T)<
-\tfrac12\log\left(1-[\sqrt{L(z_*)}-b]^2\right).}
\tag{C7}
\]

若 b≥√L(z*)，本式没有正的 sufficient risk 预算。它是审计时可展示的条件量级，不是已经测得实际 risk 小于该预算；**C7 不自动保证实际 T 的慢选择正确。**

**版本二，实际 T=Qv+r_total 的排序。** 另以 N1 在多 horizon、正确 Q0/Qh 下控制

\[
B_{\rm gap}\ge\left|
(S^T_b-S^Q_b)-(S^T_a-S^Q_a)\right|.
\]

若实际 T 的经验分数逐轴统计误差≤εstat，一个安全的充分条件是

\[
\boxed{\gamma\sqrt{1-\bar z}>
|\chi|\sqrt{\bar z}+B_{\rm gap}+2\varepsilon_{\rm stat}.}
\tag{C7a}
\]

原因是 core gap 的分母 \(1-\rho^2z\) 正且≤1；在分子已大于右侧误差预算时，可安全以下界分子代替 core gap。也可把每轴 multi-horizon remainder 与统计误差合并为 εtotal，再使用 2εtotal。**条件创新 τ 与多 horizon Bgap 是两个不同的预算，前者不能自动代替后者。**

令 \(\ell=B_{\rm gap}+2\varepsilon_{\rm stat}\)。若 0≤ℓ<γ，可定义唯一 \(z_\ell\in(0,z_*]\) 使
\(\gamma\sqrt{1-z_\ell}-|\chi|\sqrt{z_\ell}=\ell\)；将 C7 中 z* 改为 zℓ 后，才得到该保守路线的 actual-T risk 预算。若 ℓ≥γ，这条充分条件在 z=0 也不能保证。z* 仅在 Bgap=εstat=0 时才可直接用于实际 T。

## 6. 非正交部分和现有数组如何使用

实际 A 并非 Qθ 时，应把 \((A-Q)v+r(v)\) 全部放入相对指定 Q 的余项。四维一般 Q 不满足本节纯 pair 公式；任一拟用的 pair reduction必须明确。只把 affine fit residual r 放入 τ、漏掉 nonorthogonal core，将低估总误差。

可用未条件中心化量上界：

\[
\tau_{i,u}^2\le
\frac{E[(r_{{\rm total},i}(V')-E[r_{{\rm total},i}(V')\mid u])^2\mid u]}{C_{u,ii}}
\le\frac{E[r_{{\rm total},i}(V')^2\mid u]}{C_{u,ii}}.
\]

有限已暴露数组可以计算其 plug-in、λcorr、L(z*) 和 C7 的所需 risk 预算；但仍不能得到以下未证项：总体 τ 上界、真实 NLL excess 上界、训练器泛化保证，或任意非线性 representation 的 pair rotation参数。

## 7. 独立性与新颖性判断

### 7.1 严格反例：创新余项小，persistent-state 污染仍可错选

**Proved；解析构造，不是新增 HCF 实验。** 三个独立平稳 AR 因素，方差均为 1，转移系数为

\[
(0,\sqrt{1-\epsilon^2},q),\qquad q=.9,
\quad D=\operatorname{diag}(1,\epsilon^2,1-q^2),\quad0<\epsilon<1.
\]

保持 H={4,16,32}，取可逆 shear
\(T(v)=(v_1+v_2,v_2,v_3)\)，选择核心 A=I，余项 r=(v2,0,0)。它是合法的一般分解，不声称 I 是该映射的最佳 OLS 核心。dense 条件均值可以精确表示变换后的 AR 均值；最佳 regime 对角 Gaussian 的 excess为

\[
J_{\rm free}(T)=\tfrac12\log(1+\epsilon^2)\to0.
\]

因为 r°=(ξ2,0,0)，相对核心 C=D 的创新余项满足 ν=ε→0。但是 level 余项方差始终为 1。记
\(S_2=\overline{2[1-(\sqrt{1-\epsilon^2})^h]}\to0\)，\(S_3=\overline{2(1-.9^h)}\approx1.41617\)。T 的真实重新标准化分数为

\[
(S^T_1,S^T_2,S^T_3)=(1+S_2/2,S_2,S_3).
\]

对于足够小 ε，声明真最慢集合为{2,3}，实际 T 却选择{1,2}。第一轴 score error为 \(-1+S_2/2\to-1\)。N1 的增量交叉项为零，误差主要来自状态方差由 1 变为 2 的重新归一化。

该构造严格排除“conditional-innovation 余项小＋低 NLL，就可以省略 multi-horizon score 余项”的普遍推论；它不反驳 C5–C6 的 core-angle 控制，因为此处 core=I 本来没有旋转。该 baseline 只有一个 regime，没有声称满足 rich nonlinear natural-parameter full-rank，不能称为对 rich 辨识定理的反例。

**增强版：九 regime、四维、自然参数差满秩仍不能省略 score 余项。** 取 rates 为(.1,√(1−ε²),.9,.2)，u=0,…,8均匀且跨时间独立。基准均值为0、nominal std为1；u=j+1仅将轴j的nominal mean设为+1；u=j+5仅将该轴nominal mean设为−1、std设为3（j=0,…,3）。每轴再乘

\[
\text{scale}_j^2=\frac{9(1-\rho_j^2)}{19}.
\]

由于每轴 E nominal-mean=0、Var nominal-mean=2/9、E nominal-variance=17/9，总创新方差为1−ρj²；不同轴的创新交叉矩为0。因此平稳state covariance恰为I、lag二阶矩对角，但不声称非Gaussian mixture下所有状态轴统计独立。

固定当前state，基准与该轴两个特殊regime的Gaussian自然参数差，在该轴(一次项、平方项)中形成2×2块：第一列为(1/scalej,0)，第二列的平方项为4/(9scalej²)。块determinant为4/(9scalej³)>0；八个非基准regime形成block-diagonal rank8。因此每个ε>0均满足相应满秩条件。

仍取可逆T=(v1+v2,v2,v3,v4)、core I。这里

\[
J_{\rm free}(T)=\tfrac12E_u\log(1+D_{2,u}/D_{1,u})=O(\epsilon^2),\qquad
E_u\nu_u^2=E_u D_{2,u}/D_{1,u}
=\frac{\epsilon^2}{.99}\frac{145}{81}\to0.
\]

但state余项方差为1，第一轴的score为(S1+S2)/2→S1/2，仍低于S3。对足够小ε，真目标{2,3}、实际选集{1,2}；dense读出精确，第一输出轴与v1的scalar-affine R²=.5。自然参数满秩、核心正确和小创新余项没有消除persistent state污染。

**严格适用边界。** 这不推翻每个固定ε的定性可辨识性：ε→0时最小innovation variance没有统一正下界，shear也不在纯orthogonal gauge内。它只排除忽略这些条件、或省略multi-horizon/state-score余项的统一可靠性推论。九regime是一个新的解析构造，不是已执行的当前rich/weak实验，更不证明实际FHN失败具有此机制。

这里 J=.5Elog 明确是**自由 profile Gaussian head**的风险，不是固定 variance clamp 和具体有限 Flow 已实现的 actual-Fine 风险。ε→0 最终会越过当前 logstd≥−6 的 head 下限，故极限只属于所声明理论类。有限 ε=.01 时，本构造的真实与最佳候选逐轴 conditional std 均在[e⁻⁶,e²]内（解析核验保存于JSON）；这也不证明当前有限 Flow 已在全域实现该 shear，或任何已训练父模型具有此失败。

### 7.2 未执行的 Gaussian Poincaré 接口

当 V'|c 为上述 Gaussian，余项在相应 Gaussian Sobolev 类中且梯度能量可积时，逐线性方向使用 Gaussian Poincaré 并求和可得

\[
\nu_u^2\le E\!\left[\operatorname{tr}\{C_u^{-1}
D r(V')D_u D r(V')^T\}\mid u\right].
\tag{C8}
\]

这里 Dr 是余项的 Jacobian，Cu=ADuAᵀ。证明先对每个c固定Cu，给向量函数Cu⁻¹/²r套标量Poincaré，再对c|u平均。该接口使用**相对创新尺度下的平均梯度预算**，不要求 global C¹ sup 很小。Gaussian Poincaré 的标准形式可核对 [Ledoux 作者说明 Eq.(2)](https://perso.math.univ-toulouse.fr/ledoux/files/2025/07/GB-poincare-v1.pdf)，本式是其条件化和线性白化推论。

本轮没有测量实际 Fine Jacobian 或这个梯度能量，也未进行新模型推理；因此 C8 只是后续可能收紧 N10 过松 level-residual proxy 的理论接口。多数现有 N10 proxy≥1 不说明真实 ν≥1，更不说明定理被数据否定。

### 7.3 结论性质

这是把 Gaussian covariance/entropy、normalized-vector perturbation、pair correlation 与给定 rotation model 连接起来的有用受限推论。工具本身均经典；本节没有发现文献原创已确认的依据。它的科学价值在于可以明确区分“Jacobian双向连续性不成立”和“单向risk-to-core仍可成立”，并得到比 N8 更有希望的余项量级。

`correlation_risk_checks.py` 和 `CORRELATION_RISK_CHECKS.json` 检查固定小矩阵及解析信息量，不生成随机样本或物理轨迹。包括 normalized-vector bound、Gfull≥Gpair、L²/Jensen bound、二维 L(z) identity和C7阈值反演；这些检查不验证实际 Fine 的总体条件。

## 8. κ纠偏：orthogonal curvature 不控制全部 affine 方向

**Proved；固定物理尺度 gauge 中的局部仿射结论。** 对一般 \(A_t=I+tB\)，令 \(F(A)=E_uG(AD_uA^T)\)。假设相关 variance-ratio 期望有限；去除 diagonal 行缩放方向后，

\[
\boxed{F''(0)=\sum_{i<j}E_u
\left[B_{ij}\sqrt{d_j/d_i}+B_{ji}\sqrt{d_i/d_j}\right]^2.}
\tag{C9}
\]

**证明。** \((A_tD_uA_t^T)_{ij}=t(B_{ij}d_j+B_{ji}d_i)+O(t^2)\)，因此 correlation 的一阶项为括号内系数乘t。对 correlation matrix 写 \(G=-\frac12\log\det\operatorname{Corr}\)，其二阶展开为 \(\frac12\sum_{i<j}\rho_{ij}^2\)，故 \(F(t)=\frac12t^2\)乘C9右侧加o(t²)。这也核对了factor 2。diagonal B 对应未可辨识的逐轴尺度方向，不进入该二阶式。

固定 pair 的 \(x=B_{ij},y=B_{ji}\)，令

\[
R=E(d_j/d_i),\quad S=E(d_i/d_j),\quad
H_{ij}=\begin{pmatrix}R&1\\1&S\end{pmatrix}.
\]

其曲率是 \((x,y)H_{ij}(x,y)^T\)。orthogonal 生成方向(1,−1)给 \(\kappa_{\rm rot}=R+S-2\)；单向shear(1,0)给 \(\kappa_{\rm shear}=R\)。前者的Frobenius norm为√2，因此unit-Fro旋转曲率是κrot/2，不能与单位shear方向在不说明规范时直接比较。以固定Fro gauge并quotient diagonal后，覆盖该pair所有非正交方向的是λmin(H)，不是κrot。

\(RS\ge1\) 来自Cauchy–Schwarz；只有RS>1时H正定。常数但不同的variance ratio可使orthogonal rotation有正曲率，general affine Hessian仍有零方向。这是纯orthogonal起点比一般affine起点更强的具体体现。

在上节九context反例的第一/第二轴，令 \(K=145/81,c=\epsilon^2/.99\)，则

\[
R=Kc,\quad S=K/c,\quad
\kappa_{\rm rot}=K(c+c^{-1})-2=\Theta(\epsilon^{-2}),
\quad\kappa_{\rm shear}=Kc=\Theta(\epsilon^2),
\]

\[
\lambda_{\min}(H)=
\frac{R+S-\sqrt{(R-S)^2+4}}2
=(K-K^{-1})\frac{\epsilon^2}{.99}+O(\epsilon^6)>0.
\tag{C10}
\]

因此，orthogonal κ 可以显得越来越强，而当前允许的 shear 方向同时越来越弱。任意仿射及非线性 Fine 的量化可靠性不能只用正交旋转曲率代言。

这是经典 nonorthogonal Gaussian joint-diagonalization likelihood 的局部特化；相关contrast与pairwise sensitivity已有 [Afsari §3.2.3、§4.3](https://isr.umd.edu/Labs/ISL/ICA2006/Sensitivity_Final.pdf)。本节没有声称新的普适κ定义或任意Flow的最低曲率。它纠正的是本研究应测量哪个允许变换族的强度，以及为什么full-rank与κrot大仍不足以代替实际state-score误差控制。

