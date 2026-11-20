# Stage A+1：受限 Gaussian rotation 的识别强度与非渐近误差

日期：2026-11-04。承接 [Stage A 报告](../../outputs/HCF_STAGE_A_20261104/HCF-JEPA_Stage_A_科学问题与理论可行性报告.md) 与 [辨识审查](../../outputs/HCF_STAGE_A_20261104/identifiability_review.md)，不重复 Stage A。已阅读本轮完整交接要求。这里没有新增神经训练、新物理数据或 checkpoint 评分；所有数值为已存在解析值的复核或确定性矩阵公式验证。

## 1. 结论与范围

本轮可以严格闭合一条**受限总体风险到因素旋转误差**的链：当 representation 已知只剩 orthogonal gauge，并且 conditional mean 可自由吸收旋转、conditional covariance 仍被限制为 regime-specific diagonal Gaussian 时，profiled observation NLL 有显式形式。由它可以给出：

1. 二维 canonical angle 的精确风险反演及全角域非渐近上界；
2. 一般维数的风险到 signed-permutation quotient 几何距离的全局上界；
3. 不同方向的局部曲率，以及正则受限 Gaussian 模型中与 efficient Fisher information 的对应；
4. 独立轨迹条件下的受限统计推论、一个简单 minimax 难度下界。

**边界：** 起始假设“只剩 orthogonal gauge”很强。当前 Fine 是受约束有限非线性 flow，本文件没有证明它已处于此类，也没有从其经验 CAL NLL 推出总体 excess risk。所以下面的结论是可证明的受限理论，不是当前 24k 神经训练的高概率认证。

这里的基础原理与 conditional ICA、Gaussian joint diagonalization、FAR-JEPA 和标准统计局部曲率相近；本文件的严格推导不自动成为新颖理论贡献。与 slow selection 的连接还依赖目标动态交叉矩，详见主 THEORY；仅将 κ 与 γ 两个标量机械拼接会遗漏 sharp boundary 的决定因素。

## 2. 先固定无量纲 gauge 与 nuisance

真实一步状态核为

\[
s^+\mid s,a,u\sim N(m(s,a,u),D_u),\qquad D_u>0\text{ diagonal}.
\]

**所有角度与风险必须在同一逐轴 gauge 中定义。** 先固定具有相应物理单位的正尺度 bⱼ，令 H=diag(bⱼ)。可以取 bⱼ=√D₀,ⱼⱼ（reference innovation scale），也可以取 bⱼ=√Var_Q0(sⱼ)（state marginal scale，主 THEORY 为了连接G0而采用这一项）。然后令

\[
v=H^{-1}s,\quad
\Lambda_u=H^{-1}D_uH^{-1}
=\operatorname{diag}(d_1(u),\ldots,d_d(u)),\quad d_j(u)>0.
\]

d_j=Dᵤ,ⱼⱼ/bⱼ²为该gauge下的创新方差，无量纲。只有reference-innovation尺度才保证Λ₀=I；state尺度保证state covariance对角线为1，但不消除state相关。已允许的物理逐轴affine重参数化若同时按同一规则改变b，则相应canonical profiles不变，至多置换/改变符号。

**两个canonical gauge不能混用。** 同一物理线性变换在innovation-whitened坐标为orthogonal Q，在state-marginal标准化坐标中一般是对角共轭后的非正交矩阵。κcurv、Bbarrier及canonical angle都必须标注采用哪个b；即使它们分别无量纲，也不是跨gauge的同一数值。下文所有证明适用于固定任意H，不要求Λ₀=I。

**受限候选类：** 编码器相对于真实逆映射只剩 \(Q\in O(d)\)，输出为 Qv。条件均值允许通过 dense nuisance 任意精确拟合 \(Qm\) 的相应表达；每个 regime 的候选 covariance 为任意正对角矩阵，且可被优化。这里的 profiling 是总体最优化，不是对有限轨迹逐点拟合任意均值。

Q 的 Jacobian 绝对行列式为1。若允许左侧逐轴非零 scaling，其 Gaussian covariance 和 Jacobian 可以相消；我们已将该 gauge 固定，不把可任意缩放的系数作为“因素误差”。固定尺度后才能把 angle/矩阵距离与风险相联系。

对任意 u，旋转真协方差为 \(C_u(Q)=Q\Lambda_uQ^T\)。最佳正对角 Gaussian 近似恰为 \(\operatorname{diag}C_u(Q)\)，因为逐方差最小化 Gaussian cross entropy 的解是该坐标真实方差。因此 profiled excess observation NLL 为

\[
\boxed{F(Q)=\frac12\mathbb E_u
\log\frac{\prod_r [Q\Lambda_uQ^T]_{rr}}{\det\Lambda_u}.}
\tag{1}
\]

均值在此式不提供额外旋转辨识信息，因为当前理想 dense mean nuisance 可将其精确吸收。若另加强制 diagonal AR 等均值限制，风险会增加其他项，(1)不再覆盖全部辨识机制。

## 3. 四种“辨识强度”必须分名

| 对象 | 本文件记号 | 定义/单位 | 能说明什么、不能说明什么 |
|---|---|---|---|
| Gaussian natural-parameter contrast rank/最小奇异值 | rank(L)、σmin(L) | 对指定 sufficient-statistic 坐标与 context 取点构造 L；受变量单位、统计量重标定、取点和权重影响 | 支持总体唯一性证明；不能直接当风险曲率或神经轴误差常数 |
| 两轴 variance-profile 曲率 | κij,H | \(E_u[(d_i-d_j)^2/(d_i d_j)]\)，固定H下无量纲 | 等于同一H的canonical angle局部profiled risk Hessian；在本受限统计模型中还等于per-transition efficient Fisher；不能跨H直接复用 |
| 全角域 rotation barrier | Bij | \(\frac12E_u\log[1+(d_i-d_j)^2/(4d_i d_j)]\)，nats/transition | 表示45° canonical边界的风险；支持非局部角度控制，通常不等于κij |
| 神经因素误差/恢复概率 | Daxis、\(P(\text{axis gate})\) | 由数据分布、估计器、校准与样本数定义 | 需额外证明可实现性、泛化、优化与模型类包含关系；不能和前三项混为一个κ |

尤其，九个 regime 的 natural-rank 可以保持8，但若某些辨识 regime 概率趋零，总体 κ 与 B 仍会趋零；无权 rank 诊断没有控制统计信号量。对 sufficient statistics 作任意可逆缩放还可改变 σmin(L)，而不改变观察过程或上述角度风险。

## 4. 二维非渐近风险到角度

### 命题 K1：精确二维风险（Proved；应用于 Fine 为 Conditional）

只旋转两个坐标 i,j。除 permutation/sign gauge 后取 canonical \(\theta\in[-\pi/4,\pi/4]\)。记

\[
z=\sin^2(2\theta)\in[0,1],\quad
a_u=\frac{[d_i(u)-d_j(u)]^2}{4d_i(u)d_j(u)},\quad
\kappa_{ij}=4E_u a_u,\quad
B_{ij}=\frac12E_u\log(1+a_u).
\]

则

\[
\boxed{F_{ij}(z)=\frac12E_u\log(1+a_u z).}
\tag{2}
\]

**证明。** 二维旋转块的行列式保持dᵢdⱼ；对角元乘积为dᵢdⱼ＋cos²θ sin²θ(dᵢ−dⱼ)²；又cos²θ sin²θ=z/4。代入(1)。其他坐标完全相消。证毕。

若 \(P(a_u>0)>0\)，Fij(z)在[0,1]严格递增，所以有精确反演

\[
|\theta|\le\frac12\arcsin\sqrt{F_{ij}^{-1}(r)}
\quad\text{if }F(Q)\le r<B_{ij}.
\tag{3}
\]

若 r≥B，canonical角度只能被界为π/4；若 B=0，所有 angle 的该 profiled covariance 风险都为零，本类没有该方向的辨识信息。

### 推论 K1a：无需反演的全角域界（Proved）

log(1+a z)关于z凹，且在z=0为0，所以 \(F(z)\ge zF(1)=Bz\)；由log(1+x)≤x，\(F(z)\le\kappa z/8\)。因此

\[
\boxed{B_{ij}\sin^2(2\theta)
\le F(Q)\le \frac{\kappa_{ij}}8\sin^2(2\theta).}
\tag{4}
\]

又 \(\sin(2|\theta|)\ge4|\theta|/\pi\)，故

\[
\theta^2\le\frac{\pi^2}{16B_{ij}}F(Q),\qquad B_{ij}>0.
\tag{5}
\]

这不是仅在θ→0才成立的 Taylor 结论，而是整个 canonical fundamental domain 的非渐近界。

若 \(a_u\le a_{\max}<\infty\)，log(1+a)≥a/(1+a_max)给

\[
\frac{\kappa_{ij}}{8(1+a_{\max})}\le B_{ij}\le\frac{\kappa_{ij}}8.
\tag{6}
\]

因此 B 和 κ 只有在控制最大 variance ratio/contrast 后才能互相比较；弱 contrast 时 B≈κ/8。不能把 natural σmin、κcurv、Bbarrier 和训练 error 都写成同一 κ。

### 当前 rich/weak 解析量级：复用旧KL，新增曲率

**本小节只采用原始AR物理坐标/共同innovation-scale gauge。** 当前AR基准创新方差所有轴同为.12²，因此reference innovation标准化仅为共同缩放，原raw orthogonal angle不变。每条轴自己的variance regime将std乘r=1+2τ，九个regime等概率。对于任意固定两轴，恰有两个regime的a非零，而且共同为

\[
a=\frac{(r^2-1)^2}{4r^2},\quad
F(z)=\frac19\log(1+az),\quad
\kappa_{ij}=\frac{8a}{9},\quad B_{ij}=\frac19\log(1+a).
\tag{7}
\]

| law | τ | r：std倍数 | κcurv | Bbarrier：45° excess NLL |
|---|---:|---:|---:|---:|
| rich | 1 | 3 | 1.58024691358 | .113516805281 |
| weak | .01 | 1.02 | .000348618052886 | .0000435687134465 |

两个B数值已存在于 `outputs/HCF_STABLE_ID_RUN_20261027_V1/gaussian_rotation_penalty_v1/COMPLETED.json`，不是本轮新发现。κ是从同一解析核导出的新整理量，也不能作为新实验。**不能把本表数值直接接到state-unitvariance坐标的χ/γ公式**：不同AR轴的state marginal variance一般不同，FHN也一样；必须先用该state gauge重新形成Dᵤ并计算同一Q对应的F。

此时精确反演为

\[
z\le \min\left\{1,\frac{e^{9r_{\rm excess}}-1}{a}\right\},
\quad |\theta|\le\frac12\arcsin\sqrt{z}.
\tag{8}
\]

τ→0 时 a=4τ²+O(τ³)、κ=32τ²/9+O(τ³)、B=4τ²/9+O(τ³)。natural-rank 的某固定坐标矩阵可为一阶小量，而风险曲率为二阶小量；这再次说明不能把两者未经证明等同。τ=.01严格正曲率但非常弱；不是精确旋转对称。

## 5. 一般维数：局部曲率与全局几何界

### 命题 K2：局部 Hessian（Proved）

设 Q=exp(A)，A为反对称矩阵，用 \(\{A_{ij}:i<j\}\)作 canonical局部坐标，且 variance ratios在正有限区间内。由

\[
[e^A\Lambda e^{-A}]_{ii}
=d_i+\sum_{j\ne i}A_{ij}^2(d_j-d_i)+O(\|A\|^3)
\]

得到

\[
\boxed{F(e^A)=\frac12\sum_{i<j}\kappa_{ij}A_{ij}^2+O(\|A\|^3).}
\tag{9}
\]

证明：将上式代入各对角元log，常数与detΛ相消；每对(i,j)的二阶系数为\((d_j/d_i-1)+(d_i/d_j-1)=(d_i-d_j)^2/(d_i d_j)\)。没有不同pair的二阶交叉项。有限正上下界保证局部导数与期待交换。证毕。

κmin=mini<j κij>0时，Hessian在所有rotation tangent上正定；存在邻域使 \(F\ge(\kappa_{\min}/4)\sum_{i<j}A_{ij}^2\)。这个存在性邻域若没有定量 remainder，不能被冒充已知可用半径。下面给出无需该局部半径的全局界。

### 命题 K3：全d维非渐近 signed-permutation 距离（Proved）

若几乎处处 \(0<m\le d_j(u)\le M<\infty\)，令 R=M/m，κmin>0，并定义

\[
\delta_G(Q)^2=\min_{P\in\mathcal{SP}(d)}\|Q-P\|_F^2,
\]

其中SP(d)为signed permutation矩阵。则对全部Q∈O(d)，

\[
\boxed{F(Q)\ge \frac{\kappa_{\min}}{16R^2}\delta_G(Q)^2.}
\tag{10}
\]

**证明。** 令wᵣⱼ=Qᵣⱼ²，则W为doubly stochastic。由(1)，

\[
F(Q)=\frac12E_u\sum_r
\left[\log\Big(\sum_jw_{rj}d_j\Big)-\sum_jw_{rj}\log d_j\right].
\]

log在[m,M]上二阶导不大于−1/M²，强凹性给每行的Jensen gap至少为 \(\operatorname{Var}_{w_r}(d)/(2M²)\)。而

\[
\operatorname{Var}_{w_r}(d)=\sum_{i<j}w_{ri}w_{rj}(d_i-d_j)^2.
\]

因为dᵢdⱼ≥m²，期待平方差至少为m²κij，故

\[
F(Q)\ge\frac{\kappa_{\min}}{4R^2}\mathcal M(Q),\quad
\mathcal M(Q)=\sum_r\sum_{i<j}w_{ri}w_{rj}
=\frac12\Big(d-\sum_{rj}w_{rj}^2\Big).
\tag{11}
\]

再由doubly stochastic矩阵是permutation矩阵的凸组合，存在一个π使 \(\sum_rw_{r,\pi(r)}\ge\sum_{rj}w_{rj}²\)。逐项\(\sqrt w\ge w\)，并可选择每行符号，故

\[
\delta_G(Q)^2
=2d-2\max_\pi\sum_r\sqrt{w_{r,\pi(r)}}
\le2d-2\sum_{rj}w_{rj}²=4\mathcal M(Q).
\]

与(11)结合得(10)。证毕。所用doubly stochastic分解是经典线性代数/匹配事实：正支撑满足Hall条件，可减去一个正permutation的最小权重并迭代，得到有限凸分解；这里不将其包装为新引理。

该常数保守，尤其rich R=9。它的优点是数学范围明确而不依赖未知局部半径；并不声称能够认证当前实际Fine。若仅某些variance signatures相同，κmin=0，这个axis bound理应失效；可将(11)只对不同signature blocks求和，得到跨块泄漏控制，块内旋转仍不受罚。那是subspace/block结论，不能代替四轴物理恢复。

## 6. 几何旋转误差不等于已测 R²

固定本文件所选H之后，state v的中心化covariance记Cstate，必须另外假设它正定。若eigenvalues在[cmin,cmax]，那么

\[
E\|(Q-P)(v-Ev)\|²\le c_{\max}\|Q-P\|_F².
\]

这才把矩阵角度连接到state L²误差。逐轴总体重新标准化后，Hilbert空间归一化的界

\[
\left\|\frac X{\|X\|_2}-\frac Y{\|Y\|_2}\right\|_2
\le\frac{2\|X-Y\|_2}{\|Y\|_2}
\]

给每轴标准化误差平方不超过 \(4(c_{\max}/c_{\min})\|(Q-P)_{j,:}\|²\)。这仍须和horizon端点的分布覆盖/平稳性一起使用，才能接 Stage A 的速度误差链。

如果Cstate=I、二维rotation且均值已去掉，每轴直接误差为2(1−cosθ)，但FHN一般不满足state covariance=I，不能将这个方便特例硬套已有FHN R²。实际CAL拟合和TEST统计量还涉及有限样本校准误差。

## 7. 与 Fisher information 的精确关系

### 命题 K4：本受限正则模型的efficient information（Proved）

考虑单次条件Gaussian观测，θ为指定两轴rotation angle；均值是可自由调整的nuisance，所有regime diagonal logvariance也是nuisance。只在参数位于正方差内部、可微可积的正则模型内讨论。θ=0处rotation score（符号取决于旋转约定）为

\[
\dot\ell_\theta=\pm\frac{d_i-d_j}{\sqrt{d_id_j}}\xi_i\xi_j,
\quad \xi\mid(s,a,u)\sim N(0,I).
\]

因此条件Fisher为 \((d_i-d_j)^2/(d_id_j)\)。mean nuisance scores是ξ的线性函数；diagonal variance scores是ξⱼ²−1的线性组合；由于Gaussian奇数矩为0，它们与ξᵢξⱼ正交。所以投影去掉这些nuisance不损失该rotation score，per-transition efficient Fisher为κij。一般d维不同pair的score也正交，信息矩阵为diag(κij)。这与(9)的profiled-KL Hessian一致。

**严格限定：** 这不是宣称任何“KL曲率=Fisher=神经辨识强度”。 equality依赖此处的正则参数化、正确模型和特定nuisance tangent。actual flow还存在未受限nonlinear方向、有限架构、clamp边界和优化等问题；这些方向可以改变efficient information或使正则结论失效。

完整Markov轨迹中，正确conditional score可构成martingale difference；所以不同transition score的交叉期待可为0，从而在同一模型假设下累计条件信息可相加。若regime iid且每条长度T，则每条轨迹conditional Fisher为Tκ，n条独立轨迹为nTκ。**这不意味着nT个独立物理实例或nT条独立轨迹**，更不允许把未经残差化的state/Fine分数当iid样本。初态law、action政策、observed context若随参数改变，还需另外计算其信息；此处条件于共同初态并保持外生采样律。

## 8. 有限样本：可以严格说什么

### K5：风险估计事件上的条件性推论（Conditional；代数Proved）

设restricted estimator \(\widehat Q\)满足经验目标 \(\widehat J(\widehat Q)\le\widehat J(I)+o_{\rm opt}\)，并在某事件上有

\[
\sup_{Q\in O(d)}|\widehat J(Q)-J(Q)|\le\varepsilon_n.
\]

其中J为未减基准的总体profiled NLL，F(Q)=J(Q)−J(I)。则

\[
F(\widehat Q)\le2\varepsilon_n+o_{\rm opt},\quad
\delta_G(\widehat Q)^2\le\frac{16R²}{\kappa_{\min}}(2\varepsilon_n+o_{\rm opt}).
\tag{12}
\]

证明为在Q̂和I上分别添加/减去经验风险并用K3。若事件概率≥1−δ，结论同概率成立。它不需要也没有证明当前NN能够达到该经验或总体事件。

一个更可检查的理想化充分事件是：**已知正确条件均值并用真实residuals**估计每regime的innovation covariance，若 \(\|\widehat\Lambda_u-\Lambda_u\|_{op}\le\eta m\)（η<1），且聚合权重为固定真实p(u)，那么所有Q的对角variance相对误差≤η，故每个log项差不超过−log(1−η)，可取εₙ=d[−log(1−η)]/2。代入(12)给明确但较松的风险到角度界。这个事件可用独立整轨迹上的矩阵统计控制；不是实际无标签流程可直接计算的证书，未知mean的估计偏差也没有消失。

本文件没有把经验mean nuisance自由拟合后的TRAIN残差当作真实创新，也没有把同TRAIN训练的Fine当固定函数套iid集中。n必须对应真正独立的轨迹/独立抽样单位；若利用innovation martingale结构改善T依赖，须把该假设和证明写出来。

### K6：受限子类的难度下界（Proved；非当前实验结论）

取两维条件Gaussian子类，固定exogenous regime概率与已知positive diagonal Λu；每条完整轨迹可只有一次transition，或有T次具有共同初态与相同条件均值的Markov transitions。两个假设的观测innovation covariance分别为Λu和R₂αΛuR₂αᵀ，\(0<2\alpha\le\pi/4\)。dense均值nuisance固定为相同可实现的观测条件均值，故没有mean可区分信号。

直接Gaussian KL计算得每transition

\[
\mathrm{KL}(P_0(\cdot\mid u)\Vert P_{2\alpha}(\cdot\mid u))
=\frac12\frac{(d_i-d_j)^2}{d_id_j}\sin²(2\alpha).
\]

对n条独立、各T transition的完整轨迹，用conditional KL chain rule（**不是独立transition假设**）给

\[
\mathrm{KL}(P_0^{(n)}\Vert P_{2\alpha}^{(n)})
=\frac{nT\kappa_{ij}}2\sin²(2\alpha)
\le2nT\kappa_{ij}\alpha².
\tag{13}
\]

若 \(nT\kappa_{ij}\alpha²\le1/4\)，则KL≤1/2，Pinsker给TV≤1/2。任意区分两假设的测试平均错误概率至少(1−TV)/2≥1/4。任何angle估计器若在两个假设下都以概率>3/4实现严格小于α的angle误差，就能用最近中心生成错误概率<1/4的测试，矛盾。因此这种统一精度不能成立。

该下界说明受限类的有效辨识信号包含nTκ，而不由natural rank是否满决定；弱κ使有限样本困难。它是标准two-point/Le Cam原理在Gaussian rotation子类上的专门化，**不声称新的最优minimax rate**，也没有证明实际Fine的failure恰由这个下界造成。这里已知Λ、共同mean让任务更容易；将其作为未知nuisance更大类的困难子类是合法的，但不能反向声称已得到该大类上界。

## 9. 与 slow selection 的接口，以及不可省略的项

K1–K6解决的是restricted factor-angle可确定性。要预测慢目标选择，除了γ还须给出state covariance与increment covariance的几何关系。**本节必须重新固定H为state marginal standard deviation，并在这个H下重算所有dⱼ、κ、B与F；不能直接用§4旧AR数值。** 此时二维state v已总体标准化，相关记ρ，多horizon增量矩阵记

\[
M=\begin{pmatrix}S_1&C_\Delta\\C_\Delta&S_2\end{pmatrix},\quad
\gamma=S_2-S_1>0,\quad
\chi=2C_\Delta-\rho(S_1+S_2).
\]

旋转后逐轴重新标准化的真实gap为

\[
\frac{\gamma\cos2\theta-\chi\sin2\theta}
{1-\rho²\sin²2\theta}.
\]

所以risk→angle→selection的sharp boundary一般还依赖χ。κ和γ相同的系统可以有不同χ，不能声称这两个标量唯一决定全部失败边界。χ=0时pair-gap的一阶误差消掉，但**每个单独score**的一阶误差可能仍存在且相同；不要把pair cancellation误写为所有representation/score误差都二阶。

这正是比普通“κ小更难、γ小更难”故事更值得追查的受限机制：辨识风险通过具体动态几何改变目标间隔。主报告负责给完整可实现反例与新颖性判断，本文件不把尚未核实的创新性升级为已成立贡献。

## 10. 与既有成果的边界

当前Gaussian profile KL是标准计算，45°rich/weak值已在历史文件中存在；Stage A已给exact rank/rotation机制。full-d Hessian、Jensen gap、signed-permutation稳定性与Fisher/Le Cam都是已有数学工具的受限推论。与[FAR-JEPA](https://www.preprints.org/manuscript/202607.2043)的orthogonal起始gauge、过程签名消除rotation、相同签名块边界直接相邻；本文件不能声称首次以signature恢复坐标。与[GIN](https://arxiv.org/abs/2001.04872)的conditional Gaussian理论也不构成替代关系。

新得到的是一个明确、可复查的**定量接口**：κcurv、Bbarrier与natural rank分开；风险到canonical角度不再停留在无常数“identifiable”；一般d维上界有完整证明；统计下界明确独立轨迹与条件信息单位。其是否构成论文新贡献，取决于与χ、γ以及可实现动态模型结合后能否预测既有理论未覆盖的目标选择边界。当前还没有从actual neural NLL接上这个接口。
