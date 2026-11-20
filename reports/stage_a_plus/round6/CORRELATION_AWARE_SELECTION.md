# 保留动态相关与标准化抵消的 pair-gap 条件证书

2026-11-04。范围见 [本分支scope](CORRELATION_AWARE_SCOPE.md) 与 [Round6 scope](ROUND6_SCOPE.md)。只读前轮文本/JSON摘要，检查仅用固定解析矩阵；无科学数组、模型、拟合、PRNG、训练、服务器或GPU。

**结论：** 范数界把所有 signed cross terms 推到最坏方向，确可漏掉真实的分数抵消。一个合法的无标签替代接口，是对**完整、保留真解释的候选集合**同时控制候选与Fine的动态联合矩，然后直接给每个 target pair 的 robust gap；有限Gram/PSD外松弛可以保留标准化、跨轴和多时距的共享约束。它们都不需要把真实物理交叉项当作已知输入。但当前摘要没有这些完整集合/统一矩置信域，三个seed与head残差Pearson不能替代它们。

本文件的定理不主张可逆Fine已具有物理轴语义。候选解释的结构包含性、匹配覆盖与统计事件须分别成立；选集保证也不命名无标签律不能锚定的外部物理标签。

## 1. 固定分布、规范与score

固定状态参考Q0、多时距端点对Qh及预定非负权重ωh、Σhωh=1。它们可以非平稳；Qh的两个边缘不必等于Q0。用同一Q0分别将Fine参考和候选轴逐行中心化、标准化：

\[
E_{Q_0}Y_j=E_{Q_0}Z_{\theta j}=0,
\quad E_{Q_0}Y_j^2=E_{Q_0}Z_{\theta j}^2=1.
\tag{C1}
\]

Y是实际Fine的函数；Zθ是候选生成解释中的因素函数，作为同一观测X的函数定义。θ包括一个对所有时距、所有pair共同固定的轴置换。所有方差严格正且相关二阶矩有限。候选尺度按自身Q0方差定义，不把Fine的分母塞给物理因素。

\[
S_j^Y=\sum_h\omega_hE_{Q_h}(\Delta_hY_j)^2,
\quad S_j^{\theta}=\sum_h\omega_hE_{Q_h}(\Delta_hZ_{\theta j})^2,
\quad g_{ab}^Y=S_b^Y-S_a^Y.
\tag{C2}
\]

Fine的固定选集为I；所有a∈I,b∉I都须具有严格正的物理gap才能得到相同唯一top-k。FP32 tie/index规则和经验估计误差另计；本文件的严格正gap不混同机器tie判定。

## 2. Signed dynamic correction：精确保留抵消

令Eθj=Yj−Zθj，定义

\[
R_j^{\theta}=\sum_h\omega_hE(\Delta_hE_{\theta j})^2,
\quad C_j^{\theta}=\sum_h\omega_hE(\Delta_hY_j\,\Delta_hE_{\theta j}).
\]

则直接展开平方有

\[
\boxed{\delta_j^{\theta}:=S_j^{\theta}-S_j^Y
=R_j^{\theta}-2C_j^{\theta},}
\tag{C3}
\]

\[
\boxed{g_{ab}^{\theta}=g_{ab}^Y+
2C_a^{\theta}-R_a^{\theta}-2C_b^{\theta}+R_b^{\theta}.}
\tag{C4}
\]

Cj是signed cross moment，不能仅用|Cj|或其Cauchy上界代替后再声称保留了符号。不同h的项可以抵消；不同轴的修正也可以共享一个方向，从而在pair差里抵消。计算总体平均score只需要各h所列的同h联合矩；跨不同h的随机相关在联合统计置信域/同episode集中时仍须正确处理，不能把不同h当独立重复样本。

### 重新标准化并没有被省略

C3依赖C1：两个对象已经各自重新标准化。此时在Q0上必有

\[
\operatorname{Var}(E_{\theta j})
=2\operatorname{Cov}(Y_j,E_{\theta j}),
\tag{C5}
\]

而非擅自假设该covariance为0。它是两者方差都为1产生的约束，正是应保留的尺度抵消之一。

若从未标准化core X与余项R写Yraw=X+R，应先使用Round2 N1：

\[
S_{Yraw}-S_X
=\frac{2C_\Delta+R_\Delta-S_X(2c_0+v_R)}
{V_X+2c_0+v_R}.
\tag{C6}
\]

例如R=tX、t>−1时，分子严格为0；舍弃variance correction会错误产生非零分数变化。C1–C5是对同一重新标准化问题的另一种写法，不是对分母误差的忽略。经验标准化与总体标准化之间仍需有效统计界。

## 3. 为什么小L²预算不提供signed方向

若只知道||Eθj||Q0≤ηj，以及Qh边缘相对Q0密度比≤Ch,0、Ch,1，记

\[
c_h=\sqrt{C_{h,0}}+\sqrt{C_{h,1}},\quad
K=\sum_h\omega_hc_h^2,
\]

则

\[
R_j^{\theta}\le K\eta_j^2,
\quad |C_j^{\theta}|\le\sqrt{S_j^YR_j^{\theta}},
\quad |\delta_j^{\theta}|\le
\beta_j:=2\sqrt{KS_j^Y}\eta_j+K\eta_j^2.
\tag{C7}
\]

这就是前轮范数证书。小η确实限制signed cross的**幅度**，却不决定它取哪个符号，也不决定两轴修正如何联动。保持C1的归一化几何、时序PSD和可观测动态operator结构，还可比C7收紧；另一独立分支见 [DIRICHLET_SELECTION.md](DIRICHLET_SELECTION.md)。此处不把该分支与候选集分支混为一个已执行接口。

### 固定的符号不确定例

取Q0下Σ=I4、一步增量二阶矩

\[
M=\begin{pmatrix}.8&.4&0&0\\.4&1.2&0&0\\0&0&.85&0\\0&0&0&1.8\end{pmatrix}.
\]

M的特征值处于(0,4)，可由平稳Gaussian线性过程的矩实现：令lag covariance A=I−M/2、innovation covariance I−A²≻0。这里只验证矩可实现，不生成轨迹。

Y为原坐标，候选Z±仅将前两坐标旋转±.1弧度；两种候选的逐行静态距离相同，前两行为`η²=2(1−cos .1)`。但

\[
S_1^{Z_\pm}=.8\cos^2(.1)+1.2\sin^2(.1)
\ \pm .8\sin(.1)\cos(.1),
\]

分别约.88345与.72452；第三轴score仍.85。相同η预算允许一正一负的target-pair gap。例子只说明二阶/L²信息不足以指定符号，**没有把这些旋转同时当作满足full-rank生成条件的同一总体律解释**，不否定full-rank总体辨识定理。

## 4. 完整候选解释的robust pair-gap定理（Proved under coverage）

令Θ为预先声明的完整生成/因素函数类；每个θ都给一个可在观测X上定义的Zθ。令数据产生相容集Ĉ以及修正向量的置信外集D̂⊂Rᵈ。假设存在概率至少1−δC−δM−δY的联合事件，其上：

1. **Truth coverage / matching coverage**：某个真实因素解释及其合法共同匹配θ*属于Ĉ。
2. **完整矩覆盖**：对所有θ∈Ĉ，真实总体修正向量δθ属于D̂；特别包含δθ*。可用更大类的uniform事件保障，不能只覆盖看过的三个seed。
3. **Fine gap覆盖**：`|ĝ_ab−g_ab^Y|≤ν_ab` 对全部跨选集pair同时成立。若直接研究固定有限数组恒等式，可令ν=0；这不等于总体统计误差为0。

支持函数h_D(v)=sup_{d∈D}vᵀd。则同时有

\[
\boxed{L_{ab}=\widehat g_{ab}-\nu_{ab}
-h_{\widehat D}(e_a-e_b)
\le g_{ab}^{\theta_*}
\le \widehat g_{ab}+\nu_{ab}
+h_{\widehat D}(e_b-e_a)=U_{ab}.}
\tag{C8}
\]

**证明。** θ*在集合内且δθ*在外集内；C4给`gphysical=gY+(eb−ea)ᵀδθ*`，再取线性函数的sup/inf及Fine误差即可。三个事件不要求相互独立，union bound足够。证毕。

全部Lab>0即可保证I所匹配的真实因素是同Q0/Qh下严格最慢k集。Lab≤0只说明这项证书未通过，不说明Fine失败；若某Uab<0，则在这些前提下，该pair在全部相容真实解释中均反向。

### 这如何避免“假设知道真实cross term”

对任意显式候选函数Zθ，Cθ、Rθ是观测输入和候选函数的联合矩，不需读取真实state。覆盖真实候选后，其cross term自然被整体外集覆盖。困难转移为**完整候选类、真解释保留、全局搜索/上界和统一矩统计**，并没有消失。Round4的 [version-space接口](../round4/VERSION_SPACE_CERTIFICATE.md) 提供一项条件性truth-coverage机制；当前没有实际实现此完整类的覆盖或直径/支持函数上界。

θ中的匹配必须共同固定；若存在多个合法匹配，保守做法是全部纳入Ĉ。不能按pair/horizon分别挑让gap好看的排列，也不能将生成likelihood不识别的外部物理名称视为已知。若没有任何可信的轴关联/匹配前提，把所有置换都纳入后证书可能自然空泛；不能静默挑有利匹配使其通过。

仅证明“某种与I关联的因素集合最慢”不等于已证明每个Yj单独准确编码对应物理因素。若论文还主张individual-axis fidelity，仍需独立的表示/匹配保证；慢子空间或选集可以在更大的等价类上保持不变。

## 5. 可执行数学对象：联合矩支持函数，而非逐项最坏符号

### 5.1 Pair correction的ellipsoid上/下界

对单pair令w=(Ca,Ra,Cb,Rb)、`l=(2,−1,−2,1)`。若有效联合置信外集为

\[
\mathcal W=\{\widehat w+\Gamma^{1/2}u:\|u\|_2\le r\},
\qquad\Gamma\succeq0,
\]

则

\[
\boxed{\ell^T\widehat w-r\sqrt{\ell^T\Gamma\ell}
\le\kappa_{ab}\le
\ell^T\widehat w+r\sqrt{\ell^T\Gamma\ell}.}
\tag{C9}
\]

此处r是联合不确定半径，不是物理余项。Γ保留两轴/多个h统计量的共享变化；如果不确定性只沿pair共同抵消方向，`lᵀΓl=0`，即使各分量分别变化很大，pair预算仍可为0。

Γ不能直接取三seed样本协方差后宣称覆盖真解释；也不能把逐transition asymptotic covariance当独立episode下的有限置信域。它必须来自一个有效uniform事件，或作为已证明包含全部候选矩的确定性outer envelope。若D̂与Fine gap估计共享数据，可对`(SY,δ)`构造联合外集直接优化整个gap，避免拆成ν与support时再损失共享抵消；完整覆盖要求不变。

### 5.2 State/temporal Gram PSD外松弛

令`Fθ(X)=(Y(X),Zθ(X))∈R^{2d}`。所有分量相对Q0中心化。定义

\[
G_0^\theta=E_{Q_0}F_\theta F_\theta^T\succeq0,
\quad G_h^\theta=E_{Q_h}
\begin{bmatrix}F_\theta(X_0)\\F_\theta(X_h)\end{bmatrix}
\begin{bmatrix}F_\theta(X_0)\\F_\theta(X_h)\end{bmatrix}^{\!T}\succeq0.
\]

Gh是endpoint的**二阶矩**，非平稳时不把它误称为已中心的covariance。令D=(−I,I)，则增量Gram为`M_h=D G_h Dᵀ`。如下约束都是线性/PSD约束：

- G0的Y/Y块取观测参考的有效联合矩域；Y/Z、Z/Z块取完整候选函数类的uniform矩外域。
- `diag G0=1`，保存各对象自身的标准化；如有同Q0的误差预算，加入`2−2(G0)_{Yj,Zj}≤ηj²`。
- 已知端点domination时加入`(Gh)00≼Ch,0 G0`、`(Gh)11≼Ch,1 G0`。
- 将已认证的signed cross-moment区间、共享参数约束或C9的联合外集一同加入，不独立拆散它们。

在这样的外集U中，candidate物理pair gap

\[
\sum_h\omega_h\{(D G_hD^T)_{Z_bZ_b}
-(D G_hD^T)_{Z_aZ_a}\}
\tag{C10}
\]

是线性目标。最小化/最大化给安全lower/upper bounds。若U包含真Gram，PSD外松弛允许一些不可由真实函数/过程实现的矩也不会破坏安全，只会使界更保守。相反，删去真Gram的有利约束会使证书失效；仅保留PSD不能凭空推断真实cross符号。

这是具体有限维SDP或SDP加二阶锥外域的数学接口，并非本轮运行过的优化器。若将来数值求解，最小化问题的近似primal值不是已认证lower bound；需要合法dual-feasible证书、误差控制或经过验证的解析界。总体标准化或候选函数本身由数据估计时，其误差也必须并入uniform域。

## 6. 与范数界的关系、严格改进例

只用C7，修正向量落在box`Dnorm=∏j[−βj,βj]`，C8退化为

\[
g_{ab}^{physical}\ge\widehat g_{ab}-\nu_{ab}-\beta_a-\beta_b.
\tag{C11}
\]

若相同有效输入下，候选/PSD外集进一步保留完整的归一化与联合矩约束，它的修正集合可与Dnorm相交；支持函数单调性保证lower bound不差于C11。精确候选集合、有效Gram外松弛、范数box形成

\[
\inf_{\theta\in\widehat C}g_{ab}^\theta
\ \ge\ \inf_{G\in U}g_{ab}(G)
\ \ge\ g_{ab}^Y-\beta_a-\beta_b,
\tag{C12}
\]

前提是U确实外包候选矩且保留同一范数门；不同统计误差或不同支持不能用这条层级直接比较。

### 固定block-rotation类：大轴误差、零score修正

取Σ=I4、stationary lag covariance `diag(.9,.9,.2,.2)`，h=1，因而

\[
M=\operatorname{diag}(.2,.2,1.6,1.6),\quad I=\{1,2\},
\quad g_{ab}=1.4\ \ (a\in I,b\notin I).
\]

将候选解释类**明确限制**为两个block内部所有orthogonal旋转：`Zθ=(Rθ₁⊕Rθ₂)Y`。这是一个完整可写的条件候选类；若它保留真解释，所有候选都给同一最慢block。候选函数直接由观测Y及给定角度构造，其相关矩由观测Σ/M计算，不需要输入真实cross term。

对任何θ，`S_j^θ=S_j^Y`，所以修正集D={0}，robust lower gap恒1.4。取π/4旋转时，每行`η²=2−√2≈.585786`；静态individual-axis误差明显不小。令Rj、Cj如C3，有

\[
R_j=S_j^Y\eta^2,
\quad C_j=\tfrac12S_j^Y\eta^2,
\quad R_j-2C_j=0.
\]

即使额外知道精确Rj，仅保留Cauchy绝对值也给pair预算

\[
\beta_a^{CS}+\beta_b^{CS}
=(.2+1.6)(2\eta+\eta^2)>1.4,
\]

因此范数证书拒绝，而完整signed-moment证书通过。这里通过的是selected block，不是π/4混合后每个native轴都恢复了单个物理轴；正好展示selection与individual-axis fidelity的区别。

限制同样重要：当前无标签记录没有证明实际相容类恰限于这些block旋转。若完整类还含跨block、同样可信的解释，应全部加入，robust gap可能下降；不能由seed都落同block就删去未找到的解释。此例只证明两种条件界可以严格分离，不是实际Fine的新增成功结果。

## 7. 有限统计与现有摘要为何不够

对一个审计前冻结的完整有限候选类M、固定匹配和有界episode统计，可用独立完整episode的Hoeffding/union bound同时控制所有候选cross moments与pairgap；轨迹内部的时间相关和多个h共享被允许。若每个待估episode统计位于[−B,B]，n条独立轨迹、J个预定统计的统一误差可取

\[
B\sqrt{2\log(2MJ/\delta)/n}.
\tag{C13}
\]

Gaussian平方/乘积不自动有界。应另用有效tail/moment集中，或明确截断后补tail偏差；方差clamp不能把观测函数的平方统计变成有界。无限候选类需要相应complexity/uniform界；固定有限candidate若不含真解释，C13不能提供truth coverage。

模型与encoder由同一TRAIN选择时，不能把它们条件固定后对TRAIN套固定函数Hoeffding。独立audit episodes、全类uniform覆盖或有效交叉拟合须另行证明。CAL已用于选snapshot，不能再无条件作为独立确认。某个未知真实Z的cross moments不属于仅凭Y就能直接估计的统计量；只有明确候选函数及有效完整覆盖后，C3中的相关矩才具有合法的可观测外集。

旧Round3报告中，相关矩链在TRAIN为12/12，Cauchy链为9/12；CAL分别9/12与8/12。Round6 [现有输入审计](CURRENT_INPUT_AUDIT.md)重新核了这些旧摘要。它们使用真实A/D/Σ/M与余项，层级仍是oracle机制诊断。当前84份diagnostics中的Pearson是head residual对currentFine/action的关系；12份G0是逐轴std/speed摘要。二者均不提供完整候选因素到Fine的signed动态矩，更不提供完整支持函数上界。

有限二阶摘要即使很理想，也不认证条件联合分布、完整truth coverage或候选外集。三seed一致只能约束三个已找到的点；对候选直径/支持函数通常给下界或局部观察，不能作为全部相容解释的上界。

## 8. 停止位置与成果定位

| 环节 | 状态 |
|---|---|
| 标准化后signed dynamic correction与pair抵消 | **Proved**，C3–C6 |
| 完整truth-covering类的robust upper/lower gap | **Proved under coverage**，C8 |
| 联合ellipsoid支持函数、temporal/state PSD外松弛 | **Proved conditional interfaces**，C9–C10 |
| 相同输入下不弱于normbox，且可严格更紧 | **Proved / fixed analytic example**，C12与block例 |
| 当前实际Fine的完整候选类、uniform cross-moment confidence | **Unresolved**，现有摘要不齐 |
| 外部physical target语义、有限Q0/Qh统计与matching覆盖 | **Conditional / Unresolved** |
| 新方法或论文级新颖性 | **未声称**；support function、Gram PSD与robust optimization均为经典数学工具 |

本轮保留的增量是：选择可靠性可由target-specific动态联合结构决定，不能由统一余项norm是否小替代；同时给出真实cross未知时合法的候选解释接口。它不支持扩大训练，也没有把oracle相关矩的好结果包装成actual Fine的部署证书。

固定核验见 [correlation_selection_checks.py](correlation_selection_checks.py)、[CORRELATION_SELECTION_CHECKS.json](CORRELATION_SELECTION_CHECKS.json)。完成后由root/其他代理独立审查；本稿不以自检代替独立复核。
