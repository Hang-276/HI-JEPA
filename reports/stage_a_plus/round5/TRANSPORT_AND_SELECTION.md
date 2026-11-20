# 从 Gaussian anchor 的逐轴误差到实际参考分布与慢度排序

2026-11-04。本文件只给条件理论。观测密度比、尾矩和统计误差尚未在实际Fine上估计；不读取数组或执行模型。

## 1. 为什么还需要这一步

[定量闭包](QUANTITATIVE_CLOSURE.md)在一个合法条件Gaussian anchor Q下，给出中心化、方差1的Fine输出Y与标准Gaussian因素V之间的匹配：每行存在不同的i_j和非零m_j，使

\[
Y_j=m_jV_{i_j}+r_j,\quad |m_j|\le1,
\quad E_Qr_j^2\le E_j.
\]

G0使用另一个Q0作为状态方差参考，并使用多时距端点Qh。不能把Q中的单位方差或误差直接代入G0。下面给出包含尾部与重新归一化的明确桥。

## 2. 观测密度比和尾部加权迁移

假设Q0≪Q，令w=dQ0/dQ。相同可逆观测变换下，latent和observation空间的密度比相同；因此w可定义为纯观测分布的函数，不要求物理标签。它仍不是现有日志已知的量。

固定W>0及p>1。记

\[
T_{Y,j}(W)=E_{Q_0}[Y_j^2\mathbf1_{w>W}],\quad
M_{w,p}(W)=E_Q[w^p\mathbf1_{w>W}],
\]

\[
C_p=\left(E|Z|^{2p/(p-1)}\right)^{(p-1)/p},\quad Z\sim N(0,1).
\]

C_p是显式Gaussian常数，绝对矩公式为
E|Z|^q=2^(q/2)Γ((q+1)/2)/√π。若上述量有限，则

\[
\boxed{E_{Q_0}r_j^2\le
B_j(W,p):=W E_j+2T_{Y,j}(W)+2C_p M_{w,p}(W)^{1/p}.}
\tag{S1}
\]

证明：w≤W部分≤W E_Qr²；w>W上用r²≤2Y²+2m²V_i²及|m|≤1。第一尾项就是可观测T_Y，第二尾项由Hölder及Gaussian绝对矩控制。p=2时C_p=√3，因此尾项为2√3√M_w,2。此处没有把未知真因素的Q0尾矩直接作为输入，而是用可观测密度比的尾矩与anchor结构替代。

若已知全域w≤C0，可以直接用B_j=C0E_j，无须尾项。若没有L²密度比，只有在确实存在某p>1使其p阶矩有限时才可改用p接近1；绝对连续本身不保证这一点。

例如Q=N(0,1)、Q0=N(0,v)，v>1时

\[
E_Qw^p=v^{-p/2}[1-p+p/v]^{-1/2}
\]

仅当p<v/(v−1)有限。方差更宽的state law相对窄innovation anchor，可能没有有限二阶密度比；p越接近1，C_p要求的Gaussian高阶矩越大。因此这不是将support问题隐藏在一个默认取1的常数中。

## 3. 搬到Q0后重新标准化

令Z_j=m_jV_i，σY,j=sd_Q0(Y_j)>0，且σZ,j>0。对Q0中心化并各自除以自身标准差，得到Ytilde与Ztilde；Ztilde恰是相应物理轴在Q0上的标准化，符号由m_j决定。

方差的反三角不等式给

\[
\boxed{\|\widetilde Y_j-\widetilde Z_j\|_{L^2(Q_0)}
\le\eta_j:=\frac{2\sqrt{B_j(W,p)}}{\sigma_{Y,j}}.}
\tag{S2}
\]

证明：先用同一σY归一化Y−Z，再计入σZ/σY的尺度差；两项分别不超过sd(Y−Z)/σY，且sd(Y−Z)≤√B_j。若√B_j<σY,j则自动有σZ,j>0；也可由非零m_j与Q0物理轴的非退化性单独保证。

σY是观测输出的Q0统计量；Y仍用anchor中心/尺度定义，必须一并处理。若Q0=Q，直接使用闭包A19的更紧signed-unit界，不必再损失S2的常数2。经验估计这些方差时，需要可信的总体分母下界；有限正标准差不是总体下界证明。

## 4. 误差到G0评分和选集

设Qh两个端点边缘相对于Q0的密度比分别≤Ch,0、Ch,1，定义c_h=√Ch,0+√Ch,1。则

\[
\|\Delta_h(\widetilde Y_j-\widetilde Z_j)\|_2\le c_h\eta_j.
\]

用Fine自身总体score S_jh=E(ΔhYtilde_j)²，得到

\[
|S_j^{physical}-S_j^{Fine}|
\le\beta_j:=\operatorname{avg}_h
[2\sqrt{S_{jh}}c_h\eta_j+c_h^2\eta_j^2].
\tag{S3}
\]

若只知道平均score S_j和K=avg_h c_h²，由Cauchy可用

\[
\boxed{\beta_j\le2\sqrt{K S_j}\eta_j+K\eta_j^2.}
\tag{S4}
\]

对Fine选出的I，所有a∈I、b∉I若满足

\[
S_b^{Fine}-S_a^{Fine}>\beta_a+\beta_b,
\tag{S5}
\]

则匹配后的物理因素拥有相同的严格最慢k选集。它不命名外部语义目标；预声明物理目标还须等于该相同Q0/Qh下的统计慢集合。

若使用经验score和原FP32选择规则，需对总体Fine score、归一化和数值舍入给有效误差界，并在右侧加入相应pair预算。有限TRAIN阈值审计使用的是同一经验支持上的确定性代数，不能省略统计步骤后转称总体保证。

## 5. 完整条件链与非空条件

在固定合法context、同一anchor及坐标规范下：

1. [尾部后验桥](TAIL_POSTERIOR_BOUND.md)给全域odds误差R及span误差δ≤R/s，输入为proper posterior KL上界、head odds矩和尾部、真实Gaussian结构及head contrast强度。
2. 输出按anchor标准化；若原δ在未标准化输出上，必须先按闭包A24传播中心/尺度。
3. 闭包给E_j与不同轴匹配，需要L_j>0、E_j<1及同anchor Cov(Y)的λ下界满足max(E_j+E_k)<λ。
4. S1–S2搬到实际Q0并重标准化；S3–S5处理多时距端点和目标margin。
5. 若数据驱动，还须同时覆盖所有上述人口量的统计估计与候选选择；固定context下的结论不能用平均context损失直接替代。

当proper posterior误差与tail预算可共同趋0（包括尾部附录的K_q²d→0）、head s与λ及Q0方差下界固定，且可选同一序列W使W E_j→0、T_Y,j(W)→0、M_w,p(W)^(1/p)→0，端点预算保持有界且真实目标gap正时，链条在足够小误差下非空。这里要求联合控制，不是每个模型各自具有有限矩就够。若Q0=Q，可直接使用闭包的同规范界。这是带完整前提的条件定理；本轮并没有在现有Fine上核实这些前提。

## 6. 当前可观测性与实际缺口

w、T_Y、head odds矩、输出covariance都可用观测分布和模型函数定义；此处“可观测”不等于从84份残差摘要可重建，也不等于有限样本已给置信界。固定连续context通常没有重复样本，仍需合法的条件估计、设计或一致界；任取一次已观测context不能自动获得其完整future law。

若Q0取离散经验TRAIN而Q是连续Gaussian anchor，Q0不被Q支配，S1不可直接使用。要么使用总体Q0并控制经验G0统计误差，要么另外证明有限样本函数误差界；不能将此密度比桥偷换成现有TRAIN数组上已成立的误差保证。

此外，只用原始cross-entropy作posterior KL上界无法通过本轮尾部吸收门；见尾部附录的饱和证明。完整链已在数学上补齐条件接口，当前仍缺一组实际有效的输入预算。不同接口的未知量不能互相循环定义后宣称已可估计。
