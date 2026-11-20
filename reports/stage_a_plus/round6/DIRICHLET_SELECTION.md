# 用可观测动态残差保留归一化与跨时结构

2026-11-04。范围见[DIRICHLET_SCOPE](DIRICHLET_SCOPE.md)。这是一条条件选择界；不预设当前Fine接近动态本征函数，也没有读取其轨迹来估计该性质。

## 1. 动态算子由观测律定义

固定Q0与合法多时距端点律Qh，令H=L²(Q0)。对任意标量观测函数f，定义Df=(Δh f)h，range内积为各h的Qh期望平均。若端点密度比相对Q0分别≤Ch,0、Ch,1，则

\[
\|Df\|^2\le\Lambda\|f\|^2,
\qquad\Lambda=\operatorname{avg}_h(\sqrt{C_{h,0}}+\sqrt{C_{h,1}})^2.
\]

由此L=D*D是有界自伴算子，0≼L≼ΛI，且L1=0。速度分数就是S(f)=〈f,Lf〉，前提是f已在同Q0中心化并标准化。无需假设状态分量独立、平稳或时间可逆；端点law与Q0的关系必须合法。

给定Fine的某个规范化输出y，||y||=1，记

\[
S=\langle y,Ly\rangle,\qquad r_y=Ly-Sy,\qquad \rho_y=\|r_y\|.
\tag{D1}
\]

L、y、S和ρ_y都是观测过程及固定Fine的functional，不需要物理因素或真实余项。ρ_y不等于Fine一步density head的innovation residual，也不等于现存Pearson摘要。

若Qh端点权重为wh,0、wh,1，则在适当条件期望版本下

\[
Ly(x)=\operatorname{avg}_h\{w_{h,0}(x)[y(x)-E(y(X_h)\mid X_0=x)]
+w_{h,1}(x)[y(x)-E(y(X_0)\mid X_h=x)]\}.
\tag{D2}
\]

这说明总体ρ_y的估计涉及双向条件动态和覆盖。有限TRAIN上可用同一轨迹节点图的加权Laplacian精确计算经验Ly；但该图上的数值不能直接当总体ρ_y。当前G0 metadata只保存S，没有Ly或ρ_y。

## 2. 归一化使抵消具有精确形式

令z为同Q0上规范化的目标轴（匹配符号已固定），||z||=1，d=z−y。两个单位范数给〈y,d〉=−||d||²/2，故

\[
\boxed{S(z)-S(y)=2\langle r_y,d\rangle+
\langle d,(L-SI)d\rangle.}
\tag{D3}
\]

证明：展开〈y+d,L(y+d)〉−S，并将Ly=Sy+r_y代入。式中的−S||d||²正是归一化修正，不能在展开前被丢掉。对所有h先平均构成L，也保留了不同h对Ly的抵消。

这不是已经知道真实signed相关项。它用**观测输出自身的动态残差**控制最坏方向：

\[
|S(z)-S(y)|\le2\rho_y\eta+
\max(S,\Lambda-S)\eta^2,\quad\|d\|\le\eta.
\tag{D4}
\]

当ρ_y很小时，一阶误差系数也小；若ρ_y=0，界为二阶。这并不证明y是物理因素：混合后的动态本征函数也可能语义错误，仍须独立的轴匹配误差η保证。

## 3. 更紧的单边界和pair条件

假设0≤η≤√2，令

\[
q(\eta)=\sqrt{\eta^2-\eta^4/4}.
\]

写z=cy+u，u⊥y。由c=1−||d||²/2≥0得||u||≤q(η)，且|c|≤1。利用0≼L≼ΛI，

\[
S(z)\ge S-2\rho_y q-Sq^2,
\quad S(z)\le S+2\rho_y q+(\Lambda-S)q^2.
\tag{D5}
\]

推导：S(z)−S=2c〈r_y,u〉+〈u,(L−SI)u〉，再取上下界。可将区间与[0,Λ]相交。

对选中轴a与未选中轴b，令gY=Sb−Sa，则

\[
\boxed{g_Z\ge g_Y-2\rho_aq_a-2\rho_bq_b
-(\Lambda-S_a)q_a^2-S_bq_b^2.}
\tag{D6}
\]

每个pair下界严格正即保持选集。若ρ、η与score是估计量，须同时取得合法上/下置信界并对右式作保守区间计算，不能将普通点估计代入后称高概率保证。D6仍对未知误差方向取最坏值；[联合候选修正界](CORRELATION_AWARE_SELECTION.md)可另外利用方向或pair间共同结构。

本质上这是经典Rayleigh商扰动理论在本问题上的接口。相关残差、角度和二阶本征向量误差界已有系统研究，见[Zhu、Argentati、Knyazev，2013，§1–4](https://merl.com/publications/docs/TR2013-068.pdf)。不把D3/D4或“本征方向二阶稳定”声称为新原理。

## 4. 只有S、Λ和η时也可以收紧，但不能凭空得到ρ

若没有ρ_y上界，从0≼L≼ΛI只能得到ρ_y²≤S(Λ−S)。更直接的归一化几何区间如下，沿用0≤η≤√2并要求Λ>0（Λ=0时所有速度为0，无严格慢度间隔）。令

\[
\alpha=\arcsin\sqrt{S/\Lambda},\qquad
\theta=\arccos(1-\eta^2/2),
\]

则

\[
\boxed{S(z)\in\Lambda\,[\sin^2\max(0,\alpha-\theta),
\sin^2\min(\pi/2,\alpha+\theta)].}
\tag{D7}
\]

证明：令A=L/Λ，a=〈y,Ay〉、b=〈z,Az〉。Cauchy给
〈y,z〉=〈A½y,A½z〉+〈(I−A)½y,(I−A)½z〉≤√ab+√((1−a)(1−b))。
右侧=cos(arcsin√a−arcsin√b)，而〈y,z〉≥cosθ，故两个角度之差≤θ。

对pair，D7给严格分离的充分门θa+θb<αb−αa。共同η门为

\[
\eta<2\sin[(\alpha_b-\alpha_a)/4].
\tag{D8}
\]

这些是区间充分条件，不是固定实际动态算子上的必要且充分条件；更不能用它们证明实际模型失败。

## 5. 固定、真实可实现的动态例

按scope，取四维独立平稳Gaussian AR(1)，各轴方差1、ρ=(.985,.984,.9,.8)，h=(4,16,32)。在该过程的L²空间中，物理线性坐标是L本征函数，特征值sj=2[1−avg_h ρj^h]。将第一、二轴旋转.03弧度得到Fine函数y；其相对原轴的η=2sin(.015)>0，动态残差范数为|s2−s1|sin(.03)cos(.03)。取通用平稳端点上界Λ=4。

所有矩直接用解析式计算，不采样/训练。检查比较第一、二轴的pair：旧范数下界−.144094，D7下界−.127523，D6下界+.020377；真实pair gap为+.024128。全部结果见[DIRICHLET_CHECKS.json](DIRICHLET_CHECKS.json)。此处对应k=1的排序比较；主线k=2会同时选中这两轴，所以该例不展示主线top-2选集的分离优势。它仅验证同一合法动态过程上，D6确实能比两个范数区间更紧；不是当前九regime模型的完整后验认证或实际Fine测量。主线k=2的完整结构例由同轮block候选分支另行给出，仍不混作同一个后验认证过程。

## 6. 与当前记录连接的边界

仅复用Round5两份JSON，将12父全部48个pair的S、Λ和旧OLS数值floor代入D7/D8。floor是实际轴误差的**下界**，只能用于否决D7充分门；未被否决不等于获得η上界或通过。

另将floor作为最乐观η，在D6中求“若两轴动态残差均≤ρ，ρ最多允许多大”的预算。该预算只描述必要的乐观空间：真实η可能更大，实际ρ也未记录，绝不能把正数当成已获证。r=0只是该公式的理想端点，不是当前模型事实。所有父、pair及不利结果保存在JSON。

所以这条路线给出一个比真值cross-moment更可操作的观测对象，但尚未提供实际输入。未来若仅算有限TRAIN图ρ，也仍只解决经验支持问题；Round5的总体anchor到经验TRAIN缺口、η可观测上界和物理目标一致性仍须各自处理。
