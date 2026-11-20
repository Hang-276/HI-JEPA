# 有限参考样本对逐窗口慢集的影响

慢度的分母是物理坐标的参考方差。这里固定总体参考方差定义的物理慢集为目标，研究用有限参考样本代入分母后发生的变化。已有 Gaussian 模型给定完整参考律，因此这些方差原则上可以积分计算。本节的 Gaussian 部分研究有限样本归一化这一估计方式。后半部分另给参考律未知、但具备已知尾部上界时的数据置信界。 [已有联合窗口定理](../dose_joint_window_20261110/THEORY.md)

## 有限参考归一化的联合极限

沿用第一轴未知的精确剂量类
\[
h(x)=x+\ell p(\{x/\ell\}),\qquad p=F-\mathrm{id}.
\]
三个竞争轴直接可见。每个轴另有独立于评价窗口的 \(N\) 个 iid 参考点，各轴参考点相互独立。Gaussian 核验中这些点服从标准正态。令 \(Q_w(h)\) 为窗口的多时距增量均方，\(V_h\) 为总体参考方差，\(\widehat V_{h,N}\) 为以 \(N\) 为分母的中心化参考样本方差。定义
\[
D_h(w)=Q_w(h)/V_h,\qquad
\widehat D_{h,N}(w)=Q_w(h)/\widehat V_{h,N}.
\]
第一式决定物理目标。第二式决定采用该参考样本时的估计选集。

第一、二轴在极限中具有相同的 AR 系数 \(\rho\in(0,1)\)，恒等总体慢度记为 \(d>0\)。设 \(\sigma_w^2\) 为两条独立轴的总体归一化窗口分数差乘 \(\sqrt T\) 后的极限方差。参考样本方差满足 \(\sqrt N(\widehat V_{j,N}-1)\Rightarrow N(0,2)\)。一阶展开给出恒等解释的估计间隔
\[
\widehat G_{T,N}=g_T+(G_{T}-g_T)
+d\{(\widehat V_{1,N}-1)-(\widehat V_{2,N}-1)\}
+o_p(T^{-1/2}+N^{-1/2}). \tag{1}
\]
其中 \(g_T=d_{2,T}-d_1\)，\(G_T\) 是总体归一化的窗口间隔。三个竞争轴中的远端两个保持固定分离，因此中间竞争分数以概率趋于一等于第二轴分数。参考项和窗口项独立。令
\[
s_{T,N}^2=\frac{\sigma_w^2}{T}+\frac{4d^2}{N}.
\]
只要 \(N,T\to\infty\) 且两者比值有极限，\((\widehat G_{T,N}-g_T)/s_{T,N}\Rightarrow N(0,1)\)。若 \(N/T\to r\in(0,\infty)\)，原 \(\sqrt T\) 标准化下的方差增加 \(4d^2/r\)。当 \(N/T\to\infty\)，参考项消失。当 \(N/T\to0\)，\(\sqrt N\) 标准化下由参考项决定极限，方差为 \(4d^2\)。

对完整相容类，令 \(\widehat L,\widehat U\) 是 \(\widehat D_h-\widehat D_{\rm id}\) 的上下端。已有总体范围为 \([0,S_\ell]\)，\(S_\ell=c\ell^2+o(\ell^2)\)。有
\[
\sup_h\left| (\widehat D_h-\widehat D_{\rm id})-(D_h-D_{\rm id})_{\rm population}\right|
=O_p\!\left(\frac{\ell}{\sqrt T}+\frac{\ell}{\sqrt N}\right). \tag{2}
\]
证明使用相同参考样本计算所有候选。写
\[
\widehat V_h=\widehat V_X+2\ell\widehat{\mathrm{Cov}}(X,p_\ell)
+\ell^2\widehat{\mathrm{Var}}(p_\ell).
\]
\(p=F-\mathrm{id}\) 的阈值表示把参考协方差、均值和平方项变成单阈值及双阈值经验过程。它们的包络分别具有 Gaussian 矩或一致有界性，括号数对 \(\ell\) 一致，因而中心化偏差均为 \(O_p(N^{-1/2})\)。样本标准差三角不等式给 \(\inf_h\widehat{\mathrm{sd}}(h)\ge\widehat{\mathrm{sd}}(X)-\ell/2\)，在 \(\ell\to0\) 时统一远离零。将分母展开，并用上一轮的窗口一致偏差，即得式 (2)。使用共同参考使恒等分母的公共误差在候选位移中消去，余下项带有 \(\ell\)。

因此，若 \(g_T/s_{T,N}\to b\) 且 \(\ell^2/s_{T,N}\to\lambda\in[0,\infty]\)，估计归一化下的完整相容类分歧概率满足
\[
\Pr(\widehat{\mathrm{ambiguity}})\longrightarrow
\begin{cases}
\Phi(c\lambda-b)-\Phi(-b),&\lambda<\infty,\\
\Phi(b),&\lambda=\infty.
\end{cases} \tag{3}
\]
式 (2) 除以 \(s_{T,N}\) 后为 \(O_p(\ell)\)，所以两个区间端点可替换成 \(0,c\lambda\)，式 (1) 再给出上述概率。它量化的是采用样本分母后的候选一致性。

## 与总体归一化目标的错误

即使第一轴取恒等坐标，参考误差也会改变逐窗口答案。在总体并列且 \(N/T\to r\in(0,\infty)\) 时，真实窗口间隔和估计窗口间隔的正态极限相关系数为
\[
\zeta_r=\frac{\sigma_w}{\sqrt{\sigma_w^2+4d^2/r}}.
\]
二维中心正态的象限概率由旋转对称性直接积分得到，因而
\[
\Pr\{\operatorname{sign}(G_T)\ne\operatorname{sign}(\widehat G_{T,N})\}
\longrightarrow\frac{\arccos(\zeta_r)}{\pi}. \tag{4}
\]
当 \(N/T\to0\) 时该极限为 \(1/2\)，当 \(N/T\to\infty\) 时为零。有限参考归一化的候选一致性与总体参考下的物理慢集可靠性需要分别计算。共享同一参考的多条评价窗口保留共同的分母误差，重抽样单位应包括整份参考和它对应的评价面板。

若参考点本身来自一条独立的标准 Gaussian AR 轨迹，样本方差的极限方差由 2 改为 \(2(1+\rho^2)/(1-\rho^2)\)。两轴参考方差差的系数相应乘 \((1+\rho^2)/(1-\rho^2)\)。这来自 \(\operatorname{Cov}(X_0^2,X_k^2)=2\rho^{2|k|}\) 的求和。下面的 DKW 界采用 iid 参考点。

## 未知参考律下的同时方差界

现在只向判定规则提供参考样本、已知共同周期和参考尾概率包络 \(H(t)\ge\Pr(|X|>t)\)。总体参考分布允许未知。对第一轴的每个相容 \(h\) 仍计算其总体参考方差定义的目标。固定截断半径 \(R\)，令 \(Z=\operatorname{clip}(X,-R,R)\)，\(\widehat V_R(h)\) 为 \(h(Z_i)\) 的样本方差。

在 Dvoretzky–Kiefer–Wolfowitz 事件
\[
\|F_N-F\|_\infty\le\varepsilon_N,
\qquad\varepsilon_N=\sqrt{\frac{\log(2/\alpha)}{2N}},
\]
上，以下界对全部连续单调 \(h\) 同时成立。由于 \(h(R)-h(-R)\le A=2R+\ell\)，有
\[
|V_R(h)-\widehat V_R(h)|\le A^2\varepsilon_N. \tag{5}
\]
利用 \(\operatorname{Var}(Y)=\tfrac12 E(Y-Y')^2\) 展开总体乘积测度与经验乘积测度之差。对固定 \(y\)，函数 \((h(x)-h(y))^2\) 的总变差为 \((h(y)-h(-R))^2+(h(R)-h(y))^2\le A^2\)。对两次积分各用 CDF 误差乘总变差，前面的 \(1/2\) 抵消两项。这个证明同时覆盖数据依赖的候选 \(h\)。 [Massart 的 DKW 常数](https://doi.org/10.1214/aop/1176990746)

还可用样本可见的范围收紧式 (5)。若 \(U\ge\sup_h\widehat V_R(h)\)，\(t\ge\sup_h|\widehat E h(Z)-(h(-R)+h(R))/2|\)，则 \(\varepsilon_N<1\) 时可使用
\[
\delta_N=\min\left\{A^2\varepsilon_N,
\frac{\varepsilon_N}{1-\varepsilon_N}
\left(2U+2t^2+2A\varepsilon_Nt+A^2\varepsilon_N^2+\frac{A^2}{2}\right)\right\}. \tag{6}
\]
同一乘积测度证明先给出误差 \(\delta\le\varepsilon_N\{V_R+\widehat V_R+(E h-m)^2+(\widehat E h-m)^2+A^2/2\}\)，其中 \(m\) 为端点值中点。再用 \(V_R\le\widehat V_R+\delta\) 和 \(|E h-\widehat E h|\le A\varepsilon_N\)，移项后即得式 (6)。

\(U,t\) 可由可见参考直接计算。记 \(v=\widehat{\operatorname{Var}}(Z)\)，\(\bar z=\widehat E Z\)，\(u_i=\{Z_i/\ell\}\)。样本协方差的全类极值为
\[
e_-\le\widehat{\mathrm{Cov}}(Z,p(u))\le e_+,\qquad
e_\pm=\underset{s\in[0,1]}{\operatorname{ext}}\left\{
N^{-1}\sum_i(Z_i-\bar z)1_{\{u_i\ge s\}}
-N^{-1}\sum_i(Z_i-\bar z)u_i\right\}.
\]
按相位排序的加权尾和可精确计算这两个极值。于是
\[
\underline V_{\rm emp}=\max(0,v+2\ell e_-),\quad
\overline V_{\rm emp}=v+2\ell e_++\ell^2/4,
\quad U=\overline V_{\rm emp},\quad t=|\bar z|+\ell
\]
是全部候选共同有效的界。其余直接可见轴使用 \(\ell=0\)。

截断外的误差满足
\[
\|h(X)-h(Z)\|_2\le\tau_R,
\qquad
\tau_R^2=\ell^2 H(R)+2\ell\int_R^\infty H(t)\,dt
+2\int_R^\infty(t-R)H(t)\,dt.
\]
标准差的三角不等式把式 (6) 延伸为
\[
\operatorname{sd}(h(X))\in
\left[
\left(\sqrt{(\widehat V_R(h)-\delta_N)_+}-\tau_R\right)_+,
\sqrt{\widehat V_R(h)+\delta_N}+\tau_R
\right]. \tag{7}
\]
共同界可将 \(\widehat V_R(h)\) 分别换成 \(\underline V_{\rm emp},\overline V_{\rm emp}\)。若外部条件提供正态尾包络 \(H(t)=2\Phi(-t)\)，则
\[
\tau_R^2=2\{(2\ell-R)\varphi(R)+[1+(R-\ell)^2]\Phi(-R)\}.
\]
这个尾包络允许一族未知参考分布。若只知道 \(H(t)=2e^{-t^2/2}\)，则相应上界为
\[
\tau_R^2=(4+2\ell^2)e^{-R^2/2}
+4(\ell-R)\sqrt{2\pi}\,\Phi(-R).
\]

## 从有限参考界到可观测作答

评价窗口的增量分子可继续沿用相位尾和的线性极值 \(C_-,C_+\)。二次项还有一个直接可算的全类上界。记 \(u,v\) 为一个时距的两端相位，\(p_s(u)=1_{\{u\ge s\}}-u\)。由 Jensen，
\[
R_w(p)\le\sup_s R_w(p_s).
\]
对每个增量，
\[
(p_s(v)-p_s(u))^2=(v-u)^2+
(1-2|v-u|)1_{\{\min(u,v)<s\le\max(u,v)\}}.
\]
把所有时距按原权重相加，按相位端点扫描即可得到 \(R_{\max}\)。因此
\[
Q_w(h)\in
\left[\max(0,d_w+2\ell C_-),\ d_w+2\ell C_++\ell^2R_{\max}\right]. \tag{8}
\]
用分子下端除以式 (7) 的方差上端，用分子上端除以方差下端，得到总体归一化的慢度区间。方差下端为零时允许上端无穷。对每个命名轴得到区间后，只有某个二元选集的最高上界严格低于所有未选轴的最低下界时才作答。

四轴各用失效概率 \(\alpha/4\)，联合失效概率至多 \(\alpha\)。同一参考事件对任意多条评价窗口同时成立。若校准置信集合以概率至少 \(1-\alpha_{\rm cal}\) 包含真实坐标，与上述参考约束相交后，无条件错误作答的上界为 \(\alpha+\alpha_{\rm cal}\)。这个论断允许校准和参考共用随机量，因为使用的是并集界。置信集合求解和浮点容差仍需在实现中保持区间方向。

这些区间有限且可收缩。对固定 \(R\)，采样半径随 \(N\) 缩小；令 \(R\) 缓慢增大且 \(R^2\sqrt{\log(1/\alpha)/N}\to0\)，尾项和采样半径可同时趋于零。若某个窗口的分子范围与参考方差范围已经具有严格分离，有限足够大的 \(N\) 就能给出非空作答。式 (8) 本身保留相容类取界的保守性，具体预算的覆盖须用冻结实验核对。
