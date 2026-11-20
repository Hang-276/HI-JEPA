# 精确剂量下的非 Gaussian 总体与窗口歧义

沿用 [精确剂量模型](../DOSE_IDENTIFIABILITY_20261109.md) 的命名四轴和相同的物理慢度。第一观测轴 \(X_t\) 平稳，边际密度为 \(f\)，均值为零，方差为一，且在实线上有全支持。其它三轴直接可见，采用原来的独立 Gaussian AR，并独立于第一轴。第一物理轴是未知的连续严格递增双射 \(h(X_t)\)。已知物理剂量 \(\ell,2\ell\) 的总体响应均值分别是 \(x+\ell,x+2\ell\)，因此相容类在整体平移后恰是

\[
h(x)=x+\ell p(\{x/\ell\}),\qquad p(u)=F(u)-u,
\qquad F(0)=0,\quad F(1)=1,
\]

其中 \(F\) 连续严格递增。记此类周期偏移为 \(\mathcal P\)。单调性给出 \(\operatorname{osc}(p)\le1\) 和 \(|p|\le1\)。以下固定时距集 \(\mathcal H=\{4,16,32\}\)，用等权平方增量除以各解释自己的总体参考方差。结论条件于已知的精确响应和边际参考。

## 1. 总体二阶歧义的充要条件

令 \(r_j=E[X_tX_{t+j}]\)，\(\bar r=\operatorname{avg}_{j\in\mathcal H}r_j\)，\(d=2(1-\bar r)\) 为恒等解释的总体慢度。定义对称时间方向的条件均值及其残差密度

\[
m(x)=\operatorname{avg}_{j\in\mathcal H}
\frac{E[X_{t+j}\mid X_t=x]+E[X_{t-j}\mid X_t=x]}{2},
\qquad g(x)=(\bar r x-m(x))f(x).
\]

可逆过程的前向与后向条件均值相同。一般平稳过程也适用上述对称形式。\(g\in L^1\) 且 \(\int g=0\)。对 \(p\in\mathcal P\)，记 \(p_\ell(x)=p(\{x/\ell\})\)、\(e_p=E[Xp_\ell(X)]\)、\(V_p=\operatorname{Var}(p_\ell(X))\)，并令 \(K_p\) 为 \(p_\ell(X_t),p_\ell(X_{t+j})\) 协方差的时距均值。平方增量的一阶交叉项是

\[
\begin{aligned}
C(p)&=\operatorname{avg}_{j}E[(X_{t+j}-X_t)(p_\ell(X_{t+j})-p_\ell(X_t))]\\
&=2e_p-2\int m(x)f(x)p_\ell(x)\,dx .
\end{aligned}
\]

方差标准化减掉 \(d e_p\) 这一项，因此 \(C(p)-d e_p=2I_\ell(p)\)，其中 \(I_\ell(p)=\int g(x)p_\ell(x)\,dx\)。展开分子、分母得恒等式

\[
D_h-d=\frac{4\ell I_\ell(p)+2\ell^2(\bar r V_p-K_p)}{1+2\ell e_p+\ell^2 V_p}.
\tag{1}
\]

这说明控制总体一阶项的是对称条件均值偏离线性回归的部分。Gaussian AR 给出 \(m(x)=\bar r x\)，因而该项精确为零。

对任意 \(q\in L^1(\mathbb R^s)\)，记其折叠为

\[
\mathcal F_\ell q(u)=\ell^s\sum_{k\in\mathbb Z^s}q(\ell(k+u)),\qquad u\in[0,1]^s.
\]

级数在 \(L^1\) 中有定义，\(\|\mathcal F_\ell q\|_1\le\|q\|_1\)。并且 \(\mathcal F_\ell q\to\int q\) 于 \(L^1\)。证明先对紧支撑连续函数使用网格偏移一致的 Riemann 和，再利用折叠的 \(L^1\) 收缩性和紧支撑连续函数的稠密性。

令 \(G_\ell=\mathcal F_\ell g\)、\(A_\ell(v)=\int_v^1 G_\ell(u)\,du\)、\(\bar A_\ell=\int_0^1 A_\ell(v)\,dv\)。交换积分并使用 \(p=F-\mathrm{id}\) 得

\[
I_\ell(p)=\int_0^1 A_\ell(v)\,dF(v)-\bar A_\ell.
\tag{2}
\]

\(A_\ell\) 连续。用严格递增的连续分布函数将质量集中在任一阈值附近，可以任意逼近其极值。因此 \(\sup_p |I_\ell(p)|\) 等于 \(\max\{\sup_v A_\ell(v)-\bar A_\ell,\bar A_\ell-\inf_v A_\ell(v)\}\)，介于 \(\operatorname{osc}(A_\ell)/2\) 与 \(\operatorname{osc}(A_\ell)\) 之间。

由 \(V_p\le1/4\)、\(|K_p|\le1/4\) 及标准差三角不等式，式 (1) 的分母对 \(0<\ell\le1\) 有统一正下界和有限上界，二次分子有统一 \(O(\ell^2)\) 界。由式 (1)–(2) 得到**整个相容类的二阶充要条件**

\[
\sup_h |D_h-d|=O(\ell^2)
\quad\Longleftrightarrow\quad
\operatorname{osc}(A_\ell)=O(\ell).
\tag{3}
\]

只用 \(g\in L^1\) 可由折叠收敛得到 \(\operatorname{osc}(A_\ell)=o(1)\)，相应的总体保证为 \(o(\ell)\)。两者之间可以有实质差距。

## 2. 可核对的平滑条件与尖锐常数

若 \(g\in W^{1,1}(\mathbb R)\)，取 \(p-\int_0^1p\) 的周期原函数 \(P_p\)，可统一取 \(\|P_p\|_\infty\le1\)。分部积分给出

\[
I_\ell(p)=-\ell\int g'(x)P_p(x/\ell)\,dx.
\]

边界项为零，且 \(\int g'=0\)。折叠引理应用于 \(g'\) 后，\(\sup_p |I_\ell(p)|=o(\ell)\)，所以式 (1) 的总体线性分子项一致为 \(o(\ell^2)\)。

进一步假定每个评分时距的两点联合律都有密度。折叠引理应用于单点和两点密度后，分别得到均匀相位和两个独立均匀相位的 \(L^1\) 极限。由于 \(|p|\le1\)，这些极限对整个 \(\mathcal P\) 一致给出 \(V_p=\operatorname{Var}(p(U))+o(1)\) 和 \(K_p=o(1)\)，其中 \(U\) 是单位区间均匀变量。分母一致趋于 1。于是

\[
\sup_{p\in\mathcal P}\left|
\frac{D_h-d}{\ell^2}-2\bar r\operatorname{Var}(p(U))
\right|\longrightarrow0.
\tag{4}
\]

函数 \(p(u)=F(u)-u\) 是阈值函数 \(1\{u\ge v\}-u\) 对 \(dF(v)\) 的混合。每个阈值函数作用于 \(U\) 都得到方差为 \(1/12\) 的平移均匀变量。先分别中心化，再用 Jensen，可知 \(\sup_{p\in\mathcal P}\operatorname{Var}(p(U))=1/12\)，上确界由连续严格递增分布逼近阈值得到。因此 \(\sup_h |D_h-d|/\ell^2\to|\bar r|/6\)。在 \(\bar r>0\) 时，最坏的正向差也达到 \(\bar r/6\)，而负向差至多为 \(o(\ell^2)\)。这些渐近结论允许有限 \(\ell\) 的交叉项非零。若 \(\bar r=0\)，式 (4) 只给更小阶差异。

### 非线性条件均值的可逆正例

取标准 Gaussian AR(1) \(Z_t\)，自回归系数 \(0<\rho<1\)。令 \(X_t=T(Z_t)=(Z_t+aZ_t^3)/s\)，其中 \(a>0\) 且 \(s^2=1+6a+15a^2\)。\(T\) 是严格递增光滑双射，继承可逆性和各阶有限路径密度。记 \(H_3(z)=z^3-3z\)，则

\[
\begin{aligned}
E[X_{t+j}\mid X_t=T(z)]
&=\{(1+3a)\rho^j z+a\rho^{3j}H_3(z)\}/s,\\
r_j&=\{(1+3a)^2\rho^j+6a^2\rho^{3j}\}/s^2.
\end{aligned}
\]

Hermite 的两个衰减因子不同，故前向条件均值对 \(X_t\) 非线性。此时 \(g(T(z))\) 是多项式乘 Gaussian 密度再除以 \(T'(z)\)，其对 \(z\) 的导数可积，因为 \(T'(z)\ge1/s>0\)。变量替换给出 \(g\in W^{1,1}\)。因此保留尖锐总体常数 \(\bar r/6\)，并且下节的窗口结论成立。固定正弦候选 \(p(u)=\kappa\sin(2\pi u)/(2\pi)\) 对应总体常数 \(\bar r\kappa^2/(4\pi^2)\)。

## 3. 有限窗口的一阶项与慢集分歧

固定 \(T>32\) 个状态的第一轴窗口 \(w\)，令 \(d_w\) 为其恒等分数，\(b(w)=\operatorname{avg}_j D_j^\top D_j w/(T-j)\)，其中 \(D_j\) 为时距差分矩阵。记 \(u_t=\{w_t/\ell\}\)。窗口线性项包括参考方差的一阶修正

\[
L_w(p)=\sum_t b_t(w)p(u_t)-d_w e_p.
\]

同一个分数展开给出 \(D_h(w)=d_w+2\ell L_w(p)+O(\ell^2)(1+d_w)\)，余项对全部 \(p\) 和窗口一致。具体地，当 \(0<\ell\le1\)，归一化分母至少为 \(1/4\)，与 1 的差至多为 \(5\ell/4\)。窗口内 \(p_\ell\) 的增量平方均值至多为 1，且 \(|L_w(p)|\le\sqrt{d_w}+d_w/2\le1+d_w\)。代入可用 \(14\ell^2(1+d_w)\) 作为保守余项。若已知参考密度，线性项的全类极值仍能通过阈值求得。令

\[
A_w(v)=\sum_t b_t(w)1\{u_t\ge v\}
-d_w\int x f(x)1\{\{x/\ell\}\ge v\}\,dx.
\]

与式 (2) 相同，\(L_w(p)=\int A_w\,dF-\int_0^1 A_w(v)\,dv\)。极值取各相位间阈值函数的本质上下确界，可用集中的连续严格递增分布函数逼近。对一般边际，参考积分随阈值变化，单独排序观测相位还不够。若 \(xf(x)\in W^{1,1}\)，可用周期原函数界给出 \(\sup_p|e_p|=O(\ell)\)，参考线性项随即并入二阶余项。三次变换正例满足这一条件。

设整个第一轴有限路径有密度。对任何有界连续 \(a(w)\) 与非零整数向量 \(n\)，Riemann–Lebesgue 引理给出 \(E[a(w)e^{2\pi i n^\top w/\ell}]\to0\)。故 \((w,(u_t)_t)\) 联合收敛到 \((w,(U_t)_t)\)，其中相位逐点独立均匀，且独立于 \(w\)。折叠引理应用于 \(xf(x)\) 已足以保证 \(\sup_p |e_p|=o(1)\)，无需其导数条件。对固定 \(0<\kappa<1\) 的正弦候选，得到

\[
\frac{D_h(w)-d_w}{\ell}\ \Rightarrow\ Z_w=\frac{\kappa}{\pi}\sum_t b_t(w)\sin(2\pi U_t).
\tag{5}
\]

由于差分算子非零且路径有密度，\(b(w)\ne0\) 几乎处处，故 \(Z_w\) 条件于窗口为非退化对称变量。若边际有有限四阶矩，二次包络 \(C(1+d_w^2)\) 可积，从而式 (5) 的二阶矩也收敛，极限为 \(\kappa^2 E\|b(w)\|^2/(2\pi^2)\)，窗口差的 RMS 为 \(\Theta(\ell)\)。

令 \(B\) 为独立其余三轴的第二小分数。Gaussian AR 分数是至少有三个正特征值的高斯二次型，其密度在正半轴连续、严格正且有界。前两个加权平方项的卷积有界，第三项及其余项保留此界。顺序统计量 \(B\) 同样有这样的密度 \(f_B\)。第一轴进入慢集当且仅当它的分数小于 \(B\)。条件于第一轴窗口，两个固定物理解释的分歧率是 \(|F_B(D_h(w))-F_B(d_w)|\)。用 \(f_B\) 连续有界、式 (5) 与可积的一阶包络，得

\[
\ell^{-1}\Pr\{\text{恒等与固定正弦解释的慢集不同}\}
\longrightarrow E[f_B(d_w)|Z_w|]\in(0,\infty).
\]

对完整相容类，精确分数上下端的距离为 \(2\ell\operatorname{osc}(A_w)+O(\ell^2)(1+d_w)\)。参考积分在相位极限下统一为 \(o(1)d_w\)。尾和的极差在相位无并列处连续，且由 \(2\sqrt{d_w}\) 控制。于是完整类分歧率除以 \(\ell\) 的极限为

\[
2E\left[f_B(d_w)\operatorname{range}_{v\in[0,1]}
\sum_t b_t(w)1\{U_t\ge v\}\right]\in(0,\infty).
\]

用线性极值与解析二阶余项构造的外区间，和精确全类分数区间的各端点至多相差 \(O(\ell^2)(1+d_w)\)。在 \(B\) 位于外区间同一侧时才给出共同慢集。相较于知晓精确全类区间的统一零错答规则，新增拒答概率至多为 \(O(\ell^2)\)，因为条件于窗口的 \(B\) 密度有界。最优零错答的含义是对整个类中每个解释均无错。选取连续周期函数的一个可数稠密子类即可合并各解释的零测例外，分数在一致范数下连续，严格异号因此能由此子类见证。区间端点并列由 \(B\) 的独立连续密度排除。

## 4. 可逆性和光滑边际的反例

取 \(0<a_0<\sqrt{12}\)。令 \(f\) 为 \([-a_0/2,a_0/2]\) 均匀变量与独立 \(N(0,\sigma_0^2)\) 的和的密度，其中 \(\sigma_0^2=1-a_0^2/12>0\)。它光滑、严格正、对称且方差为一。记 \(t_k=2\pi k/a_0\)，其特征函数

\[
\widehat f(t)=e^{-\sigma_0^2 t^2/2}\frac{\sin(a_0t/2)}{a_0t/2}
\]

在非零整数频率 \(t_k\) 为零，并满足 \(E[X\sin(t_kX)]=-\widehat f'(t_k)=(-1)^{k+1}e^{-\sigma_0^2t_k^2/2}/t_k\)。取奇数 \(k_n=2^{n^2}+1\) 和 \(c_n=2^{-n-2}\)，定义有界连续奇函数

\[
v(x)=\tfrac14\sin(t_1x)+\sum_{n\ge1}c_n\sin(t_{k_n}x).
\]

系数绝对可和，\(\|v\|_\infty\le1/2\)。因此 \(E[v(X)]=0\)，\(b=E[Xv(X)]>0\)。在边际 \(f\) 上构造平稳转移核

\[
P(x,dy)=f(y)[1+\theta v(x)v(y)]\,dy,\qquad 0<\theta<1.
\]

该核严格正且可逆，任何有限路径都有连续密度。令 \(q=E[v(X)^2]>0\)。对任意 \(j\ge1\)，\(P^j\) 的同形式系数为 \(\theta_j=\theta^j q^{j-1}\)。故 \(E[X_{t+j}\mid X_t=x]=\theta_j b v(x)\)，\(r_j=\theta_j b^2\)。写 \(\bar\theta=\operatorname{avg}_j\theta_j\)，得到 \(g(x)=\bar\theta b[b x-v(x)] f(x)\)。

令 \(\ell_n=a_0/k_n\)，并固定 \(p(u)=-\kappa\sin(2\pi u)/(2\pi)\)，\(0<\kappa<1\)。特征函数在整数频率为零给出正弦族的精确正交性，\(E[v(X)\sin(t_{k_n}X)]=c_n/2\)。相应的 \(E[X\sin(t_{k_n}X)]\) 指数小。因而

\[
I_{\ell_n}(p)=\frac{\bar\theta b\kappa}{4\pi}\,c_n+o(c_n),\qquad
D_{h_n}-d=\frac{\bar\theta b\kappa}{\pi}\,\ell_n c_n(1+o(1)).
\]

最后一步用 \(c_n/\ell_n\to\infty\) 排除了二阶余项改变主项。候选 \(h_n=x+\ell_n p(\{x/\ell_n\})\) 的导数统一在 \([1-\kappa,1+\kappa]\)，与恒等解释共享精确剂量和被动观测律。因此即使观测过程可逆、边际光滑满支持、所有有限路径有密度、坐标统一双 Lipschitz，总体差也可沿子序列严格大于 \(\ell^2\)。反例的条件均值残差含稀疏高频，转移密度只需连续。窗口一阶结论仍满足其各项条件。

## 5. 适用范围

总体充要条件是对指定评分、已知的精确剂量观测等价类和第一轴未知物理坐标的结论。尖锐常数额外用到残差密度的可积导数与两点联合密度。窗口排序率用固定窗口长度、有限路径密度及独立的 Gaussian 竞争轴，窗口 RMS 再要求四阶矩。对有限带噪响应、多个未知或相关竞争轴，以及实际 Fine 表示，这些结构与条件需要独立建立。 [原 Gaussian 窗口理论](../dose_window_20261109/THEORY.md) · [原尖锐常数](../dose_window_20261109/SHARP_CONSTANTS.md)
