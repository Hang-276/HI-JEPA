# LeJEPA 正交恢复与命名慢因素的窗口排序

2026-11-12。以下结果针对独立、平稳、标准化的 Gaussian AR(1)。

## 设置

令 \(z_{t+1,j}=\rho_j z_{t,j}+\sqrt{1-\rho_j^2}\,\xi_{t+1,j}\)，\(0<\rho_j<1\)，各轴与各步创新独立，初始状态服从标准 Gaussian。编码器有 \(d\) 个输出，记 \(h(z)=f(g(z))\)。谈总体最优时，假设观测映射及函数类允许实现正交线性解。对固定编码器的界只要求相应矩条件。

记 \(\mathcal L(h)=\mathbb E\|h(z_{t+1})-h(z_t)\|^2\)。主要结果使用总体零均值和精确白化 \(\operatorname{Cov}(h)=I\)。严格标准 Gaussian 输出满足该条件。有限批次 SIGReg 不自动保证该条件。每个评价窗口 \(w\) 长于 32 步，\(N_j(w)\) 是 lag 4、16、32 的等权平方增量均值，每个 lag 内按所有有效端点对平均。物理目标 \(D_j(w)=N_j(w)/V_j\) 使用平稳物理参考方差 \(V_j=1\)。慢集是最小的两个命名轴。

## 1. 谱分离给出恢复误差界

定义一阶与高阶模态间隔 \(s=\rho_{\min}-\rho_{\max}^{2}\)。若 \(s>0\)，精确白化的总体最优值为 \(L_*=2d-2\sum_j\rho_j\)，全部最优解为 \(h(z)=Qz,\ Q\in O(d)\)。这些解也满足精确 Gaussian 约束，因此同一最优解结论适用于该更小的可行类。对任意精确白化编码器，记 \(\delta=\mathcal L(h)-L_*\)，\(h(z)=Mz+\nu(z)\)，\(M=\mathbb E[h(z)z^\top]\)，\(W=\mathbb E\|\nu(z)\|^2\)，则
\[
W\le\frac{\delta}{2s},
\qquad
\min_{Q\in O(d)}\mathbb E\|h(z)-Qz\|^2\le W+W^2 .
\tag{1}
\]

**证明。** 张量 Hermite 基 \(\{H_\alpha\}\) 的条件期望特征值为 \(\lambda_\alpha=\prod_j\rho_j^{\alpha_j}\)，由各轴独立及一维 Mehler 公式得到。精确白化使 \(h_i\) 构成无常数项的正交归一组。令 \(w_\alpha=\sum_i|\langle h_i,H_\alpha\rangle|^2\)，则 \(0\le w_\alpha\le1\)（Bessel），\(\sum_{\alpha\ne0}w_\alpha=d\)（Parseval），且 \(W=\sum_{|\alpha|\ge2}w_\alpha=\sum_j(1-w_{e_j})\)。因此
\[
\frac{\delta}{2}=\sum_j\rho_j(1-w_{e_j})-\sum_{|\alpha|\ge2}\lambda_\alpha w_\alpha\ge sW.
\]
若 \(\delta=0\)，\(W=0\)，白化给 \(MM^\top=I\)。任意 \(Qz\) 的损失均为 \(L_*\)。一般情形 \(MM^\top=I-\operatorname{Cov}(\nu)\)，故 \(M\) 的奇异值 \(\sigma_i\in[0,1]\)。取由 SVD 给出的正交极因子 \(Q\)，则 \(\|M-Q\|_F\le(\sum_i(1-\sigma_i^2)^2)^{1/2}\le W\)。因 \(\mathbb E[\nu z^\top]=0\)，恢复误差为 \(W+\|M-Q\|_F^2\)。

当 \(\rho_j=e^{-K_j\Delta t}\) 时，谱条件严格等价于 \(\max_jK_j<2\min_jK_j\)，即原文附录 F 的连续时间条件。式 (1) 是原文附录 A、C 论证的异质速率、精确白化版本。 [Klindt、LeCun、Balestriero，v2](https://arxiv.org/html/2605.26379v2)

## 2. 精确恢复时，表示坐标仍可抹平全部总体慢度差

取四轴相关系数 \((.90,.88,.85,.83)\)，有 \(s=.02>0\)。物理总体慢度为
\[
D_j^{\rm pop}=\frac{2}{3}\sum_{\ell\in\{4,16,32\}}(1-\rho_j^\ell),
\]
故物理排序严格，top-2 是轴 1、2。令 \(Q\) 为四阶 Hadamard 矩阵除以 2。所有 \(|Q_{ij}|^2=1/4\)，且总体参考与各 lag 增量二阶矩分别为 \(I\) 和对角矩阵 \(2\operatorname{diag}(1-\rho_j^\ell)\)。所以每个表示坐标的总体慢度均为四个物理分数的平均值。与此同时，\(Qz\) 是总体最优解，且线性解码 \(Q^\top h=z\) 逐点精确。有限窗口的增量交叉矩一般非零，旋转后无需逐窗并列。

## 3. 用滞后统计解混后，窗口排名有条件保证

### 3.1 解混误差

设 \(\rho_j\) 两两不同。给定 lag \(k\ge1\)，令 \(G_k=\operatorname{diag}(\rho_j^k)\)，\(\gamma_k=\min_{i\ne j}|\rho_i^k-\rho_j^k|\)。总体滞后协方差 \(C_k(h)=\mathbb E[h(z_{t+k})h(z_t)^\top]\) 对称。总体恢复精确时，它等于 \(QG_kQ^\top\)，故特征向量给出物理轴，带有符号和置换自由度。平方增量不受符号影响，轴名由已知命名速率或独立命名校准给出。

一般白化编码器满足
\[
\|C_k(h)-QG_kQ^\top\|_{\rm op}
\le(2\rho_{\max}^k+\rho_{\max}^{2k})W.
\tag{2}
\]
证明：一阶与高阶 Hermite 分量跨时间也正交，因此 \(C_k(h)=MG_kM^\top+C_k(\nu)\)。高阶谱界给 \(0\le v^\top C_k(\nu)v\le\rho_{\max}^{2k}W\) 对所有单位向量成立。再用 \(\|M\|_{\rm op}\le1\)、\(\|M-Q\|_{\rm op}\le W\) 控制线性项。总体滞后矩的扰动因此按残差能量 \(W\) 控制。

若对称估计量 \(\widehat C_k\) 满足 \(\|\widehat C_k-C_k(h)\|_{\rm op}\le\tau\)，设 \(e=(2\rho_{\max}^k+\rho_{\max}^{2k})W+\tau\)。若 \(e<\gamma_k/2\)，可按特征值顺序匹配 \(\widehat C_k\) 与 \(G_k\) 的特征向量。写 \(\widehat Q\) 为匹配及定向后的估计基，得到
\[
R:=\mathbb E\|\widehat Q^\top h(z)-z\|^2
\le W+\left(W+\frac{\sqrt{2d}\,e}{\gamma_k-e}\right)^2.
\tag{3}
\]
证明：由对称矩阵的极小极大原理，匹配特征值偏差至多 \(e\)。将估计向量的特征方程投影到对应真实向量的正交补，垂直分量范数至多 \(e/(\gamma_k-e)\)。取同半球符号，每根单位向量的平方差至多垂直分量平方的两倍。对 \(d\) 轴求和，再用 \(R=\|M-\widehat Q\|_F^2+W\)。

### 3.2 从解混误差得到可拒答的窗口比较

取与校准独立的评价轨迹。令 \(u=\widehat Q^\top h\)，\(\widehat N_j(w)\) 为其增量分子，\(\widehat V_j>0\) 为参考估计，\(\widehat D_j=\widehat N_j/\widehat V_j\)。令 \(E_j(w)\) 为坐标误差 \(u_j-z_j\) 的同口径平方增量均值。若 \(E_j(w)\le a_j\) 且 \(|\widehat V_j-1|\le b_j\)，则
\[
|\widehat D_j(w)-D_j(w)|\le B_j(w):=2\sqrt{\widehat N_j(w)a_j}+a_j+\widehat D_j(w)b_j .
\tag{4}
\]
证明：写 \(\Delta z=\Delta u-\Delta(u-z)\)，展开并用 Cauchy–Schwarz，得 \(|N_j-\widehat N_j|\le2\sqrt{\widehat N_j E_j}+E_j\)。再用 \(\widehat N_j=\widehat D_j\widehat V_j\)。若每个预测慢轴 \(i\) 与其余轴 \(j\) 都满足 \(\widehat D_j-\widehat D_i>B_i+B_j\)，则预测命名 top-2 正确，否则可以拒答。判定使用预测间隔。

这里有一个可能保守的概率版本。条件于独立校准，残差 \(r_j(z)=u_j(z)-z_j\) 已固定。正 AR 谱使 \(\mathbb E[r_j(z_t)r_j(z_{t+\ell})]\ge0\)，故 \(\mathbb E\sum_j E_j(w)\le2R\)。若 \(R\le R_{\rm up}\) 已成立，取 \(a_j=2R_{\rm up}/\alpha_w\)（全轴共用），对新窗口残差事件失败概率至多 \(\alpha_w\)。若校准界和参考界各有失败概率 \(\alpha_c,\alpha_v\)，错误作答概率至多 \(\alpha_c+\alpha_v+\alpha_w\)。这是单窗口无条件错误作答率的上界。作答后的条件错率还依赖覆盖率。

### 3.3 有限校准的一种显式做法

假设 \(h\) 总体白化，已知 \(K\) 使所有单位投影的四阶矩不超过 \(K\)（精确 Gaussian 输出可取 3）。从 \(m\) 条独立平稳校准轨迹各取一对 lag-\(k\) 端点，平均对称外积得 \(\widehat C_k\)。单对外积 Frobenius 范数平方的期望至多 \(d^2K\)，由 Cauchy–Schwarz 及对称化的范数收缩性得到。因此 \(\tau_k=d\sqrt{K/(m\alpha_k)}\) 给出算子范数误差界，失败概率至多 \(\alpha_k\)。若另外用 lag 1 的 \(\widehat C_1\) 和同类界，且物理速率由外部提供，则
\[
\delta\le2\left(\sum_j\rho_j-\operatorname{tr}\widehat C_1+d\tau_1\right).
\]
把右侧与 0 取较大值，代入 (1)–(3) 得到 \(R_{\rm up}\)。两个滞后估计使用同批轨迹时仍可用并集界。独立参考可用 \(m_v\) 对相互独立的状态，以 \((u_j-u'_j)^2/2\) 的均值估计 \(\widehat V_j\)。它的期望是 1，单对的方差至多 \((K+1)/2\)，故全轴可取 \(b_j=\sqrt{d(K+1)/(2m_v\alpha_v)}\)。若直接采用本模型的已知总体方差，令 \(\widehat V_j=1\) 即可。

显式方案依赖正确的模型、已知速率或其有效外界、命名匹配、总体白化及独立校准。它可以因 \(\gamma_k\) 太小、样本不足或窗口间隔太窄而全拒答。训练中的样本白化与总体白化有别，相邻时间对也通常相关。

## 4. 谱交错丢掉最快因素时，总体与窗口结论不同

把第 2 节的 \(\rho_4=.83\) 改为 \(.80\)。物理第 2、3 名的总体间隔保持原值。非恒定模态的前四个特征值变为 \(.90,.88,.85,.81\)，最后一项对应 \(H_2(z_1)=(z_1^2-1)/\sqrt 2\)。取 \(h=(z_1,z_2,z_3,H_2(z_1))\)，它是精确白化总体目标的一个最优解，损失比最优线性解低 \(2(.81-.80)=.02\)。证明：正交归一组的各模态权重介于 0 和 1 且总和为 4，最大相关和由四个最大特征值取得。这里的 \(H_2\) 输出边缘非 Gaussian。

总体最慢两轴都仍存在于表示，且已知速率时总体排序可以确定。对任意有限窗口，命名 top-2 仍可能包含未观察的轴 4。给定完整前三轴轨迹，设 \(T\) 为三轴窗口分数的第 2 小值，几乎必然 \(0<T<\infty\)。未观察轴独立于已观察轨迹，其得分是非退化 Gaussian 路径的非零半正定二次型，对任意 \(T>0\) 都有 \(0<\Pr(D_4<T)<1\)。因为路径密度处处为正，二次型在原点附近任意小，在某些开集中任意大，且无原子。因此条件于观察，轴 4 是否入选的 Bayes 错误率严格为正。即使知道全部总体速率，也无法在任何正概率的作答区域上提供零错误的逐窗保证。该结论容许低错误率作答，具体可达风险由遗漏轴分布和可观察窗口共同决定。

## 5. 足够强的各向异性下，精确 Gaussian 约束也允许非线性解优于所有线性解

取两个独立标准 Gaussian AR(1) 轴 \(X,Y\)，相关系数分别为 \(a,b\)。考虑向量场 \(v(x,y)=(-xy,x^2-1)\) 的流 \(\phi_t\)，其中 \(t\) 是编码变换参数，与 AR 过程时刻无关。该流是光滑的全局微分同胚，保持精确标准二元 Gaussian。具体而言，沿流 \((x^2+y^2)'=-2y\)，所以半径增加至多 \(|t|\)，从而有限时间内无爆炸，反向流也全局存在。对标准 Gaussian 密度 \(p\)，
\[
\operatorname{div}(p v)
=p\big(\operatorname{div}v-(x,y)\cdot v\big)=p(-y+y)=0 .
\]
因此 \(p(\phi_t(z))|\det D\phi_t(z)|=p(z)\)，即 \(\phi_t\) 是精确 Gaussian 约束的可行解。

令 \(T\) 为 AR 一步条件期望算子，\(z_i\) 为第 \(i\) 个物理坐标，\(r_i=a,b\)。由自伴性、\(Tz_i=r_i z_i\) 及两坐标函数各自的单位方差，得到精确恒等式
\[
\langle\phi_{t,i},T\phi_{t,i}\rangle-r_i
=\langle\phi_{t,i}-z_i,(T-r_i I)(\phi_{t,i}-z_i)\rangle .
\]
流的半径界保证 \((\phi_t(z)-z)/t\to v(z)\) 在 \(L^2\) 中成立。事实上它是向量场沿流的时间平均，被初始半径的二次多项式一致控制，而 Gaussian 具有有限四阶矩。\(v_1=-XY\) 是特征值 \(ab\) 的单位能量 Hermite 模态，\(v_2=X^2-1\) 是特征值 \(a^2\)、能量 2 的模态。因此总滞后相关相对任意正交线性解的增量为
\[
t^2\big[ab-a+2(a^2-b)\big]+o(t^2).
\]
例如 \(a=.90,b=.60\)，系数为 \(.06>0\)。于是足够小的任意非零 \(t\) 都让总体对齐损失严格低于所有线性可行解。保留其他独立轴可将构造嵌入四维。

此结论针对总体目标中允许实现该光滑变换的函数类。它给出一类严格反例，精确 Gaussian 目标的完整临界边界与有限 SIGReg 训练的收敛解仍待确定。

上述结果分别覆盖总体恢复、正交歧义、有校准时的概率判定、缺失因素时的有限窗口不可确定性，以及精确 Gaussian 约束下的强各向异性反例。H-JEPA、非 Gaussian FHN 和真实视觉观测需要另行检验结构假设。
