# Post-Identifiability Finite-Trajectory Slow-Factor Selection

## 1. 定义、权威来源与代码算子对齐

日期：2026-11-16。本文是低维慢因素选取的数学形式化与只读解析核对，不是新增训练、评测或新算法。目标是提供可迁入正文/附录的明确陈述与证明；不据此宣称达到某会议的录用、新颖性或形式验证标准。

**首要纠正：同窗目标与总体目标不同。** 封存代码的错误是预测集合与**同一窗口物理分数的集合**不一致；物理过程的总体排名不是评分标签。本文同时定义两个目标，但所有结果均标明对象。完美读出、相同归一化时同窗任务错误恒为零；有限轨迹可以偏离总体排名，这不构成模型同窗错误的下界。用户原要求中“完美恢复下仍有内生错选”“变慢必然使SNR恶化”“任意大lag平均一定变差”均须据此或据下文反例修正，不能作为既定结论证明。

权威顺序为封存代码/配置/回执→实验协议→SUMMARY/RESULTS。`src/slowrepr/metrics.py` 提供通用双REF版本；低维A1/A3/A4/B的实际run/score存在**真实分母固定为1**的专门版本，不能把通用版本反投射到每个旧实验。行号为本轮只读文件版本；源SHA256与完整性结果见 `results/theory_checks_20261116/`。

### 1.1 符号表与完整定义

| 符号 | 定义 |
|---|---|
| \(d,k,T\) | 因素数4，选取数2，状态数193；\(t=0,\ldots,T-1\) |
| \(\mathcal L,m\) | 评价lag集合\(\{4,16,32\}\)，\(m=3\)；不是校准解混lag |
| \(\Delta_\ell x_t\) | \(x_{t+\ell}-x_t\)，端点数严格为\(T-\ell\) |
| \(\langle a,b\rangle_\Delta\) | \(m^{-1}\sum_{\ell\in\mathcal L}(T-\ell)^{-1}\sum_{t=0}^{T-\ell-1}\Delta_\ell a_t\Delta_\ell b_t\) |
| \(\widehat S_i(x)\) | \(\langle x_i,x_i\rangle_\Delta=x_i^TMx_i\)，没有窗口去均值或窗口方差归一化 |
| \(V_i,V_i^z\) | 预测、真实REF经验方差；\(V_i=N_{ref}^{-1}\sum_n(\widehat z^{ref}_{ni}-\overline{\widehat z}^{ref}_i)^2\)，`ddof=0` |
| \(\widetilde S_i, D_i\) | 预测\(\widehat S_i(\widehat z)/V_i\)；通用真实\(D_i=\widehat S_i(z)/V_i^z\)；本轮主要低维实验真实\(D_i=\widehat S_i(z)\) |
| \(\operatorname{Top}_k(s)\) | 将\((s_i,i)\)按字典序排序，取前k个索引的集合；因此并列按0基轴索引稳定打破 |
| \(K_W,\widehat K_W,K_{pop}\) | 分别为\(\operatorname{Top}_k(D),\operatorname{Top}_k(\widetilde S),\operatorname{Top}_k(\mu)\) |
| \(\widehat\Delta\) | 预测排序后第\(k+1\)与第k分数之差，始终非负；\(\Delta_W=D_{(k+1)}-D_{(k)}\)是真实同窗gap |
| \(\gamma_i(h),\mu_i\) | 单位方差平稳源的自协方差；\(\mu_i=2m^{-1}\sum_h(1-\gamma_i(h))\)；独立AR时\(\gamma_i(h)=\rho_i^{|h|}\) |
| \(\Delta\mu,\sigma_{diff},\mathrm{SNR}\) | 名义边界物理轴2、3（代码索引1、2）的总体分差、同窗有符号差标准差及二者之比；不是排序后的随机gap |
| \(r_i,c_i,v_i,a_i,\eta_i\) | \(\widehat z_i-z_i\)，\(\operatorname{Cov}(z_i,r_i)\)，\(\operatorname{Var}(r_i)\)，\(2c_i+v_i\)，\(V_i-(1+a_i)\) |
| \(R_i^2\) | 总体版本\(1-E r_i^2/\operatorname{Var}(z_i)\)；审计代码使用独立状态经验MSE除以真实状态`ddof=0`方差；不是相关系数平方 |
| \(N_{cal}\) | B2为1024条独立长度193轨迹；A4与B1/B3阈值CAL为每组合2048条轨迹；不能按帧独立计数 |
| \(\alpha_{dyn},\alpha_{FW}\) | 动力学指数与置信错误率；局部定理若简写\(\alpha\)，由所在小节明确区分 |

本文以单位物理方差表述；非单位情况先对物理坐标与残差同时标准化。REF与TEST独立，但同一REF被许多窗口共用。冻结REF后的统计保证条件于它，不会因TEST窗口增多自动消除分母偏差。任何分母为0的坐标均不在评分定义域中；代码没有为此偷偷添加小常数。

### 1.2 逐项代码对齐表

以下相对路径均以本仓库根目录为基准。

| 项目 | 文件与行号 | 本文采用的实际定义 |
|---|---|---|
| 增量与等权 | `src/slowrepr/metrics.py:39–49`；`reports/lejepa_slow_rank_20261112/training/run.py:165–170` | 每lag先除\(T-h\)，再除lag数；长lag不因端点少而被降权 |
| 集合与gap | `src/slowrepr/metrics.py:14–17,50–58` | stable argsort；同窗真实集合为标签；第二和第三名之差 |
| 通用REF检查 | `src/slowrepr/metrics.py:7–11,32–33` | 四维有限正数；不clamp |
| 原训练R² | `reports/lejepa_slow_rank_20261112/training/run.py:158–162` | 逐轴MSE/真实经验方差；偏置误差也计入MSE |
| 原AR初始化 | 同上 `95–102` | 第一帧已为\(N(0,I)\)，不是从零启动；创新流按独立窗口生成 |
| 旧训练REF版本 | 同上 `165–170` | `.var(axis=(0,1))`，`ddof=0`；方差≤\(10^{-12}\)报错 |
| A4 REF与真实分母 | `reports/lejepa_a4_20261115/forecast.py:75–88`；`score.py:125–140` | \(N_{ref}=32768\)独立状态，仅预测除REF方差，物理分母是已知1 |
| A4校准与作答 | `forecast.py:57–72,135–143`；`score.py:116–117,137–140` | 各有向跨界pair的非负90%高估量；TEST内按得分选409/2048，不是冻结阈值规则 |
| A3阈值 | `reports/lejepa_a3_gap_20261115/forecast.py:65–73` | 汇集CAL预测分差，`np.quantile(...,method="higher")`；g50=.17748381448816608 |
| CP与Bonferroni | `src/slowrepr/metrics.py:65–94`；`reports/lejepa_a3_gap_20261115/score.py:97–121,145–146,159–176` | \(gap\ge threshold\)，单侧CP；A3为48项；观察覆盖门与风险证书不同 |
| A1 REF拆分 | `reports/lejepa_a1_joint_20261115/reference/run.py:41–57,84–101,111–126` | 同一REF的\(\widehat{Var}(z)+2\widehat{Cov}(z,r)+\widehat{Var}(r)\)精确恒等式；总体拆分另用两份131072状态估计 |
| B1/B3生成 | `reports/lejepa_b_transfer_20261115/run.py:46–60,63–75` | \(A=diag(\rho^{\alpha_{dyn}})\)，`A[1,2]=c`，\(Q=I-AA^T\)；代码Cholesky要求正定 |
| B1/B3冻结REF/阈值 | 同上 `104–126,136–151,193–211` | 各过程共用原A4 REF；新CAL取`higher`中位数；真实仍用1；128项校正 |
| B2中心与白化 | `reports/lejepa_b2_lag_20261115/run.py:45–73` | 全帧共用中心、每lag端点中心修正、对称协方差；最小特征值≤\(10^{-8}\)报错，无ridge |
| B2 multi与命名 | 同上 `71–85,89–119` | \(\{1,4,16,32\}\)四白化矩阵平均后一次eigh；NAME匹配及斜率缩放 |
| B2诊断间隔 | 同上 `101–106,114–119` | 两个边界轴分别到所有其他轴的最小间隔；bootstrap q90不是总体误差证书 |
| 最新解释限制 | `docs/RESEARCH_RECORD.md:563–579`；`docs/audit/20261116/PROGRESS_BEFORE_REORGANIZATION.md:157–171` | 自然R²未匹配、B3匹配对象与统计单位、A4/B2算法区别均保留 |

所有概率命题分别列明高斯、独立轨迹、残差、分母下界等假设；这些假设不因状态R²高而自动成立。LeJEPA可辨识允许的正交变换、理想总体最优性与有限训练读出误差是不同对象。本文不重新证明或扩展原作者的正交可辨识定理。

## 2. 正文级核心结论

完整证明与额外边界见第3节；本节中的\(D_i\)采用低维实验真实分母1。

**Proposition 1（精确有限窗口/REF分解）。** 令\(B_i=\widehat S_i(z)\)、\(K_i=2\langle z_i,r_i\rangle_\Delta\)、\(E_i=\widehat S_i(r)\)、\(V_i=1+a_i+\eta_i>0\)。逐窗口、无概率近似地，
\[
\widetilde S_i-\mu_i=(B_i-\mu_i)
 +\frac{K_i+E_i}{V_i}-\frac{B_i(2c_i+v_i+\eta_i)}{V_i}.
\]
同窗预测误差精确为\(\widetilde S_i-D_i=[K_i+E_i-B_i(2c_i+v_i+\eta_i)]/V_i\)，**没有独立添加的\(B_i-\mu_i\)**。动态交叉项依赖时序互协方差，不能由状态MSE确定；分母总体项不一定“膨胀”，因为\(c_i\)可负。第3节给出有限REF四阶矩界及精确二阶余项。

**Theorem 1（高R²不保证同窗选择；总体排名另论）。** (A) 完美匹配评分的同窗错误为0。若改问窗口是否偏离总体边界排序，则独立高斯AR二次型给精确矩与非渐近下界；\(T=193,\rho_i=.906,\rho_j=.904\)时该倒序概率至少.396369，而这不是A4的4.36%模型错率。(B) 对任意\(\varepsilon\in(0,1)\)可构造相同\(R^2=1-\varepsilon^2\)的平稳残差，归一化后一个保序，一个在明确gap条件下总体翻序并在长窗口以高概率翻序。一般时序相关残差可有\(O(\varepsilon)\)交叉影响；独立轴瞬时泄漏的总体偏差为\(O(\varepsilon^2)\)，有限窗交叉仍可为\(O(\varepsilon)\)。

**Lemma 1 / Theorem 2（精确矩与集中）。** \(M=m^{-1}\sum_\ell D_\ell^TD_\ell/(T-\ell)\)。标准高斯AR协方差\(\Sigma_i\)下，\(B_i^*=M^{1/2}\Sigma_iM^{1/2}\)，有
\[
E\widehat S_i=\mu_i,\qquad Var(\widehat S_i)=2\|B_i^*\|_F^2,
\quad P\{|\widehat S_i-\mu_i|>2\|B_i^*\|_F\sqrt x+2\|B_i^*\|_{op}x\}\le2e^{-x}.
\]
第3节给出显式lag/\(1-\rho\)范数上界、多轴充分分离条件和残差合并。\(T(1-\rho)\)进入保守有效样本量界，但不是所有固定lag增量方差的普遍渐近等价。

**Proposition 2（选择性风险证书）。** 冻结模型、REF、CAL及阈值，条件iid窗口内的作答错误数在给定作答数下服从二项分布。\(J\)项使用单侧CP上界\(U_{\alpha_{FW}/J}\)，所有风险上界同时覆盖真值的概率至少\(1-\alpha_{FW}\)。任意真实评分目标\(s\)和误差\(e=\widehat s-s\)还满足
\[
\{\widehat K\ne K,\widehat\Delta\ge\tau\}\subseteq\{2\|e\|_\infty\ge\tau\}.
\]
仅有可交换性不足；跨模型共用轨迹不能作为额外独立样本，观察覆盖≥35%也不是总体覆盖置信保证。

**Theorem 3（时间伸缩的严格极限与非单调边界）。** 令\(\lambda_i=-\log\rho_i>0\)。\(\mu_i(\alpha_{dyn})=2m^{-1}\sum_\ell(1-e^{-\alpha_{dyn}\lambda_i\ell})\)，边界gap在\(\alpha_{dyn}\to0\)为\(2\alpha_{dyn}(\lambda_j-\lambda_i)m^{-1}\sum\ell+O(\alpha_{dyn}^2)\)，在\(\alpha_{dyn}\to\infty\)也趋0，故不全局单调。固定T时方差为\(\Theta(\alpha_{dyn}^2)\)，SNR趋有限正常数；任意正固定阈值的理想覆盖趋0，但不能推出SNR普遍恶化或读出风险普遍增加。第4节给出封存条件下SNR反向的明确数值。

**Theorem 4（三角耦合的精确影响）。** 对\(A=diag(\rho)+c e_2e_3^T\)，\(|\rho_i|<1\)时\(Q\succeq0\)当且仅当\(c^2\le(1-\rho_2^2)(1-\rho_3^2)\)；代码需要严格小于。所有单轴整条轨迹的高斯分布、均值和分数方差严格不变。跨轴分数协方差为\(2c^2\|M^{1/2}GM^{1/2}\|_F^2\)，其中\(G\)是(T4.2)前定义的单位耦合跨时协方差矩阵，故名义边界差方差降低精确的\(O(c^2)\)项；这不直接保证每个模型沿用阈值通过。

**Theorem 5（校准lag谱扰动）。** 在明确线性高斯理想嵌入、有限独立CAL轨迹以及受控残差/白化误差条件下，\(\sin\angle(\widehat v_i,v_i)\le\min\{1,2E_\ell/\delta_i(\ell)\}\)。其中有限样本主项依赖\([N_{cal}(T-\ell)(1-\rho_{max})]^{-1/2}\)，并显式保留非线性残差和白化偏差。对\(\rho_i>\rho_j>0\)，令\(a=-\log\rho_i,b=-\log\rho_j\)，单个轴对gap峰值为\(\ell_*={\log(b/a)}/{(b-a)}\)。等权平均的误差上界是各误差界的平均；只有附加同向噪声或协方差条件才能推出实际角误差恶化。B2经验比值不是紧充要排名准则。

**Proposition 3（离散排序的边界测度与校准不变性）。** 修正\(s'_i=s_i-\delta_i\)改变集合，当且仅当至少一个原跨界对的修正超过它自身的pair gap（并列依stable规则）。对于\(G=s_{(k+1)}-s_{(k)}\)，\(v=E\|\delta\|_\infty^2\)，任意\(a>0\)，
\[
|Acc(s')-Acc(s)|\le P(K(s')\ne K(s))\le P(G\le2a)+v/a^2.
\]
小修正要推出小变化还需反集中/边界测度条件。净准确率小也可能来自修复与损害抵消。严格递增变换保持同一批分数的rank选择，以及`higher`次序统计量阈值的作答集合；A4实际pair依赖修正并非一个全局gap变换，因此其6339/6544重合是经验结果。

## 3. 完整证明、显式条件与实验对齐备注

证明按依赖关系组织；各小节的\(M\)始终是固定评价lag算子。为避免混淆，B2中的\(\mathcal M_{cal}\)仅表示校准lag集合，不能代入评价lag集合替换。


### 3.1 Proposition 1：精确有限窗口与有限 REF 分解

本节局部记号 $\widehat V_i$ 对应第1节的预测参考方差 $V_i$；局部 $D_i=\widehat S_i(z)$ 是未归一化物理分子（第2节的 $B_i$），局部 $s_i$ 才对应第1节的真实归一化分数 $D_i$。时间索引仍为 $0,\ldots,T-1$。输出状态暂记为 $U_i$，以免与参考方差混淆。

令 $d=4,\ k=2,\ T=193,\ \mathcal L=\{4,16,32\}$。为区分有限窗口量和总体均值，定义

\[
\langle u,v\rangle_{T,\mathcal L}
=\frac1{3}\sum_{\ell\in\mathcal L}\frac1{T-\ell}
\sum_{t=0}^{T-\ell-1}(u_{t+\ell}-u_t)(v_{t+\ell}-v_t),\qquad
D_i=\widehat S_i(z)=\langle z_i,z_i\rangle_{T,\mathcal L}.
\]

令 $N=N_{\rm ref}$。在独立参考状态 $X^{(1)},\ldots,X^{(N)}$ 上，对输出 $U_i=z_i+r_i$ 定义
\(\widehat V_i=N^{-1}\sum_n(U_i(X^{(n)})-\bar U_i)^2\)。通用真实参考为 $V_i^0>0$，真实窗分数 $s_i=D_i/V_i^0$，预测分数 $\widehat s_i=(D_i+K_i+E_i)/\widehat V_i$，其中
\(K_i=2\langle z_i,r_i\rangle\)，\(E_i=\langle r_i,r_i\rangle\)。低维 A1/A4/B 的真实目标采用 $V_i^0=1$；一般 `metrics.py` 不要求它等于 1。$A(s)$ 是稳定 top-$k$ 集；$g(s)=s_{(k+1)}-s_{(k)}$。

#### 命题陈述

假设同刻 $\operatorname{Var}(z_i)=1$，$z_i,r_i$ 二阶联合平稳；记
\(c_i=\operatorname{Cov}(z_{t,i},r_{t,i})\)、\(v_i=\operatorname{Var}(r_{t,i})\)、\(a_i=2c_i+v_i\)、\(\nu_i=1+a_i>0\)、\(\eta_i=\widehat V_i-\nu_i\)。固定任一 $\widehat V_i>0$，逐窗精确成立

\[
\boxed{\widehat s_i-\mu_i=(D_i-\mu_i)
+\frac{K_i+E_i}{\widehat V_i}
-\frac{D_i(2c_i+v_i)}{\widehat V_i}
-\frac{D_i\eta_i}{\widehat V_i}.} \tag{P1.1}
\]

$2c_i+v_i$ 是有符号的总体方差改变量，可能为负，不应统一称为“分母膨胀”。这恰好对应 A1 reference 同轴协方差、残差总体方差和有限 REF 三个分项。相对于实际的同窗真实分数，任意 $V_i^0>0$ 时有

\[
\boxed{\widehat s_i-s_i=\frac{K_i+E_i-s_i(\widehat V_i-V_i^0)}{\widehat V_i}.} \tag{P1.2}
\]

总体归一化与 REF 误差也可精确分开：令 $N_i=D_i+K_i+E_i$，则

\[
\widehat s_i-\mu_i=(D_i-\mu_i)
+\frac{K_i+E_i-a_iD_i}{\nu_i}
-\frac{N_i\eta_i}{\nu_i^2}
+\underbrace{\frac{N_i\eta_i^2}{\nu_i^2(\nu_i+\eta_i)}}_{R_{{\rm REF},i}}. \tag{P1.3}
\]

在 $|\eta_i|\le\theta\nu_i,\quad 0<\theta<1$ 上，
\[
0\le R_{{\rm REF},i}\le\frac{N_i\eta_i^2}{(1-\theta)\nu_i^3}.
\tag{P1.4}
\]
这里 $N_i\ge0$。当 iid REF 有有限四阶矩且 $\nu_i$ 固定正值时，$\eta_i=O_p(N_{\rm ref}^{-1/2})$，因而固定有限 $T$ 下该余项为 $O_p(N_{\rm ref}^{-1})$；常数依赖 $\nu_i^{-3}$、$\theta$、读出四阶矩和窗口分子。不能在 $\nu_i\to0$ 的模型序列上声称一致界。

若也要展开总体分母，精确地
\[
\frac{K_i+E_i-a_iD_i}{1+a_i}
=K_i+E_i-a_iD_i
-\frac{a_i(K_i+E_i-a_iD_i)}{1+a_i}. \tag{P1.5}
\]
在 $|a_i|\le\theta<1$ 上，余项绝对值至多
\(\frac{|a_i|}{1-\theta}(|K_i|+E_i+|a_i|D_i)\)。若在所述事件上 $D_i\le B$、$E_i\le b\varepsilon^2$、总体 $\mathbb E r_i^2\le\varepsilon^2$，则 $|c_i|\le\varepsilon$、$v_i\le\varepsilon^2$、$|a_i|\le2\varepsilon+\varepsilon^2$，且
\[
|K_i|\le2\sqrt{D_iE_i}\le2\sqrt{Bb}\,\varepsilon.
\tag{P1.6}
\]
因此 (P1.5) 的余项是 $O(\varepsilon^2)$，常数由以上显式式子给出；**总体 MSE 预算本身不能替代逐窗的 $E_i\le b\varepsilon^2$ 假设**。

#### 动态项的总体协方差／谱表达

设 $q_i(h)=\operatorname{Cov}(r_t,r_{t+h})$，$b_i(h)=\operatorname{Cov}(z_t,r_{t+h})$，$b_i(0)=c_i$，$\gamma_i(h)=\operatorname{Cov}(z_t,z_{t+h})$。逐端点平稳性给出
\[
\mathbb E D_i=\mu_i=\frac2{3}\sum_\ell[1-\gamma_i(\ell)],\quad
\mathbb E E_i=\frac2{3}\sum_\ell[v_i-q_i(\ell)],\quad
\mathbb E K_i=\frac2{3}\sum_\ell[2c_i-b_i(\ell)-b_i(-\ell)]. \tag{P1.7}
\]
不要求正反时间互协方差相等。存在谱密度时，约定 $\operatorname{Cov}(u_t,v_{t+h})=\int_{-\pi}^{\pi}e^{ih\omega}f_{uv}(\omega)\,d\omega/(2\pi)$，设 $w_{\mathcal L}(\omega)=3^{-1}\sum_{\ell\in\mathcal L}|e^{i\ell\omega}-1|^2$，则
\[
\mathbb E E_i=\int_{-\pi}^{\pi}w_{\mathcal L}(\omega)f_{rr,i}(\omega)\frac{d\omega}{2\pi},\quad
\mathbb E K_i=2\int_{-\pi}^{\pi}w_{\mathcal L}(\omega)\Re f_{zr,i}(\omega)\frac{d\omega}{2\pi}.
\tag{P1.8}
\]
状态 MSE 控制谱总质量，不能确定以上带频率权重的积分或其符号。

#### 完整证明

1. 每个增量满足 $(\Delta(z+r))^2=(\Delta z)^2+2\Delta z\Delta r+(\Delta r)^2$。所有端点权重非负且和为 1，因此加权后 $N_i=D_i+K_i+E_i$，并由 Cauchy–Schwarz 得 (P1.6)。
2. 代入 $\widehat V_i=1+2c_i+v_i+\eta_i$，计算 $N_i/\widehat V_i-D_i=[K_i+E_i-D_i(2c_i+v_i+\eta_i)]/\widehat V_i$；加减 $\mu_i$ 得 (P1.1)。将 $D_i=V_i^0s_i$ 代回即得 (P1.2)。
3. 对任何 $\nu>0,\ \nu+\eta>0$，直接通分核实 $\frac1{\nu+\eta}=\frac1\nu-\frac\eta{\nu^2}+\frac{\eta^2}{\nu^2(\nu+\eta)}$。乘 $N_i$，再用 $N_i/\nu-D_i=(K_i+E_i-a_iD_i)/\nu$ 得 (P1.3)。正分母下取绝对值得 (P1.4)。同理 $1/(1+a)=1-a/(1+a)$ 给出 (P1.5)。
4. 对 iid REF 变量 $Y$，设 $\mathbb E Y=m,\ \operatorname{Var}Y=\nu,\ \kappa_4=\mathbb E(Y-m)^4<\infty$。令 $X_n=Y_n-m,\ A=N^{-1}\sum_n X_n^2,\ B=N^{-1}\sum_n X_n$，则 $\widehat V=A-B^2$。独立性展开给出 $\mathbb E A=\nu,\ \mathbb E B^2=\nu/N$，
   \[
   \mathbb E A^2=\kappa_4/N+(N-1)\nu^2/N,\quad
   \mathbb E(AB^2)=\kappa_4/N^2+(N-1)\nu^2/N^2,\quad
   \mathbb E B^4=\kappa_4/N^3+3(N-1)\nu^2/N^3.
   \]
   用 $\mathbb E\widehat V^2=\mathbb E A^2-2\mathbb E(AB^2)+\mathbb E B^4$ 减 $(\mathbb E\widehat V)^2$，得到
   \[
   \mathbb E\widehat V=(1-1/N)\nu,\qquad
   \operatorname{Var}(\widehat V)=\frac{(N-1)^2}{N^3}\kappa_4
      -\frac{(N-1)(N-3)}{N^3}\nu^2. \tag{P1.9}
   \]
   故 $\mathbb E\eta^2=\operatorname{Var}(\widehat V)+\nu^2/N^2=O(N^{-1})$；Markov 给 $\eta=O_p(N^{-1/2})$。高斯读出特例为 $\operatorname{Var}(\widehat V)=2(N-1)\nu^2/N^2$，$\mathbb E\eta^2=(2N-1)\nu^2/N^2$；强 MLP 输出一般不是高斯，不能擅用高斯方差常数。
5. 将四项增量协方差逐项展开：$\mathbb E[(z_{t+\ell}-z_t)(r_{t+\ell}-r_t)]=2c-b(\ell)-b(-\ell)$，得到 (P1.7)。将谱表示代入并用 $|e^{i\ell\omega}-1|^2=2-2\cos(\ell\omega)$ 得 (P1.8)。证毕。

#### 与封存记录的精确对齐及限制

A1 reference 的零基索引为 18 的指定窗口，物理原分差为 $0.0019526511218441733$，参考分差项为 $-0.03286600828213094$；其 $2c,v,\eta$ 子项依次为 $+0.001258581808135238, -0.0031907250341906956, -0.030933865056075538$，精确加和。总体矩来自估计面板；代数闭合是精确的，分项的“总体归因”带估计不确定性。全档案跨界 RMS 的 $v$ 项 $.03891/.03855$ 大于 REF 抽样项 $.01960/.00965$；指定窗口与全档案主导来源不同。不能将相关分项 RMS 相加作贡献率。

### 3.2 Theorem 1：恢复度本身不提供普遍慢集保证

#### Part A：同窗零误差结论与总体排名偏离下界

若 $\widehat z=z$ 且真实和预测使用相同参考方差，则任意有限 $T,N_{\rm ref}$ 上 $\widehat s=s$，稳定 top-$k$ 完全相同，错误概率等于 0。若如 A4 真实分母为 1、预测分母为 iid REF 经验方差，则有限 REF 可以单独产生错误；当 $N_{\rm ref}\to\infty$，且真实窗边界无平分时，错误概率趋于 0。固定 $T$ 下独立非退化 高斯 AR 的任一轴对分数差具有连续分布，因此真实边界平分概率为 0。

另设两个独立标准平稳 高斯 AR(1) 轴，$0<\rho_j<\rho_i<1$，$D_i,D_j$ 是上述多 lag 未归一化分数。总体 $\mu_j>\mu_i$，但对有限 $T$ 有
\[
\boxed{P(D_j\le D_i)\ge\max\{0,\tfrac12-\sqrt{K_T(\rho_j\Vert\rho_i)/2}\}},\quad
K_T=\frac{T-1}{2}\left[v-1-\log v+\frac{(\rho_j-\rho_i)^2}{1-\rho_i^2}\right],\quad
v=\frac{1-\rho_j^2}{1-\rho_i^2}. \tag{T1A}
\]
这是**窗口排名与总体排名的偏离**，不是同窗预测错误。$T=193,\ \rho_i=.906,\ \rho_j=.904$ 时 $K_T=.0214787967694533$，右端为 $.3963689313732284$。A4 窄谱同窗模型错率 $739/16384=.04510498046875$ 并不矛盾，两者评价不同事件，前者不能为后者提供下界。关于 $\Delta\mu/\sigma_{\rm diff}$ 的精确矩及分布见 Lemma 1；“均值与标准差同阶”本身也不能给所有分布一个正的通用左尾下界。

##### Part A 证明

一致参考的结论直接由 (P1.2) 的 $K=E=0,\ \widehat V=V^0$ 得到。A4 极限由强大数律 $\widehat V_i\to1$，有限四轴分数逐点收敛、真实边界严格间隔以及有界收敛得到。

令 $P$ 是实际两轴路径分布，$Q$ 将 $j$ 轴参数替成 $\rho_i$，其他不变。在 $Q$ 下两分数独立同分布、连续，故事件 $\mathcal E=\{D_j\le D_i\}$ 的概率为 $1/2$。AR 路径密度等于标准正态初始密度乘 $T-1$ 个转移密度。两个过程初始分布相同；给定上一状态 $x$，转移分别为 $\mathcal N(\rho_jx,1-\rho_j^2)$ 和 $\mathcal N(\rho_ix,1-\rho_i^2)$。直接对正态对数似然比取期望，单步 KL 为
\[
\tfrac12\left[\log\frac{1-\rho_i^2}{1-\rho_j^2}+\frac{1-\rho_j^2}{1-\rho_i^2}-1
+\frac{(\rho_j-\rho_i)^2 \mathbb E_P x^2}{1-\rho_i^2}\right].
\]
平稳性给 $\mathbb E_Px^2=1$；链式求和正是 $K_T$。

为完整性，事件归并的 KL 界可直接证：设 $p=P(\mathcal E),\ q=Q(\mathcal E)$，在 $\mathcal E$ 和补集分别对条件密度比应用 Jensen，得 $D_{\rm KL}(P\Vert Q)\ge p\log(p/q)+(1-p)\log[(1-p)/(1-q)]$。在 $q=1/2$ 时，右式作为 $p$ 的函数在 $1/2$ 取值和导数都为 0，二阶导数为 $1/[p(1-p)]\ge4$，两次积分得不小于 $2(p-1/2)^2$。所以 $p\ge1/2-\sqrt{K_T/2}$。这是标准 KL/Pinsker 工具的本事件直接证明，不构成新的信息论不等式。证毕。

#### Part B：任意固定误差预算的等误差 供体构造

给定任意 $\varepsilon^2\in(0,1)$，令
\[
a^2=(1-\varepsilon^2/2)^{-2}-1>0.
\]
取四个独立标准 高斯 AR 轴，按总体分数 $\mu_0<\mu_i<\mu_j<\mu_3$ 排列，$i,j$ 为总体 top-2 边界。只替换 $j$ 轴，定义
\[
U_j^{(-)}=(Z_j+aZ_0)/\sqrt{1+a^2},\qquad
U_j^{(+)}=(Z_j+aZ_3)/\sqrt{1+a^2},
\]
其余输出等于真值，$r^{(\pm)}=U^{(\pm)}-Z$。两个残差平稳，具有相同单轴 MSE，且
\[
\mathbb E(r_j^{(\pm)})^2=2-2/\sqrt{1+a^2}=\varepsilon^2,\quad
R_j^2=1-\varepsilon^2,\quad \operatorname{Var}(U_j^{(\pm)})=1.
\tag{T1B.1}
\]
总体分数为 $(\mu_j+a^2\mu_d)/(1+a^2)$。快供体 总体保持边界排序；慢供体 总体严格翻转当且仅当
\[
\boxed{\Delta\mu=\mu_j-\mu_i<a^2(\mu_i-\mu_0).} \tag{T1B.2}
\]
每个 $\varepsilon^2>0$ 都可选择足够接近的 $\rho_i,\rho_j$ 满足此条件，同时固定 $\rho_0>\rho_i$。由于不等式严格，随 $T\to\infty$、固定 lag、$N_{\rm ref}\to\infty$，实际同窗真实 top-2 集合依概率趋于总体 top-2 集合，慢供体 的同窗预测错误概率趋于 1，快供体 的同窗预测错误概率趋于 0。这里不是声称给定 $T=193$ 必有高概率翻转；有限 $T$ 还需下列显式余量条件。

记任一轴过程 $X$ 的窗口分数方差 $V_T(X)=2\operatorname{tr}[(M\Sigma_X)^2]$。定义
\(m_-=\mu_i-(\mu_j+a^2\mu_0)/(1+a^2)>0\)，
\(m_+=(\mu_j+a^2\mu_3)/(1+a^2)-\mu_i>0\)。因为 $U_j$ 与 $Z_i$ 独立，Chebyshev 给
\[
P(D(U_j^-)<D_i)\ge1-\frac{V_T(U_j^-)+V_T(Z_i)}{m_-^2},\quad
P(D(U_j^+)>D_i)\ge1-\frac{V_T(U_j^+)+V_T(Z_i)}{m_+^2}. \tag{T1B.3}
\]
为得到整集合陈述，令 $A_0=\{0,i\}$，$\Delta_*=\min_{p\in A_0,q\notin A_0}(\mu_q-\mu_p)>0$。对真实排序所有四个跨界对再用并集界，
\[
P(A(D)\ne A_0)\le B_T:=\sum_{p\in A_0,q\notin A_0}\frac{V_T(Z_p)+V_T(Z_q)}{(\mu_q-\mu_p)^2}.
\tag{T1B.4}
\]
因此慢供体 错选概率至少为 $1-B_T-[V_T(U_j^-)+V_T(Z_i)]/m_-^2$（与 0 取最大）。快供体 保持整个集合的概率至少为 $1-B_T-\sum_{p\in A_0}[V_T(U_j^+)+V_T(Z_p)]/[\mathbb E D(U_j^+)-\mu_p]^2$；未修改外轴 3 的额外比较已在真实排序事件内成立。有限 REF 可另在 $\max_i|\widehat V_i-1|\le\eta$ 的事件上从分差中扣除评分分母误差，使用 Proposition 1；实验中的 32,768 个 REF 状态不能直接视为无限参考样本。

##### Part B 证明与阶数纪律

1. 独立单位方差给 $\operatorname{Var}(U_j)=1$ 和 $\operatorname{Cov}(U_j,Z_j)=1/\sqrt{1+a^2}$，所以 $\mathbb E(U_j-Z_j)^2=2-2/\sqrt{1+a^2}$。按所选 $a$，等于 $\varepsilon^2$。这匹配命名读出的逐轴 MSE；若允许重新拟合可逆的全矩阵解码，则能反解原状态，不能把它称为不可辨识反例。
2. 对一个固定窗口，写 $C_{jd}=\langle Z_j,Z_d\rangle_{T,\mathcal L}$，$\gamma=D_j-D_i$，$q=D_d-D_i$。展开平方并通分，
   \[
   (1+a^2)[D(U_j)-D_i]=\gamma+2aC_{jd}+a^2q. \tag{T1B.5}
   \]
   总体上独立轴使 $\mathbb E C_{jd}=0$，所以代入总体均值得 (T1B.2)。单窗交叉项却通常非零，$|2aC_{jd}|\le2|a|\sqrt{D_jD_d}$，当 $\varepsilon\to0$ 时 $a=\varepsilon+O(\varepsilon^3)$，这是 $O(\varepsilon)$ 的窗口效应。精确地 $\operatorname{Var}(C_{jd})=\operatorname{tr}(M\Sigma_dM\Sigma_j)$，来自对独立高斯向量先条件后取期望。正负同幅 $a$ 的总体分数和总体风险相同（反射 供体路径），但当 $|\gamma+a^2q|<2|aC_{jd}|$ 时同窗的正负两臂给相反排序。
3. 协方差 $\Sigma_U=(\Sigma_j+a^2\Sigma_d)/(1+a^2)$；高斯二次型方差公式给 (T1B.3)，Chebyshev 和并集界给 (T1B.4)。若所有相关系数绝对值小于 1，$\|\Sigma_X\|_{\rm op}$ 有固定上界，$\|M\|_{\rm op}\le4/(T-\max\mathcal L)$、$\operatorname{tr}M=2$，于是 $\operatorname{tr}[(M\Sigma_X)^2]\le\|\Sigma_X\|_{\rm op}^2\operatorname{tr}(M^2)\le8\|\Sigma_X\|_{\rm op}^2/(T-\max\mathcal L)$；各 $V_T$ 为 $O(T^{-1})$，所有严格总体间隔固定时上述概率界趋于 1。有限 REF 的一致收敛已在 Part A 说明。
4. 相比之下，完全独立于信号的 iid 残差 $r=\varepsilon\xi$ 有 $\mathbb E K=0,\ \mathbb E E=2\varepsilon^2,\ \operatorname{Var}(Z+r)=1+\varepsilon^2$，其总体分数偏差为 $\varepsilon^2(2-\mu)/(1+\varepsilon^2)=O(\varepsilon^2)$。独立且与信号同谱的残差有 $\mathbb E E=\varepsilon^2\mu$，总体归一化分数**精确不变**。两者仍可有 $O(\varepsilon)$ 有限窗交叉波动，不能把“总体二阶”误写成“每个窗口二阶”。
5. 更一般联合平稳、允许依赖历史的残差，其总体一阶交叉效应可以达到。设 $q_t=(Z_{t-1}-\rho Z_t)/\sqrt{1-\rho^2}$，$r_t=\pm\varepsilon q_t$。反向 AR 创新 $q$ 为单位方差 iid，且 $\operatorname{Cov}(Z_t,q_t)=0$。对 $\ell\ge1$，$\operatorname{Cov}(Z_t,q_{t+\ell})=\sqrt{1-\rho^2}\,\rho^{\ell-1}$，$\operatorname{Cov}(q_t,Z_{t+\ell})=0$。代入 (P1.7)，总体交叉项为 $\mp2\varepsilon\sqrt{1-\rho^2}\,|\mathcal L|^{-1}\sum_{\ell\in\mathcal L}\rho^{\ell-1}$，是显式非零 $O(\varepsilon)$。这是时间滤波读出类的例子，**不是当前逐帧 MLP 的实现**。
6. 当前独立 高斯 AR 加逐帧读出 $r_i=f_i(Z_t)-Z_{t,i}$ 有更强限制：对任意平方可积 $f$，高斯回归给 $\mathbb E[Z_{t,i}\mid Z_{t+\ell}]=\rho_i^\ell Z_{t+\ell,i}$，故 $b_i(\ell)=b_i(-\ell)=\rho_i^\ell c_i$，从而 $\mathbb E K_i=2c_i\mu_i$。在总体归一化中，一阶 $2c_i\mu_i$ 与分母 $2c_i$ 抵消，
   \[
   \frac{\mathbb E N_i}{\nu_i}-\mu_i=\frac{\mathbb E E_i-v_i\mu_i}{\nu_i}. \tag{T1B.6}
   \]
   这说明该类总体效应二阶；有限窗口的带符号交叉项仍可一阶。不能将第 5 步更广类的总体一阶反例移植为当前自然误差的机制结论。证毕。

#### Theorem 1 的实证边界

A4 的 $1430/32768=4.364013671875\%$ 与最低轴 AUDIT $R^2$ $.99520\text{--}.99636$ 证明的是这批近似读出和有限参考下高恢复仍有同窗错误。这不能证明完美恢复下存在正的不可消除错选下界。11/16 A1 自然修正中 28/28、27/28 活跃组合状态 MSE 同时改善；没有任何组合通过逐轴 $R^2$ 差 $\le.005$ 的匹配标准，因此不能把约 6 pp 收益称为已隔离状态恢复精度的方向因果效应。第二轮一个组合受影响轴 $R^2$ 下降 $.0312136$ 而准确率改善 $16.99\text{ pp}$，又排除了“自然准确率和 $R^2$ 必然严格单调”的说法。人工供体 定理和人工匹配实验不替代自然误差的匹配证据。

### 3.3 Lemma 1 与 Theorem 2：精确二次型矩与集中界

本节采用主文的同一评分算子：为缩短公式，\(S_i=\widehat S_i(z)\) 表示物理轨迹未归一化分数，\(\widehat s_i=\widetilde S_i\) 表示实际预测归一化分数。下文直接使用 \(\widetilde S_i\)；\(F_i\) 与 \(L_i\) 分别是指定矩阵的 Frobenius 范数与算子范数，\(\mathcal L\) 始终表示 lag 集合。时间索引在本节从 0 开始，轴索引仍明确区分代码的 0-based 与物理叙述的 1-based。

#### Lemma 1（实际多 lag 算子的精确高斯矩）

设 \(T>\max\mathcal L\)，\(m=|\mathcal L|\)，\(n_h=T-h\)。令 \(D_h\in\mathbb R^{n_h\times T}\) 的第 \(t\) 行为 \(e_{t+h}^T-e_t^T\)（此处时间从 0 编号），并令

\[
M=\frac1m\sum_{h\in\mathcal L}\frac{D_h^\top D_h}{n_h},\qquad
\Sigma_\rho=(\rho^{|t-s|})_{t,s=0}^{T-1},\quad 0\le\rho<1.
\]

则 \(M\succeq 0\)、\(M\mathbf1=0\)、\(\operatorname{tr}M=2\)，实际未归一化分数恰为 \(S=z^T Mz\)。

在实际 \(\mathcal L=\{4,16,32\}\) 下，\(\ker M\) 是四个 \(t\bmod4\) 同余类上分别为常数的向量空间，维数为 4；没有使用 \(\operatorname{rank}M=T-1\) 的假设。下文一般 \(d\) 表示物理轴数，仓库中 \(d=4\)。令

\[
B_\rho=M^{1/2}\Sigma_\rho M^{1/2},\quad
F_\rho^2=\operatorname{tr}(M\Sigma_\rho M\Sigma_\rho),\quad
L_\rho=\|B_\rho\|_{\rm op}.
\]

对第 \(i\) 轴简记 \(F_i=F_{\rho_i}\)、\(L_i=L_{\rho_i}\)，并记 \(h_{\max}=\max\mathcal L\)。其精确均值和方差是

\[
\mu(\rho)=\frac2m\sum_h(1-\rho^h),\qquad
\operatorname{Var}(S)=2F_\rho^2.
\tag{L1.1}
\]

独立轴 \(i,j\) 的 \(S_j-S_i\) 方差为 \(2(F_i^2+F_j^2)\)。若

\[
a_T=\frac1m\sum_h\frac1{T-h},\quad
\kappa_\rho=\frac{1+\rho}{1-\rho},
\]

则可计算的保守界为

\[
\|M\|_{\rm op}\le4a_T,\quad
\operatorname{tr}(M^2)\le8a_T,\quad
L_\rho\le4\kappa_\rho a_T,\quad
F_\rho^2\le4\kappa_\rho a_T\mu(\rho).
\tag{L1.2}
\]

特别当 \(T\ge2\max\mathcal L\) 时，\(\operatorname{Var}(S)\le16\kappa_\rho\mu/T\)。在 \(\mu\) 为常数量级且 lag 尺度与相关时间可比的区间，该式显示通常的 \(T_{\mathrm{eff}}\asymp T(1-\rho)\) 因子；**它只是统一上界，不是所有固定 lag、所有 \(\rho \to 1\) 极限的精确主阶**。固定短 lag 差分会消去慢的近常数模态，见证明中的更紧界。

##### 完整证明

**步骤 1：算子与均值。** 每个 \(D_h z\) 的第 \(t\) 项就是 \(z_{t+h}-z_t\)，故代码的均方增量等于 \(z^TD_h^TD_hz/n_h\)，等权平均得到 \(M\)。每个 \(D_h^TD_h\) 半正定、消去常数向量，且迹为 \(2n_h\)，因此 \(M\succeq0,\ M\mathbf1=0,\ \operatorname{tr}M=2\)。均值

\[
\mathbb ES=\operatorname{tr}(M\Sigma_\rho)
=\frac1m\sum_h\frac1{n_h}\sum_{t=0}^{n_h-1}
\mathbb E(z_{t+h}-z_t)^2
=\frac2m\sum_h(1-\rho^h).
\]

**步骤 2：精确二阶矩。** 写 \(z=\Sigma^{1/2}g\)，\(g\sim\mathcal N(0,I)\)。对 \(A=\Sigma ^{1/2}M\Sigma ^{1/2}\) 作正交特征分解，得 \(S=\sum_r a_rg_r^2\)，因此由独立性及 \(\operatorname{Var}(g_r^2)=2\)，

\[
\operatorname{Var}(S)=2\sum_r a_r^2
=2\operatorname{tr}(M\Sigma M\Sigma).
\]

\(A\) 和 \(B=M^{1/2}\Sigma M^{1/2}\) 分别是 \(XX^T\) 与 \(X^TX\)（\(X=\Sigma ^{1/2}M^{1/2}\)），故非零特征值、Frobenius 范数、算子范数相同。独立物理轴的评分独立，差方差是方差之和。

也可以直接用全部增量协方差计算精确 Frobenius 范数。令 \(u=t-s\)，

\[
k_{h,l}(u)=\rho^{|u+h-l|}-\rho^{|u+h|}
-\rho^{|u-l|}+\rho^{|u|}.
\]

则 Wick/Isserlis 恒等式（经典高斯矩工具）给出

\[
F_\rho^2=\frac1{m^2}\sum_{h,l}\frac1{n_hn_l}
\sum_{t=0}^{n_h-1}\sum_{s=0}^{n_l-1}k_{h,l}(t-s)^2.
\tag{L1.3}
\]

这同时给出只用有限求和的精确矩算法，完全保留多 lag 之间的协方差；**不同 lag 不能当作独立重复样本**。

**步骤 3：算子范数界。** 对任意向量 \(x\)，

\[
\|D_hx\|^2=\sum_t(x_{t+h}-x_t)^2
\le2\sum_t(x_{t+h}^2+x_t^2)\le4\|x\|^2.
\]

因此 \(\|M\|_{\mathrm{op}}\le4a_T\)。因 \(M\succeq 0\)，\(\operatorname{tr}(M^2)\le\|M\|_{\mathrm{op}}\operatorname{tr}M\le8a_T\)。协方差矩阵每行绝对值和至多

\[
1+2\sum_{u\ge1}\rho^u=\kappa_\rho,
\]

故 \(\|\Sigma_\rho\|_{\mathrm{op}}\le\kappa_\rho\)，从而 \(L_\rho\le4\kappa_\rho a_T\)。再由 \(B\succeq 0\)，

\[
F_\rho^2=\operatorname{tr}(B^2)
\le\|B\|_{\rm op}\operatorname{tr}B
\le4\kappa_\rho a_T\mu(\rho).
\]

\(T\ge2h_{\max}\) 时 \(a_T\le 2/T\)，给出上述方差界。

**步骤 4：展示固定 lag 下统一有效样本界为何可能很松。** 单 lag 分数 \(S_h=\|D_hz\|^2/n_h\) 的增量自协方差为

\[
g_h(u)=2\rho^{|u|}-\rho^{|u+h|}-\rho^{|u-h|}.
\]

若 \(|u|<h\)，由 Cauchy–Schwarz 得 \(|g_h(u)|\le g_h(0)=2(1-\rho ^h)\)；若 \(|u|\ge h\)，精确有

\[
g_h(u)=-(1-\rho^h)^2\rho^{|u|-h}.
\]

令 \(b_h=1-\rho ^h\)。于是

\[
\sum_{u\in\mathbb Z}g_h(u)^2
\le4(2h-1)b_h^2+\frac{2b_h^4}{1-\rho^2},
\]

\[
\operatorname{Var}(S_h)
=\frac2{n_h^2}\sum_{t,s}g_h(t-s)^2
\le V_h:=\frac{8(2h-1)b_h^2+4b_h^4/(1-\rho^2)}{n_h}.
\tag{L1.4}
\]

由于 \(\operatorname{Cov}(S_h,S_l)\le\sqrt{\operatorname{Var}(S_h)\operatorname{Var}(S_l)}\)，

\[
\operatorname{Var}(S)\le\frac1{m^2}\left(\sum_h\sqrt{V_h}\right)^2.
\tag{L1.5}
\]

固定 \(h\)、\(\rho \uparrow 1\) 时，\(b_h\sim h(1-\rho )\)；因此 (L1.4) 主上界是 \(O(h^3(1-\rho )^2/(T-h))\)，不是 \(O(1/[T(1-\rho)])\) 的等价式。与 Theorem 3 的精确 \(\alpha ^2\) 极限一致。证明完毕。

#### Theorem 2（高斯二次型集中及总体排名的充分条件）

沿用 Lemma 1，对任意 \(x>0\)，

\[
\Pr\{|S_i-\mu_i|\ge2F_i\sqrt x+2L_i x\}\le2e^{-x}.
\tag{T2.1}
\]

进一步得到便于代入的双体制界：

\[
\Pr\{|S_i-\mu_i|\ge t\}
\le2\exp\!\left[-\frac18\min\left\{\frac{t^2}{F_i^2},\frac t{L_i}\right\}\right].
\tag{T2.2}
\]

此为经典 Gaussian Hanson–Wright 工具的直接应用，下文给出 MGF 证明以固定常数，不列为新概率不等式。假设轴间独立，总体 top-\(k\) 集合为 \(I_*\)，并有事件 \(G\) 满足 \(\Pr(G^c)\le\eta\) 且在 \(G\) 上所有轴的实际预测分数满足 \(|\widetilde S_i-S_i|\le b_i\)。对于 \(i\in I_*,\ j\notin I_*\) 令

\[
d_{ij}=\mu_j-\mu_i-b_i-b_j>0,\quad
F_{ij}^2=F_i^2+F_j^2,\quad L_{ij}=\max(L_i,L_j).
\]

则预测 top-\(k\) 与**总体**集合 \(I_*\) 不同的概率至多

\[
\eta+\sum_{i\in I_*,j\notin I_*}
\exp\!\left[-\frac18\min\left\{\frac{d_{ij}^2}{F_{ij}^2},\frac{d_{ij}}{L_{ij}}\right\}\right],
\tag{T2.3}
\]

并取与 1 的较小值。对实际**同窗**真值 top-\(k\)，应使用 Proposition 2 的分差事件包含关系；不能把 (T2.3) 中物理过程相对总体均值的涨落当作预测相对同窗真值的误差。

##### 完整证明

**步骤 1：经典高斯 MGF。** 更一般地令 \(g\sim\mathcal N(0,I)\)、\(A\) 为任意实对称矩阵（可不定），\(Y=g^TAg-\operatorname{tr}A\)，\(F^2=\operatorname{tr}(A^2)\)、\(L=\|A\|_{\mathrm{op}}\)。对 \(0<s<1/(2L)\)，逐特征值积分标准正态密度给出

\[
\log\mathbb E e^{sY}
=\sum_r\left[-sa_r-\frac12\log(1-2sa_r)\right].
\]

对 \(|u|<1\)，Taylor 级数绝对值逐项估计得到

\[
-\log(1-u)-u=\sum_{q\ge2}\frac{u^q}{q}
\le\frac{u^2}{2(1-|u|)}.
\]

于是

\[
\log\mathbb E e^{sY}\le\frac{s^2F^2}{1-2sL}.
\tag{T2.4}
\]

取 \(s=\sqrt x/(F+2L\sqrt x)\)，对阈值 \(2F\sqrt x+2Lx\) 应用 Chernoff：

\[
-s(2F\sqrt x+2Lx)+\frac{s^2F^2}{1-2sL}=-x.
\]

因此上尾至多 \(e^{-x}\)；对 \(-A\) 重复即得 (T2.1)。这是 Gaussian Hanson–Wright 的有常数版本，所用 MGF 和 Chernoff 都是外部经典工具。

**步骤 2：双体制常数。** 对任意 \(t>0\) 取

\[
s=\min\{t/(4F^2),1/(4L)\}.
\]

因 \(1-2sL\ge 1/2\)，\(\log\mathbb E e^{sY}\le2s^2F^2\)。若 \(t\le F^2/L\)，指数 \(-st+2s^2F^2=-t^2/(8F^2)\)。若 \(t\ge F^2/L\)，指数至多 \(-t/(4L)+F^2/(8L^2)\le -t/(8L)\)。上、下尾求和即 (T2.2)；单侧不需因子 2。零范数的退化情况表示 \(Y=0\)，直接处理。

**步骤 3：带残差的总体排序。** 如果预测集合不等于 \(I_*\)，必有 \(i\in I_*,\ j\notin I_*\) 使 \(\widetilde S_j\le \widetilde S_i\)（严格无并列时为严格小于；并列按照代码的 stable 轴索引处理不影响上界）。在 \(G\) 上，这推出

\[
(S_i-\mu_i)-(S_j-\mu_j)
\ge\mu_j-\mu_i-b_i-b_j=d_{ij}.
\]

独立性使左边为 \(\operatorname{diag}(A_i,-A_j)\) 的中心高斯二次型，Frobenius 平方与算子范数分别为 \(F_i^2+F_j^2\)、\(\max(L_i,L_j)\)。用单侧 (T2.2) 后对所有跨边界对作 union bound，再加 \(\Pr(G^c)\le\eta\)，得 (T2.3)。

**步骤 4：如何把有界残差代入。** 在物理分母为 1 的情况下，若同时有 \(S_i\le u_i\)、\(E_i=S(r_i)\le e_i\)、\(|\widehat V_i-1|\le\zeta_i<1\)，则由正半定二次型 Cauchy–Schwarz，

\[
|2z_i^TMr_i|\le2\sqrt{S_iE_i},
\quad |\widetilde S_i-S_i|
\le \frac{2\sqrt{u_ie_i}+e_i+\zeta_i u_i}{1-\zeta_i}=:b_i.
\tag{T2.5}
\]

这一 \(G\) 可以由明确的残差、参考与物理分数事件组成；如没有这些控制，单独静态 \(R^2\) 不能提供每窗固定的 \(b_i\)。概率界可以非常保守；本地 \(T=193\) 与窄间隔并不保证产生非平凡的数值上界。证明完毕。

### 3.4 Proposition 2：冻结规则的选择性证书与预测分差事件

**假设。** 条件于编码器、读出、REF、全部CAL及事先固定的阈值这一σ域 \(\mathcal F\)。每一个固定模型—过程—规则单元内，\(W_1,\ldots,W_n\) 是来自同一分布的**独立完整物理轨迹窗口**，且与 \(\mathcal F\) 独立。窗口内部允许任意相关；一个窗口只产生一个二元错误。记
\[
A(W)=1\{\widehat\Delta(W)\ge\tau\},\quad
Y(W)=1\{\widehat K(W)\ne K(W)\},\quad
q=P(A=1\mid\mathcal F),\quad
p=P(Y=1\mid A=1,\mathcal F).
\]
\(q>0\) 时定义 \(p\)；\(q=0\) 时没有有意义的作答风险结论。令 \(M=\sum A_b\)，\(E=\sum A_bY_b\)。对单项错误概率 \(\eta\in(0,1)\)，定义
\[
U_\eta(e,m)=\begin{cases}1,&e=m\quad(含\ m=e=0),\\
u\in(0,1):\ \sum_{j=0}^e{m\choose j}u^j(1-u)^{m-j}=\eta,&0\le e<m.
\end{cases}\tag{P2.1}
\]
其等价写法是 \(\operatorname{Beta}^{-1}_{e+1,m-e}(1-\eta)\)，且 \(U_\eta(0,m)=1-\eta^{1/m}\)。则
\[
P\{p\le U_\eta(E,M)\mid\mathcal F\}\ge1-\eta.\tag{P2.2}
\]
对预先固定、各有 \(q_j>0\) 的 \(J\) 项（单元、规则可相互依赖），用 \(\eta=\alpha/J\)，有
\[
P\{\forall j\le J:\ p_j\le U_{\alpha/J}(E_j,M_j)\mid\mathcal F\}\ge1-\alpha.\tag{P2.3}
\]
这是 Clopper–Pearson 二项反演和 Bonferroni 并集界的应用，不是本文提出的新风险控制算法。

**证明步骤1：随机作答数仍能条件二项化。** 固定作答指示向量 \(a\)。由于窗口独立，其联合密度（或概率测度）条件于各自的 \(A_b=a_b\) 仍逐窗口分解。被选中的 \(m\) 个 \(Y_b\) 独立且同为 Bernoulli\((p)\)，所以 \(E\mid(A_1,\ldots,A_n)=a,\mathcal F\sim\mathrm{Bin}(m,p)\)。对所有有相同 \(m\) 的 \(a\) 混合仍有 \(E\mid M=m,\mathcal F\sim\mathrm{Bin}(m,p)\)。

**步骤2：证明二项反演覆盖。** 设 \(F_p(e)=P_p\{\mathrm{Bin}(m,p)\le e\}\)。对固定 \(e<m\)，\(F_p(e)\) 关于 \(p\) 严格下降，且由端点0、1确定唯一根。故 \(p>U_\eta(e,m)\Rightarrow F_p(e)<\eta\)。集合 \(\{e:F_p(e)<\eta\}\) 是一个可能为空的整数初始区间；若最大值为 \(e_*\)，则该事件概率恰为 \(F_p(e_*)<\eta\)。因此条件于每个 \(m\)，失败概率不超过 \(\eta\)；\(m=0\) 时界为1，结论平凡成立。再对 \(M\) 求全概率即(P2.2)。二项—Beta恒等式可由对二项CDF逐项求导并用边界值验证；不是额外分布近似。

**步骤3：多项同时性。** 对每项各得 \(P(p_j>U_j\mid\mathcal F)\le\alpha/J\)，于是
\(P(\cup_j\{p_j>U_j\}\mid\mathcal F)\le\sum_j\alpha/J=\alpha\)。这里从未假设各模型、规则或过程之间独立。若 \(\mathcal F\) 随机且上述条件几乎处处成立，积分也得到无条件覆盖，但那不意味着同一张区间覆盖重新训练总体的平均风险。

**步骤4：只用可交换性不够。** 设所有窗口错误都是同一个 \(B\sim\mathrm{Bernoulli}(p)\)，并全部作答。这组错误可交换，然而 \(E=nB\) 不服从 \(\mathrm{Bin}(n,p)\)。当 \(B=0\) 时上界 \(1-\eta^{1/n}\to0\)，以概率 \(1-p\) 漏掉固定正风险 \(p\)。因此本证书必须要求条件iid或等价的条件二项结构，不能只写“窗口可交换”。同一长轨迹截取重叠窗口也未自动满足条件。

**步骤5：预测分差为何有筛选意义。** 对任意确定的真实评分向量 \(s\)（可以是同窗 \(D\)，也可以是另行定义的总体 \(\mu\)），令 \(\widehat s=s+e\)，\(K=\mathrm{Top}_k(s)\)。若 \(\widehat K\ne K\)，存在漏选轴 \(i\in K\setminus\widehat K\) 和误选轴 \(j\in\widehat K\setminus K\)。真实 \(s_i\le s_j\)，预测 \(\widehat s_i-\widehat s_j\ge\widehat s_{(k+1)}-\widehat s_{(k)}=\widehat\Delta\)。因此
\[
e_i-e_j=(\widehat s_i-\widehat s_j)+(s_j-s_i)
\ge\widehat\Delta+(s_j-s_i).
\]
所以
\[
\{\widehat K\ne K,\ \widehat\Delta\ge\tau\}
\subseteq\bigcup_{i\in K,\ j\notin K}
\{e_i-e_j\ge\tau+s_j-s_i\}
\subseteq\{2\|e\|_\infty\ge\tau\}.\tag{P2.4}
\]
若分数有并列，按轴索引的stable规则决定集合；弱不等式仍成立。对 \(q>0\)，因此
\[
p\le \min\left(1,\frac{P(2\|e\|_\infty\ge\tau\mid\mathcal F)}q\right).\tag{P2.5}
\]
结合 Proposition 1 的逐轴有界误差或 Theorem 2 的尾界即可得充分条件。它不声称条件风险必随 \(\tau\) 单调下降：筛选也会改变误差分布，且分母 \(q\) 同时下降。实际部署不需知道真实参数来计算 \(\widehat\Delta\)，但解析风险解释仍需要真实误差条件；统计认证用独立TEST标签与(P2.1)。

**代码/实验对齐与边界。** `src/slowrepr/metrics.py:78–94` 恰是 \(\ge\)、单侧Beta分位数、\(e=m\) 上界1；`reports/lejepa_a3_gap_20261115/score.py:97–121` 用标准库二项尾反演实现同一上界，`145–146` 用48项校正；A2为32项，B1/B3为128项。A3 g50数值阈值 \(0.17748381448816608\) 在新TEST前固定，16/16通过；B1 coupled沿用它仍16/16通过。这些结果是条件于各次冻结材料的单元证书。

每单元的4096条独立轨迹可用于二项界；同一物理轨迹被两个训练条件使用，不能计作两条独立轨迹。B每条件65536次评价对应32768条独立物理轨迹；跨时间尺度共用创新保留配对，并不妨碍每项的边际二项证书。聚合数仅作描述。A4按TEST批次内可见得分取最高409/2048，作答决策依赖整批，不直接满足上述逐窗口冻结阈值推导，不能套用A3的同一个证书。CAL与REF共享于测试窗，证书条件化处理；不能把多个窗口当作多次独立REF实现。

`coverage >= .35` 检查的是**观察覆盖率** \(M/n\)，并无总体覆盖率的置信下界。CP的保证是风险上界覆盖，不是“观察通过就必然真实通过”，也不是每个被接受窗口正确的保证。B3 slower重校准只有14/16的上界通过，不能据此证明另两项真实风险高于1%。统计口径对齐 `docs/audit/20261116/PROGRESS_BEFORE_REORGANIZATION.md:157–171` 及 `reports/lejepa_local_closeout_20261116/AUDIT.md:19–23`。

### 3.5 Theorem 3：时间尺度变化与覆盖塌缩

#### 定理陈述（时间尺度变化：覆盖塌缩成立，SNR 单调退化不成立）

设 \(0<\rho _i<1\)、\(\lambda_i=-\log\rho_i\)、\(A_\alpha=\operatorname{diag}(e^{-\alpha\lambda_i})\)、\(Q_\alpha =I-A_\alpha  A_\alpha ^T\)，取递推 \(Z_{t+1}=A_\alpha Z_t+\xi_t\)，创新 \(\xi_t\overset{\mathrm{iid}}\sim\mathcal N(0,Q_\alpha)\)，与平稳初值 \(Z_0\sim\mathcal N(0,I)\) 独立。冻结逐帧读出 \(f\)，假定各 \(f_i\in L^2(\mathcal N(0,I))\)，并冻结正的 REF 分母 \(\widehat V_i\)。则同刻状态分布、总体静态 \(R^2\) 与冻结 REF 在所有 \(\alpha >0\) 下不变。对总体慢轴 \(i\) 和较快轴 \(j\)（\(\lambda _i<\lambda _j\)），

\[
\Delta\mu(\alpha)=\frac2m\sum_h
\big(e^{-\alpha\lambda_i h}-e^{-\alpha\lambda_j h}\big)>0,
\tag{T3.1}
\]

总体顺序不变，并且

\[
\Delta\mu'(\alpha)=\frac2m\sum_h h\big(\lambda_j e^{-\alpha\lambda_jh}
-\lambda_i e^{-\alpha\lambda_i h}\big),\qquad
\Delta\mu(\alpha)=2\alpha\bar h(\lambda_j-\lambda_i)+O(\alpha^2),
\tag{T3.2}
\]

其中 \(\bar h=m^{-1}\sum_{h\in\mathcal L}h\)，一般记 \(\overline{h^q}=m^{-1}\sum_{h\in\mathcal L}h^q\)。该间隔对 \(\alpha\) 不全局单调：\(\alpha \to 0\) 与 \(\alpha\to\infty\) 时均趋零。

对理想物理读出，任意固定正阈值 \(\tau\) 的覆盖率满足

\[
\Pr\{g_\alpha\ge\tau\}
\le \frac{2\alpha\bar h}{\tau}\sum_{i=1}^d\frac{\lambda_i}{\widehat V_i}
\longrightarrow0.
\tag{T3.3}
\]

对于每坐标全局 Lipschitz 常数为 \(L_{f,i}\) 的冻结读出，可把右端改成

\[
\frac{2\alpha\bar h}{\tau}
\left(\sum_j\lambda_j\right)
\left(\sum_i\frac{L_{f,i}^2}{\widehat V_i}\right).
\tag{T3.4}
\]

固定 \(T,\mathcal L\)，定义 \(H_{ts}=|t-s|\)、\(C=-M^{1/2}HM^{1/2}\succeq 0\)，则

\[
\operatorname{Var}(S_j-S_i;\alpha)
=2\alpha^2(\lambda_i^2+\lambda_j^2)\operatorname{tr}(C^2)+O(\alpha^3),
\tag{T3.5}
\]

因而

\[
\lim_{\alpha\downarrow0}\mathrm{SNR}(\alpha)
=\frac{2\bar h(\lambda_j-\lambda_i)}
{\sqrt{2(\lambda_i^2+\lambda_j^2)\operatorname{tr}(C^2)}}>0.
\tag{T3.6}
\]

本地参数的精确矩更给出 \(\mathrm{SNR}(.25)>\mathrm{SNR}(1)>\mathrm{SNR}(4)\)。所以不能把 B3 变慢后的同窗预测错误率上升解释为该总体边界 SNR 的下降，也不能声称重校准后的条件风险由此 SNR 唯一决定。

##### 完整证明

**步骤 1：不变的静态分布。** 因 \(Q_\alpha\succ0\) 且 \(A_\alpha I A_\alpha^T+Q_\alpha=I\)，对时间归纳得到 \(Z_t\sim\mathcal N(0,I)\)。固定的逐帧 \(f\) 对每个 \(\alpha\) 均作用于相同的输入分布，故 \(\mathbb E[(f_i(Z)-Z_i)^2]\)、\(\operatorname{Var}(Z_i)=1\)、总体 \(R_i^2\) 一致。独立审计样本如复用同一批状态，经验 \(R^2\) 也逐项相同；如各自重抽，只能说同分布。冻结 REF 的实际数值无须重新抽样，逐项严格不变。

**步骤 2：间隔与导数。** 平稳协方差为 \(\exp(-\alpha\lambda_i|t-s|)\)，套用 Lemma 1 即 (T3.1)；逐有限项求导即 (T3.2)。单个 lag 的导数由正变负的位置为

\[
\alpha_h^*=\frac{\log(\lambda_j/\lambda_i)}{h(\lambda_j-\lambda_i)}.
\]

因此总间隔至少在 \(0<\alpha<\min_h\alpha_h^*\) 上严格增加，在 \(\alpha>\max_h\alpha_h^*\) 上严格下降；中间区间不可不经分析宣称全局单调。由 \(0\le x-(1-e^{-x})\le x^2/2\)，得到显式余项

\[
\left|\Delta\mu(\alpha)-2\alpha\bar h(\lambda_j-\lambda_i)\right|
\le\alpha^2\overline{h^2}(\lambda_i^2+\lambda_j^2).
\tag{T3.7}
\]

若保留二阶系数，则余项绝对值至多

\[
\frac{\alpha^3}3\overline{h^3}(\lambda_i^3+\lambda_j^3),
\]

二阶项为 \(-\alpha^2\overline{h^2}(\lambda_j^2-\lambda_i^2)\)。所有常数只依赖固定的 \(\mathcal L\) 与 \(\lambda\)。\(\alpha\to\infty\) 时两指数均趋零，所以边界间隔也趋零。

**步骤 3：固定阈值的覆盖塌缩。** 所有预测分数非负，故排序后边界分差 \(g_\alpha\le\sum_i\widetilde S_i\)。理想读出 \(f_i(z)=z_i\) 满足

\[
\mathbb E \widetilde S_i=\frac{2}{m\widehat V_i}\sum_h(1-e^{-\alpha\lambda_i h})
\le\frac{2\alpha\bar h\lambda_i}{\widehat V_i}.
\]

Markov 不等式立刻给出 (T3.3)。若 \(f_i\) 是 \(L_{f,i}\)-Lipschitz，则

\[
\mathbb E[f_i(Z_{t+h})-f_i(Z_t)]^2
\le L_{f,i}^2\mathbb E\|Z_{t+h}-Z_t\|^2
=2L_{f,i}^2\sum_j(1-e^{-\alpha\lambda_jh})
\le2\alpha hL_{f,i}^2\sum_j\lambda_j.
\]

再平均 lag、除冻结分母、用 Markov，得到 (T3.4)。对任意目标覆盖 \(c_0>0\)，该上界明确给出一个足够小的 \(\alpha\) 使覆盖低于 \(c_0\)。这不等于已经证明实际 \(\alpha =.25\) 的精确覆盖量。

更弱地，对于任何 \(f_i\in L^2(\mathcal N(0,I))\)，相同覆盖极限仍成立但一般不提供上述线性速率。证明可用高斯 \(L^2\) 中多项式的稠密性：选多项式 \(p\) 使 \(\|f_i-p\|_2<\varepsilon\)；由两端状态的边缘分布固定，三角不等式使两端替换误差不超过 \(2\varepsilon\)；对固定多项式，\(Z_{t+h}-Z_t\to 0\) 的高斯联合矩收敛给出 \(\|p(Z_{t+h})-p(Z_t)\|_2\to0\)；先令 \(\alpha \to 0\)、再令 \(\varepsilon \to 0\)。有限个 lag、轴相加及 Markov 完成证明。

**步骤 4：固定窗口极限的方差。** 令 \(J=\mathbf1\mathbf1^T\)。逐元素 Taylor 展开为

\[
\Sigma_{i,\alpha}=J-\alpha\lambda_iH+E_{i,\alpha},\qquad
|(E_{i,\alpha})_{ts}|\le\tfrac12\alpha^2\lambda_i^2(T-1)^2.
\]

从而

\[
\|E_{i,\alpha}\|_F\le E_0\alpha^2\lambda_i^2,
\qquad E_0=\tfrac12T(T-1)^2.
\]

由于 \(M\mathbf1=0\)，\(M^{1/2}JM^{1/2}=0\)，故

\[
B_{i,\alpha}=\alpha\lambda_i C+R_{i,\alpha},\quad
\|R_{i,\alpha}\|_F\le K\alpha^2\lambda_i^2,
\quad K=\|M\|_{\rm op}E_0.
\tag{T3.8}
\]

对任意 \(u\) 满足 \(\sum_tu_t=0\)，用

\[
|t-s|=\sum_{a=0}^{T-2}
\big(\mathbf1_{t\le a<s}+\mathbf1_{s\le a<t}\big)
\]

展开后可直接求和得到

\[
u^THu=-2\sum_{a=0}^{T-2}\left(\sum_{t=0}^a u_t\right)^2\le0.
\]

\(M^{1/2}\) 的像包含于 \(\mathbf1^\perp\)，因此 \(C\succeq 0\)。又

\[
\operatorname{tr}C=-\operatorname{tr}(MH)=2\bar h>0,
\]

所以 \(\operatorname{tr}(C^2)>0\)。范数展开及 Cauchy–Schwarz 给出显式余项：

\[
\left|\operatorname{Var}(S_i;\alpha)-2\alpha^2\lambda_i^2\|C\|_F^2\right|
\le4K\|C\|_F\alpha^3\lambda_i^3+2K^2\alpha^4\lambda_i^4.
\tag{T3.9}
\]

独立轴方差相加，得到 (T3.5)；与 (T3.7) 相除即 (T3.6)。当 \(\alpha\to\infty\) 时 \(\Sigma\to I\)，差方差趋于 \(4\operatorname{tr}(M^2)>0\)，而均值差趋于零，所以 \(\mathrm{SNR}\to0\)。上述结论已足以否定“混合变慢使所定义 SNR 必然下降”的普遍命题。证明完毕。

##### 代码与封存实验对齐

实际冻结生成器 `reports/lejepa_b_transfer_20261115/run.py:46–60` 精确满足上述过程；`reports/lejepa_b_transfer_20261115/run.py:104–127` 冻结 REF 并用各 regime CAL 的 `method="higher"` 中位数设阈值。`reports/lejepa_b_transfer_20261115/SUMMARY.json` 的 `aggregate` 中 baseline 全答错率 4.5120239258%、slower 5.3924560547%、faster 5.4977416992%。在各自 CAL 中位阈值附近，错误分别为 4/32527、52/32690、7/32758；变慢减原过程 +0.1467725761 个百分点，是见过的 TEST 的探索性比较。精确物理总体 SNR 却在变慢后上升，表明它不具备跨条件解释该同窗残差错误的充分性。11/16 绝对预测 gap 区间粗配平后 slower 相对 baseline 为 -1.0998pp，也禁止写出独立于 gap 的普遍动态损害定理。

### 3.6 Theorem 4：保持边缘分布的三角耦合

#### 定理陈述（保持边缘分布的三角耦合）

令 \(A=D+c e_p e_q^T\)，\(p=2,q=3\) 为 1-based 物理轴，\(D=\operatorname{diag}\rho\)、\(|\rho _i|<1\)，\(Q=I-AA^T\)。则

\[
Q\succeq0\iff c^2\le(1-\rho_p^2)(1-\rho_q^2),
\quad Q\succ0\iff c^2<(1-\rho_p^2)(1-\rho_q^2).
\tag{T4.1}
\]

取平稳初值与独立高斯创新后，所有单轴完整轨迹分布，以及各轴分数的均值、方差与全部边际分布，与 \(c=0\) 时严格一致。令

\[
f_h=\sum_{r=0}^{h-1}\rho_p^{h-1-r}\rho_q^r
=\begin{cases}(\rho_p^h-\rho_q^h)/(\rho_p-\rho_q),&\rho_p\ne\rho_q,\\
h\rho_p^{h-1},&\rho_p=\rho_q,
\end{cases}
\]

并令 \(G_{ts}=f_{t-s}\) 当 \(t>s\)、否则为零，则

\[
\operatorname{Cov}(S_p,S_q)
=2c^2\|M^{1/2}GM^{1/2}\|_F^2,
\tag{T4.2}
\]

\[
\operatorname{Var}(S_q-S_p;c)
=\operatorname{Var}(S_q-S_p;0)
-4c^2\|M^{1/2}GM^{1/2}\|_F^2.
\tag{T4.3}
\]

在本地正 \(\rho\)、非空正 lag、\(c\ne 0\) 下，减项严格为正。此精确 \(O(c^2)\) 改变作用于跨轴协方差与差方差，不作用于单轴方差。冻结读出的动态交叉误差仍可发生 \(O(c\varepsilon )\) 改变，故没有任意读出下固定阈值迁移必然成功的定理；\(c=.12\) 的 16/16 通过状态来自封存检验。

##### 完整证明

**步骤 1：创新方差可行性的充要条件。** 除 \((p,q)\) 子块外 \(Q\) 是正对角项 \(1-\rho _i^2\)。写 \(a=\rho _p,b=\rho _q\)，特殊子块是

\[
Q_{pq}=\begin{pmatrix}1-a^2-c^2&-cb\\-cb&1-b^2\end{pmatrix},
\quad \det Q_{pq}=(1-a^2)(1-b^2)-c^2.
\]

因 \(1-b^2>0\)，Schur complement 判据给出 \(Q_{pq}\succeq0\) 当且仅当行列式非负；此时第一对角项亦非负。严格正定用严格不等号，得到 (T4.1)。代码调用 Cholesky 因而实际要求正定。宽、窄允许最大 \(|c|\) 分别为 .1792683707、.1809649485；实际 .12 均严格可行，\(\lambda_{\min}(Q)\) 分别为 .0622422065、.0649204723。

**步骤 2：矩阵幂。** 因唯一的非对角元从 \(q\) 到 \(p\)、且 \(p\ne q\)，任何非零乘积只能包含至多一次 \(e_p e_q^T\)。由归纳得到

\[
A^h=D^h+c\left(\sum_{r=0}^{h-1}a^{h-1-r}b^r\right)e_pe_q^T
=D^h+cf_he_pe_q^T.
\tag{T4.4}
\]

平稳方差为 \(I\)，故 \(\operatorname{Cov}(Z_{t+h},Z_t)=A^h\)；对角项是 \(\rho _i^h\)，完全不含 \(c\)。全部过程联合高斯，因此任一轴的任意有限维分布与独立 AR(1) 一致；分数是该路径的确定函数，其均值、方差和完整边际分布也相同。

**步骤 3：跨轴平方型协方差。** 同刻 \(p,q\) 协方差为零。若 \(t>s\)，\(\operatorname{Cov}(Z_{p,t},Z_{q,s})=cf_{t-s}\)；若 \(t\le s\) 为零，因此两轨迹的交叉协方差是 \(cG\)。对任意中心联合高斯向量 \(x,y\)，逐项展开并用 Isserlis 公式

\[
\operatorname{Cov}(x_ax_b,y_cy_d)
=K_{ac}K_{bd}+K_{ad}K_{bc}
\]

得到

\[
\operatorname{Cov}(x^TMx,y^TMy)
=2\operatorname{tr}(MKMK^T)
=2\|M^{1/2}KM^{1/2}\|_F^2.
\]

代入 \(K=cG\) 得 (T4.2)；两边际方差不变，差方差等于两方差之和减两倍协方差，即 (T4.3)。严格性亦可不用矩阵核空间推断：相同 lag/端点的一对增量有

\[
\operatorname{Cov}(Z_{p,t+h}-Z_{p,t},Z_{q,t+h}-Z_{q,t})=-cf_h\ne0,
\]

并且全部增量平方的协方差是交叉协方差的平方乘 2、权重均正，故总协方差严格正。其他轴对仍相互独立。

**步骤 4：为什么这还不是读出迁移保证。** 令逐帧残差 \(r_i(Z)=f_i(Z)-Z_i\) 有有限二阶矩，并令 \(w_i=\mathbb E[Zr_i(Z)]\)。高斯条件均值给出

\[
\mathbb E[Z_{t+h}\mid Z_t]=A^hZ_t,\qquad
\mathbb E[Z_t\mid Z_{t+h}]=(A^h)^TZ_{t+h}.
\]

逐四项展开增量交叉乘积后得到

\[
2\mathbb E[(Z_{i,t+h}-Z_{i,t})(r_i(Z_{t+h})-r_i(Z_t))]
=2e_i^T[2I-A^h-(A^h)^T]w_i.
\tag{T4.5}
\]

与 \(c=0\) 相比，额外项为

\[
-2cf_h\big(\mathbf1_{i=p}(w_i)_q+\mathbf1_{i=q}(w_i)_p\big).
\]

由 Cauchy–Schwarz，\(|(w_i)_j|\le\sqrt{\mathbb E r_i^2}=\varepsilon_i\)，所以该改变量至多 \(2|c|\,|f_h|\varepsilon_i\)（\(i=p,q\)）；这是真正可能出现的一阶 \(O(c\varepsilon )\) 项。若 \(f_p(Z)=Z_p+\beta Z_q\)，其静态误差 \(\mathbb E r_p^2=\beta^2\) 与总体分母 \(1+\beta ^2\) 不变，但额外总体评分偏差恰为

\[
-\frac{2c\beta}{m(1+\beta^2)}\sum_hf_h.
\tag{T4.6}
\]

符号随 \(c\beta\) 变化。因此物理单轴分布不变不能保证任意读出动态评分误差不变，当然也不能推出任意阈值风险不变。证明完毕。

##### 代码与封存实验对齐

对宽、窄边界轴，在 \(c=.12\) 时分数协方差分别为 .06712247645、.06704231084，差标准差由 .55681250395、.55975881242 降至 .41927939570、.42337371956；对应总体 SNR 从 .25113694836、.01904306869 升至 .33351553758、.02517757958。这些是理想物理分数的精确矩，不预测实际误差数。封存强读出旧阈值实测 \(31867/65536=48.6251831\%\) 作答、7/31867 错、16/16 通过、最坏同时上界 .60406233396%。它与该范围下没有新迁移障碍相容，但不构成任意耦合的充分保证。

### 3.7 Theorem 5：有限 CAL 下 lag 解混的条件稳定性与平均失效边界

#### Definitions：封存校准算子

本节中 \(N=N_{\mathrm{cal}}\) 始终表示独立 CAL **轨迹条数**，B2 为 \(N=1024\)，每条长度 \(T=193\)，不能把同一轨迹的帧当作独立样本。校准 lag 集合为 \(\mathcal M_{\mathrm{cal}}=\{1,4,16,32\}\)；它不同于评测 lag 集合 \(\mathcal L=\{4,16,32\}\)，也不同于评测二次型矩阵 \(M_{\mathcal L}\)。

记 CAL 嵌入为列向量 \(h_{nt}\in\mathbb R^d\)，\(n=1,\ldots,N\)，\(t=1,\ldots,T\)，权重 \(w_n\ge0,\ \sum_nw_n=1\)。令 \(\operatorname{sym}(A)=(A+A^T)/2\)。定义

\[
m=\sum_nw_n\frac1T\sum_th_{nt},\qquad
\widehat R_0=\sum_nw_n\frac1T\sum_th_{nt}h_{nt}^{\!T}-mm^T,
\tag{B2.1}
\]

\[
R_\ell^{\rm raw}=\sum_n\frac{w_n}{T-\ell}
 \sum_{t=1}^{T-\ell}\operatorname{sym}(h_{nt}h_{n,t+\ell}^{\!T}),
\quad
a_\ell=\sum_n\frac{w_n}{2(T-\ell)}
 \sum_{t=1}^{T-\ell}(h_{nt}+h_{n,t+\ell}),
\tag{B2.2}
\]

\[
\widehat\Gamma_\ell=R_\ell^{\rm raw}-a_\ell m^T-ma_\ell^T+mm^T,
\quad W=\widehat R_0^{-1/2},\quad
\widehat C_\ell=W\widehat\Gamma_\ell W,
\quad
\widehat{\bar C}=\frac14\sum_{\ell\in\{1,4,16,32\}}\widehat C_\ell.
\tag{B2.3}
\]

因此 \(\widehat\Gamma_\ell\) 也等于把每个端点都减去同一个 \(m\) 后的对称交叉矩。不能把它误写成分别用左右端点均值中心化后的协方差。

#### Theorem 5：陈述

**假设。** 物理源各轴是相互独立、平稳、标准高斯 AR(1)，\(0<\rho_d<\cdots<\rho_1<1\)。每条 CAL 轨迹独立，轨迹长度 \(T\)，基础拟合 \(w_n=1/N\)。线性基准嵌入为 \(h^0_t=Bz_t\)，\(B\) 可逆；实际嵌入允许 \(h_t=Bz_t+e_t\)，其中残差可有均值、时序相关和跨轴相关，不预先假定实际神经网络输出为高斯。涉及下述总体残差矩时，另假定 \((z_t,e_t)\) 二阶联合平稳且 \(\mathbb E\|e_t\|^2<\infty\)；保留样本残差项的代数界不需此附加分布假设。记

\[
R_* =BB^T,\quad m_*=\lambda_{\min}(R_*)>0,\quad
W_*=R_*^{-1/2},\quad O=W_*B,\quad D_\ell=\operatorname{diag}(\rho_1^\ell,\ldots,\rho_d^\ell),
\]

则 \(O\) 正交，理想白化算子为 \(C_\ell^*=OD_\ell O^T\)。

**(i) 估计误差与白化误差。** 定义精确、可分解的误差量

\[
\eta_0=\|\widehat R_0-R_*\|_{\mathrm{op}},\quad
\eta_\ell=\|\widehat\Gamma_\ell-BD_\ell B^T\|_{\mathrm{op}}.
\]

若 \(\eta_0\le m_*/2\)，则

\[
\boxed{\ \|\widehat C_\ell-C_\ell^*\|_{\mathrm{op}}\le
E_\ell:=\frac{2\eta_\ell}{m_*}
 +(2+\sqrt2)\frac{\|BD_\ell B^T\|_{\mathrm{op}}\eta_0}{m_*^2}.\ }
\tag{B2.4}
\]

下面的完全显式界给出线性高斯部分的有限 CAL 贡献。固定 \(0<\beta<1\)，令 \(q=|\mathcal M_{\mathrm{cal}}|\)，\(r=\max_i\rho_i\)，

\[
K=\frac{1+r}{1-r},\qquad K_2=\frac{1+r^2}{1-r^2},\qquad
x=\log\frac{2(q+1)(d^2+d)}{\beta},\qquad n_0=T,\quad n_\ell=T-\ell,
\]

\[
b_\ell=2\sqrt{\frac{K_2x}{N n_\ell}}+\frac{2Kx}{N n_\ell},\quad
u_\ell=\sqrt{\frac{2dKx}{N n_\ell}},\qquad
q_0=db_0+u_0^2,\quad q_\ell=db_\ell+2u_0u_\ell+u_0^2.
\tag{B2.5}
\]

设 \(D_{0,\mathrm{samp}}=\widehat R_0(h)-\widehat R_0(h^0)\)，\(D_{\ell,\mathrm{samp}}=\widehat\Gamma_\ell(h)-\widehat\Gamma_\ell(h^0)\)。以至少 \(1-\beta\) 的概率，同时有

\[
\eta_0\le\|B\|_{\mathrm{op}}^2q_0+\|D_{0,\rm samp}\|_{\mathrm{op}},\qquad
\eta_\ell\le\|B\|_{\mathrm{op}}^2q_\ell+\|D_{\ell,\rm samp}\|_{\mathrm{op}}.
\tag{B2.6}
\]

式 (B2.6) 是对实际残差项保留显式随机量的有条件界；若没有进一步控制 \(D_{\ell,\mathrm{samp}}\)，它不是实际神经网络的数值风险证书。主平方根项为 \(O\!\left(d\sqrt{\frac{x}{N(T-\ell)(1-r)}}\right)\)，线性尾项为 \(O\!\left(\frac{dx}{N(T-\ell)(1-r)}\right)\)，全帧中心化项为 \(O\!\left(\frac{dKx}{N\sqrt{T(T-\ell)}}\right)\)，常数已由 (B2.5) 给出。

**(ii) 谱方向稳定性。** 令 \(\delta_i(\ell)=\min_{j\ne i}|\rho_i^\ell-\rho_j^\ell|>0\)，\(v_i\) 与 \(\widehat v_i\) 是按同一特征值排序匹配的单位特征向量。作为经典 Davis–Kahan 工具在本算子上的应用，

\[
\boxed{\ \sin\angle(\widehat v_i,v_i)\le
\min\{1,2E_\ell/\delta_i(\ell)\}.\ }
\tag{B2.7}
\]

对连续特征值簇 \(I\)，大小 \(s\)，到外部谱的间隔 \(\delta_I>0\)，亦有
\(\|\sin\Theta(\widehat V_I,V_I)\|_F\le\min\{\sqrt s,2\sqrt s E_\ell/\delta_I\}\)。单根轴命名还须另有命名匹配稳定条件；子空间界本身不保证命名。一个与代码一致的显式充分条件如下。令 \(R\) 为已对齐候选列顺序后的命名响应矩阵，并定义

\[
\mathcal N(R)_{ij}=\frac{|R_{ij}|}{\max\{\|R_{:j}\|_2,10^{-12}\}},\qquad
U_R(\pi)=\sum_{i=1}^d\mathcal N(R)_{i,\pi(i)}.
\tag{B2.7a}
\]

若唯一最优命名 \(\pi_*\) 的效用间隔为
\(\kappa=U_R(\pi_*)-\max_{\pi\ne\pi_*}U_R(\pi)>0\)，且

\[
\xi:=\max_{ij}|\mathcal N(\widehat R)_{ij}-\mathcal N(R)_{ij}|,
\qquad \kappa>2d\xi,
\tag{B2.7b}
\]

则扰动后的命名仍为 \(\pi_*\)。此外，记 \(s_{\min}=\min_i|R_{i,\pi_*(i)}|\)；若
\(\max_{ij}|\widehat R_{ij}-R_{ij}|<s_{\min}-10^{-8}\)，则所有扰动后的命名斜率绝对值严格大于 \(10^{-8}\)，满足代码尺度检查。若所有列范数至少为 \(r_{\min}>10^{-12}\)，且 \(\zeta=\max_j\|\widehat R_{:j}-R_{:j}\|_2<r_{\min}-10^{-12}\)，则截断不激活，并有 \(\xi\le2\zeta/r_{\min}\)，从而 \(\kappa>4d\zeta/r_{\min}\) 是直接以响应矩阵误差表述的充分条件。这里并未假设封存模型天然满足这些间隔。

**(iii) lag 的总体信号。** 对 \(\rho_i>\rho_j>0\)，设 \(a=-\log\rho_i,\ b=-\log\rho_j\)，则连续 lag 变量的正间隔

\[
\delta_{ij}(\ell)=e^{-a\ell}-e^{-b\ell},\qquad
\ell_* =\frac{\log(b/a)}{b-a}\in(1/b,1/a)
\tag{B2.8}
\]

在 \((0,\ell_*)\) 严格增，在 \((\ell_*,\infty)\) 严格减，最终以 \(e^{-a\ell}\) 衰减；接近的两轴满足 \(\ell_*\approx1/|\log\bar\rho|\)。而 (B2.5) 的噪声上界随 \(T-\ell\) 减少而增大。因此“拉长 lag 增大谱差”只有局部含义，不能推出解混更准。

**(iv) 等权矩阵平均。** 有

\[
\bar C^*=O\operatorname{diag}(\bar\lambda_i)O^T,\quad
\bar\lambda_i=\frac1q\sum_{\ell\in\mathcal M_{\mathrm{cal}}}\rho_i^\ell,\quad
\bar\delta_{ij}=\frac1q\sum_\ell\delta_{ij}(\ell),\quad
\|\widehat{\bar C}-\bar C^*\|_{\mathrm{op}}\le\frac1q\sum_\ell E_\ell.
\tag{B2.9}
\]

这里单一有序轴对的间隔取正平均，但最近邻间隔须在平均之后重新取 \(\min\)。同一 CAL 和同一白化所带来的各 lag 误差相关，不能擅自把误差除以 \(\sqrt q\)。

不能只由“大 lag 的方差不减”断言平均必然更差。一个足够、可证明的劣化条件如下：在某两个物理方向张成的孤立二维块中，所有 lag 的误差仅为对称非对角项 \(e_\ell\)，总体间隔为 \(\delta_\ell>0\)。若它们同号，且 \(|e_\ell|/\delta_\ell\ge|e_{\ell_0}|/\delta_{\ell_0}\)，至少一项严格，则平均矩阵的角误差严格大于基准 lag \(\ell_0\)。精确公式为

\[
\theta_\ell=\frac12\arctan\frac{2|e_\ell|}{\delta_\ell},\quad
\bar\theta=\frac12\arctan\frac{2|\sum_\ell e_\ell|}{\sum_\ell\delta_\ell}.
\tag{B2.10}
\]

特别地，大 lag 项若具有不小的同向误差且 \(\delta_\ell\le\delta_{\ell_0},\ |e_\ell|\ge|e_{\ell_0}|\)，就满足该条件。若 \(e\) 联合零均值高斯，允许相关，只有非对角噪声，则角误差的随机大小由

\[
\frac{\sqrt{\sum_{\ell,h}\operatorname{Cov}(e_\ell,e_h)}}{\sum_\ell\delta_\ell}
\tag{B2.11}
\]

控制；该量大于基准 \(\operatorname{sd}(e_{\ell_0})/\delta_{\ell_0}\) 是平均角误差随机劣化的充分且在此二维模型中必要的条件。单独比较各 lag 方差不足以判断 (B2.11)。

#### 解释与工具归属

总体谱间隔增大，只减少 (B2.7) 的分母病态程度；有限 CAL、白化误差、残差偏差和命名不稳定均仍在分子或后续步骤。平均可能稀释信号，但也可能平均掉噪声，方向与跨 lag 协方差必须进入判断。Davis–Kahan、SOBI 的谱分离机制与高斯二次型集中均是外部成熟工具，本定理是针对封存算子的显式实例化，不是新谱扰动工具。

#### 完整证明

##### 步骤 1：代码中心化与总体极限

展开 \(\operatorname{sym}[(h_t-m)(h_{t+\ell}-m)^T]\)，交叉项的加权平均为 \(-a_\ell m^T-ma_\ell^T\)，常数项为 \(mm^T\)，得到 (B2.3)。这也说明将两个端点分别减各自样本均值会改换算子。

在线性理想情形，\(\mathbb E h_t=0\)，\(\mathbb E[h_t h_t^T]=BB^T\)，\(\operatorname{sym}\mathbb E[h_t h_{t+\ell}^T]=BD_\ell B^T\)。由 \(OO^T=W_*R_*W_*=I\) 得 \(O\) 正交，故 \(W_*BD_\ell B^TW_*=OD_\ell O^T\)。所有理想 lag 矩阵共享特征向量。

实际残差先按总体均值中心化，记 \(e_t^c=e_t-\mathbb E e_t\)。它给出的总体偏差为

\[
D_\ell^{\rm pop}=\operatorname{sym}\{B\mathbb E[z_t(e^c_{t+\ell})^T]
 +\mathbb E[e_t^c z_{t+\ell}^T]B^T+\mathbb E[e_t^c(e^c_{t+\ell})^T]\}.
\tag{B2.12}
\]

在 \(\ell=0\) 同样得到协方差偏差。若 \(\|\operatorname{Cov}(e_t)\|_{\mathrm{op}}\le v_e\)，对任意单位向量利用 Cauchy–Schwarz，三个矩阵的范数分别至多 \(\|B\|_{\mathrm{op}}\sqrt{v_e},\ \|B\|_{\mathrm{op}}\sqrt{v_e},\ v_e\)，因此
\(\|D_\ell^{\mathrm{pop}}\|_{\mathrm{op}}\le2\|B\|_{\mathrm{op}}\sqrt{v_e}+v_e\)。对称化只去掉反对称部分，不能消除 (B2.12) 中保留的偏差。把 \(D_{\ell,\mathrm{samp}}\) 写成 \(D_\ell^{\mathrm{pop}}+(D_{\ell,\mathrm{samp}}-D_\ell^{\mathrm{pop}})\) 就分离了总体有偏项和残差的有限 CAL 波动。没有残差尾假设时，不把后者写成自动随 \(N^{-1/2}\) 消失的确定量。

还可给出不需要残差分布的样本控制。设 \(u_{nt}=Bz_{nt}-m_{Bz}\)、\(v_{nt}=e_{nt}-m_e\)，在某 lag 两端定义

\[
U_L=\sum_n\frac{w_n}{n_\ell}\sum_t\|u_{nt}\|^2,\quad
U_R=\sum_n\frac{w_n}{n_\ell}\sum_t\|u_{n,t+\ell}\|^2,
\]

以及相应 \(V_L,V_R\)。逐项展开并对加权和用 Cauchy–Schwarz 得

\[
\|D_{\ell,\rm samp}\|_{\mathrm{op}}\le
\sqrt{U_LV_R}+\sqrt{V_LU_R}+\sqrt{V_LV_R}.
\tag{B2.13}
\]

零 lag 为 \(2\sqrt{U_0V_0}+V_0\)。残差幅度为 \(O(\varepsilon)\) 时，交叉偏差可为 \(O(\varepsilon)\)，残差自项为 \(O(\varepsilon^2)\)，与模块一一致。

##### 步骤 2：相关高斯帧上的精确二次型及集中

考虑一条理想物理轨迹，按时间堆叠为 \(Z\in\mathbb R^{Td}\)，协方差 \(\Omega\)。由于各轴独立，\(\|\Omega\|_{\mathrm{op}}=\max_i\|\Sigma(\rho_i)\|_{\mathrm{op}}\le K\)：每行绝对和不超过 \(1+2\sum_{h\ge1}r^h=K\)。

令 \(U_\ell\) 的 \((t,t+\ell)\) 位置为 1，其余为 0，\(F_\ell=(U_\ell+U_\ell^T)/(2n_\ell)\)，\(F_0=I/T\)；\(J_{jk}=(e_je_k^T+e_ke_j^T)/2\)。对称原始矩的 \((j,k)\) 元素恰为

\[
Z^T(F_\ell\otimes J_{jk})Z.
\tag{B2.14}
\]

\(\|F_\ell\|_{\mathrm{op}}\le1/n_\ell\)，\(\|J_{jk}\|_{\mathrm{op}}\le1\)。对 \(N\) 个独立轨迹取平均后，白化到独立标准高斯向量，其总二次型矩阵 \(A_{\ell,jk}\) 满足

\[
\|A_{\ell,jk}\|_{\mathrm{op}}\le\frac K{Nn_\ell}.
\tag{B2.15}
\]

为得到更好的 Frobenius 界，不用粗糙的 \(K^2\) 上界，而用 Wick 恒等式。对同一轴 \(j=k\)，令 \(Y_t=z_{tj}z_{t+\ell,j}\)，则

\[
\operatorname{Cov}(Y_t,Y_s)=\rho_j^{2|t-s|}
 +\rho_j^{|t-s-\ell|+|t-s+\ell|}.
\]

第一项在整数差 \(h=t-s\) 上的绝对和为 \((1+\rho_j^2)/(1-\rho_j^2)\le K_2\)。第二项的和，由两个移位序列的 Cauchy–Schwarz，至多相同的 \(K_2\)。故
\(\operatorname{Var}(n_\ell^{-1}\sum_tY_t)\le2K_2/n_\ell\)。对 \(j\ne k\)，对称项 \(Y_t=(z_{tj}z_{t+\ell,k}+z_{tk}z_{t+\ell,j})/2\) 的 Wick 展开有四项；每项的差分绝对和均由 Cauchy–Schwarz 不超过 \(K_2\)，带上系数后得到不超过 \(K_2/n_\ell\)，从而同一个 \(2K_2/n_\ell\) 上界仍成立。零 lag 亦相同。

独立轨迹平均使方差除以 \(N\)。高斯二次型的精确方差为 \(2\|A\|_F^2\)，故

\[
\|A_{\ell,jk}\|_F^2\le\frac{K_2}{Nn_\ell}.
\tag{B2.16}
\]

使用外部高斯二次型集中工具
\(\mathbb P(|g^TAg-\operatorname{tr}A|>2\|A\|_F\sqrt x+2\|A\|_{\mathrm{op}}x)\le2e^{-x}\)，由 (B2.15–16) 得每个元素误差至多 \(b_\ell\)，失败概率 \(2e^{-x}\)。\(d\times d\) 矩阵逐元素绝对值不超过 \(b\) 时，\(\|E\|_{\mathrm{op}}\le\|E\|_F\le db\)。

##### 步骤 3：有限中心化均值项

每个物理轴全帧均值的方差不超过 \(K/(NT)\)。端点均值的时间权重非负，总和 1，最大权重不超过 \(1/n_\ell\)，故权重平方和不超过 \(1/n_\ell\)，其方差不超过 \(K/(Nn_\ell)\)。一维高斯尾界与逐轴 union bound 给出 \(\|m_z\|\le u_0\)、\(\|a_{\ell,z}\|\le u_\ell\)，每个标量事件失败概率不超过 \(2e^{-x}\)。

总共需要 \((q+1)d^2\) 个二次型事件和 \((q+1)d\) 个均值事件。选择 (B2.5) 的 \(x\)，并用 union bound 后，所有事件共同成立概率至少 \(1-\beta\)。中心化增项满足
\(\|a_\ell m^T+ma_\ell^T-mm^T\|_{\mathrm{op}}\le2u_\ell u_0+u_0^2\)；零 lag 减去 \(mm^T\) 的增项至多 \(u_0^2\)。乘上 \(B\) 的左右范数后就得到 (B2.6)。没有一步把 \(NT\) 帧当成独立帧；\(K,K_2\) 正是轨迹相关性的代价。

若使用任意与数据独立的固定权重，(B2.15) 的 \(1/N\) 换为 \(\max_n w_n\)，(B2.16) 与均值方差中的 \(1/N\) 换为 \(\sum_nw_n^2\)。这不把封存 bootstrap 的经验 90% 半径自动变成严格置信界；bootstrap 是条件于已观测 CAL 的另一分布。

##### 步骤 4：白化误差传播

Weyl 不等式给 \(\lambda_{\min}(\widehat R_0)\ge m_*-\eta_0\ge m_*/2\)，故 \(\|W\|\le\sqrt{2/m_*}\)。对任意正定矩阵，

\[
A^{-1/2}=\frac1\pi\int_0^\infty t^{-1/2}(A+tI)^{-1}\,dt.
\]

用预解式恒等式及两个矩阵特征值都至少为 \(m_*/2\)，

\[
\|W-W_*\|\le\frac{\eta_0}{\pi}
 \int_0^\infty\frac{t^{-1/2}}{(m_*/2+t)^2}dt
 =\frac{\sqrt2\eta_0}{m_*^{3/2}}.
\]

记 \(\Gamma_*=BD_\ell B^T\)，恒等展开

\[
W\widehat\Gamma W-W_*\Gamma_*W_*
=W(\widehat\Gamma-\Gamma_*)W
 +(W-W_*)\Gamma_*W+W_*\Gamma_*(W-W_*).
\]

第一项范数不超过 \(2\eta_\ell/m_*\)，后二项和不超过
\(\frac{\sqrt2\eta_0}{m_*^{3/2}}\|\Gamma_*\|\frac{\sqrt2+1}{\sqrt{m_*}}\)，即 (B2.4)。

##### 步骤 5：外部 Davis–Kahan 工具及本问题的逐轴证明

归属：经典 Davis–Kahan \(\sin\Theta\) 扰动理论；采用 [Yu, Wang & Samworth, 2015, Theorem 2 / Corollary 3](https://arxiv.org/pdf/1405.0680) 的总体谱间隔形式。下面对本题所需的一维情形给出直接证明。

令 \(E=\widehat C-C_*\)，\(\eta=\|E\|_{\mathrm{op}}\)，\(\delta=\delta_i\)。若 \(\eta\ge \delta/2\)，则 \(\sin\theta\le1\le2\eta/\delta\)。否则 Weyl 给 \(|\widehat\lambda_i-\lambda_i|\le\eta\)。把 \(\widehat C\widehat v_i=\widehat\lambda_i\widehat v_i\) 投影到 \(v_i\) 的正交补：

\[
(C_* -\widehat\lambda_i I)P_\perp\widehat v_i
 =-P_\perp E\widehat v_i.
\]

该补空间上的最小奇异值至少 \(\delta-\eta\)，因此
\((\delta-\eta)\|P_\perp\widehat v_i\|\le\eta\)，得到 \(\sin\theta\le\eta/(\delta-\eta)\le2\eta/\delta\)。对连续谱簇，对每个估计列向量使用同一论证，再求平方和得 \(\sqrt s\,\eta/(\delta_I-\eta)\)；当 \(\eta\ge \delta_I/2\) 用平凡上界 \(\sqrt s\)，即所陈述的簇界。

##### 步骤 6：lag 最优点与 multi 的充分条件

先补上命名稳定性的证明。对任意排列 \(\pi\)，逐项相加有 \(|U_{\widehat R}(\pi)-U_R(\pi)|\le d\xi\)，故对每个 \(\pi\ne\pi_*\)，

\[
U_{\widehat R}(\pi_*)-U_{\widehat R}(\pi)
\ge U_R(\pi_*)-U_R(\pi)-2d\xi
\ge\kappa-2d\xi>0.
\]

因此最优排列不变。反三角不等式给 \(|\widehat R_{i,\pi_*(i)}|\ge s_{\min}-\max_{jk}|\widehat R_{jk}-R_{jk}|>10^{-8}\)。对非零列向量 \(a,b\)，

\[
\left\|\frac b{\|b\|}-\frac a{\|a\|}\right\|
\le\frac{\|b-a\|}{\|a\|}
 +\frac{|\|b\|-\|a\||}{\|a\|}
\le\frac{2\|b-a\|}{\|a\|}.
\]

逐坐标取绝对值是 1-Lipschitz，故 \(\xi\le2\zeta/r_{\min}\)。列符号翻转不改变归一化绝对响应；比较原始斜率时需先以相同符号约定对齐候选列，最终代码的斜率除法也会消去该列的符号。

对 \(\delta(\ell)=e^{-a\ell}-e^{-b\ell}\) 求导得 \(\delta'=-a e^{-a\ell}+b e^{-b\ell}\)。等式 \(\delta'=0\) 等价于 \(e^{(b-a)\ell}=b/a\)，唯一解为 (B2.8)，两侧符号直接给出单峰性。对 \(\log\) 用均值定理，有 \(\ell_*=1/\xi_{\log}\)，\(\xi_{\log}\in(a,b)\)，得到近邻峰值近似。

平均算子共享总体正交基，故特征值逐项平均，正 AR 系数保证同一轴对差分同号，得 (B2.9)。算子误差上界是三角不等式；不同 lag 使用同一 CAL，不能假设误差独立。

二维对称矩阵特征方程给出 \(\tan(2\theta)=2e/\delta\)。同号假设下，平均的噪信比满足

\[
\frac{|\sum_\ell e_\ell|}{\sum_\ell\delta_\ell}
=\sum_\ell\frac{\delta_\ell}{\sum_h\delta_h}
 \frac{|e_\ell|}{\delta_\ell}
\ge \frac{|e_{\ell_0}|}{\delta_{\ell_0}}.
\]

\(\arctan\) 严格递增，得到 (B2.10) 的角误差劣化，某项严格则总体严格。若 \(e\) 联合高斯，则 \(\sum_\ell e_\ell\) 为标准正态乘以其标准差，\(\bar\theta\) 是该绝对值除以 \(\sum_\ell\delta_\ell\) 的严格递增函数，于是 (B2.11) 与基准比值的比较给出随机序关系。

**为什么较弱的原要求不成立。** 取两个 lag，\(\delta_1>\delta_2>0\)，误差 \(e_2=-e_1\)。两者估计方差相同，大 lag 信号更弱，平均误差却严格为零。即使两误差独立且同方差，取 \(\delta_2=0.9\delta_1\)，平均噪信比是基准的 \(\sqrt2/1.9<1\)。因此必须补充误差方向/相关性或足够强的信号稀释条件，才能证明劣化。

##### 步骤 7：方向误差如何传到同窗分数，以及仍不能推出什么

为展示与模块一的衔接，在无限 REF 的二维正交理想例中令 \(\widehat z_i=\cos\theta\,z_i+\sin\theta\,z_j\)，则该坐标总体方差仍为 1。用模块一的评分矩阵 \(M_{\mathcal L}\)，精确有

\[
\widehat S(\widehat z_i)-\widehat S(z_i)
=\sin^2\theta\{\widehat S(z_j)-\widehat S(z_i)\}
 +\sin(2\theta)z_i^TM_{\mathcal L}z_j.
\tag{B2.17}
\]

独立物理轴使 \(\mathbb E[z_i^TM_{\mathcal L}z_j]=0\)，但有限窗里其方差是
\(\operatorname{tr}(M_{\mathcal L}\Sigma_jM_{\mathcal L}\Sigma_i)>0\)。因此总体偏差通常为 \(O(\theta^2)\)，同窗动态交叉误差可为 \(O(\theta)\)。有限 REF 再由 Proposition 1 加入分母误差。该公式给出风险传播机制与上界，**不证明角度一旦增大，每个窗口或每种残差方向的错误率就一定增大**。命名置换跳变也不由连续角度界覆盖。

#### Code/experiment alignment remarks：适用边界

封存实现使用同一个全帧均值与对称白化矩阵，并对四个白化 lag 矩阵先等权平均，再做一次特征分解。其零 lag 最小特征值若不大于 \(10^{-8}\) 直接报错，没有 ridge 修正；独立 REF 方差使用 `ddof=0`。所有有限校准界的独立单位均为整条轨迹。

B2 窄间隔下，`multi` 的平均“理想间隔/经验扰动半径”为 \(0.3869\)，高于 `lag16` 的 \(0.3357\)，但平均方向损失分别为 \(0.2677\) 和 \(0.2153\)，同窗 top-2 正确率分别为 \(66.02\%\) 和 \(70.81\%\)。这正是该比值不能完整排列方法细粒度顺序的封存反例。经验 bootstrap 90% 半径是局部稳定性诊断，没有在本节被提升为总体误差的严格置信上界。

本节严格证明的是给定假设下的算子误差传播、方向与命名稳定的充分条件，以及受控二维条件下等权平均增大角误差的条件。它不证明实际非线性编码器天然满足线性高斯假设；不证明任意大 lag 或任意多 lag 都更差；不把 Davis–Kahan 上界当成紧充要风险准则；不由角误差大小直接推出某个窗口或全部窗口的 0–1 选集错误率。

### 3.8 Proposition 3：离散排序的边界测度与选择分数变换

#### P3A：正确的充要条件、误差界与反集中条件

给定分数 $s$，令 $A=A(s)$，修正后 $s'_i=s_i-\delta_i$。严格无平分时，
\[
\boxed{A(s')\ne A(s)\iff\exists i\in A,j\notin A:\quad\delta_j-\delta_i>s_j-s_i.} \tag{P3.1}
\]
含平分时，将右侧补为“严格大于，或等于且 $j$ 的零基轴号小于 $i$”，即修正后 $j$ 的稳定排序键 $(s'_j,j)$ 小于 $(s'_i,i)$。这要求检查**全部**跨边界轴对；只有同时跨越原第 $k$ 和第 $k+1$ 轴这一对才可简写成超过原分差。一般在无平分时，$\max_{i\in A,j\notin A}(\delta_j-\delta_i)>g$ 仅必要、非充分；允许平分时应改为 $\max_{i\in A,j\notin A}(\delta_j-\delta_i)\ge g$ 必要、非充分。例如 $k=2,\ s=(0,1,2,100),\ \delta=(0,0,0,3)$ 的原分差 为 1，修正差可为 3，但集合不变。

记 $g=s_{(k+1)}-s_{(k)}\ge0$、$R_\delta^2=\mathbb E\|\delta\|_\infty^2$。对任意 $h>0$，无需独立性，
\[
\boxed{|\operatorname{Acc}(s')-\operatorname{Acc}(s)|\le P(A(s')\ne A(s))
\le P(g\le h)+\frac{4R_\delta^2}{h^2}.} \tag{P3.2}
\]
若还明确假设一个边界反集中条件 $P(g\le h)\le Ch/\sigma_{\rm traj}$（在所用 $h$ 范围内），则选择 $h=2(R_\delta^2\sigma_{\rm traj}/C)^{1/3}$ 得
\[
|\Delta\operatorname{Acc}|\le\min\{1,3C^{2/3}(R_\delta/\sigma_{\rm traj})^{2/3}\}. \tag{P3.3}
\]
$R^2>.99$ 或 $R_\delta\ll\sigma_{\rm traj}$ **本身均不蕴含反集中条件**，更不能不估计 $C$ 就把右端写成 $<0.05\text{ pp}$。

在联合高斯 **线性读出**的子类中，可用显式谱条件给反集中常数。将每对差写成 $G^\top B_{ij}G$，$G\sim\mathcal N(0,I)$；若每个 $B_{ij}$ 至少有四个绝对值不小于 $a_*>0$ 的特征值，则
\[
\|f_{s_i-s_j}\|_\infty\le\frac1{4a_*},\qquad
P(g\le h)\le\binom d2\frac{h}{2a_*}. \tag{P3.4}
\]
这仅适用于满足谱条件的高斯二次型分数，不能不经证明直接转给强非线性读出的分数分布。

##### P3A 证明

如果新旧集合不同，必有新进入 $j$ 和退出 $i$，故新稳定排序键 $j$ 小于 $i$。反之，只要存在外轴 $j$ 排在内轴 $i$ 前面，原集合的全部 $k$ 个轴不可能仍是新排序的前 $k$ 个。将 $s'_j<s'_i$ 移项即 (P3.1)，平分情形同理。

对所有原跨界对均有 $s_j-s_i\ge g$，而 $\delta_j-\delta_i\le2\|\delta\|_\infty$。因此 $g>h$ 且 $2\|\delta\|_\infty<h$ 时不可能换集。取事件补集、并集界及 Markov 不等式得 (P3.2)。两个正确性指示量不同必须集合不同，所以准确率差的绝对值不超过换集概率。代入反集中假设并对 $Ch/\sigma+4R^2/h^2$ 求导，得 $h^3=8R^2\sigma/C$，代回为 (P3.3)。

为证 (P3.4)，对角化 $B_{ij}$ 后，特征函数绝对值为 $\prod_r(1+4t^2\lambda_r^2)^{-1/4}$。取满足条件的四项，其余项均不超过 1，得到绝对值不超过 $(1+4a_*^2t^2)^{-1}$；该函数可积，Fourier 反演给密度上界
$(2\pi)^{-1}\int_{\mathbb R}(1+4a_*^2t^2)^{-1}\,dt=1/(4a_*)$。任一轴对落在 $[-h,h]$ 的概率至多 $h/(2a_*)$。若排序边界分差 不超过 $h$，总有某一轴对的分数差绝对值不超过 $h$；对 $\binom d2$ 对作并集界即得。证毕。

#### P3B：严格单调变换何时保持作答集

1. 对任一**严格递增**函数 $f$，单一批次分数 $g_w$ 的稳定降序排列 不变，故按固定名次 取最高 $m$ 个的作答集完全不变。若门限是 CAL 的某个实际顺序统计量 $q=g_{(r)}$，变换后按相同 $r$ 使用 $f(g_{(r)})$，则 $g_{\rm test}\ge q$ 与 $f(g_{\rm test})\ge f(q)$ 等价，作答集亦不变。常数平移和正斜率仿射变换还与 NumPy 线性插值分位数严格相容。
2. **任意严格递增非线性变换与默认线性插值经验分位数不一定相容。** CAL $\{0,2\}$ 的线性中位数是 1；取 $f(x)=x^2$（对非负分差 严格递增），变换后线性中位数为 2。TEST 分差 $1.2$ 原本通过 $\ge1$，变换后 $1.44<2$ 不通过。这反驳不加分位数实现限定的“完全相同”主张。
3. 若分数变换随模型、条件或有向轴对改变，即使每组内是平移，混池后的排列 仍可能改变。A4 使用
   \[
   g_{\rm adj}(w)=\min_{i\in A(w),j\notin A(w)}\{s_j(w)-s_i(w)-b_{ij}\}, \tag{P3.5}
   \]
   因为 $b_{ij}$ 依赖轴对，且取得最小值的轴对 随窗口变化，通常不是 $f(g_{\rm raw})$。只有所有相关 $b_{ij}$ 相同为 $b$ 时才精确退化为 $g_{\rm adj}=g_{\rm raw}-b$；单调不变性不能证明 A4 两个作答集相同。
4. 一个可用的近似界是：若 $b_{\min}\le b_{ij}\le b_{\max}$，令 $b_0=(b_{\min}+b_{\max})/2,\ \eta=(b_{\max}-b_{\min})/2$，则 $|g_{\rm adj}-(g_{\rm raw}-b_0)|\le\eta$。在同一批次各选最高 $m$ 个时，只有距原第 $m$ 大阈值不超过 $2\eta$ 的分数可能改变入选状态。因此近似重合还需要该作答边界附近的质量小；“平滑校准”本身不保证。

##### P3B 证明

严格递增函数保持每一对分数的大于、等于关系，所以保持稳定排列与顺序统计量。仿射变换与两点线性组合可交换，故保持插值分位数；非线性反例已直接计算。对 (P3.5)，每一轴对的扣减介于两个常数之间，对所有轴对取最小后仍有 $g_{\rm raw}-b_{\max}\le g_{\rm adj}\le g_{\rm raw}-b_{\min}$，即误差界。若原入选 $i$ 被原未入选 $j$ 替代，则原分数 $r_i\ge t\ge r_j$，新分数 $r'_j\ge r'_i$；$|r'-r|\le\eta$ 给 $r_i-r_j\le2\eta$，所以两者均距 $t$ 不超过 $2\eta$。证毕。

#### 与 A1／A4 封存结果的对齐与边界

`local/SUMMARY.json` 中 MSE 改善与准确率变化的比较基线不同，必须注明：

| 量 | primary | replication |
|---|---:|---:|
| `mean` 的交叉差预测 MSE | .005246092031451266 | .005537966181671965 |
| `degree2` 的 MSE | .004411668212798929 | .004764886126655563 |
| `degree2` 相对 `mean` 的 MSE 降幅 | 15.90562677226811% | 13.959638424209397% |
| `zero/raw` top-2 正确数 | 26915/28672 | 27094/28672 |
| `mean` top-2 正确数 | 26912/28672 | 27068/28672 |
| `degree2` top-2 正确数 | 26926/28672 | 27100/28672 |
| `degree2−zero` 准确率差 | +0.038364955357 pp | +0.020926339286 pp |
| `degree2−mean` 准确率差 | +0.048828125 pp | +0.111607142857 pp |
| `degree2` 相对 raw 修复／损害数 | 268／257 | 219／213 |

故 $<0.05\text{ pp}$ 是**净准确率**差，不是换集概率小于 $.05\%$；至少 $525/28672=1.83105\%$ 和 $432/28672=1.50670\%$ 的窗口正确性已改变，另可能存在“错集变成另一错集”。归档明确显示修复和损害抵消。交叉项预测误差 MSE 减少约 15% 不等于修正幅度 $R_\delta$ 很小，更不等于总评分误差 MSE 减少 15%；Proposition 1 中残差能量和分母项仍可抵消交叉项。不能仅由现有汇总反推出 (P3.3) 的数值上界已小于 $.05\text{ pp}$。

A4 两规则各接受 6544 次，共同接受 6339 次，分别独有 205 次，所有观察作答均正确。相同总准确率来源于这批数据的共同/独有区域都未观察到错误，并非两集合严格相同的数学推论。其完整 32768 次评价含共享轨迹，底层为 16384 条物理轨迹；批次 rank 选择也不满足逐窗固定独立 Bernoulli 接受规则，不能直接移用 A3 的固定阈值 CP 证书。零观察错选与配对重抽样退化区间 $[0,0]$ 均不证明总体零风险。

## 4. 解析预测与封存实验逐项对照

复现入口：

```bash
/opt/miniconda3/bin/python scripts/low_dim_ar1/verify_theory_moments.py --help
/opt/miniconda3/bin/python scripts/low_dim_ar1/verify_theory_moments.py
```

脚本只从封存目录读取12份`config.json`/`SUMMARY.json`，不导入封存run.py、不加载预测数组或模型、不生成轨迹、不重跑评测。数学计算只有固定矩阵运算与汇总计数核对；输出限制在本仓库`results/`。它兼容Python 3.11语法；本机实际可用NumPy的已有解释器是Python 3.13.11、NumPy 2.4.4，本轮未安装依赖。运行环境由输出JSON记录，不将用户环境描述误写成实际运行版本。

`SUMMARY.json`含每轴均值、方差、Frobenius/算子范数、完整分数协方差矩阵、Q最小特征值、名义边界SNR及输入SHA256。`VALIDATION.json`用独立的双lag增量协方差有限求和核对矩阵迹公式，最大绝对差约\(5.6\times10^{-17}\)；核对\(M\)的四维零空间、B1单轴矩不变、B3慢极限。纯解析数值的误差尺度不等于实验科学结论的误差尺度。

### 4.1 B1/B3：两个基础过程 × 四个条件

分差中位数列是8个单元各自中位数的算术平均，不是合池中位数；只凭SUMMARY无法恢复后者。

|过程/条件|Δμ|σ差|SNR|实测全答错率|预测分差中位数均值|固定覆盖|50%校准错/答（错率）|
|---|---:|---:|---:|---:|---:|---:|---:|
|separated_wide/baseline|0.139836193|0.556812504|0.2511369|4.2572%|0.182522|51.019%|3/16421 (0.0183%)|
|separated_wide/coupled|0.139836193|0.419279396|0.3335155|4.0009%|0.174299|49.338%|7/16665 (0.0420%)|
|separated_wide/slower|0.137781118|0.339565865|0.4057567|5.3406%|0.117188|33.008%|15/16704 (0.0898%)|
|separated_wide/faster|0.064856297|0.438476438|0.1479128|5.3070%|0.145857|41.891%|5/16432 (0.0304%)|
|separated_narrow/baseline|0.010659526|0.559758812|0.0190431|4.7668%|0.177768|50.070%|1/16106 (0.0062%)|
|separated_narrow/coupled|0.010659526|0.423373720|0.0251776|4.7150%|0.168001|47.913%|0/16203 (0.0000%)|
|separated_narrow/slower|0.010590120|0.339759718|0.0311694|5.4443%|0.110569|30.878%|37/15986 (0.2315%)|
|separated_narrow/faster|0.004932302|0.436694000|0.0112946|5.6885%|0.144472|41.879%|2/16326 (0.0123%)|

在8条件的28对描述性比较中，9对违背“SNR越高、同窗模型错率越低”。
该比较不是显著性检验。理想精确读出对同窗真值的错误率为0，全部实测错误均与此理想条件有偏离；
不能将实测误差全部归因于非线性残差，有限REF也不同，且总体SNR不描述模型误差方向。

### 4.2 A4同窗任务：总体边界倒序代理不等于模型风险

|过程|精确SNR|总体倒序正态近似（非保证）|总体倒序严格下界|实测模型错率|
|---|---:|---:|---:|---:|
|separated_wide|0.2511369|40.085%|0.000%|4.2175%|
|separated_narrow|0.0190431|49.240%|39.637%|4.5105%|

正态尾仅供展示错误目标混用的数量级；未用它拟合、认证或解释A4错误。

### 4.3 B2封存诊断（模型均值）

|过程|校准lag|准确率|理想最近间隔|经验矩阵半径q90|间隔/半径均值|旋转损失|
|---|---|---:|---:|---:|---:|---:|
|separated_wide|lag1|82.434%|0.0120000|0.0031639|3.8023|0.0039|
|separated_wide|lag4|82.019%|0.0333858|0.0101678|3.2924|0.0047|
|separated_wide|lag16|79.977%|0.0312971|0.0197788|1.5841|0.0145|
|separated_wide|lag32|71.289%|0.0090752|0.0227888|0.3983|0.1111|
|separated_wide|multi|80.307%|0.0214395|0.0111850|1.9194|0.0098|
|separated_narrow|lag1|79.010%|0.0020000|0.0032266|0.6229|0.0840|
|separated_narrow|lag4|73.044%|0.0059297|0.0102372|0.5802|0.1340|
|separated_narrow|lag16|70.810%|0.0071597|0.0213627|0.3357|0.2153|
|separated_narrow|lag32|59.436%|0.0028998|0.0232506|0.1248|0.3645|
|separated_narrow|multi|66.022%|0.0044973|0.0116440|0.3869|0.2677|

bootstrap经验半径不是已证明的总体算子误差上界；multi不是联合对角化。

### 4.4 A1 local：必须区分MSE参照、净准确率与修复/损害

|轮次|MSE相对mean下降|准确率相对raw变化|准确率相对mean变化|修复/损害|
|---|---:|---:|---:|---:|
|primary|15.9056%|+0.03836pp|+0.04883pp|268/257|
|replication|13.9596%|+0.02093pp|+0.11161pp|219/213|

全部输入路径、SHA256、精确矩、统计口径与局限见SUMMARY.json。


### 4.5 证书结果与不能由理想SNR预测的量

| 封存检验 | 观测结果 | 数学结论/限制 |
|---|---|---|
| A3 g50 | 33355/65536作答，3错；16/16通过；最坏CP上界0.4512632834% | Proposition 2的具体应用；每单元独立窗口、48项校正，合池次数不用于二项置信区间 |
| A2 strong/weak | strong 6/32934错、16/16通过；weak 145/32784错、11/16通过 | 固定阈值对读出预算不普遍稳定；拟合64标签仍有4096验证标签，恢复反差仅4/8达到原目标 |
| B1 coupled固定阈值 | 31867/65536作答，7错；16/16通过，最坏上界0.6041% | Theorem 4允许该迁移成功，但不证明强读出必须通过；边界差方差降低约43%，单轴方差不变 |
| B3 baseline固定阈值 | 33125/65536=50.5447%，4错 | 作为旧阈值覆盖参照 |
| B3 slower固定阈值 | 20934/65536=31.9427%，15错；0/16同时通过 | 覆盖未过；\(\alpha\to0\)给覆盖极限而不是\(\alpha=.25\)的精确预测 |
| B3 faster固定阈值 | 27450/65536=41.8854%，6错；16/16通过 | 不支持时间尺度变化一律导致风险证书失败 |
| B3按各自CAL中位阈值 | baseline 4/32527；slower 52/32690；faster 7/32758 | slower比baseline +0.14677pp；仅14/16上界通过，未通过不等于真实风险>1% |
| 11/16 B3全部窗口绝对分差粗配平 | slower比baseline −1.099846pp；15/16单元负差 | 比较对象不是前行接受的约50%窗口；不支持独立于gap的普遍动态损害 |

A3最坏上界的完整数值以原SUMMARY为准；近似百分数不能代替原逐单元认证。上述不同检验的Bonferroni家庭分别冻结，并无覆盖所有历史探索的一张新总体95%保证。

**对单一SNR假说的量化判断。** 两基础过程中，slower相对baseline的总体gap仅降低约1.47%和0.65%，而名义边界差标准差约降39%；因此SNR上升约61.6%和63.7%。与此同时实测全答错误分别增加约1.0834pp和0.6775pp。不能把这些偏离统称“非线性观测导致的误差”：理想SNR的目标是总体边界排序，实测是同窗预测排序，且后者含有限REF和具体读出残差。读出精确且参考分母匹配时，理想同窗任务错误为0，实测相对它的偏离为表中4%–5.7%，现有JSON不支持进一步唯一分摊成残差/参考/时间因素贡献。

在\(\alpha\to0\)极限，wide/narrow的SNR分别趋约0.489535/0.0380778，而非0；脚本在\(\alpha=10^{-5}\)的纯解析值核对该极限，不是新动力学实验。全局正常近似尾概率只作一个明确标识的辅助列，既无误差保证，也不参加任何风险证书。

**对A1“修补不敏感”的修正。** 已有数字只能说明所选修正的净准确率收益很小；修复/损害明显抵消。现有汇总没有给出校正幅度的\(L^2\)量及实际非线性分差的反集中常数，不能宣称已把Proposition 3数值化为小于0.05pp的理论上界。这一缺口用条件与诚实边界报告，本轮不新增评测来填充它。

## 5. 假设、贡献归属与不可声称清单

### 5.1 结论层级

| 层级 | 本文已经给出的内容 | 不应升级为何种结论 |
|---|---|---|
| 无分布逐样本代数 | P1分子/REF分解；P2分差错误事件；P3全部跨界pair充要条件 | 不能自动转成小风险、单调风险或自然误差因果结论 |
| 指定概率模型的严格结论 | Gaussian AR精确矩、常数明确的集中、总体倒序下界、三角耦合精确变化、慢极限、有限CAL/DK界 | 不覆盖任意非线性读出、非平稳漂移或未经控制的残差 |
| 额外条件的充分保证 | 残差/REF有界时保序；反集中时小修正换集概率小；同向/相关噪声条件下multi角误差增大 | 不是充要风险准则，不说明封存强MLP已满足全部条件 |
| 独立TEST统计证书 | 条件iid轨迹、冻结阈值、预定家庭中的CP/Bonferroni上界 | 不是总体覆盖下界、训练总体保证、逐窗口正确性保证 |
| 已封存实证 | A1–A4/B1–B3的既有结果与11/16探索分析 | 不是本轮新增独立验证，不产生新训练算法或通用阈值 |
| 外部工具与理论 | Gaussian二次型矩、Hanson–Wright、Davis–Kahan、Pinsker、CP/Bonferroni、LeJEPA可辨识与SFA/SOBI | 不作本文数学工具创新，不把已知机制改名为新方法 |

### 5.2 严格边界核对

- **评分目标：** \(K_W\)与\(K_{pop}\)分别命名。前者的完美恢复错误为零（匹配分母或无限REF），后者可以有接近一半的有限窗边界倒序概率。两者不能相互替代。
- **恢复度：** AUDIT R²是有限独立状态估计；高R²不是对每条窗口残差的确定性上界。当前逐帧独立Gaussian AR下，总体一阶协方差效应与方差归一化抵消；窗口一阶项仍存在。历史依赖读出的总体一阶反例不属于当前MLP实现。
- **分母：** `ddof=0`存在\(-\nu/N_{ref}\)偏差；四阶矩控制REF波动。非线性读出不自动具有Gaussian方差公式。冻结REF共享误差不会随测试窗数增加而消失。低维已知物理方差1与通用双REF接口明确分开。
- **时间尺度：** 有限T、固定lag下\(\alpha\to0\)和\(T\to\infty\)不能随意交换。相关时间延长不自动使增量方差增大；差分消掉慢的常数分量。不能将渐近gap坍缩当作\(\alpha=.25\)时总体gap明显变小的实证解释。
- **耦合：** B1只是一条严格三角边、匹配\(Q=I-AA^T\)的模型。单轴方差变化为零，跨轴评分协方差才为\(O(c^2)\)。固定强读出16/16通过是范围内证据，不是任意耦合的保证。
- **CAL解混：** B2没有ridge，多lag是矩阵平均。区分N条独立轨迹和NT相关帧；最近邻谱间隔不总等于2–3轴间隔；经验bootstrap半径不是自动有效的总体置信界。稳定子空间不等于命名稳定。
- **等权平均：** 大lag信号衰减且各项方差不减仍不足以推出平均必然变差，跨lag误差可能相消。正文给的是额外条件下可证的劣化，不是把B2局部现象升格为普遍规律。
- **离散变化：** top-k变更需看所有跨界pair。小净准确率变化不等于选集变化概率小；MSE下降15%也不等于修正幅度很小。A1自然修正没有恢复质量匹配对照，且存在R²下降、准确率上升的局部例外。
- **分位数：** 保序变换须严格递增，且保持rank或使用实际顺序统计量（本仓库阈值为`higher`）。任意非线性变换不与默认插值分位数交换；跨模型混池的不同平移可能大幅改rank，不能无条件叫“微扰”。A4的pair allowance通常不是全局单调gap函数。
- **统计：** 只有可交换性不足CP；完整独立轨迹才是本设计统计单位。不能把共享轨迹的模型评价当独立样本、合并家庭后沿用旧置信度或把上界未过解释成真实风险超标。A4批次rank不直接属于P2的逐窗固定阈值证书。
- **理论新颖性：** 这次工作补足严格形式化、揭示原陈述中的反例和适用条件；未证明这些组合足以构成ICML级原创贡献。文献近邻与已有理论重叠须按 `docs/LITERATURE_GAPS_AND_POSITIONING_20261116.md` 继续保留。

### 5.3 外部工具与文献归属

以下文献提供外部理论或直接近邻；本文的证明是特定代码算子的应用和完整展开。原定理的全部模型假设不可由“高R²”替代。

1. **LeJEPA总体可辨识与模态条件：** Klindt, LeCun, Balestriero, *When Does LeJEPA Learn a World Model?*，[作者v2全文](https://arxiv.org/html/2605.26379v2)。本文不重命名其正交可辨识、Hermite模态或规划保证为自己的成果。
2. **有限轨迹、慢源、谱间隔近邻：** Sprekeler, Zito, Wiskott, *An Extension of Slow Feature Analysis for Nonlinear Blind Source Separation*, JMLR 2014，[原文](https://jmlr.org/papers/volume15/sprekeler14a/sprekeler14a.pdf)。本课题的同窗具名top-k终点更具体，不意味着其基础机制无人研究。
3. **Hanson–Wright：** Rudelson, Vershynin, 2013，[作者全文](https://www.math.uci.edu/~rvershyn/papers/rv-Hanson-Wright.pdf)。本文针对Gaussian特例直接推导MGF与常数，未声称新的普遍sub-Gaussian不等式。
4. **Davis–Kahan总体谱差版本：** Yu, Wang, Samworth, 2015，[作者全文](https://arxiv.org/pdf/1405.0680)。Theorem 5的逐轴及簇界应用该成熟工具。
5. **精确二项置信区间：** Clopper, Pearson, 1934，[原文扫描](https://barestatistics.nl/uploads/1/1/7/9/11797954/clopper__pearson_1934.pdf)。Bonferroni为并集界应用；本文P2给出独立接受样本下的完整反演证明。
6. **选择性风险控制先例：** Geifman, El-Yaniv, NeurIPS 2017，[正式出版页](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)。置信分数、拒答及二项风险认证已有先例，不包装成本文新算法。

### 5.4 交付与完整性

主文档：`docs/THEORY_FORMALIZATION_20261116.md`。只读解析脚本：`scripts/low_dim_ar1/verify_theory_moments.py`。机器可读/表格/核对输出：`results/theory_checks_20261116/{SUMMARY.json,RESULTS.md,VALIDATION.json}`。

保护目录为 `reports/`、`work/hcf_stable_id_20261027/`、`archive/`。本轮开始前逐文件SHA256快照含47772个文件、33775461684字节；结束后逐文件重新计算并比较，同时检查新增/删除路径。结果、前快照及Git前后状态保存在同一输出目录的完整性文件中。Git开始时已有大量治理变更，保留原状；不把它们记成本轮修改。完整性核对证明本轮字节保持，不替代历史实验的科学复核。
