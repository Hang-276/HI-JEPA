# 无物理真值的接口：可观测自洽、exact posterior 与物理轴

2026-11-04。范围见 [预定scope](OBSERVABLE_CERTIFICATE_SCOPE.md)，现有记录见 [observability ledger](OBSERVABILITY_LEDGER.md)。本轮只读源码、JSON metadata和既有报告；没有读取数组成员或checkpoint，没有模型前向、拟合、采样或新实验。

**存在非空的替代接口，但当前记录没有给实际Fine认证。** 一条接口把真实条件regime posterior正确性与Gaussian充分统计量的满秩变化连接起来，可绕开真实A、D和非线性余项r；另一条是完整观测相容集合的表示直径。前者的exact总体逻辑在下面证明，有限近似posterior到轴误差/多时距G0余项的定量桥尚未闭合。现有残差均值、方差、Pearson与CAL NLL都不能补上这一步。

## 1. 三个不同的认证对象

1. **观测自洽**：一个已学习head能否解释观测到的条件响应或regime标签。输入可仅用x、a、u和Fine/head，不需要物理标签。
2. **物理轴等价类**：Fine是否等于真实因素的逐轴非零仿射变换和置换。必须增加真实生成类、支持和variation等结构假设；这些不是观测矩检验的结论。
3. **预声明目标的G0选集**：即使物理轴已恢复，目标还须等于相同Q0/Qh定义下唯一最慢的因素集合，并控制统计误差、支持迁移和FP32排序。轴辨识并不消除旧FHN oracle支持中目标与慢度排序不一致的反例，见 [既有pair审计](../PAIRED_SCORE_AUDIT.md)。

本文件给第1类的可实施统计对象、一个从第1类到第2类的exact条件定理，并说明第3类还缺什么。它不改变实际Fine、G0、fullFine预测通路或原目标。

## 2. 用现有head定义可观测regime posterior

令C=(X_t,A_t)，X'=X_{t+1}，U为驱动这次transition的regime。Fine是同一个、与context/regime无关的可逆映射E，令Y'=E(X')。已学习head给

\[
q_u^Y(y\mid c)=\prod_{j=1}^d\mathcal N(y_j;\mu_{ju}(c),\sigma_{ju}^2(c)).
\]

对正确且严格正的prior/propensity \(\pi_u(c)=P(U=u\mid C=c)>0\)，定义

\[
\widehat\pi_u(c,x')=
\frac{\pi_u(c)q_u^Y(E(x')\mid c)}
{\sum_v\pi_v(c)q_v^Y(E(x')\mid c)}.
\tag{O1}
\]

完整观测密度是\(q_u^X(x'\mid c)=q_u^Y(E(x')\mid c)|\det DE(x')|\)。相同未来encoder的Jacobian在O1中抵消，所以O1不需要真实inverse、真实D或真实A；这不是丢弃change-of-variables后比较绝对NLL。在当前外生iid uniform九regime设计中，正确prior为1/9；这项设计事实来自拥有源码/配置，不由有限频率“恰好接近1/9”证明。换到其他数据必须重新说明propensity。

当前head源码为\(\mu_u(c)=m(c)+\beta_u\)，其中m由当前Fine与action经线性项和MLP产生，\(\sigma_u\)仅依赖u。O1是这个head可定义的函数；**本轮没有计算任何O1值，也没有加载其embedding权重**。

多类log-loss是严格proper scoring rule：其相对真实Bayes posterior的总体excess等于\(E\,KL(\pi(\cdot\mid C,X')\|\widehat\pi(\cdot\mid C,X'))\)。实际平均log-loss本身不是这个excess，未知Bayes条件熵不能被当作0；高分类准确率亦不能推出posterior相等。

## 3. Exact posterior到axis-affine：完整平方闭包证明

这是TCL/GCL的posterior-odds/密度比原理在Gaussian head上的条件性专门化，**不作新总体辨识定理或新算法的创新声称**。原理近邻包括 [TCL 2016](https://papers.nips.cc/paper_files/paper/2016/hash/d305281faf947ca7acade9ad5c8c818c-Abstract.html)、[GCL 2019](https://proceedings.mlr.press/v89/hyvarinen19a.html)；与iVAE、GIN的关系见 [primary近邻核查](PRIMARY_NEIGHBORS.md)。下面直接给足本结论需要的证明，不机械套一般nonlinear ICA variation条件。

### 3.1 前提与结论

设U有2d+1个值0,…,2d（当前d=4、九regime）。假设：

- **真实结构包含性**：存在与context/regime无关的光滑双射观测混合f，X'=f(S')；对几乎每个c和所有u，\(S'\mid C=c,U=u\)是具有严格正对角方差、全支撑\(\mathbb R^d\)的Gaussian。均值可以由耦合动力学产生，不要求当前因素独立。这是关于真实机制的结构前提，不是unlabeled样本自动给出的结论。
- **合法候选表示/head**：E是相同支持上的context/regime无关光滑双射；上述Gaussian head方差严格正，相关函数可测。令g=E∘f。
- **正确正prior**：O1使用真实\(\pi_u(c)>0\)。
- **真实条件posterior逐点/几乎处处正确**：
  \(\widehat\pi_u(C,X')=P(U=u\mid C,X')\)，对联合观测律几乎处处、每个u成立。不是普通概率校准、ECE小、bin calibration或分类决策正确。
- **合法context上的满秩**：存在一个具有正\(P_C\)测度的context集合\(\mathcal C_R\)，其中learned Gaussian自然参数contrast矩阵\(\widehat W(c)\)满秩2d。等价地，可以直接要求它在一个属于上述posterior条件正确的满测集合的合法c上满秩。不能对任意零概率c₀指定regular conditional的版本，再以该点经验rank启动证明。

结论：存在置换p、常数\(a_j\ne0,b_j\)，使

\[
g_j(s)=a_js_{p(j)}+b_j\quad\text{对所有 }s\in\mathbb R^d.
\tag{O2}
\]

若只有局部共同开支持，以下多项式论证只保证该支持上的结论，不能外推到未覆盖输入。

### 3.2 Posterior odds给同一context的密度比恒等式

由disintegration/Fubini，posterior联合a.e正确意味着存在\(P_C\)满测集合，在其中每个c上，posterior等式对\(P_{X'|c}\)几乎处处成立。它与正测的\(\mathcal C_R\)相交，因此可选一个同时满足条件正确性和rank的c。之后全程固定此c，不拼接不同context的任意行。

对u=1,…,2d，与u=0取odds并扣除正确prior，Bayes公式给

\[
\log\frac{q_u^Y(g(s)\mid c)}{q_0^Y(g(s)\mid c)}
=\log\frac{p_u^S(s\mid c)}{p_0^S(s\mid c)}.
\tag{O3}
\]

真实/learned观测Jacobian各自在自己跨regime的比值中消失。真实条件Gaussian密度处处正、prior正，故这里的a.e集合可取为Lebesgue满测集合；不存在由零密度洞隐藏的任意函数部分。

令

\[
T(z)=(z_1,\ldots,z_d,z_1^2,\ldots,z_d^2)^\top,
\quad \eta_u=(\mu_{1u}/\sigma_{1u}^2,\ldots,\mu_{du}/\sigma_{du}^2,
-1/(2\sigma_{1u}^2),\ldots,-1/(2\sigma_{du}^2)).
\]

把\(\widehat\eta_u-\widehat\eta_0\)与真实\(\eta_u^S-\eta_0^S\)分别作为矩阵\(\widehat W,W_S\)的行，O3同时成为

\[
\widehat W\,T(g(s))+\widehat k=W_ST(s)+k_S.
\tag{O4}
\]

由于\(\widehat W\)可逆，

\[
T(g(s))=L\,T(s)+b,\qquad L=\widehat W^{-1}W_S.
\tag{O5}
\]

O5两边是连续函数；连续函数若在全支撑律下a.e相等，则处处相等。因此可以把O5作为全\(\mathbb R^d\)上的函数恒等式，而不在经验点之间擅自插值。

### 3.3 平方闭包消除非线性，再消除dense混合

O5的前d行说明每个g_j是“常数＋一次项＋逐轴平方项”的至多二次多项式：

\[
g_j(s)=b_j+\sum_i a_{ji}s_i+\sum_i q_{ji}s_i^2.
\]

后d行又说明\(g_j(s)^2\)仍为同一类至多二次多项式。若其二次齐次部分\(q_j(s)=\sum_i q_{ji}s_i^2\)非零，则\(g_j^2\)的四次齐次部分为非零多项式\(q_j(s)^2\)，不可能与至多二次右侧相等。因此所有q_ji=0，g_j必须affine。

再平方affine表达式。每个\(s_i s_k\)（i≠k）的系数是\(2a_{ji}a_{jk}\)，而O5右侧没有交叉单项式，所以\(a_{ji}a_{jk}=0\)。每行a_j最多一个非零元素。g是双射，故affine矩阵必须可逆：每行恰好一个非零元素、各行选择不同列。这恰为置换乘非零对角缩放，得到O2。没有把允许的匹配放宽为dense affine，也没有借拟合真状态来证明轴恢复。

### 3.4 对当前mean＋variance设计是否非空

当前拥有源码包含两组contrast：u1…4分别移动一轴均值；u5…8分别使一轴std乘3，即variance ratio为9。AR mean shift=.6、base std=.12；FHN mean_contrast=.035、base std=.025、contrast_strength=1且rotational=false。FHN字段variance_multiplier=3实际乘std，不能写为variance ratio3。详见ledger的拥有源码定位。

固定任意合法c，设base真实均值为m(c)、base方差d_j>0、mean shift为δ_j≠0。按mean contrasts、variance contrasts排序，真实自然参数矩阵为分块三角形式

\[
W_S(c)=\begin{bmatrix}
\operatorname{diag}(\delta_j/d_j)&0\\
\operatorname{diag}(m_j(c)[1/(9d_j)-1/d_j])&
\operatorname{diag}(-\tfrac12[1/(9d_j)-1/d_j])
\end{bmatrix}.
\tag{O6}
\]

两个对角块均非奇异，故完整自然参数rank为2d。**这只是已知合成机制下的结构兼容性证明，不是actual learned head的rank或posterior检验。** 当前有限flow/mean-MLP是否精确包含真实inverse/条件均值亦未证明。可在较宽的合法函数类中用真实inverse和真实head构造满足定理的例子，说明接口逻辑非空；不能因此断言当前实际网络已达到该例子。

全支撑连续Gaussian与光滑双射指生成数学的理想化。实际learner将观测存为FP32，严格的连续密度/双射前提不能按量化后的字节分布直接当作事实；将理想定理用于数值实现仍须控制相应近似，不能由存储格式自动忽略它。

当前learned head还有一个可观测的简化：设\(\lambda_{ju}=\sigma_{ju}^{-2}\)、\(\mu_{ju}(c)=m_j(c)+\beta_{ju}\)，contrast第一块为\(m_j(c)\Delta\lambda_{ju}+\Delta(\beta_{ju}\lambda_{ju})\)，第二块为\(-\frac12\Delta\lambda_{ju}\)。对每个j，把第一块列加\(2m_j(c)\)倍第二块列，得到与c无关的\(\Delta(\beta_{ju}\lambda_{ju})\)。所以这个特定head的rank可仅由两张regime embedding表决定，无需挑选有利context。本轮只读metadata，没有读取这些权重，actual rank仍为**未核**。

### 3.5 “校准”一词不能偷偷缩弱前提

常数预测器\(\widehat\pi(u)=P(U=u)\)可具有完全正确的普通classwise/bin校准，却忽略 informative的(C,X')。故ECE小、有限reliability bins、总体类别比例正确都不能替代本定理的真实posterior a.e正确。分类accuracy只约束决策边界，也不足够。

需要区分一个更强、不同的说法：对\(\widehat W(c)\)满秩的合法c，完整posterior向量的log-odds可逆地确定T(E(x'))，进而确定x'。因此在这些c上，**给定C和完整预测向量的exact conditional vector calibration**，\(P(U\mid C,\widehat\pi(C,X'))=\widehat\pi(C,X')\)，可推出真实posterior正确。若要声明联合观测σ代数的全局等价，须要求rank对几乎处处c成立；仅正概率的rank集合，则只在该集合上使用上述等价，已足够选取一个合法context启动平方闭包证明。这不是一般ECE/bin检验，当前也没有验证它；本定理仍直接使用posterior a.e正确的清晰表述。

### 3.6 近似版本目前没有闭合

O4若有误差e(s)，首先只得到充分统计量误差\(\widehat W^{-1}e\)。从有限posterior风险或有限witness误差到e，还需要稀有类别、posterior接近0、尾部和log-odds的控制；从近似平方闭包到接近置换affine，需要最小奇异值、支持/矩条件、平滑或多项式稳定性常数；再到实际多时距G0，还需Q0/Qh端点支持与误差传播。

这些常数未由现有metadata记录或证明，不能写成“posterior看起来好，所以Round3余项小”。将exact假设放松为小ECE、小NLL、残差zeroCorr或未定量的“近似正确”，均不能直接保留O2或选集认证。

## 4. 可落地的有限统计对象：独立episode上的posterior witnesses

对在audit数据之前固定的M个候选，预定有限有界witness\(\phi_k(C,X')\in[-1,1]\)，并把regime类别计入k，定义

\[
Z_{m,k,u}=\phi_k(C,X')\{\mathbf1(U=u)-\widehat\pi_{m,u}(C,X')\}\in[-1,1].
\tag{O7}
\]

对全部有界可测φ都满足EZ=0，当且仅当真实posterior a.e正确；有限K个witness只约束这K个条件矩，不能直接认证全部posterior。可选context/未来Fine的非线性、Fourier或分区witness，但必须预先冻结，不能看结果后挑容易通过者。它们使用的都是观测量，不需要真物理余项。

若有n条独立完整轨迹，每条轨迹先平均其合法transition上的Z，轨迹间独立、各轨迹均值仍在[-1,1]，则对预定M×K个统计量，以概率至少1−δ，

\[
\max_{m,k}|\widehat\mu_{mk}-\mu_{mk}|
\le\sqrt{\frac{2\log(2MK/\delta)}{n}}.
\tag{O8}
\]

K已包含被检验regime数；若单独写类别，再将9乘进union bound。轨迹长度不同且目标是按transition加权时，必须另写权重和相应集中界；O8针对等episode权重，不能暗换estimand。轨迹内时间相关被允许；2048个抽取transition不是2048个独立样本。现有CAL包含AR32/FHN24条轨迹，这只是来源数，不是本轮计算过置信界。

CAL已经选checkpoint且参与历史方法判断。固定七个TRAIN-only快照的统一覆盖在严格的事先候选/测试族冻结下可覆盖checkpoint选择，但不能自动覆盖整个历史的adaptive开发。新audit split或严格交叉拟合可以处理部分数据复用；仍不解决模型包含、无限witness覆盖和近似辨识常数。本轮没有新增split、计算witness或启动此协议。

这提供一个真正可定义、可拒绝的无标签自洽检验，而非只说“再做实验”：某个合法witness明显不为0就拒绝exact posterior主张；全部有限witness未拒绝，结论仅为被测条件矩范围内相容，不能称物理轴接受式证书。

## 5. Learned-coordinate residual、Gaussianity与score/Jacobian

定义可观测head residual

\[
R_j=\frac{E_j(X')-\mu_{jU}(C)}{\sigma_{jU}(C)}.
\]

对每个u，若联合律满足\((C,R)\mid U=u\sim P_{C|u}\otimes N(0,I)\)，则head完整条件密度正确，反之亦然；经可逆change-of-variables可转回观测条件密度。它是比posterior ratios更强的自洽条件。可用预定有界joint/context witnesses检验有限投影；均值0、方差1、zeroCorr、逐轴正态边缘不等于此联合条件。

一个不需要采样的反例是\(p(y_1,y_2)=\varphi(y_1)\varphi(y_2)[1+\epsilon h(y_1)h(y_2)]\)，其中\(h(z)=\cos z-e^{-1/2}\)，\(0<|\epsilon|<(1+e^{-1/2})^{-2}\)。h是even且在标准正态下均值0；两个边缘都是N(0,1)，cross-covariance为0，但ε≠0时相互依赖。因此现有有限二阶摘要即使数值理想，也不蕴含joint Gaussian或conditional independence。

若还能独立得到真实**数据条件残差密度**q_R的score，并有正则性和有限relative Fisher information，则对标准Gaussian的log-Sobolev不等式逐(C,U)积分，得到

\[
E\,KL(q_R(\cdot\mid C,U)\|N(0,I))
\le\frac12E\|\nabla_r\log q_R(R\mid C,U)+R\|^2.
\tag{O9}
\]

这是一个观测条件密度misfit的上界接口；高斯LSI的\(D\le I/2\)形式可见 [Courtade 2016，Eq.10](https://arxiv.org/html/1608.05431)。右侧需要真实数据条件score或带受控误差的独立估计；把head自带score −R与自己比较得到0是循环论证。现有记录没有数据score估计、误差界或相关导数数据，本轮不拟合它。

Fine在观测坐标的Jacobian和logdet由架构可定义；它们不等于相对真实physical inverse的Dr。参数范数投影和logstd clamp可帮助后续推导函数界，但不能直接冒充已记录的全支持Jacobian、tail或余项证书。上述路线亦不能从小密度KL无条件跳到physical轴；仍须第3节或其他定量辨识条件。

## 6. 无需显式余项的另一接口：完整相容集合

本轮 [VERSION_SPACE_CERTIFICATE.md](VERSION_SPACE_CERTIFICATE.md) 给另一条条件性接受式路线：先用完整观测likelihood的有效统一置信界保留全部相容生成解释，再认证它们在同一观测输入上的、模逐轴affine/permutation的表示直径或score-gap共识。若真实物理解释被包含，集合直径便能控制Fine到真实轴的误差，无需计算真实A/r。

这条路线仍需精确truth containment、完整候选类覆盖、有效尾部/独立episode统计界、全局匹配及直径上界；当前三seed一致不是完整置信集覆盖。有限候选若出现同样可信却轴不一致的反例，可拒绝某项稳定主张；未找到反例不能当作完整集合上界。该文也明确：离散TRAIN经验支持上的距离不能自动覆盖CAL新点。

## 7. 当前可交付判断

| 接口 | 可观测输入 | 在完整前提下的输出 | 当前状态 |
|---|---|---|---|
| O7–O8有限posterior witnesses | x,a,u、固定Fine/head | 被测有界条件矩的置信范围/拒绝 | 合法对象已定义；无逐样本posterior记录，未执行 |
| Exact posterior＋rank＋平方闭包 | 同上，另加真实Gaussian结构假设 | 物理轴模逐轴affine/permutation | exact证明闭合，真实rich设计兼容；实际posterior/rank未核 |
| 条件Gaussian residual/score | learned residual、context、独立数据score | 条件观测密度自洽或KL界 | 仅有旧二阶摘要，无joint/score证据 |
| 完整version space | 完整候选密度与表示、统一置信界 | 真解释覆盖下的轴距离/选集条件保证 | 理论接口存在；全局覆盖、直径与统计条件未执行 |
| 已存G0 scores与gap | TRAIN Fine统计metadata | 当前存储分数/规则的一致性 | 有记录；不等于population或语义目标证书 |

因此，替代余项的路线并非空白，也不能把它记成已取得的actual Fine/G0可靠性。当前最明确的可观测拒绝接口是冻结posterior/head后检验独立episode上的预定条件矩；当前最明确的exact物理辨识接口是完整posterior＋Gaussian rank。将二者连接成非空有限误差接受证书仍缺定量稳定性，已有84份摘要不足以完成它。
