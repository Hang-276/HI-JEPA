# Stage A+ 最邻近全文复核与理论路线反审

日期：2026-11-04。范围：继承最新 Proposal、Stage A 主报告与理论/辨识附件后，复核真正影响贡献边界的原始论文正文、定理和证明。没有训练、改模型或覆盖历史结果。下述“未见”仅指明确列出的全文与定理，不是对全部文献的不存在证明。

## 1. 独立结论

**把条件辨识强度 κ 接到轴误差，再接标准化慢度分数与 top-k 间隔 γ，作为一般证明链，本身不足以成为新的 ICML 理论贡献。** Gaussian likelihood 对角化对比、pairwise 动态/方差签名、轴误差随签名分离退化、同签名块内旋转，以及慢度特征的排序和有限采样局限，均有直接先行研究。

本轮真正增加了三处应收紧 Stage A 贡献措辞的证据：

1. FAR-JEPA 不只有“过程签名”的宽泛想法；其 Eq.(13) 已定义一个 pairwise population L² process gap，Theorem 2 和 Appendix A.2 已处理不同签名轴与相同签名块。[FAR-JEPA 全文](https://www.preprints.org/manuscript/202607.2043)
2. 经典 approximate joint diagonalization 已使用与我们 profiled conditional Gaussian 相同的 log-det contrast，并分析其 pairwise variance-ratio 敏感性。二维旋转的闭式 likelihood 曲率属于这一路线的专门化。[Afsari 作者稿 §3.2.3、§4.3](https://isr.umd.edu/Labs/ISL/ICA2006/Sensitivity_Final.pdf)
3. SFA 与非线性 BSS 的结合并非近期空白；ISFA 2007 与 xSFA 2014 需要纳入真正近邻，而非只引用 SFA 2002 的目标函数。[ISFA](https://www.ini.rub.de/PEOPLE/wiskott/Publications/BlasZitoWisk2007-ISFA-NeurComp.pdf)、[xSFA](https://jmlr.org/papers/volume15/sprekeler14a/sprekeler14a.pdf)

最有希望的增量仍是 **面向指定物理目标的 risk-to-selection 桥，明确耦合状态几何使慢度对轴误差出现一阶或二阶敏感性，并以此预测何时原 norm 证书失效**。这是一条可继续研究的候选，尚未证明它在文献层面足够新。不能将“找到一个闭式公式”直接改写为创新成立。

## 2. 关键全文、版本与精确覆盖

### 2.1 FAR-JEPA：最直接的近期过程签名近邻

**Wei 等，FAR-JEPA: Factorized Autoregressive Process Laws for Resolving the Orthogonal Gauge in Joint Embeddings。** 当前页面为 v1，2026-07-28 公开，未同行评审。已检查 §3–5、Appendix A–B 与 §7.2–8，而非仅摘要。[原文](https://www.preprints.org/manuscript/202607.2043)

它假设共享表示已属于 \(Qz,Q\in O(d)\)，源过程为逐轴 diagonal Gaussian AR(p)。§5 Eq.(13) 的过程间隔汇总 retained-lag 系数差与 innovation-variance 差；Theorem 2 用共同对角化得到 signed permutation。Appendix A.2 明确同签名块内任意正交变换与块投影。它没有证明任意端到端 encoder 获得所假设的 orthogonal class，也没有给当前声明物理慢集合的选择保证。当前 Fine 的 dense 均值、完整观测 likelihood 与该起点不同，不能互借保证。

### 2.2 GIN：Gaussian 逐轴 affine 的专门证明已经存在

**Sorrenson、Rother、Köthe，Disentanglement by Nonlinear ICA with General Incompressible-flow Networks，ICLR 2020。** 已读 §3.1–3.3、Appendix A Theorem 1 与 Appendix B Eq.(17)–(27)。[全文](https://arxiv.org/pdf/2001.04872)

其充分统计量关系来自满秩辅助自然参数变化；Gaussian 情况取 \((z,z^2)\)，通过平方后的四次项与跨坐标项限制，得到单因素 affine 对应。这与 Stage A 的平方统计量证明实质相同。我们对当前 \((x_t,a_t,u_t)\) 条件密度的适配可作为假设核查；不能将 polynomial closure 论证称作新辨识定理。GIN 的 volume-preserving 实现也不意味着当前一般 flow 必须采用它。

### 2.3 iVDFM：需区分模型差异与证明可借用性

**Chang、Kim，Conditionally Identifiable Latent Representation for Multivariate Time Series with Structural Dynamics。** arXiv v1 为 2026-03-24，当前 v2 为 2026-10-18；另有 ProbML 2026、PMLR 327:189–213 会议记录。已读 §3.1–3.2、Lemma 2、Theorem 3、Appendix A.3 与 B。[版本页](https://arxiv.org/abs/2603.22886)、[v2 全文](https://arxiv.org/html/2603.22886v2)、[会议记录](https://proceedings.mlr.press/v327/chang26a.html)

正文将辨识放在 conditional non-Gaussian innovation，再以 diagonal dynamics 向状态传播；这与 HCF 的 Gaussian kick、dense nonlinear conditional mean 不同。其 Gaussian 不辨识附录允许一般 covariance/dynamics 重参数化，不能据此否定固定 diagonal conditional family 的辅助辨识。其传播论证也不能在未检查滤波器关系时直接借给本研究；下面给出独立的数学审查理由。

### 2.4 Afsari 2008：Gaussian likelihood curvature 的直接经典来源

**Sensitivity Analysis for the Problem of Matrix Joint Diagonalization，SIAM Journal on Matrix Analysis and Applications 30(3):1148–1171，2008。** 已核对 Definition 2.1–2.2、Theorem 2.3、§3.2.3 Eq.(3.11) 和 §4.3 Eq.(4.17)–(4.21)。[正式记录](https://doi.org/10.1137/060655997)、[作者全文](https://isr.umd.edu/Labs/ISL/ICA2006/Sensitivity_Final.pdf)

其 Gaussian nonstationary log-likelihood contrast 是各矩阵的 \(\log[\det\operatorname{diag}(BCB^\top)/\det(BCB^\top)]\)。唯一性模量、方差比及 diagonalizer 条件数影响扰动敏感性已有明确分析。我们的二维 \(\sin^2(2\theta)\) risk 公式可有解释价值，但应标注为该 contrast 的独立专门推导。

### 2.5 SOBI 2016：pairwise 签名差到轴误差已有渐近公式

**Miettinen 等，Separation of Uncorrelated Stationary Time Series Using Autocovariance Matrices，Journal of Time Series Analysis 37(3):337–354，2016。** 正式 DOI 为 10.1111/jtsa.12159；所读作者预印本 Theorem 1(ii)、Theorem 2 给 symmetric SOBI 的渐近分析。[全文](https://arxiv.org/pdf/1405.3388)、[正式记录](https://doi.org/10.1111/jtsa.12159)

Theorem 1(ii) 的 off-diagonal unmixing 误差一阶展开，其分母正是多 lag 签名差平方和。它要求 stationary SOS 与相应样本矩收敛条件，不能直接成为 HCF flow 的有限样本学习保证；但足以排除“首次发现 κ 控制轴误差”的宽泛说法。

### 2.6 SFA、ISFA、xSFA：慢变量与源恢复的历史边界

**Wiskott、Sejnowski，SFA，Neural Computation 2002。** Eq.(2.1)–(2.4) 与 §3 已在标准化、去相关条件下最小化时间增量/导数能量，有限函数类转化为矩阵特征问题。[全文](https://papers.cnl.salk.edu/PDFs/Slow%20Feature%20Analysis_%20Unsupervised%20Learning%20of%20Invariances%202002-3430.pdf)

**Blaschke、Zito、Wiskott，ISFA，Neural Computation 19(4):994–1021，2007。** §4.1 将 second-order ICA 与 SFA 结合，已经研究利用慢度补充非线性源分离。它不承诺当前任意 coupled Gaussian-state 的严格 affine 物理恢复。[作者全文](https://www.ini.rub.de/PEOPLE/wiskott/Publications/BlasZitoWisk2007-ISFA-NeurComp.pdf)

**Sprekeler、Zito、Wiskott，An Extension of Slow Feature Analysis for Nonlinear Blind Source Separation，JMLR 15:921–947，2014。** Theorem 1、§3.3–3.4、§5.1–5.3 已研究独立源、理想函数空间下的谐波解、单调恢复，以及采样时长、相似慢度与采样频率的限制。相关理论依赖平稳、连续可微等结构，不等同于受控 FHN，但“慢度接近导致混合与采样困难”不能声称全新。[全文](https://jmlr.org/papers/volume15/sprekeler14a/sprekeler14a.pdf)

### 2.7 块辨识与子空间稳定：已有专门数值理论

**Cai、Ren-Cang Li，Perturbation Analysis for Matrix Joint Block Diagonalization，LAA 581:163–197，2019。** 所读作者预印本 Theorem 3.3、Corollary 3.4，使用 uniqueness/non-divisibility 模量控制块 diagonalizer 扰动。[作者全文](https://arxiv.org/pdf/1703.00591)、[正式 DOI](https://doi.org/10.1016/j.laa.2019.07.012)

**Cai、Ping Li，Identification of Matrix Joint Block Diagonalization，AISTATS 2021。** Theorem 2.4 给块唯一性条件，Theorem 2.6/2.8 给相应算法保证。[会议页](https://proceedings.mlr.press/v130/cai21a.html)、[全文](https://proceedings.mlr.press/v130/cai21a/cai21a.pdf)

因此，“只需跨块而非块内签名分离”和“块投影可能比个体轴稳定”具有强先行基础。把块进一步关联到指定任务目标可以是一个 specialization；不能把研究对象从轴改成块来规避当前严格可靠性标准。

## 3. 两处不能直接借用的推理：独立数学反审

### 3.1 Gaussian closure 不是当前模型族 closure

**Proved。** 对角协方差 \(D=\operatorname{diag}(1,4)\) 经 45° 旋转变为对角元 2.5、非对角元绝对值 1.5 的矩阵。它仍是 Gaussian，却不再是 diagonal Gaussian。类似地，旋转 diagonal transition 会产生 dense transition。一般 Gaussian/LDS 族的旋转等价，只有在变换仍留在给定候选族内才构成当前不可辨识反例。

这是对采用附录 closure 论证时所需条件的限制，不是凭一个例子否定 iVDFM 的全部研究。它也说明不能仅因 HCF 用 Gaussian 就给出整体 NO-GO。

### 3.2 创新逐轴映射不自动给当前状态逐轴映射

**Proved as a counterexample to the standalone propagation implication。** 令同一标量创新序列具有连续非退化独立分布，

\[
f_t=\eta_t+a\eta_{t-1},\qquad \tilde f_t=\eta_t+b\eta_{t-1},\quad a\ne b.
\]

创新之间是 identity 对应，但 \((f_t,\tilde f_t)\) 是从 \((\eta_t,\eta_{t-1})\) 的非奇异线性变换，联合分布在二维区域有密度。因此不存在仅依赖当前 \(f_t\) 的确定性函数给出 \(\tilde f_t\)。两个模型都逐分量依赖创新历史，不足以独自推出当前状态的逐分量 affine 对应。

要把传播步骤用于完整 observational-equivalence 定理，还需用相同观测律、时间局部可逆 decoder、滤波器/参数约束等排除上述自由度。此例不自动构成对完整 iVDFM Theorem 3 的反例，因为它没有同时证明两套完整模型观测律相同；它指出该局部推理不能跳步借用。

## 4. Overlap Matrix

| HCF 拟议内容 | 最直接先行覆盖 | 当前可保留的增量 | 判断 |
|---|---|---|---|
| Gaussian 自然参数满秩得到轴 affine | GIN Appendix A–B | 针对真实条件头、support、context 的适配审计 | 不是新主定理 |
| 条件方差/动态签名排除旋转 | FAR-JEPA Eq.13、Theorem 2；经典 JD | dense-mean HCF 与 diagonal-AR 的区别 | 机制已有 |
| profile likelihood 的二维旋转曲率 | Afsari §3.2.3/§4.3 | 当前归一化与参数强度的闭式常数 | 专门化推导 |
| κpair 小使轴估计敏感 | SOBI Theorem 1(ii)；JD sensitivity | 任意 flow 的真正 excess-risk-to-axis 仍未有 | 宽泛发现已有 |
| 同签名块内转动、块间可识别 | FAR Appendix A.2；joint block diagonalization | 指定物理目标是否在等价群下 well-defined | 候选任务化结论 |
| 标准化慢度找源/慢因素 | SFA 2002、ISFA 2007、xSFA 2014 | 先辨识再固定选轴的受限接口 | 组合风险明确 |
| gap 控制排序/谱稳定 | 普通 margin 与经典谱扰动；xSFA 采样分析 | 密度 risk 与目标 gap 的紧连接及失败预测 | 有限候选空间 |
| 不恢复全部轴仍恢复目标子空间 | 块辨识、SFA 子空间理论 | 明确 full-axis reliability 与任务子目标不同 | 不能降低原标准 |
| 有耦合时 score 对 axis error 一阶敏感 | Rayleigh quotient 的方向导数结构 | 当前 risk-to-declared-selection exact bridge 与可观测失败边界 | 最值得继续；新颖未确认 |

## 5. 对主代理二维 exact bridge 的独立判断

本节为本轮独立数学推导与检查，不是上述论文的结果摘要。

令两个真因素已按 \(Q_0\) 标准化，状态 covariance 与多 horizon 增量二阶矩为

\[
\Sigma=\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix},\quad
M=\begin{pmatrix}s_1&c\\c&s_2\end{pmatrix},\quad
\gamma=s_2-s_1>0,\quad|\rho|<1.
\]

对两个因素作 \(Q_\theta\) 旋转，再各自用真实方差重新标准化。直接展开两个 Rayleigh 商得到

\[
g(\theta)=\frac{\gamma\cos2\theta-\chi\sin2\theta}
{1-\rho^2\sin^2 2\theta},\qquad
\chi=2c-\rho(s_1+s_2).
\]

取与正确 signed-permutation 对齐的局部支 \(|\theta|\le\pi/4\)。当 conditional innovation covariance 为对角、均值可正确 profile 且表示严格限制在该旋转族时，Gaussian likelihood excess 可以写为

\[
F(z)=\tfrac12\mathbb E_u\log(1+a_u z),\qquad
z=\sin^2 2\theta,\quad a_u\ge0.
\]

只要 \(\kappa_{\rm barrier}=F(1)>0\)，凹性和 \(F(0)=0\) 给 \(F(z)\ge zF(1)\)。若真实总体 excess risk≤r，则

\[
z\le r/\kappa_{\rm barrier}.
\]

这明确区分了 barrier、局部曲率 \(F'(0)\)、自然参数 rank 奇异值和学习器的参数估计误差；四者不能同名为 κ。

**Proved under the restricted assumptions。** 若单轴 score 估计误差≤ε，令 \(\bar z\le1\) 控制 z，则

\[
\gamma\sqrt{1-\bar z}>|\chi|\sqrt{\bar z}+2\varepsilon
\]

足以使此对轴保持正确排序。分母正且≤1，因此该条件是安全的充分界。精确界可以进一步保留分母，而非在此损失常数。

**新意应当放在哪里。** 当χ=0时，score gap的局部退化为二阶；χ≠0时可出现一阶退化，因为 \(g'(0)=-2\chi\)。这解释为何只有 axis norm error 和 γ 的最坏界可能极松。可是 χ 也是 normalized Rayleigh 商的普通方向导数；F又是经典JD Gaussian contrast。故“首次写出χ”并非充分的创新论证。可争取的是：证明只有κ、γ无法统一决定选择可靠性，给出不可省略的几何量/必要性或相匹配的上下界，并验证它能预测尚未使用来调节结论的失败。

**不能越界的地方。** 当前 Fine 是任意 nonlinear flow；它不因 dense4 接近1就等于 \(Q_\theta s\)。从学到的 nonlinear residual 到此旋转族还需定量 remainder。经验 CAL NLL 不是这里的总体 profiled risk。一个跨边界二维旋转可解释局部失败机制，却不能自动覆盖 d=4 中多个非交换旋转、匹配变化与所有目标/非目标比较。若没有这些桥，必须保留 restricted theorem 标签。

## 6. 最值得推进的三条路线与停止门

### R1：耦合几何下的目标导向 risk-to-selection（首选，CONDITIONAL GO）

将上节作为可证明起点，主张聚焦“辨识风险相同但目标 score 受到不同方向影响”，而不是重新宣告 κ 控制轴误差。最有价值的新增结果应至少包括：

- 一个受合法动态生成模型实现的成对构造：同κ、同γ而χ不同，选择鲁棒半径不同；不能只挑无法由生成器产生的任意矩阵。
- 对正确 target ordering 的近必要/充分 risk 阈值，或至少说明一阶与二阶依赖差异无法由一般 norm 界消除。
- 明确处理 covariance 重新标准化、四维跨边界 pair 和 nonlinear remainder，或诚实把结论限定在受限族。
- 标签可观测的事后分析与无标签可计算量分开；源数据已经暴露的FHN不能充当独立确认。

若只能得到上述代数式、在旧12父上界变小，而没有新失败预测或必要性结果，应把它作为机制分析附件；不能据此宣布 ICML 理论路线 GO。

### R2：目标对等价群的可辨识性与轴标准（次选，理论澄清）

研究 “S*是否为同签名块的并集”。若等价群可在一个块内任意旋转，而目标切开该块，则目标子空间本身也不是群不变量；若完整选入该块，子空间可以保持，但每条物理轴仍不唯一。这一结论可直接证明，却与 FAR/块辨识高度相邻。

它适合回答为什么 κ_all 很小不必妨碍某个子空间用途；不适合用来降低当前 full-axis 联合门。只有若形成任务级部分辨识/不可识别区域的非平凡统计结论，才值得升级为主贡献。否则保留为必要定义与负控解释。

### R3：Fine相对raw同规则的独立价值（科学证据路线，不预设新模型）

精确定义拟争取的是恢复精度、扰动稳定性、非线性观测适用范围还是必要性，再与 raw＋相同后处理比较。现有 raw G4 已通过，故“Fine必要”不能作为已知结论。即使理论R1成立，它也没有单独证明 Fine 优于简单动态分析。

若这条路线不能找到独立价值，而理论又只剩经典组合，继续扩大GPU训练无法修复论文核心。此时应给当前固定组合的ICML方法创新路线 NO-GO；可保留问题导向的理论/诊断研究价值。

## 7. 对研究决策的具体建议

**本轮文献与独立反审支持继续有界推进 R1，但不支持开始4090正式baseline。** 最低 GO 条件应同时满足：

1. 至少一个核心结论超出“GIN辨识＋经典JD曲率＋普通排序”的直接拼接，并说明与FAR、SOBI、xSFA、块扰动理论的差别。
2. 对该结论给清楚的受限模型、非空量级与可证伪失败预测；实际 nonlinear Fine 的结论不得自动继承。
3. Fine的独立价值有明确、未被raw对照否定的可检验主张，完整预测分支不代替慢语义证据。
4. 严格逐轴 reliability 与子空间目标始终分开，不因理论路线方便而改变研究成功标准。

这里的“未证新颖”不是该公式已被某一篇论文逐字覆盖的断言；它表示本次找到的原始全文已经覆盖其大部分数学工具与宽泛结论，当前仍缺少足以排除“专门化推论”评审意见的实质增量。

