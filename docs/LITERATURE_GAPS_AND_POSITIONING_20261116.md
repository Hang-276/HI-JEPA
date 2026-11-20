# 从因素可辨识到可靠慢因素选取：文献核对与主结论备选

核对日期：2026 年 11 月 16 日。对应当前的两个缺口：①与已有研究区分贡献；②选择一条能够被证据支持的中心论断。

本文是文献调研和研究定位建议，不是论文初稿，也没有新增实验。我们的结果以[研究记录](RESEARCH_RECORD.md)及其中链接的原始报告为准。检索优先使用会议正式论文集、期刊网站、作者公开全文和官方会议录用目录。正文重点比较论文的研究对象、假设、定理和评价方式，不能仅凭标题相近认定重叠。

## 1. 先给判断

**课题可以继续，但贡献必须比“可辨识不等于可靠”“高 R² 也可能选错”“分差大时更可靠”更具体。** 这些一般认识已有相当充分的发表先例。把它们放到 LeJEPA 上重新观察，不会自动成为独立的方法或理论创新。

本轮找到的三项直接重叠尤其重要：

- Sprekeler、Zito、Wiskott 的 JMLR 2014 论文已经讨论慢源的高阶函数挤占其他源、有限轨迹导致分离误差、谱间隔和估计波动的竞争，以及采样速度的权衡。
- 选择性预测已有置信分数阈值、作答比例、作答后的错误率、精确二项上界和多项检验。我们 A3 的统计流程是这些工具的应用；无标签分位数调整覆盖率也已有先例。
- 《When Does LeJEPA Learn a World Model?》已核实进入 **NeurIPS 2026 官方录用目录**。其附录 F 已有我们使用的模态分离条件；不能继续把它当成仅供参考的未录用预印本。

**目前较值得保留的空间，是研究一个更具体的任务：因素已经可以读出以后，能否在同一段有限轨迹中，可靠地选出有物理名称的慢因素。** 这个任务不同于学习最慢的函数、恢复潜变量、估计长期平均排名，也不同于证明某个规划任务对坐标旋转不敏感。

但“任务不同”只是进入研究的理由，不能代替贡献。我们需要进一步交代：针对这个任务，得到了哪些既有结论不能直接给出的定量关系、干预证据或可复现的可靠性边界。当前最适合的是**条件与机制研究**；新校准算法的定位证据不足。

## 2. 文献证据怎么分级

| 级别 | 本文采用的判断标准 | 如何使用 |
|---|---|---|
| 已发表 | 能查到官方论文集或期刊卷期、DOI | 可以作为强重叠证据 |
| 已录用 | 官方会议目录、明确录用决定或正式 proceedings；不是只有投稿页 | 可以作为强重叠证据，注明录用状态 |
| 预印本 | 仅查到 arXiv、Preprints.org 或作者稿，没有核实正式录用 | 作为接近方向和潜在竞争的提醒，不据此直接否定我们的贡献 |
| 阅读受限 | 已确认发表，但此次只能获取摘要或较早作者版本 | 明确阅读版本，不把未核读的最终定理细节写成事实 |

“全文核读”在本文指已经打开全文，并查看与比较有关的定义、假设、定理及证明/实验章节；不是声称逐句复核了每篇的全部证明。少量背景论文只核对发表信息与摘要，另行标明。**已发表论文的作者稿放在 arXiv 上，不等于这篇论文仍是预印本。** 例如 Hall–Miller 的全文明确是期刊电子重印版。

检索覆盖五组：SFA／时序盲源分离，非线性 ICA／可辨识表示，谱表示与任务保证，选择性预测／风险控制，排名与部分可辨识。检索使用核心论文引用链和精确题名；这不是穷尽全世界文献的系统综述。“此次未发现直接覆盖”只表示当前核查范围内未找到，不等于证明无人研究。

## 3. 我们究竟要和别人比较什么

当前评价先恢复四个具名物理因素，再在一段 193 状态轨迹上，以 lag 4、16、32 的平均平方变化除以独立参考方差，选分数最低的两个因素。两个中任何一个错误，整个窗口就算错选。

如果物理轨迹和目标参考方差都精确已知，答案当然可以直接计算。研究缺口存在于数学可辨识与有限样本下的实际恢复之间，以及近似恢复经过评分、校准进入决策的过程中；不能把“精确知道物理量仍无法计算它的慢度”当作我们的主张。

必须分开以下五个目标：

| 目标 | 具体问题 | 与我们的区别 |
|---|---|---|
| 因素可辨识 | 观测能否确定潜变量，允许剩下哪些变换？ | 不直接给训练误差或当前窗口的答案 |
| 全局慢函数／慢子空间 | 哪些函数在长期平均意义下变化最慢？ | 最慢函数不一定是一组不同的物理因素 |
| 长期物理因素排名 | 哪些物理变量长期平均最慢？ | 有限窗口的实际排序可能不同 |
| 当前窗口物理慢集 | 此刻这段轨迹中，哪两个具名因素最慢？ | 这是我们目前直接评价的目标 |
| 预测或控制表现 | 用这些变量能否更好预测、规划或控制？ | 需要另做任务验证，不能从慢集正确率直接推出 |

还有一个容易混淆的地方：**证明“某个下游目标在允许的坐标变换下不变”，和证明“所有下游目标都能可靠完成”不是一回事。** 因此我们不应把原 LeJEPA 的旋转不变规划保证，解读为它已承诺逐轴物理慢度排序；也不应把我们的排序失败写成对该保证的反驳。

## 4. 已发表或已录用研究：逐类核对

### 4.1 SFA 与时序源分离：慢模态、物理因素和有限数据

**最重要近邻：Sprekeler、Zito、Wiskott，An Extension of Slow Feature Analysis for Nonlinear Blind Source Separation，JMLR 2014，15:921–947。** [正式全文](https://jmlr.org/papers/volume15/sprekeler14a/sprekeler14a.pdf)

核读 §3–5、Theorem 1 及相关附录。其独立连续源模型中，最优慢函数可由各源的谐波及其乘积组成；高斯情形对应 Hermite 多项式。某个慢源的高阶函数可能排在另一源的一阶函数之前，因此最慢的若干函数不必对应若干不同物理源。xSFA 逐次去掉已提取源的非线性函数，避免反复提取同一个源。

更直接的重叠在 §5：有限相关轨迹的有效样本少，谱差小于估计波动时难以分离；采样速度、谱拥挤和函数空间都会影响结果。文中的分辨率量级估计是启发分析，不能写成它已经证明一般非渐近概率界。其严格理论使用连续可微源与导数慢度，**没有直接给出我们的离散多 lag、固定新窗口、具名 top-2 风险保证**；但基础机制的重叠很强。

| 正式论文 | 核读位置与已有内容 | 与我们比较后的结论 |
|---|---|---|
| **Wiskott & Sejnowski，Slow Feature Analysis: Unsupervised Learning of Invariances，Neural Computation 2002，14:715–770。** [DOI](https://doi.org/10.1162/089976602317318938)／[作者全文](https://www.ini.rub.de/PEOPLE/wiskott/Reprints/WiskottSejnowski-2002-NeurComp-LearningInvariances.pdf) | §2–3，式2.1–2.4、3.15–3.18：最小化导数能量，约束零均值、单位方差、去相关；白化后作谱求解 | 慢度归一化和求慢特征本身已有；终点不是给定物理候选的窗口集合错误率 |
| **Blaschke、Zito、Wiskott，Independent Slow Feature Analysis and Nonlinear Blind Source Separation，Neural Computation 2007，19:994–1021。** [DOI](https://doi.org/10.1162/neco.2007.19.4.994)／[全文](https://www.ini.rub.de/PEOPLE/wiskott/Reprints/BlaschkeZitoEtAl-2007-NeurComp-ISFA.pdf) | §2、4、5.3–6：结合独立性与慢度；时间二阶去相关不足以保证独立；高阶累积量诊断分离 | 不能主张首次用额外统计量诊断分离失败；其诊断目标不是我们的慢集决策 |
| **Sprekeler & Wiskott，A Theory of Slow Feature Analysis for Transformation-Based Input Signals with an Application to Complex Cells，Neural Computation 2011，23:303–335。** [DOI](https://doi.org/10.1162/NECO_a_00072)／[全文](https://www.ini.rub.de/PEOPLE/wiskott/Reprints/SprekelerWiskott-2011-NeurComp-SFATheoryRFs.pdf) | §3.1、Theorems 1–2：变换驱动输入下的算子与慢函数特征方程 | 已有总体慢函数谱理论。不要把2008年 CogPrints 的另一标题误写成这篇正式论文 |
| **Belouchrani 等，A Blind Source Separation Technique Using Second-Order Statistics，IEEE TSP 1997，45:434–444（SOBI）。** [DOI](https://doi.org/10.1109/78.554307)／[原刊全文镜像](https://pzs.dstu.dp.ua/DataMining/ica/bibl/Belouchrani.pdf) | §II–III、Theorems 1–3：单 lag 分解、多 lag 联合对角化，源时间相关可区分条件 | CAL+NAME 的白化和滞后解混属于这条成熟路线；物理命名与窗口风险需另处理 |
| **Miettinen 等，Separation of Uncorrelated Stationary Time Series Using Autocovariance Matrices，JTSA 2016，37:337–354。** [期刊](https://onlinelibrary.wiley.com/doi/abs/10.1111/jtsa.12159)／[作者全文](https://arxiv.org/pdf/1405.3388) | 作者版 §2–5、Theorems 1–2：SOBI 一致性与极限分布，误差同时受自相关差、滞后协方差和白化误差影响 | “只看理想谱差不够”已有正式理论；不能把渐近源恢复界当短窗口决策界 |
| **Taskinen、Miettinen、Nordhausen，A More Efficient Second Order Blind Identification Method for Separation of Uncorrelated Stationary Time Series，Statistics & Probability Letters 2016，116:21–26（eSOBI）。** [期刊](https://doi.org/10.1016/j.spl.2016.04.007)／[作者全文](https://users.jyu.fi/~slahola/files/eSOBI.pdf) | §2.3–4：估计解混渐近方差，从预设 lag 集合选择效率更高的组合 | 更聪明地选择解混 lag 已有方法；B2 的简单平均不是其完整替代或反驳 |

### 4.2 谱扰动与动力学模态：哪些数学工具已有

| 正式论文 | 核读位置与已有内容 | 对我们的意义 |
|---|---|---|
| **Afsari，Sensitivity Analysis for the Problem of Matrix Joint Diagonalization，SIMAX 2008，30:1148–1171。** [期刊](https://epubs.siam.org/doi/abs/10.1137/060655997)／[作者全文](https://isr.umd.edu/Labs/ISL/ICA2006/Sensitivity_Final.pdf) | §2、4，Theorems 2.3、4.1–4.2：唯一性模数、条件数及扰动方向影响解混敏感性 | “唯一可辨识但病态，故有限误差下不可靠”不是我们的新概念 |
| **Zhu、Argentati、Knyazev，Bounds for the Rayleigh Quotient and the Spectrum of Self-Adjoint Operators，SIMAX 2013，34:244–256。** [期刊](https://epubs.siam.org/doi/abs/10.1137/120884468)／[作者全文](https://merl.com/publications/docs/TR2013-068.pdf) | §3–5，Theorems 3.7、3.9、Corollary 3.11：Rayleigh 商扰动、方向与残差；特征向量处存在二阶特例 | 误差交叉项、方向敏感性、总体二阶而一般窗口可能一阶，均需与该理论区分；特殊相容类的尖锐结论还需单独审查 |
| **Yu、Wang、Samworth，A Useful Variant of the Davis–Kahan Theorem for Statisticians，Biometrika 2015，102:315–323。** [期刊](https://doi.org/10.1093/biomet/asv008)／[作者全文](https://arxiv.org/pdf/1405.0680) | §2 Theorem 2 及证明：总体谱间隔与矩阵扰动控制子空间误差 | 控制子空间不等于给其中每根轴命名；应用该定理不算新谱扰动理论 |
| **Noé & Nüske，A Variational Approach to Modeling Slow Processes in Stochastic Dynamical Systems，MMS 2013，11:635–655（VAC）。** [期刊](https://epubs.siam.org/doi/10.1137/110858616)／[作者全文](https://arxiv.org/pdf/1211.7103) | §2.1–2.4、Theorems 2–3：可逆 Markov 过程的慢模态与变分原理 | 最慢动力学函数可以混合多个物理坐标，不能默认就是我们想选择的物理因素 |
| **Wu & Noé，Variational Approach for Learning Markov Processes from Time Series Data，Journal of Nonlinear Science 2020，30:23–66（VAMP）。** [期刊](https://link.springer.com/article/10.1007/s00332-019-09567-y)／[作者全文](https://arxiv.org/pdf/1707.04659) | §2、4、Theorem 2 及附录：Koopman 奇异函数、最优低秩模型、VAMP-E 验证 | 更优的总体预测模态与较低算子误差，不直接等于物理候选的同窗 top-2 风险 |

这一组的共同结论是：**我们可以使用成熟工具建立具体问题的结果，但不能把使用工具本身当成工具创新。** 对多 lag 的不同作用也要说清：训练预测时距、解混所用 lag、评价慢度所用 lag 是三个参数，不应互换。

### 4.3 非线性 ICA 与可辨识表示：恢复到了哪种坐标

| 正式论文 | 已有保证及关键条件 | 对物理慢因素选择意味着什么 |
|---|---|---|
| **Hyvärinen & Morioka，Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA，NeurIPS 2016。** [出版页](https://proceedings.neurips.cc/paper/2016/hash/d305281faf947ca7acade9ad5c8c818c-Abstract.html)／[全文](https://proceedings.neurips.cc/paper_files/paper/2016/file/d305281faf947ca7acade9ad5c8c818c-Paper.pdf) | 非平稳时间分段提供辅助信息，恢复源的逐因素变换，并有线性解混步骤 | 时间结构促进辨识已有；恢复信息不等于恢复原物理刻度 |
| **Hyvärinen & Morioka，Nonlinear ICA of Temporally Dependent Stationary Sources，AISTATS 2017。** [出版页](https://proceedings.mlr.press/v54/hyvarinen17a.html)／[全文](https://proceedings.mlr.press/v54/hyvarinen17a/hyvarinen17a.pdf) | §2–3、Theorems 1–2：非准高斯时间结构下恢复到逐因素单调变换；准高斯情形留线性混合；不同自相关可再用 SOBI 解混 | “滞后统计消除高斯旋转歧义”不是空白；非线性单轴变换仍可能改变我们的评分 |
| **Hyvärinen、Sasaki、Turner，Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning，AISTATS 2019。** [出版页](https://proceedings.mlr.press/v89/hyvarinen19a.html)／[全文](https://proceedings.mlr.press/v89/hyvarinen19a/hyvarinen19a.pdf) | Theorem 1 及补充证明：条件独立、辅助变量充分变化、光滑可逆与总体优化下，恢复到排列和逐因素可逆变换 | 不能把这种可辨识等同于平方变化量已确定；这是其等价类的数学后果，而非我们的独立新发现 |
| **Khemakhem 等，Variational Autoencoders and Nonlinear ICA: A Unifying Framework，AISTATS 2020（iVAE）。** [出版页](https://proceedings.mlr.press/v108/khemakhem20a.html)／[全文](https://proceedings.mlr.press/v108/khemakhem20a/khemakhem20a.pdf) | §4、Theorems 1–3：首先得到充分统计量间的可逆线性关系；额外条件收紧到块排列 | 不能简写成无条件线性恢复原始物理状态；仍无有限窗口选择保证 |
| **Sorrenson 等，Disentanglement by Nonlinear ICA with General Incompressible-flow Networks，ICLR 2020（GIN）。** [官方会议页](https://iclr.cc/virtual_2020/poster_rygeHgSFDH.html)／[作者全文](https://arxiv.org/pdf/2001.04872) | 生成设置、Appendix A Theorem 1：条件分布变化与总体匹配下处理因素及额外噪声维 | 与信息恢复和冗余维度有关；不是我们慢度排序的理论 |
| **von Kügelgen 等，Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style，NeurIPS 2021。** [出版页](https://proceedings.neurips.cc/paper/2021/hash/8929c70f8d710e412d38da624b21c3c8-Abstract.html)／[全文](https://proceedings.neurips.cc/paper/2021/file/8929c70f8d710e412d38da624b21c3c8-Paper.pdf) | Definition 4.1、Theorems 4.2–4.4：恢复 content 整体块，块内允许可逆变化 | 保留整块信息不自动提供具名物理轴；该区别已由论文保证的等价类明确体现 |
| **Yao 等，Learning Temporally Causal Latent Processes from General Temporal Data，ICLR 2022（LEAP）。** [官方目录](https://iclr.cc/Downloads/2022)／[作者全文](https://arxiv.org/pdf/2110.05428) | Theorems 1–2、Appendix A：充分变化的非平稳噪声，或特定 VAR 与广义 Laplace 噪声条件；恢复到排列加逐轴可逆变换及延迟因果关系 | 这类变换保持条件独立关系，但不普遍保持物理平方变化量；不能直接转用因果可辨识保证 |

**有限样本不是一个无人研究的缺口。** Lyu & Fu 的 **On Finite-Sample Identifiability of Contrastive Learning-Based Nonlinear Independent Component Analysis，ICML 2022**，在 Theorem 3.8、Eq.20 和 Appendix C 研究有限样本 GCL 的分离程度，包含样本量、函数类复杂度、逼近误差、辅助变化矩阵的最小奇异值。它需要独立样本、可逆读出和平滑性等条件，控制特定导数量；不能直接视为我们的状态 MSE 或窗口评分误差界。[出版页](https://proceedings.mlr.press/v162/lyu22a.html)／[全文](https://proceedings.mlr.press/v162/lyu22a/lyu22a.pdf)

**恢复指标本身的局限也已有直接研究。** Joshi 等的 **Who Guards the Guardians? The Challenges of Evaluating Identifiability of Learned Representations，UAI 2026，PMLR337:2618–2660**，区分数据生成过程与编码器几何对 MCC、R²、DCI 的要求，展示指标失配带来的误判。核读作者 v1 的 §2–3、Appendix F，发表状态由正式 PMLR 确认，未核对版本逐字差异。我们必须先排除这类误判，再讨论“恢复确实较好，但具体选择仍失败”。[出版页](https://proceedings.mlr.press/v337/joshi26a.html)／[核读全文](https://arxiv.org/html/2602.24278v1)

**近可辨识到稳定恢复也已有误差传播。** Nelson 等的 **Statistical and Structural Identifiability in Representation Learning，ICLR 2026**，区分跨训练一致性和真实因素恢复；Theorem 2 对 whitening＋ICA 给出 `ε′=Kε+K′ε²`，常数依赖协方差谱、ICA 目标与最优点曲率。结构恢复还需非高斯独立因素等额外条件，不能直接套在我们的高斯 OU 上。核读 §3、Theorems 1–3 和 Appendix A.3.5–A.3.6，正文按作者版。[正式论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f67e5f99b23b108a3a6665f410034bcd-Abstract-Conference.html)／[核读全文](https://arxiv.org/html/2603.11970v1)

基础背景还有 **Locatello 等，Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations，ICML 2019**：没有适当归纳假设就无法保证无监督解耦，并通过大规模实验提醒不要默认解耦指标提升会带来下游收益。本次核对其引言、贡献和公开摘要，不以未重审的证明细节作直接重叠结论。它不意味着我们在明确时序与校准假设下的任务不可能。[出版页](https://proceedings.mlr.press/v97/locatello19a.html)／[全文](https://proceedings.mlr.press/v97/locatello19a/locatello19a.pdf)

### 4.4 最接近的 LeJEPA，以及已有的“可辨识到任务”研究

**Klindt、LeCun、Balestriero，When Does LeJEPA Learn a World Model?，NeurIPS 2026 已录用。** [官方录用目录](https://neurips.cc/Downloads/2026)／[作者 v2 全文](https://arxiv.org/html/2605.26379v2)／[版本说明](https://arxiv.org/abs/2605.26379)

本次按 2026-11-09 的 camera-ready v2 核读 §3–5、限制部分及 Appendix F。其高斯 OU、维数匹配和相应总体约束下的正交恢复、近似恢复界，以及旋转不变规划保证，属于原作者。它没有承诺任意具名坐标任务都对正交变换不变，也未给训练样本量和优化过程的一般保证。

**明确的来源对应：** 附录 F 的一次／高次模态分离条件 `K_max < 2K_min`，在共同采样间隔下代入 `ρ_j=exp(−K_jΔ)`，就是 `ρ_min > ρ_max²`。本地 `s>0` 是这一条件的离散表达，沿该谱论证所得误差界属于适配推论。正交可恢复和 Hermite 竞争均不能列为我们的独立贡献。附录中把各向同性称为必要的措辞可作澄清：温和各向异性也可能满足上述条件；这处澄清不构成整篇论文的核心。

我们还必须正面承认以下已发表工作，而不是写“以往没有研究可辨识与下游任务的关系”：

| 正式论文 | 核读结果 | 与我们仍然不同的目标 |
|---|---|---|
| **HaoChen 等，Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss，NeurIPS 2021。** [出版页](https://proceedings.neurips.cc/paper_files/paper/2021/hash/27debb435021eb68b3965290b5e24c49-Abstract.html)／[全文](https://papers.nips.cc/paper/2021/file/27debb435021eb68b3965290b5e24c49-Paper.pdf) | §3–5，尤其 Theorems 4.1–4.2、5.1：增强图结构、预训练目标误差及谱条件连接到线性分类误差，并有有限样本保证 | 分类目标由标签与增强图假设连接；不是物理轨迹归一化变化量排序 |
| **Zhang、Wang、Wang，Identifiable Contrastive Learning with Automatic Feature Importance Discovery，NeurIPS 2023（triCL）。** [出版页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b6a171867138c80de2a35a6125d6757c-Abstract-Conference.html)／[全文](https://papers.nips.cc/paper_files/paper/2023/file/b6a171867138c80de2a35a6125d6757c-Paper-Conference.pdf) | §4–5、Theorem 4.5、Corollary 4.6、Theorem 5.1：谱值互异、模型充分表达时确定有序特征到符号，并分析选部分特征的分类界 | 这里的特征由增强图谱定义；不等于原始具名物理源，也不是每个窗口重新排序。不能主张首次把辨识与特征选择连接起来 |
| **Saengkyongam 等，Identifying Representations for Intervention Extrapolation，ICLR 2024。** [出版页](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aa9508b2577e0d9c824da9e2e27fe9d6-Abstract-Conference.html)／[全文](https://proceedings.iclr.cc/paper_files/paper/2024/file/aa9508b2577e0d9c824da9e2e27fe9d6-Paper-Conference.pdf) | §3–4、Theorem 3 及 Appendix D.3：在外生行动对潜变量线性作用、残差支持等条件下，仿射辨识足以支持干预外推 | 已有“某种辨识等价类足以完成特定任务”的正式范例；我们的任务对坐标变换的不变性不同 |

### 4.5 选择性预测与风险控制：A3 使用了什么成熟工具

**Geifman & El-Yaniv，Selective Classification for Deep Neural Networks，NeurIPS 2017。** [出版页](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)／[全文](https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf)

§2–4、Lemma 3.1、Algorithm 1、Theorem 3.2 已包含：固定预测器，以置信分数阈值拒答，计算覆盖率及接受区域的条件错误率，用精确二项上界与多项校正选择规则。其风险保证要求相应的同分布校准／测试条件。我们的 A3 无标签 CAL 固定 gap 阈值后用独立有标签 TEST 审计，与其具体数据角色不同，**但不形成新的统计原理**。

**Geifman & El-Yaniv，SelectiveNet: A Deep Neural Network with an Integrated Reject Option，ICML 2019。** [出版页](https://proceedings.mlr.press/v97/geifman19a.html)／[全文](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf)

核读 §2、4–5、7–8.1。其主要方法联合学习预测和拒答；与我们最直接相关的是 §5 已用独立无标签验证集的置信分数分位数调整覆盖率。因此 B3 重新取目标过程分差中位数，只是已有覆盖校准操作的应用。它调整答多少，不自动保证答得多准。

**Angelopoulos 等，Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control，Annals of Applied Statistics 2025，19(2):1641–1662，DOI 10.1214/24-AOAS1998。** [作者机构出版页](https://www.gsb.stanford.edu/faculty-research/publications/learn-then-test-calibrating-predictive-algorithms-achieve-risk)／[IMS 期刊卷](https://www.imstat.org/publications/aoas/aoas_19_2/aoas_19_2.pdf)／[核读作者全文](https://arxiv.org/pdf/2110.01052)

已确认正式发表；因最终版访问受限，定理与页码核读采用作者2022版。Definition 1、Theorem 1 和 §2 把风险目标变成多重假设检验；§3.2 明确处理接受后的条件分类错误率，§3.3 处理选择后的 MSE。将窗口 top-2 集合视为离散答案，就已有适用的通用风险控制路线。若以后主张新方法，应比较**相同标签预算下，gap 加现成 LTT／二项检验**，而不只比较未经风险校准的固定分位数。

| 正式论文 | 核读位置与保证 | 不能混用的地方 |
|---|---|---|
| **Angelopoulos 等，Conformal Risk Control，ICLR 2024。** [机构出版页](https://research.google/pubs/conformal-risk-control/)／[正式全文](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf) | §1.1–2.3，Theorem 1：交换、单调、右连续、有界损失下，控制对 CAL 和新样本共同平均的期望风险 | 不是“以95%置信度保证已接受窗口的总体错率≤1%”；接受后的条件风险随阈值也未必单调 |
| **Farinhas 等，Non-Exchangeable Conformal Risk Control，ICLR 2024。** [出版页](https://proceedings.iclr.cc/paper_files/paper/2024/hash/de04896f011beff76c91e094f72727f4-Abstract-Conference.html)／[全文](https://proceedings.iclr.cc/paper_files/paper/2024/file/de04896f011beff76c91e094f72727f4-Paper-Conference.pdf) | §2–4、Theorem 1：非交换情形的期望损失界带有分布差异项，主体定理对权重有条件 | 已有漂移下风险分析，但没有无条件解决任意动力学变化，也不能直接当选择后条件风险保证 |
| **Audibert & Tsybakov，Fast Learning Rates for Plug-in Classifiers，Annals of Statistics 2007，35:608–633。** [DOI](https://doi.org/10.1214/009053606000001217)／[作者托管正式全文](https://imagine.enpc.fr/~audibert/AudTsy_AOS0231.pdf) | §3、5，Lemma 3.1、Theorems 3.2–3.5、Lemmas 5.1–5.2：回归误差结合边界附近概率质量，转为分类风险 | “均方误差不能单独保证离散决策”“间隔相对误差重要”是已有原则；我们的增量需落在动态评分这一具体函数上 |
| **Cattelan & Silva，How to Fix a Broken Confidence Estimator，UAI 2024，PMLR244:547–584。** [出版页](https://proceedings.mlr.press/v244/cattelan24a.html)／[核读录用稿](https://arxiv.org/pdf/2305.15508) | §3–6：比较置信分数，包含前两名 logit 间隔及其归一化，区分分类准确率与错误筛选能力 | gap 信心分数及简单归一化已有直接近邻；其目标是分类 logits，不是物理慢度比值 |
| **D’Amour 等，Underspecification Presents Challenges for Credibility in Modern Machine Learning，JMLR 2022，23(226):1–61。** [出版页](https://www.jmlr.org/papers/v23/20-1335.html)／[全文](https://www.jmlr.org/papers/volume23/20-1335/20-1335.pdf) | §1–3.1：相近 iid 测试表现不约束所有部署表现，还含动力学参数未充分确定的例子 | 一般可靠性批评已有；我们的前提要更强：因素恢复成立后，仍需研究具体时间决策 |

三种保证必须分开：①单个窗口的答案有确定性／概率证书；②以高置信度控制被接受窗口群体的条件错误率；③对 CAL 与测试随机性平均的期望损失。我们的 A3 属于固定流程下的第二种独立审计，不是第一种，也不能拿第三种定理替代。

对应到 A3，独立采样单位是完整轨迹，各窗口内部的时间点不是独立试验。上界条件于固定的编码器、读出、校准和 REF；它不自动包含重新训练模型或重新抽取参考组的变化。

### 4.6 排名推断与部分可辨识：目标变化带来的真正区别

**Mogstad、Romano、Shaikh、Wilhelm，Inference for Ranks with Applications to Mobility across Neighbourhoods and Academic Achievement across Countries，Review of Economic Studies 2024，91:476–518。** [正式版作者全文](https://home.uchicago.edu/~amshaikh/webfiles/rankingsconf.pdf)／[作者机构记录](https://discovery.ucl.ac.uk/id/eprint/10142590/)

核读 §2–3，尤其 §2.3、3.3–3.4、Theorem 3.3：从成对参数差的联合置信集合构造各排名、整体排名和前若干名集合的推断。因此“给 top-k 一个可靠性保证”不是新问题。它估计的是总体参数排名；我们要判断同一条已发生轨迹里的物理评分。移用其程序需要先建立适用于我们评分误差的有效区间，不能只更换变量名称。

**Hall & Miller，Using the Bootstrap to Quantify the Authority of an Empirical Ranking，Annals of Statistics 2009，37(6B):3929–3959。** [DOI](https://doi.org/10.1214/09-AOS699)／[正式电子重印全文](https://arxiv.org/pdf/0911.3749)

核读 §3.1–3.3、Theorem 3.1：在近并列等情形，普通 n-out-of-n bootstrap 对排名分布可能不一致；不同重抽方式需要不同条件。因此旧联合重抽样没过风险要求是需要解释的实验事实，不能直接宣布发现了 bootstrap 的新缺陷；也不能未经核对就断言它必定是本文所述不一致机制。

**Imbens & Manski，Confidence Intervals for Partially Identified Parameters，Econometrica 2004，72:1845–1857。** [正式发表页](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0262.2004.00555.x)／[公开工作论文全文](https://cemmap.ac.uk/wp-content/uploads/2020/08/CWP0903.pdf)

本次读取正式摘要及2003工作论文的定义、引言；最终刊版全文未成功获取。已有研究明确区分识别集合与真参数的置信覆盖。我们“所有相容解释是否给同一个答案”的思路属于部分可辨识范畴，不能声称新框架；潜在贡献应是具体慢度目标的相容范围、速率与信息下界。本文不以未核读的最终公式作更强比较。

**Fieller，Some Problems in Interval Estimation，JRSS Series B 1954，16:175–185。** [正式发表信息与摘要](https://academic.oup.com/jrsssb/article/16/2/175/7026708)

本次仅核读出版信息和摘要，正文访问受限。比值估计中的分子、分母及其协方差有经典先例。因此不能把“联合处理分子分母优于分开处理”的一般动机包装成新统计原则。我们使用的动态二次型与共享参考仍需自己的条件推导；本文不声称 Fieller 原定理直接覆盖当前轨迹设置。

## 5. 预印本和阅读受限材料：单独处理

**H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning，2026。** 本次查看作者项目页与 arXiv v2 的 §3、Appendix F 相关内容，未核实正式录用，暂按预印本处理。其逐实体 probe、频率分离和规划结果是作者自己的成果；我们的冻结编码器评价是一项新任务测试，不是这些结果的原创来源。它不作为强重叠裁决，但我们仍应如实承认其已公开的实验内容。[项目页](https://h-jepa.com/)／[全文](https://arxiv.org/html/2610.06805v2)

**LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics，2025。** 本轮核到作者 arXiv 页面，未核实正式主会录用，暂按“正式状态待核”列示。这是训练方法来源，必须与已确认录用的 World Model 理论论文分开。[作者稿](https://arxiv.org/abs/2511.08544)

**Finite-Sample Valid Rank Confidence Sets for a Broad Class of Statistical and Machine Learning Models，Chandra & Xie，2025。** 本轮核对 arXiv v2 的摘要与版本，未找到可确认的正式录用；只作近期排名推断线索，不作为强重叠证据，也未宣称核读全部44页。一般排名推断重叠主要依据上节的已发表论文。[作者稿](https://arxiv.org/abs/2512.00316)

旧笔记另有 FAR-JEPA、Selective Prediction from Agreement 等预印本线索。本轮未完成其最新发表状态与全文版本核对，故**不把旧笔记中的摘要判断升级为强重叠结论**，也不据此排除我们的研究方向。

另有两篇**已经发表、但此次原始全文核查不完整**的相关论文，不能混入预印本类：

- **Klindt 等，Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding（SlowVAE），ICLR 2021。** 已确认正式论文信息；原始证明全文获取受限。正式摘要与 LEAP Appendix A.4 表明其利用稀疏时间转移，收紧辨识等价类。后续若主张“既有时序辨识都只保证任意非线性变换”，必须先正面核查此反例；本文不作该笼统主张。[正式论文入口](https://openreview.net/pdf?id=EbIDjBynYJ8)
- **Bates 等，Distribution-Free, Risk-Controlling Prediction Sets，JACM 2021，68(6)，Article 43，DOI 10.1145/3478535。** 发表身份由作者出版列表核实；全文访问受限，本次不声称通读。其高概率风险控制是重要背景，本文与 A3 的直接比较以已核读 SGR／LTT 为准。[作者出版列表](https://stephenbates19.github.io/research.html)／[全文入口](https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/rcps.pdf)

### 5.1 原作者成果、复用推论和我们实验的归属

| 内容 | 归属 | 我们可以怎样表述 |
|---|---|---|
| LeJEPA 训练方法与 SIGReg | 原 LeJEPA 作者 | 采用／适配已有训练方法 |
| World Model 论文的正交可辨识、近似恢复界、Hermite 谱分析及规划保证 | Klindt、LeCun、Balestriero；其谱分析还承接 SFA 等既有工作 | 引用其条件与结论，作为出发点 |
| `ρ_min > ρ_max²` 与沿其证明得到的本地谱推论 | 对原条件与论证的重写／适配 | 明确“由已有结果得到”，不列独立原创定理 |
| 交叉项恒等式、一般 gap 保序条件、CP／Bonferroni 统计流程 | 代数与既有数学统计工具 | 用于组织分析与验证，不作创新点 |
| 本地 LeJEPA 的自然误差干预、A1–A4 与 B1–B3 的具体协议和结果 | 我们独立执行的实验 | 可报告我们的新实验；其科学结论是否新颖仍逐项核对 |
| 旧剂量模型的具体歧义量级、精确相容类与拒答边界 | 本地推导的候选成果，原创性与适用性待进一步审查 | 不因“自己写过证明”就宣布首次提出 |
| 官方 H-JEPA 的模型、层级抽象与规划表现 | H-JEPA 原作者 | 引用已有模型与结果；我们的评价结果尚未产生 |

**“实验是我们做的”“结论被我们的实验支持”“结论是首次提出的”，是三种不同声明。** 用户提供参考论文不意味着授权把它的成果算作本研究贡献；本文严格区分这三者。

## 6. 将我们已有结果逐项放回文献中

| 我们的结果或拟写主张 | 最接近的既有内容 | 合适的处理 |
|---|---|---|
| 非线性慢模态可能替代另一个物理因素 | xSFA 2014；LeJEPA World Model 附录 F | 作为理论起点和复现现象，不写首次发现 |
| 条件 `ρ_min > ρ_max²` 保证一次模态排在高次模态之前 | LeJEPA 附录 F 的速率条件 | 离散时间重写与直接推论，不单列核心新定理 |
| 解混谱间隔小、估计噪声大时容易混轴 | SFA、SOBI、谱扰动、联合对角化 | 既有原理；B2 是我们管线中的实证边界 |
| 提高解混 lag 未必更好，多 lag 也未必赢 | xSFA 的采样权衡；SOBI 的效率与 lag 选择 | 本轮负结果保留，不能推广成所有多 lag 方法无效 |
| 高 R² 不足以保证慢集正确 | 可辨识评价批评、一般决策间隔理论 | 不把一般批评当主贡献；强调恢复指标有效时仍需检查特定时间决策 |
| 误差同样大，混入方向不同会使排序不同 | 二次型／Rayleigh 商扰动；任务误差传播 | 精确反例和实际自然误差干预可作新增证据；一般方向敏感性已有理论 |
| 真值交叉项可以修复一部分错误，也会制造新错 | 平方误差分解与误差抵消 | 分解是恒等式；价值在实际模型中定位机制，且明确它不能测试时部署 |
| 参考方差不匹配能翻转结果 | 比值估计、联合不确定性、SFA 归一化 | “分母重要”不是新理论；共享参考如何影响多个窗口，可能形成更具体问题 |
| 大预测分差筛出低错误窗口 | selective classification、SGR、SelectiveNet、LTT | 使用成熟方法；A3 是限定范围的独立正结果 |
| 重新取分差中位数恢复约 50% 作答量 | SelectiveNet 的分位数覆盖校准 | 不写新方法；继续问为何覆盖恢复不等于风险要求恢复 |
| 状态边缘分布与读出保持不变，改变动力学后阈值失配 | 分布变化与任务敏感性研究 | B3 的受控比较较有价值，但固定数值阈值随过程变慢而失去覆盖是直接极限 |
| bootstrap 排名置信规则失败 | Hall–Miller 排名非正则性；已有风险校准 | 保留失败，不能说首次发现 bootstrap 可能不可靠 |
| 同一观测允许多个物理解释时，慢集也可能不唯一 | 部分可辨识、非线性 ICA 等价类 | 一般框架已有；具体相容类上的尖锐速率和拒答极限才是理论候选 |

这些判断不意味着过去实验白做了。它们把几件本来混在一起的事拆清楚了：哪些现象是已知原理的预期，哪些机制在实际训练误差中出现，哪些看似合理的修复没有效果，以及哪些可见规则能在新轨迹上通过规定的可靠性要求。

### 6.1 当前较有价值的新增证据

**第一，实际误差的干预证据，而不只是人工加噪。** 自然线性混入修正在 CAL+NAME 读出上两轮分别改善约 6.17、5.84 个百分点；换成强 MLP 后，继续修小线性项、低阶交互项和增量损失没有稳定收益。这支持“机制依赖读出与剩余误差结构”，不支持一个普遍有效的修正算法。来源：[自然误差方向](../reports/lejepa_natural_direction_repair_20261113/RESULTS.md)、[动态损失](../reports/lejepa_dynamic_transfer_20261113/RESULTS.md)。

**第二，明确区分有条件成功与适用边界。** A3 在新轨迹上，以预先固定阈值取得约 50.90% 作答比例、3／33,355 次错误；逐组合经校正的风险上界和覆盖要求全部通过。A2 改变读出拟合预算后，一部分组合不再通过。但 A2 的恢复反差未在所有模型达到预定要求，因此不能把它写成“R² 单独导致失败”的完整因果实验。来源：[A3](../reports/lejepa_a3_gap_20261115/RESULTS.md)、[A2](../reports/lejepa_a2_readout_20261115/RESULTS.md)。

这里的汇总计数只作描述，风险上界按各模型—过程组合分别计算。不同模型共享物理轨迹，不能把33,355次作答都当作彼此独立的样本，再计算一个总置信区间。

**第三，将状态恢复与时间决策的变化分开。** B3 保持编码器、读出和状态边缘分布，改变时间尺度，旧阈值在变慢过程上的作答比例降至约 31.94%；新阈值恢复到约 49.88%，但仍有两个组合未确认满足风险上界。它们不是已被证明真实错误率超过 1%。这比笼统报告“换分布后效果变差”更有解释力。来源：[B1／B3](../reports/lejepa_b_transfer_20261115/RESULTS.md)。

**第四，早期理论确实比一个误差展开式更具体。** 在精确定义的剂量与相容坐标模型中，本地已有总体和有限窗口歧义的不同量级、窗口长度的不同极限，以及接近最优的拒答区间。这是备选理论主线的材料，但假设与当前监督读出 LeJEPA 不是同一套，不能直接拼接。来源：[窗口歧义理论](../reports/stage_a_plus/dose_window_20261109/THEORY.md)、[窗口与剂量共同极限](../reports/stage_a_plus/dose_joint_window_20261110/THEORY.md)。

### 6.2 四个仍可能容纳贡献的具体缺口

| 候选缺口 | 已有工作为什么还不能直接回答 | 我们已经有什么 | 还缺什么才有分量 |
|---|---|---|---|
| G1：恢复后，同一有限窗口的物理 top-2 稳定性 | 因素恢复界、总体谱模态和总体排名，都不直接等于该窗口答案；一般排序理论仍可作为工具 | 受控反例、误差分解、自然误差干预、A3 正结果 | 超过一般间隔引理的具体定量规律，或能跨预定条件复验的紧凑机制结论 |
| G2：恢复保持不变时，时间尺度如何改变可用的风险—覆盖关系 | 状态恢复只看状态分布；时间决策依赖联合轨迹分布 | B3 的受控改变，B1 的未恶化对照 | 区分单纯分差缩放与更深的评分误差变化；不要把直接缩放效应当复杂新机制 |
| G3：不完全确定物理刻度时，有限窗口答案到底能确定到什么程度 | 一般部分可辨识没有给本问题的相容变形范围与尖锐速度 | 特定剂量模型中的量级、相位区间与拒答极限 | 对齐模型假设，独立审查证明，并说明实际系统何时提供这种干预信息 |
| G4：共享参考尺度怎样限制跨窗口的可靠性 | 常规逐窗口方差分析容易条件于一个固定参考；比值与相关不确定性工具本身已有 | Fine 的分母干预、共享参考理论、LeJEPA 局部诊断 | 固定读出和参考分布，仅改变参考数量并重复独立参考组；目前未完成 |

其中 G1 最符合当前主线，G3 的理论辨识度可能更高但假设距离更大，G2 和 G4 更适合作为 G1 的条件或边界。**没有一项可以仅凭“别人没有使用我们的193步窗口”就认定新颖。**

## 7. 第二个缺口：中心论断的四个备选方案

### 方案一：因素恢复以后，什么条件才能支持可靠的慢因素决策——推荐主线

**中心句：在因素可辨识、有限样本读出已经达到较高精度时，可靠选出物理慢因素，仍需控制剩余误差经过时间差分和参考归一化后对入选边界的影响。**

这条路线关注的不是一般意义上的“表示好不好”，而是已经读出的物理量如何进入一个具体决策。所有结果围绕同一件事组织：恢复误差如何变成评分误差，评分误差如何与当前窗口的分差相互作用，以及校准规则何时仍适用。

| 项目 | 判断 |
|---|---|
| 最适合的贡献类型 | 机制与条件研究，附限定范围的可靠性验证 |
| 当前证据 | A1 的自然误差干预；A3 的独立正结果；A2、B2、B3 的边界 |
| 最强近邻 | xSFA 2014、LeJEPA 2026、Joshi 2026、SGR／LTT、排名推断 |
| 拟建立的具体增量 | 把因素恢复、同窗物理评分及条件决策风险连接起来，并用受控改变检查各环节；该连接的独立新意仍需具体结果支持 |
| 尚不能声称 | 一般充分必要条件、通用逐窗证书、普遍有效的误差校准算法 |
| 最关键的补强 | 提炼一个能解释多组干预、能被独立反例检验的定量关系；不能只有“误差小于分差”这条通用结论 |
| H-JEPA 的作用 | 检查该关系在视觉、不同层的冻结表示上是否仍有解释力；不把层级收益预先写进结论 |
| 何时应收缩 | 如果最后只能说高 R² 不够、gap 筛选有效，定位应降为应用审计，不能靠堆更多种子夸大贡献 |

一个有用的组织方式是保留评分差的分解。令 `e` 是读出误差，`K=2 avg(Δz Δe)` 是真实变化与误差变化的交叉项，`E=avg((Δe)²)` 是误差自身的变化，`V̂−V` 是参考方差偏差，则

\[
\widehat D-D=\frac{K+E-D(\widehat V-V)}{\widehat V}.
\]

**该式是记账工具，不是待宣传的新定理。** 真正需要比较的是入选因素与未入选因素的评分误差之差，而非把所有轴的误差平均成一个数。也必须区分：用测试真值事后解释失败，与在测试前提供有效的可靠性判断，是两个不同成果。

也不能把 R² 说成完全无用。在指定状态分布、方差下界、时间误差控制和足够分差等条件下，恢复误差可以参与推出保守的决策界。我们需要找的是这些额外条件及其实际适用程度，而不是证明“任何恢复指标永远不能提供帮助”。

### 方案二：物理因素已命名，但刻度仍不唯一时，慢集能确定到什么程度——理论备选

**中心句：当现有观测与校准仍允许一组物理坐标解释时，总体慢度接近，并不意味着每个有限窗口的慢集也接近；可以研究这种歧义的精确量级，以及任何可靠选择器必须放弃多少窗口。**

本地最有辨识度的候选结果来自精确剂量模型：允许的坐标为 `h(x)=x+ℓ p({x/ℓ})`；在相应假设下，总体评分偏差是二阶量级，固定有限窗口的典型评分扰动和慢集分歧可以是一阶量级。特定近并列极限由 `ℓ²√T` 决定；逐窗相位区间相对最优统一零错误规则的额外拒答只有二阶量级。这里的 `ℓ` 表示校准后仍允许的周期性刻度变化尺度，不是训练 MSE。

| 项目 | 判断 |
|---|---|
| 最适合的贡献类型 | 特定观测与干预模型下的尖锐界、下界与接近最优的决策 |
| 当前证据 | 本地已有推导、数值核对及机制材料；此次未重新完整审证明 |
| 最强近邻 | 部分可辨识、非线性 ICA、SFA 高次谱与一般排名推断 |
| 潜在增量 | 具体慢度函数上的总体／有限窗口不同量级，以及完整相容类的拒答代价 |
| 最大风险 | 主要理论依赖精确剂量、相容变形类和参考分布；与当前 LeJEPA 强监督读出可能脱节 |
| 必须先澄清 | Gaussian 是分析基准，还是所有相容物理解释都必须满足的已知分布？后一要求若结合单轴严格单调可逆性，就会排除所使用的非线性刻度反例 |
| H-JEPA 的作用 | 现有冻结评价脚本不能检验精确剂量等价类；不能把它硬当这个定理的验证 |
| 何时放弃作为主线 | 若假设不能对应任何合理的信息接口，保留为独立理论结果或附录，不拖着主线改环境去迁就定理 |

特别注意：这条路线研究的是“给定信息以后，目标是否仍有歧义”。它并没有反驳精确物理因素已完全确定时可以直接计算慢度。若选择此方案，必须把题目的可辨识层级写清楚，不能同时声称物理刻度完全可辨识，又依靠刻度不唯一证明不可能性。

具体地，若 `X` 与 `h(X)` 都必须具有同一已知标准 Gaussian 分布，且 `h` 严格递增可逆，由累积分布函数就得到 `h(X)=X` 几乎处处；允许严格递减时只增加符号翻转，它不改变平方增量慢度。上述周期非线性相容类不能同时满足这套约束。这个判断只针对单轴单调变换；SIGReg 使表示接近 Gaussian，不等于物理坐标分布已知，也不排除一般多维的 Gaussian 保持变换。

### 方案三：面向慢因素集合的风险控制方法——有条件的算法备选，目前不推荐

**中心句：利用时间误差与参考不确定性，构造比单纯预测分差更有效的作答规则，在同一错误率要求下回答更多窗口。**

这是最容易形成方法论文结构的路线，但也是当前证据最弱的路线。A4 中额外校准没有在匹配覆盖率下超过简单分差；可见交叉项预测也没有稳定增益。不能把 CP 上界、Bonferroni、分位数阈值或 LTT 重新命名成我们的算法。

要启动这条主线，至少应有一个明确可观测的新分数或有效区间，在独立数据上相对 gap 和已有风险控制程序取得稳定优势。需在相同标签、参考样本、覆盖率和风险口径下比较；若使用标签 CAL 控制风险，也应给 gap 同样的标签 CAL。**目前没有这项结果，不建议为了凑方法贡献再开一轮泛化的残差搜索。**

### 方案四：层级表示何时能保留可可靠选取的慢因素——扩展方向

**中心句：高层保留长期慢信息，不保证在每个有限窗口中仍足以恢复并选出所需物理因素；层间比较需要同时看信息保留与决策可靠性。**

这与 H-JEPA 最接近，但目前服务器结果尚未返回。原 H-JEPA 已做频率分离、逐实体 probe 和规划评价；只发现“高层较慢”或“高层丢失快因素”不足以构成新贡献。

我们的代码可以检查两层在共同轨迹、共同评分和同规格读出下的恢复与慢集风险，也可以检查第一层阈值能否直接转移。这些是关联与迁移测试。若上层无法恢复某个因素，应先把它标为前提未满足，不能混成“因素已经恢复但决策失败”；若要声称层级结构导致更可靠，还需要额外的训练或架构干预。

**建议把此方案作为方案一的外部验证，不现在改成整篇文章的中心。**

## 8. 我的建议，以及这轮完成了什么

建议用**方案一组织现有研究，方案二保留为有明确假设的理论备选**。方案三等待真正的增益证据，方案四用于检查视觉与层级范围。这样既不需要推翻已完成的 A／B 实验，也不会为了制造算法名称而重启已停止的分支。

对当前论文中心的建议表述是：

> 我们研究可辨识表示中的因素信息，何时足以支持有限轨迹上的可靠物理慢因素选取。研究区分因素恢复、坐标与参考尺度、时间评分误差，以及最终集合决策；通过受控推导和实际 LeJEPA 干预，确定恢复指标之外仍需检查的条件，并验证简单分差规则的成功范围和失效边界。

这是一条**当前证据支持的研究定位**，不是已经完成的新颖性证明，也不是 ICML 录用判断。要把它升级为有力的中心贡献，还需要从候选缺口中拿出一项真正具体的增量：超出通用间隔引理的定量边界、在合理信息条件下的尖锐不可能性结果，或可复现且有额外解释／预测能力的机制规律。

本轮已经完成：发表状态与最近邻核对、现有结论的重叠分类、可保留的缺口、四种中心论断及其证据要求。**尚未完成的是选择并建立那个最终增量；不能把“写出了贡献表述”当作这一缺口已经在科学上解决。** 本文不修改旧实验的成功／失败标准，不启动新实验。
