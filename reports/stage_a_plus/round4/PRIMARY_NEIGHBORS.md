# Round 4：Fine 价值路线的定向 primary 近邻

核查日期：2026-11-04。范围为本项目最直接的有限样本/稳定性、非线性慢源恢复及条件流方向；不是系统综述或优先权证明。均为 primary 正文或正式 proceedings；结论仅说明已读定理的范围，不能据此证明未检索文献不存在相关证书。未采纳网页摘要中的 first/complete 等优先权措辞。

## 1. 已有定量 nonlinear ICA：不能称“从来只有总体定理”

**Qi Lyu、Xiao Fu，2022，ICML**，[*On Finite-Sample Identifiability of Contrastive Learning-Based Nonlinear Independent Component Analysis*](https://proceedings.mlr.press/v162/lyu22a.html)，[全文](https://proceedings.mlr.press/v162/lyu22a/lyu22a.pdf)。重点核查 Lemma 3.7、Theorem 3.8、Assumptions 3.2/3.6。该文用 GCL、泛化与数值微分控制源混合导数指标，纳入网络逼近误差；要求 i.i.d. 数据、最优解、可逆表示、平滑/有界条件及充分 variation。它确已跨越纯无限样本辨识。它不直接给当前条件 NLL、相关轨迹、实际 G0 top-k 的可计算证书；真实 variation、逼近和高阶导数常数仍需建立，不可把时序 transition 数直接代为独立 n。

## 2. iVAE 一致性已存在，但不是实际有限训练的认证

**Ilyes Khemakhem、Diederik P. Kingma、Ricardo Pio Monti、Aapo Hyvärinen，2020，AISTATS**，[*Variational Autoencoders and Nonlinear ICA: A Unifying Framework*](https://proceedings.mlr.press/v108/khemakhem20a.html)，[全文](https://proceedings.mlr.press/v108/khemakhem20a/khemakhem20a.pdf)。Theorem 1 要求条件独立 exponential family、injective mixing 与自然参数 contrast 满秩；Theorems 2/3 进一步缩小等价类。Theorem 4 要求变分族包含真实 posterior、最大化目标及无限数据，给等价类一致性。不能将该文一般充分统计量线性歧义直接写为任意分布下逐轴 affine；也不能将其一致性理解为当前有限网络/有限训练风险上界或 G0 选集保证。

## 3. GIN 已有条件 Gaussian flow 与逐轴 affine 辨识

**Peter Sorrenson、Carsten Rother、Ullrich Köthe，2020，ICLR**，[*Disentanglement by Nonlinear ICA with General Incompressible-flow Networks (GIN)*](https://arxiv.org/abs/2001.04872)，[作者全文](https://arxiv.org/html/2001.04872v1)。§3 与 Appendix A/B 在条件 Gaussian mean/variance 变化及相应 rank 等条件下，将准确条件密度匹配约束成逐轴 translation/scaling，并实现体积保持网络；证明使用无限数据极限下相等的条件分布。当前 Fine 有一般 Jacobian 与历史相关条件核，须重新核查兼容性。条件 Gaussian flow 学源、精确 likelihood 及逐轴 affine 结论均非新；GIN 没有直接认证当前有限学习器的实际慢选集。

## 4. 慢源恢复与非线性 inverse 并非 Fine 首创价值

**Tobias Blaschke、Tiziano Zito、Laurenz Wiskott，2007，Neural Computation 19(4):994–1021**，[*Independent Slow Feature Analysis and Nonlinear Blind Source Separation*](https://www.ini.rub.de/PEOPLE/wiskott/Publications/BlasZitoWisk2007-ISFA-NeurComp.pdf)。ISFA 结合 second-order ICA 与 SFA；文中保留二阶时间相关不足以完全代表独立性的限制。这里 ISFA 是 Independent。其价值背景覆盖非线性混合中用慢性帮助源恢复，不能将“非线性表征后读慢因素”本身作为 Fine 创新。

**Henning Sprekeler、Tiziano Zito、Laurenz Wiskott，2014，JMLR 15:921–947**，[*An Extension of Slow Feature Analysis for Nonlinear Blind Source Separation*](https://jmlr.org/papers/v15/sprekeler14a.html)，[全文](https://jmlr.org/papers/volume15/sprekeler14a/sprekeler14a.pdf)。§3 分析 unrestricted Sobolev 函数类，使用独立平稳连续可微源及源/速度联合 factorization，得到 harmonic/product 结构；部分 Gaussian 情形恢复原源。其理想理论与算法不自动覆盖 coupled FHN 或离散 Gaussian kick。它提醒：非线性 inverse 的表达优势、慢源可恢复与有限学习可达性是不同结论。

## 5. 2026 已明确区分重复一致与真实结构

**Walter Nelson、Marco Fumero、Theofanis Karaletsos、Francesco Locatello，2026，ICLR**，[*Statistical and structural identifiability in representation learning*](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f67e5f99b23b108a3a6665f410034bcd-Abstract-Conference.html)，[全文](https://proceedings.iclr.cc/paper_files/paper/2026/file/f67e5f99b23b108a3a6665f410034bcd-Paper-Conference.pdf)。Theorem 1 用 decoder 的局部 bi-Lipschitz 等条件推 near-identifiability；Theorem 3 的结构恢复另需真实生成机制、独立非 Gaussian 源、重构等条件。作者明确指出局部 bi-Lipschitz 条件难以经验检验。该区分与定量近可辨识方向已存在；它不把“多次训练坐标一致”自动变成真实物理轴正确，也未直接提供当前 likelihood→G0 证书。

## 6. 条件流确有无需潜在真值的校准保证，但对象不同

**Victor Dheur、Souhaib Ben Taieb，2025，NeurIPS**，[*Multivariate Latent Recalibration for Conditional Normalizing Flows*](https://proceedings.neurips.cc/paper_files/paper/2025/hash/079923016ab82f373c0a4a080b2f3ee6-Abstract-Conference.html)，[全文](https://papers.neurips.cc/paper_files/paper/2025/file/079923016ab82f373c0a4a080b2f3ee6-Paper-Conference.pdf)。§3.1 用逆流 latent norm 定义校准；§3.3/Eq.15–16 在校准映射的 order-statistic 误差条件下给有限样本覆盖界，并联系 conformal prediction。它可以使用观测响应与独立校准信息而不使用物理源标签；不能说 conditional flows 完全没有可操作保证。其保证是 latent norm/预测覆盖，不是坐标源辨识、逐轴慢度顺序或对真实慢选集的稳定性。

本项目的直接数学边界：若基准 latent law 球对称，对任意正交 Q，\(\|Qz\|=\|z\|\)，因此 norm 校准及预测密度可以完全不变，而物理轴或不同慢度轴被混合。此例说明认证对象不同，不否定该论文的校准定理。独立校准的样本要求也不能直接套给已用于选点的 CAL。

## 7. 新增 2026 预印本 lead：需审慎，不采用其快率

**Yuwen Jiang，2026-04-10，arXiv v1，未核为正式会议**，[*Finite-Sample Analysis of Nonlinear Independent Component Analysis: Sample Complexity and Identifiability Bounds*](https://arxiv.org/abs/2604.08850)，[全文](https://arxiv.org/html/2604.08850v1)。§3.4/Eq.10 用跨 context 最小 KL 作 diversity；§4.3 的 loss-to-MCC 界依赖局部正 Hessian；§4.7 的 SGD 扩展另用 PL。本文不采纳其 rate/optimality/first 声明：跨 context KL 不等价于逐轴可辨识曲率，Lemma 4.1 的 excess-loss 非负处理有待修正，Eq.6 的逐行 max 也不是一一匹配。

独立核查理由如下，不能将作者断言当作已证事实。令 \(p(S|u)=N(0,c_uI)\)，不同 \(c_u\) 给正跨 context KL，但任何正交旋转保持全部条件 law，真实轴仍不唯一；所以仅此 KL 下界不能代替 natural-parameter contrast rank 或轴间 signature。总体最优函数的逐点 excess loss 可有正负，非负原始 loss 不推出该差逐点非负；由此不能直接用 bounded nonnegative variable 的论证获得 Bernstein 条件。该 Bernstein 条件若单独作为假设，可条件性使用，但不由这段推导自动成立。该文仍需列作近邻风险，不能反过来因其问题宣称本项目首创。

## 8. 延续前轮的直接近邻，避免重复首创声明

[FAR-JEPA（2026，未评审预印本）](https://www.preprints.org/manuscript/202607.2043) 的因子化 AR 规律/方差 signature 与 orthogonal gauge；[Afsari 的联合对角化敏感度分析](https://isr.umd.edu/Labs/ISL/ICA2006/Sensitivity_Final.pdf) 的非正交 contrast/曲率；以及 [iVDFM 前轮 arXiv 指针（2603.22886v2）](https://arxiv.org/html/2603.22886v2) 的条件 innovation/动态结构，沿用前轮核查记录。本轮没有重新复审这些全文，也没有将新 proceedings 地址与该 arXiv 版本作对应认证；不能将其 scope 再扩大到当前 Fine 的有限训练保证。

## 9. 对本项目剩余空间的准确表述

可以研究“给定等价类、真实核、局部性、余项和样本条件，指定标准化慢选集何时可靠”；不能宣称首次把 source identifiability 变成定量理论。对已核查定理而言，没有一个可直接只输入当前存储 CAL NLL，就输出当前实际 Fine/G0 的真实轴与慢选集高概率认证。原因分别是目标不同、真实结构/variation 常数、独立样本、局部入域、优化和实际多时距余项尚未核实。

这最后一句是**本项目适用性判断**，不是“文献不存在任何无真值稳定证书”的普遍命题。当前可以诚实定位为条件理论与真值辅助机制诊断；若要升级为部署证书，必须明确额外可观测前提与不依赖真值的常数/误差估计来源，不能把理想条件当成已测输入。

## 10. Exact regime-posterior 接口的直接原理来源

**Aapo Hyvärinen、Hiroshi Morioka，2016，NeurIPS**，[*Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA*](https://papers.nips.cc/paper_files/paper/2016/hash/d305281faf947ca7acade9ad5c8c818c-Abstract.html)，[全文](https://proceedings.neurips.cc/paper/2016/file/d305281faf947ca7acade9ad5c8c818c-Paper.pdf)。§3/Eq.3 从最优多类 posterior 与真实 posterior 相等导出 conditional log-density ratios；随后以指数族调制/rank 将表示联系到源统计量。它已经覆盖“辅助标签 posterior→密度比”的核心原理，不能将此桥称为新 identifiability。分类准确率或一般校准与完整 posterior 正确不同。

**Aapo Hyvärinen、Hiroaki Sasaki、Richard E. Turner，2019，AISTATS**，[*Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning*](https://proceedings.mlr.press/v89/hyvarinen19a.html)，[全文](https://proceedings.mlr.press/v89/hyvarinen19a/hyvarinen19a.pdf)。Theorem 3 给条件指数族自然参数 contrast 满秩下的充分统计量线性歧义；§5.2 说明可逆 encoded history 与原 history 的辅助信息等价；§5.4 讨论 class labels。一般 Theorem 1 的 variation 不能不加检查地套给 Gaussian location-scale；当前头应走指数族专门结果及 Gaussian 平方闭包，而非声称任意 GCL variation 自动满足。

对当前四维九 regime 头，可以形成以下**既有原理的条件性专门化**。在一个合法固定 context c=(x_t,a_t) 上，所有 regime propensity 正且真实 future 物理状态 V 的条件 law 为正定对角 Gaussian，公共支持为 R⁴；Fine 与物理观测混合的复合 H 是连续双射。若由 head 构造的完整 posterior \(\hat p(u\mid c,x')\) 等于真实 posterior（a.e.），future Jacobian 在 odds 中抵消。若该同一 c 的八个 head 自然参数 contrast 构成可逆 \(\hat L(c)\)，则取 \(T(z)=(z_1,\ldots,z_4,z_1^2,\ldots,z_4^2)\)，有

\[
T(H(v))=\hat L(c)^{-1}L_{true}(c)T(v)+b(c)\quad\text{a.e. }v.
\]

每个 H_j 因而是至多二次的 additive 多项式，其平方又必须为同类至多二次多项式。多项式恒等式强制 H_j 至多 affine；其平方没有交叉单项式，再强制该 affine 行至多一个非零 source 系数。连续双射保证各行/列组成 permutation，系数非零，故得到 axis-affine 恢复。只需一个具备这些性质的合法 c；若只有局部共同支持，结论也只在相应支持上成立。posterior 正确须在该 c 的 regular-conditional a.e. 意义下成立，不能在任意零测 context 上自由赋值后使用 rank。跨不同 c 直接拼八条 contrast 也不足以完成这个固定 c 的证明。

这里将已知 TCL/GCL 密度比原理与 GIN 类 Gaussian 充分统计量闭包用于当前 head；不将其标记为新的总体辨识定理。可计算 head rank 并不自动认证真实 Gaussian 假设或完整 posterior 正确。有限近似 posterior 还需 log-odds 误差控制、稀有类/尾部控制、最小奇异值及多项式近闭包的定量界；一般 ECE、几类 reliability diagram 或低分类错误率均不能代替。当前没有由此得到实际 Fine 的有限样本慢选集证书，也未闭合 approximate posterior→Round 3 余项的接口。
