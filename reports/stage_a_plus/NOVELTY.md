# Stage A+ 新颖性判断与贡献筛选

日期：2026年11月4日。结论：**本轮有新增可证明的精确机制结果，但未确认足够强的ICML理论或方法创新。** 完整逐篇正文核查与定理位置见[NEAREST_WORK_REVIEW](NEAREST_WORK_REVIEW.md)。下列“新增”只表示相对本项目Stage A新增，除非另有证据，不表示文献首次。

## 1 最邻近覆盖已经比Stage A更强

| 拟议主张 | 已核查的直接先行内容 | 本项目可以留下什么 |
| --- | --- | --- |
| Gaussian条件变化使物理轴affine可辨识 | [GIN Appendix A–B](https://arxiv.org/pdf/2001.04872)已有充分统计量及平方闭包推理 | 当前真实条件头、context与支持的适配审计；不是新辨识原理 |
| 过程签名κ决定轴恢复 | [FAR-JEPA §5 Eq13、Theorem2、AppendixA.2](https://www.preprints.org/manuscript/202607.2043)已有pairwise process gap、同签名块与旋转边界 | dense均值与diagonal AR的不同；不能把定义pair gap说成首次 |
| profile Gaussian NLL曲率/误差界 | [Afsari 2008 §3.2.3 Eq3.11、§4.3](https://isr.umd.edu/Labs/ISL/ICA2006/Sensitivity_Final.pdf)正是该logdet对比与方差比敏感性；[SOBI Theorem1(ii)](https://arxiv.org/pdf/1405.3388)已有签名差分母的轴误差 | 当前同一gauge下的闭式risk-to-目标排序桥；基础曲率与局部信息不是原创 |
| ICA＋慢特征能恢复源 | [ISFA 2007](https://www.ini.rub.de/PEOPLE/wiskott/Publications/BlasZitoWisk2007-ISFA-NeurComp.pdf)、[xSFA 2014](https://jmlr.org/papers/volume15/sprekeler14a/sprekeler14a.pdf)早已结合慢度与非线性BSS，并讨论采样/相似慢度问题 | 先条件辨识再选预声明物理目标的窄问题；不能泛称首次连接ICA和慢表示 |
| 只需跨块辨识，不需块内轴 | [Cai与Li，AISTATS2021](https://proceedings.mlr.press/v130/cai21a.html)及[2019块扰动理论](https://arxiv.org/pdf/1703.00591)已有块唯一性与扰动模量 | 目标对等价群是否well-defined的澄清；不能降低本研究四轴标准 |
| 条件创新＋结构动态＋预测 | [iVDFM 2026 v2 §3、AppendixA/B](https://arxiv.org/html/2603.22886v2)已是直接近邻 | Gaussian/dense-mean设定须自己证明；不能借不匹配的nonGaussian传播结论 |
| 稳定与真值对齐有区别；评价等价类重要 | [ICLR2026 statistical/structural identifiability](https://arxiv.org/abs/2603.11970)、[UAI2026 Who Guards the Guardians](https://proceedings.mlr.press/v337/joshi26a.html)已明确研究 | 与指定慢目标的具体定量失效分析，而非区分本身 |

FAR-JEPA当前公开v1为2026年7月28日，未同行评审；iVDFM当前arXiv v2为2026年10月18日，另有PMLR会议版本。两者的模型假设、版本与理论范围已经按正文核查，不以摘要判断“别人没有做”。ISFA/xSFA及经典JD是本轮补齐的更早直接近邻，Stage A的宽泛创新空间因此进一步缩小。

## 2 本轮新增结果的正确定位

### N1 精确risk-to-物理pair排序边界

[THEORY T3–T4](THEORY.md)证明

\[
g(\theta)=\frac{\gamma\cos2\theta-\chi\sin2\theta}{1-\rho^2\sin^2 2\theta},
\qquad
r_*=F\!\left(\frac{\gamma^2}{\gamma^2+\chi^2}\right).
\]

受限二维风险球内，全部角度符号下的严格排序保证恰为r<r*。数学上已完成，并有必要性方向；强于只列γ>2ε的后半链。它是profiled Gaussian联合对角化与两个归一化Rayleigh商的精确专门化，尚无依据称为新的普遍学习理论。

### N2 同整个F、同γ、不同失效边界的动态反例

三个合法conditional Gaussian VAR均保持state covariance为I，创新及完整F相同，γ=.01；因为χ不同，临界角为45°/22.5°/7.5°。这排除了“只要κ和γ确定，sharp boundary就确定”的过强故事，而且不依赖不可实现的任意矩阵拼接。它没有排除仅用κγ加全局矩界给保守保证。

**这是本轮最值得保留的理论增量。** 但精确动态几何方向导数是经典对象，尚缺对实际非线性学习器的非平凡延伸与新实例失败预测，因此不把它当成创新已确立。

### N3 FHN旧界的可定位松弛

[全12父相关矩审计](PAIRED_SCORE_AUDIT.md)显示：pair化范数界仍为6/12；读取真值signed cross-moments后才为12/12。原因主要是实际误差与真实增量的L²余弦远小于Cauchy最坏值，而非所有成功都靠跨h相消。它定位了旧证明的损失步骤，具有科学诊断价值；所用恒等式与三角界没有独立重大新颖性，而且不是无标签认证方法。

## 3 当前最强的三个贡献候选

| 候选 | 状态 | 当前可发表措辞的上限 | 不能承诺 |
| --- | --- | --- | --- |
| 指定目标下的辨识风险—耦合几何—速度间隔精确边界 | **Proved in restricted model；Novelty Unresolved** | 给受限族sharp boundary及合法same-F/same-γ反例 | 任意Flow有限样本学习保证、完整4D唯一风险半径 |
| 从粗范数到方向相关矩的完整失败审计 | **Already Supported on existing arrays** | 全12父并列；旧oracle错选保留；解释哪些松弛导致空界 | 无标签证书、新独立确认、方法认证率提高 |
| rich受控条件下稳定轴恢复及其失效机制 | **Empirically Supported, limited scope** | 4实例12父与有限扰动、weak/raw/G4/复谱/旋转边界共同报告 | 普遍恢复、Fine必要性、跨高维优势、独立原创算法 |

C1L的完整预测配对收益可作为用途评价，暂不列为可靠慢语义论文的核心贡献。Fine、G0和C1L中，**目前没有一个组件被证据确立为不可由简单适用方法替代**：Fine存在局部精度价值但raw仍过门；G0是固定排序；C1L收益属于完整预测。理论上的非线性表达能力差异也不独属于Fine。

## 4 研究路线的取舍

保留R1作为首选有界研究路线，理由是现在已有精确边界、必要性反例和旧数组中的相关几何，而不只是待证明的长链。R2作为定义/不可能性补充，避免转向子空间后暗中降低标准。R3保留为方法贡献的硬问题：如果无法找到Fine可被独立检验的增量，就不能以GPU复现更多分数来填补。

当前证据足以继续研究**耦合动态下为何相同因素辨识精度会产生不同慢目标选择风险**。尚不足以保证它能成为ICML论文；如果只能保留二维恒等式、经典曲率和真值事后界，应把这些作为机制分析，停止将固定模块组合作为强方法创新。

## 5 本轮没有跨越的边界

文献核查是有针对性的近邻全文复核，不是全领域不存在性证明。对于iVDFM的Gaussian closure与innovation→state传播，本轮只指出不能直接借用的数学前提，未声称完整论文定理已被反驳。对于FAR-JEPA，承认其orthogonal起点和process gap覆盖，不把其结果误套当前dense-mean Fine，也不因预印本状态忽略其先行性。
