# Round 4：Fine 独立价值的可成立路线与边界

2026-11-04。只做理论/文献审计，不加载模型或数组，不训练、不连接服务器。执行范围见 [FINE_VALUE_SCOPE.md](FINE_VALUE_SCOPE.md)。本文件的历史数值沿用 [Round 2 完整价值审计](../round2/FINE_VALUE_AUDIT.md)；[Round 3](../round3/ROUND3_REPORT.md) 的条件风险链没有改变那些比较结果。

## 1. 判断

**继续受限可靠性问题，收缩 Fine 方法主张，暂停“Fine 必要/普遍更稳定”的路线。** 可证明的是：一个合适的可逆坐标变换能够降低线性 reader 的近似误差；已有缓存也在若干条件下显示这种可读性增量。但可逆 Fine 不增加观测信息，理想 inverse 存在不说明当前密度目标能以有限网络和有限训练学到它；相对 raw＋线性动态规则的优势也不等于相对同容量非线性 encoder 的优势。

若论文主线固定为“Fine 是不可替代的新方法”，当前材料不足以继续扩大训练。若主线收缩为“在明确条件下，辨识风险、目标几何及非线性余项何时足以保障指定慢选集”，Round 2/3 的条件定理及失败构造仍有研究价值，但不能改称部署时无标签证书。理论工具与若干近邻已有，论文级新颖性仍待确认。

## 2. 可逆表示不增加信息：一个精确命题

设 X 为部署时观测，C 为两方法共同可见的历史、action、regime 等上下文，E 是已知的可测双射，逆可测，Y=E(X)。在固定已训练 E 后，

\[
\sigma(Y,C)=\sigma(X,C).
\]

因此对任意目标 Z 和损失，只要允许所有可测 reader，且风险有定义，

\[
\inf_h E\ell(h(Y,C),Z)=\inf_g E\ell(g(X,C),Z).
\]

证明只需双向复合：g=h∘(E,id)，h=g∘(E^{-1},id)。相应可定义的互信息也相等。条件密度建模所利用的观测信息不因确定性双射而增加。若 E 是训练集 D 的随机函数，这个命题应条件于 D（或两侧都给出 D/部署参数），不能比较一侧已知 encoder、另一侧未知 encoder 的不对称实验。

这是实数精确双射的命题。已保存 FP32 缓存与舍入后的数值实现不必是有限集合上的精确双射；本文件没有据此作逐位无损的信息论声明。可逆结构也没有替算法提供未观测的物理标签。

**改变的是函数类、坐标几何及学习偏置。** 线性、稀疏、对角 Gaussian、逐轴慢度等约束均不在一般可逆变换下保持不变，因此同一信息可以更易或更难被这些受限规则使用。该事实给 Fine 留有价值空间，但不等于已有独立方法优势。

## 3. 同一 reader 到底控制了什么

令 \(\mathcal L\) 为指定输入上的全部仿射 reader，固定 E 后的复合类为

\[
\mathcal H_E=\{x\mapsto Wx_E+b:x_E=E(x)\},
\qquad
\mathcal H_{raw}=\{x\mapsto Wx+b\}.
\]

一般 E 下，这两个类**互不必包含**。Fine 可能改善某个物理目标的仿射可读性，也可能降低；“多了 Fine”不能推出每个目标都更好。只有扩大成可选择的 encoder 类且该类含 identity 时，才能说 raw 仿射类包含在组合类中。

G4 的实际 reader 比 \(\mathcal L\) 更受限：它先在 TRAIN 用固定条件 OLS，再从特征分解产生坐标，按固定谱规则选模态，最后只作匹配及逐轴 scalar-affine 评价。对固定 TRAIN，每个合法 G4 输出都是输入的仿射函数；但任意仿射矩阵不都能由该 G4 的规定估计步骤产生。因此：

1. Fine＋G4 对 raw＋G4 是合法的移除 Fine 组件对照：后处理算法、原门、数据口径相同。
2. 它比较的是“额外学习到的坐标图＋固定动态规则”的整条组件效果。Fine 有 24k 步及 CAL NLL 选点，raw 没有对应 encoder 学习，故端到端函数类及计算不匹配。
3. 有限 CAL 全仿射 oracle 可以诊断空间可读性差异，不能替代 G4 的无标签学习性能，也不能把 CAL 上界推广到总体/TEST。
4. G4 按绝对特征值选模态；主方法 G0 按 H={4,16,32} 的标准化增量慢度选轴。G4 的恢复精度增量不能转记为 Fine＋G0 的独立增量。

## 4. 允许同容量非线性 raw encoder 时的包含关系

令 \(\mathcal E\) 是 encoder 类，\(\mathcal R\) 是 reader 类。无论给方法起什么名字，组合假设类都是

\[
\mathcal H=\{r\circ e:e\in\mathcal E,\ r\in\mathcal R\}.
\]

若 raw 非线性基线允许使用包含当前 Fine 的同一 \(\mathcal E\)，读出也允许同一 \(\mathcal R\)，则每个 Fine 路径 \(\hat r\circ\hat E\) 已在该 raw 基线的函数类中。存在性层面无法再证明 Fine 比这个类更有表达能力；训练目标、优化访问、正则化、样本使用与计算预算可以不同，实际性能仍可能不同。相同参数个数本身不保证类包含；需要架构或明确构造的包含关系。

对 G4 更精确地说：只要基线允许选择任意该 encoder，并调用同一合法 G4 拟合程序，它就能选择 \(\hat E\) 并复现 Fine＋G4 的全部计算。这个模拟命题不要求训练算法真的找到 \(\hat E\)。反过来，换一个所谓“同容量 MLP”却没有证明包含当前可逆流，也不能靠参数量断言相同可达类。

比较归因可据此分三层：

| 比较 | 能隔离的对象 | 仍未隔离 |
|---|---|---|
| raw＋G4 vs Fine＋G4 | 增加整段 Fine 学习后的组件效果 | encoder 表达类、学习预算、密度目标各自贡献 |
| 同一 encoder/reader、不同无标签目标 | 在预先匹配预算/选择规则下的目标增量 | 真实总体优化可达性、所有合理基线 |
| 相同最终函数类的 oracle 最优风险 | 类的最佳可能效果 | 有限样本学习、实际优化与无标签可选性 |

预算匹配至少需要共同训练信息/上下文、独立轨迹数、encoder 架构/约束、reader、优化更新与样本访问、选择集与调参次数。只让步数或参数数目相等不自动完成其余匹配。此表是归因条件，不是本轮授权新增实验。

Full-Fine FlowDirect/C1L 的 raw 坐标预测还可写成 \(E^{-1}\circ P\circ E\)。如果基线的 encoder、predictor、decoder 类允许这条复合并共享逆参数，同样有包含关系；逆映射虽然可共享自由参数，推理计算并不会消失。当前问题的逐轴评价与下游全信息预测必须分别比较，不能用全信息预测成功证明指定慢选集可靠。

## 5. 可以严格证明的条件性线性化增量

对给定总体分布，定义

\[
a_{raw}=\inf_{W,b}E\|S-WX-b\|^2,
\quad
a_E=\inf_{W,b}E\|S-WE(X)-b\|^2.
\]

若 X=f(S)、E=f^{-1}，且二阶矩有限，则 \(a_E=0\)。只要真实 S 不在 X 的仿射闭包中，\(a_{raw}>0\)。后半句要求该有限维仿射空间非退化/投影有定义；不要求、也没有证明有限 Fine 恰好包含 f^{-1}。

一个完全可算的例子是独立中心 Gaussian \(S_1,S_2\)，方差 \(v_1,v_2>0\)，观测

\[
X_1=S_1,\qquad X_2=S_2+\alpha S_1^2.
\]

中心化后的 \(X_2\) 与 X1 不相关，和 S2 的协方差为 v2，方差为 \(v_2+2\alpha^2v_1^2\)。故

\[
\boxed{\sup_{w,b}R^2(S_2,w^TX+b)
=\frac{v_2}{v_2+2\alpha^2v_1^2}<1\quad(\alpha\ne0).}
\]

非线性可逆坐标 \(E(X)=(X_1,X_2-\alpha X_1^2)\) 后，线性 reader 精确恢复两源。这证明了**受限类的近似误差增量**。它不证明学习算法、当前 tanh 观测分布、有限 Fine 类、条件 NLL 可达性、慢因素可靠性或创新性。固定 α 下改变状态分布也会改变 raw 难度，正好说明不能把现有 strong/weak 对照解释为只改 κ。

同样，理想状态上的线性条件均值 \(E[S_{t+1}|S_t,a_t,u_t]\) 经非线性观测后通常不再属于 raw 的线性条件均值类。这里须先取真实条件期望；有噪声时不能写 \(E[f(S')|C]=f(E[S'|C])\)。恢复状态可以降低受限动态拟合误差，但这仍是条件性表达优势；它没有消除 source 规范、有限网络、目标间隔及统计误差的要求。

对固定已学 E 和实际 reader \(\hat h\in\mathcal H_E\)，总体监督风险可恒等分解为

\[
R(\hat h)-R^*
=[R(\hat h)-\inf_{h\in\mathcal H_E}R(h)]
+[\inf_{h\in\mathcal H_E}R(h)-R^*].
\]

第二项是近似误差，第一项包含实际规则限制、估计及优化误差。存在理想 inverse 只约束一个理想 E 的第二项；它不控制当前 \(\hat E\) 的两项。Fine 的训练损失是条件观测 NLL，而非上述物理监督风险，要接通还需要 Round 2/3 那些额外条件。

## 6. 现有证据允许保留的独立增量

| 已有事实 | 合法结论 | 不合法升级 |
|---|---|---|
| 4 个 strong/.25 cohort 的 Fine＋G4 三 seed 均值高于 raw；相应 CAL span 一致改善 | 这些分布上的恢复精度/仿射可读性改善 | 普遍改进、统计显著、Fine 必要 |
| 4 个 weak/.25 cohort 的 Fine＋G4 三 seed 均值均稍低于 raw | 负例确实存在，效果有条件依赖 | 删去为“训练噪声”，或反推全部 weak 不可能受益 |
| weak/1.0 同两根上均改善，第二根较大 | 固定观测干预下的精度线索 | 新独立确认、raw 无法过原门 |
| raw 10/10 通过原门，Fine 30/30 | 既定任务门下没有必要性证据 | 将更高精度改称任务不可替代 |
| 10 cohort 来自 4 个底层轨迹/mix 根 | 有限受控证据 | 把优化 seed 或重复观测干预当独立系统 |
| strong/weak 改变状态分布及 raw CAL 线性难度 | κ 效应与观测难度未分离 | 唯一归因为条件辨识强度 |
| FHN 的 G4 在 TRAIN 复谱门 N/A | 该规则在这些父上未适用 | N/A 算 raw 错误/Fine 优势 |
| 单边 Fine 扰动稳定性、无同对象 raw 面板 | 相对稳定性仍未建立 | raw 无优化 seed 就比 Fine 更/不稳定 |

最强现有表述仍是：**Fine 在若干 raw 线性可读性较弱的受控观测分布上，提高了相同 G4 动态坐标规则的恢复精度；增量有条件依赖，尚未建立必要性或普遍稳定性优势。** 数值范围和逐父负例完整留在 Round 2，不以新理论覆盖原分母。

## 7. 文献能接通哪些段，不能接通哪些段

定向 primary 核查详见 [PRIMARY_NEIGHBORS.md](PRIMARY_NEIGHBORS.md)。已存在有限样本 nonlinear ICA、非线性慢源恢复、条件 Gaussian flow 辨识和 quantitative near-identifiability；这些方向不能作为本项目首创。iVAE Theorem 4 是真 posterior 族包含、最大化目标与无限数据下的等价类一致性。[iVAE 原文](https://proceedings.mlr.press/v108/khemakhem20a/khemakhem20a.pdf)

对本项目所需“无需物理真值的稳定证书”，本轮区分三类对象：

- 有限样本 source-identification 定理：具有实际内容，但要核查目标、独立样本、函数复杂度、真实 variation/平滑常数及误差等价类。[Lyu–Fu 2022](https://proceedings.mlr.press/v162/lyu22a.html)
- 条件流的可计算预测校准：可以不使用潜在物理真值，却认证不同对象；latent norm 校准不确定坐标方向，旋转保持范数就是直接边界。[Dheur–Ben Taieb 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/079923016ab82f373c0a4a080b2f3ee6-Abstract-Conference.html)
- 真实慢选集的稳定性：还需真实因素等价类、标准化、间隔、动态交叉矩、统计支持与实际非线性余项。conditional likelihood 或 cross-run 一致性不能单独替代这些条件。[Nelson 等 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f67e5f99b23b108a3a6665f410034bcd-Abstract-Conference.html)

“本轮核查的定理未直接覆盖当前实际 Fine”是适用性判断；不能写成“所有现有方法均未解决”或“我们首次实现稳定认证”。Round 3 的正结果依赖真值核心/创新方差/有限数组相关矩，仍是机制诊断与条件理论；它不是上述最后一类部署证书。

本轮分析的 regime-posterior 接口值得保留为一个**精确端点**：若在同一个合法 context 上，完整真实 posterior a.e. 正确、八维 head 自然参数 contrast 满秩、真实对角 Gaussian/full-support 及双射条件均成立，则 posterior odds 消去共同 future Jacobian，经充分统计量平方闭包可得到 permutation＋axis-affine。详细假设与推导在 [近邻核查 §10](PRIMARY_NEIGHBORS.md#10-exact-regime-posterior-接口的直接原理来源)。该端点是 TCL/GCL 与 Gaussian closure 的专门化，不是新总体辨识原理。它也没有证明一般校准即 posterior 正确，更没有给出有限近似 posterior 到实际慢选集余项的量化界；因此暂不改变本文件对实际 Fine 独立价值的结论。[TCL 2016 §3](https://proceedings.neurips.cc/paper/2016/file/d305281faf947ca7acade9ad5c8c818c-Paper.pdf)，[GCL 2019 Theorem 3](https://proceedings.mlr.press/v89/hyvarinen19a/hyvarinen19a.pdf)

## 8. 继续、收缩与停止建议

**继续**：保留固定物理规范下“何时因素辨识风险足以转成指定慢选集可靠”的问题；用 exact shear 失效半径、一般仿射敏感度及实际余项说明必要条件和可证范围。贡献候选是面向特定标准化目标的明确闭链及失败边界，须与经典工具和近邻逐项比较。

**收缩**：Fine 的独立价值现阶段只写条件性精度/可读性增量；把可逆性、source 辨识、稳定性和预测拆成不同命题。若仍以 Fine 为方法贡献，必须先确定一个能与同类非线性 encoder 区分的正式命题和预算/选择协议；否则 raw 线性比较只能证明引入非线性坐标有时有帮助。

**停止扩张**：不再从现有数据主张 Fine 必要、weak 必有益、G4 证明 G0 增量、跨运行一致就是真实因素正确、或 CAL oracle 已证明测试能力上限。当前不以这些理由进入 GPU/服务器；新增训练不会自动填补类包含性及比较归因缺口。本轮没有授权或执行新增比较。
