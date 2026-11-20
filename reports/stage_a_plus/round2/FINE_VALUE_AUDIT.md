# Round 2：Fine 相对 raw＋同规则动态分析的独立价值审计

日期：2026-11-04。承接最新 Proposal 2.10–2.14、[Stage A 证据审计](../../../outputs/HCF_STAGE_A_20261104/evidence_audit.md)和[Round 1 理论 §7](../THEORY.md)。本轮只读已有源码、配置、已暴露结果标量与缓存 CAL 数组；没有神经训练、模型加载/前向、新物理数据、GPU、FINAL/TEST 数组读取或原结果覆盖。

## 1. 独立判断

**已有证据支持的是特定条件下的因素恢复精度增量；不支持 Fine 必要性、对所有 weak 条件有益、或相对 raw 的普遍稳定性优势。** 完整同规则比较并不只含 Proposal 表内的两个新实例：本轮定位到开发2根＋确认2根，strong/weak各配对；再加确认两根的单次 strength=1.0 干预，共10个cohort、30个Fine优化父与10个确定性raw父，但独立底层轨迹/混合根仍只有4个。

常规观测 strength=.25 下，四个strong cohort的 Fine＋G4 均比raw＋G4精度高；四个weak cohort的Fine＋G4三seed均值则全部低于raw。strength=1.0的两个weak干预中，Fine＋G4均提高精度，尤其第二根；但这10个raw cohort全部通过原四轴门，Fine＋G4也全部通过。不能把精度改善改称完成该既定门必须有Fine。

新增 CAL 仿射空间诊断还表明：**strong/weak同时改变了观测的有效线性可读难度**。固定同一mix与strength=.25，各根strong状态的raw CAL仿射恢复上限均低于weak状态。这与τ同时改变均值kick、方差kick和状态分布一致。因此既有配对不是“只改变辨识强度、其他难度全部不动”的干预；不能把Fine增量差异唯一归于κ。

## 2. 实际比较对象与规则核查

| 项目 | Fine＋G4 | raw＋G4 | 结论 |
|---|---|---|---|
| 输入 | 同一cohort观测经锁定Fine编码的完整4维缓存 | 相同cohort原始4维观测 | 移除Fine的组件对照；不等计算预算 |
| Fine学习 | 每父24k、观测CAL NLL选点，3优化seed | 无Fine/无优化seed | raw不能复制成3个独立父 |
| 动态拟合 | TRAIN-only：标准化状态4维＋action2维＋regime one-hot9维的15→4 OLS | 同一函数与固定常数 | 无额外ridge、超参救援或物理标签 |
| 坐标规范化 | 行转移矩阵的右特征向量；按特征值绝对值降序；TRAIN mode方差归一化 | 相同 | 单个固定G4规则，未改为学习器 |
| 时间/数值 | 原生t0全部合法一步pair；FP64拟合与变换，FP32输出 | 相同 | 不能混入旧common t≥11结果 |
| 适用性门 | std≥1e−9、设计rank15、虚部≤1e−10、特征值距离>1e−8、基条件数≤1e8 | 相同 | N/A保留分母，不按结果放宽 |
| 轴评价 | CAL一一匹配及逐轴scalar-affine，锁后在既有机制TEST评分 | 相同 | 当前审计仅读取这些已暴露分数 |
| 原门 | mean R²≥.95、worst≥.90＋原rank/命名条件 | 相同 | 本轮未改变阈值 |

代码依据为 `work/hcf_stable_id_20261028/conditional_dmd_eigenframe_v1/math_core.py`。raw开发runner直接调用同一个 `F.fit`；新raw控制通过AST复用原数值函数，新确认 Fine runner 的 `CANON_SHA=4634618363f8348e6907a50a8230558eceb09fbc03536ca389548ffb6ebc886f` 固定该数学源；strength1桥继续用该canonicalizer。没有在本轮重新运行这些函数。

G4的“慢坐标”是按绝对特征值取前两模态，**不是G0在H={4,16,32}上排序的同一个选择规则**。本文件比较Fine＋G4与raw＋G4是合法的同后处理对照；不能将这个增量直接记成当前Fine＋G0主方法增量。G4产生新坐标也不能将原weak Fine轴门改成通过。

## 3. 全部有效AR同规则对照

下表只重新汇总保存的已暴露TEST标量。Fine列为同一cohort三个优化seed的均值，raw列每格仅一个确定性父；Δ是R²绝对差，不是百分点外的风险减少，未做显著性检验。

| 物理/混合根与阶段 | 条件 | strength | Fine＋G4 mean R² | raw＋G4 mean R² | Δ Fine−raw |
|---|---|---:|---:|---:|---:|
| paired_ar_fresh1，开发 | strong | .25 | .998093006 | .997238164 | +.000854842 |
| paired_ar_fresh1，开发 | weak | .25 | .998168325 | .998981401 | −.000813076 |
| paired_ar_fresh2，开发 | strong | .25 | .998897379 | .988535346 | +.010362033 |
| paired_ar_fresh2，开发 | weak | .25 | .997868509 | .998001649 | −.000133140 |
| dmd_confirm_ar_fresh1，确认 | strong | .25 | .999192742 | .997626751 | +.001565991 |
| dmd_confirm_ar_fresh1，确认 | weak | .25 | .996594229 | .997085002 | −.000490773 |
| dmd_confirm_ar_fresh2，确认 | strong | .25 | .998801888 | .995731501 | +.003070388 |
| dmd_confirm_ar_fresh2，确认 | weak | .25 | .998576973 | .998607101 | −.000030128 |
| 同确认fresh1，已暴露weak观测干预 | weak | 1.0 | .994102936 | .992664902 | +.001438035 |
| 同确认fresh2，已暴露weak观测干预 | weak | 1.0 | .998011267 | .976003154 | +.022008113 |

已保存原门：开发Fine＋G4 12/12、raw4/4；新确认Fine＋G4 12/12、raw4/4；strength1干预Fine＋G4 6/6、raw2/2。合计30/30、10/10分别是模型父/cohort计数，**不是40个独立环境**。两强度的weak分数共享同一底层两根，strong/weak也共享随机原语/mix。

三个seed的方向并非每格都完全相同：weak/.25开发fresh2和确认fresh2各有一个Fine seed稍高于同一次raw，尽管三seed均值低于raw；另外两个weak cohort三seed都低于raw。strong四格和strength1两格则各自三seed全部高于其raw。完整逐父值、每轴R²、raw worst、原目标方差以及输入SHA见[FINE_VALUE_AUDIT.json](FINE_VALUE_AUDIT.json)。这些seed是条件于相同轨迹/观测的优化重复，不能独立检验“跨系统有益概率”。

strength1的两实例 `1−meanR²` 描述性减少仍为19.60%、91.71%；相应raw worst=.984509、.944324，均高于原.90门。开发strong fresh2也有较大精度差，但其raw worst=.963895，同样过门。不能只列最大的相对误差减少而省略raw的成功。

### 结果来源

- 开发Fine：`outputs/HCF_STABLE_ID_RUN_20261028_V2/conditional_dmd_eigenframe_v1/RESULTS.json`；raw：`conditional_dmd_raw_obs_v1/RESULTS.json`。
- 新确认Fine：`dynamic_eigenframe_confirmation_eval_v1/RESULTS.json`；raw：`dynamic_eigenframe_fresh_raw_control_v1/RESULTS.json`。
- 单次强观测：`weak_AR_observation_intervention_eval_v1/RESULTS.json`及保留58/65来源差异的 `weak_AR_observation_intervention_recovery_v2/ALL6_FINE_AND2_RAW_G_LOCK.json`。

原独审分别覆盖开发、新确认与强观测的operator/映射/评分算术；本轮没有重新重建全部G4或读取其TEST数组。当前新确认也已经暴露，不能再次称为未触碰的确认集。

## 4. 观测强度到底改变了什么

原始 `observation_transform` 先施加随机正交混合，再串联4个可逆tanh shear，最后逐轴scale/offset：每层更新一个坐标为

\[
x_i\leftarrow x_i+\alpha\tanh(c x_j+b),\quad i\ne j.
\]

c的绝对值在[.7,1.3]，bias在[−.2,.2]；本轮历史干预只把α从.25固定到1.0，同一组q/shears/scales/offset。α不是innovation std，也不是可辨识参数τ；“1.0比.25更大”不等于函数在实际状态支持上的不可线性恢复误差必然按4倍变化。tanh在部分区间接近线性，多层组合的有效难度还依赖状态幅度和相关性。

强观测样本实际来自

\[
x^{(1)}_{\rm FP32}=\operatorname{FP32}\left[g_{1.0}\left(g_{.25}^{-1}(\operatorname{FP64}(x^{(.25)}_{\rm FP32}))\right)\right],
\]

并复用原注册物理标签、action与regime。它没有生成新latent轨迹；同时也不是从原未舍入FP64状态重新观测的逐位等同数据。历史roundtrip和量化误差门已保存，本轮没有重新调用逆映射或随机参数生成器。

所以本轮可以说“同底层weak机制、固定更大观测shear系数下出现精度增量”，不能说“新独立环境已证明任意强非线性下Fine必要”。

## 5. 新增CAL仿射空间oracle：可以建立的有限上界

### 5.1 定义、合法范围与读取纪律

本轮脚本 [fine_value_audit.py](fine_value_audit.py) 只读取10个raw CAL、10个相应物理CAL与30个缓存Fine CAL，共50个member。每个CAL均32条×129时点×4维，均为已存在数组，记录文件及typed-array SHA。脚本逐member白名单读取，不打开任何TEST/FINAL数组、不导入模型、不加载PT、不调用PRNG。没有为模型选择提供物理标签。

对同一CAL，令X=[1,中心化/标准化raw]，S为真实物理值。用固定rcond=1e−12、rank5的最小二乘投影P_X，定义逐轴

\[
R^{2,\mathrm{raw\ span}}_j
=1-\frac{\|(I-P_X)S_j\|²}{\|S_j-\bar S_j\|²}.
\]

这是**这些有限CAL点上、任意点态仿射reader可取得R²的上界**。固定的raw G4＋scalar-affine readout是该类的子集，因此同一CAL口径下不能超过这个oracle（舍入容差除外）。它不是总体或TEST上界，也不是未标注学习器能达到的保证；CAL已用于原Fine选点，不能将它当新确认。

同样在Fine缓存上计算全affine projection，用来判断当前表示是否扩大/改善了相对真实物理因素的线性可读空间。两个oracle都使用同CAL真值，所以差异是表示空间的事后诊断，**不是无标签认证、机制归因或正式训练成绩**。

### 5.2 全10格结果

| 根 | 条件/strength | raw CAL最佳affine mean R² | raw最差轴 | Fine CAL最佳affine（三seed均值） | Fine−raw |
|---|---|---:|---:|---:|---:|
| 开发fresh1 | strong/.25 | .998735147 | .996544412 | .999145575 | +.000410428 |
| 开发fresh1 | weak/.25 | .999797858 | .999354780 | .999137170 | −.000660688 |
| 开发fresh2 | strong/.25 | .989713977 | .965955118 | .999187020 | +.009473043 |
| 开发fresh2 | weak/.25 | .998584174 | .995697573 | .998378564 | −.000205611 |
| 确认fresh1 | strong/.25 | .998343124 | .996820825 | .999522333 | +.001179209 |
| 确认fresh1 | weak/.25 | .999901930 | .999790606 | .999466295 | −.000435635 |
| 确认fresh2 | strong/.25 | .996384695 | .990937991 | .999328067 | +.002943371 |
| 确认fresh2 | weak/.25 | .999152715 | .998149288 | .999159924 | +.000007209 |
| 确认fresh1 | weak/1.0 | .996060926 | .988909957 | .997900303 | +.001839377 |
| 确认fresh2 | weak/1.0 | .974569112 | .937989796 | .998432068 | +.023862956 |

两个结果值得保留。第一，强观测fresh2存在明确的**有限CAL仿射可读性增量**：raw同CAL全部affine reader的mean上限约.97457，缓存Fine的affine oracle约.99843。第二，weak/.25不是Fine改善线性化的统一正例；三根反而降低，另一根仅约7.2×10⁻⁶提高。

所有raw CAL affine oracle的mean/worst都仍高于原数值门。即便把这些oracle当作更强的有标签能力参照，也没有出现“在这些已存CAL上，所有raw affine读出必定达不到原门”的证明。不能用上限低于Fine精度但仍高于门的事实，改称raw任务不可完成。原G4的无标签拟合与命名/rank门还必须另行检查，本节不改变其原通过数。

### 5.3 Strong/weak的观测难度混杂

四根strong raw CAL affine mean上限分别约.998735/.989714/.998343/.996385；各自weak上限提高到.999798/.998584/.999902/.999153。这个同向事实说明，τ弱化后，同一非线性观测族在实际状态支持上更易用线性reader恢复。

源码中τ同时缩小逐轴mean kick(.6→.006)与std倍数(3→1.02)，因此改变state分布；本轮未将此与辨识signature的贡献分离。不能只用strong/weak的Fine＋G4差异宣称“Fine尤其在强/弱κ下有独立作用”，也不能据这些CAL值声称所有weak轴失败均已由观测难度解释。

一个严格但**不冒充实际tanh生成器**的解析例说明这种混杂可能发生：独立中心Gaussian s1、s2的方差为v1、v2，观测x1=s1、x2=s2+αs1²。因中心化平方项与s1、s2不相关，任何raw affine reader恢复s2的最佳总体

\[
\boxed{R^{2,\mathrm{raw\ affine}}_{s_2}
=\frac{v_2}{v_2+2\alpha²v_1²}.}
\]

α完全不变，v1/v2改变也会改变raw线性难度；非线性inverse仍精确恢复。它推广了Round1的v1=v2=1例，说明“固定观测函数”不保证“固定观测恢复难度”。这不是对当前两个AR/FHN数据分布的已证明定量解释。

## 6. FHN与高维：不能拿不适用当优势

FHN历史六Fine父的G4在TRAIN实谱检查即因复共轭特征值拒绝，保留N/A；没有进入CAL/TEST物理评分。当前记录不能构成“Fine＋G4成功、raw＋G4失败”的FHN对照，更不能把复谱N/A当raw恢复R²=0。Fine在rich FHN能恢复原轴是另一条有限证据；与raw同规则动态分析的独立差异尚未由此建立。

同速旋转负控中的dense4/slow2高分而轴门失败表明总体对称性/评价对象的区别，不能按“加Fine后模型可重复”宣称独立贡献。Lorenz96/Burgers的完整C1L预测也没有形成全面raw优势；Burgers无独立轴标签。它们不补足Fine的因素恢复必要性。

本轮也没有得到“Fine比raw更稳定”的新证据。已有Fine换mix/替换TRAIN稳定性缺少相同raw-G4扰动面板；raw每cohort的确定性意味着没有优化seed波动，却不说明其统计或mix稳定性。两边的稳定性对象必须匹配后才能比较。

## 7. 经典ISFA与nonlinear source recovery的实际约束

[Independent Slow Feature Analysis and Nonlinear Blind Source Separation](https://www.ini.rub.de/PEOPLE/wiskott/Publications/BlasZitoWisk2007-ISFA-NeurComp.pdf)，Blaschke、Zito、Wiskott，Neural Computation 2007，结合second-order ICA与SFA研究非线性盲源分离。作者全文也明确指出time-delayed second-order independence不足以保证其案例中的源分离；因此不能把“用动态二阶统计规范化坐标”无条件推广成因素唯一性。这里的ISFA是**Independent**，不是information-preserving小写iSFA。[原NeurIPS 2004论文](https://proceedings.neurips.cc/paper_files/paper/2004/hash/47fd3c87f42f55d4b233417d49c34783-Abstract.html)已公开这一组合方向。

[An Extension of Slow Feature Analysis for Nonlinear Blind Source Separation](https://jmlr.org/papers/volume15/sprekeler14a/sprekeler14a.pdf)，Sprekeler、Zito、Wiskott，JMLR 2014，正文§3在独立源及其速度联合分解、平稳可微等条件下分析unrestricted function space的SFA，讨论产品/harmonic输出、源选择及有限相关时间影响。其Gaussian情形中部分最优函数可以对应原源，非线性观测反演与slow source recovery显然不是Fine首次提出的价值。

这些结论不能直接套到coupled FHN或离散Gaussian-kicked状态上；函数类无限、独立性、可微样本路径以及等价类都要重查。Fine可争取的剩余价值应是：在给定真实条件核假设、有限函数类和预先规定扰动范围内，相比同规则raw/其他非线性source recovery，展示独立精度或可靠性增量。仅用理想非线性inverse优于raw线性类的shear例不足以确立方法创新或学习可行性。

## 8. 对Stage A+的决策输入

| 可争取主张 | 当前支持 | 不能据此声称 |
|---|---|---|
| 在特定rich AR条件下提高同规则动态轴恢复精度 | 开发与新确认4个strong cohort方向一致；CAL span诊断一致 | 普遍优势、显著性、必要性 |
| 较强固定观测shear下改善弱AR精度 | 同两根strength1.0均改善，第二根较大；CAL线性span显示表示增量 | 新独立确认、任意非线性混合适用域、raw无法过门 |
| weak下Fine必需或必有益 | **不支持**；.25下4个均值均稍差于raw，raw全过门 | 以弱辨识作为Fine既有独立优势 |
| Fine拓展无标签轴可靠性的条件范围 | 目前只有候选机制线索 | 已证明新的可靠区域/实际flow的高概率认证 |
| Fine提高跨mix/样本扰动稳定性相对raw | 缺少同对象raw稳定性对照 | 用raw无seed或Fine单边稳定支持相对优势 |
| Fine是不可由简单方法替代的组件 | **目前没有这样的已建证据** | 把保持更高R²精度改成不可替代 |

当前最强可保留的独立价值表述为：**“Fine在若干具有较低raw线性可读性的受控观测分布上改善了相同动态坐标规则的恢复精度；该增量具有条件依赖，尚未建立必要性或普遍可靠性优势。”** 这是一项真实有限证据，足以指导科学命题，但不能单独填补ICML理论/方法创新缺口。继续研究时需先明确精度、稳定性、适用域或必要性中究竟哪一项是正式主张；本轮不新增模型或实验矩阵。

## 9. 可复核性

[FINE_VALUE_AUDIT.json](FINE_VALUE_AUDIT.json)保存10格原结果差异、全部30父Fine CAL oracle、10格raw CAL oracle、每轴值、OLS rank及normal-equation残差、50次CAL member读取和59个输入文件SHA。脚本只做固定线性代数诊断，无超参网格；所有计算结果均保留。CAL oracle的性质是事后有限数组projection，不能与原heldout分数混列为同一评价，也不能用于改选checkpoint、G4或门。
