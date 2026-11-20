# Round 6：当前实际 Fine 的可观测输入与相关矩证据审计

2026-11-04。按 [固定范围](CURRENT_INPUT_SCOPE.md) 只读指定confirmation_v2的84份diagnostics、12份Fine.json、12份G0.json及拥有源码，并复核两份旧JSON结果。没有读取科学数组或checkpoint，没有模型执行、新拟合、随机采样、训练、服务器或GPU操作。

**结论：当前指定记录仍不足以实例化后验KL上界或无真值动态相关结构证书。** 这不是仓库全局不存在相关信息的断言，也不是这些量原则上不可观测。旧相关矩链确有数值作用，但其输入依赖物理真值，不能直接转作接受式无标签证书。

可重现schema、全部110份JSON输入及3份源码的SHA256和24个旧情境重计数见 [CURRENT_INPUT_AUDIT.json](CURRENT_INPUT_AUDIT.json)，实现为 [current_input_audit.py](current_input_audit.py)。计数是单次脚本执行读取数，前置人工阅读不计入该字段。

## 1. 84份snapshot记录并未保存后验误差所需的信息

审计范围固定为四slot（AR rich fresh1/fresh2、FHN ε=.08/.5）×三优化父×七snapshot。共有12父、4个物理实例；84份诊断不是84个独立环境。七个step为1500、3000、4500、6000、12000、18000、24000。

84份记录的顶层、TRAIN/CAL、逐regime和bounds schema分别完全相同。每份TRAIN/CAL记录2048个sampled pairs，包含九regime聚合残差均值、方差、坐标相关，以及残差与current Fine/action的4×6 Pearson矩阵。每父七snapshot使用相同sampled pair indices。CAL来源为AR每父32条、FHN每父24条轨迹；2048不等于独立样本量。CAL参与七snapshot的NLL选点。

| 所需对象 | 指定记录实际保存 | 本轮可成立的判断 |
|---|---|---|
| 每个(c,x′)的九类posterior/odds/logits | 未保存；只有实际u下NLL的聚合值 | 不能重建posterior log-loss或bin校准表 |
| 同一pair上全部regime条件likelihood | diagnostics只计算实际u的transition | 不包含后验归一化所需counterfactual-u项 |
| posterior proper-risk excess的有效上界 | 无Bayes风险下界、全局优化gap或覆盖证明 | 当前CAL NLL不能直接填A |
| 分箱/局部条件校准 | 无bin×regime计数、bin内posterior波动上界 | 不能给分箱误差上界，更没有从分箱到完整posterior的余项 |
| learned自然参数contrast强度 | 只有参数hash、范数与logstd范围 | 实际最小奇异值未测，源码结构兼容不等于实际强度下界 |
| anchor尾矩、协方差下界及Q0密度比尾矩 | 无相应总体/置信预算 | 现有残差二阶矩不提供这些对象 |

关键源码：[`diagnostics`](../../../work/hcf_method_freeze_20261025/fine_development.py) 第225行调用 `model.transition(y,a[r,t],u[r,t])`，即实际u；第253–258行只写聚合诊断。`fit_one` 第414–424行用同CAL诊断选择最低NLL snapshot。[`run_fine`](../../../work/hcf_stable_id_20261028/confirmation_v2/ours_bridge.py) 第110–113行保存的FEATURES仅为TRAIN/CAL Fine，再构造G0；本轮未打开FEATURES文件。

这并非可以由实际u的NLL均值补齐：posterior log-loss还包含每个pair的 `log Σ_v π_v h_v(x′|c)`，而当前记录既无该归一化项，也无逐pair的全部h_v。聚合后的NLL与残差矩不能恢复它。

## 2. 已有“可观测相关矩”与需要的误差相关矩不同

当前诊断里的Pearson矩阵对象是**learned head标准化残差与当前Fine/action**，不是Fine相对真实物理因素的多时距误差，也不是任意相容解释与Fine之间的差异。

12份G0.json只记录各轴TRAIN中心/尺度、平均速度、FP32排序和选集等。它们足以追溯当前有限TRAIN的选择，却没有完整跨时距cross-error矩或候选解释的联合分数修正区间。源码明确其estimand为 `adaptive_unclipped_TRAIN_slowness_not_population_certificate`。

若用完整相容生成模型集合替代显式真值，则每个候选解释与Fine之间的差异原则上可在观测空间计算；但还须证明真实解释包含在集合中，并对**全体**候选的signed pair correction给统一界。现有三seed只是已找到的三个点；未提供完整集合、覆盖或support-function上界。它们的共识不能代替这些条件。

## 3. 保留相关结构确实改善过旧诊断，但属于oracle证据

本轮只从 [Round3保存JSON](../round3/CHAINED_SCORE_AUDIT.json) 重新逐项核对24个parent/split情境、96个pair，没有重读旧数组或重拟合。对每pair检查 `affine_lower_bound − stored_remainder_bound = chained_lower_bound`，以及该下界不大于已存exact实际gap；均在1e−12数值容差内一致。

| 旧闭链 | TRAIN父覆盖 | CAL父覆盖（固定TRAIN尺度） | TRAIN pair覆盖 | CAL pair覆盖 |
|---|---:|---:|---:|---:|
| 已知仿射下界＋Cauchy评分余项 | 9/12 | 8/12 | 36/48 | 35/48 |
| 已知仿射下界＋真值相关矩评分余项 | 12/12 | 9/12 | 48/48 | 36/48 |

三份FHN ε=.5 TRAIN模型的相关矩闭链最小下界分别约+.009121、+.008626、+.014600，Cauchy版本均为负。相关矩值得保留的证据来自**同一批固定模型上它能避免部分最坏方向松弛**，不是新增科学确认，也不证明学习算法在新实例上满足该界。

这条旧链使用真实物理轴拟合的A、真实创新D、物理Σ/M及core-to-Fine余项。它并非观测NLL→实际Fine的总体链。其相关预算也仍对若干分量取绝对值，不能把它说成已经掌握任意方向的完全signed修正。

CAL的FHN ε=.5三个模型仍不通过相关矩链；共享的物理CAL实例本身目标margin为−.0209892692，不能凭表示误差上界救回不成立的目标排序前提。此为有限CAL支持诊断，未更改部署时TRAIN冻结G0，也未改原12/12联合成功记录。[旧诊断解释](../round3/CHAINED_SCORE_AUDIT.md)

## 4. 新旧范数门的失败不能混算

Round3的Cauchy链是已知affine core→实际Fine评分余项；Round5则从normalized单物理轴误差反推共同/逐轴范数门。两者不是同一个充分条件。

只读复核 [Round5数值oracle摘要](../round5/ORACLE_GATE_FLOOR_AUDIT.json) 保留：共同η门数值否决6/12，分轴同界仅否决FHN ε=.5的3/12。该诊断已扣数值OLS最优性修正与显式敏感性余量，但不是interval形式证明；只否决同一有限TRAIN支持上的当前充分范数门。其余9父未被分轴下界否决不等于获得保证。

因此，旧相关矩链的TRAIN12/12与Round5分轴范数门的3/12否决并不冲突。它们共同指向应保留目标相关结构，而不能证明无标签的相关结构上界已经存在。

## 5. 本轮实际可实例化性结论

| 接口 | 现有记录足够实例化？ | 缺少的具体输入 |
|---|---|---|
| 观测posterior→小A | 否 | 完整posterior记录；合法Bayes/excess风险上界或覆盖与正则性条件 |
| 固定context的Round5 tail/closure | 否 | 合法固定context条件估计；head contrast、anchor总体矩/尾部/协方差置信预算 |
| anchor→G0参考分布 | 否 | 密度比与尾矩、非退化尺度及总体/经验桥 |
| signed相关结构→实际选集 | oracle可诊断，无标签接受式证书尚不能 | 真包含的候选解释集合及全类signed pair correction上界；统计覆盖 |
| 当前有限TRAIN G0排序复现 | 是，metadata层面 | 这不是物理语义或总体可靠性保证 |

本分支未给任何实际父新增无真值证书；这是当前输入缺口，而非对未来任何证法的不可能性断言。范围内没有发现能直接把旧oracle相关预算替换成可观测上界的已存记录。若保留目前证据，应定位为**条件选择理论＋固定受控实例上的相关结构机制诊断**，不宣称部署时可靠性认证、Fine必要性或同容量方法优势。
