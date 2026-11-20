# Round 4：可观测量与现有记录ledger

2026-11-04。遵守 [scope](OBSERVABLE_CERTIFICATE_SCOPE.md)：只读Python源码、JSON metadata和既有Markdown；无数组成员、checkpoint权重、NN前向或新科学数值。本ledger区分“函数可定义”“记录存在”“总体条件已证明”三件事。构造性接口与完整证明见 [OPTIONS](OBSERVABLE_CERTIFICATE_OPTIONS.md)。

## 1. 拥有源码与当前接线

当前来源为Fine＋固定TRAIN G0、fullFine C1L预测。G0的两轴输出是并行慢轴语义输出，不是fullFine forecast的瓶颈。以下定位来自源码静态读取，没有调用相应函数。

| 来源 | 拥有内容/实际字段 | 本轮复核边界 |
|---|---|---|
| `work/hcf_method_freeze_20261025/fine_development.py::read_train_cal` | learner仅x/actions/regimes，u∈0…8；读TRAIN/CAL | 源码字段契约；本轮未调用loader |
| 同文件 `diagnostics` | 用实际u算learned标准化residual；完整观测NLL减future logdet；逐u二阶摘要和Pearson | 静态核schema与已有JSON；无新residual计算 |
| 同文件 `fit_one` | 七固定snapshot；CAL NLL选best；TRAIN gradients；详细diagnostics独立写JSON | 不能把CAL说成独立确认集 |
| 历史拥有文件 `hcf_8h_extension_20261025/fhn/core.py::FineDensity` | E可逆；mean=linear＋MLP(currentFine,a)＋regime_mean；std仅regime、logstd clamp[-6,2] | 代码支持定义O1；未加载actual embedding表 |
| `work/hcf_method_finalize_20261027/v1/explicit_selector.py` | TRAIN FP64 mean/std与speed，FP32 speed排序；h=(4,16,32)，取两轴，index破tie | 存储规则/有限TRAIN排序；无population interval |
| `work/hcf_stable_id_20261028/confirmation_v2/ours_bridge.py` | 固定命名TRAIN/CAL诊断对；FEATURES仅raw Fine train/cal | 不包含counterfactual-u posterior或逐样本residual |
| 同目录 `frozen_prediction_v2.py` | 已冻结预测/编码接口 | 若调用会读状态/执行NN，本轮未调用 |
| 同目录 `data_owner.py` | learner边界为x/actions/regimes；拥有真状态另隔离；AR mode=rich；原数学、V2 RNG接线 | 只核接口与源码，不读physical或FINAL |

历史拥有文件根为 `/Users/yuzihang/Documents/Codex/2026-10-23/referenced-chatgpt-conversation-this-is-an-3/work/`。Fine source中有encode/decode和change-of-variables，不等于从metadata恢复了真实inverse或其导数。

## 2. Mean＋variance contrasts：不能漏掉mean组

拥有源码 `hcf_identified_g_20261024/fine_stage.py::generate_case`（约第74–80行）明确：AR rich u1…4逐轴mean shift .6，u5…8逐轴std×3，base std=.12。`hcf_8h_extension_20261025/fhn/env_fhn.py::regimes`（约第36–40行）明确：FHN u1…4逐轴mean_contrast，u5…8逐轴std乘 `1+(variance_multiplier−1)*contrast_strength`；当前配置mean_contrast=.035、kick_std=.025、variance_multiplier=3、contrast_strength=1、rotational=false。`contrast_rank`（约第42–48行）使用同一past/action下完整Gaussian自然参数contrast。`generate`的更新为predicted＋means[u]＋stds[u]×noise。

因此当前rich法则不是variance-only：同一context上的rank8接口在结构上兼容，见OPTIONS O6的解析分块证明。若删除mean contrasts，固定context的variance-only自然参数rank至多d；那是另一法则，不能套到当前四槽。这里使用拥有生成源码仅核理论前提兼容性，**没有用真实D/means做无标签actual证书，也没有重新计算原有contrast_rank数值**。

当前learned head的两张regime embedding表原则上足以计算其自然参数rank，且该特定head的rank与context无关；但JSON只记表的hash/范数与logstd范围，没有数值表或奇异值。本轮不读checkpoint，故actual learned rank标未核，不能用source rank代替。

## 3. 12父、84 snapshot：实际保存了什么

共同metadata根为

`outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2/execution/jobs/representation/OursFine4/<slot>/seed<rep>/model/`

四slot为 `ar_rich_fresh1`、`ar_rich_fresh2`、`fhn_fresh_eps008`、`fhn_fresh_eps05`；每slot三representation父，共12父、四物理实例。每父有七份 `diagnostics/stepXXXXXX.json`，合计84份，不是84个独立环境或数据集。七step为1500、3000、4500、6000、12000、18000、24000。同一父用固定sampled pair indices重复评估；每snapshot TRAIN/CAL各2048对。

| slot | rep0选step | rep1选step | rep2选step | CAL来源轨迹数/父 |
|---|---:|---:|---:|---:|
| ar_rich_fresh1 | 24000 | 24000 | 24000 | 32 |
| ar_rich_fresh2 | 24000 | 18000 | 18000 | 32 |
| fhn_fresh_eps008 | 18000 | 24000 | 24000 | 24 |
| fhn_fresh_eps05 | 12000 | 12000 | 12000 | 24 |

这些轨迹是抽样对的来源，表格不声称每条都产生相同数量的有效对，也不把2048当独立n。metadata的per_trajectory给每条trajectory的pairs与mean_NLL；不是逐transition日志。32/24也不是本轮计算过集中界的样本量。

`Fine.json`保存checkpoint/来源/状态hash、best step/CAL_NLL、scalar history和成本；history没有完整残差诊断。详细记录在独立diagnostics目录，schema为：

- 顶层：step、TRAIN、CAL、bounds、projection_activity_cumulative、factor_labels_used、TEST_arrays_materialized。
- TRAIN/CAL：observation_NLL、pairwise_mean_observation_NLL、pairs、pair_indices_sha256、standardized_innovation_by_regime、residual_vs_currentFine_action_correlation、context_columns、per_trajectory、interpretation。
- 每u：pairs、mean、variance_ddof0、coordinate_correlation，全部九regime均保留。
- residual与context的Pearson是4×6，context列为Fine0…3、action0…1；它不是对完整conditional law的检验。
- bounds：parameter_norms、logstd_min/max、mixing_generator_norms。没有功能性全支持Jacobian界、观测tail bound或conditional data score。

已保存的schema还明确写：`Finite sampled diagnostics; not an axis-identification or uniform calibration certificate`。本轮只验证记录类型和数量，不聚合其科学goodness、不新算残差矩。

## 4. Observability ledger

“可计算”表示在明确范围的只读推理/统计流程中有函数定义；不表示本轮执行过，亦不表示现有metadata足够重建。

| 量/前提 | 只需观测与learned对象？ | 现有记录 | 可合法支持的结论 / 缺口 |
|---|---|---|---|
| x,a,u与驱动next的u索引 | 是 | loader/owner源码契约、原array身份metadata | 可定义unlabeled estimand；本轮未读arrays |
| 完整观测条件NLL | 是，须包含future logdet | sampled scalar NLL与per-trajectory means | 选点/有限拟合诊断；真实熵未知，非excess KL |
| Fine/head参数 | 是，learned对象 | checkpoint路径和hash；权重未读 | 功能存在；不能从hash得posterior/rank |
| O1完整counterfactual-u posterior | 是，正确positive prior | 未保存9类逐对posterior/odds/log-loss | 后续可定义；当前不能检验posterior a.e正确 |
| learned自然参数rank/最小奇异值 | 是，当前head可由两张regime表求 | 无数值表/contrast矩阵/singular values | source rank不等于learned rank；当前未核 |
| 真实conditional posterior a.e正确 | 一个总体性质，可用observed U拒绝有限投影 | 无完整posterior/witness日志或统一界 | ECE、accuracy、类别比例不够；exact前提未获证 |
| learned标准化residual二阶矩 | 是 | 84 snapshot、九u的均值/variance/Pearson | 有限必要检查；非joint Gaussian/conditional independence |
| residual逐样本向量与原context | 是 | diagnostics未保存；FEATURES不是残差日志 | 不能仅凭聚合矩恢复非线性witness或条件密度 |
| residual与context的完整独立性 | 是总体性质 | 仅current Fine/action Pearson | zeroCorr不推出独立；有限线性instrument覆盖不足 |
| 条件Gaussianity/条件covariance函数 | 是可观测条件律 | per-u聚合矩，非给定完整C的conditional函数 | 非条件均值估计；非实测conditional innovation residual |
| 真实数据conditional score | 原理上是观测分布对象 | 无估计器、逐样本score或估计误差界 | O9不能用head score代替 |
| E在raw坐标的Jacobian/logdet | learned函数可定义 | 源码支持；NLL使用logdet；无逐点矩阵或统一导数界 | 不等于相对physical inverse的Dr |
| 参数范数/logstd clamp | 是 | bounds与projection_activity | 架构限制；不自动给样本tail/局部余项/全局certificate |
| 正确regime prior/propensity | 当前设计公开 | owner/source为外生iid uniform九类 | 当前可取1/9；不可泛化到其他assignments |
| 真实Gaussian kernel、可逆观测、mean/variance variation | 结构假设，不由观测摘要完全证实 | 合成拥有源码/配置支持；没有actual head posterior证据 | 可做机制兼容性声明；部署需明确truth containment |
| 独立audit/cross-fit覆盖 | 统计设计条件 | CAL已用于七snapshot选点和历史开发 | 不可当独立确认；固定候选的uniform accounting须另证明 |
| 有效有界loss/tail/复杂度界 | 需额外数学条件 | 无针对Gaussian NLL或G0增量的统一界 | logstd clamp不使NLL有界；transition数不等独立n |
| G0 TRAIN mean/std/speed/gap | 是 | 12个G0.json有FP64/FP32 stats、selected_pair、tie、hash | 当前有限stored-score选择可追溯；非population gap证书 |
| 完整相容类、表示直径/score共识 | 可在观测空间定义 | 三seed是有限point candidates，无完整集合上界 | 不能把三seed一致当coverage；本轮另给条件接口 |
| 物理轴名称、外部慢目标、target=唯一最慢集合 | 不单由无标签likelihood确定 | 既有POSTLOCK truth audit；当前unlabeled metadata无anchor | 轴等价类和统计慢序不命名语义；旧oracle反例保留 |
| Round2/3 true A、D、Σ/M、r及pair余项 | 否，既有truth-assisted对象 | 已有JSON/报告存在，本轮未重算 | 仅真值有限数组诊断，不能填本表unlabeled缺口 |

均值0、方差1、逐轴Gaussian边缘和zeroCorr仍可能联合依赖：OPTIONS给了\(\varphi(y_1)\varphi(y_2)[1+\epsilon h(y_1)h(y_2)]\)的解析反例。现有84份摘要即使在这些量上理想，也只覆盖必要检查。

## 5. CAL与G0支持口径

Fine以TRAIN优化，用固定CAL对上的observation NLL选择七个snapshot中的best。现有详细诊断与选点共享CAL支持，不能重新包装为选择后独立确认。若未来只检验完全事先冻结的七候选，可采用同时覆盖，但候选族、witness族、整个历史开发的适应性和独立episode计数都要显式处理。

G0的实际选点是TRAIN归一化后的h=(4,16,32)平均速度，经FP32存储后按speed/index选择。CAL若用于评估该分数，仍应沿用固定TRAIN尺度/variance、只换合法增量支持；CAL自归一化是另一个estimand。当前无标签接口没有重算任何CAL分数，也没有取消前轮负margin或旧oracle反例。

既有physical轴匹配使用了POSTLOCK truth，它不是本轮允许的unlabeled匹配输入。version-space或exact posterior定理中的置换应作为理论等价类或候选之间可观测匹配处理，不能偷用旧真值mapping命名物理轴。

## 6. 复核身份与可追溯hash

以下为被选snapshot的diagnostics JSON字节SHA256，不是checkpoint内容验证或metadata数值真实性的独立复现。共同根见第3节。

| slot/rep | step | diagnostics SHA256 |
|---|---:|---|
| ar_rich_fresh1/0 | 24000 | `e7a22be114c0b7392697c9ae409c2cae65bd4711a66e4ad79b2c8886e89ef78b` |
| ar_rich_fresh1/1 | 24000 | `4e2c1f402da61d5c9ebc3ac60a3fbdafa10ccb7de876b044f5028aa595a65d02` |
| ar_rich_fresh1/2 | 24000 | `16b1aac3891c699151e4d38b55f782c8cc79de10a04089d51d1f04c1d2ebb0ca` |
| ar_rich_fresh2/0 | 24000 | `72a99084e708cbb3bb5d7538581ed23e5b19c0739187419395f28f6fa87dcbd9` |
| ar_rich_fresh2/1 | 18000 | `02ecdcfa181b0c436894ed89dbf46837ed50eb2bc55b2d7a84ee3810591f4753` |
| ar_rich_fresh2/2 | 18000 | `5cbccb4b762042ffcf70423212bbeb53b7a87268ffdca7b1f01080d171587e5f` |
| fhn_fresh_eps008/0 | 18000 | `c2af8147be2bebca083c80427f642e8b1bc96e1eec3fc322a7c357650578d20e` |
| fhn_fresh_eps008/1 | 24000 | `1fbef2185cd86adc3acea90482cdb96d0b52cdaac7ea1931631c3f4b0ae1fe51` |
| fhn_fresh_eps008/2 | 24000 | `1cec8a307f17139ef9fc10fef76a48c6801978abb3518b133a0304f4afd11fab` |
| fhn_fresh_eps05/0 | 12000 | `ff2370f878b34dce69b862324142bd3d89a522430aa7097bcc7e56053660b351` |
| fhn_fresh_eps05/1 | 12000 | `a75c7ba029f3f10af5faee04885ae90a85df5fcaa2c024ae8db1205dde1e9c2e` |
| fhn_fresh_eps05/2 | 12000 | `e110c6ff8f56edb28a5272edb5df75de1b13edc79ed3d999027dc59840716ea2` |

关键当前源码字节SHA256：

- `fine_development.py`：`bdd0b47afa875ae2aca3c697d8a54d022b46343015f34d767973827132686c1a`
- `explicit_selector.py`：`e39e988a6406d2f70b1755bbb6a456dcde9a2ab56f4caf6647d768463becd112`
- `ours_bridge.py`：`f7cd48b18f5e237d71d63c9962c02dc7123a3a62a9843d2d5fbc481129ace04f`

本轮未改任何原科学源码、模型、配置、旧报告或JSON；没有把只读metadata盘点记为新的模型通过/失败实验。可交付的是接口与缺口的精确位置，而不是actual Fine物理认证。
