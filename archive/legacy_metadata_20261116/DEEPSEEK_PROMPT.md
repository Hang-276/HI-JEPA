# 可直接粘贴给DeepSeek的启动提示词

你接手我的本地HCF-JEPA研究。请实际完成工程、运行已批准实验、分析和独立复算，不要只回复计划，不要等我一步步指导。我将睡觉；我回来明确说“可以停止”“我睡醒了”或同义指令时，你应停止新增启动/更新，在安全边界保存现场并汇报。

## 立即阅读与边界

工作目录：`/Users/yuzihang/Documents/ChatGPT/HI-JEPA`。先读根目录`DEEPSEEK_START_HERE.md`列出的5项，再按需读其精确链接；不用遍历巨大outputs或重写已有手册。这里原有HI-JEPA旧谱模态工程，当前接手的是HCF-JEPA FlowDirect分支，不要误跑根README旧formal脚本。

我已批准FlowDirect原有限计划并给11小时总窗口。起点2026-10-25 23:43:48，截止2026-10-26 10:43:48，时区Asia/Shanghai。**换成DeepSeek不重新获得11小时。** 我的停止指令优先；到窗口上限停止新增科学更新、保存和汇报。不要为填满时间增加矩阵，也不要因一个脚本结束就把整个研究任务当成完成。

已批科学上限：阶段0–1先做，最多20神经拟合＋4OLS、288000更新；按原开发门通过后推进原阶段2–4，完整最多74神经＋4OLS＝78新拟合、1032000更新。普通工程修复和忠实受控适配可自主完成；不得悄悄变损失、放松失败门或增加种子/参数网格。RESULT_DEPENDENT_EXTENSION_PLAN中的C26拟合、D52拟合是额外提案，不能据“自主”自动叠加到已批预算。5903与CONTRAST继续停止。

## 研究是什么：必须保住的主线

我们研究的是**稳定、可辨识的动力学因素**。预测检验这些因素的用途，低MSE不替代因素可辨识性。数据是受控四维双FHN配置族，物理状态可理解为(v1,w1,v2,w2)，w是预声明慢因素；经未知平滑可逆混合得到四维观测。caseA/B是同一配置族的两个data/mixing实例，不是两个机制族。ε=.08/.5，两维动作、九种已知regime，三个优化seed。当前预测任务允许完整有序未来动作/regime路径，不能外推到未知未来控制。

Fine E是4块可逆flow，在原条件单步Gaussian NLL＋未来Jacobian下24k训练，CAL无物理标签选点。丰富regime均值/方差对比、正确条件族和精确population匹配下有逐轴仿射歧义的条件论证；有限训练、带测量噪声和一般有限样本稳定性尚无保证。Fine gate为mean/worst轴R²≥.95/.90；G为≥.90/.80、慢集合正确且Fine通过。这些经验门不能反写为理论。

旧V1是G选coarse2，P预测未来coarse、仿射OLS Q补未来其余两维，再用冻结E逆映射。当前coarse2丢掉其余当前信息，完整预测误差高。G learnable/ST相对TRAIN速度top2的独立增量未建立，ND在本面板也没有独立增量。历史memory加入过去信息显著改善受限任务，但没有消除完整当前信息的差距，且4499整体替代门失败；不要恢复那条队列。

新的主候选**FlowDirect=冻结Fine完整RAW4→3个独立64×64 Tanh残差MLP(h4/16/32)→预测RAW Fine4→同一Fine.decode→完整观测**，头参数50828。它不是flow matching，也不是新的可逆转移模型。旧2107清洁split NMSE约.010323/.095378/.297753；旧5103清洁约.010871/.101618/.312772，σ=.1约.021408/.125851/.331160。不同面板/支持/分母不能混表；高分部分可能来自完整当前信息，必须与同信息ObsDirect比较。

coarse先作为固定慢轴视图：原TRAIN标准化、多lag4/16/32速度，固定top2原Fine轴；当前和预测RAW4使用同一pair及G自己的mean/std读出。它不限制主预测输入、不dense重混、不接新H/P/Q。不能把切开四维输出当新算法，也不能因慢轴正确就宣称coarse闭合或提高完整分数。

文献与原文分数已在索引内。重点：iVAE已有flow特例，LEAP提供动态辨识，IDF ICLR2025已比较LEAP＋离线MLP两阶段预测。组合本身不是自动新颖；各论文MCC/MSE/任务与本地FHN不能直接排名。

## 交接时真实状态：从这里开始，不要虚构续跑

截至本交接，**FlowDirect新分支0科学拟合、0科学更新；阶段0旧36快照CAL审计脚本也没有执行，没有新确认数据生成/读取。** 所有代理已收到停止新增科学启动。旧累计297拟合/2388000更新不属于本轮。

已有12组冻结Fine/G与旧Flow，36个3k/6k/12k Flow快照已做过字节SHA核对；旧Flow科学选择仍固定12k，12/12诊断latent CAL最低在3k，但不得事后改挂旧TEST或抹掉12k成本。父/旧输出原地只读，新增源用`work/hcf_method_freeze_20261025/flowdirect_branch_v1/`，新增结果用`outputs/HCF_METHOD_FREEZE_20261025/FLOWDIRECT_BRANCH_V1/`。

当前交接模块：

- `stage0_cal_audit.py`：已静态审查并补历史snapshot SHA绑定、恢复terminal后张量哈希检查；**未运行**。复用旧qe.verify完整88头和TRAIN/CAL检查，再审36快照的161/149 CAL；不读数值TEST/物理真值。第一次运行前检查源码/当前环境、固定身份，不能写成已通过运行。
- `direct_models.py`：工程模块，由交接代理收尾；以实际模块和dummy SELF_CHECK收据为准，**不等于整套科学runner已完成**。需要你核对接口、参数/梯度/恢复边界，并实现父加载、矩阵、CAL家族选择、源锁、科学账本及最终评价桥接。没有未确认的科学CLI命令可直接照抄。
- `leap_adapter.py`若存在也只属源审查/工程；LEAP的CPU、控制条件、label-free选点与完整训练/推断保真门**尚未通过**，不可当作已准备好的baseline。
- `vendor/leap_official/`只读来源：官方alpha commit `216a83d0236777eab3e7a2e7fe8afcab2487b90e`；不是已运行模型。不要运行其shell安装/训练脚本。

交接工程收尾的精确身份（只表示工程证据）：`direct_models.py` SHA256 `e423fff98e1fe2b7051c4fba34b2b89b5edd4d9857f85d0d60de488d60ac6d7e`，`fixtures_dummy_v1/SELF_CHECK.json` SHA256 `811b14220737c05eff2ab4de78f07d1065e68d2f51f3c45c8043687386254a5c`，19/19 dummy检查通过；`leap_adapter.py` SHA256 `40eaeba7ad137d836036436b2c48cae4bce2578be6e22af8212abcfaebc657cd`，其10/10 dummy通过。它们均没有完整科学入口或独立科学终审，不能把这些计入模型拟合/有效性结果。LEAP最新差异和缺口读`work/hcf_method_freeze_20261025/flowdirect_branch_v1/vendor/LEAP_FIDELITY_REVIEW.md`。

Direct模块最小API是`fit_direct(kind,train,cal,out,external_identity,seed,fine=...,resume=False,fixture=None)`；序列keys为`x,z,actions,regimes`，Flow外部身份还需`coarse_pair/coarse_mean/coarse_std`。`load_direct/load_snapshot/verify/native_arrays/evaluate_cal`以实际定义为准；EDMD入口为`fit_edmd({'x','actions','regimes'},out,external_identity)`。CLI只有dummy self-check，不存在真实科学一键启动器。重启故障的snapshot-before-latest、completed checkpoint缺receipt尚未分别注入，接手时不要把这些恢复分支视为已测试。

**测量评价缺口特别注意：** 当前`native_arrays/evaluate_cal`只支持clean TRAIN/CAL；不得直接把noisy整序列塞进去而让未来目标也变成noisy。必须新建显式`noisy_current/clean_future`评价builder并用dummy检查泄漏隔离后才跑科学噪声面板。

环境入口：`/opt/miniconda3/envs/PyTorchProject1/bin/python -B`。单科学进程、Torch/OMP/OPENBLAS/VECLIB均1线程。不要安装或升级包改变现有环境，不连接远程服务器。历史依赖目录`/Users/yuzihang/Documents/Codex/2026-10-23/referenced-chatgpt-conversation-this-is-an-3`只读；FineDensity实现位于其`work/hcf_8h_extension_20261025/fhn/core.py`。完整解码必须是`fine.decode(y)=encoder.inverse(y)*fine.std+fine.mean`，不能漏掉观测反标准化。

## 当前文档未写完整的实现重点

1. **最近邻保真先做。** LEAP原NP prior没有action输入；非平稳版本ct每样本一个、整段选同spline；官方入口用val_mcc物理标签选checkpoint。已批准的是明确标名的“LEAP-NP＋Direct受控适配”，不是逐字复现。允许把观测到的a/regime加入prior条件、逐transition选择噪声spline、用原label-free val_elbo替代val_mcc，但要保留NP inverse dynamics的逐轴Jacobian、条件noise flow、原recon/KL/TC与交替判别器等核心目标，逐项对照来源。删TC/flow、普通VAE冒名、物理标签选点或未来泄漏均不放行。缺lightning/ipdb/yaml/scipy/sklearn/tqdm/disentanglement_lib，先验证；可将数学等价的CPU基础操作用现有torch/numpy适配并做dummy对等检查，不擅自安装环境。官方5000步warm-start也须计入24k表征更新上限，不能当免费预训练。若核心无法保真，记录阻塞并停受影响路径，不换弱基线掩盖。
2. **因果上下文和149支持。** LEAP每起点仅12帧[t−11,…,t]，推断/训练一致，双向GRU只能在该前缀内使用；绝不看t之后观测。新Direct全部t12..160，TRAIN96×149、CAL24×149，三个h共用(r,t)。旧Flow训161不能当新149的完全匹配控制。G速度仍用原合法189/177/161人口。LEAP前缀比Fine单帧长，作为完整配方比较，不能只归因E。
3. **只两种Flow目标＋匹配Raw。** Flow-Latent原normalized Fine MSE；Flow-Obs通过冻结fine.decode对obs TRAIN149 future-var风险求梯度，仅head更新，父无梯度且全部张量不变。ObsDirect同50828头、初始化、r/t流、观测损失与CAL选择。Adam .001/b128/clip10，完整12k，3k/6k/12k按三h等权CAL R_x联合选一个点、tie earlier。不要暗中early-stop追减成本。Flow-Latent vs Raw属于不同loss配方；严格Fine增量只由Flow-Obs vs Raw判断。
4. **家族门和开发停止门照原计划。** 四开发单元是caseA/B×ε、Fine seed0。Obs需CAL总体至少1%增益、四组非劣、每h Fine4/selected2误差≤Latent1.05才选，否则Latent。进入剩余八单元前，主家族/Raw每h四单元均值风险比≤1.05且各单元三h均值比≤1.10；不通过就封存，不加网格/seed救分。具体归一化/聚合以批准计划为准，写机器协议消除歧义。
5. **基线身份。** DMDc已有4个OLS，只复用不伪造三seed；它使用全部192一步转移，不是149严格控制。EDMD同一步支持，固定64确定中心RBF、median正距离宽度、FP64 rcond1e-10、无ridge/裁谱；规则见计划。递推不用未来teacher force。最近邻24k预算不等于论文原算力；不把未收敛结果冒充忠实性能复现。
6. **稳定性要真分开。** 原12个Fine/head配对seed不隔离head随机性；新增8头在固定Fine下补两个head seed，另6个已有replace10 Fine/G父需重训匹配完整Flow头，不能跨gauge直接复制旧头。按同输入的观测输出计算D_x，不用NN权重距离或TEST Procrustes。
7. **封锁与确认。** 工程fixture只用dummy种子/数据。科学协议、源码哈希、全部选择/模型锁好后才生成全新确认家族（4数据集各96轨迹、σ0/.01/.05/.1）。5101工程污染、5903另阶段均不用作本轮全新确认；不要调用旧5103 owner复用其固定科学seed来冒充新数据。新tag与RNG在封锁前注册，生成只在全锁后开放。未来目标始终clean，输入扰动共享Z；不能先看TEST再选家族/λ/模型。所有方法在同一个新家族重评，不能拼旧baseline分数。

## 自主执行与汇报方式

先快速盘点文件/进程和当前账本，防重复训练。完成工程、dummy验证、来源门、源锁，按已批阶段推进；无需为每个普通修复问我。长期任务用可恢复checkpoint、保存真实进程/日志；每30–60分钟或阶段变更写进度文件，不高频轮询耗token。不要把工程耗时叫训练耗时，不把hash/smoke当科学有效性。

原资格门不放松：Fine/G轴集合、预测主张、固定Fine重训（风险CV≤10%、最大配对D_x/配对平均R_x≤.25）、TRAIN替换18格（风险比≤1.10且D_x/base R_x≤.25）分别判。失败保留，未知/N/A保留。严格预测增量需相对匹配Raw总体风险比≤.95、每h正增益且≥9/12胜、四组非劣；保分不能写成预测优势。

每次新增实验前先写一句：它解决哪个稳定辨识/机制缺口，以及什么结果会让我们放弃该解释。不要堆更多baseline、反复换学习率或只追一个最好的数。主线完成且确认/审计/报告齐全后可待命；若门失败，完成原因定位与其它独立已授权工作，不强行继续科学路径。新增分支先按现有提案做0拟合诊断；涉及扩大科学预算须重新获得我的明确授权。

结束/我叫停时交付一个Markdown报告：实际完成/未完成清单、精确fit/update/墙钟与CPU成本、同口径完整预测/因素恢复/风险＋函数稳定表、coarse作用、方法Go/No-Go、最近邻适配差异、失败与仍未知项、可恢复位置。保存源锁、原数组、选择规则和失败记录，不能只给结论。停止时不杀其他项目进程，不删除旧结果，不新增下一轮。

现在开始接手：先读索引、确认当前状态，完成尚缺工程，再运行已批准的有限程序；不要复述整份计划后等待我回复。
