# DeepSeek交接入口：当前只接FlowDirect分支

更新时间：2026-10-26，Asia/Shanghai。**这是当前阅读索引。不要扫描整个outputs、重读所有旧手册或运行根README的旧formal实验。** 本文件只整理入口，科学协议/结果继续引用原件。

## 先按顺序读这5项

| 顺序 | 文档 | 作用与当前身份 |
|---|---|---|
| 1 | [DEEPSEEK_PROMPT.md](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/DEEPSEEK_PROMPT.md) | 用户可直接粘贴的接手指令；补齐最新授权、精确交接状态、实现缺口、停止办法 |
| 2 | [当前执行授权](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/CURRENT_EXECUTION_AUTHORIZATION.md)及[11小时窗口](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/FLOWDIRECT_BRANCH_PLAN_20261025/EXECUTION_WINDOW_11H.md) | 用户已批准FlowDirect原有限程序；换执行者不重置计时，不恢复旧队列 |
| 3 | [已批准的实验计划](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/FLOWDIRECT_BRANCH_PLAN_20261025/PLAN_FOR_APPROVAL.md) | 文件名仍是历史“待审批”，现已批准；其8小时由11小时窗口覆盖；科学预算、开发/确认门不变 |
| 4 | [FlowDirect本地证据](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/FLOWDIRECT_BRANCH_PLAN_20261025/local_evidence.md) | 现有源码、分数、36快照、信息优势、稳定性缺口。它较早建议保161；最终批准计划明确新Direct训149，以最终计划为准 |
| 5 | [既有方法卡](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/CURRENT_METHOD_CARD.md) | 研究问题、Fine/G/P/Q与既有证据；正文是旧V1快照，不是要求新分支继续coarse-only预测 |

先读完以上再实施。不要因目录名有“frozen/latest/current”就跳过时间和范围核对。

## 按任务补读，不要求全部重读

| 任务 | 精确入口 |
|---|---|
| 最近邻和原文分数 | [可辨识动态文献](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/FLOWDIRECT_BRANCH_PLAN_20261025/identified_dynamics_literature.md)、[预测基线审查](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/FLOWDIRECT_BRANCH_PLAN_20261025/forecast_baseline_review.md)；均保留原文链接和可比性边界 |
| 后续结果分支 | [RESULT_DEPENDENT_EXTENSION_PLAN.md](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/FLOWDIRECT_BRANCH_PLAN_20261025/RESULT_DEPENDENT_EXTENSION_PLAN.md)：A/B为诊断，C/D为新增训练提案；不是自动执行队列，不与原78拟合混账 |
| 条件辨识理论 | [THEORY_AFFINE_GAUSSIAN_NOTE.md](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/THEORY_AFFINE_GAUSSIAN_NOTE.md)、[THEORY_BOUNDARY_NOTE.md](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/THEORY_BOUNDARY_NOTE.md)；精确population前提与有限样本稳定性必须分开 |
| 历史为何转向完整状态 | [4303记忆结果](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/COORDINATE_MEMORY_PILOT_V1/RESEARCH_REPORT.md)、[4499移植失败门](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/COORDINATE_STATE_TRANSFER_V1/RESEARCH_REPORT.md)、[5103测量扰动](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261025/MEMORY_OBSERVATION_NOISE_V2/RESEARCH_REPORT.md) |
| 原Flow/G与父加载 | [heads.py](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/work/hcf_method_freeze_20261025/q_output_v1/heads.py)、[parents_and_heldout.py](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/work/hcf_method_freeze_20261025/q_output_v1/parents_and_heldout.py)、[evaluate.py](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/work/hcf_method_freeze_20261025/q_output_v1/evaluate.py)；只调用明确TRAIN/CAL验证函数，不能误调旧TEST评价 |
| 正在交接的工程 | [flowdirect_branch_v1目录](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/work/hcf_method_freeze_20261025/flowdirect_branch_v1)：工程草稿/源适配，**没有已确认完整科学runner**。以实际文件、fixture收据和缺口为准 |
| LEAP最新保真缺口 | [LEAP_FIDELITY_REVIEW.md](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/work/hcf_method_freeze_20261025/flowdirect_branch_v1/vendor/LEAP_FIDELITY_REVIEW.md)：官方来源、CPU工程检查与未实现部分；不是科学放行 |

## 文档冲突与历史材料

优先级：用户最新直接指令 → 当前授权/11小时窗口 → 已批准FlowDirect计划 → 本交接索引/提示词 → 既有结果原件 → 旧方法卡/旧队列/历史提案。发现冲突记录差异，不覆盖历史收据。

- 根README其余正文、docs/V2/V3、scripts/run_formal等属于更早HI-JEPA谱模态工程，**不是本次FlowDirect启动入口**。
- CONTINUATION_AUTHORIZATION_20261025_V2.md及旧CONTINUED_TASK_QUEUE的持续授权已被“收尾”结束；本次只恢复FlowDirect。
- 5903 CURRENT_COORDINATE_LOSS_ROLE_V1、旧CONTRAST、P/Q优化失败候选：继续封存，不恢复。
- 5101 V1存在工程阶段提前接触随机量的偏离，正式评价改用5103 V2；不能重用5101当全新确认。
- 带DRAFT、fixtures、engineering、dummy的文件是工程/草稿证据；缺结果写未运行/N/A，不写0或通过。

仓库既有科学累计为297拟合、2,388,000更新（截至5103报告）；本次FlowDirect新分支交接时正式新增仍为0。接手者另建本分支账本，工程夹具不冒充科学实验。
