# Round 6 posterior 分支范围

2026-11-04。先读总 [ROUND6_SCOPE.md](ROUND6_SCOPE.md)、Round5 `TAIL_POSTERIOR_BOUND.md`、Round4 `OBSERVABILITY_LEDGER.md`。本分支独立核查真实 regime posterior KL 的合法可观测上界，不预设当前资料能给肯定结论。

只新增本 `round6/` 中的 `OBSERVABLE_POSTERIOR_BOUND.md`、本范围文件、来源清单，以及有价值时的固定公式检查脚本/JSON。允许纸笔推导、primary 文献定向核查、已有文本/JSON摘要只读；不得修改其他分支作者文件或先前报告。

禁止并不执行：训练、模型/checkpoint加载或前向、科学数据数组读取、新拟合、新随机采样/物理轨迹、服务器/GPU。固定概率向量及解析常数检查不作为训练/确认实验。

重点审查：posterior KL 与 proper risk/Bayes entropy 的方向；参考分类器比较是否只能给下界；finite-bin校准与完整posterior间的剩余信息；有独立样本、覆盖和正则性时可定量实现的上界；平均context风险与固定context的区别；当前已存摘要是否足够填入。未知regularity、coverage、approximation与optimization常数不得用有利值补齐。
