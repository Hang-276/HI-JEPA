# Round 6 dynamic-correlation selection scope

2026-11-04。遵守本目录 `ROUND6_SCOPE.md`。本分支仅证明条件性的选集保证：保留 signed increment cross moments、逐轴重新标准化及 target-pair 共享抵消；通过完整候选解释集合或有效联合矩置信域，给物理 pair gap 的 upper/lower bounds。

输入只读：Round2 exact score identity、Round3 affine/评分报告、Round5 transport/selection 及其已有文本/JSON摘要。旧真值摘要仅作 oracle 机制背景，不能作为实际无标签接受证据。

固定检查只用手工矩阵、解析二阶矩和确定性线性代数，不创建物理样本，不用 PRNG，不读取科学数组、模型/checkpoint，不拟合、不前向、不训练，不接服务器/GPU。只写本 `round6/` 新文件。

交付 `CORRELATION_AWARE_SELECTION.md` 与固定公式脚本/JSON。明确完整候选类的真解释覆盖、matching、uniform statistics 和端点支持前提；不把有限 seed、有限摘要或小 L² norm 解释为已掌握真实 signed cross terms。完成后交独立代理只读复核。
