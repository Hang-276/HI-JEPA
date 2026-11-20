# Stage A+ Round 4

2026-11-04。本轮只推进无物理真值的理论接口、有限样本限制与Fine独立价值判断，没有读取科学数组或执行模型。

先读 [总报告与路线判断](ROUND4_REPORT.md)。结论：总体可辨识不等于有限可认证；当前实际Fine尚无可用的无真值余项证书。保留条件稳定性问题，Fine价值收缩为部分分布上的受限可读性增量，服务器/GPU阶段继续暂缓。

| 内容 | 主文件 | 核验与来源 |
|---|---|---|
| 无标签可识别对象、满秩近邻的有限检验下界 | [UNLABELED_LIMITS](UNLABELED_LIMITS.md) | [13类解析检查](UNLABELED_LIMITS_CHECKS.json)、[脚本](unlabeled_limits_checks.py)、[范围](UNLABELED_LIMITS_SCOPE.md) |
| 精确regime后验接口及可观测替代 | [OBSERVABLE_CERTIFICATE_OPTIONS](OBSERVABLE_CERTIFICATE_OPTIONS.md) | [现有记录账本](OBSERVABILITY_LEDGER.md)、[输入身份校验](OBSERVABILITY_INPUT_HASHES.json)、[范围](OBSERVABLE_CERTIFICATE_SCOPE.md) |
| 近似后验到充分统计量span的条件界 | [APPROXIMATE_POSTERIOR_BRIDGE](APPROXIMATE_POSTERIOR_BRIDGE.md) | [固定向量核验](APPROXIMATE_POSTERIOR_CHECKS.json)、[脚本](approximate_posterior_checks.py) |
| 完整观测相容集合的表示直径证书 | [VERSION_SPACE_CERTIFICATE](VERSION_SPACE_CERTIFICATE.md) | [固定公式核验](VERSION_SPACE_CHECKS.json)、[脚本](version_space_checks.py)、[范围](VERSION_SPACE_SCOPE.md) |
| Fine与raw的函数类、比较归因和保留主张 | [FINE_VALUE_ROUTE](FINE_VALUE_ROUTE.md) | [范围](FINE_VALUE_SCOPE.md) |
| 直接primary近邻与适用性 | [PRIMARY_NEIGHBORS](PRIMARY_NEIGHBORS.md) | [本地来源和阅读记录](FINE_VALUE_SOURCE_MANIFEST.json) |
| 独立复核 | [后验桥与下界审查](POSTERIOR_AND_LIMITS_REVIEW.md)、[精确后验证明审查](EXACT_POSTERIOR_REVIEW.md) | 相容集合接口的独审在UNLABELED_LIMITS §7 |
| 交付完整性 | [DELIVERY_CHECKS](DELIVERY_CHECKS.json) | [SHA256SUMS](SHA256SUMS) |

三个不能混写的对象：posterior几乎处处正确与普通概率校准；近似充分统计量span与近似逐轴仿射恢复；已搜索seed共识与完整相容集合直径上界。总报告和附录均保留这些边界。

现有比较数字沿用 [Round 2 Fine价值审计](../round2/FINE_VALUE_AUDIT.md)，没有重读TEST/FINAL数组或改评价门。Round 3的真值辅助闭链继续保留，没有被本轮理论入口改称无标签认证。

确定性脚本只枚举固定解析参数/概率向量和小矩阵，不生成科学样本。已有结果应按SHA核验；如需重算，使用隔离输出以保留本交付。
