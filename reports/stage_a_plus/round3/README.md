# Stage A+ Round 3

2026-11-04。新增一般非正交仿射的有限邻域理论、指定慢目标敏感性、单向混合精确选集边界，以及现有Fine评分余项的有限数组闭链。只读已有数据和记录，没有训练或服务器执行。

先读 [总报告与去留判断](ROUND3_REPORT.md)。本轮保留受限机制研究；实际Fine总体认证、相对raw独立价值和论文级新颖性仍未建立，服务器/GPU阶段继续暂缓。

| 内容 | 理论或报告 | 全记录与复算入口 |
|---|---|---|
| 一般仿射风险、显式余项与目标gap | [条件定理](AFFINE_LOCAL_THEORY.md) | [公式检查](AFFINE_LOCAL_CHECKS.json)、[脚本](affine_local_checks.py) |
| 单向混合完整选集精确边界 | [理论与近邻](SHEAR_BARRIER_REVIEW.md) | [公式检查](SHEAR_BARRIER_CHECKS.json)、[脚本](shear_barrier_checks.py) |
| 已有12个核心的Hessian/局部量级 | [审计报告](AFFINE_GEOMETRY_AUDIT.md)、[预定范围](AFFINE_GEOMETRY_SCOPE.md) | [JSON](AFFINE_GEOMETRY_AUDIT.json)、[脚本](affine_geometry_audit.py) |
| 核心风险界接实际Fine评分余项 | [闭链报告](CHAINED_SCORE_AUDIT.md)、[范围](CHAINED_SCORE_SCOPE.md) | [JSON](CHAINED_SCORE_AUDIT.json)、[脚本](chained_score_audit.py) |
| 已有4实例全96条单向混合 | [边界审计](SHEAR_BARRIER_AUDIT.md)、[预定范围](SHEAR_AUDIT_SCOPE.md) | [JSON](SHEAR_BARRIER_AUDIT.json)、[脚本](shear_barrier_audit.py) |
| 独立审查 | [局部证明](AFFINE_INDEPENDENT_REVIEW.md)、[单向证明与实现](SHEAR_INDEPENDENT_REVIEW.md)、[数值应用](NUMERICAL_INDEPENDENT_REVIEW.md) | [单向独审检查](SHEAR_INDEPENDENT_CHECKS.json)、[脚本](shear_independent_review.py) |
| 交付与完整性 | [交付检查](DELIVERY_CHECKS.json) | [SHA256SUMS](SHA256SUMS) |

已有Fine/raw比较继续引用 [Round 2完整审计](../round2/FINE_VALUE_AUDIT.md)，本轮不重新读取其TEST/FINAL数组、不调整门或补跑模型。

注意三项边界：逐点条件不是整球条件；真值辅助有限数组闭链不是总体或无标签认证；最小单向失效风险是一般仿射严格排序失效成本的上界，并不是全类安全半径。

公式与审计脚本使用环境已有的NumPy Python。已有结果文件应保留；需要复核时优先检查JSON记录和SHA，若需重算应复制到独立目录避免覆盖本交付。首次仿射审计的序列化失败及同科学逻辑重算已保留在 [执行记录](AFFINE_GEOMETRY_SERIALIZATION_FAILURE.md) 中；单向审计的初次系统Python导入失败发生在任何科学输入读取之前。
