# HCF-JEPA Stage A+ 第二轮交付

2026年11月4日。**先读[主报告与研究决策](ROUND2_REPORT.md)。** 结论为CONDITIONAL GO：保留受控机制研究，尚未建立实际Fine的非空总体可靠性保证，服务器阶段继续后置。

本轮最重要的理论结果是：**正交旋转方向的辨识曲率可以很强，单向shear方向却很弱。** 一个四维、九regime、自然参数满秩的解析构造中，可预测慢因素混入快轴，似然代价及条件创新余项可很小，标准化后的慢集合仍会错选。它明确了κ的方向范围，以及条件创新余项与多时距评分余项不可互相替代；没有声称这是实际已训练Fine的失败机制或文献首次发现。

实际数据诊断保留了三个限制：小静态非仿射余项仍不足以让FHN小间隔实例通过范数界；一个已有FHN CAL物理实例的排序margin为负；Fine相对raw的增量目前限于部分条件下的精度，10个raw同规则cohort全部通过原门。

## 报告与证明

| 文件 | 内容 |
| --- | --- |
| [ROUND2_REPORT.md](ROUND2_REPORT.md) | 本轮发现、证据边界、创新性判断、继续及停止条件 |
| [NONLINEAR_THEORY_REVIEW.md](NONLINEAR_THEORY_REVIEW.md) | 完整归一化score恒等式、最大熵单向总体风险桥、相对创新尺度预算与高频反例 |
| [CORRELATION_RISK_EXTENSION.md](CORRELATION_RISK_EXTENSION.md) | 更紧相关风险桥、实际T所需评分余项、满秩持久因素shear反例、一般仿射Hessian |
| [NONLINEAR_REMAINDER_AUDIT.md](NONLINEAR_REMAINDER_AUDIT.md) | 四槽12父仿射/polar诊断、全部安全界失败、固定TRAIN尺度CAL负margin |
| [INNOVATION_BUDGET_AUDIT.md](INNOVATION_BUDGET_AUDIT.md) | 原噪声law核验、全部regime预算及相关桥经验量级失败 |
| [FINE_VALUE_AUDIT.md](FINE_VALUE_AUDIT.md) | 10个同规则cohort比较、CAL仿射空间上限、strong/weak的观测难度混杂 |

## 数值证据与复算入口

| JSON | 对应脚本 |
| --- | --- |
| [NONLINEAR_FORMULA_CHECKS.json](NONLINEAR_FORMULA_CHECKS.json) | [nonlinear_formula_checks.py](nonlinear_formula_checks.py) |
| [CORRELATION_RISK_CHECKS.json](CORRELATION_RISK_CHECKS.json) | [correlation_risk_checks.py](correlation_risk_checks.py) |
| [NONLINEAR_REMAINDER_AUDIT.json](NONLINEAR_REMAINDER_AUDIT.json) | [nonlinear_remainder_audit.py](nonlinear_remainder_audit.py) |
| [INNOVATION_BUDGET_AUDIT.json](INNOVATION_BUDGET_AUDIT.json) | [innovation_budget_audit.py](innovation_budget_audit.py) |
| [INNOVATION_CORRELATION_BUDGET.json](INNOVATION_CORRELATION_BUDGET.json) | [innovation_correlation_budget.py](innovation_correlation_budget.py) |
| [FINE_VALUE_AUDIT.json](FINE_VALUE_AUDIT.json) | [fine_value_audit.py](fine_value_audit.py) |

拟合前固定的[非线性余项协议](NONLINEAR_REMAINDER_PROTOCOL.md)、[创新预算scope](INNOVATION_BUDGET_SCOPE.md)与[相关预算追加scope](INNOVATION_CORRELATION_SCOPE.md)均单独保留。增量分析按已有结果适应性推进，均是开发/事后分析，不称独立预注册确认。

公式脚本只做确定性解析/矩阵检查。数据脚本读取既有TRAIN/CAL缓存、原映射、配置和已暴露结果标量；读取成员与输入哈希随JSON保存。50个CAL成员用于Fine价值诊断，12次TRAIN OLS与12次polar用于余项分解，两类操作不是神经训练。后续预算复用A/Q，不重新拟合。

本轮零神经训练、模型前向、checkpoint反序列化、PRNG、新物理轨迹、TEST/FINAL数组读取、GPU或服务器操作。历史源码、数据、模型、阈值及上一轮交付均不覆盖。有限CAL上限、余项plugin与真值充分界不能称总体或无标签证书。

[DELIVERY_CHECKS.json](DELIVERY_CHECKS.json)记录交付核验；[SHA256SUMS](SHA256SUMS)覆盖本目录文件（自身除外）。上一轮入口为[Stage A+第一轮](../README.md)。
