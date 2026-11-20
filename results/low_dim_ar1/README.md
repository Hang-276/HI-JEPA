# 低维封存实验

本目录按原分支名提供链接。阅读报告时优先使用下面的实目录入口，保持旧报告内部相对链接的含义。

| 分支 | 原报告 |
|---|---|
| A1自然修正 | [结果](../../reports/lejepa_natural_direction_repair_20261113/RESULTS.md) |
| A1参考与二次机制 | [局部状态](../../reports/lejepa_a1_joint_20261115/local/RESULTS.md)、[参考](../../reports/lejepa_a1_joint_20261115/reference/RESULTS.md) |
| A4新模型 | [结果](../../reports/lejepa_a4_20261115/RESULTS.md) |
| A3固定分差 | [结果](../../reports/lejepa_a3_gap_20261115/RESULTS.md) |
| A2拟合预算 | [结果](../../reports/lejepa_a2_readout_20261115/RESULTS.md) |
| B1/B3动力学 | [结果](../../reports/lejepa_b_transfer_20261115/RESULTS.md) |
| B2解混间隔 | [结果](../../reports/lejepa_b2_lag_20261115/RESULTS.md) |
| 11月16日本地收口 | [结果](../../reports/lejepa_local_closeout_20261116/RESULTS.md) |

文件与回执原样保留；所有链接解析回 `reports/`，避免改变原封存来源键。可重演输入准备见[低维复现说明](../../docs/LOW_DIM_REPRODUCTION.md)，不可在原结果目录直接重跑。

## 后续理论与合成机制分析

以下是清理后新增的分析成果，保存在独立目录；不混入上表的封存神经网络结果。

| 成果 | 报告 | 配套结果 |
|---|---|---|
| 数学形式化与严格条件 | [完整证明](../../docs/THEORY_FORMALIZATION_20261116.md) | [确定性解析核对](../theory_checks_20261116/RESULTS.md) |
| 谐波、慢子空间与有限窗口机制 | [探索性 NumPy 沙盒](../../docs/EXPLORATORY_MECHANISM_PROBES_20261116.md) | [探针一](../exploratory_20261116/probe1_summary.json)、[探针二](../exploratory_20261116/probe2_summary.json)、[封存锚点投射](../exploratory_20261116/ANCHOR_COMPARISON.md) |

脚本、图形与安全复现说明见[统一入口](../../docs/README.md)。两报告坚持同窗集合错误与总体排序倒序分开；沙盒没有建立单一 SNR 风险定律，也没有将 `.85` 准入门证明为理论必要条件。
