# H-JEPA slow selection

这是固定模型与本次数据分布上的独立测试结果；PASS 不表示一般性定理。

| 模型/层 | 阈值来源 | 最低 R² | 接受比例 | 选错窗口/接受窗口 | 错误率上界 | 状态 |
|---|---|---:|---:|---:|---:|---|
| seed42_level1 | own_cal | 0.563 | 51.0% | 30/1045 | 4.52% | RECOVERY_NOT_VERIFIED |
| seed42_level1 | level1_cal | 0.563 | 51.0% | 30/1045 | 4.52% | RECOVERY_NOT_VERIFIED |
| seed42_level2 | own_cal | 0.251 | 48.9% | 70/1002 | 9.38% | RECOVERY_NOT_VERIFIED |
| seed42_level2 | level1_cal | 0.251 | 52.9% | 77/1084 | 9.41% | RECOVERY_NOT_VERIFIED |
| seed43_level1 | own_cal | -0.195 | 48.9% | 73/1002 | 9.72% | RECOVERY_NOT_VERIFIED |
| seed43_level1 | level1_cal | -0.195 | 48.9% | 73/1002 | 9.72% | RECOVERY_NOT_VERIFIED |
| seed43_level2 | own_cal | -0.051 | 49.6% | 105/1015 | 13.11% | RECOVERY_NOT_VERIFIED |
| seed43_level2 | level1_cal | -0.051 | 12.2% | 29/249 | 17.99% | RECOVERY_NOT_VERIFIED |
| seed44_level1 | own_cal | -0.053 | 45.4% | 96/929 | 13.24% | RECOVERY_NOT_VERIFIED |
| seed44_level1 | level1_cal | -0.053 | 45.4% | 96/929 | 13.24% | RECOVERY_NOT_VERIFIED |
| seed44_level2 | own_cal | -0.098 | 50.7% | 109/1039 | 13.24% | RECOVERY_NOT_VERIFIED |
| seed44_level2 | level1_cal | -0.098 | 40.7% | 82/834 | 12.86% | RECOVERY_NOT_VERIFIED |

RECOVERY_NOT_VERIFIED：本读出未通过四个因素的恢复检查，不能据此认定表征丢失了信息。
CRITERIA_NOT_MET：恢复检查通过，但本次阈值未同时达到覆盖率与错误率要求。
PASS：恢复检查通过，覆盖率至少 35%，且 Bonferroni 校正后的单侧错误率上界不超过 1%。
覆盖率是测试集观测比例，不是总体覆盖率的置信下界。

请将这个运行目录整体压缩发回；分析不需要 assets 中的图像或官方权重。
