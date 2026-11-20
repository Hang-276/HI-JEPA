# 创新预算追加：逐轴相关界的固定量级检查

2026-11-04。由根代理明确追加。原N8预算与所有既有文件保留。

原 `INNOVATION_BUDGET_AUDIT.json` 未保存足以恢复各轴残差二阶矩的完整协方差，因此只最少重读同12父既有TRAIN/CAL Fine、同4槽物理与regime数组；复用已存A/Q/b、TRAIN尺度、各u的C矩阵。不重复OLS、polar、Cholesky、inverse/solve或神经计算。

按驱动next状态的u分组，计算未中心化及仅u样本均值中心化的每轴经验 `tau_j²=mean(r_next,j²)/C_ujj`。对固定四个target/non-target pair均计算 `e_ab,u=2(tau_a+tau_b)`，及可选保守截断 `min(e,2)`，分别按真实regime设计权重1/9和经验频率给L²预算。四pair全部保留，不据结果挑轴或选择core。

可与固定core的 `sqrt(E_u rho(C_u)_ab²)` 比较相关风险桥是否仍给零正风险下界；这只是把总体公式代入经验余项表达式的量级检查，不测实际conditional covariance/innovation residual、不估条件均值、不认证总体NLL或无标签可靠性。actual A/Q是4维core，不宣称它们属于任一purepair二维旋转族。

只新写本scope、`INNOVATION_CORRELATION_BUDGET.json`和独立脚本；在主创新预算报告附录汇总。新增拟合/随机数据/前向/FINAL/GPU全部为零。
