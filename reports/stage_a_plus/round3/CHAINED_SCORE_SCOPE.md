# 已有JSON闭链诊断范围

2026-11-04。根代理追加授权；本文件先于本闭链计算写入。已有Round2/Round3文件均保留。

仅join `AFFINE_GEOMETRY_AUDIT.json` 的逐pair affine下界与Round2 `NONLINEAR_REMAINDER_AUDIT.json` 中 `affine_to_actual` 的同pair score预算。不重新读取数组、拟合或评价模型。固定原12父×TRAIN/CAL共24个parent/split情境，每情境全部4个target/non-target pair，共96行；不将24情境当24独立物理实例。

兼容性需逐项实核：相同slot/rep/target/轴匹配，Q0=同TRAIN尺度、Qh=同split/H合法端点；Abar是原A非零逐行缩放，所以自有TRAIN方差归一化score不变。对比两个JSON保存的affine pair gaps与actual score，若不吻合则标N/A停止该行，不重拟合补救。

固定两条主要预算：Round2 `CS_bound_measured_Q0_correction` 与 `truth_assisted_pair_correlation_triangle_bound`。对每pair分别计算 `lower_actual_CS=lower_affine−B_CS`、`lower_actual_corr=lower_affine−B_corr`，保留全部正/负下界及现成exact实际gap用于核查，不选择有利预算掩盖另一条失败。全部四pair下界>0才计一个情境通过。

这是已知仿射core、真实物理矩、真实余项的有限数组数学闭链，不是从J_actual/CAL NLL到实际Fine的总体或高概率认证。CAL负margin与原部署G0的TRAIN冻结选集保持原解释。

只新写本scope、`CHAINED_SCORE_AUDIT.md/json/py`，保存输入/scope/脚本hash。新增数组访问/OLS/polar/NN/PRNG/physical data/FINAL/GPU均为零。
