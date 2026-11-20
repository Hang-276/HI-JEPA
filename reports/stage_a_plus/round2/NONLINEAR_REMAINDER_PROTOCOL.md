# Round 2 非仿射余项审计：固定执行范围

2026-11-04。此文件在任何本轮OLS/polar数值拟合之前写入。

## 已完成的访问检查

已读取既有 `reports/stage_a_plus/paired_score_audit.py` 的支持/标准化/映射实现。已检查NPZ目录：四槽各一个既有 `private/TRAIN_CAL/<slot>.npz` 均含 `train__physical.npy` 和 `cal__physical.npy`；四槽×seed0/1/2的12个既有 `FEATURES.npz` 均含 `train.npy` 与 `cal.npy`；12份原CAL `MAPS.json` 和原 `G0.json` 均可用。没有生成替代数据，也不访问任何FINAL数组。

## 固定对象与数值操作

1. 四槽固定为 AR rich fresh1/fresh2、FHN fresh ε=.08/.5；每槽原seed0/1/2全部保留，不追加或选择父模型。AR目标[0,1]，FHN目标[1,3]，H=(4,16,32)。
2. 使用原CAL一一匹配、slopes和intercept对齐Fine，不重新拟合该映射。物理V与Fine W分别以各自完整TRAIN逐轴均值/标准差标准化，ddof0、FP64；相同TRAIN尺度直接用于CAL，不对CAL重新标准化。
3. 每父仅作一次预定full-TRAIN OLS：列向量约定 `w=A v+b`；NumPy行样本实现 `W=V@A.T+b`。使用全部TRAIN点和截距、`numpy.linalg.lstsq(..., rcond=None)`，不正则化、不选择子空间或超参。若设计秩不足，保留N/A，不补救。
4. 每父对A只作一次polar分解：A=UΣVᵀ，Q=UVᵀ，不强制det(Q)=1。保留general affine core `A v+b`、orthogonal polar core `Q v+b`、nonorth affine差 `(A−Q)v`、非仿射OLS余项 `w−(Av+b)`。Q是A的Frobenius最近正交因子，不另做Procrustes/旋转拟合。
5. 在TRAIN和可用CAL上分别计算物理、actual Fine、affine core和polar core速度。所有分母均为该对象在TRAIN Q0的方差，CAL仅更换Qh的增量支持。每h只用本split合法端点。
6. 固定比较三种分解：affine→actual、polar→affine、polar→actual。每轴报告exact增量交叉项、残差增量能量、TRAIN方差修正及其和；每父报告全部4个target/non-target pair、exact差和norm/Cauchy安全界，允许界失败。CAL的state余项均值/方差只作外推诊断，不替换TRAIN分母修正。
7. 原G0仍按已存TRAIN FP32分数排序。CAL上“分数选集”只是假想支持敏感性诊断，不是改变部署G0。CAL已用于checkpoint选择和轴匹配，不能称独立确认。

## 恒等式与边界

固定Q0=TRAIN，对core Z和余项R定义 V=Var_Q0(Z)、c0=Cov_Q0(Z,R)、vr=Var_Q0(R)，W=Z+R，Vw=V+2c0+vr。对Qh=TRAIN或CAL合法端点，C=E_Qh[ΔZ ΔR]、Rinc=E_Qh[(ΔR)²]、S_core=E_Qh[(ΔZ)²]/V，则

`S_actual−S_core=[2C+Rinc−S_core(2c0+vr)]/Vw`。

OLS只保证TRAIN静态cross为零，不保证TRAIN/CAL的滞后或增量cross为零。该拟合使用物理真值，是事后诊断；不修改Fine/G0，不以κ推实际NLL risk，不宣称无标签证书。

## 产物与成本上限

只写本目录的新 `NONLINEAR_REMAINDER_AUDIT.md/json`、独立复算脚本及本protocol；读取文件/member与SHA256、脚本及protocol SHA全部记录。上限12次full-TRAIN多输出OLS、12次4×4 polar-SVD；CAL不拟合。零神经训练、零forward/checkpoint反序列化、零PRNG、新物理数据、FINAL数组、GPU、模型/门槛修改。所有失败/N/A/负margin完整报告。
