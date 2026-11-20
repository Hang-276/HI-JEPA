# Round 3：非正交仿射辨识几何的固定量级诊断

2026-11-04。本文件在本轮任何Hessian/梯度/核心风险数值计算之前写入。前轮报告、数据、模型与门全部保留。

## 对象与最少读取

只复用Round2 `NONLINEAR_REMAINDER_AUDIT.json`中固定12父的原CAL对齐A/Q/b、物理TRAIN均值/std，及 `INNOVATION_BUDGET_AUDIT.json`中已核噪声law/state gauge的九个D_u。已检查四槽物理NPZ均含train__physical/cal__physical，可最少只读这8个既有成员来恢复完整TRAIN Σ与TRAIN/CAL增量二阶矩M。没有Fine新前向、拟合或FINAL数组。

Q0始终TRAIN，Σ=Cov_TRAIN(v)供TRAIN与CAL两套分数共同使用；CAL仅改变各h合法增量支持，不自行重标准化。H=(4,16,32)，四槽原target不变，全部六i<j创新H块与全部四target/non-target比较保留。

## 固定数值定义

每槽、六pair取真实regime law uniform1/9：

`H_ij=[[E d_j/d_i,1],[1,E d_i/d_j]]`，坐标顺序 `(B_ij,B_ji)`；列出旋转角度曲率、unit-Fro旋转曲率、两个单位shear方向曲率及最小特征值。稳定计算λmax=(a+c+hypot(a−c,2))/2，λmin=(ac−1)/λmax；若非正则记N/A，不修regularizer。它只属于固定尺度下的局部仿射Gaussian profile，不是actual global Flow κ。

每父仅作行尺度规范化 `Abar_ij=A_ij/A_ii`，B=Abar−I。数值适用性门预定为任意|A_ii|≤1e−8则该父N/A，不修改阈值救援；这是分母保护，不是科学性能门。检查F(Abar)=F(A)，比较exact F(Abar)与0.5 vec(B)^T H vec(B)，但不把其接近当作局部理论认证；不重算polar、不挑core。

每split用固定TRAIN Σ与该split M，对每target a/non-target b在identity处计算off-diagonal梯度：row b为 `2(M_bj−S_bΣ_bj)/Σ_bb`，row a为其负对应项，其余为0。Q0 unit variance下分母为1。计算全12维block-diagonal H的平方对偶范数 `ell^T H^-1 ell`，与 `||ell||²/min_pair λmin(H_pair)` 比较。固定H与offdiag坐标欧氏/Fro范数约定，不改变坐标以优化结果。

本轮先报告这些有限oracle/局部代数量。等待theory代理提供有限邻域常数后再评估非渐近保证，不自行选择邻域或调阈值。CAL负margin完整保留，不重选G0。

## 有限邻域补充（首次计算前追加）

theory代理已完成 `AFFINE_LOCAL_THEORY.md` 的独立常数核验。对每父固定r=||B||F，不搜索半径：R=max_u,i,j d_j/d_i，c(r)=r/[3(1−r)]+R²r²/4；r<1且c≤κ_aff/4才标记uniform radius-r球的充分条件成立。另固定报告pointwise预算 `0.5 E_u sum_i[q_iu−log1p(q_iu)] + ||B||F² ||B||op/[3(1−||B||op)]`（op<1），其中q_iu=sum_j≠i Bij² d_j/d_i。

pointwise条件预定为该预算/||B||F²≤κ_aff/4；它只验证当前已知B的truth-assisted域条件，不能认证整个球。也另列预算≤0.25 bᵀHb这一直接pointwise二次风险保证，分名显示，不据通过与否选择有利标准。B=0时按精确identity边界处理。上述域/预算失败完整保留，不调阈值或拟合。

若上述任一合法coercivity条件成立，且score分母邻域条件成立，可按正式附录同r计算各轴C_j(r)、有限数组仿射pair下界 `gamma−2 sqrt(Psi F(A))−4 max(C_a,C_b)F(A)/kappa`；不代入actualFine/CAL NLL，不加入未估的统计概率或非线性认证。记录这是uniform球条件还是仅当前B的pointwise条件，所有负下界保留。

## 输出与上限

只新写本scope、`AFFINE_GEOMETRY_AUDIT.md/json`及独立脚本。记录旧A/Q/噪声审计、物理成员、scope和脚本SHA。仅24个2×2 H块、12父的固定矩阵与每split4pair算术；零新OLS/polar、NN、PRNG、新数据、conditional均值拟合、FINAL、GPU或实际风险认证。
