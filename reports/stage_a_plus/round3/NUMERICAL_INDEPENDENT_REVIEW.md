# 仿射几何与评分闭链：独立复核

日期：2026-11-04。复核对象为 `affine_geometry_audit.py`、`AFFINE_GEOMETRY_AUDIT.json/md`，以及新增的 `chained_score_audit.py`、`CHAINED_SCORE_AUDIT.json`。未运行作者审计脚本，未再次读取物理/Fine数组；独立算术复核只使用已写JSON中的小矩阵、预算与标量。

**结论：未发现实质应用错误。** 行尺度、Hessian坐标、pointwise条件、实际Σ对角线与12/12、1/12计数均正确；闭链保持了总体认证与有限真值诊断的区别。

## 1. 行尺度与H坐标

`Abar=A/diag(A)` 是按行除以相应非零对角元。对自由Gaussian profile contrast，任意非零行缩放同时改变每行方差与整体determinant，精确相消；逐轴重新标准化的平方增量score同样不变。代码拒绝|Aii|≤1e−8，不作补救。本次全部对角元非零。

主JSON保存的最大风险行尺度差约2.96×10⁻¹⁵，最大score行尺度差约1.33×10⁻¹⁵。它们处于记录的普通FP64容差内，不称为interval proof或固定clamp actual-head风险的尺度不变性。

H与b、ell都按 `(0,1),(1,0),(0,2),(2,0),…` 排列。每pair块为 `[[E(dj/di),1],[1,E(di/dj)]]`，梯度相应投到同一两个坐标。稳定λmin=det/λmax与全矩阵eigenvalue结果一致；没有遗漏unit-Fro旋转的1/2规范。

## 2. FP64下Σjj略偏离1的处理

保存的TRAIN Σ对角线与1最大差约1.01×10⁻¹⁴。代码没有简单丢掉该偏差，而采用一般baseline方差 `v0=Σjj>0`：

\[
s_j=M_{jj}/v_0,\quad T_j=M-s_j\Sigma,
\quad L_j=2e_j^TT_jb_j/v_0.
\]

由 \(|e_j^T\Sigma b_j|\le\sqrt{v_0K_\Sigma}\|b_j\|\)，分母至少为 \((\sqrt{v_0}-\sqrt{K_\Sigma}r)^2\)。实际代码的score余项常数为

\[
C_j=\frac{\|T_j\|+4\|T_je_j\|\sqrt{K_\Sigma/v_0}
+2\|T_je_j\|K_\Sigma r/v_0}
{(\sqrt{v_0}-\sqrt{K_\Sigma}r)^2},
\]

这是主文单位方差证明的正确一般化，不是额外的CAL重标准化。CAL只换M；Q0分母始终使用同一个TRAIN Σ。

## 3. Pointwise与uniform条件接A6

统一球条件 `c_r≤κ/4` 保证整个相应邻域coercivity，本次只1/12通过。使用给定B的δlog＋δdet则只检查当前矩阵：

- 若该预算≤κ||B||F²/4，则F≥bTHb/4；
- 直接方向条件预算≤bTHb/4也足够得到同一结论。

第二项即使不写成统一球假设，也可在这个固定B上支持 `bTHb≤4F` 与 `||B||F²≤4F/κ`，因此接A6的dual与二阶项是合法的。它不保证任意其他B进入同域，不从低risk推断入域，更不是实际学习器分布上的认证。

独立重计结果：

| 对象 | 通过数/父数 |
|---|---:|
| uniform radius-r ball | 1/12 |
| pointwise min-κ预算 | 12/12 |
| pointwise directional quadratic预算 | 12/12 |
| TRAIN全部4pair仿射下界正 | 12/12 |
| CAL全部4pair仿射下界正 | 9/12 |

CAL失败为FHN ε=.5同一物理CAL下的三个优化父，不能记作三个独立环境。它们保留负physical margin，不改变部署G0的TRAIN选集。

## 4. 独立JSON矩阵复算

没有复用作者函数，直接以JSON中的D、A、Σ、M构造dense covariance与12×12 block Hessian。与保存量的最大差为：H=0、κ约1.11×10⁻¹⁶、风险约1.92×10⁻¹⁵、score约1.33×10⁻¹⁵、Ψ约1.39×10⁻¹⁷、A6下界约8.49×10⁻¹⁴。最后一项包含不同浮点运算顺序造成的风险误差传播，仍小于原1e−12容差。

这证明已存小矩阵上的公式应用一致；未独立重读/重建物理矩阵，也没有认证总体矩、实际Fine NLL或统计泛化。

## 5. CHAINED_SCORE的兼容性与计数

闭链使用相同slot/rep、原A/Q、target与pair索引；core gap来自同一个TRAIN尺度和同split合法增量，两个JSON对应值在1e−12内匹配。Abar的非零行缩放不改变score，所以可以把新仿射core下界与旧 `affine_to_actual` 的同pair score余项预算相接。

独立检查全部96行后，两种减法 `lower_affine−B_CS` 与 `lower_affine−B_corr` 和保存结果的差均为0；预算均非负，compatibility failure为0。全部四pair均正才计一个parent/split情境：

| 支持 | 仿射下界 | 接Cauchy score余项 | 接真值相关矩score余项 | 已存actual精确gap |
|---|---:|---:|---:|---:|
| TRAIN | 12/12 | 9/12 | 12/12 | 12/12 |
| CAL（固定TRAIN尺度） | 9/12 | 8/12 | 9/12 | 9/12 |

24个parent/split情境来自12父、4物理槽；不把TRAIN/CAL或相同物理槽的多个seed当独立环境。相关矩预算使用真值，不能把其12/12覆盖改称无标签证书。

## 6. 最终证据边界

这一闭链已把**给定数据导出的仿射core、真值有限矩、已知创新law和真值score余项**连到有限数组上的actual score下界。它没有把actual总体NLL excess接入，也未补齐一般非线性余项的总体概率界、uniform候选类事件或训练优化保证。原审计与主文保留了这些限制，未发现越界应用。
