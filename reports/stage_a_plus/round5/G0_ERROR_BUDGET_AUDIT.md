# 现有 G0 分数需要多小的轴误差：同支持门槛与 oracle 否决

2026-11-04。范围见 [FEASIBILITY_SCOPE](FEASIBILITY_SCOPE.md)。只读取12份原生G0.json和一份既有Round2 summary，不读取数组、不加载模型或重新拟合。这里的门槛是**同一经验TRAIN支持**上的确定性充分条件，不是总体保证。

## 1. 不使用物理标签反推门槛

对每个原生Fine坐标，原G0以全TRAIN的L个时点归一化，各h使用L−h个合法起点。因此端点相对全时点经验measure的密度比≤L/(L−h)。令

\[
K=\operatorname{avg}_{h\in\{4,16,32\}}4L/(L-h).
\]

如果Fine与某一signed-permuted物理轴均在同一TRAIN参考下标准化，且逐轴L²差≤η_j，则 [S4](TRANSPORT_AND_SELECTION.md) 给

\[
\beta_j\le2\sqrt{K S_j}\eta_j+K\eta_j^2.
\]

S_j直接来自原生G0的FP64平均score；原FP32选集保持不动。每个选中a和未选中b都需gap>β_a+β_b。若使用共同η，设A_ab=2√K(√S_a+√S_b)、B=2K，严格门为

\[
\eta<\min_{a,b}\frac{2g_{ab}}{A_{ab}+\sqrt{A_{ab}^2+4Bg_{ab}}}.
\]

实际FP32选集在全部12份记录上也对应正FP64跨边界gap；数值舍入差保留在JSON。未读取物理标签或重新选轴。完整48个pair见 [G0_ERROR_BUDGET_AUDIT.json](G0_ERROR_BUDGET_AUDIT.json)。

| 物理实例槽（各3父） | 当前有限TRAIN最小gap范围 | 共同每轴η严格门范围 |
|---|---:|---:|
| AR rich fresh1 | .565204–.567152 | .048098–.048256 |
| AR rich fresh2 | .605450–.607465 | .051778–.051943 |
| FHN ε=.08 | .212409–.213587 | .025684–.025834 |
| FHN ε=.5 | .021989–.024684 | .001913–.002148 |

这个表只反推“上界需要有多小”。它没有测得实际η，也没有保证这些分数代表总体。四槽只有4个物理实例、12个优化父。

## 2. 旧真值诊断只用于否决充分界

Round2已经保存同TRAIN支持、标准化Fine对完整物理仿射空间的OLS拟合。任意normalized signed-permuted单轴都属于该仿射空间，所以精确OLS最小residual MSE是对应轴差平方的下界。这个下界可用于问“即使有理想上界，当前充分门能否通过”，不能改称无标签接受式证书。

数值OLS解的residual本身原则上是最小值的上界。最终审计按独审意见，使用已存design最小奇异值和包含intercept的正规方程残差扣除

\[
\frac{5g_\infty^2}{\sigma_{min}(design)^2/N},
\]

再扣1e−10 MSE敏感性余量。前一项约10⁻³⁰–10⁻²⁹；后一项未改变任何结论。存储统计仍是FP64，**这不是interval arithmetic形式证明**。修正过程见 [REVIEW_FIXES](REVIEW_FIXES.md)。

| 槽 | 共同η的数值oracle下界范围 | 共同η门被否决 | 分轴η_j的同一范数门被否决 |
|---|---:|---:|---:|
| AR rich fresh1 | .03366–.03513 | 0/3 | 0/3 |
| AR rich fresh2 | .03126–.03592 | 0/3 | 0/3 |
| FHN ε=.08 | .03779–.03961 | 3/3 | 0/3 |
| FHN ε=.5 | .02178–.02427 | 3/3 | 3/3 |

分轴版本将旧summary的matched_indices只用于把各输出的OLS residual norm还原到原生Fine坐标，再分别代入β_j；不重拟合匹配。FHN ε=.08共同门失败不能推广到分轴门，改用分轴下界后它们并未被否决。**未被否决的9父不等于通过**，因为下界不是可用的实际误差上界。

FHN ε=.5三个父即使只代入各轴误差的旧oracle下界，当前分轴范数充分界也已超过相应gap。这是当前证法的明确限制，不是模型选错，也不否定更紧的signed pair-correlation界。Round3的真值相关矩闭链仍可通过这些TRAIN父，两个结论不矛盾。

完整原MSE、数值校正、逐轴下界和pair否决见 [ORACLE_GATE_FLOOR_AUDIT.json](ORACLE_GATE_FLOOR_AUDIT.json)。上述否决只限同一有限TRAIN支持，不直接否定一个尚未建立的总体或新支持证书。

## 3. 样本和总体条件的地位

前轮定义的独立episode有界witness界，在δ=.05、一个固定模型、九个预定类别witness下，n=24/32的保守半径约.700/.607。这只是预先规定的Hoeffding设计量级，未运行新witness；也没有把选点CAL当独立审计。

它不是所有方法的样本复杂度下界。可证明的mixing、martingale结构、参数限制或更紧方差界可能利用更多transition信息；当前没有建立这些替代统计条件。不能将2048个抽样transition默认当独立，也不能反过来声称最多只有24/32条信息。

本轮没有实际posterior KL上界、anchor covariance下界、总体head尾矩或anchor到Q0的密度比预算，因此没有任何父被宣布通过无标签总体证书。原科学结果和评价门未改变。
