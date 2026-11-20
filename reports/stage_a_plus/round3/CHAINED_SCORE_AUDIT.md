# Round 3：已有仿射下界与Fine评分余项的有限数组闭链

日期：2026-11-04。按 [追加scope](CHAINED_SCORE_SCOPE.md)，只join两份已完成JSON；没有重新读取数组、拟合、前向、随机数或FINAL/GPU操作，已有文件均未改。

**结果：TRAIN使用Cauchy余项的闭链覆盖9/12父，使用真值相关矩余项覆盖12/12；CAL分别8/12和9/12。** 两条预算全部保留。这里闭合的是“已知仿射core的局部风险几何＋真实有限Σ/M＋真实Fine评分余项”的确定性链，**不是**“观测CAL NLL或J_actual→实际Fine”的总体、高概率或无标签认证。

完整96行pair记录与输入hash见 [CHAINED_SCORE_AUDIT.json](CHAINED_SCORE_AUDIT.json)，脚本为 [chained_score_audit.py](chained_score_audit.py)。

## 1 为什么两份结果可直接连接

仿射来源为 [AFFINE_GEOMETRY_AUDIT.json](AFFINE_GEOMETRY_AUDIT.json)，余项来源为 [Round2 NONLINEAR_REMAINDER_AUDIT.json](../round2/NONLINEAR_REMAINDER_AUDIT.json) 的 `affine_to_actual` 分解。

逐父核对相同slot/rep、原A/Q、target标签与TRAIN尺度；逐pair核对归一化core gap。Round3的Abar只是原A的非零逐行缩放，自有TRAIN方差归一化score不变。TRAIN/CAL各使用相同合法horizon/端点，CAL始终沿用TRAIN方差。96/96 pair兼容，0 N/A，未重拟合或挑选core。

仿射下界L_A已经使用正式局部余项常数。uniform球条件仅1/12通过，而pointwise已知B条件12/12通过；闭链沿用这一区分，没有把pointwise成立冒充整个球成立。它也没有把拟合A的解析Gaussian profile风险改称实际Fine风险。

## 2 固定两条非精确评分预算

对目标a、非目标b，Round2给

\[
g_W=g_A+D_{A\to W},\qquad |D_{A\to W}|\le B_{ab}.
\]

因此已有\(g_A\ge L_A\)时，

\[
g_W\ge L_A-B^{CS}_{ab},\qquad
g_W\ge L_A-B^{corr}_{ab}.
\]

本轮固定使用Round2的 `CS_bound_measured_Q0_correction` 与 `truth_assisted_pair_correlation_triangle_bound`。前者是保留TRAIN方差差的Cauchy界；后者对增量cross、残差能量、归一化修正的pair贡献分别取绝对值。两者都需要有真值的core/余项，且都不是将exact实际gap直接当作证明。

每父/split全部四target pair下界严格正，才计为该情境覆盖。现成exact gap只用于核查下界，未用于选预算或改变门。

| 增量支持 | affine下界四pair均正 | 闭链CS四pair均正 | 闭链相关界四pair均正 | exact实际四pair均正 |
|---|---:|---:|---:|---:|
| TRAIN | 12/12 | 9/12 | 12/12 | 12/12 |
| CAL，固定TRAIN尺度 | 9/12 | 8/12 | 9/12 | 9/12 |

TRAIN逐pair覆盖CS为36/48、相关界48/48；CAL为35/48与36/48。这里原12父来自4槽；TRAIN/CAL共24个parent/split情境不等于24个独立物理环境。

## 3 全12父的最小闭链下界

下表每格为同父相应split全部四pair下界的最小值。负数全部保留，未改目标或作结果后救援。

| 槽/rep | TRAIN CS下界 | TRAIN相关下界 | CAL CS下界 | CAL相关下界 |
|---|---:|---:|---:|---:|
| AR1/0 | +.3105029 | +.4911233 | +.3278890 | +.4970037 |
| AR1/1 | +.3165279 | +.5014269 | +.3298983 | +.5082519 |
| AR1/2 | +.3401679 | +.5111484 | +.3559047 | +.5205656 |
| AR2/0 | +.3258484 | +.4923525 | +.2600492 | +.4135236 |
| AR2/1 | +.3488983 | +.5255163 | +.2904325 | +.4559782 |
| AR2/2 | +.3744302 | +.5377200 | +.3170725 | +.4647958 |
| FHN.08/0 | +.0289403 | +.1261451 | +.0039385 | +.0948527 |
| FHN.08/1 | +.0153739 | +.1074968 | −.0078326 | +.0809751 |
| FHN.08/2 | +.0349746 | +.1314096 | +.0117802 | +.0989280 |
| FHN.5/0 | −.1105736 | +.0091211 | −.1781119 | −.0680831 |
| FHN.5/1 | −.0967393 | +.0086262 | −.1822216 | −.0761386 |
| FHN.5/2 | −.0726143 | +.0145995 | −.1433620 | −.0458612 |

FHN.5三个TRAIN父的CS闭链都失败，相关矩闭链仍可给正数；不能只报告后一列。CAL上FHN.08 rep1的CS下界为−.0078326而exact实际margin为+.1657515，属于充分条件过松。FHN.5的CAL负下界则同时对应真实物理目标margin已负和actual Fine margin已负，不能用表示近似保证救回不成立的目标排序前提。

## 4 CAL负例与部署G0的边界

FHN.5 CAL的真实physical gap为−.0209892692，来自一个已有24轨迹物理实例；三Fine父共享该物理样本，不计为三个独立环境。CAL只是假想替换增量支持的诊断。原部署G0仍按TRAIN冻结选集，经固定CAL一一匹配后对应物理轴[1,3]（0-based），部署使用各父原生Fine索引；原Fine＋G0联合成功12/12没有改变。

上述闭链也没有给总体gap置信界，未处理新数据上的Fine/OLS误差，未将条件innovation余项估成总体量，未从actual NLL排除远处等价解或证明学习器进入局部chart。Round2创新预算量级失败仍保留，未被本次有限数据正下界覆盖或推翻。

## 5 执行与成本

仅读取两份既有JSON并核对其输入来源hash，保存全部24情境/96pair及兼容性结果。新增数组读取、OLS/polar、神经训练/前向、PRNG、新物理数据、checkpoint加载、FINAL或GPU任务均为0。普通FP64下全部现成exact实际gap均不小于相应闭链下界（容差1e−12）；这不是interval arithmetic形式化浮点证明，也不是新的独立科学确认。
