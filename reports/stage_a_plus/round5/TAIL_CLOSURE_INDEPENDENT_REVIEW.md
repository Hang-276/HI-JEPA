# Round 5：尾部、定量闭包与迁移的独立数学复核

2026-11-04。只读核对同轮作者稿件，不修改其文件；没有新文献检索、模型/科学数组读取、训练或采样。此前固定解析 checks 保持不变且不重跑。

## 1. QUANTITATIVE_CLOSURE

[QUANTITATIVE_CLOSURE.md](QUANTITATIVE_CLOSURE.md) 的 tilt-Hermite 提取逐项核对如下：A9 为6a_i²；A12 为ℓ_iℓ_k；A14 为3√2ℓ_i a_i；A15 为√2ℓ_i a_k。方差1/2下的缩放、Gaussian正交及奇偶性均给出这些系数。

A10 的极值点为 t=(7±√22)/2，C4*<1成立；A14、A15的有界witness常数均严格小于2，故K3=2有效。A18的常数 Cd=8d²+16d(d−1)/e²、A19 signed-unit换算及 A20 pairwise covariance 非塌缩门均正确。cubic 提取把quadratic系数压到 O(δ)，最终每行表示L2为O(δ)，而非误将平方误差当成L2；近塌缩/小旋转构造的系数也一致。未发现实质数学错误。

唯一文字限定：只有**仅依赖被tilt坐标I**的有限Hermite多项式乘rI必有界；实际所有witness都符合这一点，故不影响证明。所选context、Gaussian总体norm、输出规范与 covariance λ 门仍为真实假设，不是实际Fine已测事实。

衔接时，本分支 T7 已在 anchor Q 下并含 mixed-law A→anchor 的 π0 换算；不能再重复除√π0。闭包§7.2针对另一个“已有mixture-law误差”的输入所写换算本身正确。

## 2. TRANSPORT_AND_SELECTION

[TRANSPORT_AND_SELECTION.md](TRANSPORT_AND_SELECTION.md) 的 S1 可由 r²≤2Y²+2m²V² 与Hölder直接得到；Cp的Gaussian绝对矩公式正确，p=2尾项常数确为2√3。Q=N(0,1)、Q0=N(0,v)时密度比矩的有限条件 p<v/(v−1) 正确。

S2 的归一化常数2/σY正确，且非零m/参考方差条件已保留；S3/S4 的score交叉项、平均Cauchy以及S5完整cross-target margin均正确。Q0≪Q、存在p>1可积、共同观测双射、同Q0尺度与合法endpoint coverage是必要前提；文稿未将经验TRAIN支持或单一anchor直接偷换为这些条件。未发现实质公式错误。

若将“非空链条”写成渐近命题，应明确选择W的联合条件：W E_j→0且TY,j(W)、Mw,p(W)^(1/p)→0，并有正的总体方差/目标gap及端点预算。固定W下的保守迁移尾项可能不随anchor误差归零；“迁移尾矩可控”应按上述意义理解，不能仅指某个有限正数。单个固定预算的S1–S5无需这一渐近补充也已成立。

## 3. 本分支尾部界的两项收尾修正

[TAIL_POSTERIOR_BOUND.md](TAIL_POSTERIOR_BOUND.md) 已补：非空极限除了A/ε→0、t/Htail→0，还必须有Kq²d→0；固定α、π0而ε→0时，(A/ε)log²(1/ε)→0是明确充分条件。单独A/ε→0不足以控制head坏集合能量。CE饱和证明另单列Q(Ω)=0时直接t=1/k=1，避免用严格“0>0”。这些修正不改变固定解析三例或脚本/JSON。

本轮数学接口已条件性接齐，实际posterior KL、Gaussian结构、总体规范/λ、迁移尾矩和统计置信预算并未因证明接通而获得。来源与本分支最终文件哈希见 [TAIL_POSTERIOR_SOURCE_MANIFEST.json](TAIL_POSTERIOR_SOURCE_MANIFEST.json)。
