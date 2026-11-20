# Round6 理论接口独立复核

2026-11-04。只读复核，不执行作者脚本，不读科学数组或模型，不采样、不拟合。本记录不把自检写成独立复核：`CORRELATION_AWARE_SELECTION.md`由本代理撰写，已交root独立审C3–C13并通过；下面的Dirichlet、综合稿和posterior接口由本代理独立检查。

## 1. Dirichlet 与总体同时区间

`DIRICHLET_SELECTION.md` 的D2 adjoint包含两个端点各自的density-ratio weight，不缺额外1/2。D3是精确归一化恒等式；D5在η≤√2时使用的q(η)单调性正确。D6 pair的上/下界方向、D7 fidelity区间和D8共同η门的角度因子均正确。有限TRAIN图算子与总体观测算子已区分；旧OLS数值floor只用于否决充分门或计算乐观空间，没有当作误差上界。

`ROUND6_REPORT.md` §4将D6写为

`Sb(1−qb²)−Sa(1−qa²)−Λqa²−2ρa qa−2ρb qb`。

在η≤√2时q²≤1，因此对Sb取下置信端点、Sa取上端点安全；动态残差取有效非负上界亦安全。前端η、tail/transport、匹配和所有置信事件须同时有效，正文保留了这些条件。当前实际Fine没有新增通过事件的表述准确。

## 2. Fine-specific 完整后验向量的信息充分性

若对几乎处处合法C，learned 8×8 contrast L(C)可逆，q的九个分量严格正，且共同Fine E有可测逆，则

`T(E(x′))=L(C)⁻¹[log(q1/q0),…,log(q8/q0)]−L(C)⁻¹b(C)`。

取前四个坐标并用E逆，显式恢复x′，故`σ(C,q)=σ(C,x′)`。因此`I(U;X′|C,q)=0`，完整conditional vector auto-calibration等价于真实posterior正确。此信息充分性本身不要求真实kernel为Gaussian；真实Gaussian/full-rank结构仍是随后axis-affine辨识的另一个前提。

不能把constant-q的一般coarsening反例用于否定这个注入性特例。有限bin B(q)仍可能有`I(U;q|C,B)>0`；普通ECE、不含C的校准与完整conditional vector calibration不同。actual learned rank未验证，FP32有限量化也不能当成exact实数向量的注入性证明。

## 3. Posterior/Brier主式的先行审查

综合稿中的精确式

`EΩ||p−q||² = BrierRiskΩ(q) − Σb μb(1−||pb||²) + Σb μb E[||p−pb||²|bin b]`

方向正确，不要求q在bin内恒定。需将bins明确限定为Ω内cells；等价地用`μb=P(Ω∩{B=b})`及`pb=E[p|Ω,B=b]`，不能减去域外的bin噪声。以`KL(p||q)≤||p−q||²/ε`进入Ω下restricted KL亦正确。最后的within-bin true-posterior variation是必须控制的剩余项，不能用q自身光滑性或有限calibration点值替代。

`OBSERVABLE_POSTERIOR_BOUND.md` 作者尚在落盘；其有限样本细节在文件出现后再补最终审查。此状态不应被记录成完整posterior分支已经独审完成。
