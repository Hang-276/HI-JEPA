# Exact posterior 接口独立复核

2026-11-04。只读复核 `OBSERVABLE_CERTIFICATE_OPTIONS.md` §3、`OBSERVABILITY_LEDGER.md` 及 `ROUND4_REPORT.md` §3–4。未读取科学数据数组或模型权重，未做模型执行、拟合、采样或新数值诊断；未修改被审文件。本记录审查数学推导和范围转述，不重新认证其文献新颖性或源码机制。

**结论：主证明成立，未发现需要修改结论的数学错误。两处语义精确化已反馈作者。**

## 1. Exact 主证明

1. 联合 posterior a.e. 正确经 disintegration 给出满测的合法 context 集。它与正测满秩集相交，选其中同一个 c 即可；无需把不同 c 的 contrast 行拼接。
2. 真实 Gaussian 全支撑、正 prior 及共同可逆表示使同一 c 的 posterior odds 等于真实/learned 密度比。正确 prior 被扣除，得到 O4。只需 learned contrast 矩阵可逆，就能解出 O5；无需另外假设真实 contrast 可逆。
3. O5 两侧连续且在具有正密度的真实因素律下 a.e. 相等，故在整个因素空间相等。前 d 行将每个 g_j 限制为无交叉项的至多二次多项式；后 d 行控制 g_j²。非零二次齐次项平方产生不可消除的四次项，故 g_j 为 affine；平方中的交叉项又迫使每行至多一个非零系数。可逆性给出置换和逐轴非零 affine，结论 O2 成立。
4. 因为 g 本身不依赖 c，在一个合法 c 得到该函数恒等式已经足够推出共同表示的结论。实际 posterior 正确性、learned rank、连续总体与 FP32 记录之间的近似均未由本证明认证；正文已明确保留这些缺口。

## 2. 当前 head 的 rank 简化

设 λ_ju=σ_ju⁻²，μ_ju(c)=m_j(c)+β_ju。自然参数 contrast 的一次列为

`m_j(c) Δλ_ju + Δ(β_ju λ_ju)`，

二次列为 `−Δλ_ju/2`。因此将一次列加 `2m_j(c)` 倍二次列，恰好消去 context 项。这是可逆列操作，故 rank 与 c 无关；该结论仅使用当前 head 的均值加法结构与 regime-only 方差。它没有认证 embedding 的实际数值 rank。O6 的 mean/variance 分块三角矩阵亦正确，variance ratio 为 9。

## 3. 完整向量 conditional calibration 的范围

对某个合法且 Ŵ(c) 满秩的 c，完整预测概率向量的 log-odds可逆地确定 T(E(x′))，其前 d 个分量确定 E(x′)，进而确定 x′。这些是标准 Borel 空间中的可测单射，故在该 c 上，预测向量与 x′ 生成相同信息。因此 exact conditional full-vector calibration 在该 c 上等价于真实 posterior 正确。

建议 OPTIONS §3.5 将“二者生成相同的观测 σ 代数”明确限定为上述满秩的合法 c，或满秩 context 集上的限制。若要直接声称全局 `σ(C,X′)=σ(C,π̂(C,X′))`，则应要求 Ŵ(C) 几乎处处满秩。当前主定理只需正测满秩，不必增强它：在该正测集合中由完整向量校准得到 posterior 正确，再选一个合法 c 即可启动主证明。

这项限定不改变 exact 结论；普通 ECE、有限分箱或 classwise calibration 仍不足够。

## 4. 总报告 §3–4

反例被准确限定为未给辅助变化强度下界的整个 full-rank 类；没有否定固定 rich 配方的总体辨识，也没有声称实际 Fine 已失败。`TV≤1/4` 且错误解释下误认证概率 `≤δ` 时，正确解释下通过概率 `≤δ+1/4` 的量化表述正确，没有错误排除严格正但很小的随机 power。

唯一建议是把 §3 “完整观测风险仍≤τ²/3”明确写为“固定 Gaussian head 的完整观测 NLL 总体 excess≤τ²/3”。该构造控制的不是绝对 NLL。§4 的 exact 端点与 finite/approximate 缺口转述准确；后验下界只能在受控区域使用，统计量近似关系没有被冒充成轴余项或 G0 保证。

**复核状态：** exact 理论接口通过；actual Fine 的总体 posterior、rank、有限误差与选集认证保持未解决。
