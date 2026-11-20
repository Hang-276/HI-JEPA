# Posterior 补桥与无标签下界的独立复核记录

2026-11-04。无新文献/物理数据/数组/模型读取，无采样或训练。固定概率向量脚本仅核算确定性不等式，不作为模型实验。

## 1. 近似 posterior 桥

请求的常数正确：自然对数 Pinsker＋log 的 ε⁻¹-Lipschitz 给 \(E\|r\|^2\le4(m-1)a/\epsilon^2\)；同一 context 的 \(\sigma_{min}(\hat L)\ge s\) 再给统计量 span 残差 L2≤\(2\sqrt{(m-1)a}/(\epsilon s)\)。六组固定 m=9 概率向量以及 context 平均、Ω 归一化的确定性检查全部通过，见 [检查 JSON](APPROXIMATE_POSTERIOR_CHECKS.json)。

该结论必须保留四个限制：全域 Gaussian/full-support/full-rank 与所有 posterior 分量的统一正 ε 不能兼容；平均 context 风险不能变成任意固定 context 风险；有限 Ω 的尾部和概率质量须另控；近似 sufficient-statistic span 未证明为 axis-affine 余项。σmin 的值还依赖 latent gauge 与 basis norm，不能任意 rescale 后作物理强度比较。详见 [APPROXIMATE_POSTERIOR_BRIDGE.md](APPROXIMATE_POSTERIOR_BRIDGE.md)。

精确 KL excess 需要 Bayes 基准，不等于任何可观测上界都不可能：总体 KL≤观测 regime 的 multiclass cross-entropy，后者无需物理源标签，但含不可约 entropy，posterior 正确时仍通常为正。本轮没有证明这个粗界足够产生当前非空预算，未计算当前数据。

## 2. UNLABELED_LIMITS 只读数学复核

独立阅读 [UNLABELED_LIMITS.md](UNLABELED_LIMITS.md)；U1 的 mean/variance Gaussian KL 常数、每 τ>0 的 natural rank8、twist 的 Jacobian/导数界、U4 的级数 \((1+2k)^{-2}\) 及 residual 系数2、K0≤τ²/3、归一 Hellinger 的两点 product-TV 界 \(2\sqrt{nK0}\) 均正确，**未发现实质数学错误**。每 τ full-rank 但无 uniform variation 下界、U2 为 IID且未证明正 slow gap、有限实际 flow 包含性未证，这些 scope 限定均必要且已保留。

一处表述宜量化：“小误报和非空 power 不能同时保证”若将非空理解为任何正数则过强；独立随机以 δ 概率 pass 即有 false-positive=power=δ。U3 实际给出 \(power\le false\ positive+TV\)；例如 TV≤1/4 时，误报≤δ 推出 power≤δ+1/4。故应写“不能统一保证预定的非平凡 power 与小误报”，或直接写“两端错误之和至少3/4”。此为文字边界建议，不改变构造或公式。

本复核没有修改 UNLABELED_LIMITS 作者文件。来源文本 SHA256、补桥文件及脚本/JSON SHA256 见 [FINE_VALUE_SOURCE_MANIFEST.json](FINE_VALUE_SOURCE_MANIFEST.json)。

## 3. 总报告指定段落复核

只读核对 [ROUND4_REPORT.md](ROUND4_REPORT.md) §4 的近似式、tail/gauge，以及 §6 Fine 价值与 §7 去留建议。近似式常数、Gaussian 全域 ε 的不兼容性、clipping/basis 限定和 Fine 证据边界均正确。

已反馈两处精度建议：区域公式明确 a 与 L2 是否均条件于同一个 Ω；若 a 不归一而 L2 条件于 Ω，需额外除以 \(\sqrt{P(\Omega\mid c)}\)。固定 E 的 raw affine 与 E 后 affine 类“一般不能断言彼此包含”，不能说每个 E 下都互不包含，因为 affine/identity E 给相等的类。其余指定段落未发现实质数学或科学结论错误；本复核未修改作者总报告。
