# 一般仿射局部理论：独立只读数学复核

日期：2026-11-04。复核对象：[AFFINE_LOCAL_THEORY.md](AFFINE_LOCAL_THEORY.md) 与 [affine_local_checks.py](affine_local_checks.py)。本次只读推导和脚本文本，没有执行作者脚本、读取新数组、训练、模型前向或改动作者文件。

**结论：未发现实质数学错误。** 以下说明已独立检查的常数、逻辑与必要范围。固定矩阵核验不能替代证明，主文的条件限制也不能省略。

## 1. Risk Taylor 与有限半径

diag(B)=0使每行条件方差精确等于 \(d_j[1+\sum_{k\ne j}B_{jk}²d_k/d_j]\)。配合 \(\det(ADA^T)=\det(A)²\det D\)，得到主文精确log-diagonal减logdet表达式。

\(0\le x-\log(1+x)\le x²/2\) 加上 \(\sum_j\|B_{j,:}\|⁴\le\|B\|_F⁴\)，确实给出variance项的 \(R²\|B\|_F⁴/4\) 余项。对非对称/非正规B，仍有

\[
|\operatorname{tr}B^k|\le\|B\|_F\|B^{k-1}\|_F
\le\|B\|_F²\|B\|_{op}^{k-2},
\]

所以主文的trace级数界及 \(r/[3(1-r)]\) 常数成立。op norm<1保证I+sB沿s∈[0,1]非奇异，det为正；本chart内无需漏记absolute determinant。

二次项恰为 \(b^THb/2\)：log-diagonal给两个方差比平方项，\(\operatorname{tr}B²/2\) 给每pair的 \(B_{ij}B_{ji}\)。H的off-diagonal为1，没有额外factor2。

若余项不超过κ||B||F²/4，则F≥bTHb/4。r₀中的 \(3κ/16\) 与 \(\sqrt{κ/(2R²)}\) 分别把两个预算压到κ/8，常数正确。

## 2. Uniform ball 与 fixed candidate

主文对整球使用统一R、r和κ；可选的δlog＋δdet保留指定B的精确variance截断误差及operator-norm determinant界。它合法地证明**该固定B**的coercivity，不证明整个相同Fro半径球都满足同样点态条件。

主文已明确这一点，也没有由小F直接推出候选进入chart。若后续用现有父模型的点态通过数，应继续保持“truth-assisted fixed-candidate域检查”的名称，不能变成训练器或全risk-ball的概率证书。

## 3. Score余项与Ψ

Tⱼ=M−sⱼΣ满足其jj元素为0。A4的分数恒等式同时保留分子交叉项与重新标准化分母；Σ正半定、Σjj=1即可使用seminorm反三角。

当√KΣ r<1时，分母至少为(1−√KΣ r)²；分子中线性项乘方差偏差的上界为

\[
2a_j^*\|b_j\|(2\sqrt{K_\Sigma}\|b_j\|+K_\Sigma\|b_j\|²),
\]

因此A5的 \(t_j^*+4a_j^*\sqrt K+2a_j^*Kr\) 系数正确。

按pair排列的off-diagonal向量使用全部B元素一次，norm恰为||B||F。ell的系数与脚本row-gradient实现一致。H-positive下

\[
|\ell^Tb|\le\sqrt{\ell^TH^{-1}\ell}\sqrt{b^THb}
\le2\sqrt{\Psi F}.
\]

pair二阶余项用 \(C_b\|b_b\|²+C_a\|b_a\|²\le C_{ab}\|B\|_F²\) 即可，不需要再额外乘2。因此A6中的 \(4C_{ab}F/κ\) 没有漏常数。统计score各轴ε相加为2ε也正确。非线性输出必须再加T−A的multi-horizon gap预算，主文已保留。

Σ=I、M diagonal时一阶敏感性为0但A8二阶项一般非零；它不等于无选择误差。该限定与Round2持久因素反例兼容。

## 4. 脚本文本检查

脚本固定context方差、Σ、M与B，不含PRNG、模型或数据读取。offdiag_vector与H/ell的排列一致。较大fixture超出uniform coercivity域时，其coercivity/gap断言明确标为仅在适用条件下检查；它仍检查Taylor和score余项，不把域外点作为A6实例。single-shear检查使用exact式，不借局部半径声称全仿射保证。

本次未重新运行脚本，因而结论是对公式及实现逻辑的独立只读复核，而非另一次数值执行记录。

## 5. 不可省略的研究边界

必须固定真实物理尺度、命名、diag(A)=1行尺度规范、finite M/Σ及uniform regime variance-ratio界。H必须正定；若某pair variance ratio在regime间恒定，general-affine Hessian可奇异，即使正交κ为正。

小risk没有证明局部chart事件；pointwise truth-fit A也没有自动成为实际神经表示的总体核心。现有CAL矩、经验NLL或单批余项不能替代总体参数/概率事件。该理论是具有完整余项与半径的受限数学连接，尚不是实际Fine训练的端到端可靠性结论。
