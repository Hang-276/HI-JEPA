# 单向 Shear 风险边界独立复核

日期：2026-11-04。审查对象：`SHEAR_BARRIER_REVIEW.md`、`shear_barrier_audit.py`、`SHEAR_BARRIER_AUDIT.json`。原作者文件未修改，原审计main未重跑；本次仅读取保存的矩与参数，并增加固定解析corner-case，不读取新物理数组、不前向模型、不训练。

## 1. 结论

**未发现实质数学或实现错误。** S1 Gaussian profile risk、S2 normalized score、二次gap的有限根/无有限根/切触分类、沿指定shear的严格唯一top-k风险球边界以及负baseline处理均成立。当前结果也正确保留“全affine失败inf的上界”与“全affine安全保证”之间的方向差别。

`shear_independent_review.py` 与 `SHEAR_INDEPENDENT_CHECKS.json` 逐项复核96个direction/split记录、168个合法baseline下的gap约束，并另外测试八个root分支。八类检查全部通过。存在84条正baseline shear记录与12条负baseline记录；没有将后者继续当作正margin路径计算。

## 2. 实际finite矩的一般Σ公式正确

论文正文选择了Σjj=1的物理规范；脚本实际上保留了任意正对角Σ的形式。这是正确的数值防护。令 \(S_k=M_{kk}/\Sigma_{kk}\)、\(\sigma=+1\) 表示affected轴是非目标，\(\sigma=-1\) 表示其为目标，则脚本使用

\[
q_k(t)=\sigma\{M_{ii}-S_k\Sigma_{ii}
+2t(M_{ij}-S_k\Sigma_{ij})
+t^2(M_{jj}-S_k\Sigma_{jj})\}.
\]

它恰为目标排序gap乘以affected轴方差
\(V_i(t)=\Sigma_{ii}+2t\Sigma_{ij}+t^2\Sigma_{jj}\)。只要该2×2 principal covariance正定，所有有限t的分母都正。归一化输出仍用同一TRAIN-Q0 covariance；CAL只改变Qh的增量矩M。这没有把CAL自己的variance混入分母。

独立核验既复查保存Σ/M，也加入了明确非单位对角Σ的小矩阵fixture；未发现依赖“浮点Σ对角恰好为1”的错误。脚本risk所用D也保持同一TRAIN物理标准化尺度。

## 3. root、切触与无有限tie

对g>0，\(q=g+2\beta t+\alpha t^2\) 的全部分类正确：

- α<0具有异号穿越根；α=0且β≠0为线性根。
- α=β=0没有有限根，normalized gap只在无穷参数处趋零。
- α>0且判别式负，没有有限根且全方向gap有正下界。
- α>0且判别式零为有限切触，严格唯一性在该点失败，但不意味着根两侧反序。
- α>0且判别式正为同号两根，中间反序。

最近根半径 \(g/(|\beta|+\sqrt{\beta^2-\alpha g})\) 与符号选择均正确；α=β=0被单独处理，没有错误计算0/0。Decimal高精度分类没有把近切触强行调成成功或失败。补充fixture验证了现有结果中没有出现的limiting tie与精确切触分支；generic浮点多项式求根对重根会产生小分裂，因此独立fixture对切触使用精确重复根，而非放宽判别阈值。

所有相关q在0处为正；最近root之前不可能先变负，因而min root确实给沿该shear的完整固定选集边界。对source j本身的比较也包含在相应target/non-target集合中，没有只检查混合pair而漏掉其他target边界。

## 4. 从单shear到一般affine的正确不等式方向

将数学目标明确为严格唯一top-k，定义

\[
R_{\rm aff}^{\rm fail}=\inf\{F(A):A\text{可逆且固定目标不再是唯一top-k}\}.
\]

每个单shear族均是此类的子集，所以

\[
R_{\rm aff}^{\rm fail}\le\min_{i\ne j}\mathcal R_{ij}^{\rm shear}.
\]

这是全affine失败成本的**上界**，不能倒转成全affine的安全半径下界。若某些方向没有有限tie，其barrier为∞；这不证明其他affine方向安全。最小barrier只足以控制列出的单shear族的并集，不能控制多shear复合。

还应持续保留**严格唯一**这个目标限定：对FP32/lex实现，切触且priority偏目标的点可能并非实际错选点。当前正文§6已说明，因此没有将严格tie边界误称为实现层必要充分保证。建议任何转述JSON的`is_upper_bound_on_all_affine_failure_radius`时都同时写“strict unique-top-k”，避免字段名脱离scope后被误读。当前保存结果没有切触行，这一措辞主要保护理论外推。

## 5. baseline、风险与证据层级

FHN ε=.5的CAL物理目标margin约−.020989，identity已不是声明目标的严格top-k。因此该split的全部12个方向均记`BASE_TARGET_NOT_STRICT`、半径0且不继续计算root，是正确处理；不能把负baseline通过改变shear方向重新解释成已有可靠性。

其余七个slot/split合计58个有限strict-tie方向、26个无有限tie方向；96行总数与8个baseline矩记录一致。risk由给定D、uniform1/9 regime law和理想free-profile式求得；score几何来自保存TRAIN/CAL有限矩。这个组合是明确的truth-assisted sensitivity诊断，不是已证明总体矩、实际Fine总体excess或新的独立确认。

独立核验只确认上述公式、字段与解析边界一致，没有重新读取训练数据证明Moments估计精度，也没有验证任何checkpoint的神经实现。未发现需要纠正原审计数值的事项。

