# 本轮独审修正记录

首次oracle门槛草稿把已存浮点OLS residual MSE直接当作数学最小MSE下界。独审指出：数值解的residual原则上是最小值的上界，须扣除正规方程残差误差，不能直接称形式化不可能性证书。

最终审计仅用已存summary补扣 `5*g_inf²/lambda_min(designᵀdesign/N)`；g_inf字段确实包含四个状态列与intercept，design singular values未除sqrt(N)，来源静态核对见Round2脚本。另减1e−10的MSE敏感性余量，完整保留原MSE、校正项及余量。此余量是显式数值敏感性检查，不是假装已证明的interval rounding bound。

最终措辞改为“FP64 oracle对这个充分界的否决”，不称形式化机器证书，不推断actual模型错误，也不外推到总体。新增axis-specific同一范数界的检查，仍只连接旧summary，无新数组、拟合或模型执行。此前初稿文件在本轮交付前修正；原Round2/3/4交付与原科学结果未动。

FEASIBILITY_SCOPE在首个G0阈值算术后追加了这项旧真值summary的否决用途。原G0结果保留执行时scope哈希；最终scope明确标明追加发生在oracle连接之前，不把它伪称为首次计算前已有。初始范围正文可由当前scope的追加标题前正文恢复并核对该哈希。
