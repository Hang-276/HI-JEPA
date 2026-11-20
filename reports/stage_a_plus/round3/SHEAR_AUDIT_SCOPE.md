# Round 3 单向shear边界：固定计算范围

2026-11-04。在本轮shear数值计算前固定。只读Round 2已存四槽物理TRAIN尺度、创新D_u及四份既有physical TRAIN/CAL数组。物理目标、H={4,16,32}、TRAIN方差口径不变；不读取Fine缓存、模型或FINAL，不做拟合、前向、训练、随机采样或服务器操作。

每槽以同一TRAIN尺度计算Σ=Cov_TRAIN(v)和M_split=平均_h E_split[Δ_hv Δ_hvᵀ]。CAL只换增量支持，不换Σ。每个支持均报告全部12条有向shear A=I+t e_i e_jᵀ（i≠j），不选择或删除方向。

每条shear只改变第i行。对冻结的物理坐标命名和原S*，计算所有涉及i的target/non-target差的二次多项式；第一次任一差为零的最小|t|为严格unique top-k边界。用80位Decimal对已算出的binary64系数求根与判别符，保留切触/穿越、无有限根及limiting tie区别。它不是FP32或lexicographic tie规则的精确错选阈值，也不代替原axis gate。

对应风险使用同一state gauge的真实创新矩阵，F_shear(t)=0.5*mean_u log(1+t² d_j(u)/d_i(u))，9regime权重均为1/9。这是数学候选仿射族的自由profile总体风险；没有测量actual Fine NLL。评分边界来自现有有限数组的矩，不冒充总体选择边界。

若支持在t=0已有非正物理target margin，记录BASE_TARGET_NOT_STRICT及零安全半径，不虚构正边界。全部方向风险边界的最小值只给一般仿射候选类首次失效风险的一个上界；不能据此保证所有较低风险的任意仿射/非线性表示都安全。优化后的轴重匹配不允许进入本诊断。

输出独立SHEAR_BARRIER_AUDIT.json/md和脚本，不覆盖之前报告或历史结果。所有输入、脚本及本scope记录SHA256。数值核验针对明确矩阵/多项式公式，不是总体统计认证。
