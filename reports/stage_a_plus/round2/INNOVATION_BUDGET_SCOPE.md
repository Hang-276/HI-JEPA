# 追加有界诊断：创新尺度预算

2026-11-04。在本追加诊断任何分组预算计算之前写入。

目的仅为评估 `NONLINEAR_THEORY_REVIEW.md` N8/N10的量级。使用已完成12父的A/Q、b、原CAL映射及全部TRAIN尺度，不重复OLS/polar、不改任何已完成文件。重新只读既有TRAIN/CAL Fine、物理数组，以及驱动一步next状态的 `regimes[:,t]`。不使用未来时刻的下一个regime。

来源已按文本/JSON核查，不import生成模块：当前confirmation/data_owner从固定 `registration/INSTANCE_LOCK.json` 指定物理参数；AR原 `fine_stage.py` 的rich分支逐轴base std=.12，u=5…8分别对应一轴std×3；FHN原 `env_fhn.py::regimes(cfg)` 使用kick_std=.025、variance_multiplier=3、contrast_strength=1、rotational=false，当前两个FHN配置measurement_noise=0。执行时保存源码与manifest哈希并核对原登记引用。若不匹配或缺失则保留失败，不更换模型/配置。

固定state gauge为前一审计真实物理TRAIN逐轴std，`D_u=diag(physical_innovation_std(u)^2 / physical_TRAIN_std^2)`。对每父两core B=A与B=Q，残差为 `r_next=W_next−(B V_next+b)`；计算九regime内的经验 `mean[r_nextᵀ(B D_u Bᵀ)^−1 r_next]`，以及仅按regime减样本均值后的版本。后者只作分组矩统计，不拟合条件均值/神经网络。

全部12父×两core×TRAIN/CAL×9regime保留；只对ν<1的格报告 `4 log((1+ν)/(1−ν))`，其余N/A。可并列按经验频率和均匀1/9加权的预算，但任一regimeν≥1则完整九regime预算N/A，不删该格后平均。可列同固定核心的解析Gaussian diagonality contrast，不能称actual Fine风险实测。

这些是N10总体上界表达式的**有限数组经验plug-in**；不是条件中心化创新余项的实测、不是总体ν上界、不是无标签或高概率风险认证。ν<1经验成立也不据此认证N8条件。Gaussian-head条件KL/NLL excess语义保留，不把它泛称mutual-information lower bound。

成本上限：不新增数据/PRNG、OLS、polar或NN；每父core/regime至多一次4×4 Cholesky（12×2×9=216），每split的分组残差各一次三角solve（共432），并保存九regime计数、全部raw/centered量。零训练/前向/checkpoint反序列化/FINAL数组/GPU。只新写本scope、`INNOVATION_BUDGET_AUDIT.md/json`及独立脚本。
