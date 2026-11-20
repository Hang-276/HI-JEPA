# 当前确实使用 LeJEPA，但结论范围是本地低维机制研究

已核对实际训练记录及当前加载的冻结权重。16 个编码器来自论文作者仓库 `klindtlab/lejepa-identifiability`，固定 commit `de7503f1b2291ba1e9fc03d9c433180794803440`。原样调用作者的 MLP 构造、`alignment_loss` 和 `SIGReg`，训练目标为 `(1−λ) alignment + λ SIGReg`，λ=.001。损失和模型文件的哈希同时记入训练记录与当前加载校验。

本研究自行设定四个独立 Gaussian AR(1) 因素，通过可逆三角非线性和 Hadamard 混合生成观测；把相邻观测送入同一个编码器。没有另设预测头。使用本地 CPU、较小网络和训练预算，因此属于作者目标及网络构造的低维适配，不是原论文全部实验的复现，也不是大规模视觉 LeJEPA。

后续机制检查使用同一批冻结编码器输出的命名谱读出和监督 MLP 物理读出。当前只在历史强恢复的14个监督 MLP 上定位与修正残差，编码器和原 MLP 权重不更新。物理 CAL 标签用于解释误差及构造修正，不能称无监督修正收益。

可核对证据：

- [训练代码](../lejepa_slow_rank_20261112/training/run.py)，第19–24行原作者模块导入，第56–80行实际目标。
- [实际训练记录](../lejepa_slow_rank_20261112/training/runs/results.json)，记录作者来源、commit及两个源文件哈希。
- [训练设置](../lejepa_slow_rank_20261112/training/protocol.json)。
- [本轮沿用的冻结加载器](../lejepa_natural_direction_repair_20261113/loaders.py)，核验旧权重与命名参数的来源。

因此可以说“正在研究 LeJEPA 表示何时能可靠选出物理慢因素”，目前不能说“已经在视觉 LeJEPA 上验证了普遍规律”。
