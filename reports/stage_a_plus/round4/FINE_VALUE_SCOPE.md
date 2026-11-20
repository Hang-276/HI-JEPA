# Round 4：Fine 独立价值与 primary 近邻核查范围

日期：2026-11-04。任务由根代理分配，承接 Round 2/3，不重做 Stage A。

本分支只完成假设类/复合函数分析、已有 Markdown 报告的只读复核及定向 primary 论文查证。仅在本 `round4/` 目录新增 `FINE_VALUE_ROUTE.md`、`PRIMARY_NEIGHBORS.md`、本范围文件及 `FINE_VALUE_SOURCE_MANIFEST.json`（来源哈希/联网阅读记录）；后续追加数学核验另新增 `APPROXIMATE_POSTERIOR_BRIDGE.md`、`approximate_posterior_checks.py`、`APPROXIMATE_POSTERIOR_CHECKS.json`、`POSTERIOR_AND_LIMITS_REVIEW.md`。不修改原报告、模型、配置、评价门或历史结果。

读取的本地证据：`../round2/FINE_VALUE_AUDIT.md`、`../round3/ROUND3_REPORT.md`；追加只读独立数学审查 `UNLABELED_LIMITS.md` 及 `ROUND4_REPORT.md` 指定的 §4/§6/§7。原 Fine/原动态规则的实现事实和数值仅按前两份已经审计的报告转述，本轮不重新执行模型或数据诊断，不把报告转述升级为新复验。

联网范围：有限样本/定量 nonlinear ICA，iVAE 一致性，GIN 条件流，ISFA/xSFA 慢源恢复，2026 statistical/structural near-identifiability，以及条件流 latent recalibration。按根代理后续信息，另核查 TCL 2016 posterior→density-ratio 与 GCL 2019 exponential-family sufficient-statistic 结果，用于 exact regime-posterior 接口的近邻定位。采用论文作者全文或正式 proceedings；新预印本独立标记。检索是定向近邻核查，不能据此断言全文献不存在结果或本项目首创。

禁止并未执行：checkpoint/模型加载、模型前向、数组诊断或读取、TRAIN/CAL/TEST/FINAL 数组、新随机采样、新物理轨迹、训练、服务器连接、GPU 作业。以上执行计数均为 0。初始价值分析为纸笔形式；追加脚本只使用手工固定概率向量和对角矩阵核算不等式，不采样、不读取物理数组或模型。

主判据：不增加信息与改善受限可读性必须分开；同一个后处理规则与相同端到端函数类/预算必须分开；存在理想 inverse 与实际学习可达性必须分开；真值辅助机制诊断与部署时无真值认证必须分开。保留 raw 10/10 原门成功、weak/.25 四格均值负例、FHN N/A、G4≠G0、仅四个底层独立根及 strong/weak 分布混杂。
