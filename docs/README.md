evidence_tier: repository_integration_only

# 理论与机制报告统一入口

论文问题：**When Does Factor Identifiability Yield Reliable Slow Representations?**

本页把两份已完成报告及配套产物接入清理后的 HI-JEPA 仓库。报告原件、数据、脚本和原交付回执保留原路径与内容。阅读顺序是先核对数学对象和条件，再看合成实验是否支持候选机制；实际运行封存的配置、代码、回执仍优先于解释性文档。

## 两份报告与证据等级

| 报告 | 解决的问题 | 配套证据与范围 |
|---|---|---|
| [严格数学形式化与证明](THEORY_FORMALIZATION_20261116.md) | 同窗目标、有限 REF 分解、高 R² 反例、排序稳定性、滞后谱扰动和条件选择性保证 | [解析核对](../results/theory_checks_20261116/RESULTS.md)、[完整数值](../results/theory_checks_20261116/SUMMARY.json)、[交付收据](../results/theory_checks_20261116/DELIVERY.json)。确定性计算与封存汇总只读对齐，不是新增模型实验。 |
| [探索性机制沙盒](EXPLORATORY_MECHANISM_PROBES_20261116.md) | 谐波与实际字典差异、残差谱偏置、800 格有限窗口可靠性、128 个真实锚点和快轴退化 | `exploratory_numerical_sandbox`。[探针一数据](../results/exploratory_20261116/probe1_summary.json)、[探针二数据](../results/exploratory_20261116/probe2_summary.json)、[交付清单](../results/exploratory_20261116/DELIVERY.json)。人工高斯生成与只读锚点投射，不是训练出的 LeJEPA/H-JEPA 新结果。 |

两份报告支持的主线是：因素恢复之后，慢子空间还需满足残差谱偏置与有限 CAL 误差相对真实谱间隔的条件；物理慢因素选取还需满足归一化评分误差相对同窗口分差的条件。总体边界 SNR 不等于同窗错率，增加 T 不能普遍消除固定 REF 偏差；实际 Poly2 不应直接套用理想 Hermite 临界值，`.85` 是操作性准入门。具体假设、反例和负结果以报告正文为准。

## 表格与图形

- [128 个封存锚点及 240 条选择规则对照](../results/exploratory_20261116/ANCHOR_COMPARISON.md)；[80 组完整窗口扫描](../results/exploratory_20261116/TMIN_FULL.md)。
- [谐波对照](../results/exploratory_20261116/figures/harmonic_boundary.svg)、[慢子空间相图](../results/exploratory_20261116/figures/subspace_phase.svg)、[全答风险相图](../results/exploratory_20261116/figures/phase_all_risk.svg)、[选择性风险相图](../results/exploratory_20261116/figures/phase_own_cal_risk.svg)、[快轴退化](../results/exploratory_20261116/figures/stress.svg)。
- 平表：[800 格 phase](../results/exploratory_20261116/phase_cells.csv)、[128 个 anchor](../results/exploratory_20261116/anchor_cells.csv)、[72 个 stress 单元](../results/exploratory_20261116/stress_cells.csv)；后者含 56 个请求单元和单独标记的 16 个极限控制。

## 脚本与复现范围

Python 3.11+ 与 NumPy；不需要 SciPy/PyTorch。实际数值运行环境及种子在收据中，本机使用 `/opt/miniconda3/bin/python`（Python 3.13.11）。先读帮助：

```bash
python scripts/low_dim_ar1/verify_theory_moments.py --help
python scripts/exploratory_20261116/probe1_harmonic_and_subspace.py --help
python scripts/exploratory_20261116/probe2_phase_diagram_alignment.py --help
```

需要复跑时，选尚未存在的新输出目录；下列命令会实际执行解析计算或合成采样，不加载模型。此轮导航整合没有再次执行它们：

```bash
python scripts/low_dim_ar1/verify_theory_moments.py --output-dir results/theory_checks_replay_01
python scripts/exploratory_20261116/probe1_harmonic_and_subspace.py --output-dir results/exploratory_20261116/replay_01
python scripts/exploratory_20261116/probe2_phase_diagram_alignment.py --output-dir results/exploratory_20261116/replay_01
```

三个入口：[理论矩核对脚本](../scripts/low_dim_ar1/verify_theory_moments.py)、[探针一](../scripts/exploratory_20261116/probe1_harmonic_and_subspace.py)、[探针二](../scripts/exploratory_20261116/probe2_phase_diagram_alignment.py)。沙盒图表与报告生成器按默认路径读取本次原收据，复跑子目录不会自动替换原交付。

原报告的可复现性检查见[理论验证](../results/theory_checks_20261116/VALIDATION.json)和[沙盒验证](../results/exploratory_20261116/validation.json)。两个 NumPy 主脚本已独立复跑，数值完全一致，含启动及写盘的墙钟约 2.50 秒／5.99 秒。复现同一种子不增加科学独立性。

`reports/`、活跃锁定 `work/` 和归档来源路径必须保持不变；新克隆若缺少原输入应取得原证据包，不能重写期望哈希。原报告的完整性收据及行号引用对应其形成时的版本；本次只追加研究进展，不改报告字节或原回执。

## 项目进展与整合记录

- [FourRoom 正式回包与结果核验](FOURROOM_HJEPA_RETURN_20261116.md)：2026-11-16 运行 COMPLETE；12 项均 `RECOVERY_NOT_VERIFIED`，不能把完成运行当作科学门槛通过。原包、原始结果、环境修复与缺项均已独立记录。
- [研究进展同步](PROGRESS_SYNC_JEPA.md)：第6节记录两份报告整合，第7节记录正式视觉回包。
- [完整研究记录](RESEARCH_RECORD.md)：第13节连接证明与沙盒，第14节记录回包结果及停止归因决定。
- [研究 TODO](ICML_RESEARCH_TODO.md)：第七节区分已完成报告和仍未完成的独立科学验证。
- [本次导航整合与核验](audit/report_integration_20261116/README.md)；[原目录清理审计](audit/20261116/README.md)保留历史状态。

两份报告的导航整合没有重算其旧指标；后续FourRoom接入核验了已返回数组并另存诊断，见[回包审计](audit/fourroom_return_20261116/README.md)。没有启动新训练、轨迹采样或依赖安装，视觉结论以正式回包证据为准。
