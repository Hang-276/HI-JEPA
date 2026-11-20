# Reliable Slow Representations / JEPA

研究问题：**因素已经可以辨识或恢复以后，还需要什么条件，才能可靠地选出有限轨迹中的物理慢因素？**

当前主线是低维 LeJEPA 受控实验与官方 FourRoom 两层 H-JEPA 视觉扩展。低维 A／B 阶段已有完成的结果和停止边界；**FourRoom 正式结果已于 2026-11-16 回包归档，运行 COMPLETE，但 12 项均为 RECOVERY_NOT_VERIFIED（0 PASS）**。完整性核验、逐轴恢复与原标准判定见[回包报告](docs/FOURROOM_HJEPA_RETURN_20261116.md)。

- [理论与机制报告统一入口](docs/README.md)：两份报告、配套脚本、结果表、相图与核验收据。
- [严格数学形式化与证明](docs/THEORY_FORMALIZATION_20261116.md)：同窗目标、REF 分解、排序稳定性及统计保证的条件。
- [纯 NumPy 机制沙盒](docs/EXPLORATORY_MECHANISM_PROBES_20261116.md)：800 格扫描、128 个封存锚点对照与快轴退化压力测试；探索性合成证据。
- [进展同步：数值、证据等级、差异与交接](docs/PROGRESS_SYNC_JEPA.md)
- [完整研究记录及回包分析方案](docs/RESEARCH_RECORD.md)
- [研究 TODO](docs/ICML_RESEARCH_TODO.md)
- [文献与成果归属](docs/LITERATURE_GAPS_AND_POSITIONING_20261116.md)

## 结构

| 目录 | 用途 |
|---|---|
| `src/slowrepr/` | 当前图像实验、物理读出、评分与统计 |
| `src/hijepa/` | 早期 HI-JEPA 实现，保留兼容；不是官方 H-JEPA |
| `configs/fourroom_hjepa/` | 当前图像配置 |
| `configs/low_dim_ar1/` | 低维封存配置的只读链接 |
| `configs/legacy_hijepa/` | 历史配置，原路径保留链接 |
| `scripts/fourroom_hjepa/` | 图像输入准备与运行入口 |
| `scripts/low_dim_ar1/` | 在新目录准备低维复现，不覆盖原结果 |
| `scripts/low_dim_ar1/verify_theory_moments.py` | 确定性解析矩与封存汇总核对，不采样或运行模型 |
| `scripts/exploratory_20261116/` | 纯 NumPy 合成机制探针、图表导出与核验 |
| `scripts/check_repository.py` | 只读语法、路径、配置与来源检查 |
| `tests/` | 数值单测与历史测试，须按名称选择 |
| `docs/` | 研究记录、协议入口和本次审计 |
| `results/low_dim_ar1/` | 低维结果统一导航 |
| `results/theory_checks_20261116/` | 数学形式化的解析核对与完整性收据 |
| `results/exploratory_20261116/` | 合成沙盒数据、相图、锚点对照与复现收据 |
| `results/fourroom_hjepa/` | `fourroom_01/` 保留服务器原始结果；独立 analysis 目录保存回包核验 |
| `results/legacy_fine/` | 早期 Fine 理论与实验入口 |
| `reports/` | 原低维封存实目录，必须保留原位 |
| `archive/` | 历史输出、快照、原交付和备份 |

`reports/` 和 `work/hcf_stable_id_20261027/` 是封存兼容目录：回执还校验 `Path.resolve()` 后的来源路径键，改名可能破坏校验。`results/` 因而使用链接导航，原内容不改。其他旧根目录链接主要保障历史查阅；旧脚本若依赖固定祖先目录，重演前须恢复原布局。

## 环境与最小命令

以下训练与服务器命令是未来使用说明；目录整理没有安装依赖、运行模型或重算旧结果。其后完成的确定性理论核对与 NumPy 合成沙盒已单独归档，复现范围及命令见[报告统一入口](docs/README.md)。

只读结构检查使用 Python 3.11+ 标准库，不导入研究模块：

```bash
python scripts/check_repository.py
```

根 `requirements.txt` 只列 NumPy／SciPy 数值工具，**不是历史实验环境锁**。当前包可用 `pip install -e .` 安装；视觉 GPU 环境由独立安装脚本准备。旧 Python3.13 环境声明保存在 `archive/legacy_metadata_20261116/`。

低维 A4 的最小复现准备命令：

```bash
python scripts/low_dim_ar1/prepare_a4.py --destination /absolute/path/to/new-a4-reproduction
```

准备器只核对并复制13个封存输入，目标必须全新且位于仓库外；随后打印该目录内的训练命令。详见 [低维复现说明](docs/LOW_DIM_REPRODUCTION.md)。它只覆盖 A4 训练和状态审计准备，不含后续 TEST 评分；原种子重复不是独立验证。不得直接在原 `reports/` 上执行训练、forecast、score 或重分析脚本。

官方 FourRoom 分支使用 Python3.11、PyTorch2.7.1+cu128。正式服务器已将依赖修为 transformers 5.9.0 / huggingface-hub 1.5.0；本地安装脚本仍保留旧版本范围，不能原样作为已验证环境。以后另行允许启动计算时，须先按[已核实环境差异](docs/FOURROOM_HJEPA.md)准备独立环境并使用新运行名：

```bash
bash scripts/fourroom_hjepa/setup.sh
.venv-fourroom/bin/python scripts/fourroom_hjepa/run.py --name fourroom_02 --device cuda
```

正式原始结果保存在 `results/fourroom_hjepa/fourroom_01/`，原包保存在 `archive/server_results/20261116/`。本次实际运行以回包内 `scripts/` 为准，不用重构后的入口替代；旧交付仍保存在 `archive/releases/hjepa_slow_selection_20261115/`。正式运行已证明服务器流程完成，但 `stable_worldmodel` 的运行时模块路径/独立版本未单独封存。详见 [图像实验说明](docs/FOURROOM_HJEPA.md)。

仅固定数组数值单测，不读取实验结果或运行模型：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_fourroom_metrics.py' -v
```

需要 NumPy、SciPy。**不要无筛选执行全仓测试**：旧 `test_protocol.py` 和 `test_v3_spectral.py` 包含训练、模型前向或优化。静态检查通过不等于服务器流程或科学假设通过。

## 证据管理

TEST／AUDIT／REF／CAL预测与真值、回执、provenance、最优checkpoint、封存配置和服务器原包均保留。`.gitignore` 排除大文件与历史数据，忽略不等于删除；新克隆不会自动包含全部本地证据，缺少输入时须取得原结果包。

整理前的48,449项非Git文件／链接清单、原工作树改动、逐文件删除记录及前后哈希核查均在 [docs/audit/20261116](docs/audit/20261116)。旧根README、环境声明、提示词和校验清单已归档。根Git历史及嵌套Git元数据未清理；本轮没有提交或推送。

两份后续报告按现有 `docs/`—`scripts/`—`results/` 结构接入，保持报告、结果与原回执的字节和路径不变。导航整合记录另存于 [report_integration_20261116](docs/audit/report_integration_20261116/README.md)，不覆盖目录治理的历史审计。

采用 LeJEPA／H-JEPA 不代表其已有模型或理论是本项目原创。当前贡献与尚未排除的解释以进展同步和文献归属记录为准。
