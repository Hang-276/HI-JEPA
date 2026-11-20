# HI-JEPA / Reliable Slow Representations 工作区协作与科研指引

本文件位于项目根目录，由 Antigravity 在打开本工作区时**自动加载**。无论谁以何种 Antigravity 账户打开本目录（含整文件夹打包拷贝），都必须严格遵守以下主线、红线、进度事实与操作规范。

---

## 1. 核心研究主线与论文定位

- **核心问题**：**When Does Factor Identifiability Yield Reliable Slow Representations?**
  即：**当物理因素已经可以辨识或恢复以后，还需要什么条件，才能在有限轨迹窗口内可靠地选出物理慢因素？**
- **核心两阶段框架与误差机制**：
  1. **门一（`AUDIT` 全因素恢复，`min R² >= 0.85`）**：阻断快因素退化导致的两种失效分支（Proposition 4）——**Regime II-A**（高频残差占优、分数上移、局部速度翻转窗系统性漏检）与 **Regime II-B**（快轴方差坍缩、微弱慢谐波泄漏被 `REF` 归一化放大为主导份额、常规窗大面积劫持）。门一不可被纯下游有标签校准（LTT / SGR）替代（Proposition 5）。
  2. **门二（无标签 `own_cal` 选择性分差门控）**：在过门一前提下，结合匹配 `REF` 分母消偏、残差时间自相关与有限 `CAL` 抽样/谱间隔比边界，以 **0 条带标签校准轨迹**提供约 50% 覆盖与族校正 `<1%` 的选择性风险认证。
- **文献归属红线**：采用 LeJEPA、H-JEPA、LeWM、SFA、Clopper–Pearson、Learn-Then-Test (LTT) 等既有模型/工具，不代表其为本项目原创；原创贡献与既有理论的边界以 `docs/LITERATURE_GAPS_AND_POSITIONING_20261116.md` 和 `docs/THEORY_AND_LITERATURE_SUPPLEMENT_20261117.md` 为准。

---

## 2. 科研与工程核心行为准则（全员强制）

1. **锚定主线与结构纪律**
   - 一切实验设计、架构调整与分析必须对齐上述主线，严禁提出偏离主线的发散改动；未经负责人明确许可，禁止擅自增删模块、改动核心评测口径或引入无关依赖。
2. **批判性思维与绝对防作弊红线**
   - 对已有假设、方案及自身输出保持高度批判性，主动指出逻辑漏洞、公平性偏差与风险；不迎合、不夸大结论。
   - **严禁**：修改测试脚本/评测定义制造改善、硬编码结果、屏蔽报错、把 `NaN`/`Inf` 静默替为 `0`、事后放宽预设门槛（`audit_min_r2=0.85`、`minimum_coverage`、`maximum_selective_error`）、或在偷看 `TEST` 真值后调参/改阈值。
   - 严格区分证据等级（`formal`、`deterministic_theory_check`、`exploratory_numerical_sandbox`、`preflight_audit`）：运行 `COMPLETE` 不等于科学门槛通过，探索性沙盒与诊断不等于独立确认性实验。
3. **事实核验与先对齐再动手**
   - 涉及文献、SOTA、Baseline 或未知概念时，必须先联网检索核实一手原文，严禁凭记忆编造；查无实据直接说明“不知道”。
   - 写代码或推进关键科研节点前，若需求存在歧义或缺少上下文，**必须先向用户提问确认**；复杂长程任务遵循“完整计划 -> 确认 -> 分步实现 -> 全面核验”。

---

## 3. 封存资产与不可变红线（最高优先级）

以下路径与文件为**只读封存资产**，绝对禁止修改、覆写、重命名、删除或就地重跑：

1. **物理目录原位锁定**：`reports/` 与 `work/hcf_stable_id_20261027/` 必须保持物理实目录（不得改为符号链接或移动），因为 `docs/audit/20261116/SEALED_SOURCE_CHECKS.json` 校验 `Path.resolve()` 路径键及 **232 项封存源 SHA-256**。
2. **48,889 份受保护资产基线**：`results/closeout_20261117/protected_before.json` 对以下范围做了全量 SHA-256 锁定——`reports/`、`work/hcf_stable_id_20261027/`、`archive/`、`src/`、`configs/`、`tests/`、`results/theory_checks_20261116/`、`results/exploratory_20261116/`、`results/fourroom_hjepa/`、`results/visual_sprites_lejepa/`、`results/low_dim_ar1/`、`results/legacy_fine/`，以及 `2026-11-17` 之前既有的 `docs/` 与 `scripts/` 文件。
3. **逐字节一致校验**：`src/slowrepr/metrics.py` 与 `configs/fourroom_hjepa/default.json` 必须与 `archive/releases/hjepa_slow_selection_20261115/` 中对应原件逐字节一致（`scripts/check_repository.py` 强校验）。
4. **新增产物的正确做法**：
   - 任何新复跑、新分析或新报告必须写入**全新的带日期输出目录/新文件**，严禁覆盖既有结果或原回执。
   - 运行 Python 脚本时优先加 `-B` 或设置 `PYTHONDONTWRITEBYTECODE=1`，避免在受保护目录下生成 `__pycache__` 触发完整性基线告警。
   - `src/hijepa/` 为早期历史兼容实现，**不是官方 H-JEPA**；当前视觉/慢表征代码在 `src/slowrepr/`、`scripts/visual_sprites_lejepa/`、`scripts/fourroom_hjepa/` 与 Wave-1 回包目录。

---

## 4. 当前最新研究状态快照（截至 2026-11-18）

> **重要防误导提示**：`README.md`、`docs/PROGRESS_SYNC_JEPA.md`、`docs/ICML_RESEARCH_TODO.md` 停留在 `2026-11-16`；`docs/ANTIGRAVITY_HANDOFF_READY_20261116.md` 停留在 `2026-11-17 14:29`（仍称 32×32 正式实验计算中）。**请务必以本节时间线及最新结果文档为准，切勿用旧文档误判当前进度。**

| 阶段 / 日期 | 状态 | 核心已证实事实与权威入口 |
|---|---|---|
| **低维 AR(1) LeJEPA A1–A4 / B1–B3** (`2026-11-12`–`16`) | 已封存完成 | A3 `g50` `3/33,355` 错、`16/16` PASS；A2 `strong/weak` `16/16` vs `11/16`；A4 高恢复全答仍错 `1,430/32,768` (`4.36%`)、20% 筛选 `0/6,544`；B1 耦合 `16/16` PASS；B2 窄间隔 `lag1 79.01% > lag16 70.81% > multi 66.02%`；B3 变慢后原阈值覆盖跌至 `31.94%`，重设阈值 `49.88%`、`14/16` PASS。入口：`docs/LOW_DIM_REPRODUCTION.md`、`docs/PROGRESS_SYNC_JEPA.md`、`docs/ICML_RESEARCH_TODO.md`。 |
| **理论形式化与纯 NumPy 机制沙盒** (`2026-11-16`) | 已封存完成 | 同窗目标、有限 `REF` 分解、排序稳定性、滞后谱扰动与 800 格/128 真实锚点沙盒（后者属 `exploratory_numerical_sandbox`）。入口：`docs/README.md`、`docs/THEORY_FORMALIZATION_20261116.md`、`docs/EXPLORATORY_MECHANISM_PROBES_20261116.md`。 |
| **官方 FourRoom 首轮回包 `fourroom_01`** (`2026-11-16`) | 已封存完成 | 6 层×2 规则共 12 项全部 `RECOVERY_NOT_VERIFIED`（0 PASS）；按 `own_cal` 原接受集拆分，常规窗 `2/5,550` 错，翻转窗 `481/482` 错。入口：`docs/FOURROOM_HJEPA_RETURN_20261116.md`。 |
| **32×32 双实体视觉 `visual_sprites_lejepa/formal_v1`** (`2026-11-17 16:40`) | **已完成并核验** | 6 组正式盲测、32 份预测文件 SHA-256 全匹配（2,048 窗口含 159 翻转窗 `7.76%`）。`visual_lejepa` **3/3 PASS**（min AUDIT R² `0.9815–0.9833`，覆盖 `53.56%–54.35%`，错选 `1/1097–1113`，CP 上界 `0.61%–0.62%`）；`long_horizon_pred` **3/3 `RECOVERY_NOT_VERIFIED`**（seed101/103 属 Regime II-A，seed102 属 Regime II-B 相对慢泄漏 `65.52%`、常规窗全答错 `71.89%`）。入口：`results/visual_sprites_lejepa/formal_v1/RESULTS.md`、`SUMMARY.json`。 |
| **本机收口 `L1–L5` 与理论文献增补** (`2026-11-17 23:53`) | **已完成并核验** | L1 实证验证 Proposition 4 正交方差分解；L2 实测真实编码器残差 lag-1 自相关 `0.51–0.71` 及 B2 抽样/间隔比 $a$ 单调恶化（`1.24→6.15`）；L3 解耦有限 `REF` 与窗口长度 $T$（`unit` 分母拉长 $T$ 反恶化，`shared` 分母一阶相消）；L4 补齐 Proposition 4/5 与原文核读；L5 实证门一失效时有标签 LTT（1,024 条标签轨迹）覆盖全 `0.0%`，中低预算下无标签 `own_cal` 占优。入口：`docs/LOCAL_CLOSEOUT_SYNTHESIS_20261117.md`、`docs/THEORY_AND_LITERATURE_SUPPLEMENT_20261117.md`。 |
| **FourRoom Wave-1 正式回包 `fourroom_hjepa_wave1_20261118`** (`2026-11-18 16:12`) | **已回包（待整合进 `docs/`）** | 证实原 FourRoom 失效源于 **Type-II `[CLS]` 汇聚瓶颈**而非 L1 编码器丢弃 distractor（L1 `grid4x4` 空间网格读出在 LeWM/L2/L4 三族全通过，`ridge` R² `0.991–0.997`）。Family A（24 项）：**6/24 PASS**（全来自 L1 `grid4x4` 160k MLP/ridge，0 错选）；Family B（51 项，LeWM/L2/L4）：**23/51 PASS**（LeWM L1 `9/9` PASS；L2 L1 仅 grid `6/6` PASS；L4 L1 grid `6/6` + CLS `2/3` PASS，seed43 `0.834` 未过门；深层 CLS L2–L4 随层深单调下降且全 `RECOVERY_NOT_VERIFIED`）。L1 网格读出通常不接受翻转窗（`accepted=0`，全答准确率 `0.75–0.94`）。入口：`results/fourroom_hjepa_wave1_20261118/extracted/results/fourroom_depth_v2_combined/FINAL_SUMMARY.md`、`preflight_audit/STAGE1_METHODOLOGICAL_REPORT.md`。 |

### 当前已知未收尾 / 待整合项
- `2026-11-18` 的 `fourroom_hjepa_wave1_20261118` 正式结果尚未写入 `docs/` 主同步文档（如新整合报告或同步更新，注意不得改动 `protected_before.json` 已锁定的历史旧文件）。
- 根目录 Git 工作树自早期提交 `6eb6de6` 以来所有新进展均未提交；**未经用户明确指令，禁止擅自 `git commit`、`git push` 或清理历史 Git 元数据**。

---

## 5. 新接手推荐阅读顺序（按时效与信息密度）

1. `results/fourroom_hjepa_wave1_20261118/extracted/results/fourroom_depth_v2_combined/FINAL_SUMMARY.md`（最新 FourRoom Wave-1 正式总表与结论）
2. `results/fourroom_hjepa_wave1_20261118/extracted/results/preflight_audit/STAGE1_METHODOLOGICAL_REPORT.md`（官方探针源码审计与 `[CLS]` vs `grid4x4` 发现）
3. `docs/LOCAL_CLOSEOUT_SYNTHESIS_20261117.md`（L1–L5 本机机制收口与 32×32 正式结果拆解）
4. `docs/THEORY_AND_LITERATURE_SUPPLEMENT_20261117.md`（Proposition 4/5 与最新原文文献核读）
5. `docs/README.md` 与 `docs/THEORY_FORMALIZATION_20261116.md`（核心数学形式化与历史报告统一入口）
6. 按需查阅：`docs/LITERATURE_GAPS_AND_POSITIONING_20261116.md`、`docs/RESEARCH_RECORD.md`、`docs/ICML_RESEARCH_TODO.md`、`README.md`

---

## 6. 整目录拷贝协作、环境与安全校验命令

- **打包拷贝注意事项**：
  - 拷贝/压缩整个工作区时必须保留符号链接（如 `tar`、`cp -a`、`rsync -a`），且不要破坏 `reports/` 与 `work/hcf_stable_id_20261027/` 相对路径。
  - 目录内 `.venv_visual_cpu/bin/python` 是指向原机器绝对路径（`/opt/miniconda3/envs/evg-agent/bin/python`）的符号链接，历史文档中的 `/opt/miniconda3/bin/python` 与服务器路径 `/home/kemove/yuzihang/hjepa-slow-selection` 亦绑定原机；**拷贝到新电脑后需新建本地 Python 3.11+ 环境**并安装 `requirements.txt`（基础数值核验仅需 `numpy` / `scipy`；GPU 实验按 `scripts/fourroom_hjepa/setup.sh` 及 `docs/FOURROOM_HJEPA.md` 独立配置）。
- **安全校验命令（请使用 Python 3.11+，因 `check_repository.py` 依赖 `hashlib.file_digest`）**：
  1. 只读仓库语法、软链与 232 项封存源核验（纯标准库，不导入研究模块）：
     ```bash
     python3.11 -B scripts/check_repository.py
     ```
  2. 仅跑固定数组数值单测（需 `numpy`、`scipy`）：
     ```bash
     PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3.11 -B -m unittest discover -s tests -p 'test_fourroom_metrics.py' -v
     ```
     **严禁无筛选执行全仓 `tests/`**：旧 `tests/test_protocol.py` 和 `tests/test_v3_spectral.py` 包含训练/前向/优化流程。
  3. 仅在明确需要复核 48,889 份受保护文件零改动时执行全量 SHA-256 校验（约 24 秒；注意该脚本会更新 `results/closeout_20261117/INTEGRITY_CLOSEOUT_20261117.json` 的核验时间戳）：
     ```bash
     python3.11 -B scripts/closeout_20261117/snapshot_and_verify.py verify --baseline results/closeout_20261117/protected_before.json
     ```
