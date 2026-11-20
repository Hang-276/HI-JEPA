evidence_tier: repository_integration_only

# 两份报告接入清理后仓库

日期：2026-11-16。目标仓库为 `/Users/yuzihang/Documents/ChatGPT/HI-JEPA`，即已完成目录治理的项目根目录。两份报告及配套文件原已在该仓库中；本次补齐统一入口、结果导航与研究进展，不另复制一个分叉仓库。

接入原件：

- [数学形式化报告](../../THEORY_FORMALIZATION_20261116.md)，连同 `scripts/low_dim_ar1/verify_theory_moments.py` 与 `results/theory_checks_20261116/`。
- [探索性机制沙盒报告](../../EXPLORATORY_MECHANISM_PROBES_20261116.md)，连同 `scripts/exploratory_20261116/` 与 `results/exploratory_20261116/`。

新增 [docs/README.md](../../README.md) 统一入口；更新根 README、两级结果 README、进展同步、研究记录和 TODO。进展记录采用追加，研究记录第563–579行保持不变。两份报告、脚本、原数据、图表、原 DELIVERY 及旧治理审计不改写。

整合前六份导航文档原样备份在 `before/`，索引见 [BEFORE.json](BEFORE.json)。检查结果与整合后的文档指纹见 [INTEGRATION.json](INTEGRATION.json)，包含：

1. 理论 10 项、沙盒 32 项交付条目的哈希比对。
2. 对前轮受保护的 47,786 个文件按完整清单再核验，无变化、缺失或受保护目录新增。
3. 更新文档的本地链接、两报告入口可达性及研究记录原行号内容检查。
4. Python 3.11+ 运行 `python3 scripts/check_repository.py` 的只读结果。

此轮只整合文档和导航，未启动训练、合成采样、旧评测重算或服务器任务；未安装依赖，未提交或推送 Git。原工作区已有的治理改动保留，不能把当前整合称为干净工作区或远端发布。
