# 2026-11-16 审计与目录治理记录

本目录记录文件审计，不是新增实验结果。未运行训练、特征提取、模型评测或旧统计重算。

| 文件 | 含义 |
|---|---|
| `PROGRESS_BEFORE_REORGANIZATION.md` | 移动前已完成的进展审计初稿，保留当时路径与占位 |
| `FILES_BEFORE.jsonl` | 整理前48,449项非Git文件／链接的路径、大小、mtime与SHA256 |
| `GIT_STATUS_BEFORE.txt`、`WORKTREE_BEFORE.patch` | 整理前已有的未提交变化 |
| `SEALED_SOURCE_CHECKS.json` | 13份来源锁、232个唯一源／资产的整理前核查 |
| `SEALED_PREDICTION_CHECKS.json` | 144个预测文件的原回执hash与数组头核查 |
| `REORGANIZATION.json` | 实际移动、备份、链接、删除及空缓存目录清单 |
| `DELETED_FILES.tsv` | 每个实际删除文件的原位置、新位置、大小、SHA256与原因 |
| `PRESERVATION_CHECK.jsonl`、`PRESERVATION_SUMMARY.json` | 每项原文件保留位置及内容核验，或明确缓存删除依据 |
| `STATIC_CHECKS.json` | 语法、路径、JSON及封存来源的整理后只读检查 |
| `DOCUMENT_LINKS.json` | 当前导航文档的本地链接检查 |
| `HJEPA_REFACTOR_REVIEW.md` | 新图像入口与原交付的科学逻辑一致性及help修复 |
| `NEW_SOURCE_HASHES.json` | 本次新工作代码／配置的指纹，不替代旧回执 |

原始科学证据的实路径主要仍在 `reports/`；大量历史产物仅移动至 `archive/`。Git元数据不在全文件hash清单内，原位或随完整目录保留，未执行Git对象清理。固定数组单测依赖SciPy，本机缺失，因此没有运行；也没有运行含模型任务的历史测试。
