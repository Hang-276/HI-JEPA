# FourRoom H-JEPA：标准目录入口与复现边界

本入口用于以后在服务器上运行官方冻结 H-JEPA 的物理因素读出与慢因素选取评价。它从2026-11-15交付版本迁移而来，**只调整目录、导入和代码封存清单，没有更改数据生成、模型、读出、评分或统计规则**。历史交付版本保存在 `archive/releases/hjepa_slow_selection_20261115/`；当前服务器若使用该旧版，仍应以其实际结果包为准，不能把新入口当作旧运行的源码。

本次仓库整理没有执行安装、下载、数据采集、模型前向、训练或评测。下列命令是未来复现说明，不代表已成功运行。

**2026-11-16 正式回包更新：** `fourroom_01` 已完整归档，状态 `COMPLETE`，但 12 项均未通过四轴恢复准入，均为 `RECOVERY_NOT_VERIFIED`。实际运行代码在[包内 scripts](../results/fourroom_hjepa/fourroom_01/scripts/run.py)，核验与逐项结果见[回包报告](FOURROOM_HJEPA_RETURN_20261116.md)。上段“本次仓库整理”是原目录迁移的历史范围，不代表现在仍无正式结果。

## 1. 文件与结果位置

| 内容 | 仓库路径 |
|---|---|
| 主实现、数据生成和读出训练 | `src/slowrepr/fourroom_hjepa.py` |
| 评分与统计函数 | `src/slowrepr/metrics.py` |
| 运行入口 | `scripts/fourroom_hjepa/run.py` |
| 依赖与官方输入准备 | `scripts/fourroom_hjepa/setup.sh` |
| 科学配置 | `configs/fourroom_hjepa/default.json` |
| 固定数组数值单测 | `tests/test_fourroom_metrics.py` |
| 独立运行环境 | `.venv-fourroom/` |
| 官方源码、权重及本次原始数据缓存 | `assets/fourroom_hjepa/` |
| 一次运行的完整结果 | `results/fourroom_hjepa/<运行名称>/` |

这条视觉分支使用Python3.11和自己的环境，**不要用归档中旧 HI-JEPA 的 Python3.13 环境声明或根数值工具依赖代替它**。新结果名称不得重复；运行脚本拒绝覆盖已存在的运行配置。新名称但相同配置会复用同一组生成种子，不代表独立重复实验。

## 2. 环境与已知缺口

需要Linux、NVIDIA GPU、Python3.11，以及能够访问官方源码和模型下载站的网络。原交付建议一张24GB GPU、至少16GB内存和30GB磁盘；本次回包确认使用 RTX 4090，run.log 从 16:29:09.242 到 21:44:24.829，约5小时15分钟，未包含此前安装时间，峰值资源未记录。

安装脚本保留原设置：PyTorch `2.7.1+cu128`、torchvision `0.22.1+cu128`；官方源码提交固定为 `97054ecaff83e13839195ec4b92d74c08a0127c5`，模型 revision 固定为 `7b8ba64389690345045650c51c6d6fea0ba4bd02`。其余依赖沿用原版本范围或未锁定声明，不宣称环境已完全锁定。实际版本由运行结果中的 `environment.txt` 和输入准备记录保存。

**服务器修复与本地脚本的差异已核实：** 回包 `environment.txt`、`provenance.json` 中 setup 记录及 `scripts/setup.sh` 一致确认 `transformers==5.9.0`、`huggingface-hub==1.5.0`；替代原来的 `transformers>=4.50,<5`、`huggingface-hub>=0.30,<1`。Python 为3.11.18、torch为2.7.1+cu128、torchvision为0.22.1+cu128；未改变正式科学配置、读出或验收阈值。本地 `scripts/fourroom_hjepa/setup.sh` 和历史交付原件仍保持原样，因此其旧依赖范围不能直接用于复现这次已完成的服务器环境。

官方固定源码提交自带 `stable_worldmodel/`，已核对源码树的37个路径条目，包含 FourRoom 环境和策略；它无需作为独立发行包出现在 pip freeze 中。正式流程已经跑通，但运行时实际导入的模块文件路径仍未封存，不能因此声称来源已完全锁定。用户说明权重经 HF-Mirror 下载且 `smoke_03` 完成；正式包未附镜像下载日志和该 smoke 的 status/log，二者按用户说明记录。三个权重的回包 SHA 与官方固定 revision 的 LFS 元数据已核对，详见回包报告。

在仓库根目录执行输入准备：

```bash
bash scripts/fourroom_hjepa/setup.sh
```

该命令会安装依赖并下载权重；本次回包核验没有执行。它保留原依赖范围及导入检查，未来使用前须处理上文已记录的版本兼容差异，不能把原样命令当作本次服务器环境锁。

## 3. 未来运行命令

仅固定数组数值测试，不读取实验结果、不运行模型：

```bash
.venv-fourroom/bin/python -m unittest discover -s tests -p 'test_fourroom_metrics.py' -v
```

**不要使用无筛选的全仓库测试命令。** 旧分支的一些测试实际包含训练、读出优化或模型前向。

服务器工程冒烟测试：

```bash
.venv-fourroom/bin/python scripts/fourroom_hjepa/run.py --name smoke_01 --smoke --device cuda
```

冒烟会执行真实模型前向和小规模读出拟合，只有在允许启动计算时才运行；`SMOKE_COMPLETE` 只证明流程跑通，不是科学验收。

正式实验：

```bash
mkdir -p results/fourroom_hjepa
mkdir results/fourroom_hjepa/fourroom_02 && nohup .venv-fourroom/bin/python scripts/fourroom_hjepa/run.py --name fourroom_02 --device cuda > results/fourroom_hjepa/fourroom_02/launch.log 2>&1 &
```

日志及状态均在同一个运行目录中：`launch.log`、`run.log`、`collection.log`、`status.json`。正式 `COMPLETE` 表示运行结束，不表示科学假设通过；判定看 `RESULTS.md` 与 `summary.json`。失败时保留完整目录，程序不会自动重跑。

结果打包：

```bash
tar -czf fourroom_02_results.tar.gz -C results/fourroom_hjepa fourroom_02
```

只需返回整个结果目录，不需收集其他位置的分析文件。原始图像、官方权重在 `assets/fourroom_hjepa/`，暂时保留，便于必要时补导出校准真值或复核输入；无需一并发回。

## 4. 保持不变的科学设置

使用三个官方模型种子42、43、44，冻结两层编码器。四个物理因素为角色 x、角色 y、干扰物 x、干扰物 y。每层使用同一时刻图像的逐帧表示，上层不跳帧。读出为 `d→128→128→4`，两层隐藏激活为SiLU，训练3,000步；只用TRAIN拟合，只用VAL选择检查点。

共3,776个episode，按完整episode划分：TRAIN 256、VAL 64、AUDIT 128、REF 256、CAL 1,024、TEST 2,048。TRAIN与VAL每段固定取32帧；其他用途每段一个193帧窗口。评价lag为4、16、32步，平方增量等权平均。真实评分除以共同独立REF上的物理方差，预测评分除以本读出在同组REF上的预测方差；不使用短窗口内部方差。

采样仍基于官方FourRoom环境，将expert policy的 `action_repeat_prob` 从官方默认0.05设为0，以避免原实现上一段动作状态对下一episode的影响。这是原交付已存在的采样分布变化，不是本次重构新增，也不宣称精确复现官方原评价分布。

两条规则保持不变：每层自己的CAL分差中位数 `own_cal`；第一层CAL阈值直接用于两层 `level1_cal`。在TEST预测前冻结读出、REF和阈值；全部TEST预测保存并写入回执后，才读取TEST真值。当前代码不包含联合线性解混baseline，也没有增加恢复质量匹配筛查或新校准算法。

正式12项比较均保留：3个模型×2层×2规则。每项同时满足以下条件才标记 `PASS`：四轴独立AUDIT R²都至少0.85；TEST实际作答比例至少35%；按12项Bonferroni分配置信预算的单侧Clopper–Pearson错选率上界不超过1%。不同模型与层共享episode；每项内部以独立episode窗口统计，不能把帧或六份预测当作更多独立样本。恢复未通过、作答不足、风险未确认需分别解释。

## 5. 新运行的封存材料

运行根目录保存有效配置、数据划分、输入来源和权重散列、环境版本、阈值、预测回执、共同物理REF与TEST真值。每个模型与层目录保存读出权重、AUDIT真值与预测、REF预测、CAL预测、完整TEST预测和评分。

新的 `scripts/` 快照保持仓库相对目录，至少包含实际执行的主模块、评分函数、包初始化、运行入口、安装脚本、默认配置与本文档；`script_receipt.json` 记录这七项的SHA256。实际运行配置另存为结果根目录的 `config.json`，其中包括冒烟设置。该新增封存清单仅改善代码可追溯性，不改变科学计算。

已回包的 `fourroom_01` 使用旧交付布局，实际有6份平铺脚本/文档快照，没有新入口才生成的 `script_receipt.json`；这不是该旧协议必需材料的缺失。本次从原压缩包逐文件验证并记录了全部76文件的 SHA；`forecast_receipt.json` 另锁定阈值及6份 TEST 预测。

CAL物理真值与encoder中间表征沿用原实现，未单独导出到结果包。不得把这一缺项解释成这些数据已用于某种新校准。后续分析与补跑条件见[研究记录](RESEARCH_RECORD.md)第11.5节。
