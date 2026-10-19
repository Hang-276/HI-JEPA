# HI-JEPA 代码与实验配置

本项目实现 proposal 的三组低维机制实验：目标模态恢复与扰动稳定性、联合约束与伪解、部分模态恢复与分层价值。代码在 Mac CPU 完成集成验证；正式主结果、对照和多种子实验统一留在 Linux＋CUDA 执行。

本机集成结果仅说明代码路径可以执行，不构成模态恢复成功、分层有收益或完整学习定理的验证。CUDA 可迁移接口已经实现，实际 CUDA 执行与跨设备数值对齐须在目标机器完成。

## 安装和入口

```bash
git clone https://github.com/Hang-276/HI-JEPA.git
cd HI-JEPA
```

目标 Python 版本为 3.13。开发环境实测为 Python 3.13.15、PyTorch 2.14.0、NumPy 2.5.3。运行时仅依赖 NumPy 与 PyTorch，JSON 配置由标准库读取，测试使用 unittest。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py doctor --device cpu --output doctor_cpu.json
```

也可安装为命令行包：`python -m pip install -e .`，随后使用 `hijepa` 替代 `python run.py`。`run.py` 直接支持完整源码目录，代码没有硬编码开发机路径。

## 各实验启动脚本

各实验脚本都从同一份 canonical matrix 选择任务，自动补齐基准验收和结果复用的依赖，不改方法、训练预算或验收阈值。实验一、二、三和 `run_all.sh` 默认使用 **formal** profile；在 Mac 上需要显式传 `local`。

| 脚本 | 用途 | 调用示例 |
| --- | --- | --- |
| `scripts/run_experiment1.sh` | 实验一：谱间隔、观测扰动、轴／簇恢复与稳定性 | `bash scripts/run_experiment1.sh formal runs/e1` |
| `scripts/run_experiment2.sh` | 实验二：候选伪解审计、普通分层／块结构／完整模型对照 | `bash scripts/run_experiment2.sh formal runs/e2` |
| `scripts/run_experiment3.sh` | 实验三：direct／no_fine／full、细节保留与冻结 G/K 迁移 | `bash scripts/run_experiment3.sh formal runs/e3` |
| `scripts/run_all.sh` | 三组实验的完整矩阵 | `bash scripts/run_all.sh formal runs/formal` |
| `scripts/run_formal.sh` | 正式全量入口，固定 Linux CUDA | `bash scripts/run_formal.sh runs/formal` |
| `scripts/run_local.sh` | 本机完整集成入口，固定 CPU | `bash scripts/run_local.sh runs/local` |

单组实验的通用语法为 `bash scripts/run_experimentN.sh [formal|local] [OUTPUT] [--manifest-only]`，其中 N 为 1、2 或 3。`run_all.sh` 使用相同参数格式；`run_formal.sh` 和 `run_local.sh` 已固定 profile，只接受 `[OUTPUT] [--manifest-only]`。省略 OUTPUT 时使用项目 `runs/` 下的对应目录。脚本可从任意工作目录调用；显式相对 OUTPUT 按调用者的当前目录解析。

先查看任务清单、不训练：

```bash
bash scripts/run_experiment1.sh formal runs/e1 --manifest-only
bash scripts/run_experiment2.sh formal runs/e2 --manifest-only
bash scripts/run_experiment3.sh formal runs/e3 --manifest-only
```

清单位于 OUTPUT 下的 `planned_manifest.json`，`launch/selection.json` 说明包含的依赖、复用关系和验收阶段。正式 e1/e2/e3/all 分别包含 39/12/12/57 条计划记录；e2/e3 包含必要的 e1_baseline 依赖。不可认证条件仍保留，不会被过滤成“成功任务”。本机 e1/e2/e3/all 分别为 12/8/6/24 条记录。

实际 formal 运行依次完成 CUDA doctor、固定 checkpoint 的 CPU→CUDA 对齐、既有基准 seed 0/1 验收，再接续所选矩阵；已完成任务通过 `--resume` 复用。e2 和 all 还会保存 `pseudo_solution_audit.json`。任一前置检查失败即停止。`--manifest-only` 不调用设备检查、审计或训练。

整套实验推荐使用一次 `run_formal.sh`，由全局矩阵共享基准结果，避免分开启动三个独立输出目录时重复训练依赖。不同入口不要复用同一输出目录的不同矩阵；源码或协议改变时保留旧目录并使用新目录。

可用 `HIJEPA_PYTHON=/path/to/python` 指定解释器，用 `HIJEPA_PARITY_BUNDLE=/path/to/bundle.pt` 指定跨设备对齐包。默认对齐包随仓库提供，不需要先在服务器重做一轮本机实验。

## 本机集成

```bash
python -m unittest discover -s tests -v
bash scripts/run_local.sh runs/local
```

这是一套固定集成矩阵，覆盖 5 种模型、实模态、振荡、近重簇、观测扰动、不可认证的局部访问、两个种子及 full/no_fine 的完整迁移路径。完全相同的 e2_full 源训练复用 e1_real 产物，明确标记为复用，不计作独立重复实验。

矩阵会保留预设的 `unresolved` 条件。它们表示当前访问／误差预算不足以认证目标，不是应该调到通过为止的工程错误。实现故障记为 `failed` 并返回非零退出码。

## Linux CUDA 正式运行

在目标 Linux 机器安装相同版本的 NumPy 与 **带 CUDA 支持的 PyTorch 2.14.0**，并使用相容 NVIDIA 驱动。不同发行渠道的 PyTorch wheel 可能不同，先执行 doctor 验证实际 CUDA 构建；不能仅凭包版本推断 GPU 可用。

也可使用附带 Dockerfile：在 Linux NVIDIA 宿主机构建镜像，以 `docker run --gpus all` 运行并挂载结果目录；构建参数 `TORCH_INDEX_URL` 可指定官方 PyTorch CUDA wheel 源。Docker 构建及 GPU 容器运行尚未在本机验证，宿主驱动与 NVIDIA Container Toolkit 需在目标机器配置。

```bash
python run.py doctor --device cuda --output doctor_cuda.json
python -m pip freeze > environment-linux.lock.txt
python run.py manifest --config configs/matrix_formal.json --output formal_manifest.json
python run.py matrix --config configs/matrix_formal.json --output runs/formal --resume
```

推荐直接执行上面的 `bash scripts/run_formal.sh runs/formal`，自动完成所有前置检查与双种子验收。此处逐条 Python 命令保留供手动检查使用。容器运行时需把本机对齐包挂载进去。

正式 profile 强制 Linux＋CUDA；CUDA 不可用时明确失败，不回退 CPU。所有模型、对照和种子沿用同一环境。主配置使用 FP32，无 AMP／TF32；谱选择与证书数值部分始终在 CPU float64 上生成。随机投影由 CPU RNG 生成后移至训练设备。

正式矩阵先对 e1_baseline 的 seed 0、1 记录仅基于 train 的防坍塌验收；失败时保留结果并停止余下正式矩阵。有效秩、coarse floor 裕量和动态信息的工程阈值在 `anticollapse` 中明示，属于实施选择，不是 proposal 的原文数值。本机短程集成只记录此诊断，不据此宣布正式验收通过，也不反复调参使其通过。

`base_formal.json` 明确列出正式执行预算，`matrix_formal.json` 列出所有因素和对照。正式预算是本次实施选定的配置，并非 proposal 原文规定的数值。

## 跨设备对齐

从本机一个有效 checkpoint 导出固定输入、权重和投影 RNG：

```bash
python run.py parity-export --checkpoint runs/local/e1_real_seed0/checkpoint.pt --output parity_bundle
```

将源码目录和 `parity_bundle/` 一起迁移到 Linux：

```bash
python run.py parity-check --bundle parity_bundle/bundle.pt --device cuda --output parity_cuda.json
```

检查同一 checkpoint 与固定 batch 的高层输出、K、每项损失和参数梯度。默认 rtol=1e-4、atol=1e-6 是工程比较阈值，可通过命令行显式调整并记录；不能将其当作科研恢复阈值。CPU 自对齐通过不代表 CUDA 对齐已经通过。完整训练轨迹与最终权重不要求逐位一致。

## 实现与 proposal 的对应

| 模块 | 职责 |
| --- | --- |
| `systems.py` / `data.py` | 光滑收缩系统、可逆非线性观测、独立轨迹划分、NPZ 与数据 hash |
| `spectral.py` | 声明的局部查询、完整 Jacobian 误差预算、轮廓与重数、目标维数与谱域 |
| `models.py` | E/G/F、实块与二维旋转缩放 K、5 种对照、固定参数与光滑性预算 |
| `losses.py` | 两层预测、状态复合 rank、逐模态／逐块定标、0.1 SIGReg 与固定 covariance floor |
| `evaluation.py` | calibration 冻结匹配、轴与簇评价、共同原始空间预测、细节探针、迁移 monitor |
| `diagnostics.py` | u³／重复编码／共振审计、局部导数残差、动态信息与跨种子统计 |
| `runner.py` | 冻结协议、数据缓存、断点恢复、源训练、冻结 G/K 的迁移与矩阵汇总 |
| `parity.py` | 一次固定输入的跨设备前向／损失／梯度对齐 |

`full` 使用所有约束；`block_only` 去掉 rank；`hierarchical` 用一般非线性高层预测器且无 rank，保留相同目标选择、定标和防坍塌配置。`direct` 是相同 E→G 深度及参数预算的直接 m 维输出网络，保留内部 z 的共同 SIGReg，去掉细粒度预测支路。`no_fine` 保留与 full 相同 E/G/F 参数，只在源预训练去掉细粒度预测损失。

迁移阶段 full/no_fine 使用相同任务损失，冻结 G、K，只更新 E、F。源→目标仅改变细节机制，粗尺度规律与观测接口保持一致。共同达标阈值使用 step-zero 冻结的匹配与 decoder，在目标 calibration 的独立 monitor 轨迹上检查；test 不决定首达标时间。预算内未达标报告 `not_reached_within_budget`，不计作同精度成本。

## 数据和证据边界

模态选择采用预先声明、所有对照共享的局部模拟器映射查询。当前实现不是“只凭被动轨迹就完成目标认证”。selector 不读取真模态或真实谱标签；真值仅用于独立兼容性审计和评价。状态观测映射的可微访问用于 rank，所有对照享有相同访问权限。

默认系统的局部状态 Jacobian 为正规实／旋转块矩阵。非线性状态变换在平衡点导数为 I，只作用于细节维；显式 `nonnormal_extension` 被标记为主理论之外，主运行器不将其接纳为默认理论设定。

局部谱证书在声明的 C3 与 IEEE／tanh 误差模型下给出 J(0) 误差及谱轮廓重数；没有使用 directed-rounding 区间算术。它不等于全域谱证书或学习恢复证书。当前参数和固定参数预算的解析光滑性界保守且可能很松；完整有限样本恢复证书、完整比较解 joint cost 和优化误差界没有由训练自动获得。

真解审计分别报告 rank、scale、floor、谱域和逆收缩的兼容性；有限 batch 的正则代价与总体矩的代价可能不同。共振混合审计明确区分“谱域／floor／rank 接受”与严格逆收缩条件；canonical 共振例位于严格条件之外。

## 产物与恢复

每个训练任务保存 config、environment、data_reference、selection、compatibility、initial checkpoint、带 RNG／optimizer 的恢复 checkpoint、训练日志、评价和 seed 输出。源训练与迁移分别保存。

GitHub 仓库包含源码、配置、脚本、验证汇总和必需的 parity bundle。完整本机轨迹、优化器 checkpoint、缓存及新生成的 `runs/` 不入 Git，继续保留在本地；克隆后可直接启动正式流程。

`SHA256SUMS` 校验入库文件；本地轨迹缓存由各自的数据 manifest 单独校验。

数据缓存身份包含生成代码、配置与 NumPy 版本，读取时校验 NPZ hash。源码或协议改变时禁止混用已有矩阵和结果；选择新的输出目录保留旧记录。相同源码与协议的 `--resume` 可跳过已完成任务、接续中断任务。不同设备恢复仍不保证相同后续训练轨迹。

汇总位于 `summary.json` 与 `SUMMARY.md`，包括每种配置的种子均值／标准差和 calibration 冻结的跨种子表示一致性。缺测为 null／N/A，绝不填零。

## 来源

方法依据：[HI-JEPA Proposal](https://bcnpo29t80vs.feishu.cn/wiki/WPJCwzObTiURfbkSzjFcMt5Unzc)，本次核对的页面修改日期为 2026-10-18。

SIGReg 按官方 [LeJEPA Epps–Pulley](https://github.com/galilai-group/lejepa/blob/main/lejepa/univariate/epps_pulley.py) 与 [随机投影切片](https://github.com/galilai-group/lejepa/blob/main/lejepa/multivariate/slicing.py) 算法独立实现；保留有限样本 N 缩放、Gaussian 特征函数目标、对称梯形积分及单位 Gaussian 投影。17 个积分点、64 个投影的配置固定用于所有对照，没有替换为方差正则。

跨设备复现遵循 PyTorch 的[复现边界](https://docs.pytorch.org/docs/2.14/notes/randomness.html)和[数值精度说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)。
