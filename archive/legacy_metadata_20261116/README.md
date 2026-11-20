# HI-JEPA 代码与实验配置

> **当前方法已固定并停止本轮实验（2026-10-27）：[定型报告与 baseline 比较](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FINALIZE_20261027/V1/FINAL_REPORT.md)、[固定 workflow](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FINALIZE_20261027/V1/REPORT_V2/CURRENT_WORKFLOW.md)、[模型配置](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FINALIZE_20261027/V1/REPORT_V2/METHOD_CONFIG_FROZEN.json)。采用 C1L：冻结 Fine＋显式 G＋完整 Fine 三成员共识预测；移除 learned G/ST、ND、旧 Q 与 native 目标微调。FHN 匹配 Raw NMSE 改善 2.17%，高维负结果保留；裁剪评分诊断 0/4 确认，仅作诊断。未修改飞书。下方交接与实验说明保留历史时点。**

> **Coarse 唯一性边界实验已完成并停止（2026-10-26）：见 [正式结果报告](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261026/COARSE_BOUNDARY_V1/RESULTS.md)。18个Fine均通过；大间隔6/6确认单例，近并列原G仅3/6选对且全部保留歧义，完全并列全部保留6个候选。独立复算通过，无后续训练排队。**

> **稳定辨识新轮已完成（2026-10-26）：见 [Fine/G新混合与训练替换结果](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261026/IDENTIFIABILITY_CORE_V1/RESULTS.md)。新24组Fine/G与旧18组复评均通过原轴门；本轮已停止。**

> **最新完成结果（2026-10-26）：本轮训练与复评见 [FlowDirect多任务报告](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/outputs/HCF_METHOD_FREEZE_20261026/LATEST_OURS_BENCHMARKS_20261026.md)。本轮队列已全部完成。下方DeepSeek入口为此前交接历史。**

> **当前HCF-JEPA / FlowDirect实验交接（2026-10-26）：先读 [DEEPSEEK_START_HERE.md](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/DEEPSEEK_START_HERE.md)，再用 [DeepSeek启动提示词](/Users/yuzihang/Documents/ChatGPT/HI-JEPA/DEEPSEEK_PROMPT.md)。下方正文是原HI-JEPA谱模态工程说明，不是本次FlowDirect实验启动入口。**

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
| `scripts/run_baseline_v2.sh` | v2 GCSB 双 seed baseline 验收（本轮只跑 seed 0/1） | `bash scripts/run_baseline_v2.sh runs/formal_v2` |
| `scripts/run_formal_v2.sh` | v2 完整矩阵入口（本轮不启动 57 条） | `bash scripts/run_formal_v2.sh runs/formal_v2_full` |
| `scripts/run_v3_spectral.sh` | v3 局部模态约束读出：冻结 v2 特征，已授权同步后尝试正式 seed0／1 各 20k | `bash scripts/run_v3_spectral.sh runs/formal_v3_spectral runs/formal_v2 --force` |

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

## v2：GCSB 修订（本轮 baseline 验收）

v2 只替换 v1 的 hinge covariance floor，其他损失、模型、K 谱块、数据、训练预算和评估协议不变。新损失是**批次协方差软谱惩罚**：

```text
L_GCSB(c) = (mu/m) * sum_j max(0, log((lambda0 + eps) / (lambda_j(S) + eps)))
```

其中 `S` 是当前 train 批次的 coarse 表示 `c = G(E(x))` 的无偏协方差，`lambda_j` 是其升序特征值；固定 `lambda0=0.04`、`eps=0.005`、`mu=0.10`。它在特征值低于 `lambda0` 时给出梯度，在健康方向为零；不旋转、不白化、不改变 K 的 block 划分或推理路径，也不使用真实模态标签。

**它是软惩罚，不提供硬性防坍塌保证。** 正式门禁、阈值和主指标不变；v2 必须重新通过 `e1_baseline` 双 seed 验收，失败即保留结果并停止，不通过调参或放宽阈值“救回”。

v2 使用独立配置和输出目录：

- `configs/base_formal_v2.json`：在 `base_formal.json` 上只覆盖 barrier 键；
- `configs/matrix_baseline_v2.json`：本轮 seed 0/1 的 `e1_baseline` 验收矩阵；
- `configs/matrix_formal_v2.json`：与 v1 相同的 57 条完整矩阵，本轮不启动；
- `python scripts/make_v2_parity_bundle.py`：用新源码的初始化 checkpoint 和固定输入生成近共线、GCSB 激活的 parity bundle（无训练）；
- `bash scripts/run_baseline_v2.sh runs/formal_v2`：先执行 CUDA doctor、新源码的 CPU→CUDA parity，再从头训练 baseline seed 0/1（各 20,000 步）；
- `bash scripts/run_formal_v2.sh runs/formal_v2_full`：完整 v2 矩阵入口，仅在未来显式调用时运行。

由于 v2 修改了 `src/hijepa`，v1 随仓库提供的 parity bundle 的源码摘要会失配；v2 必须使用新源码下导出的 bundle（`HIJEPA_PARITY_BUNDLE` 可覆盖路径）。v1 的 `runs/formal` 原产物和配置保留，不与 v2 输出混用。

本轮 v2 baseline seed 0/1 均完成 20,000 步，四项防坍塌门禁均通过；但 coarse recovery 仍未达到 0.25 阈值，detail 的线性/二次探针仍在均值基线（NMSE≈1），fine 的长程预测较 v1 变差。门禁通过与恢复成功是两件事，详细数字见 `runs/formal_v2/BASELINE_REPORT.md`。

## v3：局部模态一致性的受约束非线性读出

已完成的服务器 V3 直接受约束拟合中，两个 seed 各运行新增 20,000 步，原结构约束和防坍塌门禁通过，但 coarse sin² 为 **0.493493／0.457402**，仍超过 0.25，`both_met=false`。二维函数子空间诊断已明显改善，但第二单轴模态仍未正确分离。原结果与 checkpoint 保留，不能把预测残差降低当作辨识成功。

随后已完成逐 block 局部模态约束实验：seed0 的两个实 block 局部残差分别为 **0.005997／0.006855**，FP64 与实际 FP32 结果近似一致，未满足约束；seed1 两个残差为 **0.000104817／0.000020862**，满足局部约束，但 coarse sin² 为 **0.335402**，仍未达到 0.25。两个 seed 均实际执行 200 次 AL 更新，21 是外层损失日志条数；罚参数在第 600 步达到上限。随后按 AL 改变重置 Adam 的完整运行发生了 **199 次重置，仍失败**，未支持“仅清空历史状态即可解决”的原假说；也没有证据将失败归因于 GPU 并行。

`exact_nullspace_projection_v1` 随后的服务器 seed0 已核验完整 20k，但 **accepted=0**，累计 499,449 次内部恢复和 3,332,776 次回溯；初末固定批次目标同为 **0.3370691679477298**。其 spec 指标相对 raw 的改善来自 query 初始化，不能称为训练收益。seed1 被用户中止，仅保留 partial 产物。

**本轮默认修订为 `solver_revision="feasible_block_sqp_v1"`。用户停止旧实验后的本地修复阶段已完成；最新授权为原位同步并尝试启动 seed0／1 各 20k 的正式 baseline，完成或失败后停止，不扩展 57 条矩阵。当前说明授权和待执行流程，不表示已上传或已启动。** 源训练继续复用 v2 双 seed checkpoint，E、F 和 G 隐藏层冻结。完整定义见 `docs/V3_RECOVERY_PLAN.md`：

- 保留非线性字典 `φ(x)=[1; E(x); U(E(x))]`、全 train 共同起点的 FP64 配对矩残差、原 K 块和参数域、原 rank/κ、batch scale 与 GCSB。`length=64` 有 65 个观测，三个训练跨度共同使用起点 0–56；不求 `C00` 逆，不截断字典。
- 固定点 `x*=observe(0)` 的 bias 由非 bias 权重派生，使 `c(x*)=0`；实际 FP32 固定点输出再次检查，bias 预算仍保留。
- 新模态残差使用观测 Jacobian `C=D_x c(x*)` 和 selector 的 observed-space `Jhat`，逐 block 约束 `C_i Jhat−K_i C_i`。原 rank 仍使用状态复合 Jacobian，二者不可混用。query center 和误差证书必须匹配。
- block 内 Cholesky 左预条件化只用于局部残差的谱范数度量，不改变模型前向、不跨 block 白化。query 的 induced∞ 误差乘 `sqrt(d)` 转为保守谱范数预算；数值容差单独记录，不按 test 调整阈值或新损失权重。
- W 的当前／未来两端和原参数化 K 接收梯度；默认额外 **每 seed 20,000 步**，继承源学习率、batch 和 seed，取末迭代，无重启、早停或测试选点。特征提取／拟合为 CPU FP64，`--device` 只选择端点评价设备。
- 将非 bias 读出精确写成 `M=C A⁺+Z Nᵀ`，`A=D_x φ_nonbias(x*)`，`N` 为完整 SVD 给出的 `Aᵀ` 零空间，`Z` 全部保留为自由参数。满列 rank 检查失败即停止，不截断字典。实际优化使用正交坐标，避免直接乘 `A⁺` 造成梯度预条件失衡，K 的原域参数继续可训练。
- 每个 outer step 仍采样一次原 train batch，依次更新 Z 和局部 `(T,Kraw)`，两块各有独立 Adam。只接受满足原约束且通过同 batch 原目标 Armijo 检查的步；拒绝时回滚该块参数及 Adam state，局部块失败不再撤销先前已接纳的 Z 更新。C／K／Z 始终可变，原目标和阈值不变，不添加损失。
- 局部方向模型线性化完整条件化残差矩阵 F，结合谱范数球、原联合权重／bias 预算和 state-rank 的 PSD 局部内近似；固定 ADMM 最多 64 次、rho=10。有限内迭代只产生候选方向，不声称精确投影或全局最优，最终依靠原非线性约束与原目标检查接纳。
- 真实源 C／K 的直接 25 次恢复未收敛，因此初始化改为确定性 selector-query 局部谱构造：按原 block 顺序匹配 `Jhat` 左特征行，统一缩放 C 使原 state-rank 最小奇异值为 `1.01κ`，K 初始化为同一 query 谱且必须处于原域，保留源全部 Z 系数。两个真实源 seed 的构造均已联合可行，无需缩小 Z；这只证明初值可行，不是训练成功。后续 C／K／T／Z 均可变，不固定特征向量或 K。记录初值投影范数、源 Z 保留情况及约束审计；匹配／原域或可行性检查失败即明确停止，不承诺一般可行保证。这是数值初始化变化，不使用全局 `Px` 或真模态标签。
- 循环中在 step1、原 checkpoint interval、间隔达到 30 秒及末步实时更新简表 stdout、`progress.json`／`progress.jsonl` 和原子 `readout_latest.pt`。checkpoint 包含双 Adam、完整坐标基、RNG、参数、统计及身份哈希。fresh 新 fit 显式清除自身旧 latest，避免误继承旧运行；resume 绝不删除 checkpoint／进度。`--resume` 仅适用于新 solver，须存在匹配 checkpoint、沿用原 planned budget，不重新初始化；已有 progress 却丢 checkpoint 会拒绝。未开始的 seed 可以按原预算首次启动。
- 旧 `exact_nullspace_projection_v1` 配置仍可复现旧 solver；缺少 revision 的旧保存配置继续走 AL／Adam 路径。新 runtime 集中在唯一新增文件 `diagnostics/readout_runtime.py`，不增加第三方依赖。聚合报告只纳入本轮 active seed 目录，并在每 seed 完成时更新。
- 最终实际 FP32 端点重查原 K／预算／state-rank 和新增局部约束。结构／方法合法性、防坍塌门禁、原科学恢复阈值独立报告；无效端点标为 diagnostic，默认加载器拒绝。端点保存冻结 source provenance 和新增 readout method config。local query 的 Jacobian 此次明确用于读出训练，属于方法修订。

最新授权的正式入口如下；先将旧 exact 的完整 seed0 和 partial seed1 合并为单个归档，再原位同步并 fresh 启动新 solver：

```bash
python -m unittest discover -s tests -p test_v3_spectral.py -v
# 默认新配置 configs/base_formal_v3_modal.json：
bash scripts/run_v3_spectral.sh runs/formal_v3_spectral runs/formal_v2 --force
# 仅新 solver 的后续同 revision 恢复，不用于旧 exact 产物，不与 --force 混用：
bash scripts/run_v3_spectral.sh runs/formal_v3_spectral runs/formal_v2 --resume
# 复现此前无局部模态约束的 V3（旧配置不改）：
HIJEPA_V3_CONFIG=configs/base_formal_v3.json \
  bash scripts/run_v3_spectral.sh runs/formal_v3_spectral runs/formal_v2 --force
# CPU 端点评价：
# HIJEPA_V3_DEVICE=cpu bash scripts/run_v3_spectral.sh runs/formal_v3_spectral runs/formal_v2 --force
```

脚本保持 `[OUTPUT] [SOURCE_ROOT] [CLI flags ...]` 接口；`HIJEPA_V3_CONFIG` 可显式选配置，相对路径按调用者工作目录解析。默认新配置继承 `configs/base_formal_v3.json`，旧配置继续保留原方法；入口复用 `diagnostics/spectral_readout.py`，约束计算集中在一个 `diagnostics/local_modal.py` 模块，源训练 `src/hijepa/*.py` 和 v1/v2 配置不改。当前只接纳 `full` baseline，不自动扩展其他 variant 或 57 条矩阵。`--force` 可替换输出目录的生成结果，执行前须保留冻结归档；v1/v2 源目录受保护。

本轮完整测试 **102 项，101 项通过、1 项因无 CUDA 跳过，耗时 8.717 秒**，无失败。覆盖 same-batch 原目标实际下降、零伪接纳、防无效训练报告、连续／断点累计 8 步逐位一致，以及 fresh 同身份旧 latest 清理边界。

真实 checkpoint 工程回归使用每 seed **64 条 train 轨迹、各 65 个观测时刻、8 个 outer steps**，运行于 Mac CPU、2 线程：seed0 接纳 8 个 outer／8 个 Z／4 个 local 更新，固定批次目标 **0.219797→0.185914**；seed1 接纳 8／8／8，目标 **1.174036→1.012891**。FP64 和实际 FP32 的 modal／rank／预算检查均通过。wall time 约 0.391／0.0857 秒，包含 moments／checkpoint，只是此规模与线程设置的工程耗时。局部 ADMM 模型可行检查均通过，但收敛标志为 0，不能将后验接纳等同于内层精确求解，也不能外推正式速度或恢复。

上述检查表明本轮代码发生了有效可行更新，不能说明正式模态恢复成功或外推正式速度；服务器 CUDA parity 和完整预算结果尚未测量。每 seed 总 20k outer budget 不变，每 outer 两个块的内部计算量与旧 solver 不同，分别审计。固定点一阶身份与轨迹残差仍不构成完整全局辨识证明。本轮不使用真模态标签、全局 `Px` 或 HCF 机制，不改阈值／cluster，不覆盖 v1／v2 旧结果。已获授权的下一阶段为归档旧 exact 结果、原位同步、fresh 启动双 seed 正式 baseline；不 resume 旧 exact、不干扰 HCF／已有 tmux，也不启动 57 条矩阵。正式启动状态及科学结论须以实际服务器产物为准。

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
| `diagnostics/spectral_readout.py` | v3 读出：FP64 配对矩、原 K／预算／rank、可选局部模态约束与增广拉格朗日、raw/spec 评价、端点及 parity；保留旧方法复现 |
| `diagnostics/local_modal.py` | 观测坐标局部模态约束、固定点 bias 消元、误差预算与数值控制；不改变前向表示 |

`full` 使用所有约束；`block_only` 去掉 rank；`hierarchical` 用一般非线性高层预测器且无 rank，保留相同目标选择、定标和防坍塌配置。`direct` 是相同 E→G 深度及参数预算的直接 m 维输出网络，保留内部 z 的共同 SIGReg，去掉细粒度预测支路。`no_fine` 保留与 full 相同 E/G/F 参数，只在源预训练去掉细粒度预测损失。

迁移阶段 full/no_fine 使用相同任务损失，冻结 G、K，只更新 E、F。源→目标仅改变细节机制，粗尺度规律与观测接口保持一致。共同达标阈值使用 step-zero 冻结的匹配与 decoder，在目标 calibration 的独立 monitor 轨迹上检查；test 不决定首达标时间。预算内未达标报告 `not_reached_within_budget`，不计作同精度成本。

## 数据和证据边界

模态选择采用预先声明、所有对照共享的局部模拟器映射查询。当前实现不是“只凭被动轨迹就完成目标认证”。selector 不读取真模态或真实谱标签；真值仅用于独立兼容性审计和评价。状态观测映射的可微访问用于 rank，所有对照享有相同访问权限。

默认系统的局部状态 Jacobian 为正规实／旋转块矩阵。非线性状态变换在平衡点导数为 I，只作用于细节维；显式 `nonnormal_extension` 被标记为主理论之外，主运行器不将其接纳为默认理论设定。

局部谱证书在声明的 C3 与 IEEE／tanh 误差模型下给出 J(0) 误差及谱轮廓重数；没有使用 directed-rounding 区间算术。它不等于全域谱证书或学习恢复证书。当前参数和固定参数预算的解析光滑性界保守且可能很松；完整有限样本恢复证书、完整比较解 joint cost 和优化误差界没有由训练自动获得。

真解审计分别报告 rank、scale、floor、谱域和逆收缩的兼容性；有限 batch 的正则代价与总体矩的代价可能不同。共振混合审计明确区分“谱域／floor／rank 接受”与严格逆收缩条件；canonical 共振例位于严格条件之外。

## 产物与恢复

每个训练任务保存 config、environment、data_reference、selection、compatibility、initial checkpoint、带 RNG／optimizer 的恢复 checkpoint、训练日志、评价和 seed 输出。源训练与迁移分别保存。

v3 复用 `runs/formal_v3_spectral`，保存 `REPORT.md`、`report.json`、聚合与逐 seed 的 CPU/CUDA parity、`seedN/source.json`、`optimization.json`（新增预算、初始／最终约束与训练日志）、`moment_diagnostics.json`、`spectral_readout.json`、`raw_endpoint.pt`、`spec_endpoint.pt`，以及 `raw/`、`spec/` 的原评价、多跨度残差和 train-only 门禁记录。它不写回 `runs/formal` 或 `runs/formal_v2`；当前 `candidates.json` 只声明未枚举 EDMD 候选，历史 EDMD 路径才保存候选列表。

GitHub 仓库包含源码、配置、脚本、验证汇总和必需的 parity bundle。完整本机轨迹、优化器 checkpoint、缓存及新生成的 `runs/` 不入 Git，继续保留在本地；克隆后可直接启动正式流程。

`SHA256SUMS` 校验入库文件；本地轨迹缓存由各自的数据 manifest 单独校验。

数据缓存身份包含生成代码、配置与 NumPy 版本，读取时校验 NPZ hash。源码或协议改变时禁止混用已有矩阵和结果；选择新的输出目录保留旧记录。相同源码与协议的 `--resume` 可跳过已完成任务、接续中断任务。不同设备恢复仍不保证相同后续训练轨迹。

汇总位于 `summary.json` 与 `SUMMARY.md`，包括每种配置的种子均值／标准差和 calibration 冻结的跨种子表示一致性。缺测为 null／N/A，绝不填零。

## 来源

方法依据：[HI-JEPA Proposal](https://bcnpo29t80vs.feishu.cn/wiki/WPJCwzObTiURfbkSzjFcMt5Unzc)，本次核对的页面修改日期为 2026-10-18。

SIGReg 按官方 [LeJEPA Epps–Pulley](https://github.com/galilai-group/lejepa/blob/main/lejepa/univariate/epps_pulley.py) 与 [随机投影切片](https://github.com/galilai-group/lejepa/blob/main/lejepa/multivariate/slicing.py) 算法独立实现；保留有限样本 N 缩放、Gaussian 特征函数目标、对称梯形积分及单位 Gaussian 投影。17 个积分点、64 个投影的配置固定用于所有对照，没有替换为方差正则。

跨设备复现遵循 PyTorch 的[复现边界](https://docs.pytorch.org/docs/2.14/notes/randomness.html)和[数值精度说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)。
