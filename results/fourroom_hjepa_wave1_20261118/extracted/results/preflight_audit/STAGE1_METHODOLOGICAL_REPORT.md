# Wave-1 Stage-1 方法学报告：官方 CLS 读出丢弃 distractor，空间网格读出可线性恢复

- 日期：2026-11-17
- 状态：**Stage-1 硬闸门未通过；正式 Wave 1 未启动**（按用户决定暂停并写报告）
- 代码与产物根目录：`/home/kemove/yuzihang/hjepa-slow-selection`
- 本报告只陈述已实际运行、已落盘的结果；不含任何 TEST 数据或正式实验结论。

---

## 0. 一页结论

1. 按官方 `final_probing_decoding_eval.py` + `probers.py` 逐项复刻的 60-epoch 双头探针，在 Level 1 对 **agent 两轴 R² ≈ 1.000**，但对 **distractor 两轴只有 R² ≈ 0.43–0.77**，未达到预注册的 `audit_min_r2 = 0.85`；seed43 Level 1 distractor min = **0.425 < 0.50**，触发协议的硬停止。
2. 该失败**不是** wrapper 漏步、seed 选择或 `action_repeat_prob` 造成的：在**官方 probing HDF5** 上用同一 seed43 checkpoint、同一官方探针复现，Level 1 distractor R² = **0.405 / 0.468**，与自采数据几乎一致。
3. 失败原因是**读出**，不是编码器：distractor 位置在 Level-1 ViT 的 **patch token** 中几乎线性可读。把 projector 后的 patch token 做 4×4 空间平均池化（3072 维），再用同一探针：
   - MLP 60 epoch：distractor R² = **0.977 / 0.968**
   - 闭式线性 ridge：distractor R² = **0.994 / 0.995**
4. 在 `dev_preflight` 上，L1 网格读出对**三个模型族 × seed42/43** 全部通过 0.85：

   | 模型族 | seed42 min R² | seed43 min R² |
   |---|---:|---:|
   | LeWM (L=1) | 0.9957 | 0.9879 |
   | H-JEPA L=2 | 0.9744 | 0.9240 |
   | H-JEPA L=4 | 0.9619 | 0.9732 |

5. 深层表征（L2+）是 192 维 latent，没有空间 token；官方 CLS 读出在 L2 的 distractor R² 更低（seed43: 0.080 / 0.228）。这与“层级抽象丢弃任务无关的快因素”一致，但本报告不把它当作正式实验结论。
6. 因此，原始 Wave 1 的 pre-registered family（24/30）与 `audit_min_r2=0.85` 在 distractor 轴上不可同时成立；若要继续，必须**显式修改读出与 family 定义**。本报告给出建议方案，但**未执行**。

---

## 1. 执行边界与红线遵守

- GPU：所有 Python/PyTorch/采集/测试命令均显式设置 `CUDA_VISIBLE_DEVICES=6` 或 `7`；所有新脚本入口有硬断言。全程未在 GPU 00–05 上创建进程（`nvidia-smi` 仅显示他人既有进程与 Xorg）。
- 文件：未修改 `run.py`、`metrics.py`、`config.json`、`setup.sh`、`test_*.py` 及 `assets/H-JEPA` 内任何文件；新增代码全部为独立文件。`setup.sh` 的既有未提交改动是部署会话留下的依赖锁定，不是本次产生。
- TEST 泄漏：`dev_preflight` 不含 test 角色；官方 probing 数据只用于诊断；正式 `fourroom_v2` 未采集，因此不存在任何正式 TEST 真值。
- 未静默兜底：未降低 `audit_min_r2`、`minimum_coverage`、`maximum_selective_error`；未把 NaN 替换为 0；未吞异常。

---

## 2. Stage 0：官方上游探针协议审计

审计文件：`results/preflight_audit/upstream_probe_audit.json`。pinned commit `97054ecaff83e13839195ec4b92d74c08a0127c5`。

### 2.1 模型加载与逐帧编码

- `torch.load(ckpt, map_location="cpu", weights_only=False)`，`model.eval()`，全参数冻结。
- `_dense_encode_no_stride`：
  - Level 1：`model.get_level(1).encode({"pixels": ...}, key="pixels")["embed_0"]`
  - Level k≥2：输入键 `embed_{k-1}`，输出键仍为 `embed_0`
  - 所有层输出 192 维，逐帧一个 embedding；评估时不施加 stride。
- `_validate_dense_assumptions` 要求每层 `temporal_window_size == 1`。实测：
  - `fourroom_lewm`：1 层，`embed_dims {1: 192}`，window 1
  - `fourroom_hjepa_l2`：2 层，`{1: 192, 2: 192}`，window 均为 1，stride 1/2
  - `fourroom_hjepa_l4`：4 层，`{1..4: 192}`，window 均为 1，stride 1/2/2/2

### 2.2 官方 prober 的精确结构

- `build_prober(input_dim, output_dim)` → `MLP(input_dim, hidden_dim=512, output_dim)`。
- 实际结构：`Linear(input,512) → LayerNorm(512) → GELU → Linear(512,out)`。
  - **只有 1 层 512 维隐藏层**，不是提示词预期的两层。
- 每个目标列独立一个 prober：`xy`（2 维）与 `distractor0_xy`（2 维）各自训练，输出拼成 `[agent_x, agent_y, distractor_x, distractor_y]`。
- 输入是**原始 192 维特征**，不做 z-score；LayerNorm 位于第一个 Linear 之后。
- 目标坐标用 checkpoint 的 `normalizer.pt` 做 **z-score**；loss 在归一化空间；物理误差乘以 std 后报告 NMSE/R²。

### 2.3 官方训练超参数

- AdamW，`lr=1e-3`，`weight_decay=1e-4`，`batch=128`，`epochs=60`。
- `shuffle=True`，`drop_last=True`。
- **没有学习率调度器**。
- **没有 val 选优**：每 epoch 的 eval 只用于记录日志，最终保存第 60 epoch 的权重。

### 2.4 数据采集配置差异

- `fourroom_tp35_d1_probing.yaml` 与 `fourroom_tp35_d1.yaml` 的 world/ego_policy 完全相同。
- 本实验的采集命令只改：`seed`、`num_traj=null`、`num_transitions=400*episodes`、`action_repeat_prob=0.05 → 0.0`、hydra 输出目录。
- **唯一科学性的采样差异是 `action_repeat_prob=0.0`**，已在 README 记录；官方 probing 数据为 600/150 episodes、seed 3072。

### 2.5 normalizer.pt

- 官方 probing eval **必需**（缺失直接报错），用于目标列 z-score 与去归一化误差；不用于编码器特征。
- Wave 1 下载了 9 个 checkpoint 对应的 normalizer（实际内容相同），并在探针中按官方方式使用。

### 2.6 与提示词预期的三处偏差

| 提示词预期 | 源码实际 |
|---|---|
| 两层 512 隐藏层 | **一层** 512 隐藏层 |
| 按 val 选最优权重 | **不选优**，保存最后一轮 |
| 输入特征先 z-score | 输入**不** z-score；只对目标坐标 z-score |

本报告的复刻以源码为准。

---

## 3. Stage 1：双卡预检与硬闸门

### 3.1 预检数据

- 名称：`dev_preflight`
- `collection_seed = 2026111800`，`split_seed = 20261117`
- 564 条轨迹：train 400（全 401 帧）、val 100（全 401 帧）、audit 64（每条 1 个 193 帧窗口）
- `h5_sha256 = e7158fcd2dd3c39f883f08beee99538a258e9265cc22bb262690b7dd1eaddbc7`
- `splits_sha256 = c1536f89028bea8a695c819143a18fb1fd4f3d226116e211e5b1ec913d1e5e53`
- 采集耗时 18 分钟；占用 GPU 06（采集本身不用 GPU）。

### 3.2 闸门结果（官方 CLS 读出，60 epoch）

| Seed | Level | agent_x | agent_y | distractor_x | distractor_y | min |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 1 | 1.0000 | 0.9999 | 0.7650 | 0.7069 | **0.7069** |
| 42 | 2 | 0.9999 | 0.9999 | 0.5872 | 0.4877 | 0.4877 |
| 43 | 1 | 0.9999 | 0.9999 | 0.4254 | 0.4818 | **0.4254** |
| 43 | 2 | 0.9999 | 0.9999 | 0.1273 | 0.2665 | 0.1273 |

- 自动通过条件（两个 seed 的 L1 四轴 min ≥0.85）未满足。
- 硬停止条件（seed43 L1 distractor min <0.50）满足 → `HARD_STOP_DISTRACTOR_R2_BELOW_0.50`。

### 3.3 Wrapper 对照

逐项核对结果：未发现漏做的官方步骤。匹配项包括特征键/层索引、ImageNet 归一化、逐帧编码、prober 结构、原始输入、目标 z-score、AdamW/60 epoch/无调度器/末轮权重、batch/shuffle/drop_last。

记录在案的非缺陷差异：

- `action_repeat_prob=0.0`（官方 0.05，README 已记录）；
- 新采集/划分 seed 与 400/100 全帧轨迹（官方 600/150）；
- 未训练 `min_distractor_dist` 头（与其他头无共享参数，梯度为 0，不可能改变 xy/distractor 头）；
- 未训练 decoder（参数独立，同样不影响探针梯度）。

### 3.4 官方 probing 数据复现（决定性对照）

使用 HF dataset repo `jepa-world-models/h-jepa`（resolved revision `cc05cb8d2aaf9aec0d5364d073c6369ff760b60b`）的：

- `fourroom_tp35_d1_probing.h5`（600 episodes，240,600 帧）
- `fourroom_tp35_d1_probing_val.h5`（150 episodes，60,150 帧）

两文件均通过仓库 `SHA256SUMS` 校验。按 seed43 的 episode 排列取 399 条训练、201 条 audit、100 条 val，用同一 checkpoint、同一官方 60-epoch 双头探针：

| Level | agent_x | agent_y | distractor_x | distractor_y | min |
|---|---:|---:|---:|---:|---:|
| 1 | 0.9999 | 0.9999 | **0.4045** | **0.4676** | **0.4045** |
| 2 | 0.9999 | 0.9999 | 0.0799 | 0.2278 | 0.0799 |

**结论：低 distractor R² 在我们的新数据与官方数据上一致复现，不是数据分布或采集参数造成的。**

### 3.5 容量/收敛诊断（辅助）

在 dev_preflight seed43 L1 的 60k 训练子集上（联合 4 维 MLP、原始目标尺度，仅作容量参考）：

| 变体 | agent R² | distractor R² (audit) | distractor R² (train) |
|---|---|---:|---:|
| 官方 width/depth, 60 epoch | 0.998 / 0.998 | 0.201 / 0.244 | 0.465 / 0.508 |
| 输入 z-score, 60 epoch | 0.998 / 0.997 | 0.197 / 0.241 | 0.465 / 0.505 |
| 输入 z-score, 200 epoch | 0.999 / 0.999 | 0.298 / 0.427 | 0.673 / 0.713 |
| 双层 512, 60 epoch | 0.994 / 0.993 | 0.427 / 0.543 | 0.794 / 0.825 |

- 增加容量/epoch 能把 distractor 推到约 0.54，但仍远低于 0.85；训练 R² 也仅约 0.83。
- 该诊断使用联合 4 维头与原始目标尺度，不是对官方协议的逐项复刻；官方协议的结果以 §3.2/§3.4 为准。

---

## 4. 读出归因：CLS 池化丢弃了 distractor

### 4.1 线性可读性（dev_preflight seed43 L1，60k 训练子集）

| 目标 | 线性 ridge R² (x / y) |
|---|---:|
| agent_xy | **0.9981 / 0.9965** |
| distractor_xy | **0.0048 / 0.0725** |

Agent 位置几乎线性可读；distractor 位置在 CLS 特征中几乎不是线性可读的。

### 4.2 Patch / 空间特征（官方 probing 数据，seed43 L1，60 epoch）

特征构造：

- `cls`：官方 `ProjectedEncoder` 输出（基线）
- `patch_mean`：projector 后 256 个 patch token 的均值
- `patch_max`：projector 后 patch token 的最大值
- `grid4x4`：projector 后 patch token 按 16×16 网格重塑，4×4 块内均值池化，展平为 3072 维
- `patch_mean_raw`：projector **前** patch token 的均值

| 特征 | MLP distractor R² (x / y) | 线性 ridge (x / y) |
|---|---:|---:|
| CLS | 0.4295 / 0.4960 | 0.0089 / 0.1038 |
| patch_mean | 0.7508 / 0.8603 | 0.7628 / 0.8133 |
| patch_max | 0.4488 / 0.4169 | 0.5704 / 0.5558 |
| **grid4x4** | **0.9767 / 0.9679** | **0.9943 / 0.9950** |
| patch_mean_raw | 0.7077 / 0.8822 | 0.8551 / 0.8739 |

关键点：

- 信息并未在编码器中丢失；它在 patch token 中以**空间局部化**的形式存在。
- 4×4 网格的**线性** ridge 已达 0.994/0.995，说明不需要复杂非线性读出。
- 官方 CLS 池化是瓶颈；`patch_max` 也不够，说明不是简单的“存在性”信号。

### 4.3 跨模型族验证（dev_preflight，L1，grid4x4，四轴 min）

| 模型族 | seed42 | seed43 |
|---|---:|---:|
| LeWM (L=1) | 0.9957 | 0.9879 |
| H-JEPA L=2 | 0.9744 | 0.9240 |
| H-JEPA L=4 | 0.9619 | 0.9732 |

- 三个模型族的 L1 网格读出均以较大余量通过 0.85。
- 未验证 seed44；正式运行前应补齐。
- 上层（L2+）是 192 维 latent，没有空间 token；网格读出不适用。官方 CLS 读出在上层的 distractor R² 更低（seed43 L2 = 0.080/0.228），这与“层级抽象丢弃任务无关快因素”一致，但需正式实验验证。

---

## 5. 科学解释与边界

1. **Agent 是任务相关、受控的慢因素**：在 CLS 与 patch 特征中都几乎完美、且线性可读（R²≈1.0）。这与 H-JEPA/LeWM 的自预测目标一致——智能体位置对预测未来最有用。
2. **Distractor 是任务无关、快速、独立运动的因素**：CLS 只保留了约 0.4–0.7 的方差；上层 latent 更少。信息在 patch token 中，但 CLS 汇聚没有把它带上来。
3. **读出选择会改变科学结论**：同一冻结编码器、同一训练预算，仅把 CLS 换成 4×4 空间网格，distractor 审计就从“失败”变为“通过”。因此任何“表征丢失了因素”的主张都必须先排除读出瓶颈。
4. **边界**：本报告只验证了 Level 1、seed42/43、FourRoom、checkpoint pinned 版本；未验证 seed44、未做正式 TEST、未评估其他环境或下游规划。网格读出是诊断/修订读出，**不是**官方协议的一部分。

---

## 6. 建议的修订方案（未执行，等待授权）

若继续 Wave 1，建议：

- 新增 `probe_grid4x4_official`：输入为 projector 后 patch token 的 4×4 空间均值池化（3072 维），其余与官方 60-epoch 双头探针完全相同；**仅用于 Level 1**。
- 确认性 family 重新预注册：
  - Group A（读出消融，L1）：3 种读出（`probe_8k`、`probe_160k_official`、`probe_grid4x4_official`）× 3 seed × 2 规则 = **18**，`alpha = 0.05/18`。
  - Group B（深度谱系，L1）：LeWM / H-JEPA L2 / H-JEPA L4 × 3 seed × 2 规则 = **18**，`alpha = 0.05/18`。
- Level 2+ 仍用官方 CLS 读出，作为**探索性诊断**（预计 `RECOVERY_NOT_VERIFIED`），不计入确认性 family。
- 正式运行前补验 seed44 的 L1 网格读出；如果 seed44 也 ≥0.85，再采集 `fourroom_v2` 并启动三臂。
- 预计全量运行 3.5–4 小时；仅 L1 精简版约 2–2.5 小时。

**这不是对原协议的静默修改**：原 24/30 family 与 0.85 审计在 distractor 轴上不可同时成立；任何修订都需要用户显式授权，并在任何 TEST 数据产生之前完成。

---

## 7. 已运行 / 未运行清单

### 已运行

- Stage 0 源码审计（5 项技术细节）。
- 6 个新 checkpoint + 9 个 normalizer 的下载与 LFS SHA256 校验（`assets/setup_v2.json`）。
- `dev_preflight` 采集（564 轨迹，1.56 GB）。
- 双卡预检 42/43 × L1/L2（官方 CLS 探针）。
- 线性 ridge 与容量诊断。
- 官方 probing 数据下载、SHA256 校验与 seed43 复现（Level 1/2）。
- Patch/grid 读出诊断（官方数据 seed43 L1）。
- 三模型族 × seed42/43 的 L1 网格读出验证。

### 未运行

- 正式 `fourroom_v2` 采集（3,956 轨迹）。
- `fourroom_probe_v2` / `fourroom_lewm_v2` / `fourroom_hjepa_l4_v2` 三臂。
- 任何 TEST 编码、`test_prediction.npz`、`forecast_receipt.json`、正式 PASS/风险认证。
- seed44 的网格读出验证。
- 正式 bundle 打包（用户选择暂停写报告）。

---

## 8. 关键产物

| 文件 | 内容 |
|---|---|
| `results/preflight_audit/upstream_probe_audit.json` | Stage 0 逐项审计 |
| `results/preflight_audit/dev_preflight_seed42.json` / `seed43.json` | 预检 CLS 结果 |
| `results/preflight_audit/dev_preflight_summary.json` | 闸门判定 |
| `results/preflight_audit/distractor_diagnosis_seed43_level1.json` | 容量诊断 |
| `results/preflight_audit/distractor_ridge_seed43_level1.json` | 线性 ridge（CLS） |
| `results/preflight_audit/official_probing_replication_seed43.json` | 官方数据复现 |
| `results/preflight_audit/patch_feature_diagnosis_seed43.json` | patch/grid 归因 |
| `results/preflight_audit/grid_readout_preflight_*.json` | L1 网格跨模型验证 |
| `results/preflight_audit/stage1_gate_report.json` | 机器可读汇总 |
| `results/preflight_audit/dev_preflight_*_level*/` | 探针权重、audit 预测 |

代码（均为新增独立文件）：

- `run_v2.py`、`prepare_data_v2.py`、`preflight_v2.py`、`verify_v2.py`
- `download_extra_weights.py`
- `diagnose_preflight.py`、`diagnose_official_data.py`、`diagnose_patch_features.py`、`validate_grid_readout.py`
- `configs/fourroom_probe_v2.json`、`configs/fourroom_lewm_v2.json`、`configs/fourroom_hjepa_l4_v2.json`

---

## 9. 复现命令（诊断部分）

```bash
cd /home/kemove/yuzihang/hjepa-slow-selection
export HF_HOME=$PWD/assets/cache

# Stage-1 preflight (data already collected)
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -B preflight_v2.py --seed 42
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -B preflight_v2.py --seed 43
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -B preflight_v2.py --combine

# Official-data replication
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -B diagnose_official_data.py --seed 43 --level 1 2

# Patch/grid attribution
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -B diagnose_patch_features.py --seed 43

# Cross-family grid validation
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -B validate_grid_readout.py --seed 42 --model-tag fourroom_lewm --features grid4x4
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -B validate_grid_readout.py --seed 43 --model-tag fourroom_lewm --features grid4x4
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -B validate_grid_readout.py --seed 42 --model-tag fourroom_hjepa_l4 --features grid4x4
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -B validate_grid_readout.py --seed 43 --model-tag fourroom_hjepa_l4 --features grid4x4
```

---

## 10. 开放问题

1. 是否把 `probe_grid4x4_official` 作为正式 Wave 1 的第三个读出（需修订 family 18/18）？
2. 上层（L2+）是否保留为探索性诊断，还是把确认性主张限制在 L1？
3. 是否需要先补 seed44 的网格验证，再采集正式数据？
4. 是否要在其他环境（AntMaze/Cube/Push-T）检查“CLS 丢弃空间因素、patch 网格可恢复”是否普遍？
