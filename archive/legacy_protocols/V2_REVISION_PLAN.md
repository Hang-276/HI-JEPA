# HI-JEPA v2 最小修订方案

日期：2026-10-19  
状态：**方案稿。暂不修改训练代码，不启动训练。**  
依据：`runs/formal_diagnostics/REPORT.md`、`summary.json`、`conformance.json`、`gradient_conflicts.json`、`recovery.json`、`nonlinear_readout.json`。  
原则：保留 `runs/formal` 原正式结果；主方案只引入一种直接增强 coarse 全局非退化、且兼容现有 K 谱块结构的机制；不做事后 0.92/0.90 合并，不放宽门禁阈值，不使用真实模态标签参与训练。

---

## 0. 本轮实施调整（2026-10-19）

按用户指示，本轮实现采用以下调整，后文相应阶段以本节为准：

- **取消独立的 2,000 步 pilot**。完成单元/数值/梯度/配置测试、CUDA doctor 和新源码 CPU→CUDA parity 后，直接从头运行 `e1_baseline` seed 0/1，各 20,000 步。
- **本轮只做双 seed baseline 验收**，不自动扩展 57 条矩阵，不实施 v2.1；完整矩阵脚本和配置保留，但本轮不启动。
- **parity 不额外训练**：使用新源码下的初始化 checkpoint 和固定输入；初始化时故意让 G 的末层两行近共线，使 GCSB 在固定批次上激活。脚本为 `scripts/make_v2_parity_bundle.py`，bundle 在 `runs/formal_v2_preflight/parity_bundle/`。
- v2 输出目录为 `runs/formal_v2`；v1 `runs/formal` 原产物和配置保留。
- 实现文件：`src/hijepa/losses.py`（GCSB）、`src/hijepa/config.py`（可选校验）、`configs/base_formal_v2.json`、`configs/matrix_baseline_v2.json`、`configs/matrix_formal_v2.json`、`scripts/run_baseline_v2.sh`、`scripts/run_formal_v2.sh`、`scripts/make_v2_parity_bundle.py`、`tests/test_v2_barrier.py`、README。
- 训练中可记录诊断，但不根据中途结果调整 λ0/ε/μ 或门禁阈值。
- **本轮结果**：seed 0/1 均完成 20,000 步，四项防坍塌门禁均通过；但 coarse recovery 仍未达到 0.25，detail 探针仍在均值基线，fine 长程预测较 v1 变差。门禁通过与恢复成功分开报告；详见 `runs/formal_v2/BASELINE_REPORT.md`。本轮未扩展 57 条矩阵，未实施 v2.1。

---

## 1. 背景与 v2 目标

v1 正式实验在 `e1_baseline_seed1` 的 coarse covariance floor 门禁停止：

- seed1 最终 coarse 相关系数 0.9924，最小协方差特征值 0.00501 < 0.032399；coarse effective rank 1.025。
- 协议核验未发现实现错误：损失重算逐项一致、stop-gradient 正确、状态复合 Jacobian 与有限差分一致、参数投影与 K 约束成立。
- 最终 G 上 coarse 梯度 6.27、scale 7.36、floor 仅 0.186；coarse 与 scale 的夹角余弦 −0.975；rank 项在最终点为 0。净 G 梯度在坍塌方向上接近驻点。
- seed0 在原点处的二维慢子空间已恢复（sin²≈0.0005），但全局 c 的线性读出 R² 仅 0.444；z→selected R²=0.995；这是全局 coarse 坐标/优化问题，不是数据或 E 完全丢失信息。

v2 的目标不是“证明方法有效”，而是做一个可解释、可复现的最小机制改动，验证能否在不改变 K 谱块结构、不改变正式评估阈值的前提下，消除 seed1 型全局 coarse 坍塌。**主 v2 不同时修改 fine 目标、不合并模态簇、不放宽阈值。**

---

## 2. 诊断口径修正（与本方案相关的部分）

- **NMSE 的均值基线是 1**。`normalized_mse = MSE / Var(target)`；1 等于均值预测。detail 指标≈1.0 表示只达到均值基线，线性/二次探针未检测到 detail；不能据此证明严格信息丢失。
- **SIGReg 比较计入 0.1 权重**。有效 SIGReg 梯度为 seed0 E `6.41`、seed1 E `4.49`（原始未加权值 ×0.1）；仍主导 E，但优势约为 coarse 的 12.8/4.6 倍。
- **收紧结论**：没有严格证明 seed1 是局部最优；没有证明全局不可逆或严格信息丢失；近简并 1D 混合的损失下界未直接测量。本方案只把这些作为设计依据，不把它们写成已证明定理。

---

## 3. 主方案：Global Coarse Spectral Barrier（GCSB）

### 3.1 机制与公式

只替换 v1 的 hinge covariance floor，其他损失项、数据、模型、K 参数化、训练预算、评估协议全部不变。

对当前 train 批次的 coarse 表示 \(c_i=G(E(x_i))\in\mathbb R^m\)（本实验 m=2），令

\[
\bar c=\frac1B\sum_{i=1}^B c_i,\qquad
S=\frac{1}{B-1}\sum_{i=1}^B (c_i-\bar c)(c_i-\bar c)^\top,
\]

\(S\) 的升序特征值为 \(\lambda_1\le\cdots\le\lambda_m\)。定义 **单侧谱 barrier**：

\[
\boxed{
L_{\mathrm{GCSB}}(c)=
\frac{\mu}{m}\sum_{j=1}^m
\left[
\log\!\left(\frac{\lambda_0+\epsilon}{\lambda_j+\epsilon}\right)
\right]_+,
\qquad [x]_+=\max(x,0).
}
\]

其中：

| 符号 | 取值 | 含义 |
| --- | ---: | --- |
| \(\lambda_0\) | `0.04` | 与 v1 floor std=0.2 对应的方差下限（\(0.2^2\)） |
| \(\epsilon\) | `0.005` | 数值平移；把 \(\lambda\to0\) 时的梯度上界限制为 \(\mu/(m\epsilon)=10\) |
| \(\mu\) | `0.10` | barrier 强度；在 seed1 的 \(\lambda=0.005\) 处单特征值梯度约 \(0.1/(2\times0.01)=5.0\) |
| \(m\) | 2 | 当前 baseline 的 coarse 维数；按 \(1/m\) 取平均，保持量纲 |

**梯度性质**（\(\lambda_j<\lambda_0\) 时）：

\[
\frac{\partial L_{\mathrm{GCSB}}}{\partial \lambda_j}
=-\frac{\mu}{m(\lambda_j+\epsilon)},
\]

当 \(\lambda_j\ge\lambda_0\) 时该特征值贡献为 0（ReLU 的次梯度取 0）。因此：

- 它只在**低于 floor**时施加惩罚，不会像 log-det 那样奖励方差膨胀，不会与 scale 项争夺“把方差变大”；
- 梯度随 \(\lambda\to0\) 增大（被 \(\epsilon\) 截断），直接补上 v1 floor 在坍塌点只有 0.186 的弱梯度；
- 在 seed1 的 \(\lambda=0.005\) 处，barrier 对单个特征值的梯度约 5.0；注意 coarse/scale 的 6.3/7.4 是 **G 参数空间**的梯度范数，二者不能直接等同，参数空间中的 barrier 梯度需在 Phase 1 实测。健康方向 \(\lambda\approx0.4\) 处为 0。

v2 总损失保持不变的形式，只把 \(L_{\text{floor}}\) 换成 \(L_{\mathrm{GCSB}}\)：

\[
L =
\lambda_f L_{\text{fine}}
+\alpha L_{\text{coarse}}
+\beta L_{\text{rank}}
+\gamma L_{\text{scale}}
+0.1\,L_{\text{SIGReg}}(z)
+L_{\mathrm{GCSB}}(c).
\]

full 变体仍取 \(\lambda_f=1,\alpha=\beta=\gamma=1\)。没有新增数据、标签或旁路目标。

### 3.2 为什么选这一种机制

1. **直接对应失败模式**：v1 floor 是唯一直接惩罚 coarse 协方差最小特征值的项，但其 hinge 在 seed1 的 0.005 处 G 参数梯度只有 0.186；GCSB 在同一状态的**特征值空间**梯度约 5.0，且越接近 0 越强。参数空间的实际梯度需在 Phase 1 实测；设计目标是使其与 coarse/scale 同量级，而不是已经证明。
2. **兼容 K 谱块结构**：GCSB 只读取 \(c\) 的协方差特征值，不改变 \(c\) 的坐标、不旋转、不白化、不改变 K 的 block 划分、不改变 real/rotation block 的参数化，也不改变逐 block 的 scale 规则。K 仍作用在同一 \(c\) 坐标上。
3. **不是跨模态联合白化**：v1 的 floor 本来就是全 \(c\) 协方差最小特征值的约束；GCSB 没有新增跨模态耦合，只是把同一 floor 的软 hinge 换成单侧谱 barrier。它不把协方差变成单位阵，不强制跨模态零相关，也不对 \(c\) 做线性变换；它只给最小特征值加一个单侧下界势能。它不会破坏旋转块内部的协方差结构。
4. **不改推理路径**：barrier 是训练正则项；模型 forward、checkpoint 结构、评估和 K 约束都不变。唯一变化是损失项，因此改动面小。
5. **不使用真实模态标签**：只用模型输出 \(c\) 和 train 观测；真实模态只在评估中使用。
6. **不叠加第二种机制**：主 v2 不加入 coarse/scale 解耦、不改 fine 目标、不合并 0.92/0.90、不改阈值。这样可以把门禁结果归因于 GCSB，而不是混合改动。

### 3.3 与理论、动力学坐标的兼容性

- **理论层面**：v1 协议已经有“固定 covariance floor（std 0.2）”的设定；GCSB 是把同一个 floor 从 hinge 软惩罚改成单侧谱 barrier，不引入新的结构假设，也不提供新的全局恢复证书。原局部谱选择、局部 rank 证书、K 逆收缩约束保持不变。
- **坐标层面**：barrier 在 K 实际作用的 \(c\) 坐标中定义；不改 \(G\) 的输出坐标，不做 PCA/whitening，因此不会改变 K 的特征方向与 block 语义。对 rotation block，barrier 只通过全 c 协方差的特征值防止出现接近零的方向，不强制该 block 的各向同性。
- **与 scale 的关系**：v1 的 scale 逐 block 约束均值与方差；GCSB 只在协方差特征值低于 \(\lambda_0\) 时施加梯度，健康方向上梯度为 0，因此不会主动与 scale 争夺方差。它可能间接改变 coarse/scale 的平衡；这一点需要在 validation 中测量，而不是假设已经解决。
- **门禁兼容**：\(\lambda_0=0.04\)，v1 门禁阈值仍为 \((0.2\times0.9)^2-10^{-6}=0.032399\)。v2 不修改门禁、不修改阈值、不合并簇；v2 必须用同一个门禁重新验收。

### 3.4 预计改动文件

只做版本化新增/最小修改，不改 v1 的 `runs/formal`：

| 文件 | 改动 |
| --- | --- |
| `src/hijepa/losses.py` | 新增 `coarse_spectral_barrier(c, mu, epsilon, lambda_floor)`；`compute_losses` 根据 `losses.coarse_floor_mode` 选择 `"hinge"`（默认，v1 行为）或 `"spectral_barrier"`（v2） |
| `src/hijepa/config.py` | 仅在配置显式提供时校验 `losses.coarse_floor_mode / coarse_barrier_mu / coarse_barrier_epsilon / coarse_barrier_lambda_floor`；若 mode=`spectral_barrier`，要求三个参数显式且为正；若 mode=`hinge` 或未提供，保持 v1 行为。**不把这些键加入 `DEFAULTS["losses"]`**，避免改变 v1 已解析配置的 `protocol_fingerprint` |
| `configs/base_formal_v2.json` | 新建；`extends: base_formal.json`，只覆盖 `name` 与 `losses` 的 barrier 键；数据/模型/K/训练/评估/迁移预算完全沿用 v1 |
| `configs/matrix_formal_v2.json` | 新建；`base: base_formal_v2.json`，jobs/seeds 与 `matrix_formal.json` 完全一致（19 设定 × 3 seeds = 57 条） |
| `scripts/run_formal_v2.sh` | 新建；与 `run_formal.sh` 相同流程，但设置 `HIJEPA_MATRIX=${repo_root}/configs/matrix_formal_v2.json` 和 `HIJEPA_PARITY_BUNDLE=<v2 bundle>`，输出 `runs/formal_v2` |
| `scripts/_experiment.py` | 最小改动：canonical matrix 路径可由 `HIJEPA_MATRIX` 覆盖；默认仍为 `configs/matrix_{profile}.json`，v1 入口行为不变 |
| `tests/test_protocol.py` | 新增 barrier 数值/梯度/配置测试：有限性、低于 floor 的非零梯度、高于 floor 的零惩罚、v1 默认仍为 hinge、配置校验 |
| `docs/V2_REVISION_PLAN.md` | 本方案 |
| `docs/PROTOCOL.md` / `README.md` | 增加 v2 入口与“只改 coarse floor”的说明（可选但建议） |
| `SHA256SUMS` | 若沿用校验清单，更新新增/修改文件 |

不改动：`models.py`（K block 参数化）、`spectral.py`（选择/证书）、`systems.py`、`data.py`、`evaluation.py`（主指标）、`runner.py` 的训练/恢复逻辑（`compute_losses` 内部自动生效）、v1 的 `configs/base_formal.json` / `matrix_formal.json`。

### 3.5 预计解决与仍未解决的问题

**预期改善（待验证）**

- seed1 型全局 coarse 坍塌：在 \(\lambda\to0\) 时提供强于 v1 hinge 的梯度，增加坍缩解的成本。
- 门禁方向的鲁棒性：barrier 直接作用于门禁所测的 covariance min eigenvalue，而不是间接依赖 scale。
- 由于是单侧 barrier，健康方向不额外奖励方差膨胀，避免与 scale 形成新的“越大越好”冲突。

**仍未解决**

- **coarse/scale 反平行**：GCSB 不直接解耦两者；它增加第三个力，但不保证消除净梯度抵消。validation 中要记录 `cos(coarse,scale)` 与净 G 梯度。
- **seed0 的全局非线性/部分可读 coarse 码**：GCSB 只防止 rank 退化，不强制全局线性、单射或可逆。0.92/0.90 的轴混合仍可能存在；主 v2 不做事后合并，单轴指标可能仍然差。
- **近简并轴的可辨识性**：不改变 cluster 规则；主结果仍按 singleton 簇和 signed permutation 评估。
- **detail 保留**：主 v2 不改 fine 目标；见 §4 的独立建议。
- **理论保证**：barrier 是训练正则，不构成全局恢复/可逆证书；原局部证书范围不变。
- **超参数与批次噪声**：μ/ε 只做了梯度量级设计，未做正式验证；batch=256 的协方差估计有噪声，未使用 EMA/运行统计（避免叠加第二种机制）。若门禁仍失败，按 stop rule 停止，不通过调 μ 使其通过。
- **迁移阶段**：`compute_losses` 在 source 与 transfer 都使用同一损失；GCSB 也会作用于迁移阶段。需要监控迁移状态与 G/K 冻结检查，但不在主 v2 额外改迁移目标。

---

## 4. fine 细节保留：单独修订建议（v2.1，不并入主 v2）

诊断事实：

- `z→fast modes` 逐坐标线性 R²≈0（seed0 0.0015/0.0009；seed1 ≈0）。
- detail 的线性/二次 probe NMSE≈1.0，即均值预测基线；不是简单的线性/二次可读。
- 原始观测线性包含全部 4 个模态（R²≈0.998），所以不是数据缺失。
- fine 预测 h1 归一化 MSE 0.048/0.043（好），但 fine 目标是 future z；z 可以在不保留 fast detail 的情况下把 future z 预测好。

**结论：需要改变训练目标，而不只是换评估探针。** 仅增加非线性 probe 不能保证 detail 被保留；若 detail 保留是科学目标，必须在训练中给 fine 分支一个观测空间/细节目标。

**v2.1 单独建议**（不与主 v2 同轮执行，避免混淆 coarse 机制）：

\[
L_{\text{detail}} = \lambda_d\,
\mathbb E\left[\left\|D(z_t)-x^{\text{clean}}_{t+1}\right\|^2\right],
\]

其中 \(D\) 是一个小的观测空间 decoder（线性或一层非线性），\(x^{\text{clean}}_{t+1}\) 是 future clean observation；只使用 train 观测，不使用真实模态标签。保留原 fine 的 future-z 目标，或把 fine 目标改为“future z + observation detail”的加权组合。\(\lambda_d\) 需预注册；先在单独输出目录做小预算消融，确认不损害 coarse/SIGReg，再考虑是否进入下一版正式矩阵。该建议不改变 main v2 的结论，也不把 detail 指标当作 main v2 的成功条件。

---

## 5. 衔接原正式预算的验证安排

### Phase 0：静态与单元测试（CPU，开发环境）

- `python -m unittest discover -s tests -v`：v1 全部测试 + 新 barrier 测试。
- barrier 测试至少覆盖：
  - 低于 floor 时返回正标量、梯度有限且非零；
  - 高于 floor 时惩罚为 0（或数值上 <1e-8）、梯度为 0/次梯度；
  - μ/ε 非法配置被拒绝；
  - v1 默认 `coarse_floor_mode="hinge"` 的解析配置和损失值不变。
- 不启动正式训练。

### Phase 1：CUDA 工程预检（非正式证据）

1. `python run.py doctor --device cuda --output runs/formal_v2_preflight/doctor_cuda.json`。
2. **新 parity bundle**：因为 v2 修改了 `src/hijepa/*.py`，仓库自带的 v1 bundle 的 `source_sha256` 会失配，不能复用。用一个短 v2 engineering checkpoint（local_integration、小预算，仅用于导出）执行：
   - `python run.py parity-export --checkpoint <v2_engineering_checkpoint> --output runs/formal_v2_preflight/parity_bundle`
   - `python run.py parity-check --bundle runs/formal_v2_preflight/parity_bundle/bundle.pt --device cuda --output runs/formal_v2_preflight/parity_cuda.json`
3. `python run.py manifest --config configs/matrix_formal_v2.json --output runs/formal_v2_preflight/planned_manifest.json`；核对 57 条记录、v1/v2 仅 `losses` barrier 键不同。
4. **本轮取消独立的 2,000 步 train-only pilot**（按用户指示）。上述单元/数值/梯度/配置测试、CUDA doctor 和新源码 parity 作为检查；检查通过后直接进入 Phase 2 的 20,000 步 baseline。训练中记录诊断，但不根据中途结果调整 λ0/ε/μ 或门禁阈值。

### Phase 2：v2 正式门禁（与原预算一致）

- 用 `e1_baseline` seed0/seed1，**canonical formal budget**：20,000 source steps、batch 256、train/calibration/test=4096/512/1024、length 64、z=32、width=128、FP32、无 AMP/TF32。
- 使用与 v1 完全相同的四项 anticollapse 检查与阈值：
  - fine effective rank ≥1.5；
  - coarse min covariance eigenvalue ≥0.032399；
  - fine/coarse dynamic fraction ≥0.01；
  - 仅 train，不访问 test。
- **两个 seed 都必须通过**；若任一失败，保留 `runs/formal_v2`，停止，不调 μ、不合并簇、不放宽阈值、不启动余下矩阵。该失败本身作为 v2 的工程结果报告。

### Phase 3：v2 完整矩阵（仅当 Phase 2 通过；本轮不执行）

- 运行与 v1 相同的 57 条记录（19 设定 × 3 seeds），源训练 20k，迁移最多 10k、每 250 步 monitor，相同 reuse 规则；输出 `runs/formal_v2`，`--resume`。
- 不覆盖/不混用 `runs/formal`；v1 作为冻结对照保留。
- 主指标、阈值、cluster 规则、test 使用方式完全不变；不做事后 0.92/0.90 合并。
- 可预注册一个**次要诊断**（不改主指标）：报告二维慢子空间 sin²、一般线性/二次/RBF 读出，以及 coarse/scale 梯度夹角。它只作为解释性 secondary endpoint，不能替代主门禁或主恢复指标。
- 预算：57 条记录、39 次独立源训练、6 次迁移；GCSB 每步只多一个 2×2 对称特征分解，预计 <1% 额外计算，整体仍按 v1 的 GPU 工期量级安排。迁移阶段同样使用 GCSB；监控 `G_K_bitwise_unchanged` 与迁移 monitor。

### 停止与解释规则

- Phase 2 失败：停止，不通过调 μ/ε 或改阈值“救回”；需要新协议版本时另开目录。
- Phase 2 通过但 Phase 3 出现非有限值/恢复指标恶化：按原协议保留记录并报告，不用 test 调参。
- 任何 μ/ε 变更都必须作为新协议版本记录；不能在同一正式矩阵中根据 test 结果调整。
- 训练全程禁止真实模态标签；GCSB 只使用模型输出 c；真实模态只在评估中使用。

---

## 6. 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| barrier 是软约束，不能保证 λ≥λ0 | 保持原门禁；失败即停；不把 barrier 当成证书 |
| barrier 与 scale 相互作用 | 单侧 barrier 在高 λ 处为 0；pilot 记录 coarse/scale 夹角与净梯度；不在主 v2 同时改 scale |
| batch 协方差噪声 | 与 v1 floor/scale 使用同一 batch；不引入 EMA（避免叠加机制）；记录特征值分布 |
| 特征值退化/数值不稳定 | `eigvalsh` + clamp(λ≥0) + ε；梯度上界 μ/(mε)=10；pilot 检查有限性 |
| 迁移阶段行为变化 | 同一损失在 transfer 生效；监控冻结 G/K、monitor 记录与最终评估 |
| 源码指纹变化导致 v1 parity bundle 失效 | 新建 v2 engineering checkpoint → 新 parity bundle；v1 结果保留但不用 v2 源码 resume |
| 近简并轴/单轴指标仍差 | 不在主 v2 合并簇；可预注册 secondary 子空间诊断；主结论不夸大 |
| detail 仍无保留 | 主 v2 不改变 fine 目标；detail 作为独立 v2.1 建议，不混入主结论 |

---

## 7. 推荐方案与选择理由

**推荐：采用 Global Coarse Spectral Barrier（GCSB）作为 v2 唯一主机制，替换 v1 的 hinge covariance floor；参数固定为 \(\lambda_0=0.04,\epsilon=0.005,\mu=0.10\)；其余训练/模型/K/评估/预算不变。**

理由：

1. **直接命中诊断出的缺口**：v1 floor 在 seed1 的坍塌点 G 参数梯度仅 0.186，而 coarse/scale 为 6.3/7.4；GCSB 在同一状态的特征值空间梯度约 5.0，且随 \(\lambda\to0\) 增强，设计上能提高坍缩解的成本。参数空间梯度是否与 coarse/scale 同量级需 Phase 1 实测，不能仅凭特征值空间数值断言。
2. **最小且可归因**：只改一个损失项；不引入新网络、不改 K 参数化、不改推理路径、不改数据/评估；单机制改动使门禁结果可归因。
3. **兼容 K 谱块结构**：barrier 在 K 作用的同一 c 坐标上，不旋转/不白化/不改变 block 划分；逐 block scale 仍在；real/rotation block 语义不变。
4. **不做被禁止的事**：不是跨模态联合白化；不事后合并 0.92/0.90；不放宽阈值；不使用真实模态标签。
5. **不会主动与 scale 争夺方差**：单侧 barrier 在 \(\lambda\ge\lambda_0\) 时为零，不会奖励方差膨胀；只在接近坍塌时介入。
6. **验证路径与原正式预算一致**：20k × seed0/1 门禁，再 57 条矩阵；v1 结果作为冻结对照保留。

**未选择/未堆叠的方案：**

- **跨模态联合白化**：被明确禁止，且会改变 K block 语义/推理路径。
- **对 c 加 SIGReg**：会施加完整的分布匹配，可能改变 block 尺度与旋转块结构；改动更大，且与现有 z-SIGReg 叠加。
- **只做逐 block 方差 floor**：seed1 的两个坐标方差都约 0.66，逐坐标方差 floor 无法阻止跨坐标相关导致的全局最小特征值坍塌。
- **硬 batch 投影/白化层**：依赖 batch 统计、改变推理/前向路径，且更接近被禁止的白化；不作为最小改动。
- **修改 K 或谱块**：破坏现有 K 谱块结构与理论兼容性。
- **事后合并 0.92/0.90 或放宽阈值**：会改写原正式结果，禁止。
- **主 v2 同时解耦 coarse/scale 或改 fine 目标**：会与 GCSB 混淆，违反“一次只验证一种机制”；这些作为后续独立实验。

---

## 8. 本方案明确不做什么

- 现在不修改任何训练代码、不启动任何训练、不修改 `runs/formal`。
- 不改变 v1 的正式配置、门禁阈值、评估指标或结果。
- 不把 0.92/0.90 事后合并成一个簇来“改善”v1/v2 指标。
- 不在训练中使用真实模态标签；GCSB 只用 c。
- 不在主 v2 中加入第二或第三种机制（SIGReg-on-c、coarse/scale 解耦、fine 目标修改、detail decoder 等）。
- 不承诺全局恢复、可逆性或理论证书；v2 只验证 coarse 全局非退化机制。

---

## 附录 A：诊断证据到设计选择的映射

| 诊断证据 | 对 v2 设计的含义 |
| --- | --- |
| seed1 c 相关 0.9924，min eig 0.00501；G floor 梯度 0.186 vs coarse 6.27/scale 7.36 | 需要直接作用于全局 covariance 的更强梯度；GCSB 在 λ=0.005 处的特征值空间梯度约 5.0，参数空间梯度待 Phase 1 实测 |
| rank 最终为 0；局部 B 满秩、全局 c rank-1 | rank 不能作为主修复；需要全局谱约束；GCSB 直接作用于全局 c 协方差 |
| coarse 与 scale cos≈−0.92/−0.975 | 主 v2 不叠加 scale 解耦；使用单侧 barrier 避免新增方差奖励；pilot 监控冲突 |
| seed0 原点局部子空间 sin²≈0.0005，但 c→truth 线性 R² 0.444；z→truth 0.995 | GCSB 只修非退化，不承诺轴/全局线性恢复；不做 cluster 合并；可另报 secondary 子空间诊断 |
| NMSE 1=均值基线；detail NMSE≈1.0；z→fast R²≈0 | detail 问题需要改训练目标，单独 v2.1；不混入主 v2 |
| 实现核验全部通过 | 不需要为 v2 修 bug；只做机制替换，避免把实现修复和机制验证混在一起 |

## 附录 B：v2 配置示意（仅示意，暂不创建/执行）

```json
{
  "extends": "base_formal.json",
  "name": "formal_v2_gcsb",
  "losses": {
    "coarse_floor_mode": "spectral_barrier",
    "coarse_barrier_lambda_floor": 0.04,
    "coarse_barrier_epsilon": 0.005,
    "coarse_barrier_mu": 0.10
  }
}
```

`matrix_formal_v2.json` 只需把 `base` 指向该文件，jobs/seeds 与 `matrix_formal.json` 完全相同。v1 的 `base_formal.json`、`matrix_formal.json` 和 `runs/formal` 不改。
