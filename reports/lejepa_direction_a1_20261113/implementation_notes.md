# A1 实现和可行性核对

## 已封存权重和读出

主候选为历史强恢复组的 14 个 checkpoint。名单来自 `reports/lejepa_dynamic_transfer_20261113/analysis/HISTORICAL_GROUPS.json`，判据是上一轮独立复验中，两个评价过程、四个物理轴的冻结 MLP 读出最低 R² ≥ .85。两个谱交错 seed 7811 checkpoint 在该组之外。选固定 B 阶段 MLP，避免再按现有评估准确率为每个模型挑臂。强组涵盖四个动力学训练条件。

- 原编码器在 `reports/lejepa_slow_rank_20261112/training/runs/{case}_{seed}.pt`。从原协议取 `hidden` 和 `layers`，用原 `make_mlp_encoder(4, hidden, layers, device="cpu")` 建模，以 `torch.load(..., weights_only=True)` 载入。编码器散列在原 `prospective/pipeline_forecast_receipt.json`。
- B 阶段物理读出在 `reports/lejepa_mechanisms_20261113/readout/runs/{case}_{seed}_readouts.pt`。该文件是可信本机产物，包含 NumPy 统计量和 PyTorch 参数。按 B 复验 `FORECAST_RECEIPT.json` 验散列后用 `weights_only=False` 读取。用原 `mlp(4,128)` 建读出并加载 `payload["mlp"]`。调用 `embed(encoder, states)` 编码，最后一维摊平到四轴，按 `payload["input_mean"]` 和 `payload["input_std"]` 标准化，再经 `predict_mlp(x, model)`，恢复输入路径的形状。对本轮 REF 与 TEST 都用同一冻结标准化。B 权重、readout 定义和编码输入的版本均应进新回执。
- 可选 C 对照为既有 `state` 臂的两个种子 `experiment/fits/{case}_{seed}_{probe_seed}_state.pt`，键为 `state,input_mean,input_std`，可 `weights_only=True`。C 的 `decoder(128)`、`predict(model, encoded_tensor)` 和 `load_encoder` 支持复用。若纳入，对两个既有 probe seed 等权，并保存每个结果。主实验只需 B 读出。
- 各模块文件都叫 `run.py`。加载本轮或旧模块时用 `importlib.util.spec_from_file_location` 指定唯一模块名，避免旧模块内部 `from run import ...` 导入调用端。B 的复验脚本已有可用的加载例子。

## 数据和评价定义

沿用四维标准化、独立、平稳 Gaussian AR(1)，输入先由原 `observe` 生成，之后只通过冻结编码器和监督读出。使用未用过的新 master，角色隔离为 CAL-mix、CAL-gap、REF、TEST；若需设计可行性 pilot，另设种子。不同模型和干预臂在相同物理条件上共享轨迹。跨条件是否共享底层随机创新在冻结协议写明。C 的 `trajectories(master, role, rates, count, length)` 展示生成方式。不要直接复用它固定的角色枚举来编码两个 CAL 子集，须给不同角色标识。

物理轴名已经由 B 的冻结标签监督确定，使用固定的轴索引。预测分数是 lag 4、16、32 三组平方增量均值的等权平均，除以该干预臂在独立 REF 上的自身经验方差。物理评价使用真实总体方差 1。每个 REF 面板的同一批物理状态经所有干预臂前向。可选以较大 REF 或多个独立 REF 检查方差扰动。实验只生成新轨迹和前向，14 模型不训练，最小预算建议 CAL-mix 128×193、CAL-gap 128×193、TEST 1,024×193、REF 32,768 个 iid 状态。按整条轨迹统计。

## 建议的最小干预和预测

在每个模型和物理条件内，固定只改变 recipient 轴 2（零基索引，物理总体第三慢），分别加 donor 轴 0（总体最慢）或轴 3（总体最快）。先在 CAL-mix 对 recipient 做共享的仿射校准或保持 B 原尺度，二者择一冻结。供体预测以 CAL-mix 自身均值中心化。记原 recipient 残差为 $e_2$，中心化供体为 $q_k$。令两臂的额外 CAL recipient MSE 都为预设 $\tau>0$，只用 CAL-mix 解析求出正混入系数：

$$
b_k=E_{\rm CAL}[e_2q_k],\qquad
c_k=E_{\rm CAL}[q_k^2],\qquad
a_k=\frac{-b_k+\sqrt{b_k^2+c_k\tau}}{c_k}.
$$

于是两臂具有相同的 CAL recipient 状态 MSE，其余输出逐元素相同。固定近似匹配阈值，并在揭盲后报告实际 TEST MSE 差；若额外匹配门失效，保留结果并缩小解释。固定供体极性、recipient、可选 $\tau$ 网格或唯一值，在看 TEST 结果前确定主臂。单次排序预测先看固定 1–2 物理边界对。另报全 named top-2；实际跨界身份随窗口变化，可在只由预测定义的候选近边界样本做预注册主/辅分析。保留包括未翻转窗在内的全部数据。

独立 CAL-gap 用真实物理标签拟合一个冻结的物理差预测，例如用名义冻结读出的窗口滞后矩特征预测 $D_2-D_1$，或者用 CAL-gap 估计固定的名义分数有符号偏差。候选分数和候选差在 TEST 直接由冻结读出、已知干预与 REF 得到。预测物理差的符号与候选差符号相反，即预报这对物理因素的次序错置。保存物理差估计、候选差、预测事件和对照评分，再独立揭盲真实物理差、true top-2 与正确性。事前固定 gap 阈值和打平处理。若想预测“干预改变基线对错”，先由名义差与两臂候选差预测发生差异的窗口，再由 CAL-gap 的物理差估计判断方向。单独报告物理差估计在独立 TEST 上的准确率，防止干预方向和物理差校准误差相互掩盖。

方向盲基线使用同一个 CAL-gap 物理差预测，附加两臂共用的 CAL MSE 信息、基线 margin 和事件分数。另可用仅基线分数是否接近零作为风险对照。两臂共用状态误差匹配结果，但供体身份及候选分数有差异。对比模型应使用相同的候选样本和覆盖／报警预算。连续分数可先报 paired Brier 或排序指标，并按固定阈值给出 flip 命中、漏报、虚警。物理真值只参与封存后的核验。

## 解释界线

- 对冻结候选坐标从 TEST 可见预测直接计算其分数，属于可计算的候选读出。用 TEST 真物理轴和逐窗残差精确重建分数差属于事后恒等分解，不能算物理 flip 的预测。
- 本设计检验已知定向干预和标注 CAL 下的机制。已知物理 rho 的解析总体差只能做先验或真值对照，无法给出该次新轨迹的窗口真值。可逆人工混轴也不模拟真实学习过程中自动产生的所有误差。
- 匹配状态 MSE 仍可能伴随供体增量能量、动态交叉矩和参考方差不同。它们是排序机制的组成部分。分别保存 CAL 与 TEST 的有符号差、候选分母和经验增量矩，避免将影响都叫作几何方向的纯效应。
- 窄总体差不保证每条窗口接近边界。事后真实边界分层仅作机制核对。主筛选规则若限定候选近边界，应由冻结预测与 CAL 阈值给出。固定轴对排序只为全 top-2 的辅助或在预先定义条件下作为代理。
- 同一物理轨迹被多个 checkpoint 和干预臂复用。重抽样需同步按轨迹及参考面板进行。结果条件于已选冻结模型、监督读出、四因素生成过程和配对干预。
