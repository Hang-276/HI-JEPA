# HI-JEPA v3：局部模态一致性的受约束非线性读出

日期：2026-10-20  
状态：**用户停止旧实验后的本地修复／验证阶段已完成。最新授权为原位同步 `feasible_block_sqp_v1` 并尝试启动 seed0／1 各 20k 的正式 baseline；完成或失败后停止，不重训源模型、不扩展 57 条矩阵。以下记录授权和待执行流程，不表示已上传、已启动或科学成功；服务器 CUDA parity 和新 solver 正式恢复结果尚未测量。**

HI-JEPA 继续研究 proposal 的部分模态目标：保留非线性细粒度模型，稳定辨识少量预先选定的高层动力学模态及其子空间。此次修订加强高层模态的动力学身份约束，保留非线性字典和原恢复评价。训练期 LSGA、固定全局观测目标 $Px$、HCF 条件密度机制均不属于本方案。

## 1. 已完成结果与修改依据

v2 的两个完整 baseline 均通过原 train-only 防坍塌门禁，但 coarse sin² 为 0.561234／0.813493，超过原阈值 0.25。fine forecast mean raw MSE 为 0.007645／0.010752；fine 长程退化和 detail 探针接近均值基线是独立问题。

历史 EDMD 路径得到的候选不满足原 K 块和权重预算，已作为诊断保留。随后服务器完成了冻结非线性字典上的直接受约束 V3 拟合，两个 seed 各新增完整 20,000 步：

| seed | 拟合时间 | 原结构约束／防坍塌 | raw coarse sin² | V3 coarse sin² | `both_met` |
| --- | ---: | --- | ---: | ---: | --- |
| 0 | 54.59 s | 通过／通过 | 0.561234 | 0.493493 | false |
| 1 | 51.99 s | 通过／通过 | 0.813493 | 0.457402 | false |

这些是**此次局部模态修订之前**的正式 V3 结果，来自已下载并校验的服务器归档；原结果和 checkpoint 保留。结果不能再描述为“尚未运行完整 checkpoint 拟合”。

只读诊断进一步表明：V3 的二维函数子空间 mean sin² 已改善至约 0.088／0.090，但第一模态单轴 sin² 约 0.028／0.019，第二模态仍约 0.959／0.896。局部目标二维子空间原本已较准确；满秩、协方差门禁和较小的平均轨迹残差，没有保证每个输出 block 对应正确模态。训练末点未获全局最优或严格驻点证书，不能排除优化问题，也不能据此宣称原网络无解。

新方法因此在保留全局配对残差的同时，加入**逐 block 局部动力学一致性约束**。它是一次方法修订，不只是原优化器的数值修补。

随后服务器已完成局部模态约束实验。seed0 两个实 block 的条件化局部残差分别为 0.005997／0.006855，FP64 与实际 FP32 结果近似一致，均未满足约束；seed1 两个残差分别为 0.000104817／0.000020862，局部约束满足，但 coarse sin² 为 0.335402，仍未达到原恢复阈值。seed1 全 train 配对残差为 0.1097，旧 V3 为 0.00435，说明局部可行与全局目标同时优化好仍有差距。两 seed 实际各有 200 次 AL 更新；21 是外层日志条数，不能当作 AL 更新次数。罚参数在第 600 步达到上限，固定步数完成和罚参数封顶均不能证明问题不可行或排除优化预算／收敛问题。

上一轮曾提出巨大梯度经 Adam 历史二阶矩持续抑制有效步长的数值假说。随后 AL 改变时清空 optimizer state 的完整运行实际执行了 **199 次重置，仍失败**；因此未支持“仅重置 Adam 即可解决”的解释，不能继续以它作为充分根因。本轮改为精确零空间参数化和可行性恢复，不增加方法损失、不调整查询误差或恢复阈值，也不将失败归因于与 HCF 并行或 FP32 导出。

此后 `exact_nullspace_projection_v1` 的服务器 seed0 已核验完成 20,000 个 outer steps，但 accepted 为 **0**，累计内部恢复 499,449 次、回溯 3,332,776 次。初末固定批次目标完全相同，均为 **0.3370691679477298**。其 spec 结果相对 raw 的改善来自 selector-query 初始化，不能归为训练收益；代码退出与完整预算完成也不能作为发生有效更新的证据。seed1 已由用户中止，只有 partial 产物，不能当作完整正式 seed。当前改用独立 block 接纳与可恢复进度，保留旧 revision 复现。

## 2. 冻结源模型、非线性字典与预算

直接复用 `runs/formal_v2/e1_baseline_seed0/1/checkpoint.pt`，核对完整源训练步数、`phase=source`、源码／协议指纹与数据 hash。源 E、F 和 G 隐藏层冻结，v1／v2 源配置与产物不变。

记 $U(z)$ 为原 G 的最后一个非线性隐藏层：

$$
\phi(x)=[1;E(x);U(E(x))],\qquad c(x)=W\phi(x),\qquad W=[b,V].
$$

非 bias 系数从原输出层初始化为 $V_{\rm raw}=[0,W_{G,\rm final}]$。K 的可训练参数复制自源模型，保留原实块／旋转块的类型、顺序、参数化和参数域。新约束启用后，bias 由固定点锚派生，见下一节；raw 端点仍保留原 bias 作参照，不是失败后的备选输出。

字典允许 $c(x)$ 对原观测保持非线性。它只覆盖已学固定特征的线性张成空间，不等于重新训练 E／G 隐藏层后原架构的全部表达空间。

## 3. 新增局部模态约束

### 3.1 固定点锚：消去 bias

令 $x_*=g(0)$，固定 $\phi_* = \phi(x_*)$。取

$$
b=-V\phi_{*,\rm nonbias},
$$

使 $c(x_*)=0$。bias 不再是独立优化变量；其原预算仍需满足。该消元只固定一个点的函数值，没有把全局 $c(x)$ 限制成线性观测读出。最终 FP32 计算的固定点输出仍需重新检查，不能把 FP64 代数消元等同于 FP32 精确零。

### 3.2 观测 Jacobian 与原 state-rank 分开

selector 已声明的 local query 提供观测坐标下的 $\widehat J_x\approx D F_x(x_*)$。定义

$$
C=D_xc(x_*)=W D_x\phi(x_*),\qquad R_i=C_i\widehat J_x-K_iC_i.
$$

实现必须检查 selector 的 `query_center` 与 $x_*$ 一致。新的 $C$ 不能与旧的状态复合 Jacobian 混用。原 rank 使用

$$
B=D_s(c\circ g)(0),\qquad \sigma_{\min}(B)\ge\kappa,
$$

仍按原规则独立保留。$B$ 的满秩表示局部非退化；$R_i$ 则约束第 $i$ 个 block 的动力学身份。

### 3.3 按 block 条件化残差，不白化前向

对每个原 block 令 $L_i=\operatorname{chol}(C_iC_i^\top)$，检查

$$
r_i=\left\|L_i^{-1}(C_i\widehat J_x-K_iC_i)\right\|_2
\le \epsilon_{J,2}.
$$

这里 $\|\cdot\|_2$ 是矩阵谱范数。$L_i^{-1}C_i$ 的行正交，因而不会通过单纯放大 $C_i$ 降低残差，也不会让二维旋转 block 内的小奇异方向被总 Frobenius 范数掩盖。奇异或非有限的 block 不可视为满足约束。

**Cholesky 只用于局部残差度量。** 它不进入模型前向，不改变 $c(x)$ 或 K，不跨 block 混合，不对训练批次的 coarse 表示做白化，也不要求跨模态零相关。

selector 的 certificate 使用 induced infinity norm，若维数为 $d$、误差预算为 $\epsilon_{J,\infty}$，本实现取保守转换

$$
\epsilon_{J,2}=\sqrt d\,\epsilon_{J,\infty}.
$$

记录原范数、原误差及转换后的值；不读取真实模态或真实谱来设阈值。FP64／FP32 数值检查容差分别为 $10^{-8}$／$10^{-6}$，与 query error 分开报告。局部观测 Jacobian 可非正规，不能仅凭普通特征值间隔就把该残差转换成无条件的轴恢复保证。

## 4. 保留的全局目标与新增约束求解

### 4.1 全量共同起点配对残差

沿用 $\mathcal H=\{1,k,2k\}$：

$$
\mathcal R(W,K)=\sum_{q\in\mathcal H}\frac1{Nm}
\sum_{(i,t)}\left\|W\phi(x_{i,t+q})-K^qW\phi(x_{i,t})\right\|^2.
$$

$N$ 为每跨度配对数，保持跨度求和和坐标平均。`length=64` 实际有 65 个观测时刻；baseline 的 $k=4$，三个跨度共同使用 $t=0,\ldots,56$，每轨迹每跨度 57 对，4096 条 train 轨迹共 233,472 对。

通过未中心化配对 Gram 矩在全部 train 配对上精确计算经验残差，特征提取／拟合固定 CPU FP64。分块只限制临时内存，不丢弃训练配对。不求 $C_{00}^{-1}$、不截断字典、不要求 W 是 EDMD 特征向量。W 的当前／未来两端和 K 均接收梯度，保持原 V3 语义；这与源训练的未来分支 stop-gradient 更新不同。

### 4.2 原目标与完整零空间参数化

原目标继续为

$$
L_0=\alpha\mathcal R+\beta L_{\rm rank}
+\gamma L_{\rm scale}+L_{\rm GCSB}.
$$

沿用源 $\alpha,\beta,\gamma,\kappa$、原 state-rank、逐 block scale 及 v2 GCSB（$\lambda_0=0.04,\epsilon=0.005,\mu=0.10$）。scale／GCSB 每步使用原 batch 规则：独立轨迹不放回采样，各轨迹随机取同一合法支持内的时刻。不以全体合并协方差代替批次正则期望。fine 与 $0.1\,\mathrm{SIGReg}$ 在冻结 E／F 后是常数。

本轮固定 `spectral_readout.solver_revision="feasible_block_sqp_v1"`。设非 bias 特征维数为 $p$、观测维数为 $d$，令 $A=D_x\phi_{\rm nonbias}(x_*)\in\mathbb R^{p\times d}$。使用完整 SVD，检查 $A$ 满列 rank；数值 rank 不足时明确失败，不截断字典或用低 rank 近似掩盖退化。完整零空间参数化沿用旧 exact revision，但优化与接纳事务不同。

令 $N\in\mathbb R^{p\times(p-d)}$ 为 $\ker(A^\top)$ 的完整正交基。非 bias 读出采用精确参数化

$$
M=C A^\dagger+Z N^\top,\qquad MA=C,
$$

其中 $C$ 为局部观测 Jacobian，$Z$ 保留全部自由系数。它覆盖原完整读出空间，没有把高阶非线性零导数方向冻结或删除。令薄 SVD 为 $A=U\Sigma V^\top$，实际优化正交坐标

$$
T=C V\Sigma^{-1},\qquad C=T\Sigma V^\top,\qquad M=T U^\top+Z N^\top,
$$

避免在优化参数映射中直接使用病态 $A^\dagger$。这里的基变换只重参数化完整读出，不对前向 coarse 表示白化。$K$ 的原 raw 参数继续可训练，原实块／旋转块、顺序和参数域不变。bias 仍由 $b=-M\phi_{*,\rm nonbias}$ 派生，所有预算均作用于重构的 $M,b$。

### 4.3 独立 block 更新、有限局部模型与原约束接纳

每个 outer step 按原 RNG／支持规则只采样一次 train batch，先更新完整零空间 $Z$，再更新局部 $(T,K_{\rm raw})$，两个 block 使用独立 Adam。两块都基于原 $L_0$、同一 batch 检查下降，原目标与科学阈值不变，不增加损失项。局部 block 失败时只撤回该 block 的参数及 Adam state，已接纳的 $Z$ 不回滚；拒绝 $Z$ 时也恢复该块 Adam state，避免把未执行的数值步写入历史。

局部候选方向对完整矩阵

$$
F_i(T,K)=\operatorname{chol}(C_iC_i^\top)^{-1}
\left(C_i\widehat J_x-K_iC_i\right)
$$

作一阶线性化，而非只线性化标量最大奇异值。小模型结合逐 block 谱范数球、重构权重／bias 的原联合预算及 state-rank 的 PSD 充分局部内近似。固定 ADMM 最多 64 次、数值 rho=10，有限内求解仅生成方向；不声称得到精确投影、内层全局解或原非凸问题最优解。$Z$ 仅在此次局部 block 中固定，后续 outer steps 继续更新。

候选必须重新通过**原非线性约束**（local modal、state-rank、权重／bias、固定点）及**同 batch 原目标 Armijo**后验检查，满足局部小模型本身不等于可接纳。回溯仍不能接纳时，恢复该 block 的参数及 optimizer state。实时记录原约束、接纳／拒绝、模型可行性、有限 ADMM 收敛标志及模型迭代，不将它们合并成一个“优化成功”标记。真实 source C／K 直接进行 25 次旧 Gauss–Newton 恢复未收敛，属于旧实现记录，不是新 solver 的迭代规则。

当前使用**确定性的 selector-query 局部谱初值**：按原 block 顺序匹配 $\widehat J_x$ 的左特征行构造 $C$，统一缩放使原 state-rank 的 $\sigma_{\min}$ 为 $1.01\kappa$；$K$ 初始化为相同 query 的匹配谱，必须处于原 K 参数域。$Z$ 初值保留 source 非 bias 读出在完整零空间中的全部系数，不以零值替换非线性自由部分。两个真实 source seed 的这一构造均已通过联合初值可行性检查，无需缩小 $Z$，但这不是优化过程或恢复成功的证据。

这是明确的数值初始化变化，不固定训练期特征向量或 K。初始化后 $C/K$ 及其正交坐标 $T$、完整零空间参数 $Z$ 都继续更新。构造只使用既有 selector query、原 state-rank 接口及冻结特征，不使用真实模态、calibration／test、全局观测目标 $Px$ 或历史成功输出选点。记录初值投影范数、source Z 是否完整保留、K 原域及全部约束；谱匹配／原域检查失败或找不到可行初值时明确停止，不宣称一般初始化可行保证。历史无训练代数见证也不等于候选步的有限次恢复过程一定成功。

同一 seed 输出目录保存完整双 Adam、参数布局、候选拒绝／局部模型审计；不把有效预条件器单独当成实际参数更新或恢复证据。数值常量固定进入运行记录，不作为按 test 调节的超参数。本轮不新增第三方依赖。

`exact_nullspace_projection_v1` 旧保存配置仍走原联合候选／恢复路径；缺少 `solver_revision` 的旧配置走原 AL／Adam，继续按其 `optimizer_state_policy` 使用 `retain` 或 `reset_on_al_change`，缺省为 `retain`。不能将新 solver 静默用于旧配置复现。

学习率、batch、seed 和总外层迭代数继承各自源配置，CPU FP64 拟合保持 8 线程。canonical baseline 为每 seed **额外 20,000 次读出外层迭代**，每 outer 的两个 block 分别审计；拒绝候选仍计入外层预算，不补跑至 20k 接纳步，不把新旧 solver 的外层数相同表述为运算量相同。不重跑源训练。非 bias 联合权重与派生 bias 均保留原预算；K 保留原参数域。不随机重启、不早停、不按 train 门禁或 calibration／test 挑点。固定预算输出最后状态；失败如实保留，不能称已证明问题不可行。

微型工程 fixture 可显式缩短 `spectral_readout.solver_steps`，记录 `engineering_budget_override=true`；这不构成正式实验或方法筛选。当前仍只支持 `model.variant=full` baseline，不自动扩展到其他 variant 或 57 条矩阵。

## 5. 实际端点、provenance 与接纳层次

拟合后按源 dtype 构建实际端点（canonical FP32），重新检查 K 域、联合权重／bias 预算、原 state-rank、固定点输出和逐 block 模态约束。直接对实际端点求观测输入 Jacobian，不能只对 FP64 特征 Jacobian 乘以 cast 后的 W。FP64 拟合残差与实际部署残差分别报告。

三层接纳保持分开：

1. **方法与结构合法性**：原 K 域、预算、rank 加新增固定点／局部模态约束均满足，才可标为 `valid_main_model`。失败时端点为 diagnostic，默认加载器拒绝；诊断加载需显式 `allow_diagnostic=True`。
2. **原防坍塌门禁**：fine effective rank、coarse 最小协方差特征值及 fine／coarse 动态比例保留原 train-only 规则和阈值。
3. **原恢复标准**：calibration 冻结匹配、test、cluster／signed-permutation 和共同阈值不变。只有合法性、原门禁、原恢复标准通过且所请求的 parity 无失败，才报告此设定恢复成功。

端点区分保存冻结 source config／协议指纹与新增 readout method config／指纹，保存 local query 的中心、J、误差预算和约束记录。源 selector certificate 不改写，但新方法明确记录 **selector Jacobian 已用于读出训练约束**；这是比旧 V3 更强的训练使用。拟合不读取真模态标签、calibration／test 或全域 Jacobian，也不使用固定全局 $Px$ 目标。原评价仅在末迭代冻结后执行。

CPU/CUDA parity 覆盖新增 observed-feature Jacobian、固定点、局部约束、目标／梯度及实际端点；激活约束 fixture 与最终端点分别检查。本机无 CUDA 时记录未知／跳过，不能声称通过。固定输入 parity 不证明优化轨迹逐位相同。

## 6. 入口与旧实验复现

新配置 `configs/base_formal_v3_modal.json` 继承保持不变的 `configs/base_formal_v3.json`，启用 `spectral_readout.local_modal`，显式设置 `solver_revision="feasible_block_sqp_v1"`。旧配置／保存配置保持原 solver 语义。入口和测试继续复用 `diagnostics/spectral_readout.py` 与 `tests/test_v3_spectral.py`；局部约束使用 `diagnostics/local_modal.py`。唯一新增的内聚模块 `diagnostics/readout_runtime.py` 负责原子 checkpoint、身份验证和实时进度，不增加另一套源码目录。`src/hijepa/*.py`、v1／v2 配置和原产物不改。

此前“只在本地修复、先不上传／重启”的阶段已经完成。用户现已授权原位同步并尝试启动正式实验；先将旧 exact 的完整 seed0 和 partial seed1 合并为单个归档，再使用以下 fresh 入口，不 resume 旧 exact：

```bash
python -m unittest discover -s tests -p test_v3_spectral.py -v
# 默认使用新增局部模态配置；OUTPUT / SOURCE_ROOT 接口不变：
bash scripts/run_v3_spectral.sh runs/formal_v3_spectral runs/formal_v2 --force
# 新 solver 的后续同 revision 恢复：同预算、同身份，不用于旧 exact checkpoint：
bash scripts/run_v3_spectral.sh runs/formal_v3_spectral runs/formal_v2 --resume
# 显式复现此前不含局部模态约束的 V3：
HIJEPA_V3_CONFIG=configs/base_formal_v3.json \
  bash scripts/run_v3_spectral.sh runs/formal_v3_spectral runs/formal_v2 --force
# 仅将端点评价改为 CPU；拟合本身始终 CPU FP64：
# HIJEPA_V3_DEVICE=cpu bash scripts/run_v3_spectral.sh runs/formal_v3_spectral runs/formal_v2 --force
```

脚本接受 `[OUTPUT] [SOURCE_ROOT]` 和后续 CLI flags；`HIJEPA_V3_CONFIG` 可选择配置，显式相对路径按调用者工作目录解析。默认端点评价设备 CUDA。已授权的下一阶段先单归档旧完整／partial 证据，再原位同步并复用原输出目录，fresh 运行 seed0／1 各 20k。`--force` 允许替换指定生成结果，不能覆盖唯一历史证据；`--resume` 不应配合 `--force` 使用，也不能用于旧 exact checkpoint。脚本保护 v1／v2 源目录；不干扰 HCF 或既有 tmux 进程，不扩展 57 条矩阵。此处不预报同步或启动已完成。

主要产物继续集中在同一输出目录：`config.json`、`environment.json`、`REPORT.md`、`report.json`、聚合 parity，以及 `seedN/` 下的 source／optimization／约束记录、raw／spec 端点、原评价、残差和门禁。`candidates.json` 声明不枚举 EDMD 候选。历史 `solver="edmd"` 仅用于旧诊断。聚合 `report.json` 每完成一个 seed 便更新，`active_seed_directories` 只列本轮有效目录；旧目录残留不能算当前运行证据。

### 6.1 循环内实时进度与可信恢复

新 solver 在 **step1、原 source checkpoint interval、距离上次发布至少 30 秒、末步** 于实际循环内发布简表 stdout 并 flush，同时原子更新 `progress.json`、追加 `progress.jsonl`、原子替换 `readout_latest.pt`。记录 completed／planned outer steps、两个 block 接纳数、损失／约束与局部模型审计；不等到训练返回后才补写进度。进度按完成 outer 边界保存，未完成的外层迭代不伪记完成。

`readout_latest.pt` 包含完整双 Adam、全部参数、完整坐标基、NumPy／Torch／Python RNG、累计统计和不可改写的 identity。写 checkpoint 失败时保留上一个已发布文件并清理临时文件；checkpoint 与进度不是跨文件原子事务，恢复以最后可信 checkpoint 为准。

fresh 新 fit 在自身输出目录显式清理旧 `readout_latest.pt`，避免同身份的旧 checkpoint 被误当成本次进度；resume 不删除 checkpoint 或进度。runtime 的 `reset_progress()` 仍仅删除 `progress.json`／`progress.jsonl`，latest 清理由 fresh fit 显式负责，不改变该 API 契约。

`--resume` **仅支持 `feasible_block_sqp_v1`**，必须在指定输出中找到 checkpoint，检查 kind／schema／revision、完整 source provenance、model／train features、config／selection、solver code 哈希、声明的 Torch／NumPy 环境版本、线程数和原 planned budget，并逐项验证坐标基和参数布局。不匹配明确拒绝，不重新初始化、不增加预算、不清空既有进度。某 seed 尚未开始且完全没有进度／checkpoint 时，可在同次恢复调用中首次运行其原预算；已有 progress 却丢 checkpoint 的 seed 拒绝恢复。完整 seed 结果只在完成时进入当前聚合报告；checkpoint／partial 不能被称为正式恢复成功。

## 7. 停止条件与未证明部分

局部模态约束规定固定点附近的一阶动力学身份，全局轨迹残差约束已覆盖区域中的非线性形状；两者不能自动排除覆盖外误差、高阶非线性污染或有限样本问题。局部约束通过不是稳定全局辨识证书，固定字典求解失败也不是原网络表达能力的不可行性证明。

不更改 0.25 恢复阈值，不事后合并 0.92／0.90，不改变真值使用边界，不同时增加 decoder/detail 目标。E／F 和 fine 评价路径保持冻结，本修订不声称修复 fine 长程或 detail 问题。任一正式 baseline 未通过原接纳标准即保留并停止，不扩展 57 条矩阵。

本次本地代码与工程检查的实际结果以本轮检查输出为准；此前 V3／AL reset／exact-nullspace 的测试计数和服务器 parity 不能作为本轮新 solver 的验证结果。已完成的本地阶段只使用小规模 train 子集；下一阶段已获授权同步并尝试正式双 seed baseline，结果以实际服务器产物为准。

## 8. 验证记录与本轮交付边界

本轮 `feasible_block_sqp_v1` 完整测试 **102 项，101 项通过、1 项因无 CUDA 跳过，耗时 8.717 秒**，无失败。覆盖 same-batch 原目标实际下降、零伪接纳断言、防无效训练报告、断点恢复与连续累计 8 步逐位一致，以及 fresh 同 identity 旧 latest 的清理边界。本机无 CUDA，不能声称服务器 parity 已通过。

本轮真实 checkpoint 工程回归使用 **64 条 train 轨迹、每条 65 个观测时刻、每 seed 8 个 outer steps**，运行于 Mac CPU、2 线程，没有读取 test 作方法筛选：

| seed | outer 接纳 | Z 接纳 | local 接纳 | 初末固定 train 审计目标 | wall time |
| --- | ---: | ---: | ---: | --- | ---: |
| 0 | 8／8 | 8／8 | 4／8 | 0.219797 → 0.185914 | 0.391 s |
| 1 | 8／8 | 8／8 | 8／8 | 1.174036 → 1.012891 | 0.0857 s |

两个 seed 的 FP64 和实际 FP32 modal／rank／权重与 bias 预算检查均通过；参数确实发生更新，局部块拒绝没有撤销先前接纳的 Z。局部 ADMM `model_feasible` 检查均通过，但 `converged` 计数为 **0**，因此只报告有限模型和后验原约束检查通过，不称内层已精确收敛或全局最优。wall time 包含 moments／checkpoint，仅适用于 64 条 train 轨迹、Mac 2 线程的此小规模工程，不能外推正式速度／恢复率。工程回归只验证可行有效更新及产物，不证明完整预算模态恢复成功。

另使用两个完整冻结源 checkpoint 已保存的局部特征 Jacobian 做了**无训练代数可行性见证**。令 $A=D_x\phi_{\rm nonbias}(x_*)$，取 selector 的 $\widehat J_x$ 按原实 block 排序的左特征行 $C$，以公共尺度使原状态复合 Jacobian 的最小奇异值为 0.101，再构造

$$
V=C A^\dagger,\qquad b=-V\phi_{*,\rm nonbias}.
$$

该历史代数见证未读取真实模态标签或 test，当时也未作为优化初始化；其最小范数构造与当前保留 source 全部 $Z$ 的局部谱初值应分别记录。观测特征 Jacobian $A$ 在两个 seed 均为 rank 4，条件数分别约 $1.066\times10^5$／$1.199\times10^4$。

| seed | 非 bias 权重范数 | bias 范数 | state-rank 最小奇异值 | 两实 block 局部残差 |
| --- | ---: | ---: | ---: | --- |
| 0 | 1.333292641 | 0.006777464 | 0.101 | $5.41\times10^{-14}$／$3.80\times10^{-14}$ |
| 1 | 0.034559066 | 0.000312378 | 0.101 | $2.89\times10^{-15}$／$1.59\times10^{-15}$ |

该见证仅说明原权重／bias 预算内的**局部条件**可实现；它不检验 coarse batch 防坍塌门禁、全轨迹配对残差或恢复，不能推导完整受约束问题可行，也不能说明新 optimizer 策略一定成功。高条件数本身仍提示特征局部求解可能数值敏感。

`base_formal_v2.json` 与旧 `base_formal_v3.json` 均与冻结结果包逐字节一致。`src/hijepa/*.py` 指纹仍为 `4cb657d2560eb7ded47d299bda974de8433e5289935bf0c685ac472da8d92e48`；原结果归档 SHA256 仍为 `12a92f59898841073533def8df3bf83b255365191d4814da045b1b3726889cfc`。本轮实际 CPU→CUDA parity、完整预算约束可行性和恢复率尚未测量。

改动集中在以下文件，不增加备份版本目录：

| 文件 | 本轮变更 |
| --- | --- |
| `configs/base_formal_v3_modal.json` | 显式启用完整零空间／可行性恢复 solver，旧配置保留 |
| `diagnostics/local_modal.py` | 完整 F 局部方向模型、有限 ADMM、原约束与旧 solver 兼容数值工具 |
| `diagnostics/spectral_readout.py` | 独立 Z／local 事务、同 batch Armijo、真实数值进度、恢复入口与当前聚合报告 |
| `diagnostics/readout_runtime.py` | 唯一新增模块：原子 checkpoint、严格 identity、循环内进度及完整状态恢复 |
| `tests/test_v3_spectral.py` | 针对性数值／工程验证 |
| `README.md`、本文件、`SHA256SUMS` | 使用说明、审阅记录和文件完整性 |

当前授权边界为**保留旧 exact 完整 seed0／partial seed1 的单个归档、原位同步本地修复、fresh 尝试运行新 solver 的 seed0／1 各 20k 正式 baseline，然后停止**。此前只在本地修复且未上传／重启的阶段已完成；当前记录下一阶段授权，不表示已执行。v1／v2 源结果不改，不 resume 旧 exact，不干扰 HCF／已有 tmux 或其他进程，不启动 57 条矩阵，不新增备份版本目录或独立报告文件。服务器实际 CUDA parity、新 solver 的正式启动／完成状态和恢复判定均以实际产物为准，不由小规模有效更新预报成功。
