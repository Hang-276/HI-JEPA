# JEPA 研究进展同步

审计日期：2026-11-16。研究问题：**因素已经能够辨识或恢复以后，还需要什么条件，才能可靠地选出有限轨迹中的物理慢因素？**

本同步文档用于主线科研讨论与后续交接。权威顺序为**实际运行封存的配置、代码和回执 → 实验协议 → 既有汇总记录**。结果数值只读取已有文件；本次没有启动训练、特征提取、评测推理或重算既有实验指标。审计与目录治理本身不提高科学证据等级。

**审计与目录整理已完成。** 整理实绩、完整文件映射和删除清单见第5节。封存 `reports/` 与活跃锁定依赖 `work/hcf_stable_id_20261027/` 必须保留原位；即使添加旧路径符号链接，移动后 `Path.resolve()` 仍会改变封存来源键，可能破坏回执核查。标准 `results/low_dim_ar1/` 已以分支链接提供导航。这是保护证据链的兼容决定，不是漏整理。

## 1. 全局进度快照

当前低维主线已有独立新模型／轨迹上的成功范围和失败边界，但自然误差方向与恢复精度尚未完全分开，也没有建立独立于分差的新动态损害机制。视觉扩展已交付实现，用户告知服务器运行中，本地尚无实际回包。不能把“运行中”升级为“真实图像实验已完成”。

本表只采用七种等级：待设计／已实现／通过工程检查／已完成真实运行／获得局部现象／已排除竞争解释／停止。“停止”表示按范围收口，不表示没有结果或所有相关方向被否定。

### 1.1 完整性与权威来源

- 对 A1 local、A2 strong/weak、A3、A4 panel/forecast、B1/B3、B2 的 13 份 DESIGN_LOCK / RECEIPT / FORECAST 中 `sources` 核查，共232个唯一源/封存资产，**全部匹配、零缺失**。完整明细已保存到 `docs/audit/20261116/SEALED_SOURCE_CHECKS.json`。
- 对 A4 16、A3 16、A2 32、B1/B3 64、B2 16，共144个封存预测NPZ逐文件核对回执哈希，**全部匹配**。只读NPY header确认形状：A4 `(2048,4)`分数、`(2048,)`原/调整分差、`(2048,2)`top2；A3/A2/B1/B3 `(4096,4)`分数；B2 `(5,4096,4)`分数。
- 这些检查不重新验证数值统计正确性，也不把本次审计升级为独立科学复验。
- 历史脚本保存在 reports 内，引用原 work vendor。封存代码不能直接改路径；否则 sources 哈希失效。最终保留上述活跃锁定目录原位，以标准目录链接导航；不修改原封存代码。

### 1.2 低维实验的真实设置

四维平稳标准高斯 AR(1)，各轴 `z[t+1]=rho*z[t]+sqrt(1-rho^2)*eps`。观测是可逆三角非线性变换：`[z0+.55 sin(1.5z1), z1+.4(z2²−1), z2+.55 sin(1.5z3),z3]` 乘4×4归一化Hadamard。代码 `reports/lejepa_slow_rank_20261112/training/run.py:44`。作者 SIGReg 和 make_mlp_encoder 来自 `work/hcf_stable_id_20261027/v1/vendor/lejepa_identifiability`；不是自创LeJEPA。

A4 新训练种子20151–20154，训练 separated_wide `[.93,.918,.892,.88]` 与 crossed_wide `[.93,.918,.892,.76]`。评价 separated_wide 与 separated_narrow `[.93,.906,.904,.88]`。编码器 hidden64/layers3/steps3200/batch256/AdamW lr .003/SIGReg权重 .001/slices24；读出MLP宽128/steps3000/batch256/lr .001/拟合8192/VAL4096。正式来源 A4 panel/DESIGN_LOCK.json 内 config/original_encoder_config/original_probe_config；该文件sources匹配。

所有主线评分保持窗口193、lag4/16/32等权平方增量，选择最低两个命名物理因素，任何一个选错算一次。低维物理方差已知为1，预测分母为独立REF输出经验方差；不同于FourRoom未知总体方差、以共同物理REF定义答案。

### 1.3 各模块当前状态

| 模块 | 证据等级 | 边界 |
|---|---|---|
| 早期 HCF/Fine 方向迁移与理论分支 | 停止 | 保留受限反例及真实运行，11月5日技术报告停止判断仅属此阶段 |
| 人工受控误差方向对照 | 已排除竞争解释 | 在该受控构造中排除了仅由误差幅度解释；不等于自然误差已隔离 |
| A1 自然修正、交叉项与二次机制 | 停止 | 有实测局部现象，修正收益仍与恢复改善混杂；不再搜索残差 |
| A4 额外校准方法比较 | 停止 | 8个新模型真实运行完成；两规则0错误，无法区分校准额外价值 |
| A3 固定分差规则 | 获得局部现象 | 当前8模型、两过程、原REF下16/16达到门槛；不是通用证书 |
| A2 读出预算边界 | 停止 | 预算减少后风险上升；预定恢复差异仅4/8达到，未隔离R²因果 |
| B1 耦合外推 | 获得局部现象 | 唯一耦合 .12 保持状态恢复，原规则16/16达标；无普遍耦合规律 |
| B2 lag与多lag平均 | 停止 | 预定改善失败；谱间隔/扰动诊断并非通用选lag法 |
| B3 时间尺度 | 获得局部现象 | 状态恢复相同仍有阈值覆盖边界；未建立独立于分差的新机制 |
| 11/16 A1、B3旧数据分析 | 停止 | 探索性重分析完成，没有新增模型、轨迹或可靠性条件 |
| 官方FourRoom两层H-JEPA实现 | 通过工程检查 | 历史记录有7项数值单测和语法检查；服务器实际执行到哪一步待回包 |
| H-JEPA回包后的机制分析方案 | 已实现 | 研究记录第11.5节已固定顺序与停止条件，尚未执行 |
| 文献归属核对与研究记录 | 已实现 | 已区分原论文成果、复用工具与本地实证；不是新颖性证明 |


## 2. 真实图像扩展 H-JEPA 专项核查

### 2.1 实际证据与证据等级

- 当前官方 FourRoom 图像实验只找到 `archive/releases/hjepa_slow_selection_20261115/` 的8个交付文件与独立 `.git`，没有 `assets/`、`results/`、运行日志、`provenance.json`、`setup.json`、`forecast_receipt.json` 或该实验的预测/真值包。
- 全仓（包含隐藏、忽略文件）按 H-JEPA/FourRoom/预测/回执名称检索；另只读取20个 ZIP/TAR 的成员目录，无 `fourroom_01`、`fourroom_02`、`hjepa_slow_selection`、`test_prediction.npz`、`forecast_receipt.json`、`reference_truth.npz` 成员命中。没有解压或执行内容。
- 用户称服务器正在运行，但本地没有回包，无法核实执行到数据采集、编码、读出或评测哪一步。准确状态：**通过工程检查；正式图像运行待回包**。工程检查依据 `VALIDATION.md:9-14` 的历史记录，不是本轮重新运行结果。
- `archive/outputs/HCF_STABLE_ID_RUN_20261027_V1/hjepa/` 中 AR/FHN 模型是旧四维适配，不是此次图像 H-JEPA。旧 `formal_default_v2/STATE.json` 为 completed=0,total=21；`ROOT_DISPOSITION.json` 为 deferred_gpu_at_user_instruction、formal_training_started=false。不能混入本次12项视觉科学验收。
- H-JEPA 独立仓库唯一提交 `145daec9d5dab2c389fb9c910d6a91d99cd0d6a4`，2026-11-15T18:02:53+08:00，8个文件同时加入，当前干净。父仓库唯一提交 `6eb6de6` 为2026-10-19；父仓库把整个实验目录视作未跟踪，不能以父 Git 历史声称近期视觉代码逐次变更。

### 2.2 精确配置（仅交付设置，未冒充运行事实）

`config.json` 与 `run.py` 一致：种子42/43/44，采样种子2026111541，划分种子20261115，3,776 episodes，TRAIN256/VAL64/AUDIT128/REF256/CAL1024/TEST2048。每段401帧；TRAIN/VAL每段32帧，合计8192/2048带标签状态。其余角色每段一个193帧窗口。四因素为agent_x,agent_y,distractor_x,distractor_y。

两层各用同一时刻RGB逐帧编码；`run.py:86-114` 检查两层temporal_window_size=1。读出 `d→128→128→4`，SiLU，AdamW，3000steps，batch256，lr=.001，weight_decay=.0001；seed=模型seed×10+layer，每层只有一次拟合。TRAIN标准化，VAL选择最佳模型；`run.py:117-155,216-218`。所有特征只在内存，`run.py:240,254`删除，没有latent缓存。

原验收：逐轴AUDIT R²≥.85、观测coverage≥.35、12项Bonferroni单侧总体95%选择性错率上界≤.01。单项alpha=.05/12。科学统计单位为TEST episode窗口，不是193帧或6个模型-层复制。

| 实际运行环节 | 本地可核查状态 |
|---|---|
| 3,776 episodes采集、原始数据与数据哈希 | 待回包，实际执行进度未核实；仅有预定配置，不能确认服务器进度 |
| 官方3个模型的provenance与安装环境 | 待回包，实际执行进度未核实；静态有核验逻辑，无实际provenance和setup回执 |
| 两层 encoder 编码 | 待回包，实际执行进度未核实；工程单测不等于真实图像前向 |
| 六个 d→128→128→4 readout 拟合 | 待回包，实际执行进度未核实；本次probe权重和拟合日志尚未归档 |
| AUDIT、CAL、REF与TEST评分 | 待回包，实际执行进度未核实；下表无实测数值，不能填零代替缺失 |

“待回包”表示本地缺少执行证据，不否认用户告知服务器已经开始运行；本轮没有访问服务器。

### 2.3 AUDIT表：6个单元均待回包

| seed | layer | agent_x R² | agent_y R² | distractor_x R² | distractor_y R² | 四轴全≥.85 | 归一化状态MSE |
|---|---|---|---|---|---|---|---|
|42|1|待回包|待回包|待回包|待回包|待回包|待回包|
|42|2|待回包|待回包|待回包|待回包|待回包|待回包|
|43|1|待回包|待回包|待回包|待回包|待回包|待回包|
|43|2|待回包|待回包|待回包|待回包|待回包|待回包|
|44|1|待回包|待回包|待回包|待回包|待回包|待回包|
|44|2|待回包|待回包|待回包|待回包|待回包|待回包|

`audit.json`只保存四轴R²和eligible，`audit.npz`保存预测和真值。代码R²的分母为AUDIT真值逐轴方差，不是REF方差。没有显式保存归一化状态MSE字段；按AUDIT方差归一化的MSE在定义上等于1−R²，而用共同REF归一化是另一口径。当前无回包，也遵守不重算结果红线，不能伪造该指标。

### 2.4 原12项选择性验收：均待回包

|seed|layer|规则|阈值|实际coverage/≥35%|错误/作答数|CP上界/≤1%|
|---|---|---|---|---|---|---|
|42|1|own_cal|待回包|待回包|待回包|待回包|
|42|1|level1_cal|待回包|待回包|待回包|待回包|
|42|2|own_cal|待回包|待回包|待回包|待回包|
|42|2|level1_cal|待回包|待回包|待回包|待回包|
|43|1|own_cal|待回包|待回包|待回包|待回包|
|43|1|level1_cal|待回包|待回包|待回包|待回包|
|43|2|own_cal|待回包|待回包|待回包|待回包|
|43|2|level1_cal|待回包|待回包|待回包|待回包|
|44|1|own_cal|待回包|待回包|待回包|待回包|
|44|1|level1_cal|待回包|待回包|待回包|待回包|
|44|2|own_cal|待回包|待回包|待回包|待回包|
|44|2|level1_cal|待回包|待回包|待回包|待回包|

### 2.5 评分、阈值与封存顺序

- `metrics.py:39-61`：4/16/32步平方增量各自平均再等权。真实分数除共同物理REF方差；预测分数除各读出在同一REF上的预测方差；variance均默认ddof=0。排序stable；精确平局固定因素顺序。分差是预测第3名−第2名；top2集合不同为错选。
- `run.py:203-205,229-238`：共同REF物理真值，读出各自REF预测，CAL只读取预测，阈值为预测分差中位数。own_cal每层独立；level1_cal将同种子第一层阈值用于两层。第一层两规则相同，原12项仍保留。
- `metrics.py:78-94`：gap≥threshold作答；CP上界beta.ppf(1−alpha,k+1,n−k)，k=n时上界1（包括未作答n=0）；coverage是样本比例，不是总体coverage置信下界。
- 静态顺序正确：`run.py:241-243`阈值保存→245-254全部TEST预测→255-256保存预测与阈值SHA256回执→257读取TEST真值→258保存真值→263评分。只能证明交付代码顺序，**没有实际回执可做运行核验**。
- provenance静态实现：`run.py:174-197`核上游commit/dirty、checkpoint SHA256与setup回执，记录环境、设备、HDF5 SHA256。正式输出配置与脚本快照在`run.py:312-325`保存。未收到实际provenance，不能声称官方权重/数据哈希已验真。

### 2.6 新增对照、文档差异和边界

- 图像实验**没有新增联合线性解混baseline、R²匹配干预或额外诊断运行**。联合线性解混及.005匹配是旧低维A1分析，不得算成H-JEPA结果。
- `docs/RESEARCH_RECORD.md:581-642`是回包后的诊断计划，尚未执行。保存完整TEST预测可支持未来同窗MSE与动态误差诊断；CAL真值未导出，latent features未保存；缺独立重复读出、独立REF重复及新机制确认集。
- 相比官方环境，action_repeat_prob由.05改0（`config.json:5`，`run.py:41`，README:100）；这是评价采样分布变化，不能写成纯工程变化或精确复现官方评价。
- 视觉真实评分使用有限REF方差，低维AR1使用已知总体方差；不能跨两个分支直接解释绝对错率差异为视觉/层级作用。
- README第5节“完整服务器实验尚未运行”及VALIDATION为11月15日交付状态；11月16日研究记录已改成“用户告知运行中但未回包”。这是证据时间层次差异，不应修改历史交付快照来抹平。
- `setup.sh:43-45`未显式列出stable-worldmodel，57/61行import该模块。当前没有官方源码assets可确认该模块是否由PYTHONPATH指向的固定上游源码直接提供，也无实际安装日志。因此记为**独立环境依赖核验缺口**，不能未经核验就断言必然报错。正式包回传时检查environment和安装/启动日志；本轮不安装或联网下载。


## 3. 近期代码与协议变更审计

### 3.1 Git与近两日文件变化

- 根仓库 `origin=https://github.com/Hang-276/HI-JEPA.git`，仅一个提交：`6eb6de6`，提交时间 `2026-10-19T17:55:40+08:00`，题目 `Implement HI-JEPA experiments and portable launch scripts`。
- 根工作树修改了 `README.md`、`SHA256SUMS`、`scripts/_experiment.py`、`src/hijepa/config.py`、`src/hijepa/losses.py`；其余新实验、研究文档、历史输出大部分未跟踪。因此不能用根 Git 日志宣称最近两日没有工作，也不能将工作树与唯一提交的差异全部归为最近两日。
- 现有修改中的真正条件变化：`coarse_floor_mode=spectral_barrier` 软谱屏障及参数校验；`HIJEPA_MATRIX` 环境变量可改配置入口。它们不是单纯路径重构。这些文件 mtime 属于较早工作，不能标为近两日新增。
- `archive/releases/hjepa_slow_selection_20261115/` 是独立、干净的嵌套 Git 仓库，远端 `https://github.com/Hang-276/hjepa-slow-selection.git`，提交 `145daec`，时间 `2026-11-15T18:02:53+08:00`，题目 `Add standalone H-JEPA slow-selection server experiment`。
- 其他嵌套 `.git`：`work/hcf_method_freeze_20261025/flowdirect_branch_v1/vendor/leap_official/.git`；`outputs/PaperDesk_public_20261030/PaperDesk/.git`。保留，不当缓存删除。

按文件 mtime（不是 Git 变更证据）统计 2026-11-15 00:00 起：H-JEPA目录8文件；docs中的研究记录/TODO/文献归属3文件；A4 57文件、A3 26、B2 25、A2 65、本地收口10、B1/B3 75、A1 joint 18。mtime只能表明最近写入，不能推导运行成功或未变更旧协议。

H-JEPA当前代码 mtime：metrics 11-15 17:12:28；test_metrics 17:14:27；config 17:28:09；setup 17:51:32；VALIDATION 17:54:08；README/run 18:01:10。研究记录等为11-16更新。

### 3.2 条件变化与旧结果的可比性

近两日范围为2026年11月15日至16日，不能把更早未提交变化全部算成这两天新增。

| 推进 | 改动性质 | 与旧结果的比较边界 |
|---|---|---|
| A4新模型与读出 | 新训练种子及模型，不是工程重构 | 保持物理评分，用于独立模型范围验证 |
| A3固定阈值 | 科学作答规则变化 | A4按测试批次排名取20%，A3为固定数值阈值，不可当成同一覆盖协议 |
| A2降低拟合标签 | 8192→64拟合预算，VAL不变 | 误差方向与参考方差也会变化，不能只归因R² |
| B1／B3耦合与时间尺度 | 改变轨迹联合分布 | 保持标准高斯状态边缘与冻结读出；不是纯工程变化 |
| B2解混lag | 校准算子变化，评价lag不变 | 多lag仅等权矩阵平均，不是一般联合对角化 |
| A1参考及二次机制 | 有限范围的诊断和独立新轨迹验证 | 使用真值的诊断不能作为可部署方法 |
| 11月16日A1／B3收口 | 旧TEST探索性分析 | 未新增模型或改变旧门槛；本次审计不再次执行这些分析 |
| 官方图像H-JEPA交付 | 新视觉评测实现 | 动作重复率.05→0改变评价采样分布，不是与官方原分布完全等价的纯修复 |
| 文献、研究记录与回包方案 | 来源、解释和执行边界整理 | 不是新增科学结果 |

近期封存配置中的窗口193及评价lag4/16/32保持；读出预算、解混lag、动力学等有意变化均按分支记录，不静默合并。

### 3.3 应明确列出的差异／缺口

1. 整理前根README把2026-10-27 C1L/FlowDirect称作当前方法；这与11月主线LeJEPA + 视觉H-JEPA不一致。旧README已原样保存到 `archive/legacy_metadata_20261116/README.md`，新根README已改为当前入口。
2. `reports/stage_a_plus/STEP7_FINAL_DECISION.md`称11月5日技术报告收尾，不等于当前整个研究取消；是早期Fine方向迁移分支的历史决定。
3. A1 natural +6pp并非“恢复质量匹配下方向独立作用”；11月16日实际匹配结果0/28，应以最新解释限制旧表述。不要覆盖原结果。
4. A2 预定至少6/8恢复差异，实际4/8；不能把已完成运行说成预定操纵成功。
5. A4原20%规则是测试批次内排名选择，与A3/A2固定数值阈值单窗作答不同；不能直接当同一覆盖协议结果相互复验。
6. B3 变慢风险上界未过不等于真实风险已超过1%；新CAL中位数只改作答，不修复表示。
7. B2 multi为四矩阵等权平均后一次特征分解，不是一般多lag联合对角化算法。
8. 最新低维锁/配置哈希未见不一致。仅本轮所核查13份source锁与144份forecast，不应把它表述成所有历史产物均已科学复核。


## 4. 统一格式交接条目

下述数值读取既有封存摘要和报告，没有重算指标。逐组合详细数据保留在各 `SUMMARY.json`。百分比、百分点与错误计数分别记录；共享轨迹的模型评价次数不是独立样本数。

### 1. A1 参考误差来源诊断
- 【问题】旧窗口扣交叉项后反而变差，源于残差总体方差还是有限REF抽样？
- 【状态】获得局部现象。
- 【一句话结论】指定窗口主要受这次REF偏差影响，全体窗口的残差总体方差项却更大；单窗口不代表整体。
- 【协议与配置】`reports/lejepa_a1_joint_20261115/reference/PROTOCOL.md`、同目录run.py、SUMMARY.json。14冻结强MLP，两份各131072 iid状态，master2026111501/02，role61，原CAL/REF/旧TEST保持。
- 【结果】首轮窄过程 separated_wide_1942 row18，参考分差−.032866；同轴协方差+.001259（近似95% CI −.000147,+.002664），残差总体方差−.003191（−.003367,−.003015），有限REF项−.030934（−.032335,−.029533）。全体跨界RMS初轮/复验：残差总体方差 .03891/.03855；有限REF .01960/.00965。
- 【限制】区间只反映新iid总体矩估计；原窗口固定，真值诊断不是可见规则。
- 【证据入口】`reports/lejepa_a1_joint_20261115/reference/{RESULTS.md,SUMMARY.json,PROTOCOL.md,run.py}`。
- 【待办与卡点】无必要重做；需新假设才能重开。

### 2. A1 局部二次残差验证
- 【问题】状态低阶二次项能否解释动态交叉项，并改善慢集？
- 【状态】停止。
- 【一句话结论】解释部分交叉项，但慢集准确率几乎不变，收益集中于两个模型。
- 【协议与配置】`reports/lejepa_a1_joint_20261115/local/`；14冻结强MLP，2轮×2过程，每轮32768标签拟合状态、每过程1024新窗口；master2026111521/22，角色101/102；DESIGN_LOCK与CAL_RECEIPT锁定。
- 【结果】全部二次项相对CAL均值交叉差MSE下降15.91%/13.96%，单侧95%下界11.39%/8.70%；同轴加权项15.02%/14.22%，下界10.87%/10.27%；全部二次扣项top2仅+.038/+.021pp，原准确93.872%/94.496%，约88.4%/88.3%的MSE降幅来自两个模型。
- 【限制】使用真实物理TEST状态，不能部署；区间条件于固定模型/CAL/REF。
- 【证据入口】`reports/lejepa_a1_joint_20261115/local/{PROTOCOL.md,DESIGN_LOCK.json,CAL_RECEIPT.json,SUMMARY.json,RESULTS.md}`。
- 【待办与卡点】停止二次项扩展与新残差预测器。

### 3. A4 新模型与额外校准
- 【问题】带标签校准的分差高估修正是否优于直接分差？
- 【状态】停止。
- 【一句话结论】新8模型已真实训练，两规则在约20%作答时均0错，未验证额外校准收益。
- 【协议与配置】`reports/lejepa_a4_20261115/config.json` SHA256 `868e79d231b58d2ac0ebdbaa4ee58775557048b9f00f3690462e74db38ddcf33`；REF32768/CAL2048/TEST2048，每模型×2过程；见panel/RECEIPT。
- 【结果】四轴最低AUDIT R²范围 .9952011702307183–.9963637632483042。每规则0/6544作答、19.97%，共同6339、各独有205均对；全答1430/32768错。`primary_gate.passed=false`，增益0，退化bootstrap[0,0]不能作为零风险保证。32768组合评价对应16384独立物理轨迹。
- 【限制】非通用阈值；校准优势不能区分。
- 【证据入口】`reports/lejepa_a4_20261115/panel/{protocol.json,DESIGN_LOCK.json,RECEIPT.json,runs/}`；`reports/lejepa_a4_20261115/{FORECAST_LOCK.json,NEW_FORECAST.json,new_forecast/,SUMMARY.json}`。
- 【待办与卡点】按原门停止，不为发现收益放宽作答比例。

### 4. A3 固定分差阈值
- 【问题】仅CAL预测固定阈值能否在独立新轨迹上可靠作答？
- 【状态】获得局部现象。
- 【一句话结论】g50在本面板所有16组合达到覆盖35%/风险1%门。
- 【协议与配置】`reports/lejepa_a3_gap_20261115/config.json` SHA256 `1c6454fee0148f8ef307e2fef41e831b5a005d8ab57d29db11bd689669b044f3`；TEST4096/组合，role613；3阈值×16组合=48项Bonferroni。
- 【结果】g20=.37577294958683716：0/13389错、20.4299927%作答、最高上界.8776323%、0/16全门（覆盖不足）；g50=.17748381448816608：3/33355错、50.8956909%、最坏上界.4512633%、16/16；g80=.05942476563385046：206/52830错、80.6121826%、最坏上界1.3702124%、11/16。65536模型窗口来自32768独立轨迹。
- 【限制】固定模型/读出/参考/动力学条件；合计不作独立二项CI。
- 【证据入口】`reports/lejepa_a3_gap_20261115/{THRESHOLDS.json,DESIGN_LOCK.json,FORECAST.json,forecast/,SUMMARY.json,score.py}`。
- 【待办与卡点】等待视觉验证，不追加原低维任务。

### 5. A2 拟合预算
- 【问题】降低标签量时阈值迁移是否保留？
- 【状态】停止。
- 【一句话结论】覆盖相近而风险增加，但恢复反差门未达，不能给R²因果门槛。
- 【协议与配置】`reports/lejepa_a2_readout_20261115/config.json` SHA256 `a10c0c24415f7a902f9c59969cf332eb7efb76f2c6d7d1b19520d67e74b37f58`；8编码器，8192/64拟合，同VAL4096；新AUDIT16384，TEST4096，32项校正，阈值固定g50。
- 【结果】strong6/32934错、50.2532959%、最坏上界.4268684%、16/16；weak145/32784错、50.0244141%、最坏上界2.5064724%、11/16。weak最低轴R² .96838–.99293；预定下降至少.01仅4/8，要求6/8。共同作答且gap差≤.02子集9822次，1次仅weak错。
- 【限制】读出误差方向与REF方差同时变化；子集不是控制R²的因果实验。
- 【证据入口】`reports/lejepa_a2_readout_20261115/weak/{protocol.json,DESIGN_LOCK.json,RECEIPT.json,runs/}`；`reports/lejepa_a2_readout_20261115/{AUDIT_GATE.json,FORECAST.json,SUMMARY.json}`。
- 【待办与卡点】停止追加标签剂量。

### 6. B1/B3 部署耦合与时间尺度
- 【问题】状态边缘分布/编码器/读出/REF保持时，耦合与速率是否破坏规则？
- 【状态】获得局部现象。
- 【一句话结论】本次耦合与变快可沿用旧阈值；变慢覆盖下降，重设阈值恢复覆盖但仅14/16风险上界通过。
- 【协议与配置】`reports/lejepa_b_transfer_20261115/config.json` SHA256 `99e0e96d335bc1e49b2638456f08a6f5fc33eb41442f25d735a3bbe70bab4a22`；A=diag(rho^exponent)+coupling e2e3^T，Q=I-AA^T；coupled=.12，slower exponent=.25，faster=4。4种×16组合×2规则=128项；CAL2048、TEST4096、共同AUDIT16384；角色811/813/817。
- 【结果】固定阈值：baseline4/33125错、coverage50.5447388%、最坏上界.4954986%、16/16；coupled7/31867、48.6251831%、.6040623%、16/16；slower15/20934、31.9427490%、1.2633522%、0/16；faster6/27450、41.8853760%、.6150966%、16/16。重校准：baseline4/32527、49.6322632%、.5045049%、16/16；coupled7/32868、50.1525879%、.5862720%、16/16；slower52/32690、49.8809814%、1.1739640%、14/16；faster7/32758、49.9847412%、.5115410%、16/16。
- 【限制】slower两未过组9/2048与7/1978，观察均<1%，只是同时上界未过；共享创新/模型评价不可当独立。B1整体未出现需额外限制的失败，不是证明任意耦合安全。
- 【证据入口】`reports/lejepa_b_transfer_20261115/{DESIGN_LOCK.json,THRESHOLDS.json,CAL_GAPS.npz,FORECAST.json,forecasts/,SUMMARY.json,THEORY.md}`。
- 【待办与卡点】按原有限范围停止。

### 7. B2 校准lag
- 【问题】扩大理想谱间隔或平均多lag是否改善解混及慢集？
- 【状态】停止。
- 【一句话结论】本轮预定方案失败，估计扰动和物理轴恢复一起恶化，不能只看理想间隔。
- 【协议与配置】`reports/lejepa_b2_lag_20261115/config.json` SHA256 `56f39df6efd9e194e54c439a4e6d81d49e4aa320f3df1414ca71f689295090ec`；8编码器，CAL1024/NAME256×±.15/REF32768/AUDIT16384/TEST4096；lag1/4/16/32与等权矩阵multi；CAL bootstrap64，窗口bootstrap4096，两主比较校正。
- 【结果】宽/窄top2：lag1 82.43/79.01%，lag4 82.02/73.04%，lag16 79.98/70.81%，lag32 71.29/59.44%，multi80.31/66.02%。窄主差lag16 −.082000732421875、单侧下界−.0872802734375；multi−.1298828125、下界−.135345458984375。逐seed lag16 −.00061035,+.00134277,−.16394043,−.16479492；multi −.00158691,−.20471191,−.08959961,−.22363281，均未过主门。
- 【限制】多lag仅本轮等权平均；谱间隔/扰动比值解释不了所有方法顺序，无通用自动选lag。
- 【证据入口】`reports/lejepa_b2_lag_20261115/{DESIGN_LOCK.json,FORECAST.json,forecast/,SUMMARY.json,run.py}`。
- 【待办与卡点】停止扩大lag网格。

### 8. 11/16 A1 旧结果收口分析
- 【问题】自然方向修正+6pp能否独立于恢复精度改善解释？
- 【状态】停止。
- 【一句话结论】原收益存在，但恢复质量匹配对照为空，未建立独立方向效应。
- 【协议与配置】`reports/lejepa_local_closeout_20261116/a1/ANALYSIS_PLAN.md`，analyze.py与SUMMARY输入散列；仅用旧自然修正五分支预测、校准元数据、AUDIT及REVEAL，无新模型或轨迹。最大逐轴R²差≤.005描述筛查。
- 【结果】primary/replication +6.167603/+5.838013pp；活跃28组中恢复MSE改善28/28、27/28；与base/reverse/wrong_donor/joint_linear恢复匹配均0/28。第二轮一个组合受影响轴R²降.0312而选取增16.99pp。joint_linear75.20/74.58%准确，高于定向66.40/65.82%。
- 【限制】旧TEST探索性；每轮65536评价来自2048共享创新行；不能把例外作普遍规律，也不能说收益全由MSE解释。
- 【证据入口】`reports/lejepa_local_closeout_20261116/a1/{ANALYSIS_PLAN.md,analyze.py,SUMMARY.json,RESULTS.md}`；原 `reports/lejepa_natural_direction_repair_20261113/`。
- 【待办与卡点】停止残差搜索；存档缺逐窗口状态预测，不补前向。

### 9. 11/16 B3 旧结果收口分析
- 【问题】变慢效应是否只有固定阈值回答更少？
- 【状态】停止。
- 【一句话结论】匹配约50%作答后错率仍升，但全量窗口的绝对分差配平不支持普遍额外损害。
- 【协议与配置】`reports/lejepa_local_closeout_20261116/b3/ANALYSIS_PLAN.md`、analyze.py/SUMMARY散列；旧CAL给20/50/80%阈值；原gap箱；配对创新行bootstrap1000，seed2026111603；仅旧3种速率，未做新前向。
- 【结果】50%处slower−base错率差 .0014677257606060816，探索95%区间[.0009281719562894315,.002005545339701309]；faster差 .00009071349992586619，区间[−.00015495736596245896,.0003348977126418633]。绝对预测gap配平全窗口差−1.099846pp；真实gap−1.160267pp；CAL分位排名+.985121pp。gap<.06比例19.56%→29.43%。
- 【限制】全量配平非50%作答子集，不解释尾部全部差异；旧TEST探索，非独立新条件。
- 【证据入口】`reports/lejepa_local_closeout_20261116/b3/{ANALYSIS_PLAN.md,analyze.py,SUMMARY.json,RESULTS.md}`；`reports/lejepa_local_closeout_20261116/AUDIT.md`。
- 【待办与卡点】收口，不加新预测器。


### 10. 11月15日官方视觉H-JEPA交付

- 【问题】低维恢复到慢因素选取的条件能否迁移至官方视觉两层表示；主要竞争解释是读出质量、物理评分和作答窗口不同。
- 【状态】通过工程检查。
- 【一句话结论】实现与固定协议齐全，尚无可审计的正式视觉数值；用户报告服务器运行中不替代回包。
- 【协议与配置】`archive/releases/hjepa_slow_selection_20261115/config.json` SHA256 `a35678c9ce7ac22add97b476adec604d5f9a2a791e497d649a89c5b9d2acb761`；3776episodes，3种子×2层×2规则，T193、lag4/16/32、逐轴R²≥.85、coverage≥35%、Bonferroni CP上界≤1%。完整配置见第2节。
- 【结果】6组逐轴恢复及12项规则全部待回包；历史VALIDATION记载7项数值单测及语法检查通过，没有完整模型前向证据。
- 【限制】单次读出、单组REF、无新机制确认集；CAL真值及latent不随包保存；动作采样与官方默认不同。
- 【证据入口】`archive/releases/hjepa_slow_selection_20261115/{config.json,run.py,metrics.py,README.md,VALIDATION.md,setup.sh}`，独立Git提交145daec；当前没有实际视觉运行结果；新入口未来输出到 `results/fourroom_hjepa/`。run.py SHA256 `0e6abc8d92d2193d42c4471cf8790dd547a5448064399db683e9145b9758ac3d`；metrics.py `ede7d23d218789f9ff66a362fed128d34a5290f4ace907bf7771fed51d219c8a`。
- 【待办与卡点】只待完整结果目录及实际代码、provenance与receipt。历史记录下载遇网络／TLS问题，当前无新的错误日志；不下载权重、不补跑。实际环境中的stable_worldmodel来源待核。

### 11. 11月16日文献与成果归属核对

- 【问题】哪些结论来自参考论文或既有SFA／源分离／风险控制工具，哪些是独立执行的实验，哪些可能形成新增认识？
- 【状态】已实现。
- 【一句话结论】已建立归属表和候选缺口；尚未证明核心科学新意已经成立。
- 【协议与配置】优先已发表或已录用论文，预印本只作参考；区分阅读全文、仅核摘要和发表身份。配置不适用；文献出处与访问限制保存在报告中。
- 【结果】推荐围绕有限窗口物理慢因素决策的适用条件与边界组织主线；自然误差干预仍有解释缺口，简单分差与分位数校准不能当新算法。没有新增模型、样本或实验数值。
- 【限制】LeJEPA正交可辨识、Hermite谱分析及相关条件来自原作者；一般排序间隔、误差恒等式、CP／Bonferroni不是原创定理。视觉范围扩大不能自动补齐新颖性。
- 【证据入口】`docs/LITERATURE_GAPS_AND_POSITIONING_20261116.md`第5.1、6、7、8节，以及其中逐项原始来源。
- 【待办与卡点】缺少具有额外解释或预测价值的具体机制证据；不为凑方法重新开启残差搜索。

### 12. 11月16日研究记录、TODO与H-JEPA回包分析边界

- 【问题】怎样保留真实研究经过，避免把待回包、探索性分析或局部成功混成论文贡献？
- 【状态】已实现。
- 【一句话结论】研究记录和原A1／B3条目已纳入本地收口结论，回包分析与停止条件已写明；方案尚未执行。
- 【协议与配置】研究记录第11.4—11.5节保持原12项视觉验收和门槛；先核包，再同窗口比较，再做限定的三类误差诊断，最后决定是否需要最小对照。日期是制定时间，不是已完成新机制验证。
- 【结果】没有新增科学数值；完成的是归属、解释边界和执行条件整理。旧TODO部分总项未勾选不表示A4新模型验证或本轮文献核对没有发生。
- 【限制】原TEST用于既定规则评价，其余机制发现不能直接当独立确认；目前没有另行封存的新机制确认集。
- 【证据入口】`docs/RESEARCH_RECORD.md`、`docs/ICML_RESEARCH_TODO.md`；`reports/lejepa_local_closeout_20261116/{RESULTS.md,AUDIT.md}`。
- 【待办与卡点】只有具体解释能改变一句论文主张且最小对照能区分竞争解释时，才提出有限补充；无额外价值、无法区分或验证失败即收缩并停止，不追加样本直到原统计门通过。

### 13. 11月16日本次审计与目录治理

- 【问题】把当前进展、历史快照与原始证据分清，并在不损坏封存来源的前提下建立标准入口。
- 【状态】通过工程检查。
- 【一句话结论】完成全仓文件清点、近期证据核查和归档；保留原封存路径，新图像入口不改变科学配置与评分逻辑。
- 【协议与配置】本次红线为不训练、不推理、不重算结果、不删除核心证据。逐文件记录见 `docs/audit/20261116/FILES_BEFORE.jsonl`，操作表见 `REORGANIZATION.json`；科学配置沿用原hash，新代码指纹另见 `NEW_SOURCE_HASHES.json`。
- 【结果】完成33项路径移动、6份原件备份；仅删除1,039个明确缓存文件，共23,468,044字节，移除177个已空缓存目录。静态检查28个Python、11个Shell、26个JSON及35个链接通过，232个来源hash再次匹配。两个新入口的轻量help通过，未加载研究模型。完整保存核验见第5节。
- 【限制】本机没有SciPy与PyTorch；未运行固定数组数值单测、依赖安装、服务器smoke、低维准备器或模型计算。新环境声明是工具依赖，不是旧实验的完整环境锁。原数据与旧脚本的完整科学复现仍需其封存环境及输入。
- 【证据入口】`src/slowrepr/`、`scripts/fourroom_hjepa/`、`scripts/low_dim_ar1/prepare_a4.py`、`scripts/check_repository.py`、`docs/audit/20261116/`；旧交付在 `archive/releases/hjepa_slow_selection_20261115/`。
- 【待办与卡点】等待服务器原包；stable_worldmodel来源、完整运行与新入口服务器验证仍待核。不得将目录治理或help成功算作视觉科研进展。

## 5. 仓库整理变更表

本节记录已实际完成的操作。全仓清点覆盖48,449项非Git文件／文件链接，普通文件总量33,733,392,715字节；Git元数据单独保留。历史产物全部进入文件级完整性核查，但科学内容的重点审查范围是近两日主线及其封存依赖，**不声称逐个重做或科学复核所有历史实验**。

### 5.1 新旧核心路径映射

| 原路径 | 整理后路径 | 处理与兼容方式 |
|---|---|---|
| `reports/lejepa_*/` | 实目录不变；导航 `results/low_dim_ar1/lejepa_*/` | 保留Path.resolve与封存来源键，源码/结果字节不改 |
| `reports/stage_a_plus/` | 实目录不变；导航 `results/legacy_fine/` | 历史Fine证据完整保留 |
| `work/hcf_stable_id_20261027/` | 原位 | 活跃封存依赖，不改名 |
| 原独立图像交付 | `archive/releases/hjepa_slow_selection_20261115/` | 原8文件与独立Git保留，旧experiments路径链接 |
| 图像run.py（新工作副本） | `src/slowrepr/fourroom_hjepa.py`＋`scripts/fourroom_hjepa/run.py` | 只调整路径、导入、帮助和代码快照；不冒充旧服务器运行源码 |
| 图像metrics.py／config.json（新副本） | `src/slowrepr/metrics.py`／`configs/fourroom_hjepa/default.json` | 与原交付逐字节一致 |
| 新图像结果目录 | `results/fourroom_hjepa/<name>/` | 仅未来输出位置，当前无正式视觉结果 |
| 低维五个主config.json | `configs/low_dim_ar1/` | 链接封存原件，不生成新科学参数 |
| 根旧配置 | `configs/legacy_hijepa/` | 原路径保留文件链接 |
| 根README与旧环境元数据 | `archive/legacy_metadata_20261116/` | 保留旧原件；新README、pyproject、requirements明确分支环境 |
| 旧src/hijepa与scripts源码 | 原位 | 整理前已有未提交修改保留，标为历史兼容实现 |

根包由旧HI-JEPA打包声明改为当前 `slowrepr` 分析工具包，最低Python3.11、NumPy/SciPy基础依赖；这改变安装入口，不改变任何已运行实验环境。旧声明另存。官方图像仍使用独立Python3.11／CUDA环境。新H-JEPA的10个核心函数AST一致、配置和metrics字节一致；主流程仅路径及封存清单变化，见 [重构复核](audit/20261116/HJEPA_REFACTOR_REVIEW.md)。

### 5.2 实际归档快照及移动清单

| 原路径 | 实际新路径 | 原位置兼容链接 |
|---|---|---|
| `outputs` | `archive/outputs` | 保留 |
| `runs` | `archive/legacy_runs` | 保留 |
| `diagnostics` | `archive/legacy_checks/diagnostics` | 保留 |
| `smoke` | `archive/legacy_checks/smoke` | 保留 |
| `validation` | `archive/legacy_checks/validation` | 保留 |
| `HCF-JEPA-Next` | `archive/snapshots/HCF-JEPA-Next_20261021` | 保留 |
| `experiments/hjepa_slow_selection` | `archive/releases/hjepa_slow_selection_20261115` | 保留 |
| `work/hcf_baseline_report_20261027` | `archive/work_snapshots/hcf_baseline_report_20261027` | 保留 |
| `work/hcf_method_finalize_20261027` | `archive/work_snapshots/hcf_method_finalize_20261027` | 保留 |
| `work/hcf_method_freeze_20261025` | `archive/work_snapshots/hcf_method_freeze_20261025` | 保留 |
| `work/hcf_method_freeze_20261026` | `archive/work_snapshots/hcf_method_freeze_20261026` | 保留 |
| `work/hcf_multibench_c1l_20261027` | `archive/work_snapshots/hcf_multibench_c1l_20261027` | 保留 |
| `work/hcf_stable_id_20261028` | `archive/work_snapshots/hcf_stable_id_20261028` | 保留 |
| `README.md` | `archive/legacy_metadata_20261116/README.md` | 无；新文档使用新路径 |
| `SHA256SUMS` | `archive/legacy_metadata_20261116/SHA256SUMS` | 无；新文档使用新路径 |
| `Dockerfile` | `archive/legacy_metadata_20261116/Dockerfile` | 无；新文档使用新路径 |
| `.dockerignore` | `archive/legacy_metadata_20261116/.dockerignore` | 无；新文档使用新路径 |
| `DEEPSEEK_PROMPT.md` | `archive/legacy_metadata_20261116/DEEPSEEK_PROMPT.md` | 无；新文档使用新路径 |
| `DEEPSEEK_START_HERE.md` | `archive/legacy_metadata_20261116/DEEPSEEK_START_HERE.md` | 无；新文档使用新路径 |
| `实现与本机验证报告.md` | `archive/legacy_metadata_20261116/实现与本机验证报告.md` | 无；新文档使用新路径 |
| `docs/PROTOCOL.md` | `archive/legacy_protocols/PROTOCOL.md` | 保留 |
| `docs/V2_REVISION_PLAN.md` | `archive/legacy_protocols/V2_REVISION_PLAN.md` | 保留 |
| `docs/V3_RECOVERY_PLAN.md` | `archive/legacy_protocols/V3_RECOVERY_PLAN.md` | 保留 |
| `configs/base_formal.json` | `configs/legacy_hijepa/base_formal.json` | 保留 |
| `configs/base_formal_v2.json` | `configs/legacy_hijepa/base_formal_v2.json` | 保留 |
| `configs/base_formal_v3.json` | `configs/legacy_hijepa/base_formal_v3.json` | 保留 |
| `configs/base_formal_v3_modal.json` | `configs/legacy_hijepa/base_formal_v3_modal.json` | 保留 |
| `configs/base_local.json` | `configs/legacy_hijepa/base_local.json` | 保留 |
| `configs/evidence_policy.json` | `configs/legacy_hijepa/evidence_policy.json` | 保留 |
| `configs/matrix_baseline_v2.json` | `configs/legacy_hijepa/matrix_baseline_v2.json` | 保留 |
| `configs/matrix_formal.json` | `configs/legacy_hijepa/matrix_formal.json` | 保留 |
| `configs/matrix_formal_v2.json` | `configs/legacy_hijepa/matrix_formal_v2.json` | 保留 |
| `configs/matrix_local.json` | `configs/legacy_hijepa/matrix_local.json` | 保留 |

另有6份原件备份，用于保留本次编辑前的工作树内容：

| 被更新的工作文件 | 原件备份位置 |
|---|---|
| `.gitignore` | `archive/legacy_metadata_20261116/.gitignore` |
| `pyproject.toml` | `archive/legacy_metadata_20261116/pyproject.toml` |
| `requirements.txt` | `archive/legacy_metadata_20261116/requirements.txt` |
| `docs/RESEARCH_RECORD.md` | `archive/legacy_metadata_20261116/docs/RESEARCH_RECORD.md` |
| `docs/ICML_RESEARCH_TODO.md` | `archive/legacy_metadata_20261116/docs/ICML_RESEARCH_TODO.md` |
| `docs/LITERATURE_GAPS_AND_POSITIONING_20261116.md` | `archive/legacy_metadata_20261116/docs/LITERATURE_GAPS_AND_POSITIONING_20261116.md` |

`.gitignore`仅控制是否意外提交，不删除数据。archive主体和二进制不随普通Git提交携带；因此新克隆还需取得必要原始产物。根Git约39GB及三个历史嵌套Git都没有清理或重写。Git状态中的部分D代表已移动到归档，不代表实际删除证据。原工作树差异保存在 `docs/audit/20261116/WORKTREE_BEFORE.patch`；未进行commit、push或远端操作。

### 5.3 实际删除清单

| 删除类型 | 文件数 | 字节 | 完整逐文件清单 |
|---|---:|---:|---|
| macOS metadata | 3 | 30,732 | [原路径、实际删除路径、SHA256与原因](audit/20261116/DELETED_FILES.tsv) |
| Python bytecode cache | 1,036 | 23,437,312 | [原路径、实际删除路径、SHA256与原因](audit/20261116/DELETED_FILES.tsv) |

合计1,039文件、23,468,044字节；另移除177个已经为空的 `__pycache__` 目录，目录清单在 `REORGANIZATION.json`。没有删除任何.bak／_copy／_old副本、压缩包、空科研包文件或不明日志。两个.bak和名字含old/copy的有效源码全部随原目录保留。逐文件清单是本表的完整附件，不能把分组统计替代删除记录。

### 5.4 完整性、引用与检查边界

**整理后47,410项原文件／链接均找到保留位置；全部保留普通文件SHA256与整理前一致，未解释的丢失或改变为0。** 原研究文档和环境声明若更新，核查的是其原样备份；科研结果、预测、回执和checkpoint没有改写。保留普通文件共33,709,924,671字节。

- [整理前全文件清单](audit/20261116/FILES_BEFORE.jsonl)、[每项保留位置与核查结果](audit/20261116/PRESERVATION_CHECK.jsonl)、[完整性摘要](audit/20261116/PRESERVATION_SUMMARY.json)。内容校验只计算SHA256，不重算实验指标。
- [只读结构检查](audit/20261116/STATIC_CHECKS.json)：Python／Bash／JSON／链接与232个封存源通过。H-JEPA指标、配置原bytes一致。原单测需要本机缺少的SciPy，未安装、未执行；旧全量测试会训练，禁止运行。
- [新入口审核](audit/20261116/HJEPA_REFACTOR_REVIEW.md)：最初help因提前导入h5py失败，已仅在新wrapper修为先解析帮助再导入；复查help退出0且未加载科学模块。治理检查器最初把所有receipt路径误按根目录解释，已按A2/A3/A4代码明确的相对基准修正，未修改任何封存回执。
- 旧封存文档中的原路径不回写；当前记录使用实际新路径或明确的原位兼容路径。归档历史入口的固定祖先目录假设不保证自动迁移，不能把链接存在等同于全流程可运行。
- 本轮未启动训练、特征提取、评测、统计重算、安装、下载或新的实验；新A4准备器也只检查了语法与help，没有实际复制复现实验或执行模型。
