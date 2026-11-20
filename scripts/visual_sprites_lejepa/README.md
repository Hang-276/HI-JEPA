# 32×32 双实体视觉 LeJEPA 对照

使用分离颜色通道渲染两个连续移动的高斯斑块。两臂共享卷积编码器结构、图像流和探针协议。视觉 LeJEPA 使用相邻帧对齐和作者 SIGReg，远期臂使用 lag 24 的预测与较弱 SIGReg。论文结果以各次正式运行的设计锁和预测回执为准。

无噪声且采用实数运算时，红、蓝通道去掉已知背景后的高斯强度比可分别确定两个中心。实际像素有微量噪声和有限精度，可恢复精度由独立 AUDIT 测量。评分目标是渲染前的物理 AR 坐标 `z`，图像坐标使用 `tanh` 映射。

## 本机环境与开发检查

本机使用被 Git 忽略的 `.venv_visual_cpu/`，基于既有 `/opt/miniconda3/envs/evg-agent` 并复用其包。实际导入版本为 Python 3.11、PyTorch 2.12.0、SciPy 1.17.1、NumPy 2.4.6。当前运行上下文报告 MPS 不可用。该环境的 PyTorch 与 SciPy 默认会加载两份 OpenMP 库，以下调用使两者共享 PyTorch 自带的同一份库：

```bash
DYLD_LIBRARY_PATH=/opt/miniconda3/envs/evg-agent/lib/python3.11/site-packages/torch/lib \
  .venv_visual_cpu/bin/python -B scripts/visual_sprites_lejepa/render.py
```

`-B` 防止在复用的环境或封存作者源码旁生成字节码。该命令生成开发种子的物理窗口与独立 REF，并输出默认设计的翻转率。用 `--scan` 可复核允许范围内的系数网格。配置和已有收据见 [default.json](../../configs/visual_sprites_lejepa/default.json)、[翻转率预检](../../results/visual_sprites_lejepa/development/FLIP_PREFLIGHT.json)和[CPU预检](../../results/visual_sprites_lejepa/development/RUNTIME_PREFLIGHT.json)。

运行入口支持 `--development`、`--formal`、`--smoke` 与 `--verify`。训练模式使用上面的隔离环境和库路径，脚本换成 `run.py`。正式入口先检查开发翻转率及同代码、配置和设备下的单种子开发恢复收据，再创建 `formal_v1/`。当前默认动力学未达到开发翻转率范围。

双臂工程验证使用独立开发流、各三步训练和一个独立盲测小面板，耗时 7.2183 秒。它验证了从图像到封存再到评分的连通性。可用以下命令复核 12 个文件的哈希和时序：

```bash
.venv_visual_cpu/bin/python -B scripts/visual_sprites_lejepa/run.py --verify --name smoke_receipt_v1
```

实验结果应从 [工程收据](../../results/visual_sprites_lejepa/smoke_receipt_v1/FORECAST_RECEIPT.json) 与 `evidence_tier` 区分。当前只有开发预检和工程验证，`formal_v1/` 尚未建立。

## 正式结果的判读

正式配置在接触 TEST 前连同代码散列封存。图像生成需要在渲染进程内短暂计算潜在坐标，预测阶段只接收图像。保存各单元预测、预测 REF 和 CAL 阈值，再写入带 UTC 时间及哈希的 `FORECAST_RECEIPT.json`。评分阶段核验回执后，才取得独立重放的 TEST 坐标。CPU `float64` 评分和 CP 上界直接使用仓库评分模块。

每个单元报告四轴 `AUDIT R²`、`own_cal` 覆盖、对六项应用 Bonferroni 后的单侧 CP 上界、综合状态，以及常规窗与物理翻转窗的全答／选择性错选。跨单元共享同一 TEST 面板。翻转子集错误计数用于诊断，正式 CP 作用于整个接受集。

## 联合改动力学的正式入口

新版将两慢轴设为 `.964/.961`，两快轴设为 `.780/.760`。开发搜索的两个面板分别为 10.2661% 和 8.4167% 翻转，第三个独立开发面板为 8.4106%。预测 lag 为 15，慢轴在此跨度的相关约为 `.577/.551`，快轴为 `.024/.016`。数据和网格见 [联合搜索收据](../../results/visual_sprites_lejepa/development/JOINT_RHO_SEARCH.json)。

```bash
DYLD_LIBRARY_PATH=/opt/miniconda3/envs/evg-agent/lib/python3.11/site-packages/torch/lib \
  .venv_visual_cpu/bin/python -B -u scripts/visual_sprites_lejepa/run_joint.py \
    --config configs/visual_sprites_lejepa/joint_v1.json --attempt 1 --device cpu
```

开发使用单个独立模型种子训练两臂。正向四轴须全部达到 `.90`，远期臂两慢轴须达到 `.90`，且两快轴均低于 `.60`。同时通过后入口继续执行 `formal_v1`。开发失败后允许一次配置调整和第二次开发尝试，之后仍有未过项即结束，并保留两次收据。原 `default.json` 对应早期工程小样，联合设计保存在独立配置文件。
