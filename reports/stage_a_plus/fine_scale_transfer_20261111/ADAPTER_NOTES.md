# 两个冻结Fine模型的适配与工程核验

`adapter.py`提供两个预先指定的模型，不进行训练或checkpoint选择：

| key | 模型来源 | 评价工况 | 历史轴排列 |
|---|---|---|---|
| `eps05_seed1_to008` | `fhn_fresh_eps05`，seed1，源ε=0.5 | ε=0.08 | [0,2,1,3] |
| `eps008_seed0_to05` | `fhn_fresh_eps008`，seed0，源ε=0.08 | ε=0.5 | [2,3,0,1] |

路径中的`eps05`表示0.5。轴排列来自既有`step6_fine_20261105/forecast_lock.json`，属于此前使用物理对应得到的归属信息。本次仅复用该信息，报告时应明确这一条件。预测阶段不读取本次评价或参考的物理位置水平值。

接口：

- `KEYS`、`CONFIGS`：固定实验身份与来源。
- `open_model(key)`：返回Fine `encode`、4×4 `beta`、`available`、JSON `provenance`；`raw`子项提供同一公开TRAIN标准化下的raw编码与beta。另保留工程用`frozen`、`loader`对象，主预测只序列化`provenance`。
- `public_panel(key,count,seed)`：该模型指定target工况的公开数据与回执。
- `endpoint_branches(key,public,receipt,selected,dose)`：选定轨迹全部状态的可见命名轴±dose响应，形状`(selected,time,4,4)`。
- `node_branches(key,public,receipt,ids,dose)`：只返回指定扁平节点的响应，形状`(nodes,4,4)`。
- `replay_physical(receipt)`：独立评分阶段调用；编码、校准和查询输出不调用这个入口。
- `sources()`：适配器、依赖源码、模型、G0、公开TRAIN、历史匹配、CAL回执与特征缓存哈希。
- `base`：复用的`point_sets`和`margin`等评分辅助函数。

两个beta都用旧active实验第0组的target CAL重放拟合：

| key | CAL96种子 | 已有query种子 |
|---|---:|---:|
| `eps05_seed1_to008` | 2026160000 | 2026175000 |
| `eps008_seed0_to05` | 2026159000 | 2026174000 |

使用轨迹索引`%4==0`的24条、固定8时刻、+.035的四个命名轴变化做列RMS标准化OLS；与原Stage3拟合规则一致。`%4==1`的24条和−.0175剂量只做检查；既有+.035√2响应未参与拟合。原始query哈希逐项复现，且校准只提供位移响应和已知剂量。

工程核验只读或重放既有TRAIN、既有CAL和旧评价轨迹，没有生成新正式样本。`ADAPTER_ENGINEERING.json`记录：

| 检查 | ε=.5 seed1→.08 | ε=.08 seed0→.5 |
|---|---:|---:|
| TRAIN CPU回放最大G0单位误差 | 6.06×10^-7 | 4.59×10^-7 |
| CAL OLS条件数 | 1.03046 | 1.02780 |
| 正剂量fit相对RMSE | 3.518% | 5.783% |
| 负剂量check相对RMSE | 3.684% | 6.404% |
| 受限节点与整轨迹分支一致 | 是 | 是 |

TRAIN回放阈值沿用2×10^-5；校准fit/check误差只记录，没有据此调参。第一种source映射还与旧`endpoint_branches`逐数组和哈希完全一致；第二种source映射通过公共数据/RNG回执核验及受限节点交叉核验。

可选同源对照入口`source_panel(key,count,seed)`已提供，使用相同模型和观测混合、把epsilon置回模型来源；尚未抽取此类新样本。它将继续使用当前target CAL得到的beta，若主协议需要另行同源CAL，应明确区分该变化。当前正式协议和实验执行由主流程负责，本适配器自检不代替正式通过条件。
