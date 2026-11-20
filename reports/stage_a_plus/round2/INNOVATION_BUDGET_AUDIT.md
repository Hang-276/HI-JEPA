# 追加审计：创新尺度预算与Gaussian-head风险桥的量级

日期：2026-11-04。依照 [追加scope](INNOVATION_BUDGET_SCOPE.md) 执行，随后按根代理授权增加 [相关矩预算scope](INNOVATION_CORRELATION_SCOPE.md)。原有余项报告、A/Q与尺度全部保留，没有重新OLS/polar或修改正式模型。

**结论：本次有限数组plug-in没有为实际Fine建立非空风险认证。** 小state-MSE并不意味着相对innovation covariance的余项小。N8式的完整九regime经验预算在A core只有7/12父能形式计算，其中6个AR预算约1 nat、另一个FHN预算约11.47 nats；Q core只有AR六父能形式计算，FHN六父的九regime均出现ν≥1。仅按u中心化没有改变这些计数。更紧相关桥的逐轴经验上界表达式也仍饱和：针对已拟合core的全部四target/non-target pair均不能给出正Gaussian-head风险下界。

这些是**N10总体上界表达式的经验plug-in**，不是实测的条件创新余项、不是总体上界、不是实际NLL excess，更不是无标签或高概率证书。不能以经验ν≥1断言真实条件中心化ν≥1，也不能以经验ν<1断言N8的总体前提已经认证。

主结果见 [INNOVATION_BUDGET_AUDIT.json](INNOVATION_BUDGET_AUDIT.json)，脚本为 [innovation_budget_audit.py](innovation_budget_audit.py)。相关桥附录见 [INNOVATION_CORRELATION_BUDGET.json](INNOVATION_CORRELATION_BUDGET.json) 与 [独立脚本](innovation_correlation_budget.py)。

## 1 来源、噪声law与时间索引核查

本轮未import任何生成模块，只读取源码文本、JSON、NPZ指定成员。确认链为当前 `confirmation_v2/data_owner.py` → `registration/INSTANCE_LOCK.json` →原physics manifest及原生成代码。四份manifest哈希全部与登记引用相符；四个关键源码哈希与原 `ROOT_TRAINING_RELEASE.json` 相符。

| 系统 | 已核实真实innovation law | 当前配置边界 |
|---|---|---|
| AR rich | base std=.12；u=5…8各使对应一轴std×3；其余轴保持base；独立Gaussian innovations | mode=rich；manifest variance_ratio=9；u为外生iid uniform0…8；未改为weak law |
| 两槽FHN | 确定性积分后加独立Gaussian kicks；base std=.025；u=5…8各使一轴std×3 | 源字段variance_multiplier=3实际乘在std上，故高/低方差比为9；contrast_strength=1、rotational=false、measurement_noise=0；ε=.08/.5原样保留 |

源码分别为历史 `hcf_identified_g_20261024/fine_stage.py`、`hcf_8h_extension_20261025/fhn/env_fhn.py::regimes(cfg)`，FHN接线另由当前 `identifiability_core_v1/data_owner.py::_physical` 核查。后者实际执行 `current=predicted+means[u_t]+stds[u_t]*noise_t`。

因此本轮使用 **regimes[:,t] 对应next state[:,t+1]**，不是next状态之后的下一个regime。每个split的t=0…T−2全部合法一步转换保留。TRAIN和CAL均采用上一份余项审计的固定TRAIN物理/Fine尺度；CAL不重新标准化。

设物理TRAIN std为s_j，state gauge中

\[
D_u=\operatorname{diag}\left(\sigma^2_{physical,j}(u)/s_j^2\right).
\]

每父两个core B=A和B=Q及截距b直接读取原审计，不重拟合；核心innovation covariance为 \(C^B_u=BD_uB^T\)。所有矩阵正定，未加ridge或修改噪声。

## 2 真正测量的量与未测量的量

在所有next状态上重建固定缓存残差

\[
r_{t+1}=W_{t+1}-(BV_{t+1}+b).
\]

对每个u分别计算

\[
\bar\nu^2_{raw,u}
=\frac1{N_u}\sum_{i:u_i=u}r_i^T(C^B_u)^{-1}r_i,
\]

\[
\bar\nu^2_{u\text{-centered},u}
=\frac1{N_u}\sum_{i:u_i=u}(r_i-\bar r_u)^T(C^B_u)^{-1}(r_i-\bar r_u).
\]

后者仅减去同regime的样本均值（ddof0），没有拟合 \(E[r\mid c]\)、条件均值网络或动力学模型。它满足经验恒等式“raw能量=centered能量+样本均值能量”，所以更小；但样本中心化本身存在估计误差，不能称有限样本总体上界。

真正N8里的量是

\[
\nu_u^2=E[\|(C^B_u)^{-1/2}(r-E[r\mid c])\|^2\mid u],
\quad c=(v_t,a_t,u_t).
\]

N10给总体投影嵌套不等式：该量不大于仅u中心化的总体二阶矩，再不大于未中心化总体二阶矩。**本轮测的是这两个上界表达式的有限数组对应量，而非ν本身，也没有证明样本值覆盖总体值。**

每格经验ν<1时，列出形式表达式

\[
\delta_{u,plugin}=4\log\frac{1+\bar\nu_u}{1-\bar\nu_u}.
\]

任一regime经验ν≥1，则该父完整九regime预算记N/A，不丢弃失败regime后再平均。完整平均分别保存真实设计权重1/9和经验频率N_u/N；下表列经验频率版本。只给ν的RMS作描述，**没有**将RMS先代入log来代替逐u计算再平均。

## 3 全部九regime与12父结果

九regime在所有split均有样本。JSON保存12父×两core×TRAIN/CAL×9，共432格的计数、raw/centered ν²、ν、N/A及样本均值，不删除任何格。

| Core/支持 | raw全部9u<1的父 | raw u<1格数/108 | 仅u中心化全部9u<1的父 | 仅u中心化u<1格数/108 |
|---|---:|---:|---:|---:|
| A/TRAIN | 7/12 | 71/108 | 7/12 | 71/108 |
| A/CAL（TRAIN尺度） | 7/12 | 76/108 | 7/12 | 76/108 |
| Q/TRAIN | 6/12 | 54/108 | 6/12 | 54/108 |
| Q/CAL（TRAIN尺度） | 6/12 | 54/108 | 6/12 | 54/108 |

这些父/格数是预算表达式的适用性诊断，不是模型成功率或独立环境数。A的额外第7父为FHN ε=.5 rep2；Q的6父全部是AR。FHN六父的Q core在每个split的9个regime均ν≥1。

### 3.1 一般仿射core A

每格“TRAIN/CAL”使用同一A、TRAIN尺度和noise law，只有已有next状态样本支持不同。F_core是固定矩阵的解析Gaussian-head profile contrast，见下一节；不是actual Fine的NLL实测。

| 槽/rep | max ν raw：TRAIN/CAL | ν<1的u数：TRAIN/CAL | 完整δ_plugin：TRAIN/CAL | F_core，uniform9 |
|---|---:|---:|---:|---:|
| AR1/0 | .172755 / .168973 | 9/9 / 9/9 | 1.08584 / 1.03305 | .00126758 |
| AR1/1 | .154501 / .150138 | 9/9 / 9/9 | 1.07252 / 1.04061 | .00097730 |
| AR1/2 | .149971 / .151279 | 9/9 / 9/9 | .96901 / .94647 | .00081754 |
| AR2/0 | .184837 / .178820 | 9/9 / 9/9 | 1.16353 / 1.13017 | .00256402 |
| AR2/1 | .159406 / .152922 | 9/9 / 9/9 | 1.01108 / .99925 | .00169377 |
| AR2/2 | .161827 / .154419 | 9/9 / 9/9 | .94090 / .90129 | .00147372 |
| FHN.08/0 | 1.223161 / 1.195224 | 1/9 / 2/9 | N/A / N/A | .00183055 |
| FHN.08/1 | 1.201415 / 1.200063 | 1/9 / 2/9 | N/A / N/A | .00233895 |
| FHN.08/2 | 1.231635 / 1.186146 | 1/9 / 2/9 | N/A / N/A | .00180234 |
| FHN.5/0 | 1.141540 / 1.193135 | 2/9 / 2/9 | N/A / N/A | .00155021 |
| FHN.5/1 | 1.055401 / 1.075805 | 3/9 / 5/9 | N/A / N/A | .00179672 |
| FHN.5/2 | .930065 / .960466 | 9/9 / 9/9 | 11.47435 / 11.42391 | .00068774 |

AR即便经验ν都<1，完整预算约.9–1.2 nats/transition，比其固定core contrast大数百倍以上；FHN.5 rep2虽形式可计算，预算约11.47 nats而core contrast约.000688。**若把这些经验量当作总体量代入，N8也会非常空泛；但本轮没有资格将其当作总体量。**

### 3.2 Polar core Q（包含nonorth affine与非仿射总余项）

| 槽/rep | max ν raw：TRAIN/CAL | ν<1的u数：TRAIN/CAL | 完整δ_plugin：TRAIN/CAL | F_core，uniform9 |
|---|---:|---:|---:|---:|
| AR1/0 | .249229 / .252188 | 9/9 / 9/9 | 1.37611 / 1.33619 | .00033684 |
| AR1/1 | .489875 / .492725 | 9/9 / 9/9 | 2.15130 / 2.13161 | .00882891 |
| AR1/2 | .349958 / .350343 | 9/9 / 9/9 | 1.59642 / 1.58701 | .00347715 |
| AR2/0 | .362627 / .346341 | 9/9 / 9/9 | 1.93049 / 1.86629 | .00119699 |
| AR2/1 | .287441 / .289797 | 9/9 / 9/9 | 1.67287 / 1.64890 | .00706535 |
| AR2/2 | .303907 / .298669 | 9/9 / 9/9 | 1.57405 / 1.51706 | .00472184 |
| FHN.08/0 | 3.387324 / 3.354673 | 0/9 / 0/9 | N/A / N/A | .01622793 |
| FHN.08/1 | 3.263935 / 3.250740 | 0/9 / 0/9 | N/A / N/A | .01347821 |
| FHN.08/2 | 3.004229 / 2.984949 | 0/9 / 0/9 | N/A / N/A | .01085642 |
| FHN.5/0 | 1.677616 / 1.748741 | 0/9 / 0/9 | N/A / N/A | .00121969 |
| FHN.5/1 | 1.627886 / 1.666967 | 0/9 / 0/9 | N/A / N/A | .00062596 |
| FHN.5/2 | 1.203683 / 1.217395 | 0/9 / 0/9 | N/A / N/A | .00043670 |

这与前一审计的“小state余项MSE”不矛盾：innovation很小，且Q余项额外包含(A−Q)v；相对innovation归一化后可能远大于1。不能删去nonorth部分后把A的预算移作Q/κ的预算。

### 3.3 仅u中心化没有解决量级

所有raw/centered ν<1的格数与父数都相同。AR A的TRAIN完整δ从raw约.94090–1.16353变为约.92321–1.15353；FHN.5 rep2从11.47435变为11.46190。Q AR预算有所减小，但仍约1.26685–1.79039 nats。FHN Q所有九regime仍未进入ν<1域。只按u去均值不能消除随当前state/action变化的余项条件均值，不能冒称已经测得r°。

## 4 与Gaussian-head KL/NLL的对应，不能改称一般MI界

对固定core，真实Gaussian innovation covariance是C_u。这里计算

\[
G(C_u)=\tfrac12\log\frac{\prod_j C_{u,jj}}{\det C_u},
\qquad F_{core}=\tfrac19\sum_{u=0}^8G(C_u).
\]

它是在均值自由profile、方差仅按regime取对角形式时，固定仿射core的Gaussian-head条件KL/NLL excess。actual Fine的条件分布一般非Gaussian、真实条件均值未知；本轮没有计算actual Fine的条件covariance或NLL excess。

最大熵桥针对“完整conditional observation NLL减正确真实条件entropy baseline”的excess及对应Gaussian covariance contrast。它**不是**本轮实测的一般mutual-information lower bound，不允许直接把某个观测CAL NLL数填为J_actual。已有CAL参与Fine选点，且本次A/Q来自同TRAIN标签拟合，有限样本泛化仍未认证。

N8/N10数学有效性与当前量级可用性是两件事。本次粗矩表达式很松，并不反驳N8；真实条件中心化余项可能显著更小，但本轮没有估计它，也不追加条件均值模型来救结果。

## 5 追加附录：逐轴相关风险桥仍饱和

按追加scope，复用已存C_u、A/Q和尺度，最少重读同一固定缓存，计算

\[
\bar\tau^2_{j,u}=\frac{\widehat E[r_j^2\mid u]}{C_{u,jj}},
\quad
\bar\tau^2_{j,u,centered}=\frac{\widehat{\operatorname{Var}}(r_j\mid u)}{C_{u,jj}},
\quad e_{ab,u}=2(\bar\tau_{a,u}+\bar\tau_{b,u}).
\]

这些仍是N10式上界表达式的样本量，不是条件中心化\(\tau\)总体上界。对全部四target/non-target pair保存均匀1/9及经验频率的\(e_{rms}\)，并保留逐u可用的\(\min(e,2)\)版本；没有挑选有利pair或core。

相关桥的总体形式为

\[
\|\rho(C_u)_{ab}\|_{L^2(u)}
\le \sqrt{1-e^{-2J_{actual}}}+\|e_{ab,u}\|_{L^2(u)}.
\]

若把本轮plugin暂代总体量，一个正风险下界至少需要\(\rho_{core,rms}>e_{rms}\)。本次在A/Q、TRAIN/CAL、raw/u-centered、uniform/经验权重的**每一种设置中，48个pair全部不满足**；即便逐u cap e到2，仍全部饱和，形式正风险下界为0。

| Core/支持（uniform9，raw） | 48 pair的e_rms范围 | 对应core ρ_rms范围 | 无正下界pair |
|---|---:|---:|---:|
| A/TRAIN | .209168–2.667659 | .001548–.046012 | 48/48 |
| A/CAL，TRAIN尺度 | .194979–2.647528 | 同一C：.001548–.046012 | 48/48 |
| Q/TRAIN | .286269–7.670466 | .001064–.165164 | 48/48 |
| Q/CAL，TRAIN尺度 | .278357–7.602099 | 同一C：.001064–.165164 | 48/48 |

raw与仅u中心化的完整逐轴τ、四pair逐u e及两种权重结果全部保存在相关预算JSON。该检查只比较**已拟合4D core**的相关量；不能把它当作purepair二维Q的角度认证，也没有计算整个假想临界旋转族的统一余项预算。因此不宣称“所有目标critical barrier都无解”，只说明当前可直接读取的粗余项矩预算不足以从这个相关桥推出正actual风险下界。

## 6 对研究决策的含义

- 数学上，条件Gaussian-head最大熵单向桥可继续保留；本次不反驳它。
- 实证上，不能把前份小state余项或高dense读出直接当作innovation尺度的小ν。本次已对全部12父、两core、九regime保留失败。
- 以本次可直接算出的N10表达式进入N8，量级不够；改用逐轴相关桥的同类粗表达式也未获得非空正风险信息。不能仅因公式更紧就宣布已接上实际Fine。
- 尚未解决的关键是条件中心化余项、其可估计性及总体上界；本轮没有进一步训练/拟合条件均值模型，也没有据此扩大实验。
- 所有这些诊断均有真值、使用已暴露TRAIN/CAL且A/Q为事后拟合，不能变成无标签证书或独立确认。

## 7 来源、复核与新增成本

主诊断保存scope/hash、独立脚本hash、旧A/Q审计JSON hash、四关键源码hash及所有读取文件/member。四关键源码与原189源训练锁中的对应hash相符，physics manifests与原INSTANCE_LOCK引用相符。

主预算新增216次4×4 Cholesky、432次分组whitening solve，执行约.10秒；相关附录只算逐轴矩与确定代数，新增OLS/polar/Cholesky/solve均为0。两部分均零条件均值模型拟合、NN训练/forward、PRNG、新物理数据、checkpoint反序列化、FINAL数组或GPU。原余项结果、正式方法、噪声law、目标和门槛没有改变。
