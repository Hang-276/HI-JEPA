#!/usr/bin/env python3
"""Render the numerical report from saved receipts; no simulation or training."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/exploratory_20261116'
TIER = 'exploratory_numerical_sandbox'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])+'\n'


def pct(x):
    return '—' if x is None else f'{100*x:.3f}%'


def main():
    p1 = json.loads((OUT/'probe1_summary.json').read_text())
    p2 = json.loads((OUT/'probe2_summary.json').read_text())
    validation = json.loads((OUT/'validation.json').read_text())
    phase, anchors, stress = p2['phase'], p2['anchors'], p2['stress']
    harmonic, sub = p1['harmonic'], p1['subspace_phase']
    parts = [r'''evidence_tier: exploratory_numerical_sandbox

# When Does Factor Identifiability Yield Reliable Slow Representations?

## 1. 核心判断与证据范围

本次完成纯 NumPy 机制沙盒：162 个谐波对照、420 个子空间条件（每格 3 次 CAL 重复，共 1,260 次拟合）、完整 800 格慢集可靠性扫描、128 个封存模型单元只读投射，以及 56 个请求压力单元和 16 个另列的极限对照。没有训练或重跑封存神经网络，也没有官方视觉 FourRoom/H-JEPA 的新结果。

**更适合论文主线的是“因素恢复之后还需满足两层条件”：先控制残差时间谱相对真实谱间隔的偏置，才能恢复目标慢子空间；再控制归一化评分误差相对同窗口分差的大小，才能可靠选择物理慢因素。** 本次为这条有条件的主线提供机制图和反例，尚不足以声称已有一条普适、陡峭的 ICML 级经验相变定律。

| 候选论断 | 本次结果 | 写入论文的边界 |
| --- | --- | --- |
| 可辨识表征避免观测非线性畸变 | 支持：人工提供近线性因素表征明显改善恢复 | 尚未证明训练的 LeJEPA 自动达到此条件，更未证明只有这条路线可行 |
| 实际 Poly2-SFA 在 ρ₁=ρ₀² 精确劫持 | 不支持这一精确断言；只有 oracle latent-Hermite 字典具有该精确边界 | 必须保留“观测字典”与“理想因素字典”的差别 |
| 残差谱偏置跨过 eigengap 后增加 CAL 也无法修复 | 总体公式清楚，数值支持；有限样本过渡受第二个比值控制 | 最干净的正面机制候选，需在真实编码器残差上最小确认 |
| 单一 population SNR 解释同窗错误 | 不支持：全答 Spearman −0.110，own_cal −0.157 | population pair inversion 与同窗集合误差是不同事件 |
| 拉长 T 总能获得 <1% 的 T_min 定律 | 不支持：固定 REF 偏差与事件定义导致非单调，许多曲线网格内不达标 | 只能报告观察到的首次网格过线与未过线 |
| 快轴退化可击穿 own_cal | 支持有限窗口失效，最坏 23.360%；请求网格没有总体快轴替换 | 相对方差成分决定机制，不能说任意微弱泄漏必然劫持 |
| AUDIT R²≥.85 是理论必要门槛 | 不成立；缩放不变性提供直接反例 | 这是既有操作性准入门，不能当必要或充分定理 |

本报告百分比差用“百分点”（pp）；所有统计拟合、Wilson 区间与网格最小值均为探索性描述。共享 TEST、REF、嵌套轨迹前缀和模型的格子不能视作独立重复。

## 2. 口径、实际环境与复现入口

严格只读对齐 [metrics.py](../src/slowrepr/metrics.py)、[原始观测生成函数](../reports/lejepa_slow_rank_20261112/training/run.py) 和 [前序理论文档](THEORY_FORMALIZATION_20261116.md)。四轴编号为 0,1,2,3。主评分为

\[
S_i(w)=\frac{1}{3}\sum_{\ell\in\{4,16,32\}}
\frac{1}{T-\ell}\sum_{t=0}^{T-\ell-1}(w_{t+\ell,i}-w_{t,i})^2,
\quad s_i=S_i(z)/v_i,\quad\widehat s_i=S_i(\widehat z)/\widehat v_i.
\]

每个 lag 在先平均自身有效端点后等权；不把长 lag 的端点数错误地混到总权重。预测分母是独立 32,768 状态 REF 上自身的 `ddof=0` 方差，不夹成正数、不替换成物理方差。主真值采用封存低维实验的已知物理方差 `v_i=1`；另列共享物理 REF 估计分母的敏感性对照。top2 使用稳定排序、随后集合索引排序；分差是第三小减第二小；接受条件 `gap >= threshold`。

主要事件是 **同一个窗口中** 预测 top2 与物理评分 top2 不同，不是预测集合偏离总体 `{0,1}`。这一区别决定了 SNR 和窗口长度的解释。

三个 R² 口径必须分开：

- 探针一令信号功率参数 `p=1−ε`，`h=√p Qz+√(1−p)e`。直接逆旋转的总体 MSE-R² 是 `2√p−1`，最优线性读出 R² 才为 `p`。
- 探针二 800 格令 `zhat=z+√(1−r)e`，白噪声独立；扫描的 `r` 就是总体 MSE-R²。实际 REF 上的逐轴估计也保存。
- 压力测试快轴参数 `s` 是信号功率系数，实际总体 MSE-R² 是 `1−(√s−1)²−leak²−σ²`，另有独立 AUDIT 实测值。不能把 `s=.85` 解释为准入边界。

实际可用数值解释器是 `/opt/miniconda3/bin/python`（Python 3.13.11、NumPy 2.4.4），没有安装任何包；脚本通过 Python 3.11 语法解析。系统默认 `python3` 较旧，仓库检查最初因缺少 `hashlib.file_digest` 失败；把已有数值解释器目录放到 PATH 前端后，用同一条 `python3 scripts/check_repository.py` 原样通过，未改检查脚本。

```bash
/opt/miniconda3/bin/python scripts/exploratory_20261116/probe1_harmonic_and_subspace.py --help
/opt/miniconda3/bin/python scripts/exploratory_20261116/probe2_phase_diagram_alignment.py --help
# 使用尚未存在的新结果子目录，保留本次原始数值收据：
/opt/miniconda3/bin/python scripts/exploratory_20261116/probe1_harmonic_and_subspace.py --output-dir results/exploratory_20261116/my_rerun
/opt/miniconda3/bin/python scripts/exploratory_20261116/probe2_phase_diagram_alignment.py --output-dir results/exploratory_20261116/my_rerun
```

探针一种子 `2026111601`；探针二种子 `20261116`；压力测试种子 `20263116`。设计在全量运行前写入 [DESIGN.json](../results/exploratory_20261116/DESIGN.json)，该文件的 `seed_base` 不是所有探针共享的单一 RNG 种子。所有实际种子以各收据为准。
''']
    parts.append(table(['主脚本', '首次收据内部计时', '独立复跑总墙钟（含启动与写盘）', '数值复现'], [
        ['探针一', f"{p1['runtime_seconds']:.3f} s", f"{validation['reproducibility'][0]['wall_seconds_including_startup_and_write']:.3f} s", '非计时字段逐项完全一致'],
        ['探针二', f"{p2['elapsed_seconds']:.3f} s", f"{validation['reproducibility'][1]['wall_seconds_including_startup_and_write']:.3f} s", '非计时字段逐项完全一致']]))
    parts.append(r'''
两份主脚本均低于 60 秒。主网格采用并行轨迹行、共享噪声和有限长度缓存的充分统计量；只按时间及少量 lag 循环，没有逐条蒙特卡洛轨迹的 Python 循环。锚点只读 11 份 JSON，不加载模型或预测张量。

## 3. 探针一：谐波与无标签慢子空间

### 3.1 实际观测 Poly2 不等于因素 Hermite 字典

复刻可逆三角非线性观测与归一化 Hadamard 旋转。SFA 使用无标签端点合并协方差白化与对称滞后协方差；谐波实验预算是 **独立 lag-1 状态对数**（1024/4096/65536），不是一条同长度轨迹。每个条件只有一个 CAL 抽样，样本预算嵌套；16,384 个独立 AUDIT 状态统一评测。原始观测与 Poly2 的拟合不使用真实因素；oracle 字典是额外诊断控制。

下表 N=65536，角度为函数空间最大主夹角正弦；H₂ 污染是 H₂(z₀) 在两维输出上的多元投影 R²（0–1，亦为单一谐波对输出的典型相关平方）。
''')
    rows = []
    for case, rates in harmonic['design']['rho_cases'].items():
        by = {r['method']: r for r in harmonic['rows'] if r['case'] == case and r['N_independent_lag1_pairs'] == 65536}
        raw, poly, oracle, ident = [by[m] for m in ('raw_linear_sfa', 'poly2_observed_sfa', 'oracle_latent_hermite_dictionary', 'identifiable_power_1')]
        rows.append([case, f"{raw['function_subspace_sin_theta_max']:.4f}",
                     f"{poly['function_subspace_sin_theta_max']:.4f} / {poly['harmonic_H2_projection_R2']:.4f}",
                     f"{oracle['function_subspace_sin_theta_max']:.4f} / {oracle['harmonic_H2_projection_R2']:.4f}",
                     f"{ident['function_subspace_sin_theta_max']:.4f}"])
    parts.append(table(['ρ₁／条件', 'Raw 角度', 'Poly2 角度 / H₂', 'oracle 角度 / H₂', 'identifiable p=1 角度'], rows))
    parts.append(r'''
在独立高斯 AR(1) 的因素 Hermite 字典中，一阶 z₁ 的 lag-1 特征值是 ρ₁，二阶 H₂(z₀) 是 ρ₀²=.9025。低于临界点，oracle 第二慢模式是 H₂；高于临界点则是 z₁；**等号处第二特征空间不唯一**，不能拿一次样本选出的模式证明等号属于某一侧。

实际 `Poly2(observe(z))` 不包含精确 z₀/H₂(z₀) 的保证，因此不能继承该临界值。它在本次 `.870→.885` 网格间已发生明显转变；`.885/.895` 虽低于 .9025，H₂ 污染却很低。Raw 和 Poly2 仍有持续的线性慢子空间偏差，但本次大样本对照不支持把 oracle 的精确临界值直接移植到观测 Poly2；有限 N 结果本身也不是后者总体极限的证明。谐波本身是合法慢函数；“劫持”专指它挤占不同物理因素的表示容量。

以下固定 ρ₁=.895，展开三个预算和三种 identifiable 功率，给出角度、多元慢因素线性回归 R² 与 H₂ 污染。1024→65536 的单次配对抽样变化不是误差条或渐近证明。
''')
    rows = []
    for r in harmonic['rows']:
        if r['case'] == 'rho1_0.895':
            rows.append([r['N_independent_lag1_pairs'], r['method'], f"{r['function_subspace_sin_theta_max']:.4f}",
                         f"{r['slow_multivariate_linear_regression_R2']:.4f}", f"{r['harmonic_H2_projection_R2']:.4f}"])
    parts.append(table(['独立对数', '方法', 'sinΘ_max', '慢因素回归 R²', 'H₂ 投影 R²'], rows))
    parts.append(r'''
![Harmonic comparison](../results/exploratory_20261116/figures/harmonic_boundary.svg)

### 3.2 残差偏置与 CAL 波动需要分别控制

CAL 为 128 或 1024 条独立平稳轨迹，每条 T=193；三个独立 CAL 重复；TEST 为 2048 条独立轨迹；REF 为 32768 个独立状态。lag 取 1/4/16。各向同性残差自相关 .5；方向性残差在快轴方向为 .94，在慢轴方向为 .5，随后与信号共同旋转。模型是在原因素坐标方向加入独立慢过程，不是把真实慢变量复制到快轴（后者在探针二压力测试）。

写 ε=1−p，δℓ=ρ₁^ℓ−ρ₂^ℓ，κ=(.5,.5,.94,.94)。总体白化后各模式特征值为

\[
\lambda_i=(1-\varepsilon)\rho_i^\ell+\varepsilon\kappa_i^\ell,\qquad
\lambda_1-\lambda_2=(1-\varepsilon)\delta_\ell-\varepsilon(.94^\ell-.5^\ell).
\]

在本次排序结构中真实目标子空间的总体边界为

\[
\varepsilon_{\rm crit}=\frac{\delta_\ell}{\delta_\ell+.94^\ell-.5^\ell}.
\]

各向同性残差没有上述相对谱偏置；方向性残差过线后，不论 CAL 多大，都收敛到错误目标子空间。有限样本还取决于白化与 lag 协方差波动。保存的无量纲诊断为偏置比 `b=ε(.94^ℓ−.5^ℓ)/[(1−ε)δℓ]` 和抽样比 `a=||Ĉ_whitened−C_population||op/[(1−ε)δℓ]`；这些使用已知合成总体，不能直接当真实数据上无标签可观测的风险证书。
''')
    rows = []
    for gap in sub['design']['boundary_rho_gaps']:
        bs = [b for b in sub['population_boundary_conditions'] if b['gap_rho'] == gap]
        rows.append([f'{gap:.3f}']+[f"{b['critical_error_power_1_minus_R']:.6f}" for b in bs])
    parts.append(table(['Δρ', 'ε_crit lag1', 'ε_crit lag4', 'ε_crit lag16'], rows))
    parts.append('窄间隔 .002 时，三个边界对应直接 MSE-R² 分别为 .995470、.991795、.981005；不是统一的 R² 门槛。\n\n以下均为 Ncal=1024，3 次 CAL 拟合均值。loading angle 只比较信号方向；function angle 额外包含独立噪声的不可恢复底限，每次拟合满足 `sin²Θ_function=1−p cos²Θ_loading`；表内分别平均后的角度不必满足同一非线性恒等式。同窗错误是四轴 oracle 命名后的诊断。\n')
    rows = []
    for c in sub['cell_means']:
        if c['N_cal_trajectories'] == 1024 and c['gap_rho'] in (.002, .026) and c['signal_power_parameter'] in (.995, .99, .95) and c['cal_lag'] in (1, 16):
            rows.append(['iso' if c['residual']=='isotropic_ar05' else 'slow', f"{c['gap_rho']:.3f}", f"{1-c['signal_power_parameter']:.3f}", c['cal_lag'],
                         f"{c['signal_loading_subspace_sin_theta_max_mean']:.3f}", f"{c['function_subspace_sin_theta_max_mean']:.3f}",
                         pct(c['oracle_aligned_four_axis_same_window_top2_error_mean'])])
    parts.append(table(['残差', 'Δρ', 'ε', 'lag', 'loading angle', 'function angle', 'oracle 同窗错误'], rows))
    parts.append(r'''
各向同性残差 Ncal 从 128 增至 1024，平均 loading angle 从 .2256 降至 .0942，超过 45° 的拟合记录数从 18/315 降至 4/315。方向性残差平均角度从 .4748 降至 .4058，但超过 45° 的拟合记录数仍为 108/315。因而“多收集 CAL”主要缓解抽样部分，不能消除总体偏置。

方向性残差下，角度对纯抽样比值的 Spearman 为 .5873，对 `a+b` 为 .8717；各向同性为 .8456。`a+b` 对同窗错误的相关分别为 .7092/.6025。相关变强不代表一维坍缩：`ε=.02, Δρ=.006, N=1024, lag16` 的 `a+b=1.242`、角度 .153；`ε=.20, Δρ=.05, N=1024, lag4` 的 `a+b=1.283`、角度 .997，后者已跨总体偏置边界。

方向性残差的总体错误子空间与经验 >45° 崩溃恰好都为 216/630，但交叉计数为 `[[392,22],[22,194]]`，有 44 条拟合记录不一致，不能宣称阈值完美预测。这些 315/630 计数包含每条件三次 CAL 重复，不代表同等数量的独立条件。lag16 提高偏置容忍度，却不保证有限 CAL 更准：例如 iso、Δρ=.026、ε=.005，lag1/16 角度为 .0153/.0631。

**命名边界：** 白化与特征分解不使用真值；角度评测使用真值；四轴 top2 错率还使用真实坐标的符号和排列对齐。它调用全部四个特征向量，不能解释成仅输出两维慢表示即可获得的无标签物理命名能力。主空间错误与四轴对齐后错误也不必单调。

![Subspace phase](../results/exploratory_20261116/figures/subspace_phase.svg)

## 4. 探针二：800 格可靠性图与窗口长度

两过程 × 5 个 α × 10 个 T × 8 个 MSE-R²；每格 CAL2048、TEST4096、REF32768。`own_cal` 为各格独立 CAL 分差的 `method=higher` 中位阈值，其 TEST 覆盖率只接近 50%，不强制精确一半。固定门限为封存 g50 的 `.17748381448816608`。

### 4.1 解析 SNR 检验的是哪一个事件

令 v=1−r，M 为精确等权 lag 平方差矩阵，Σᵢ[t,s]=ρᵢ^|t−s|+v 1[t=s]。计算全部跨 lag 项与端点重叠，非独立 lag 近似：

\[
\mu_i^y=\frac{2\,\operatorname{avg}_{\ell}(1-\rho_i^\ell)+2v}{1+v},\qquad
\operatorname{Var}(s_i^y)=\frac{2\operatorname{tr}(M\Sigma_iM\Sigma_i)}{(1+v)^2},
\quad \mathrm{SNR}_{eff}=\frac{\mu_2^y-\mu_1^y}{\sqrt{\operatorname{Var}(s_1^y)+\operatorname{Var}(s_2^y)}}.
\]

另保存用户字面形式“物理均值差/预测差标准差”，以及给定本次实际 REF 后分别用各轴 REF 分母的条件矩。正态尾 `Φ(−SNR)` 是固定总体边界 pair 的倒序代理，不是同窗 top2 集合误差公式。
''')
    d = phase['diagnostics']
    parts.append(table(['检验', '数值', '解释'], [
        ['SNR vs 全答错误 Spearman', f"{d['spearman_snr_vs_same_window_all_risk']:.5f}", '不支持紧主曲线'],
        ['SNR vs own_cal 错误 Spearman', f"{d['spearman_snr_vs_same_window_own_cal_risk']:.5f}", '不支持紧主曲线'],
        ['SNR vs 固定阈值覆盖率 Spearman', f"{d['spearman_snr_vs_fixed_coverage']:.5f}", '覆盖不能由 population boundary 单独解释'],
        ['物理分子版 vs 全答 / own / 覆盖', ' / '.join(f'{d[k]:.5f}' for k in ('spearman_requested_physical_gap_snr_vs_same_window_all_risk','spearman_requested_physical_gap_snr_vs_same_window_own_cal_risk','spearman_requested_physical_gap_snr_vs_fixed_coverage')), '改变分子未修复目标差异'],
        ['正态代理 vs population pair 倒序 MAE', f"{100*d['normal_proxy_population_pair_mae']:.4f} pp", '同目标比较'],
        ['给定 REF 的正态代理同目标 MAE', f"{100*d['normal_proxy_conditioned_ref_population_pair_mae']:.4f} pp", '条件化 REF 改善数值预测'],
        ['正态代理 vs 同窗集合错误 MAE', f"{100*d['normal_proxy_vs_same_window_risk_target_mismatch_mae']:.4f} pp", '目标混用的偏差，不能归结为正态近似不准']]))
    parts.append(r'''
### 4.2 T_min 是网格描述，未得到单调长度定律

80 条固定 (process,α,R²) 的 T 曲线中，全答只有 4 条曾低于 1%，这 4 条随后均回到 ≥1%；own_cal 为 61 条曾低于 1%，其中 1 条后来回穿。定义“首次观察过线”与“首次逐格 Wilson 95% 上界过线”分别列出；后者不是在 800 格选择后仍有效的同时置信保证。网格未过线标为 `>1537`，不外推为永远不可能。

下表每格为“全答首次 / own_cal 首次”；星号表示后来回穿。完整 80 格及 Wilson 上界版本见 [TMIN_FULL.md](../results/exploratory_20261116/TMIN_FULL.md)。
''')
    rows = []
    for proc in ('separated_wide', 'separated_narrow'):
        for a in phase['grid']['alphas']:
            fields = []
            for r2 in (.8, .95, .995, 1.):
                row = next(c for c in phase['t_min'] if c['process']==proc and c['alpha']==a and c['r2_population']==r2)
                def cross(rule):
                    q = row[rule]
                    return str(q['first_observed_below_1pct'] or '>1537')+('*' if q['below_then_above_on_grid'] else '')
                fields.append(cross('all')+' / '+cross('own_cal'))
            rows.append([proc.replace('separated_', ''), a]+fields)
    parts.append(table(['过程', 'α', 'R²=.80', 'R²=.95', 'R²=.995', 'R²=1'], rows))
    parts.append(r'''
正面但有条件的例子：wide、α=.25、R²=.80 的 own_cal 在 T769 为 17/1920=.8854%，Wilson 上界 1.4134%；T1153 为 10/1981=.5048%，上界 .9268%，到 T1537 为 .2422%。同条件全答到 T1537 仍为 8.911%。narrow、α=.25、R²=.80 的 own_cal 在 T1537 仍为 1.344%；wide、α=4、R²=.95 为 2.389%。本次支持“部分设置可通过长度与拒答改善”，不支持“所有窄间隔或慢动力学总可由延长 T 修复”。

固定阈值覆盖也会随 T 显著变：wide、α=.25、R²=.80 从 T65 的 43.31% 降至 T1537 的 12.96%。在 T193、R²=.995，wide 的 slower/baseline/faster 覆盖为 33.28%/50.56%/42.36%，narrow 为 30.03%/50.07%/41.77%。固定阈值与 own_cal 的约 50% 覆盖不能混报。

### 4.3 REF 偏差揭示了为何“完美读出仍非零”

主目标物理分母为 1，预测分母为独立有限 REF 的样本方差。因此即使预测逐状态等于真值，评分分母仍不同。R²=1 的 100 格主错误率范围 .4883%–11.1084%；**把真值也除以同一物理 REF 后，100/100 格严格 0 错**。这既是算子核验，也是对“高 R² 必然低错误”的条件说明。

narrow、α=1、R²=1：T65/T193/T1537 的全答错误为 .6592%/1.0010%/2.4170%。同一个固定 REF 的轴间偏差不会因增加 TEST 窗口长度而消失。40 个 narrow 的 (α,R²) 条件中有 28 个在本次 REF 下预测总体 2–3 gap 变负，wide 为 0/40；例如 narrow、α=4、R²=1，真实总体 gap=.004932，条件 REF 后为 −.008587。这些数字条件于一次 REF 实现；没有重复 REF 得到一般失效频率，也不证明所有长窗口最终都变坏。

![All-answer phase](../results/exploratory_20261116/figures/phase_all_risk.svg)

![Selective phase](../results/exploratory_20261116/figures/phase_own_cal_risk.svg)

## 5. 128 个封存神经网络锚点：实测 vs 沙盒

128=16 A4 +16 A3 +32 A2 +64 B。所有 AUDIT、REF、rho、α、T、来源 SHA 和逐格差值都在 `probe2_summary.json.anchors`。分别按真实 AUDIT 的 mean/min 对相应 T193、过程和 α 网格做最近邻及相邻 R² 线性插值；插值仅为描述，不是新运行或拟合后的预测保证。完整逐项表（含选择规则、坐标距离）见 [ANCHOR_COMPARISON.md](../results/exploratory_20261116/ANCHOR_COMPARISON.md)。

下表按 8 模型、相同过程和实验条件按窗口数汇总。括号中差值为实测减对应插值预测；coupled 只与 c=0 代理比较。
''')
    rows = []
    for a in anchors['aggregate']:
        rows.append([a['experiment']+' '+a['arm']+' '+a['regime'], a['process'].replace('separated_', ''), pct(a['actual_all_risk']),
                     pct(a['mean_interpolated_phase_risk'])+f" ({a['mean_difference_pp']:+.3f}pp)",
                     pct(a['min_interpolated_phase_risk'])+f" ({a['min_difference_pp']:+.3f}pp)", 'c=0 proxy' if a['coupling_proxy'] else 'diagonal'])
    parts.append(table(['实验', '过程', '实测全答', 'mean-R² 插值', 'min-R² 插值', '动力学匹配'], rows))
    parts.append(r'''
排除 16 个 coupled 代理，112 格 mean/min 投射的未再拟合预测决定系数分别为 −1.0870/.1213，MAE 为 1.976/1.236 pp，预测错误与实际错误的 Spearman 为 .369/.403；SNR 与真实错误的 Spearman 为 −.318/−.311。负决定系数说明该原样代理甚至不如用本批均值预测；这些是共享模型/轨迹的外部描述性误差，不是独立样本的统计显著性。

A2 weak 实测平均 8.794%，mean/min 代理只为 4.636%/6.187%；两种投射均在 16/16 格低估。strong 实测 4.352%，代理 2.987%/3.679%；weak−strong 实际增加 4.442 pp，代理只增加 1.649/2.508 pp。由此可排除“指定的各向同性白噪声加 scalar mean/min R² 已解释实测误差”的说法；**不能把剩余差异全部归因为残差方向**，因为真实各轴异质性、时间谱、REF、模型误差与插值偏差没有单独控制。

选择规则的对应严格保留：A3 g50、A2 fixed、B fixed 数值门限相同；A3 g20/g80 与 A4 nominal/adjusted（每格 TEST 内取 409/2048）没有匹配的本次 phase 规则，所以只列实测，不虚构选择性预测。B recalibrated 是每 regime 混池 16 model-process CAL 单元的中位数；phase own_cal 是单条件中位数，均约 50% 覆盖但协议不等价。B coupled 为 `A[1,2]=.12, Q=I−AAᵀ`，未在对角 800 格里模拟它，不把 c=0 代理误差当机制检验。A2、A3/A4、B 的 AUDIT 来源也分别保留，未用一个共同 R² 替代。

## 6. 快轴退化与 REF 归一化压力测试

ρ=[.96,.95,.70,.68]；慢轴读出 `z+√.06 ξ`（总体 MSE-R²=.94）。快轴读出 `√s z_fast+λ z_donor+σ ξ`，供体分别为真实慢轴 0/1。请求网格为 7 个 s ×4 个 λ ×2 个噪声分支，共 56 格。CAL2048/TEST4096/REF32768/AUDIT32768，全部独立面板；格间共享面板用于配对比较。额外 `s=.0001/0` 的 16 格显式另列，没有替代请求参数。

### 6.1 决定退化的是相对方差成分

总体快轴方差是 `s+λ²+σ²`，归一化总体慢度是

\[
\overline s_f=\frac{s\mu_f+\lambda^2\mu_d+2\sigma^2}{s+\lambda^2+\sigma^2},
\quad \mu_i=2\operatorname{avg}_{\ell}(1-\rho_i^\ell).
\]

分母变小本身不是原因：分子同时缩放；泄漏只有在其**相对方差份额**变大时才主导。设 b 是两个正常慢读出中较大的总体慢度，则该快轴超过慢集边界当且仅当（无并列时）

\[
s< s_{crit}=\frac{\lambda^2(b-\mu_d)-\sigma^2(2-b)}{\mu_f-b},
\]

且右侧为正。σ=.02、λ=.1 时，轴2/3 的临界 s 为 .00195517/.00023753，均低于请求下限 .01；σ=.1 时两者为负，白噪声地板阻止此总体劫持。λ=.1、s=.01、σ=.02 的泄漏方差占比为 49.02%；绝对 .1 的振幅此时已不再是相对微扰。

**56 个请求格均没有总体快轴进入 top2，但有限窗口与 own_cal 仍会严重失效。** 下面的 min R² 是独立 AUDIT 四轴最小值；高信号时最小值来自慢轴。
''')
    rows = []
    for c in stress['requested_cells']+stress['additional_limit_controls']:
        if c['fast_signal_fraction'] in (.9, .05, .01, .0001) and c['leakage_amplitude'] in (0., .1):
            rows.append(['额外' if c['extra_limit_control'] else '请求', c['noise_sd'], c['fast_signal_fraction'], c['leakage_amplitude'],
                         f"{c['audit_r2']['min']:.4f}", pct(c['all']['risk']), pct(c['own_cal']['risk']),
                         pct(c['own_cal']['coverage']), '是' if c['population_fast_axis_enters'] else '否'])
    parts.append(table(['范围', 'σ', 's', 'λ', 'AUDIT min R²', '全答错误', 'own_cal 错误', 'own 覆盖', '总体快轴进入'], rows))
    parts.append(r'''
主网格最坏 σ=.02、s=.01、λ=.1：全答 1350/4096=32.959%，own_cal 488/2089=23.360%；同 s 无泄漏为 106/4096=2.588%，own_cal 0/2050。增加 σ=.1 后最坏 own_cal 为 3.532%，说明白噪声地板可削弱慢泄漏的相对主导性，却不自动保证低错率。额外 s=.0001、λ=.1、σ=.02 才跨总体边界，观察到全答 93.408%、own_cal 90.591%；此行仅是极限控制。

### 6.2 `.85` 门槛的地位

每个噪声分支中 12 个请求格会被实际 min AUDIT-R²<.85 拒绝，其中 11 格 own_cal 观察错率仍 <1%。例如 λ=0、s=.01、σ=.02，AUDIT min R²=.18947，own_cal 0/2050，描述性 Wilson 上界 .1870%。这说明门槛可以保守；另有低于 1% 的点其 Wilson 上界仍超过 1%，不能统称“已认证安全”。

更直接的必要性反例是纯正缩放：若 `zhat_j=a_j z_j, a_j>0`，则 `S(zhat_j)/Var_REF(zhat_j)=S(z_j)/Var_REF(z_j)` 每个样本精确成立，而 MSE-R²=`1−(a_j−1)²` 可低于 .85。共享 REF 真值时此操作保留零错；已知物理方差 1 的主口径下错误也与缩放 a 无关。因此 .85 不能是归一化慢集正确性的理论必要阈值。

反过来，两个分支各 16 个准入格虽然本次 own_cal 均 0 错，全答最高仍约 3.20%/3.42%；准入门也不是全答 <1% 的充分保证。合理定位是：AUDIT 限制论文所声称的“物理因素恢复”前提；部署风险还需独立、协议一致的选择性校准与验证。

![Stress probe](../results/exploratory_20261116/figures/stress.svg)

## 7. 可写入机制章节的结论，以及最小真实确认

可写成有条件、可证伪的两步论述：（一）近似 O(d) 因素表征消除了一个非线性字典歧义来源，但可靠慢子空间还要求残差谱偏置低于真实 lag eigengap，并控制有限 CAL 白化误差；（二）物理慢因素选择依赖归一化评分误差与同窗口 margin，而不是仅依赖总体动力学 SNR、平均 R² 或更长 T。若逐轴评分误差最大为 η，则同窗真实分差大于 2η 是足够的集合稳定条件；它与前序理论主线一致，并非本次拟合出来的新定律。

这比“唯有可辨识性”“单一 SNR 主曲线”“T_min 普适定律”“.85 必要性”更受当前证据支持。是否达到 ICML 创新性与主实验证据要求，不能由本次合成沙盒判定。推荐把残差谱偏置边界作为正面机制核心，把谐波对照、有限 REF 和快轴压力作为条件与失效图。

后续最小确认应在已有严格划分协议下进行，当前没有启动任何新训练或服务器实验：

1. 冻结真实编码器与读出，测量残差零阶/滞后协方差的方向与谱；用独立 CAL 估计误差和 eigengap，检查“偏置/间隔”和“抽样/间隔”两轴是否预示子空间失效。解混拟合仍无标签；真值只用于独立诊断。
2. 固定模型，重复独立 REF 并改变 REF 预算，分离 REF 分母偏差与轨迹长度效应；不要只重抽 TEST。以相同窗口定义记录 score perturbation、分差、错误与接受事件。
3. 在实际 Layer1/Layer2 结果返回后核对逐轴 AUDIT、预测 REF 方差与慢泄漏份额，验证是否存在本压力模型中的相对方差转变。Layer2 信息抹除尚不能由本次模拟宣称已发生。
4. 用未参与图形选择的模型/seed/过程确认阈值和最终曲线；对多设置与自适应选择重新设定同时保证，不能把本次逐格 Wilson 区间当封存 CP 证书。

## 8. 算子核验、产物与封存保护

独立核验直接从 `metrics.py` 的 AST 取出三个纯 NumPy 评分函数，未导入其 SciPy 统计模块；观测公式仅把原函数的 sin/stack/square 原语转写成 NumPy，未导入 PyTorch。流式充分统计与原评分的最大差为 1.78×10⁻¹⁵，Gram 投影与原评分最大差 3.55×10⁻¹⁴，精确解析方差与独立稠密 trace 核对最大差 1.67×10⁻¹⁶；稳定并列排序一致，匹配 REF 的完美读出 100 格全部严格零错。压力测试充分统计核对最大差 1.33×10⁻¹⁵。

两份完整主结果已用相同种子独立复跑，去除计时元数据后逐项完全一致，见 [validation.json](../results/exploratory_20261116/validation.json)。测试提供算子与复现核验，不等于对真实模型机制的科学验证。所有新增 Python 文件还进行 Python3.11 语法检查；仅在实际可用 Python3.13.11 上数值执行。

全量 SHA-256 审核覆盖任务前 47,786 个受保护文件、33,787,529,235 字节：`reports/`、`work/hcf_stable_id_20261027/`、`archive/`、`src/slowrepr/`、前序理论核验结果，以及已有理论文档和核验脚本。前后完整清单及变化计数在 [INTEGRITY.json](../results/exploratory_20261116/INTEGRITY.json)、[protected_before.json](../results/exploratory_20261116/protected_before.json)、[protected_after.json](../results/exploratory_20261116/protected_after.json)。最终以该收据的 `changed/missing/added` 空数组与仓库检查 `failures=[]` 为准；不把先前已有工作区改动说成干净状态。

主要入口：

- [探针一脚本](../scripts/exploratory_20261116/probe1_harmonic_and_subspace.py) 与 [probe1_summary.json](../results/exploratory_20261116/probe1_summary.json)：全部 162 条谐波记录、1260 条拟合记录、420 格均值与总体临界值。
- [探针二脚本](../scripts/exploratory_20261116/probe2_phase_diagram_alignment.py) 与 [probe2_summary.json](../results/exploratory_20261116/probe2_summary.json)：全部 800 格、128 锚点、56 请求与16额外压力控制。
- [phase_cells.csv](../results/exploratory_20261116/phase_cells.csv)、[anchor_cells.csv](../results/exploratory_20261116/anchor_cells.csv)、[stress_cells.csv](../results/exploratory_20261116/stress_cells.csv)：平表入口；完整统计语义以 JSON 为准。
- [图形生成脚本](../scripts/exploratory_20261116/render_figures.py)、[逐项表导出脚本](../scripts/exploratory_20261116/export_tables.py)、[独立复现与核验](../scripts/exploratory_20261116/verify_sandbox_alignment.py)、[保护核验](../scripts/exploratory_20261116/verify_integrity.py)。

所有输出 JSON 第一键、Markdown 第一行均标注 exploratory_numerical_sandbox；图也标记同一证据等级。封存结果仍按原证据等级存在于原路径，本次只读投射不会将合成证据升级为神经网络验证。
''')
    integrity_path = OUT/'INTEGRITY.json'
    if integrity_path.exists():
        integrity = json.loads(integrity_path.read_text())
        check = integrity['repository_check']
        parts.append(f"最终核验已通过：{integrity['protected_files']:,} 个受保护文件，"
                     f"变化 {len(integrity['changed'])}、缺失 {len(integrity['missing'])}、新增 {len(integrity['added'])}；"
                     f"全量哈希耗时 {integrity['hash_scan_seconds']:.3f} 秒。仓库只读检查耗时 {check['seconds']:.3f} 秒，"
                     f"核对 {check['result']['sealed_sources']} 个封存源，`failures=[]`。\n")
    (ROOT/'docs/EXPLORATORY_MECHANISM_PROBES_20261116.md').write_text('\n'.join(parts)+'\n')
    print(json.dumps({'evidence_tier': TIER, 'report': 'docs/EXPLORATORY_MECHANISM_PROBES_20261116.md', 'status': 'rendered_from_receipts'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
