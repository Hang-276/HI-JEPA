# Round 2：实际Fine的仿射核心、非仿射余项与速度排序

日期：2026-11-04。承接Stage A+，固定四槽12父、原目标、H=(4,16,32)、原Fine和G0。**本轮只读已有TRAIN/CAL缓存，完成预声明的12次TRAIN OLS与12次polar-SVD；没有新神经训练、前向、物理数据、FINAL数组读取或GPU执行。**

执行前已检查输入路径和原审计脚本，并写入 [固定范围协议](NONLINEAR_REMAINDER_PROTOCOL.md)。协议SHA在拟合前记录，执行后核对未变。完整逐轴、逐horizon、全部target/non-target pair见 [NONLINEAR_REMAINDER_AUDIT.json](NONLINEAR_REMAINDER_AUDIT.json)，复算入口为 [nonlinear_remainder_audit.py](nonlinear_remainder_audit.py)。

## 1 主要结论

1. **已存rich Fine在这些数组上可被一般仿射核心较好近似，但不是已证明的正交剩余误差族。** TRAIN上最差轴非仿射OLS余项MSE为.000474–.001569（actual Fine各轴TRAIN方差为1）；CAL对应为.000452–.001402。A的奇异值覆盖.962344–1.067261；\(\|A-Q\|_F=.021007\text{–}.078574\)。这些是事后拟合的数值，不是NLL或总体结构的保证。
2. **小静态OLS余项不消除动态交叉项。** TRAIN静态预测子/余项交叉矩最大约1.45e−15；但affine→actual的增量交叉矩可达.008913。OLS只约束TRAIN同刻矩，不约束跨时刻矩或截断端点上的同刻矩。
3. **归一化修正不可省略。** 对general affine或polar core，各自TRAIN方差不同。实际score误差需要同时计入增量cross、余项增量能量及TRAIN方差差。本轮全部恒等式最大浮点残差约3.51e−14。polar→actual时两项可约.06而彼此抵消，直接逐项取绝对值仍会给空泛界。
4. **Cauchy充分条件仍可失败。** TRAIN affine→actual的pair-Cauchy界只覆盖9/12父，FHN ε=.5三父全部失败；有真值的三项pair triangle界覆盖12/12。polar→actual即便使用有真值三项界也只覆盖9/12，不能只报有利分解。
5. **新增的支持敏感性负例必须保留。** Q0固定为TRAIN、只将Qh换为已有CAL合法端点后，FHN ε=.5真物理target margin为−.0209892692，而TRAIN为+.0206498870；actual Fine三父及两种core在CAL也都出现负target margin。这来自**一个已有24条轨迹的CAL物理实例**，不是三个独立环境；**不是部署G0错选**。原G0一直使用原TRAIN选集[1,3]，从未改为CAL选轴。

## 2 固定输入、尺度与拟合约定

四槽固定为AR rich fresh1/fresh2、FHN fresh ε=.08/.5，每槽seed0/1/2全部保留。AR物理目标[0,1]，FHN[1,3]。

| 槽 | TRAIN状态数组 | 既有CAL状态数组 |
|---|---|---|
| AR rich fresh1/fresh2（各槽） | 128×129×4 | 32×129×4 |
| FHN fresh ε=.08/.5（各槽） | 96×193×4 | 24×193×4 |

使用原冻结CAL一一匹配、slopes、intercept将Fine对齐到物理顺序；没有重拟合轴映射或使用稠密映射重新宣告原轴成功。物理V和Fine W分别用自己的完整TRAIN均值/标准差作逐轴标准化（FP64、ddof0）。同一TRAIN尺度用于CAL，**CAL不自行重新标准化**。

每父唯一一次OLS使用全部TRAIN点、全部四物理坐标和截距：

\[
w=Av+b\quad\text{（列向量约定）},
\qquad W_{rows}=V_{rows}A^T+b.
\]

`numpy.linalg.lstsq(..., rcond=None)`，没有ridge、子空间选择、超参或seed搜索。12个设计矩阵均秩5，无N/A或补救。A的polar分解取

\[
A=U_A\Sigma V_A^T,\qquad Q=U_AV_A^T,
\]

不强制det(Q)=1、不另作数据加权Procrustes。Q只是A在Frobenius距离下的最近正交因子。

固定两个core与两个余项：

\[
Z_A=Av+b,\quad Z_Q=Qv+b,
\quad N=(A-Q)v,\quad R=W-Z_A,
\quad W=Z_Q+N+R.
\]

R称“非仿射OLS余项”，仅表示当前有限数组上不能由该全仿射拟合解释的部分；不把它单独归因为某一种神经网络非线性机制。全部拟合使用真值，只用于科学诊断，不修改实际Fine/G0。

## 3 正确的score误差恒等式

### 命题 R1，Proved（固定支持上的确定性代数）

对任意core Z、output W=Z+R，所有状态尺度来自固定Q0=TRAIN。定义

\[
V_j=\operatorname{Var}_{Q0}(Z_j),\quad
c_{0j}=\operatorname{Cov}_{Q0}(Z_j,R_j),\quad
v_{rj}=\operatorname{Var}_{Q0}(R_j),
\quad V_{wj}=V_j+2c_{0j}+v_{rj}>0.
\]

在Qh=TRAIN或CAL的各h合法端点上，定义

\[
A_{jh}=\mathbb E_{Qh}(\Delta_h Z_j)^2,\quad
C_{jh}=\mathbb E_{Qh}(\Delta_h Z_j\Delta_h R_j),\quad
R_{jh}=\mathbb E_{Qh}(\Delta_h R_j)^2,
\qquad S_{Z,jh}=A_{jh}/V_j.
\]

则

\[
\boxed{S_{W,jh}-S_{Z,jh}
=\frac{2C_{jh}+R_{jh}-S_{Z,jh}(2c_{0j}+v_{rj})}{V_{wj}}.}
\]

**证明。** \(\mathbb E(\Delta W)^2=A+2C+R\)，且\(\operatorname{Var}_{Q0}(W)=V+2c_0+v_r\)。两个归一化score通分相减即可。不同h允许不同Qh，不要求平稳、时间独立或增量均值为零。

CAL公式中的 \(c_0,v_r,V,V_w\) **仍来自TRAIN**；只有A/C/R与分数的增量支持改为CAL。将CAL方差放入同一主公式会改变研究对象，不是当前审计。

对h平均后记三种贡献

\[
\alpha_j=2\overline C_j/V_{wj},\quad
\beta_j=\overline R_j/V_{wj},\quad
\nu_j=-\overline S_{Z,j}(2c_{0j}+v_{rj})/V_{wj}.
\]

则逐轴误差 \(\delta_j=\alpha_j+\beta_j+\nu_j\)，目标a/非目标b的pair误差为

\[
D_{ab}=(\alpha_b-\alpha_a)+(\beta_b-\beta_a)+(\nu_b-\nu_a).
\]

JSON对三种固定分解 `affine_to_actual`、`polar_to_affine`、`polar_to_actual`，在TRAIN/CAL分别保存全部逐轴、逐h与四个target pair。没有仅选择最有利core。

### OLS静态正交不等于动态正交

TRAIN OLS包含截距，故R与V、常数的样本内积为零；Z_A与N均为V的仿射函数，TRAIN上\(\operatorname{Cov}(Z_A,R)=0\)与\(\operatorname{Cov}(N,R)=0\)（至浮点精度）。但是

\[
\mathbb E_{Qh}[\Delta Z\Delta R]
=\mathbb E_{Qh}[Z_+R_++Z_-R_--Z_+R_--Z_-R_+]
\]

含有不受OLS约束的跨时刻项；两个同刻项还使用截断端点支持，未必等于全TRAIN Q0矩。因此不能在score公式里令C=0。CAL也没有样本OLS正交保证。

## 4 非精确安全界与其失败

由Cauchy–Schwarz，一个保留实测TRAIN方差差的逐轴界为

\[
B^{CS}_j=\frac1{|H|}\sum_h
\frac{2\sqrt{A_{jh}R_{jh}}+R_{jh}
+|S_{Z,jh}|\,|2c_{0j}+v_{rj}|}{V_{wj}}.
\]

若连state cross也仅用范数控制，可把\(|2c_0+v_r|\)替换为\(2\sqrt{Vv_r}+v_r\)，得更保守的\(B^{allCS}\)。两者都是非精确安全界，且仍需要真值拟合core/余项；不是无标签证书。

逐pair可用 \(B^{CS}_{ab}=B^{CS}_a+B^{CS}_b\)。另一个有真值的三项相关矩界为

\[
B^{corr}_{ab}=|\alpha_b-\alpha_a|+|\beta_b-\beta_a|+|\nu_b-\nu_a|.
\]

core目标gap为g_Z时，\(g_Z>B_{ab}\)足以保证output对应pair仍正确；全部四pair均满足才计一父通过。即使exact排序正确，安全界也允许失败；若core本身gap为负，误差足够小不能提供“保持正确”保证。

下列计数不是原Fine/G0成功率，而是**固定事后分解在指定Qh上的充分条件覆盖数**。

| 支持 | 固定core→output | core/output的正目标margin父数 | CS（实测Q0修正） | 全范数CS | 真值三项pair界 |
|---|---|---:|---:|---:|---:|
| TRAIN | affine→actual | 12/12、12/12 | 9/12 | 9/12 | 12/12 |
| TRAIN | polar→affine | 12/12、12/12 | 9/12 | 6/12 | 9/12 |
| TRAIN | polar→actual | 12/12、12/12 | 8/12 | 6/12 | 9/12 |
| CAL，固定TRAIN尺度 | affine→actual | 9/12、9/12 | 9/12 | 6/12 | 9/12 |
| CAL，固定TRAIN尺度 | polar→affine | 9/12、9/12 | 8/12 | 6/12 | 9/12 |
| CAL，固定TRAIN尺度 | polar→actual | 9/12、9/12 | 6/12 | 6/12 | 9/12 |

TRAIN affine→actual的CS失败全部是FHN ε=.5三父；同样三父使polar→actual的真值三项界失败。不能因为一般仿射拟合更好，就跳过polar近似误差或把actual Fine视为严格正交变换。

### 方差修正与cross抵消的实例

FHN ε=.5 rep0，TRAIN的affine→actual pair(a=1,b=2)：core gap=.0209694210，actual gap=.0226111038。三项依次为

\[
+.0017833592\quad +.0004491953\quad -.0005908718,
\]

合计exact差+.0016416828。相关矩界=.0028234264，小于core gap；CS界=.1114601284，大于core gap，仍失败。

同父的polar→actual pair(a=3,b=2)：core gap=.0494890093、actual gap=.0486849844。三项为

\[
-.0584928072\quad -.0010247785\quad +.0587135607,
\]

exact差仅−.0008040250，但三项triangle界=.1182311463，超过core gap。这里cross增量与方差修正强烈抵消；即便读取真值后，过早逐项取绝对值仍会空泛。这不否定恒等式或实际选对，而是明确指出该余项理论还会在哪个步骤损失精度。

## 5 全12父的静态近似与CAL外推余项

W在TRAIN每轴方差均为1。下表“最大MSE”是单父四轴中最大值；CAL一列用相同TRAIN映射/尺度，不在CAL重拟合或重标准化。它们不是物理逐轴恢复的原R²门，也没有预设一个MSE通过阈值。

| 槽/rep | TRAIN最大非仿射MSE | CAL最大非仿射MSE | TRAIN最大nonorth affine MSE | \(\|A-Q\|_F\) | \(\|Q-I\|_F\) |
|---|---:|---:|---:|---:|---:|
| AR1/0 | .001234379 | .001189355 | .000208167 | .0247413 | .0166019 |
| AR1/1 | .001174441 | .001236423 | .000989846 | .0504455 | .0543661 |
| AR1/2 | .001133284 | .001120732 | .000486114 | .0341629 | .0337860 |
| AR2/0 | .001172360 | .001009557 | .000518878 | .0408243 | .0308488 |
| AR2/1 | .001290205 | .001173570 | .000653527 | .0441746 | .0525181 |
| AR2/2 | .000977168 | .000888452 | .000523680 | .0383789 | .0439403 |
| FHN.08/0 | .001569119 | .001402409 | .002966754 | .0785739 | .0878213 |
| FHN.08/1 | .001427885 | .001296857 | .002493046 | .0748837 | .0817511 |
| FHN.08/2 | .001479156 | .001333325 | .001972955 | .0696995 | .0758924 |
| FHN.5/0 | .000498337 | .000483536 | .000688269 | .0413933 | .0458401 |
| FHN.5/1 | .000589015 | .000573817 | .000442018 | .0375255 | .0342171 |
| FHN.5/2 | .000474173 | .000452316 | .000158386 | .0210068 | .0287589 |

TRAIN非仿射余项最大MSE对应每轴解释方差至少约.998431。这个描述支持“这些缓存上存在较好的全仿射近似”，不支持“实际Fine只剩旋转”或“总体nonlinear remainder已受控”。尤其FHN ε=.08的nonorth affine项可大于非仿射余项；保留一般A和将其投影到Q是不同近似。

CAL只提供**对这个TRAIN拟合分解的支持外推诊断**。它已参与Fine观测checkpoint选择和原轴匹配，不能充当新的独立确认集。低CAL余项MSE也不能推出未见分布上的remainder或NLL界。

## 6 全部父的score margin：TRAIN与CAL不能混称

每个对象的方差分母固定为该对象自己的TRAIN Q0方差。表中CAL只替换增量端点Qh，绝非重新在CAL估计标准差。physical每槽只有一份，三父重复引用不是三个物理证据。

| 槽/rep | TRAIN physical | TRAIN actual | TRAIN affine | TRAIN polar | CAL physical | CAL actual | CAL affine | CAL polar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AR1/0 | .5650573 | .5652045 | .5650474 | .5648381 | .5777651 | .5800089 | .5756333 | .5761264 |
| AR1/1 | 同槽 | .5660102 | .5658987 | .5661984 | 同槽 | .5783584 | .5745684 | .5777425 |
| AR1/2 | 同槽 | .5671523 | .5662789 | .5657546 | 同槽 | .5794352 | .5771079 | .5766356 |
| AR2/0 | .6053248 | .6074647 | .6059404 | .6052716 | .5554060 | .5592149 | .5540987 | .5551500 |
| AR2/1 | 同槽 | .6054500 | .6043850 | .6045294 | 同槽 | .5546246 | .5529079 | .5555104 |
| AR2/2 | 同槽 | .6066583 | .6056932 | .6048289 | 同槽 | .5610293 | .5540237 | .5554076 |
| FHN.08/0 | .2319547 | .2124090 | .2225230 | .2265244 | .1844706 | .1623940 | .1730443 | .1794131 |
| FHN.08/1 | 同槽 | .2135865 | .2223682 | .2258483 | 同槽 | .1657515 | .1739972 | .1789365 |
| FHN.08/2 | 同槽 | .2134297 | .2234999 | .2273550 | 同槽 | .1635081 | .1743325 | .1802377 |
| FHN.5/0 | .0206499 | .0226111 | .0209694 | .0203242 | −.0209893 | −.0250364 | −.0227039 | −.0288341 |
| FHN.5/1 | 同槽 | .0246840 | .0224535 | .0216001 | 同槽 | −.0191293 | −.0179784 | −.0223650 |
| FHN.5/2 | 同槽 | .0219890 | .0210206 | .0208686 | 同槽 | −.0217749 | −.0226781 | −.0253499 |

### FHN ε=.5的新增有限支持负例

物理CAL分数为

\[
(1.9192096646,\;1.9401989338,\;1.9406296474,\;1.9203964555),
\]

按此Qh作假想top2会得[0,3]，不是预声明目标[1,3]；最小pair为(a=1,b=0)，gap=−.0209892692。这是**一个已有CAL物理实例的有限数组排序**。TRAIN物理分数仍选择[1,3]，正gap=.0206498870。

原部署G0从TRAIN确定并冻结，依然选[1,3]；本轮没有用CAL重新选G0、没有改原Fine＋G0联合成功数。CAL实际Fine分数负margin只说明在固定TRAIN尺度、换用CAL增量支持时，同一外部目标不再是该有限数组的严格最慢两轴。不能写为“新发现部署G0三个父失败”，也不能把同一CAL的三优化父当作三个独立负例。

这里Q0保持TRAIN未动，所以比旧`t≥11`同时改端点/标准化的oracle更直接展示：**仅更换有限增量支持也可能改变目标排序**。TRAIN/CAL使用同一生成配方和相同合法时间窗口，差别是已有不同轨迹样本及样本数；因此本结果首先是有限样本支持波动证据，并非受控动力学或分布干预，更不能证明总体margin为负。固定TRAIN分母本身也来自有限样本。旧caseA FHN ε=.5/t≥11的负oracle和后续两root样本路径12格成功均继续保留，不被本次结果覆盖或重标。

## 7 实际Fine能否接入κ理论

本次只证明存在一个在现有TRAIN上较好的**一般仿射拟合**，且在既有CAL上有近似量级的非仿射余项；没有证明：

- actual Fine是严格正交残差，或A的polar Q就是训练器剩余的真实误差机制；
- Q来自二维单pair旋转，或其角度可由上一轮κ/F反演；
- dense条件均值与对角高斯风险可精确profile当前实际encoder的完整误差；
- 小静态MSE能控制Jacobian、NLL、条件创新结构或未见支持；
- CAL观测NLL是总体profiled excess risk。

因此不能以本次Q矩阵代入κ公式，宣称已经获得实际Fine的风险认证。可以保留的条件接口是：若未来理论允许一般仿射core加可控余项，则本次提供了方差修正完整的score误差恒等式与已暴露数组中的数值检查；但最坏情况Cauchy界在FHN ε=.5仍空泛，有真值分解也不能替代无标签可估计性。

## 8 证据层级、复现与成本

本轮范围是**truth-assisted posthoc decomposition**。物理标签只用于这份审计，没有进入实际Fine训练、G0选择、checkpoint选点或阈值修改；没有新增物理数据或独立确认。

保存内容包括：全部12父OLS A/b与rank、polar Q与奇异值、TRAIN/CAL余项逐轴统计、3种transition×2个支持×4轴×3h的精确分解、每父全部4个target pair及安全界、四槽oracle、原CAL maps与G0的输入哈希。没有删掉任何负margin或失败界。

实际读取56个member/JSON记录、40个唯一既有文件；协议、脚本及每次输入读取都保存SHA256。实际12次full-TRAIN多输出OLS、12次4×4 polar-SVD、0次CAL拟合。数值执行约.29秒。12个原G0的FP64分数均与已存分数精确重现，原FP32选集保持一致。普通FP64计算的恒等式/界按1e−12容差核查，不称interval arithmetic形式化证书。

零新NN训练、零模型前向、零checkpoint反序列化、零新物理轨迹/随机数、零FINAL数组读取、零GPU任务；所有写入均位于本轮独立报告目录，正式代码、模型、评价门和历史结果未改。
