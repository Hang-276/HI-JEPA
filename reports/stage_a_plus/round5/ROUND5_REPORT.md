# Stage A+ Round 5：条件定量链已闭合，实际证书仍未建立

2026-11-04。承接 [Round 4](../round4/ROUND4_REPORT.md)。本轮完成尾部、近似平方闭包、参考分布迁移和慢度排序的条件定理；只读已有JSON metadata/报告，不读取科学数组、不加载模型、不训练、不连接服务器或使用GPU。

## 1. 本轮判断

**“近似 regime 后验→近似逐轴恢复→慢因素排序”的条件数学链已经补齐，具有明确的非空误差域。当前实际 Fine 尚未得到非空、无物理真值的保证。** 两个判断必须分开。

正进展有三项：去掉真实后验统一pmin；消去未知真实log-odds系数范数；将充分统计量近似误差δ以显式 **O(δ)** 界转成逐轴误差，并用可观测输出协方差门排除多个输出挤到同一源轴。再通过观测密度比、尾矩与重新归一化，接到G0。

实际限制也更具体：单用可观测cross-entropy作后验KL粗上界，必定无法通过本轮尾部吸收门；现有日志没有更紧的有效上界。另一方面，旧真值数值诊断已否决FHN ε=.5三个父的同支持分轴范数充分门。继续提高这条链的前端估计精度，也无法让这个既定末端充分门通过它们。

**路线建议：保留条件稳定性定理和失败边界；暂停将当前范数链作为实际Fine/G0证书的主线，不据此进入训练阶段。** 进一步投入实际认证，需要可验证的小posterior-excess上界和更紧的目标相关误差预算。当前并未建立这两项；Fine相对raw的独有方法价值也没有因本轮理论而增加。

## 2. 尾部：消除真实pmin与真实系数范数

固定合法context c=(x_t,a_t)，取某个baseline regime的真实Gaussian future law为anchor Q；混合future law为P，已知propensity π0>0。物理Gaussian参数只用于证明结构，算法不读取真D、真A或物理状态。

由head定义可观测后验q，在Ω={min_u q_u≥ε}内工作。设A是未归一的真实posterior KL上界 E_P[1Ω KL(p‖q)]，α∈(0,1)固定，φα=1−α+αlogα。真实p没有被假设远离0。

若某类p_u<αq_u，KL至少εφα；所以anchor下的坏集合质量由

\[
\bar d=t+\min\{1-t,A/(\pi_0\epsilon\phi_\alpha)\},
\quad t=Q(\Omega^c)
\]

控制。真实log-odds是Gaussian坐标上的二次多项式，hypercontractivity给小集合收缩系数

\[
k(\bar d)=\inf_{p\ge2}(p-1)\bar d^{1/2-1/p}.
\]

这项计算在dbar<e⁻⁴时允许严格吸收。令N为head log-odds的anchor L²范数，g为好集合误差预算，h为head坏集合/尾部能量预算，则完整odds误差满足

\[
\boxed{R\le\frac{g+kN+h}{1-k}.}
\]

未知真实odds范数通过 ||真实odds||≤R+N 被移到左侧。N、t、head尾部能量都是观测law与固定模型的函数，不依赖物理标签。再除以同一坐标规范下head contrast矩阵的最小奇异值s，得到全域anchor充分统计量误差δ≤R/s。

这里的“可观测函数”仍需合法总体置信界；固定连续context没有自动提供重复未来样本。渐近非空还必须共同控制K_q²d→0，其中K_q随log(1/ε)增长；仅A/ε→0不够。详细公式、修正后的联合条件和独审见 [TAIL_POSTERIOR_BOUND](TAIL_POSTERIOR_BOUND.md)。

### 可观测交叉熵为什么在这里不够

A≤CEΩ是合法粗界。但Ω上每类q≤1−8ε，因此CEΩ≥P(Ω){−log(1−8ε)}。代入本轮坏集合预算必使dbar=1、k=1，无法吸收；若Q(Ω)=0也直接有t=1。

这严格否决的是**只使用绝对CE的这一条充分证法**，不是所有无标签方法。可行的小A必须来自更有信息的风险上界。例如在固定模型类包含真实posterior、uniform generalization误差≤ζ、经验最优差距≤ξ且逼近误差≤a_app时，proper log-loss理论给A≤2ζ+ξ+a_app（全域界也控制restricted量）。当前Fine按观测NLL训练，现有记录没有这套posterior目标的全局差距或覆盖证明；与一个普通参考分类器比较，也不能自动给上界。

若能证明完整条件观测NLL的总体excess J，则joint-KL分解还给posterior KL≤J；但存储的CAL NLL没有正确entropy baseline，仍不能直接充当J。数据交叉拟合可处理部分复用偏差，不能自动提供上述oracle、模型包含或优化条件。

## 3. 近似平方闭包：从δ到逐轴误差

在同一Gaussian anchor中将真实因素逐轴标准化为V~N(0,I_d)，Fine输出Y按anchor中心化且每轴方差1。若Y_j与Y_j²分别接近加性二次空间span{1,V_i,V_i²}，误差为δ1j、δ2j，令

\[
e_j=\delta_{2j}+2\delta_{1j},\quad
L_j=1-\delta_{1j}^2-2d e_j,
\]

\[
E_j=\delta_{1j}^2+
\frac{[8d^2+16d(d-1)/e^2]e_j^2}{L_j}.
\]

其中分母常数中的e为自然对数的底。在L_j>0时，存在单一物理轴i_j与系数m_j，使

\[
\|Y_j-m_jV_{i_j}\|_2^2\le E_j.
\]

若E_j<1，还得到同规范signed-unit轴差平方≤2(1−√(1−E_j))。若观测输出的anchor协方差满足Cov(Y)≽λI，且max_{j≠k}(E_j+E_k)<λ，则所有i_j互异，构成置换。**可逆本身不足以给这个定量非塌缩条件。**

证明使用有界Gaussian tilt×Hermite测试函数提取四次、三次和交叉一次系数；三次项使最终轴L²误差成为O(δ)，避免只由四次项得到更松的O(√δ)。没有调用未经定量的多项式闭包连续性。

d=4、总span误差≤δ时，显式版为

\[
L=1-\delta^2-24\delta,\quad
E(\delta)=\delta^2+
\frac{9(128+192/e^2)\delta^2}{L}.
\]

完整证明、近塌缩反例和17项固定核验见 [QUANTITATIVE_CLOSURE](QUANTITATIVE_CLOSURE.md)。tail T7已在anchor中包含π0转换，接入这里不再重复除√π0；输出中心/尺度若未规范，还须按附录A24转换误差。

## 4. 从anchor搬到实际Q0，并控制G0排序

设w=dQ0/dQ，是同一观测空间中的密度比。给定W>0、p>1，定义head输出尾能量T_Y,j=E_Q0[Y_j²1(w>W)]、密度比尾矩M_w,p=E_Q[w^p1(w>W)]。Gaussian绝对矩常数C_p可显式计算，则

\[
E_{Q_0}(Y_j-m_jV_{i_j})^2
\le W E_j+2T_{Y,j}+2C_p M_{w,p}^{1/p}=:B_j.
\]

搬到Q0后各自重新标准化，轴差至多η_j=2√B_j/sd_Q0(Y_j)。多时距端点coverage再将其变成Fine与物理轴的分数误差β_j；若每个选中/未选中pair的Fine gap>β_a+β_b，便保留严格慢选集。

这一步没有直接输入未知物理Q0尾矩，但增加了可观测密度比及head尾矩的有效上界要求。绝对连续不保证某个p>1的密度比矩有限；窄innovation anchor与宽state分布之间可能严重放大误差。若Q0是离散经验TRAIN，连续anchor不能直接支配它，需要总体参考加统计控制，或另外的有限样本函数误差界。

有限条件定理和联合非空条件见 [TRANSPORT_AND_SELECTION](TRANSPORT_AND_SELECTION.md)。所有尺度、measure和端点预算都明确保留，未把anchor直接当成原G0 TRAIN分布。

## 5. 数学非空与实际非空的区别

在固定四维九regime Gaussian解析例中，tail预算得到三组span上界3.4013、.0266895、4.50015×10⁻⁵。前两组不能通过本轮闭包门；第三组得到anchor每轴误差上界约 **.00167679**。

但第三组使用的已知生成参数posterior-KL上界约 **2.22×10⁻²¹**，anchor尾概率约4.98×10⁻¹⁵。这是一个明确的条件可行例，不是必要误差率、最优样本复杂度或当前Fine的实测值。encoder在该解析例中是identity，head有极小指定偏差，也不证明非线性学习可达性。

再将第三组预算与单独固定的score矩( .3,.5,1.4,1.8 )、Q0=anchor、K=4组合，最小剩余gap约.8873。这个算术检查说明条件接口不恒空；这些score矩未被宣称来自同一个完整动态生成过程或实际Fine。两组失败仍保留。见 [CHAIN_BUDGET_CHECKS.json](CHAIN_BUDGET_CHECKS.json)。

**当前实际父模型没有任何一个被本轮宣布获得无标签证书。** 缺少有效A、anchor输出covariance下界、head总体尾矩、固定context统计覆盖及Q0迁移预算；数学非空示例不能填补它们。

## 6. 现有G0记录给出的真实门槛限制

仅用12份原生G0 metadata，在同一有限TRAIN支持上可反推共同每轴normalized L²误差η必须小于的充分门：

| 槽（各3父） | 共同η门范围 | 旧OLS数值oracle下界 | 共同门被否决 | 分轴同界被否决 |
|---|---:|---:|---:|---:|
| AR rich fresh1 | .04810–.04826 | .03366–.03513 | 0/3 | 0/3 |
| AR rich fresh2 | .05178–.05194 | .03126–.03592 | 0/3 | 0/3 |
| FHN ε=.08 | .02568–.02583 | .03779–.03961 | 3/3 | 0/3 |
| FHN ε=.5 | .00191–.00215 | .02178–.02427 | 3/3 | 3/3 |

旧全仿射OLS最小残差是任何单轴匹配误差的下界。本轮只把已存真值summary用于**否决当前充分界**，没有把它当作无标签接受证据。浮点OLS已扣正规方程误差和显式数值敏感性余量；这是FP64诊断，非interval形式证明。

共同门的6/12否决不能直接推广到分轴版本；分轴检查只否决FHN ε=.5三个父。其余9父未被下界否决，不等于通过。该结论只限同一有限TRAIN支持，也不否定尚未建立的总体或其他支持证书。

这解释了当前路线的实际困难：为了保证所有可能误差方向，范数界丢掉了signed动态交叉项及归一化修正的抵消。Round3的真值相关矩闭链仍能通过这些TRAIN父，实际选集也没有被本轮改判。更紧的目标相关证书仍可能存在，但需要额外论证和可观测输入。

完整48个pair、数值修正和scope见 [G0_ERROR_BUDGET_AUDIT](G0_ERROR_BUDGET_AUDIT.md)。

## 7. 可观测条件清单与去留

| 条件/量 | 是否需要物理标签 | 当前证据 |
|---|---|---|
| posterior q、head contrast、原生G0 scores | 函数可由模型与观测定义 | G0有metadata；posterior及contrast未评估 |
| 小的proper posterior KL上界A | 可定义于完全观测的(U,C,X′)律 | 无有效上界；单独CE在本证法中必失败 |
| 固定合法Gaussian anchor | 真实结构假设；不是物理数组输入 | 合成理想核兼容；当前连续context的估计/覆盖未建立 |
| head odds总体矩、tail和输出covariance下界 | 原理上可观测 | 84份二阶摘要不包含所需预算 |
| Q0/anchor密度比及尾矩 | 观测law函数 | 未估计；不存在默认有限domination |
| G0总体score统计误差、支持与外部目标一致性 | 部分可观测，另需结构/目标定义 | 现有经验margin不是总体置信界 |
| Fine相对同容量非线性表示的独有增量 | 比较证据 | 未建立；前轮raw同规则结果不变 |

本轮解决了此前“近似平方闭包没有量化”的数学缺口，也给出不依赖真实odds系数的尾部处理。继续把当前证法当成实际证书推进，已经碰到末端范数门对现有FHN的数值否决，以及前端风险上界的不可直接获得性。

因此建议保留本轮条件定理与反例作为研究产出，暂停扩张实际Fine证书/独有方法主张。若后续继续，应先证明更有信息的可观测posterior误差上界，并明确目标相关误差如何避开已暴露的松弛；当前结果不足以支持上服务器或扩大训练。论文级新颖性仍未确认，Gaussian/Hermite/hypercontractivity工具及有限样本nonlinear ICA近邻均不能忽略。

交付与独立复核见 [README](README.md)、[DELIVERY_CHECKS.json](DELIVERY_CHECKS.json)。原评价门、模型、旧报告和全部负结果保留。
