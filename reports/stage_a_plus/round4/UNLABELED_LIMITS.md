# Round 4：无标签余项、总体可辨识与有限数据的严格边界

日期：2026-11-04。执行范围先固定于 `UNLABELED_LIMITS_SCOPE.md`。承接 Round 2 的单向风险/余项证明及 Round 3 的局部仿射闭链；不改旧文件。本轮不读取物理数组、不加载或前向模型、不训练、不采样轨迹、不连接服务器或GPU。

## 1. 独立判断

**不能一概说“无物理标签就永远无法控制实际Fine余项”。** 若生成族正确、完整总体观测律已知且所需full-rank可辨识条件成立，因素等价类原则上由观测律决定，部分规范化后的余项也成为观测律的可识别functional。它未必有可实现且数值有效的估计器。

**也不能由定性full-rank推出当前有限TRAIN/CAL可以给出同时低误报且高通过概率的认证。** 本文件给两个严格近不可辨识构造：对每个τ>0都full-rank，但任意有限总数据量下可令两个不同因素解释难以区分。一例给轴恢复障碍；另一例使同一个已知encoder相对真因素的最佳仿射余项由0变为固定正量，同时观测分布仍任意接近。后者甚至保持创新variance下界、观测映射光滑性与可逆条件数不退化；退化的是辅助变化强度。

结论是：当前无标签输入没有免费关闭Round2/3的总体余项缺口。可以研究有完整模型覆盖的相容集证书；有限网络共识、低NLL、可逆性或raw坐标Jacobian小均不能直接代替其假设。

## 2. 先把待认证量写清楚

部署可读信息为x、a/u、Fine E及head、它们的导数、TRAIN/CAL和训练记录。以下均不是默认可读量：物理v、真实f或其inverse、真D、真值匹配、真A。

Round2的分解 \(T=E\circ f=Av+b+r(v)\) 需要一个**确定的核心定义**。任取A可以任意改变r，故“r是否小”本身还不是不依赖定义的科学问题。一个合理对象是给定Q0后，令Av+b为E(f(v))在真实因素affine span上的L²最佳投影；同时固定输出尺度和有限二阶矩。

若把潜因素换成允许的 \(v'=PDv+c\)，affine span不变，投影函数与余项r(X)不变。因而该**输出单位下的最佳affine residual**不是必然受潜变量缩放/置换影响的不可识别对象。相对innovation covariance也应连同A、D一起变换；\(C_u=AD_uA^T\)在该重参数化下不变。

反之，指定某个未经固定的A、某个任意物理单位的coordinatewise remainder或未命名target，可能只是规范未定，不能把它与统计不可辨识混为一谈。

## 3. 总体观测律下：何时可以，何时不可以

### 3.1 Full-rank成立时的正面结论（Conditional）

假定真实条件Gaussian生成族、context-free可逆混合、共同支持、辅助自然参数充分变化以及Stage A相应正则条件确实成立。若完整观测条件律已知，则因素映射可识别至permutation＋逐轴affine；这沿用已有conditional ICA理论及本项目Gaussian专门化，不是本轮首创。参考 [iVAE正文§4](https://proceedings.mlr.press/v108/khemakhem20a/khemakhem20a.pdf)及 [GIN Appendix A–B](https://arxiv.org/pdf/2001.04872)，不可把iVAE的一般充分统计量结论直接改写为全部情况下的轴affine结论。

在此条件下，真实因素的affine span也被识别，所以§2所定义的最佳affine投影、r(X)、同一支持下的多时距余项矩成为观测律的functional。条件创新余项 \(r(X')-E[r(X')\mid x,a,u]\) 和Cu也可在理论上沿该已识别生成解释定义。这里“不需要事先拿到物理标签”与“当前有限数据已经能准确算出它”是两回事。

若某个encoder/head在总体上**精确**给出真实观测条件密度，且它与真实物理解释都在规定可辨识模型族内，相应可辨识定理可以认证其坐标等价类。不能将“校准图看起来好”或“有限样本未拒绝”替代这种完整条件密度相等。

### 3.2 Rank不成立时的永久歧义（Proved construction）

令所有regime下潜状态均为标准Gaussian，观测解释一为X=S，解释二为X=g(S)，其中g是下面§5的Gaussian-preserving nonlinear twist。两者总体观测律完全相同；E(X)=X及标准Gaussian head在两者中同样完美。第一种解释下非线性余项为0，第二种为严格正量。因此这些物理余项不是该观测律的functional。

**本例rank退化，只用于证明退化族中的总体歧义。它不能否定rank成立时的总体定性辨识。** 下一节单独处理每个τ>0皆full-rank的有限数据障碍。

### 3.3 外部语义名称与统计慢集合

permutation等价下，一个任意外部名字如“物理变量w1”不会由无语义锚点的观测自动命名。此时可识别的是因素集合及其内在统计属性。若目标定义为相同Q0/Qh下唯一最慢k因素，它在允许重命名下可以良好定义；若预声明物理目标与该集合是否一致尚未知，则认证统计慢集合不等于认证那个外部目标。

这不是要求恢复时禁止置换，而是要求主张与允许的等价类一致。已知且轴对齐的干预/物理约束可能提供额外命名信息；本任务未默认拥有它们。

## 4. 每个模型均full-rank，有限数据仍无统一的低误报/高通过保证

### 定理 U1：四维九context两点障碍（Proved）

令U在{0,…,8}均匀。定义源条件族：u=0时 \(N(0,I_4)\)；u=j+1时均值为τej、covariance I；u=j+5时均值0、covariance \(I+\tau e_je_j^T\)，j=0,…,3，τ>0。

Gaussian自然参数 \((\mu_j/d_j,-1/(2d_j))\) 的八个非基准差形成一次/二次统计量上的block diagonal矩阵；每轴两个方向的系数分别为τ与τ/[2(1+τ)]，故rank为8，对每个τ>0成立。所有innovation variances在[1,1+τ]，没有小variance退化。

比较两个观测模型：

\[
P_\tau:X=S,\qquad Q_\tau:X=R_\theta S,
\]

其中R只在前两轴旋转，取θ=π/4。两者都满足上述full-rank latent条件族与可逆光滑混合。标准Gaussian公式给单样本联合(U,X)的KL

\[
\boxed{K_\theta(\tau)=D_{\rm KL}(P_\tau\|Q_\tau)
=\frac{\tau^2}{9}
\left[2(1-\cos\theta)+\frac{\sin^2\theta}{1+\tau}\right].}
\tag{U1}
\]

**证明。** 两个mean context各贡献τ²(1−cosθ)，两个variance context各贡献 \(\tfrac12\sin^2\theta\,\tau^2/(1+\tau)\)，其余context相同；除以9即可。determinants相同，不产生额外logdet项。

给定n个独立(U,X)，Pinsker与product KL给

\[
\operatorname{TV}(P_\tau^n,Q_\tau^n)
\le\sqrt{nK_\theta(\tau)/2}.
\tag{U2}
\]

任意检验/认证pass事件C都满足

\[
P_\tau(C^c)+Q_\tau(C)\ge1-\operatorname{TV}(P_\tau^n,Q_\tau^n).
\tag{U3}
\]

当C表示“E=identity的物理轴正确”时，P下它确实正确，Q下前两轴45°混合而非允许的逐轴affine，best scalar R²接近.5。可令τ按1/√n足够小，使TV≤1/4，则缺乏power与错误认证概率之和至少3/4。固定full-rank不排除这种序列。

该结论不否定总是abstain的安全判别器，也不排除独立随机pass所产生的严格正power；它排除的是在整个未给强度下界的full-rank类中，同时保证两端错误都小，或保证预定高通过概率的统一有限数据认证。具体地，若Q下误认证概率≤δ且TV≤1/4，则P下通过概率至多δ+1/4。经典testing-to-estimation背景见 [Bin Yu，Assouad, Fano, and Le Cam，1997](https://web.stanford.edu/class/stats300a/REFS/yu1997assouad.pdf)；U1的Gaussian构造与常数在此完整给出，不借摘要推断。

**训练记录及导数不增加超出数据的信息。** 若Fine/head、seed、优化轨迹及其导数由同一允许数据和独立随机性产生，它们是该数据的Markov kernel；total variation不增加。上述下界因此覆盖任意此类训练后诊断。若权重或结构另编码了外部物理inverse/参数，那是额外信息，已超出本任务输入。

### 正边界慢集合的时间扩展（Proved conditional construction）

U1不依赖慢度目标；如需保持一个正的slow-set gap，可取真动态矩阵 diag(.9,.9,.2,.2)，初态采用与该矩阵匹配且在前两轴各向同性的共同Gaussian；两个解释分别用I与Rθ观测。转移均值变换只旋转辅助mean，因为动态矩阵与R交换；一步条件KL仍为U1，trajectory KL按chain rule累加，无需把连续帧假装独立。τ=0时两慢轴对两快轴有正boundary gap；固定有限h与窗口的矩随τ连续，因此足够小τ仍有正gap。

这时无标签G0可能选对整个慢子空间，但E=identity在Q解释下仍未恢复慢块内两条物理轴，违反本项目严格联合Reliability。此扩展不将子空间正确改称轴恢复正确。

## 5. 非线性余项本身也可有限不可认证

### 定理 U2：full-rank近邻下0与固定正affine residual（Proved）

令第一二坐标z∈R²，t=||z||²，定义固定α≠0的径向twist

\[
g_\alpha(z)=R_{\alpha e^{-t}}z,
\]

其余两轴不变。半径保持不变，极角加一个只依赖半径的角度，所以g保持标准Gaussian，Jacobian determinant恒为1，inverse是同半径下的反向旋转。由
\(\|Dg\|,\|Dg^{-1}\|\le1+2|\alpha|/e\)，可逆条件数和导数不随τ退化。

在同一个U1 full-rank源族上比较
\(P_\tau:X=S\) 与 \(\widetilde Q_\tau:X=g_\alpha(S)\)。取已知encoder E=identity。P下最佳affine residual为0；当τ→0时，Q下该余项能量趋向一个严格正值。

**正余项的精确表达。** 在τ=0时，最佳affine核心的二维块是aI+bJ，其中

\[
a+ib=\sum_{k=0}^\infty
\frac{(i\alpha)^k}{k!(1+2k)^2},\qquad
\boxed{\delta_\alpha^2=2[1-|a+ib|^2]>0.}
\tag{U4}
\]

因为对Z~N(0,I2)，weighted measure \(||Z||^2/2\) 下的e^{−k||Z||²}期望为1/(1+2k)²。角度是该非退化measure上的非恒定随机变量，其单位复数均值模严格小于1。连续性与Gaussian矩控制保证τ>0足够小时余项仍大于δα²/2。该对象是最佳affine投影残差，不是任取一个坏A制造的假余项。

**观测距离。** 令P0为U独立标准Gaussian观测参考。源族满足

\[
K_0(\tau)=D_{\rm KL}(P_\tau\|P_0)
=\frac29[\tau^2+\tau-\log(1+\tau)]\le\tau^2/3.
\]

以 \(h^2=1-\int\sqrt{pq}\) 定义Hellinger距离。g保持P0，所以triangle、变换不变性及KL≥2h²给

\[
h^2(P_\tau,\widetilde Q_\tau)\le4h^2(P_\tau,P_0)\le2K_0(\tau).
\]

product affinity与TV–Hellinger关系于是给

\[
\boxed{\operatorname{TV}(P_\tau^n,\widetilde Q_\tau^n)
\le2\sqrt{nK_0(\tau)}\le2\tau\sqrt{n/3}.}
\tag{U5}
\]

取τ同时小于上述余项连续性所需阈值，并满足τ≤√3/(8√n)时，TV≤1/4。因此任何试图区分“最佳affine余项0”和“至少δα²/2”的有限数据方法，都面临U3相同的power/soundness限制；没有排除严格正但很小的随机通过概率。

**甚至低actual population excess也不提供无强度常数的统一余项界。** 对identity encoder和固定标准Gaussian head q=φ，g保持φ且detDg=1，所以两个解释中的完整Gaussian observation NLL excess均恰为K0(τ)→0；Q下最佳affine余项仍不消失。这不否定随可辨识强度退化的条件性模量，也不否定每个固定τ>0的总体唯一性。

本例属于可逆光滑生成族；本轮没有证明该固定g及inverse被当前有限Flow精确包含，更没有声称某个已训练Fine出现该失败。它是对一般无标签推论的理论边界。其IID源族也没有声明正slow margin；如要针对带固定正slow-gap的非线性余项类建立同样下界，需要另外检查动态结构，不能借上节线性AR扩展替它完成。

## 6. 哪些可观测诊断不充分

| 可观测量 | 确实可以说明什么 | 单独不能认证什么 |
|---|---|---|
| E可逆、cycle error小 | 信息未被编码器结构删除、数值inverse一致 | 物理轴或非线性余项小 |
| DE条件数/logdet/光滑性 | raw坐标下的映射几何 | Dr=DE(f(v))Df(v)−A，因Df与A未知 |
| encoded marginal whitening/Gaussianity | 该输出分布性质 | τ=0的twist反例具有完全相同输出分布 |
| pooled residual cross-covariance小 | 某个平均矩接近0 | conditional/regime矩相消、近弱签名、Gaussian family失配 |
| regime-wise residual诊断 | 可以拒绝某些head的条件独立/方差设定 | 有限样本未拒绝不等于完整posterior/density正确 |
| learned-head自然参数rank | 该拟合head输出的代数变化 | 真生成族、共同支持、实际误差或定量强度已经满足 |
| 多seed共识 | 已搜索解的重复性 | 所有相容解释的上界、真实语义 |
| G0 empirical margin/重估稳定 | 固定表示内的经验选集分离 | 物理目标与该表示的因素轴对应 |
| held-out observation NLL | 当前模型风险估计或与指定对照的风险差 | 正确的总体entropy baseline、全类最优风险与axis modulus |

这不是说这些诊断没有研究价值。它们适合拒绝、定位失配、估计指定模型的统计量；接受式语义认证需要把缺失前提补齐。特别是conditional posterior a.e正确比普通校准、分类accuracy或若干moment匹配强得多，不能互相替换。

## 7. 相容集合接口：逻辑上可行，当前尚不可执行为证书

已独立审查 `VERSION_SPACE_CERTIFICATE.md`。在完整模型类包含真实物理解释、完整observation loss的统一事件以及完整相容集表示半径上界成立时，证明正确：真实解释留在相容集内；signed-permutation quotient与同Q0逐轴标准化给相对真表示误差；端点coverage后用Fine自身score上界而不是未知physical gap作充分条件。

该接口不要求先显式估真r，但并未消除真模型包含和完整集合求上界的成本。找到一个相容且轴不同的解释足以拒绝small-diameter条件；有限候选全都一致只能给已搜索集合的直径，通常是完整集合直径的下界。

**重要支持限定：** 若Q0取离散经验TRAIN，而Qh来自不同CAL点，其端点measure通常不被经验TRAIN measure支配。不能从候选在TRAIN上的L²距离自动推出CAL或新支持距离；需要总体覆盖、明确支持关系、函数复杂度/导数条件或独立的统一泛化界。文稿已明确要求coverage，不应在后续应用中省略。

从候选群内的最慢集合到外部物理目标，还需要原目标一致性假设；version-space命题没有替研究者识别任意外部名字。它也没有把L²表示误差直接改称现有TEST R²门通过。

U1/U2与这类证书并不矛盾：在弱签名有限数据下，诚实覆盖集合应保留不同解释，直径可能大，证书应abstain；当固定τ>0、正确有限复杂度类且样本增加时，集合可能收缩。当前没有足够依据宣布当前Fine已经获得该无标签证书。

## 8. 本轮允许写到哪里

**Proved：** 规范固定后的affine residual对允许latent affine重参数化不变；rank退化下的总体非识别；full-rank九context近邻的有限数据检验下界；非线性Gaussian-preserving twist的严格正affine residual及Hellinger界。

**Conditional：** 正确full-rank族与完整总体律下的因素/余项functional可识别；完整相容集的无标签充分证书。前者不是当前估计成功，后者不是当前实现完成。

**Unresolved：** 当前实际Fine能否在无物理标签下得到非空创新及多h余项预算；完整模型类包含性、可计算集合上界与有限数据强度下界；外部physical target的语义锚定；论文级新颖性。

本轮下界使用经典Le Cam/距离工具，不能将这种基本“不可能性证明方法”称为首创。最有价值的纠偏是：**总体可辨识、有限数据可认证和可观测proxy充分性是三件不同的事；应给出有强度与支持条件的接受式证书，或诚实保留拒绝/abstain边界。**

必要的解析核验见 `unlabeled_limits_checks.py`、`UNLABELED_LIMITS_CHECKS.json`；只枚举context参数、计算Gaussian KL/矩和收敛级数，不生成样本或物理轨迹。

