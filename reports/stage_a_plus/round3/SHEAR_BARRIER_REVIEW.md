# Round 3：单向 shear 下完整固定选集的精确风险边界

日期：2026-11-04。只作受限总体理论与确定性公式核验；不训练、不采样、不生成物理轨迹、不改历史报告。本文将 Round 2 的持久因素反例推广为**任意 d、任意固定 k 下，沿一个指定单向 shear 路径的完整目标选集边界**。它仍不是实际非线性 Fine 的全类认证，也不宣称首创。

## 1. 问题范围和不可省略的假设

在同一固定物理尺度下令 v 的各轴在 Q0 上方差为1，state covariance 为Σ。多时距平方增量矩阵记

\[
M=|\mathcal H|^{-1}\sum_hE_{Q_h}[\Delta_hv\Delta_hv^T],
\quad S_\ell=M_{\ell\ell}.
\]

M是二阶矩，不要求增量均值为零，不要求state平稳；各Qh可有不同合法端点。真实声明集合S*先固定，0<k=|S*|<d，并假设它是原分数的**唯一**top-k：

\[
\gamma=\min_{b\notin S^\star}S_b-\max_{a\in S^\star}S_a>0.
\]

若γ≤0，下面从正gap出发的公式不能使用；严格正确性的baseline已经不成立，严格风险鲁棒半径直接记0或`baseline_not_unique_target_topk`。

只研究单向 shear

\[
A_t=I+t e_i e_j^T,\qquad i\ne j,\qquad t\in\mathbb R.
\]

它对所有有限t可逆，\(A_t^{-1}=I-t e_i e_j^T\)、detAₜ=1；**没有旋转的±45° fundamental-domain限制**。设ρ=Σij，并要求|ρ|<1，以保证变换轴方差

\[
V_i(t)=1+2\rho t+t^2=(t+\rho)^2+1-\rho^2>0
\]

在全部有限t上不退化。Σ正定是一个充分条件；若|ρ|=1，某有限t可能产生零方差，必须另记normalization N/A，不能继续套本定理。

坐标命名及S*沿路径固定，不在每个t重新匹配/重命名。本文给严格唯一top-k的边界；它不证明被混合轴仍满足逐轴物理恢复门。G0实现的FP32排序与固定索引tie-break在边界上须另处理，见§6。

## 2. 精确 Gaussian profile 风险

假设真实一步创新在同一物理尺度下满足positive diagonal Gaussian covariance Dᵤ，且mean nuisance可精确吸收 Aₜ 的坐标变换。regime概率p(u)固定，有限regime、各 dℓ(u)>0。则对完整 Gaussian observation NLL 作自由mean和regime diagonal-variance profile后，shear的excess为

\[
\boxed{F_{ij}(t)=\frac12E_u\log\left(1+t^2\frac{d_j(u)}{d_i(u)}\right).}
\tag{S1}
\]

**证明。** 只有(i,j)条件协方差块改变，为

\[
\begin{pmatrix}d_i+t^2d_j&td_j\\td_j&d_j\end{pmatrix}.
\]

其det为dᵢdⱼ，对角元乘积为(dᵢ+t²dⱼ)dⱼ。最佳diagonal Gaussian covariance是该对角线，Gaussian KL为½log(两者比值)。其他轴相消。证毕。

F为偶函数，关于|t|连续严格递增，F(0)=0；有限positive regime下F(|t|)→∞。局部

\[
F(t)=\tfrac12\kappa_{\mathrm{shear},ij}t^2+o(t^2),\qquad
\kappa_{\mathrm{shear},ij}=E_u[d_j(u)/d_i(u)].
\]

这不是orthogonal κ，两个方向i←j与j←i一般非常不同。这里F是population free-profile风险；有限head的clamp/函数限制、真实CAL NLL与神经训练误差未被等同于它。

## 3. 只有一个score变化，但须检查完整选集

输出轴i为vᵢ+t vⱼ，其余输出不变；每轴都使用变换后的自身Q0方差重新标准化。因此

\[
\boxed{S_i(t)=\frac{S_i+2tM_{ij}+t^2S_j}{1+2\rho t+t^2},
\quad S_\ell(t)=S_\ell\quad(\ell\ne i).}
\tag{S2}
\]

若i∈S*，只需检查i与每个非目标b：

\[
S_b-S_i(t)=\frac{q_b(t)}{V_i(t)},\quad
q_b(t)=g_b+2\beta_b t+\alpha_b t^2,
\]

\[
\boxed{g_b=S_b-S_i>0,\quad
\beta_b=S_b\rho-M_{ij},\quad\alpha_b=S_b-S_j.}
\tag{S3a}
\]

若i∉S*，只需检查i与每个目标a：

\[
S_i(t)-S_a=\frac{q_a(t)}{V_i(t)},
\]

\[
\boxed{g_a=S_i-S_a>0,\quad
\beta_a=M_{ij}-S_a\rho,\quad\alpha_a=S_j-S_a.}
\tag{S3b}
\]

所有不涉及i的target/non-target comparisons保持原正gap。由于Vᵢ(t)>0，完整S*仍为唯一top-k的**必要且充分**条件就是上述全部qℓ(t)>0。这里不是只检查旋转/混合的两个坐标；source轴j可能与目标边界不是同一轴。

## 4. 单个二次gap的全部情形

令q(t)=g+2βt+αt²，g>0，\(\Delta=\beta^2-\alpha g\)。下表处理正负两种t。

| 情形 | 有限tie | tie性质 | 最近tie的绝对参数 |
|---|---|---|---|
| α<0 | 两根、异号，Δ>0 | 穿越后可反序 | g/(\|β\|+√Δ) |
| α=0、β≠0 | 一根−g/(2β) | 穿越 | g/(2\|β\|) |
| α=0、β=0 | 无 | q恒正；归一化gap在\|t\|→∞趋0 | ∞，仅limiting tie |
| α>0、Δ<0 | 无 | 对所有有限t及极限方向保持正gap | ∞，无limiting tie |
| α>0、Δ=0 | 一重根−β/α（β≠0） | 切触，只有tie、无反序 | g/\|β\| |
| α>0、Δ>0 | 两根、同号 | 两根之间反序 | g/(\|β\|+√Δ) |

故在存在有限tie时可统一用数值较稳定的式子

\[
\boxed{r_q=\frac{g}{|\beta|+\sqrt{\beta^2-\alpha g}}.}
\tag{S4}
\]

**证明。** α≠0时根为(−β±√Δ)/α。利用(β²−Δ)=αg进行有理化，绝对值较小的根得到S4；α=0、β≠0时S4也退化为正确线性根。α=β=0不能代入0/0，必须单列。Δ=0、g>0只可能对应α>0和β≠0，切触根合法。证毕。

若β≠0，最近tie的符号为−signβ；β=0且α<0时正负最近根等距。对α>0的切触/双根，β必须非零。

**广义Rayleigh商解释。** gap为

\[
\frac{(1,t)\begin{pmatrix}g&\beta\\\beta&\alpha\end{pmatrix}(1,t)^T}
{(1,t)\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}(1,t)^T}.
\]

分母矩阵正定。α>0、Δ<0时分子矩阵正定，故全实线gap有统一正下界；α=β=0时它半正定且唯一零方向是t=∞，所以没有任何有限tie但不存在全实线统一正margin。切触情形的半正定零方向在有限t处。这些都是经典二次型性质。

## 5. 完整选集的 sharp population risk barrier

对§3中全部相关比较ℓ，按§4得到r_qℓ；无有限root时记∞。定义

\[
t_*:=\min_\ell r_{q_\ell},\qquad
\mathcal R_*:=\begin{cases}F_{ij}(t_*),&t_*<\infty,\\
\infty,&t_*=\infty.\end{cases}
\tag{S5}
\]

### 定理 S1（Proved；沿指定shear族的严格唯一选集）

在前述假设下，对任意有限风险半径R≥0，风险球

\[
\{t\in\mathbb R:F_{ij}(t)\le R\}
\]

中的**每个**表示都以唯一top-k选出原S*，当且仅当

\[
\boxed{R<\mathcal R_*.}
\tag{S6}
\]

若 \(\mathcal R_*=\infty\)，每个有限R均满足保证。它可能是严格uniform positive-gap情形，也可能只有|t|→∞的limiting tie；二者须分开报告，不能把limiting tie写为一个可执行有限参数的错误。

**证明。** 全部q在t=0为正，且连续。在区间|t|<t*内没有任何root，故全部为正；若t*有限，至少一比较在某个|t|=t*处为零，同时所有其他比较非负。由F关于|t|严格递增，R<F(t*)当且仅当风险球的|t|范围严格位于t*内；R≥F(t*)包含一个临界tie，不能保证唯一top-k。若没有有限root，全部q在全部有限t上都为正，每个有限风险球安全。证毕。

定理允许k>1、source j在目标内/外任意位置，并覆盖完整target/non-target比较。它**只覆盖一个指定shear方向**；若对多个(i,j)取最小barrier，只能声称所有列出的单shear均受控，不能推广到多shear复合或一般A的全risk ball。

## 6. Tie-break与实际G0：严格边界不是机器排序的等价式

严格S1把跨边界tie视为唯一性失败。当前G0使用存储FP32 score和索引tie-break，数值选集在tie点可能仍等于S*；因此不能把S6未经说明称为其逐位实现的必要充分式。

若在population exact scores上另固定一个无关t的lex priority，则完整选集正确当且仅当每个相关比较满足q>0，或q=0且目标的priority优先于非目标。由此：

- 穿越root：R<R*安全；R>R*包含反序点而失败；R=R*是否仍正确取决于所有最近tie的priority。
- 切触root：若tie priority偏目标，该比较在所有t上都不会导致实际lex错选，可从actual-failure约束中移除；若偏非目标，该孤立tie本身是错点。
- 无有限root：有限t都不因该比较失败；limiting tie不属于任何有限shear参数。

如果采用此tie-aware目标，应先去掉有利的纯切触约束，再找其余约束的首个bad/tie crossing，并保留半径端点是否包含的区别。更直接的可核查判据是逐比较的(q,priority)条件；FP32舍入还需单独误差预算。本文主定理采用严格unique top-k，避免把一个数值索引选择当物理辨识信息。

## 7. Round 2 持久因素反例的完整4D/k=2边界

沿用九regime四维解析构造，state covariance=I，rates

\[
(\rho_1,\rho_2,\rho_3,\rho_4)=(.1,\sqrt{1-\epsilon^2},.9,.2),
\quad \mathcal H=\{4,16,32\},\quad S^\star=\{2,3\},
\]

本节使用1-based索引。M对角，\(S_\ell=\overline{2(1-\rho_\ell^h)}\)。选择non-target i=1，source j=2。与目标2的gap为

\[
S_1(t)-S_2=\frac{S_1-S_2}{1+t²}>0,
\]

没有有限tie，只有|t|→∞时趋零。决定完整选集边界的是与目标3比较：

\[
S_1(t)-S_3=\frac{S_1-S_3+t²(S_2-S_3)}{1+t²}.
\]

对足够小ε，S2<S3<S4<S1，故

\[
\boxed{t_*²=\frac{S_1-S_3}{S_3-S_2}>0.}
\tag{S7}
\]

九regime中令c=ε²/.99，则variance ratio d2/d1有7次c、1次c/9及1次9c。因此

\[
\boxed{\mathcal R_*=\frac1{18}
\left[7\log(1+ct_*²)+\log(1+ct_*²/9)+\log(1+9ct_*²)\right].}
\tag{S8}
\]

t*趋向一个正正常数，真实物理γ=S4−S3约.582759；但R*按ε²趋零。原shear t=1位于错误侧，故低free-profile risk可以与明显错误选集并存。该现象的参数机制是shear方向曲率低，不能被正交旋转曲率大代替。

这不推翻固定ε的定性rank8辨识结果；ε→0过程中最小innovation variance没有统一正下界，且有限head clamp和有限flow包含性仍有Round2所述限制。ε=.01的确定性数值见附带JSON；它只评价数学构造，不是新增实验或实际Fine前向。

## 8. 近邻判断：数学工具已知，目标化组合需谨慎评价

[Bijan Afsari, Sensitivity Analysis for the Problem of Matrix Joint Diagonalization](https://isr.umd.edu/Labs/ISL/ICA2006/Sensitivity_Final.pdf)全文§3.2.3给出一般非正交变换的log-likelihood diagonality contrast（Eq.3.11），§4讨论其敏感性、方差profile及接近奇异的影响。这直接覆盖S1的数学母式；单shear代入和Hessian强弱不是新发明。该文关注joint diagonalizer的恢复/敏感性，本文不把它自动引用成物理慢目标top-k正确性定理。

[Sprekeler、Zito、Wiskott, An Extension of Slow Feature Analysis for Nonlinear Blind Source Separation, JMLR 2014](https://jmlr.org/papers/volume15/sprekeler14a/sprekeler14a.pdf)全文§2.2已将线性系数上的慢度写成增量/状态二次型及广义特征值问题；本文S2–S4属于这一Rayleigh quotient代数的明确限制。源恢复、harmonic排序和声明物理目标集合也须区分，不能因同用速度准则便声称已有全部target selection结论。

本轮定向核查只用于界定重合，不是穷尽新颖性检索，也未确认“同一个shear下完整固定物理选集的risk first-tie闭式”从未出现。最稳妥表述是：**一个由经典Gaussian contrast与Rayleigh商组成、给出完整case distinction的目标化受限定理**。它比只列κ/γ或只证明pair rotation更直接，但需要与实际余项、统计事件和新独立失败预测建立有效连接，才能判断是否有论文级贡献。

## 9. 已完成与尚未完成

已完成：S1精确风险、S2全selector scores、全部二次边界情形、S6严格全选集风险球充分必要式、lex tie区别、持久因素4D反例的明确临界参数/风险及确定性公式核验。

尚未完成：实际Fine属于该单shear类、总体M/Σ/D的无标签可得性、相对创新/score余项控制、有限神经训练excess上界、多shear或一般affine类的全risk ball、新颖性确认。不可将本文件有限解析结论升级为这些未完成项。
