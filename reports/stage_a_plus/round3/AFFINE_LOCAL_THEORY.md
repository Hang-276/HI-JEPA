# Round 3：一般仿射核心的有限邻域风险与慢目标分数

日期：2026-11-04。继承 Round 2 的 pair Hessian。本文件仅作总体条件理论与固定小矩阵的确定性核验，没有神经网络训练、模型前向、随机样本/物理轨迹生成或服务器操作。

## 1. 结论与边界

本轮完成的不是“从实际 Fine 的低 CAL NLL 自动得到可靠慢轴”，而是一个有显式半径和余项的局部桥：

\[
\text{已对齐且进入指定仿射邻域}
\Longrightarrow F(A)\gtrsim\kappa_{\rm aff}\|A-I\|_F^2
\Longrightarrow\text{指定 target gap 的条件下界}.
\]

分数一阶敏感性由 \(\Psi_{ab}=\ell_{ab}^TH^{-1}\ell_{ab}\) 控制，可能比统一最弱曲率界更有针对性。独立 AR 的标准矩条件可使一阶项为零，但二阶分数误差仍存在。

**所有下界均要求候选已经位于所列局部 chart。** 小 risk 本身不证明这个事件，不排除远处置换、其他等价解或任意非线性 flow。没有将经验 CAL 的矩或 NLL冒充总体输入。

## 2. 固定 chart、尺度与风险

物理状态按固定 Q0 逐轴标准化，\(\Sigma=\operatorname{Cov}_{Q_0}(v)\)，\(\Sigma_{jj}=1\)。逐轴命名已固定，行尺度利用允许的非零 affine 等价类固定为 \(\operatorname{diag}A=1\)。该 chart 要求对应对角元非零；实际匹配和尺度是否可这样稳定固定是另一个条件。

写

\[
A=I+B,\qquad B_{jj}=0,\qquad\|B\|_F\le r<1.
\]

真一步条件创新为 \(N(0,D_u)\)，D对角、正定，条件均值允许任意 dense 作用。自由 profile 的观测 NLL excess是

\[
F(A)=\frac12E_u\log\frac{\prod_j(AD_uA^T)_{jj}}{\det(AD_uA^T)}.
\]

假设逐regime的variance ratios满足
\(d_k(u)/d_j(u)\le R\) 对全部j,k成立，R≥1。该R固定在当前物理尺度gauge；不把它与自然参数矩阵奇异值混用。Round 2 已解释为何κrot不能代替这里的general-affine κ。

## 3. 显式 Taylor 余项与有限邻域 coercivity

以每个 i<j 的 \((B_{ij},B_{ji})\) 排列 off-diagonal 向量b；因此 \(\|b\|_2=\|B\|_F\)。定义 block-diagonal Hessian

\[
H=\operatorname{blockdiag}_{i<j}
\begin{pmatrix}E(d_j/d_i)&1\\1&E(d_i/d_j)\end{pmatrix},
\qquad \kappa_{\rm aff}=\lambda_{\min}(H)>0.
\]

### 命题 A1：显式余项（Proved）

\[
\boxed{F(I+B)=\frac12b^THb+\mathcal R_F(B),\qquad
|\mathcal R_F(B)|\le
\left[\frac{r}{3(1-r)}+\frac{R^2r^2}{4}\right]\|B\|_F^2.}
\tag{A1}
\]

**证明。** 因diagB=0，令 \(x_j(u)=\sum_{k\ne j}B_{jk}^2d_k/d_j\ge0\)，有精确式

\[
F(I+B)=\frac12E\sum_j\log(1+x_j)-\log\det(I+B).
\]

由 \(0\le x-\log(1+x)\le x^2/2\)，第一项的二次截断余项≤\(R^2\|B\|_F^4/4\)。当||B||op≤r<1时，logdet的trace级数绝对收敛；trB=0，二次项是−trB²/2。

\[
\left|\log\det(I+B)+\tfrac12\operatorname{tr}B^2\right|
\le\sum_{k\ge3}\frac{|\operatorname{tr}B^k|}{k}
\le\frac{r}{3(1-r)}\|B\|_F^2,
\]

其中 \(|\operatorname{tr}B^k|\le\|B\|_F^2\|B\|_{\rm op}^{k-2}\)。两部分的二次主项相加恰为bTHb/2，得到A1。det(I+B)>0由从I到I+B的非奇异连续路径保证。

### 推论 A2：有限半径保证（Proved under assumptions）

令

\[
c_r=\frac{r}{3(1-r)}+\frac{R^2r^2}{4}.
\]

若 \(c_r\le\kappa_{\rm aff}/4\)，则

\[
\boxed{F(I+B)\ge\frac14b^THb
\ge\frac{\kappa_{\rm aff}}4\|B\|_F^2.}
\tag{A2}
\]

因为 \(\|B\|_F^2\le b^THb/\kappa_{\rm aff}\)，直接代入A1即可。这不是从Hessian正定跳到非局部结论。

一个显式充分半径是

\[
r\le r_0:=\min\left\{\frac12,
\frac{3\kappa_{\rm aff}}{16},
\sqrt{\frac{\kappa_{\rm aff}}{2R^2}}\right\}.
\tag{A3}
\]

在r≤1/2时，第一预算≤2r/3≤κ/8，第二预算≤κ/8。弱仿射辨识或大variance-ratio下，r0可能非常小、现实上空泛；本文件不将保守半径当作已核实的Fine范围。

更一般可保留 \(\alpha_r=1/2-c_r/\kappa_{\rm aff}>0\)，得到 \(F\ge\alpha_r b^THb\)，不必固定1/4常数。

**可选固定候选的收紧。** 对实际给定B，不用uniform R²预算而保留

\[
\delta_{\log}=\tfrac12E\sum_j[x_j-\log(1+x_j)],\quad
\delta_{\det}\le\frac{\|B\|_F^2\|B\|_{\rm op}}
{3(1-\|B\|_{\rm op})},\quad\|B\|_{\rm op}<1.
\]

则 \(|\mathcal R_F|\le\delta_{\log}+\delta_{\det}\)。若这一预算≤κaff||B||F²/4，同样得到该固定B上的A2。它可比uniform ball常数明显紧；但使用实际B计算是truth-assisted域检验，既不证明整个同半径球安全，也不证明低risk自动推出入域。

## 4. 一般仿射慢分数的线性项与二阶余项

固定全部Qh与horizon权重，令

\[
M=\overline{E_{Q_h}[\Delta_hv\Delta_hv^T]}\succeq0,
\qquad s_j=M_{jj}.
\]

M是增量二阶矩，不要求增量均值为零或平稳。第j输出行写为 \(a_j=e_j+b_j\)，其中 \((b_j)_j=0\)。分数为

\[
S_j(A)=\frac{a_j^TMa_j}{a_j^T\Sigma a_j}.
\]

定义 \(T_j=M-s_j\Sigma\)、\(u_j=e_j^T\Sigma b_j\)、\(v_j=b_j^T\Sigma b_j\)。直接通分得

\[
\boxed{S_j(A)-s_j
=\frac{2e_j^TT_jb_j+b_j^TT_jb_j}{1+2u_j+v_j}
=L_j(B)+\mathcal R_j(B),}
\tag{A4}
\]

其中

\[
L_j(B)=2e_j^TT_jb_j,
\quad\mathcal R_j=
\frac{b_j^TT_jb_j-L_j(B)(2u_j+v_j)}{1+2u_j+v_j}.
\]

### 命题 A3：显式二阶预算（Proved）

令 \(K_\Sigma=\|\Sigma\|_{\rm op}\)，假设 \(\sqrt{K_\Sigma}r<1\)。设 \(a_j^*=\|T_je_j\|_2\)、\(t_j^*=\|T_j\|_{\rm op}\)，定义

\[
C_j(r)=\frac{t_j^*+4a_j^*\sqrt{K_\Sigma}+2a_j^*K_\Sigma r}
{(1-\sqrt{K_\Sigma}r)^2}.
\]

则

\[
\boxed{|\mathcal R_j(B)|\le C_j(r)\|b_j\|_2^2.}
\tag{A5}
\]

**证明。** Σ-seminorm的反三角给分母 \(\|e_j+b_j\|_\Sigma^2\ge(1-\sqrt{K_\Sigma}\|b_j\|)^2\)。又有 |L_j|≤2a*j||bj||、|b_jTT_jb_j|≤t*j||bj||²，以及 |2u+v|≤2√KΣ||bj||+KΣ||bj||²。代入A4得A5。若Σ正定，也可用λminΣ给另一分母下界；这里保留与局部标准化更直接的形式。

## 5. 指定目标的 dual-Hessian sensitivity

对目标a与非目标b，令 \(\gamma_{ab}=s_b-s_a>0\)。按上述off-diagonal排列，将 \(L_b-L_a\) 的系数记为 \(\ell_{ab}\)：第b行系数为 \(2(M_{bk}-s_b\Sigma_{bk})\)，第a行取负，其余为零。定义

\[
\Psi_{ab}=\ell_{ab}^TH^{-1}\ell_{ab}.
\]

它是指定gap对受风险约束扰动的一阶敏感性；不是新的基本统计概念，也不是所有输出的统一轴误差。

### 定理 A4：风险约束下的有限邻域 gap（Proved under assumptions）

在A1–A5条件下、\(c_r\le\kappa_{\rm aff}/4\)时，设 \(C_{ab}=\max\{C_a,C_b\}\)，则

\[
\boxed{S_b(A)-S_a(A)\ge\gamma_{ab}
-2\sqrt{\Psi_{ab}F(A)}
-\frac{4C_{ab}}{\kappa_{\rm aff}}F(A).}
\tag{A6}
\]

**证明。** H-inner-product Cauchy给 \(|\ell^Tb|\le\sqrt{\Psi}\sqrt{b^THb}\le2\sqrt{\Psi F}\)。二阶差不超过 \(C_b\|b_b\|^2+C_a\|b_a\|^2\le C_{ab}\|B\|_F^2\le4C_{ab}F/\kappa\)。在A4中相减即得。

如果 \(F(A)\le\varepsilon_F\)，仿射输出每轴统计评分误差≤εstat，那么对全部target/non-target pair满足

\[
\gamma_{ab}>2\sqrt{\Psi_{ab}\varepsilon_F}
+\frac{4C_{ab}}{\kappa_{\rm aff}}\varepsilon_F
+2\varepsilon_{\rm stat}
\tag{A7}
\]

即可保证该仿射表示的选集正确。若进一步输出为非线性T=Av+r，还须额外加入Round 2 N1得到的pair multi-horizon score余项预算 \(B^{T-A}_{\rm gap}\)。条件创新余项不是这项score预算的替代。

Ψ满足 \(\Psi_{ab}\le\|\ell_{ab}\|^2/\kappa_{\rm aff}\)，因此它可能比统一最弱方向预算小。特别是当敏感性几乎不投向弱辨识方向时，这个差别有意义。反过来，若状态gap最敏感的正是弱shear方向，Ψ会变大；不能只报告较强κrot。

## 6. 一阶零不等于分数零误差

若 \(\Sigma=I\)、M=diag(s)，则所有Lj=0，Ψab=0。然而有精确式

\[
S_j(I+B)-s_j
=\frac{\sum_{k\ne j}(s_k-s_j)B_{jk}^2}
{1+\sum_{k\ne j}B_{jk}^2}.
\tag{A8}
\]

故误差是二阶且通常非零。该结构适用于满足相应二阶矩条件的独立平稳AR，也适用于具有这些矩而非边际独立的系统；不能从“一阶不敏感”宣称选择不受混合影响。A6此时只剩O(F/κ)项，正符合该现象。

## 7. 与 exact single-shear 路线的分工

对 \(B=tE_{ij}\)，B²=0，A的det恒为1，精确风险是

\[
F(I+tE_{ij})=\tfrac12E\log(1+t^2d_j/d_i).
\]

在ratio≤R时，它满足 \(t^2E(d_j/d_i)/[2(1+t^2R)]\le F\le t^2E(d_j/d_i)/2\)。这条一维方向式无需r0式的小球，可明显更紧；但它只控制该指定shear，不能替代多方向general-affine定理。

本文件的优势是给全部off-diagonal方向一个确有余项和半径的统一局部结论，并明确target-specific dual sensitivity。局限是半径与二阶常数保守；弱条件下可能不如exact单shear路线有实际量级。两者应并列，不能用单ray的好界宣称整个仿射邻域已覆盖。

## 8. 经典性、可证对象与开放项

Gaussian joint-diagonalization contrast、local Hessian、Taylor余项与dual quadratic norm都是经典工具；[Afsari的nonorthogonal JD分析](https://isr.umd.edu/Labs/ISL/ICA2006/Sensitivity_Final.pdf)已经提供直接背景。本轮完成的是将它们适配到当前固定行尺度、真实G0标准化及指定目标gap的完整条件证明；未确认强ICML新颖性。

已证明：A1显式Taylor余项、A2–A3有限半径coercivity、A4–A5分数精确式与二阶界、A6–A7目标敏感性条件。未证明：实际Fine进入该chart、其非线性余项小、CAL risk提供总体εF、固定数据矩已具有所需总体概率界，或实际训练轴已被认证。

`affine_local_checks.py` 和 `AFFINE_LOCAL_CHECKS.json` 只使用固定小矩阵、context参数及解析log-det式验证这些公式与不等式，不生成样本或轨迹。核验通过只能支持代数实现一致，不能证明真实父模型满足总体假设。

