# Hermite 残差修正与慢因素排序

本实验使用独立、标准化的平稳 Gaussian AR(1) 物理轴 \(Z_k\)。慢度目标是 lag 4、16、32 等权的逐窗平方增量均值，除以物理参考方差 1。可见读出 \(U\) 冻结。每个修正分支使用同一组独立参考状态估计其自身坐标的方差。物理答案始终由未修改的 \(Z\) 定义。

## 多因素交互、逐轴变形和其他非线性项

记 \(\psi_\alpha(z)=\prod_k{\rm He}_{\alpha_k}(z_k)/\sqrt{\alpha_k!}\)，其中 \({\rm He}_m\) 是概率论 Hermite 多项式，\(|\alpha|\) 为总次数。对标准独立 Gaussian，这组函数是正交归一基。先用 CAL 冻结逐轴仿射坐标 \(V_j=(U_j-b_j)/A_{jj}\)。同坐标参考归一化使这一变换保持基线分数。随后将 \(e_j=V_j-Z_j\) 在 0–3 阶基上联合拟合，保留常数和线性项，避免有限样本的经验相关性落入非线性解释。实际有限拟合系数应在独立状态上审计。

- 二、三阶共有 30 项。至少有两个不同坐标非零的 22 项为本轮的多因素交互组。
- 仅含受体自身的两项 \({\rm He}_2(Z_j)/\sqrt{2}\) 和 \({\rm He}_3(Z_j)/\sqrt{6}\) 组成自身非线性组。
- 剩余六项是其他物理轴各自的二、三阶项。单个供体的非线性串入也属于跨轴误差，但与多因素交互是不同的分析对象。

对某个组 \(B\)，理论真值更新为 \(g_j(Z)=\sum_{\alpha\in B}b_{j\alpha}\psi_\alpha(Z)\)，其中 \(b\) 只由 CAL 拟合。可执行更新只能取可见 \(U\) 的函数。若代理先在 CAL 上剔除常数和物理仿射项，这项正交性仍需在独立状态审计。代理不一定保持其他非线性组。含自身轴的非线性映射也不自动全局可逆。本轮不沿用线性单行剪切的可逆性结论。

## 状态 MSE 与总体慢度可以朝不同方向变化

若第 \(k\) 个 AR 系数为 \(\rho_k\)，则条件期望对 Hermite 模态的特征值是 \(\eta_\alpha=\prod_k\rho_k^{\alpha_k}\)。由平稳性与 Hermite 正交性，

\[
E[(\psi_\alpha(Z_{t+h})-\psi_\alpha(Z_t))
   (\psi_\beta(Z_{t+h})-\psi_\beta(Z_t))]
=2(1-\eta_\alpha^h)\,\mathbf 1_{\alpha=\beta}.
\]

因此对平方可积读出 \(V_j=\sum_\alpha u_\alpha\psi_\alpha(Z)\)，非零阶模态的总体参考方差为 \(\mathcal V=\sum_{\alpha\ne0}u_\alpha^2\)，总体慢度是

\[
D=\frac{\sum_{\alpha\ne0}u_\alpha^2d_\alpha}{\mathcal V},\qquad
d_\alpha=\frac{2}{3}\sum_{h\in\{4,16,32\}}(1-\eta_\alpha^h).
\]

若精确去掉组 \(B\) 的总体投影，设 \(E_B=\sum_{\alpha\in B}u_\alpha^2\)，\(d_B=\sum_{\alpha\in B}u_\alpha^2 d_\alpha/E_B\)，并要求 \(0<E_B<\mathcal V\)，则

\[
D_{\rm delete}-D=\frac{E_B(D-d_B)}{\mathcal V-E_B}.
\]

当组内模态比整个读出变化更快，即 \(d_B>D\)，去掉它们会使读出变慢。如果去掉的是二阶及以上的真实投影，该操作同时降低状态 MSE \(E_B\)。一个原本不该入选的因素可能因此被误纳入。实际 CAL 拟合只近似这些总体系数，可见代理也可能带入其他误差，所以不预设实测排序收益或损害的符号。

当前两个谱分离过程的 \(\rho_{\max}=.93\)、\(\rho_{\min}=.88\)。所有二阶及以上的纯 Hermite 模态满足 \(\eta_\alpha\le .93^2=.8649<.88\)，因此都比任何单个真实物理轴快。它们仍可能比含其他非线性模态的某个读出整体更慢。总体正交性不保证单个窗口交叉项为零。上述推导使用已知 Gaussian–Hermite 谱工具，用来区分误差分量抑制、状态恢复和慢集选取这三个终点。

## 有限窗口的可核验分解

设某个分支用 \(g_j\) 修正受体为 \(V_j-g_j\)，则同坐标的动态分子和参考方差变化为

\[
\delta N_j=-2\overline{\Delta V_j\Delta g_j}+\overline{(\Delta g_j)^2},
\qquad
\delta V_j=-2\operatorname{Cov}_{\rm REF}(V_j,g_j)+\operatorname{Var}_{\rm REF}(g_j).
\]

若参考分母仍为正，分数变化为 \((\delta N_j-D_j\delta V_j)/(\operatorname{Var}_{\rm REF}V_j+\delta V_j)\)。反号对照仅翻转两个交叉项的符号。独立状态上的目标组衰减、仿射及其他组变动、物理 top-2 的修复与新增错选需要分别报告。用真实 \(Z\) 构造更新只用于事后诊断，其分数变化不可计作可执行方法效果。
