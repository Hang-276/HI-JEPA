# Round 5：近似 posterior 尾部桥范围

2026-11-04。承接 Round 4 的 `APPROXIMATE_POSTERIOR_BRIDGE.md`，本分支研究如何避免对真实 posterior 施加全域统一 pmin。只在本 `round5/` 目录新增尾部理论、固定解析检查及来源记录；不修改 Round 4 或历史模型/报告/评价门。

允许：纸笔不等式；已有 Markdown 的只读参考；对 Gaussian 多项式 norm/hypercontractivity 和直接有限样本 nonlinear ICA 近邻作定向 primary 查证；手工固定概率向量、固定解析积分/矩公式的确定性算术检查。固定检查不采样、不拟合模型，不生成物理轨迹。

禁止并不执行：任何科学数据数组读取、checkpoint/模型加载或前向、训练、新随机样本/轨迹、服务器/GPU。不会用未知真实 posterior KL、物理 Gaussian 系数或尾常数冒充已测可观测输入。

本轮要分别回答：仅有 head posterior q 的区域下界和真实 KL 小，是否足以控制 log-odds；加入真实 Gaussian odds 的矩/系数预算后能否控制坏集合与未归一 restricted 误差；这些常数是否需真值、何时所得界趋零或可非空；KL 与可由观测 regime 标签估计的 cross-entropy 粗界有何差别。本分支最多证明充分统计量 span；同轮 quantitative closure 附录虽已另补条件定理，仍不把其真实前提当成实际 Fine 已满足或已获得 G0 证书。

后续优先采用 Gaussian-anchor 小集合收缩的吸收接口，消去 RHS 的未知真实 odds norm，而不把真系数估计当作已解决。新增交付为 `TAIL_POSTERIOR_BOUND.md`、`tail_posterior_checks.py`、`TAIL_POSTERIOR_CHECKS.json`、`TAIL_POSTERIOR_SOURCE_MANIFEST.json`。检查脚本一轮执行全部通过，使用固定概率向量、固定 Gaussian 参数与解析截断矩；未读取任何数据数组或训练产物。

追加只读独立核验同轮 `QUANTITATIVE_CLOSURE.md` 的 Gaussian tilt-Hermite 系数、C4/K3常数、O(δ)行误差与非塌缩门；没有修改作者文件或新算样本。该阅读来源及反馈摘要保存在本分支 source manifest。

随后只读复核 `TRANSPORT_AND_SELECTION.md` 的Hölder、重规范与score/选集公式，新增 `TAIL_CLOSURE_INDEPENDENT_REVIEW.md` 记录；自有尾部稿补齐Kq²d→0和零质量Ω的限定。未重跑原已通过的解析checks。
