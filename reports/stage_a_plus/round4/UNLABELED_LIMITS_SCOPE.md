# Round 4：无标签余项与辨识边界审查范围

日期：2026-11-04。只推进理论和固定解析公式，不读取新的物理数组，不训练、模型前向、加载checkpoint、采样轨迹、服务器/GPU操作或TEST/FINAL访问。

允许的部署信息：观测x、action/regime、已训练可逆Fine/head及其导数、TRAIN/CAL及训练记录。禁止把物理state、真实观测inverse、latent生成参数、真值对齐、真D或真A当作无标签输入。

分别研究：有限数据上的非空认证/检验、完整总体观测律、full-rank条件成立与不成立。不能用rank退化例否定rank成立时的总体定性辨识；若给full-rank近不可辨识序列，必须指出强度退化和有限样本/统一性限定。

交付限于本目录的UNLABELED_LIMITS.md与必要解析检查。确定性检查只计算固定分布/矩阵公式，不生成观测或latent样本。已有Round2/3文件一律不改。
