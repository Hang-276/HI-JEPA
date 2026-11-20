# Round 4：可观测认证接口与记录盘点范围

2026-11-04。仅作理论分析与已有代码/配置/报告metadata的只读核查；不作新数组诊断。

允许讨论的对象仅为观测x、action a、regime u、已学习Fine/head/G0函数及其可计算导数/密度。禁止执行模型、加载checkpoint、生成数据、读取TEST/FINAL或physical数组；不把已有trueD、A、物理inverse或有标签余项当无标签输入。理论中的真实生成类/目标包含性若必要，必须显式标不可观测假设。

只读检查：当前Fine/head/selector源码、Fine.json中既有观测NLL/标准化残差诊断字段、G0.json与训练来源配置、纯推理/导数接口源码，以及已完成报告的范围说明。只盘点记录是否存在，不重新计算残差、score、Jacobian或模型输出；不访问.pt内部权重或.npy/.npz成员。

输出：`OBSERVABLE_CERTIFICATE_OPTIONS.md`、`OBSERVABILITY_LEDGER.md`及本scope。root另写 `VERSION_SPACE_CERTIFICATE.md`，本文件仅概述并链接该接口，不重复其长证明。区分可观测self-consistency、统计最慢坐标选择与物理轴/预声明目标；构造性定理需给完整前提，明确哪些仍不可由数据认证。

零server/GPU/train/forward/PRNG/新物理数据/新数组诊断/模型或评价门修改；不重跑任何前轮审计。
