# 首次输出序列化失败记录

首次确定性算术执行完成全部12父/4槽检查，在 `json.dumps(report)` 写出前遇到 NumPy bool 未转换为原生bool的TypeError，因此没有生成审计JSON。没有修改数据、模型、目标或任何科学公式。修复仅给JSON增加NumPy scalar序列化转换，然后用已存在的bundled Python重复确定性算术；不安装依赖、不修改项目环境。

失败发生于 `affine_geometry_audit.py` 原第196行，异常：`TypeError: Object of type bool is not JSON serializable`。首次尝试只读同8个物理成员，无OLS/polar/NN/随机数/FINAL/GPU；复算增加相同8成员读取与确定性小矩阵工作，必须与成功一次执行的成本分别说明。

首次失败版本脚本SHA256：`e034a7fa8fb70c3e39361ad8e199bb5ab4d5b893ec8c90a0326a10ac7df404c8`。成功版脚本SHA另存于正式审计JSON。
