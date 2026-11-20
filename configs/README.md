# 配置入口

- `fourroom_hjepa/default.json`：新标准入口配置，与2026-11-15视觉交付版逐字节相同。
- `low_dim_ar1/*.json`：五个主线配置的链接，解析回封存原件。A1等其他阶段的参数在各阶段协议、`protocol.json` 或 `DESIGN_LOCK.json` 中，以[进展同步](../docs/PROGRESS_SYNC_JEPA.md)的逐项指纹为准。
- `legacy_hijepa/`：早期HI-JEPA配置，根配置文件名保留兼容链接。

封存配置不能原位修改后继续冒充原实验；新比较须另存新协议和新输出目录。
