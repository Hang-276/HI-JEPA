# HI-JEPA 运行汇总

证据范围：engineering_only。

完成 20，不可认证 4，失败 0，待运行 0。

| 任务 | 状态 | 秒每步 | 迁移结果 |
| --- | --- | --- | --- |
| e1_real_seed0 | completed | 0.00254 | N/A |
| e1_real_seed1 | completed | 0.00252 | N/A |
| e1_rotation_seed0 | completed | 0.00252 | N/A |
| e1_rotation_seed1 | completed | 0.00253 | N/A |
| e1_near_seed0 | unresolved | N/A | N/A |
| e1_near_seed1 | unresolved | N/A | N/A |
| e1_near_resolved_seed0 | completed | 0.00252 | N/A |
| e1_near_resolved_seed1 | completed | 0.00255 | N/A |
| e1_observation_noise_seed0 | completed | 0.00249 | N/A |
| e1_observation_noise_seed1 | completed | 0.00276 | N/A |
| e1_uncertified_access_seed0 | unresolved | N/A | N/A |
| e1_uncertified_access_seed1 | unresolved | N/A | N/A |
| e2_hierarchical_seed0 | completed | 0.00406 | N/A |
| e2_hierarchical_seed1 | completed | 0.00381 | N/A |
| e2_block_only_seed0 | completed | 0.00235 | N/A |
| e2_block_only_seed1 | completed | 0.00246 | N/A |
| e2_full_seed0 | completed | 0.00254 | N/A |
| e2_full_seed1 | completed | 0.00252 | N/A |
| e3_direct_seed0 | completed | 0.00307 | N/A |
| e3_direct_seed1 | completed | 0.00234 | N/A |
| e3_no_fine_seed0 | completed | 0.00290 | reached |
| e3_no_fine_seed1 | completed | 0.00231 | not_reached_within_budget |
| e3_full_seed0 | completed | 0.00261 | reached |
| e3_full_seed1 | completed | 0.00252 | not_reached_within_budget |

local_integration 的指标仅用于确认代码路径，不构成正式恢复或分层收益结论。
不可认证的预设条件被保留；缺测使用 N/A。正式主结果应在冻结的 Linux CUDA 环境统一生成。
