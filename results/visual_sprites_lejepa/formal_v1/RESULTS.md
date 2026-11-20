# Two-Sprite visual comparison

Evidence tier: `formal_visual_sprites`
Device: cpu
Elapsed seconds: 8177.0
Per-cell Bonferroni alpha: 0.008333333
Forecast files verified: 32

| Arm and seed | agent_x R² | agent_y R² | distractor_x R² | distractor_y R² | Eligible | Coverage | Errors / accepted | CP upper | Result |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| visual_lejepa_seed101 | 0.996502 | 0.994930 | 0.983358 | 0.983345 | True | 54.0527% | 1/1107 | 0.6170% | PASS |
| visual_lejepa_seed102 | 0.995049 | 0.995314 | 0.984091 | 0.981474 | True | 54.3457% | 1/1113 | 0.6136% | PASS |
| visual_lejepa_seed103 | 0.996761 | 0.994913 | 0.981779 | 0.985908 | True | 53.5645% | 1/1097 | 0.6226% | PASS |
| long_horizon_pred_seed101 | 0.997885 | 0.996883 | 0.125669 | 0.260997 | False | 50.0977% | 45/1026 | 6.1609% | RECOVERY_NOT_VERIFIED |
| long_horizon_pred_seed102 | 0.998575 | 0.998374 | -0.002740 | -0.004478 | False | 48.3887% | 606/991 | 64.8536% | RECOVERY_NOT_VERIFIED |
| long_horizon_pred_seed103 | 0.996332 | 0.995815 | 0.262116 | 0.338990 | False | 52.4414% | 28/1074 | 4.0164% | RECOVERY_NOT_VERIFIED |

| Arm and seed | Subset | All-answer errors / windows (risk) | own_cal errors / accepted (risk) |
|---|---|---:|---:|
| visual_lejepa_seed101 | regular | 19/1889 (1.0058%) | 0/1104 (0.0000%) |
| visual_lejepa_seed101 | flip | 36/159 (22.6415%) | 1/3 (33.3333%) |
| visual_lejepa_seed102 | regular | 19/1889 (1.0058%) | 0/1109 (0.0000%) |
| visual_lejepa_seed102 | flip | 40/159 (25.1572%) | 1/4 (25.0000%) |
| visual_lejepa_seed103 | regular | 19/1889 (1.0058%) | 0/1095 (0.0000%) |
| visual_lejepa_seed103 | flip | 30/159 (18.8679%) | 1/2 (50.0000%) |
| long_horizon_pred_seed101 | regular | 369/1889 (19.5341%) | 10/984 (1.0163%) |
| long_horizon_pred_seed101 | flip | 119/159 (74.8428%) | 35/42 (83.3333%) |
| long_horizon_pred_seed102 | regular | 1358/1889 (71.8899%) | 519/874 (59.3822%) |
| long_horizon_pred_seed102 | flip | 120/159 (75.4717%) | 87/117 (74.3590%) |
| long_horizon_pred_seed103 | regular | 148/1889 (7.8348%) | 8/1048 (0.7634%) |
| long_horizon_pred_seed103 | flip | 109/159 (68.5535%) | 20/26 (76.9231%) |

Subgroup risks are descriptive. The six-cell family contains the prespecified overall own_cal claims. All cells share the same TEST episodes. Local UTC timestamps and named hashes document the forecast-before-scoring order.
