# Wave-1 最终汇总：FourRoom H-JEPA / LeWM（修订读出与 Family）

日期：2026-11-18。正式 TEST 数据只在 `forecast_receipt.json` 落盘后读取。

## Family 定义

- Family A（`fourroom_probe_v2`）：L1 4 读出 × 3 seeds × 1 rule = 12；L2 2 readouts × 3 seeds × 2 rules = 12；family_size=24，alpha=0.05/24。
- Family B（`fourroom_lewm_v2` + `fourroom_probe_v2` + `fourroom_hjepa_l4_v2`）：LeWM 9 + L2 15 + L4 27 = 51；family_size=51，alpha=0.05/51。

## 表 1：Stage-1B seed44 补验与 CLS 层深（dev_preflight，AUDIT min R²）

| 模型 | Seed | CLS L1 | CLS L2 | CLS L3 | CLS L4 | grid MLP | grid ridge |
|---|---:|---:|---:|---:|---:|---:|---:|
| lewm | 42 | 0.979 | — | — | — | 0.996 | 0.993 |
| lewm | 43 | 0.977 | — | — | — | 0.988 | 0.996 |
| lewm | 44 | 0.973 | — | — | — | 0.993 | 0.991 |
| hjepa_l2 | 42 | 0.707 | 0.488 | — | — | 0.974 | 0.997 |
| hjepa_l2 | 43 | 0.425 | 0.127 | — | — | 0.924 | 0.996 |
| hjepa_l2 | 44 | 0.240 | -0.038 | — | — | 0.974 | 0.995 |
| hjepa_l4 | 42 | 0.922 | 0.718 | 0.434 | 0.202 | 0.962 | 0.991 |
| hjepa_l4 | 43 | 0.804 | 0.526 | 0.191 | 0.114 | 0.973 | 0.997 |
| hjepa_l4 | 44 | 0.944 | 0.776 | 0.628 | 0.338 | 0.990 | 0.997 |

> 所有单元的 agent_x / agent_y AUDIT R² 在 0.990–1.000 之间；下方逐轴表重点列出受约束的 distractor 两轴。

## 表 2：Family A 四读出配对消融（`fourroom_probe_v2`，L=2 模型，family_size=24）

| Lv | Readout | Seed | Rule | a_x | a_y | d_x | d_y | min | Cov | Err/Acc | Upper | Status | Flip win | Flip acc | Flip acc/err |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | cls_8k_small | 42 | own_cal | 0.997 | 0.997 | 0.584 | 0.511 | 0.511 | 0.516 | 36/1057 | 0.0532 | RECOVERY_NOT_VERIFIED | 208 | 0.163 | 36/36 |
| 1 | cls_8k_small | 43 | own_cal | 0.996 | 0.996 | -0.084 | 0.052 | -0.084 | 0.538 | 107/1102 | 0.1253 | RECOVERY_NOT_VERIFIED | 208 | 0.000 | 107/107 |
| 1 | cls_8k_small | 44 | own_cal | 0.996 | 0.996 | -0.065 | -0.047 | -0.065 | 0.497 | 116/1018 | 0.1452 | RECOVERY_NOT_VERIFIED | 208 | 0.115 | 104/102 |
| 1 | cls_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.780 | 0.695 | 0.695 | 0.493 | 9/1010 | 0.0211 | RECOVERY_NOT_VERIFIED | 208 | 0.231 | 9/9 |
| 1 | cls_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.428 | 0.472 | 0.428 | 0.491 | 23/1005 | 0.0399 | RECOVERY_NOT_VERIFIED | 208 | 0.010 | 23/23 |
| 1 | cls_160k_official | 44 | own_cal | 1.000 | 1.000 | 0.377 | 0.434 | 0.377 | 0.513 | 36/1050 | 0.0536 | RECOVERY_NOT_VERIFIED | 208 | 0.000 | 36/36 |
| 1 | grid4x4_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.982 | 0.978 | 0.978 | 0.501 | 0/1026 | 0.0060 | PASS | 208 | 0.750 | 0/0 |
| 1 | grid4x4_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.972 | 0.975 | 0.972 | 0.499 | 0/1022 | 0.0060 | PASS | 208 | 0.745 | 0/0 |
| 1 | grid4x4_160k_official | 44 | own_cal | 1.000 | 0.999 | 0.982 | 0.973 | 0.973 | 0.497 | 0/1017 | 0.0061 | PASS | 208 | 0.808 | 0/0 |
| 1 | grid4x4_160k_ridge | 42 | own_cal | 1.000 | 0.999 | 0.996 | 0.996 | 0.996 | 0.494 | 0/1012 | 0.0061 | PASS | 208 | 0.933 | 0/0 |
| 1 | grid4x4_160k_ridge | 43 | own_cal | 1.000 | 1.000 | 0.995 | 0.996 | 0.995 | 0.496 | 0/1015 | 0.0061 | PASS | 208 | 0.938 | 0/0 |
| 1 | grid4x4_160k_ridge | 44 | own_cal | 1.000 | 1.000 | 0.994 | 0.995 | 0.994 | 0.498 | 0/1019 | 0.0060 | PASS | 208 | 0.899 | 0/0 |
| 2 | cls_8k_small | 42 | own_cal | 0.998 | 0.997 | 0.325 | 0.336 | 0.325 | 0.516 | 81/1056 | 0.1030 | RECOVERY_NOT_VERIFIED | 208 | 0.038 | 81/81 |
| 2 | cls_8k_small | 42 | level1_cal | 0.998 | 0.997 | 0.325 | 0.336 | 0.325 | 0.475 | 74/972 | 0.1036 | RECOVERY_NOT_VERIFIED | 208 | 0.038 | 74/74 |
| 2 | cls_8k_small | 43 | own_cal | 0.993 | 0.991 | -0.072 | -0.052 | -0.072 | 0.518 | 123/1061 | 0.1467 | RECOVERY_NOT_VERIFIED | 208 | 0.067 | 113/113 |
| 2 | cls_8k_small | 43 | level1_cal | 0.993 | 0.991 | -0.072 | -0.052 | -0.072 | 0.114 | 29/234 | 0.1972 | RECOVERY_NOT_VERIFIED | 208 | 0.067 | 29/29 |
| 2 | cls_8k_small | 44 | own_cal | 0.993 | 0.990 | -0.067 | -0.043 | -0.067 | 0.510 | 136/1044 | 0.1627 | RECOVERY_NOT_VERIFIED | 208 | 0.087 | 111/108 |
| 2 | cls_8k_small | 44 | level1_cal | 0.993 | 0.990 | -0.067 | -0.043 | -0.067 | 0.505 | 136/1035 | 0.1641 | RECOVERY_NOT_VERIFIED | 208 | 0.087 | 111/108 |
| 2 | cls_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.556 | 0.518 | 0.518 | 0.483 | 38/990 | 0.0593 | RECOVERY_NOT_VERIFIED | 208 | 0.130 | 38/38 |
| 2 | cls_160k_official | 42 | level1_cal | 1.000 | 1.000 | 0.556 | 0.518 | 0.518 | 0.399 | 30/818 | 0.0596 | RECOVERY_NOT_VERIFIED | 208 | 0.130 | 30/30 |
| 2 | cls_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.135 | 0.301 | 0.135 | 0.515 | 38/1054 | 0.0557 | RECOVERY_NOT_VERIFIED | 208 | 0.000 | 38/38 |
| 2 | cls_160k_official | 43 | level1_cal | 1.000 | 1.000 | 0.135 | 0.301 | 0.135 | 0.474 | 34/971 | 0.0554 | RECOVERY_NOT_VERIFIED | 208 | 0.000 | 34/34 |
| 2 | cls_160k_official | 44 | own_cal | 1.000 | 1.000 | 0.060 | 0.144 | 0.060 | 0.512 | 81/1049 | 0.1037 | RECOVERY_NOT_VERIFIED | 208 | 0.005 | 81/81 |
| 2 | cls_160k_official | 44 | level1_cal | 1.000 | 1.000 | 0.060 | 0.144 | 0.060 | 0.461 | 72/944 | 0.1042 | RECOVERY_NOT_VERIFIED | 208 | 0.005 | 72/72 |

## 表 3：Family B 三模型深度谱系（family_size=51）

| Model | Lv | Readout | Seed | Rule | a_x | a_y | d_x | d_y | min | Cov | Err/Acc | Upper | Status | Flip win | Flip acc | Flip acc/err |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| lewm | 1 | cls_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.978 | 0.981 | 0.978 | 0.501 | 0/1027 | 0.0067 | PASS | 208 | 0.750 | 0/0 |
| lewm | 1 | cls_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.978 | 0.980 | 0.978 | 0.494 | 0/1011 | 0.0068 | PASS | 208 | 0.784 | 0/0 |
| lewm | 1 | cls_160k_official | 44 | own_cal | 1.000 | 1.000 | 0.983 | 0.976 | 0.976 | 0.488 | 0/1000 | 0.0069 | PASS | 208 | 0.798 | 0/0 |
| lewm | 1 | grid4x4_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.994 | 0.996 | 0.994 | 0.503 | 0/1030 | 0.0067 | PASS | 208 | 0.913 | 0/0 |
| lewm | 1 | grid4x4_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.988 | 0.988 | 0.988 | 0.499 | 0/1022 | 0.0068 | PASS | 208 | 0.865 | 0/0 |
| lewm | 1 | grid4x4_160k_official | 44 | own_cal | 1.000 | 1.000 | 0.990 | 0.990 | 0.990 | 0.488 | 0/1000 | 0.0069 | PASS | 208 | 0.865 | 0/0 |
| lewm | 1 | grid4x4_160k_ridge | 42 | own_cal | 0.998 | 0.998 | 0.991 | 0.992 | 0.991 | 0.489 | 0/1001 | 0.0069 | PASS | 208 | 0.885 | 0/0 |
| lewm | 1 | grid4x4_160k_ridge | 43 | own_cal | 0.999 | 0.999 | 0.994 | 0.995 | 0.994 | 0.486 | 0/995 | 0.0069 | PASS | 208 | 0.880 | 0/0 |
| lewm | 1 | grid4x4_160k_ridge | 44 | own_cal | 0.998 | 0.998 | 0.987 | 0.987 | 0.987 | 0.490 | 0/1003 | 0.0069 | PASS | 208 | 0.812 | 0/0 |
| l2 | 1 | cls_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.780 | 0.695 | 0.695 | 0.493 | 9/1010 | 0.0223 | RECOVERY_NOT_VERIFIED | 208 | 0.231 | 9/9 |
| l2 | 1 | cls_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.428 | 0.472 | 0.428 | 0.491 | 23/1005 | 0.0415 | RECOVERY_NOT_VERIFIED | 208 | 0.010 | 23/23 |
| l2 | 1 | cls_160k_official | 44 | own_cal | 1.000 | 1.000 | 0.377 | 0.434 | 0.377 | 0.513 | 36/1050 | 0.0553 | RECOVERY_NOT_VERIFIED | 208 | 0.000 | 36/36 |
| l2 | 1 | grid4x4_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.982 | 0.978 | 0.978 | 0.501 | 0/1026 | 0.0067 | PASS | 208 | 0.750 | 0/0 |
| l2 | 1 | grid4x4_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.972 | 0.975 | 0.972 | 0.499 | 0/1022 | 0.0068 | PASS | 208 | 0.745 | 0/0 |
| l2 | 1 | grid4x4_160k_official | 44 | own_cal | 1.000 | 0.999 | 0.982 | 0.973 | 0.973 | 0.497 | 0/1017 | 0.0068 | PASS | 208 | 0.808 | 0/0 |
| l2 | 1 | grid4x4_160k_ridge | 42 | own_cal | 1.000 | 0.999 | 0.996 | 0.996 | 0.996 | 0.494 | 0/1012 | 0.0068 | PASS | 208 | 0.933 | 0/0 |
| l2 | 1 | grid4x4_160k_ridge | 43 | own_cal | 1.000 | 1.000 | 0.995 | 0.996 | 0.995 | 0.496 | 0/1015 | 0.0068 | PASS | 208 | 0.938 | 0/0 |
| l2 | 1 | grid4x4_160k_ridge | 44 | own_cal | 1.000 | 1.000 | 0.994 | 0.995 | 0.994 | 0.498 | 0/1019 | 0.0068 | PASS | 208 | 0.899 | 0/0 |
| l2 | 2 | cls_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.556 | 0.518 | 0.518 | 0.483 | 38/990 | 0.0611 | RECOVERY_NOT_VERIFIED | 208 | 0.130 | 38/38 |
| l2 | 2 | cls_160k_official | 42 | level1_cal | 1.000 | 1.000 | 0.556 | 0.518 | 0.518 | 0.399 | 30/818 | 0.0617 | RECOVERY_NOT_VERIFIED | 208 | 0.130 | 30/30 |
| l2 | 2 | cls_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.135 | 0.301 | 0.135 | 0.515 | 38/1054 | 0.0574 | RECOVERY_NOT_VERIFIED | 208 | 0.000 | 38/38 |
| l2 | 2 | cls_160k_official | 43 | level1_cal | 1.000 | 1.000 | 0.135 | 0.301 | 0.135 | 0.474 | 34/971 | 0.0572 | RECOVERY_NOT_VERIFIED | 208 | 0.000 | 34/34 |
| l2 | 2 | cls_160k_official | 44 | own_cal | 1.000 | 1.000 | 0.060 | 0.144 | 0.060 | 0.512 | 81/1049 | 0.1060 | RECOVERY_NOT_VERIFIED | 208 | 0.005 | 81/81 |
| l2 | 2 | cls_160k_official | 44 | level1_cal | 1.000 | 1.000 | 0.060 | 0.144 | 0.060 | 0.461 | 72/944 | 0.1067 | RECOVERY_NOT_VERIFIED | 208 | 0.005 | 72/72 |
| l4 | 1 | cls_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.911 | 0.924 | 0.911 | 0.477 | 0/976 | 0.0071 | PASS | 208 | 0.587 | 0/0 |
| l4 | 1 | cls_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.883 | 0.834 | 0.834 | 0.488 | 6/1000 | 0.0180 | RECOVERY_NOT_VERIFIED | 208 | 0.284 | 6/6 |
| l4 | 1 | cls_160k_official | 44 | own_cal | 1.000 | 1.000 | 0.968 | 0.958 | 0.958 | 0.498 | 0/1020 | 0.0068 | PASS | 208 | 0.712 | 0/0 |
| l4 | 1 | grid4x4_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.982 | 0.954 | 0.954 | 0.491 | 0/1006 | 0.0069 | PASS | 208 | 0.750 | 0/0 |
| l4 | 1 | grid4x4_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.967 | 0.983 | 0.967 | 0.495 | 0/1013 | 0.0068 | PASS | 208 | 0.837 | 0/0 |
| l4 | 1 | grid4x4_160k_official | 44 | own_cal | 1.000 | 0.999 | 0.992 | 0.994 | 0.992 | 0.490 | 0/1004 | 0.0069 | PASS | 208 | 0.889 | 0/0 |
| l4 | 1 | grid4x4_160k_ridge | 42 | own_cal | 1.000 | 0.999 | 0.992 | 0.992 | 0.992 | 0.492 | 0/1008 | 0.0068 | PASS | 208 | 0.846 | 0/0 |
| l4 | 1 | grid4x4_160k_ridge | 43 | own_cal | 1.000 | 1.000 | 0.996 | 0.997 | 0.996 | 0.494 | 0/1011 | 0.0068 | PASS | 208 | 0.918 | 0/0 |
| l4 | 1 | grid4x4_160k_ridge | 44 | own_cal | 0.999 | 0.999 | 0.996 | 0.997 | 0.996 | 0.490 | 0/1003 | 0.0069 | PASS | 208 | 0.938 | 0/0 |
| l4 | 2 | cls_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.730 | 0.766 | 0.730 | 0.523 | 10/1071 | 0.0224 | RECOVERY_NOT_VERIFIED | 208 | 0.130 | 10/10 |
| l4 | 2 | cls_160k_official | 42 | level1_cal | 1.000 | 1.000 | 0.730 | 0.766 | 0.730 | 0.509 | 10/1042 | 0.0230 | RECOVERY_NOT_VERIFIED | 208 | 0.130 | 10/10 |
| l4 | 2 | cls_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.732 | 0.589 | 0.589 | 0.483 | 19/990 | 0.0368 | RECOVERY_NOT_VERIFIED | 208 | 0.096 | 19/19 |
| l4 | 2 | cls_160k_official | 43 | level1_cal | 1.000 | 1.000 | 0.732 | 0.589 | 0.589 | 0.483 | 19/990 | 0.0368 | RECOVERY_NOT_VERIFIED | 208 | 0.096 | 19/19 |
| l4 | 2 | cls_160k_official | 44 | own_cal | 1.000 | 1.000 | 0.908 | 0.828 | 0.828 | 0.517 | 4/1058 | 0.0139 | RECOVERY_NOT_VERIFIED | 208 | 0.438 | 4/4 |
| l4 | 2 | cls_160k_official | 44 | level1_cal | 1.000 | 1.000 | 0.908 | 0.828 | 0.828 | 0.496 | 4/1015 | 0.0145 | RECOVERY_NOT_VERIFIED | 208 | 0.438 | 4/4 |
| l4 | 3 | cls_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.487 | 0.551 | 0.487 | 0.536 | 45/1097 | 0.0630 | RECOVERY_NOT_VERIFIED | 208 | 0.077 | 45/45 |
| l4 | 3 | cls_160k_official | 42 | level1_cal | 1.000 | 1.000 | 0.487 | 0.551 | 0.487 | 0.439 | 35/899 | 0.0631 | RECOVERY_NOT_VERIFIED | 208 | 0.077 | 35/35 |
| l4 | 3 | cls_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.539 | 0.347 | 0.347 | 0.504 | 53/1032 | 0.0762 | RECOVERY_NOT_VERIFIED | 208 | 0.058 | 53/53 |
| l4 | 3 | cls_160k_official | 43 | level1_cal | 1.000 | 1.000 | 0.539 | 0.347 | 0.347 | 0.380 | 31/779 | 0.0663 | RECOVERY_NOT_VERIFIED | 208 | 0.058 | 31/31 |
| l4 | 3 | cls_160k_official | 44 | own_cal | 1.000 | 1.000 | 0.803 | 0.649 | 0.649 | 0.508 | 27/1040 | 0.0450 | RECOVERY_NOT_VERIFIED | 208 | 0.269 | 27/27 |
| l4 | 3 | cls_160k_official | 44 | level1_cal | 1.000 | 1.000 | 0.803 | 0.649 | 0.649 | 0.387 | 16/792 | 0.0408 | RECOVERY_NOT_VERIFIED | 208 | 0.269 | 16/16 |
| l4 | 4 | cls_160k_official | 42 | own_cal | 1.000 | 1.000 | 0.216 | 0.249 | 0.216 | 0.502 | 89/1029 | 0.1169 | RECOVERY_NOT_VERIFIED | 208 | 0.072 | 89/89 |
| l4 | 4 | cls_160k_official | 42 | level1_cal | 1.000 | 1.000 | 0.216 | 0.249 | 0.216 | 0.414 | 72/847 | 0.1186 | RECOVERY_NOT_VERIFIED | 208 | 0.072 | 72/72 |
| l4 | 4 | cls_160k_official | 43 | own_cal | 1.000 | 1.000 | 0.307 | 0.188 | 0.188 | 0.510 | 86/1045 | 0.1118 | RECOVERY_NOT_VERIFIED | 208 | 0.034 | 86/86 |
| l4 | 4 | cls_160k_official | 43 | level1_cal | 1.000 | 1.000 | 0.307 | 0.188 | 0.188 | 0.377 | 50/772 | 0.0968 | RECOVERY_NOT_VERIFIED | 208 | 0.034 | 50/50 |
| l4 | 4 | cls_160k_official | 44 | own_cal | 1.000 | 1.000 | 0.601 | 0.424 | 0.424 | 0.494 | 47/1012 | 0.0706 | RECOVERY_NOT_VERIFIED | 208 | 0.096 | 47/47 |
| l4 | 4 | cls_160k_official | 44 | level1_cal | 1.000 | 1.000 | 0.601 | 0.424 | 0.424 | 0.398 | 32/815 | 0.0650 | RECOVERY_NOT_VERIFIED | 208 | 0.096 | 32/32 |

## 关键发现

- Family A：24 行中 **6 行 PASS**，全部来自 L1 的 `grid4x4_160k_official` 与 `grid4x4_160k_ridge`；CLS/8k 在 L1 全部 RECOVERY_NOT_VERIFIED，L2 全部 RECOVERY_NOT_VERIFIED。
- Family B：51 行中 **23 行 PASS**；LeWM L1 的三种 160k 读出全部 PASS；L2 L1 只有两种 grid 读出 PASS；L4 L1 的 grid 读出全部 PASS，CLS 在 seed43（0.834）差 0.016 未过门。
- **同读出下的深度效应**：LeWM 的 L1 CLS 能恢复 distractor（0.976–0.978），而 H-JEPA L2 的 L1 CLS 只有 0.377–0.695；H-JEPA L4 的 L1 CLS 为 0.834–0.958。L1 网格读出在所有三个模型族上都 ≥0.954。
- 深层 CLS（L2–L4）全部 RECOVERY_NOT_VERIFIED；AUDIT min R² 随层深单调下降（L4 seed42: 0.911 → 0.730 → 0.487 → 0.216）。
- 局部翻转窗：网格读出通常不接受翻转窗（accepted=0），全答准确率 0.75–0.94；CLS 读出会接受少量翻转窗并几乎全部错选。

## 完整性

- 三个正式目录状态均为 `COMPLETE`；Forecast receipt 的 thresholds/test_prediction SHA256 100% 匹配；无 NaN/Inf。
- 单测：`Ran 7 tests ... OK`。原 `run.py` / `metrics.py` / `config.json` / `test_metrics.py` 零改动；`assets/H-JEPA` 干净。
- 原始 `setup.sh` 仅有部署会话遗留的 1 行依赖锁定差异（非本次产生）。
- Family B 单臂诊断界 M=9/18/27 已逐行写入 `fourroom_depth_v2_combined/summary.json`。

