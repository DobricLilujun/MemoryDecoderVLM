# V4 实验结果

> 数据划分：每个 (task_type × difficulty) cell 70 条 train / 30 条 val（seed=0）。
> 不再保留 V3 的 test split——所有评估都在 val 集上。

**Split 规模**

| split | train | val |
| :-- | --: | --: |
| perception | 1680 | 720 |
| reasoning | 1680 | 660 |

## 1. 有 Zoom 组（仅 perception 训练）

> **训练数据**：FineSightBench/perception 的 train split。
> **评估**：perception-val（同分布）+ reasoning-val（分布外泛化）。

### 1.1 perception-val（同分布） (eval = perception)

| 配置 | Overall | easy | medium | hard | extreme | Δ vs base | N |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| base | 71.94 | 96.11 | 91.11 | 73.89 | 26.67 | — | 720 |
| zd_24L_perc | 74.44 | 96.11 | 93.89 | 82.22 | 25.56 | +2.50 | 720 |
| zd_priv_24L_perc | 74.86 | 96.11 | 95.00 | 82.22 | 26.11 | +2.92 | 720 |
| zd_zoompriv_24L_perc | 75.69 | 96.11 | 95.00 | 83.33 | 28.33 | +3.75 | 720 |
| zd_24L_noap_perc | 74.03 | 96.11 | 92.22 | 80.00 | 27.78 | +2.09 | 720 |
| zd_24L_nosw_perc | 75.00 | 96.11 | 94.44 | 82.22 | 27.22 | +3.06 | 720 |

**按任务分解**

| 配置 | animal_recognition | block_recognition | color_block_recognition | letter_recognition | shape_recognition | text_recognition |
| :-- | --: | --: | --: | --: | --: | --: |
| base | 70.83 | 50.00 | 89.17 | 78.33 | 74.17 | 69.17 |
| zd_24L_perc | 70.00 | 67.50 | 89.17 | 77.50 | 71.67 | 70.83 |
| zd_priv_24L_perc | 71.67 | 68.33 | 89.17 | 76.67 | 72.50 | 70.83 |
| zd_zoompriv_24L_perc | 70.00 | 71.67 | 89.17 | 77.50 | 75.00 | 70.83 |
| zd_24L_noap_perc | 70.83 | 60.83 | 89.17 | 77.50 | 74.17 | 71.67 |
| zd_24L_nosw_perc | 70.00 | 67.50 | 89.17 | 77.50 | 75.00 | 70.83 |

### 1.2 reasoning-val（分布外泛化） (eval = reasoning)

| 配置 | Overall | easy | medium | hard | extreme | Δ vs base | N |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| base | 9.85 | 11.11 | 8.33 | 7.78 | 13.33 | — | 660 |
| zd_24L_perc | 10.45 | 10.56 | 10.00 | 9.44 | 12.50 | +0.60 | 660 |
| zd_priv_24L_perc | 12.27 | 14.44 | 11.67 | 10.56 | 12.50 | +2.42 | 660 |
| zd_zoompriv_24L_perc | 10.91 | 11.11 | 10.56 | 9.44 | 13.33 | +1.06 | 660 |
| zd_24L_noap_perc | 12.27 | 12.78 | 12.78 | 11.11 | 12.50 | +2.42 | 660 |
| zd_24L_nosw_perc | 13.18 | 15.00 | 13.89 | 11.11 | 12.50 | +3.33 | 660 |

**按任务分解**

| 配置 | blur_chain | chain_reasoning | comparison_chain | counting_chain | text_counting_chain | text_reading_chain |
| :-- | --: | --: | --: | --: | --: | --: |
| base | 0.00 | 0.00 | 0.00 | 0.00 | 47.50 | 6.67 |
| zd_24L_perc | 0.00 | 6.00 | 3.00 | 0.00 | 45.00 | 5.00 |
| zd_priv_24L_perc | 0.00 | 6.00 | 4.00 | 0.00 | 44.17 | 15.00 |
| zd_zoompriv_24L_perc | 0.00 | 6.00 | 3.00 | 0.00 | 47.50 | 5.00 |
| zd_24L_noap_perc | 0.00 | 6.00 | 3.00 | 0.00 | 46.67 | 13.33 |
| zd_24L_nosw_perc | 0.00 | 6.00 | 3.00 | 0.00 | 46.67 | 18.33 |

## 2. 无 Zoom 组（perception + reasoning 联合训练）

> **训练数据**：perception.train ∪ reasoning.train，shuffle 后混合 batch。
> **评估**：perception-val + reasoning-val 分别报告。

### 2.1 perception-val (eval = perception)

| 配置 | Overall | easy | medium | hard | extreme | Δ vs base | N |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| base | 71.94 | 96.11 | 91.11 | 73.89 | 26.67 | — | 720 |
| zd_priv_24L_all | 74.58 | 96.11 | 94.44 | 81.67 | 26.11 | +2.64 | 720 |
| zd_knn_6L_all | 77.78 | 96.67 | 95.00 | 86.67 | 32.78 | +5.84 | 720 |
| zd_knn_6L_noap_all | 77.50 | 96.67 | 95.00 | 86.67 | 31.67 | +5.56 | 720 |
| zd_knn_6L_nosw_all | 78.06 | 96.67 | 95.00 | 86.67 | 33.89 | +6.12 | 720 |
| zd_knn_6L_plain_all | 77.78 | 96.67 | 95.00 | 86.67 | 32.78 | +5.84 | 720 |
| zd_knn_24L_all | 77.78 | 96.67 | 95.00 | 86.67 | 32.78 | +5.84 | 720 |
| lora_r16_all | 85.42 | 100.00 | 99.44 | 88.33 | 53.89 | +13.48 | 720 |

**按任务分解**

| 配置 | animal_recognition | block_recognition | color_block_recognition | letter_recognition | shape_recognition | text_recognition |
| :-- | --: | --: | --: | --: | --: | --: |
| base | 70.83 | 50.00 | 89.17 | 78.33 | 74.17 | 69.17 |
| zd_priv_24L_all | 72.50 | 67.50 | 89.17 | 75.83 | 70.83 | 71.67 |
| zd_knn_6L_all | 71.67 | 80.00 | 89.17 | 78.33 | 75.83 | 71.67 |
| zd_knn_6L_noap_all | 70.83 | 80.00 | 89.17 | 77.50 | 75.83 | 71.67 |
| zd_knn_6L_nosw_all | 71.67 | 80.00 | 89.17 | 78.33 | 77.50 | 71.67 |
| zd_knn_6L_plain_all | 71.67 | 80.00 | 89.17 | 78.33 | 75.83 | 71.67 |
| zd_knn_24L_all | 71.67 | 80.00 | 89.17 | 78.33 | 75.83 | 71.67 |
| lora_r16_all | 67.50 | 100.00 | 99.17 | 88.33 | 81.67 | 75.83 |

### 2.2 reasoning-val (eval = reasoning)

| 配置 | Overall | easy | medium | hard | extreme | Δ vs base | N |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| base | 9.85 | 11.11 | 8.33 | 7.78 | 13.33 | — | 660 |
| zd_priv_24L_all | 8.94 | 10.56 | 5.56 | 8.89 | 11.67 | -0.91 | 660 |
| zd_knn_6L_all | 12.42 | 13.89 | 11.67 | 10.00 | 15.00 | +2.57 | 660 |
| zd_knn_6L_noap_all | 13.33 | 15.00 | 12.78 | 11.67 | 14.17 | +3.48 | 660 |
| zd_knn_6L_nosw_all | 12.42 | 13.89 | 12.22 | 10.00 | 14.17 | +2.57 | 660 |
| zd_knn_6L_plain_all | 12.73 | 13.33 | 12.22 | 11.67 | 14.17 | +2.88 | 660 |
| zd_knn_24L_all | 11.97 | 13.33 | 11.11 | 10.00 | 14.17 | +2.12 | 660 |
| lora_r16_all | 35.61 | 38.33 | 40.56 | 36.67 | 22.50 | +25.76 | 660 |

**按任务分解**

| 配置 | blur_chain | chain_reasoning | comparison_chain | counting_chain | text_counting_chain | text_reading_chain |
| :-- | --: | --: | --: | --: | --: | --: |
| base | 0.00 | 0.00 | 0.00 | 0.00 | 47.50 | 6.67 |
| zd_priv_24L_all | 0.00 | 0.00 | 0.00 | 0.00 | 46.67 | 2.50 |
| zd_knn_6L_all | 0.00 | 6.00 | 4.00 | 0.00 | 54.17 | 5.83 |
| zd_knn_6L_noap_all | 0.00 | 6.00 | 4.00 | 0.00 | 55.00 | 10.00 |
| zd_knn_6L_nosw_all | 0.00 | 6.00 | 4.00 | 0.00 | 54.17 | 5.83 |
| zd_knn_6L_plain_all | 0.00 | 6.00 | 4.00 | 0.00 | 55.83 | 5.83 |
| zd_knn_24L_all | 0.00 | 6.00 | 4.00 | 0.00 | 53.33 | 4.17 |
| lora_r16_all | 32.00 | 39.00 | 39.00 | 18.33 | 58.33 | 27.50 |
