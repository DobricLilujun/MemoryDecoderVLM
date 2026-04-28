# V4 Experiment Report

This report is generated automatically from local artifacts.

## Local Artifacts

- Training history CSV: analysis_v4/train_history.csv
- Evaluation summary CSV: analysis_v4/eval_overall.csv
- With-zoom loss curve: analysis_v4/loss_with_zoom.png
- Without-zoom loss curve: analysis_v4/loss_without_zoom.png
- With-zoom eval bars: analysis_v4/eval_with_zoom.png
- Without-zoom eval bars: analysis_v4/eval_without_zoom.png

## With Zoom (trained on perception only)

| Model | Perception Val | Reasoning Val |
| :-- | --: | --: |
| zd_24L | 74.44 | 10.45 |
| zd_priv_24L | 74.86 | 12.27 |
| zd_zoompriv_24L | 75.69 | 10.91 |
| zd_24L_noap | 74.03 | 12.27 |
| zd_24L_nosw | 75.00 | 13.18 |

![With-zoom loss](analysis_v4/loss_with_zoom.png)

![With-zoom evaluation](analysis_v4/eval_with_zoom.png)

## Without Zoom (trained on perception + reasoning)

| Model | Perception Val | Reasoning Val |
| :-- | --: | --: |
| base | 71.94 | 9.85 |
| zd_priv_24L | 74.58 | 8.94 |
| zd_knn_6L | 77.78 | 12.42 |
| zd_knn_6L_noap | 77.50 | 13.33 |
| zd_knn_6L_nosw | 78.06 | 12.42 |
| zd_knn_6L_plain | 77.78 | 12.73 |
| zd_knn_24L | 77.78 | 11.97 |
| lora_r16 | 85.42 | 35.61 |

![Without-zoom loss](analysis_v4/loss_without_zoom.png)

![Without-zoom evaluation](analysis_v4/eval_without_zoom.png)
