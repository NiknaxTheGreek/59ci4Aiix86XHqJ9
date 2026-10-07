# Phase 11D — O grayscale-domain fixed-config ablation

Frozen configuration: **C14**. No HPO, validation, early stopping, or threshold tuning.

| Condition | F1 | Precision | Recall | Accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| RGB-trained / RGB-tested | 0.9983 | 1.0000 | 0.9966 | 0.9983 | 0.9998 | 0.9998 |
| RGB-trained / grayscale-tested | 0.6840 | 0.5197 | 1.0000 | 0.5511 | 0.9311 | 0.9342 |
| Grayscale-trained / grayscale-tested | 0.9735 | 1.0000 | 0.9483 | 0.9749 | 1.0000 | 1.0000 |

## Paired video/group bootstrap delta-F1

- grayscale_domain_minus_original_rgb: -0.0248 (95% CI [-0.0447, -0.0092])
- testtime_grayscale_minus_original_rgb: -0.3143 (95% CI [-0.4128, -0.2320])
- grayscale_domain_minus_testtime_grayscale: +0.2895 (95% CI [+0.2014, +0.3874])
