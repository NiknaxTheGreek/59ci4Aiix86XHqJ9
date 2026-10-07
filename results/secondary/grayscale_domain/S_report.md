# Phase 11D — S grayscale-domain fixed-config ablation

Frozen configuration: **C18**. No HPO, validation, early stopping, or threshold tuning.

| Condition | F1 | Precision | Recall | Accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| RGB-trained / RGB-tested | 0.8196 | 0.8232 | 0.8161 | 0.8377 | 0.9002 | 0.8995 |
| RGB-trained / grayscale-tested | 0.6225 | 0.4519 | 1.0000 | 0.4519 | 0.8980 | 0.8898 |
| Grayscale-trained / grayscale-tested | 0.7788 | 0.6953 | 0.8851 | 0.7727 | 0.8669 | 0.8471 |

## Paired video/group bootstrap delta-F1

- grayscale_domain_minus_original_rgb: -0.0409 (95% CI [-0.1537, +0.0636])
- testtime_grayscale_minus_original_rgb: -0.1971 (95% CI [-0.3515, -0.0580])
- grayscale_domain_minus_testtime_grayscale: +0.1562 (95% CI [+0.0426, +0.2879])
