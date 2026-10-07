# Phase 11D — T grayscale-domain fixed-config ablation

Frozen configuration: **C06**. No HPO, validation, early stopping, or threshold tuning.

| Condition | F1 | Precision | Recall | Accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| RGB-trained / RGB-tested | 0.8800 | 0.9693 | 0.8057 | 0.8894 | 0.9518 | 0.9637 |
| RGB-trained / grayscale-tested | 0.6695 | 0.5032 | 1.0000 | 0.5032 | 0.8500 | 0.8763 |
| Grayscale-trained / grayscale-tested | 0.9069 | 0.9887 | 0.8376 | 0.9135 | 0.9849 | 0.9878 |

## Paired video/group bootstrap delta-F1

- grayscale_domain_minus_original_rgb: +0.0269 (95% CI [-0.0002, +0.0546])
- testtime_grayscale_minus_original_rgb: -0.2105 (95% CI [-0.3052, -0.1260])
- grayscale_domain_minus_testtime_grayscale: +0.2374 (95% CI [+0.1538, +0.3295])
