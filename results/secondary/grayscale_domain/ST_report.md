# Phase 11D — ST grayscale-domain fixed-config ablation

Frozen configuration: **C07**. No HPO, validation, early stopping, or threshold tuning.

| Condition | F1 | Precision | Recall | Accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| RGB-trained / RGB-tested | 0.7040 | 0.8980 | 0.5789 | 0.7702 | 0.9082 | 0.9150 |
| RGB-trained / grayscale-tested | 0.7109 | 0.5556 | 0.9868 | 0.6211 | 0.7511 | 0.7216 |
| Grayscale-trained / grayscale-tested | 0.8372 | 0.7500 | 0.9474 | 0.8261 | 0.8709 | 0.8445 |

## Paired video/group bootstrap delta-F1

- grayscale_domain_minus_original_rgb: +0.1332 (95% CI [-0.0558, +0.3554])
- testtime_grayscale_minus_original_rgb: +0.0069 (95% CI [-0.2184, +0.2391])
- grayscale_domain_minus_testtime_grayscale: +0.1263 (95% CI [+0.0227, +0.2620])
