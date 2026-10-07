# MonReader — V2 scientific rebuild

MonReader is a binary computer-vision project for classifying a single frame as `flip` or `notflip`.

## Frozen scratch-CNN champions

| Regime | Champion | F1 | Interpretation |
|---|---:|---:|---|
| O | C14 | **0.9983** | supplied source-joint benchmark |
| S | C18 | **0.8196** | held-out ENV-03 |
| T | C06 | **0.8800** | later frames of known videos |
| ST | C07 | **0.7040** | held-out ENV-03 + later frames |

See `reports/synthesis/CROSS_REGIME_SYNTHESIS.md` for 5,000-resample frame and video/group uncertainty.

## Fixed ImageNet ResNet18 comparator

One prospectively frozen non-adaptive ResNet18 recipe was trained independently in O/S/T/ST: 5 head-only epochs + 15 full-network epochs, AdamW, final epoch-20 checkpoint, no validation, no early stopping, threshold 0.5.

| Regime | Scratch champion F1 | ResNet18 F1 | ΔF1 (ResNet − scratch) | Paired video/group 95% CI |
|---|---:|---:|---:|---:|
| O | 0.9983 | 0.9931 | -0.0052 | -0.014 to +0.002 |
| S | 0.8196 | 0.7807 | -0.0390 | -0.169 to +0.076 |
| T | 0.8800 | 0.8807 | +0.0007 | -0.032 to +0.030 |
| ST | 0.7040 | 0.7196 | +0.0156 | -0.212 to +0.259 |

None of the four paired video/group intervals excludes zero. The fixed pretrained comparator therefore does not support replacing any frozen scratch champion.

See `reports/resnet18/RESNET18_COMPARATOR_REPORT.md` and `manifests/resnet18/RESNET18_REGISTRY.yaml`.

## Key interpretation

O, S, T and ST answer different generalization questions and are not a global model ranking. O is highly source-joint and near-frame-correlated; S isolates unseen-source difficulty; T isolates later-frame difficulty within known sources; ST combines unseen source and later-frame stress.

## Next gate

**Phase 11 — controlled secondary analyses.** These are explicitly post-selection and cannot change Best O/S/T/ST. Priority analyses include controlled augmentation, visual/data ablations, shortcut sensitivity, and temporal/video deployment diagnostics.

## Future work

Hands are part of the natural MonReader input domain, so hand/arm removal is **not required for deployment**. It was investigated only as a secondary analysis to test whether hand-related visual information influences model behaviour.

A promising pretrained removal approach did not translate into acceptable end-to-end reconstruction, and subsequent tuning did not resolve the core failure modes. The supplied dataset also provides no true ground truth beneath occlusions, and attempts to recover covered page pixels from related or nearby frames were not sufficiently complete or consistent for reliable use.

Naturalistic hand/arm removal is therefore excluded from the reported results and no hand/arm-removed corpus is used in the presented analysis. It is retained only as a future-work topic because a defensible solution would require document-aware reconstruction, reliable recovery or validation of hidden content, and stronger QA than currently available.

See `reports/future_work/HAND_ARM_REMOVAL.md`. No hand/arm-removal performance tables or transformed-corpus results are presented.
