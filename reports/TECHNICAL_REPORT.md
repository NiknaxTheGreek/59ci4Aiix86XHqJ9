# MonReader Technical Report

## Executive summary

MonReader is a binary image-classification project for identifying whether a video frame depicts a page `flip` or `notflip`. The project evolved through two scientific phases: an earlier notebook-based workflow using explicit validation sets and Bayesian W&B sweeps, and a later V2 rebuild using a prospectively frozen train/test-only protocol.

This report documents the **submission-ready historical notebook workflow** while preserving the rule that **current V2 results are authoritative**.

The historical analysis produced a strong but important result: performance was near ceiling under the nominal supplied split, yet declined under source-safe and temporal stress. That finding motivated the stricter V2 rebuild and remains the most important methodological lesson of the project.

---

## 1. Problem statement

The operational task is to classify an individual MonReader video frame as:

- `flip`
- `notflip`

The main scientific risk is that neighbouring frames from the same source video are strongly correlated. A model can therefore appear extremely accurate if train and test sets contain near-neighbour frames or the same recording sources.

The historical workflow explicitly created four evaluation regimes to expose this risk.

---

## 2. Data and source structure

The historical audit contained **2,989 frames from 117 source videos**.

The four historical regimes were:

| Regime | Training | Validation | Context | Test | Test source videos |
|---|---:|---:|---:|---:|---:|
| O | 2,152 | 240 | 0 | 597 | 115 |
| S | 2,147 | 232 | 0 | 610 | 24 |
| T | 2,387 | 301 | 0 | 301 | 117 |
| ST | 1,904 | 475 | 489 | 121 | 24 |

O is source-joint: its test sources also occur in development. S and ST explicitly hold test sources out of development. T evaluates later frames from known videos. ST combines source and temporal stress.

This means the regimes answer **different generalization questions** and their scores are descriptive robustness comparisons rather than causal effects.

---

## 3. Historical scratch-CNN selection

The historical scratch-CNN study used independent Bayesian W&B sweeps for O, S, T and ST. Each sweep targeted validation F1, with the following search space:

- optimizer: Adam, AdamW, SGD or RMSprop;
- learning rate: `1e-5` to `1e-2` log-uniform;
- batch size: 8 to 256;
- optional positive weight decay: `1e-6` to `1e-2`;
- dropout: 0.0 to 0.5;
- depth: 1 to 7 convolutional blocks;
- starting filters: 8 to 64.

The final controller audit reached **30 qualifying completed runs per regime** after excluding invalid/OOM/zero-F1 attempts.

A later correction pass was important: stale W&B summary values changed the frozen winner identity for T and ST. The submission notebook therefore reports the **corrected frozen winners**, not the earlier stale sweep-summary winners.

Historical tuned-scratch held-out Macro F1 was:

| Regime | Macro F1 |
|---|---:|
| O | 0.9899 |
| S | 0.4400 |
| T | 0.8305 |
| ST | 0.7906 |

The source-safe S collapse was the clearest signal that O was not a sufficient robustness estimate.

---

## 4. Historical ResNet18 comparison

An ImageNet-pretrained ResNet18 was evaluated on the same frozen historical populations.

| Regime | Scratch Macro F1 | ResNet18 Macro F1 | Δ ResNet − Scratch |
|---|---:|---:|---:|
| O | 0.9899 | 0.9950 | +0.0050 |
| S | 0.4400 | 0.9584 | +0.5184 |
| T | 0.8305 | 0.9091 | +0.0786 |
| ST | 0.7906 | 0.8772 | +0.0867 |

Historically, ResNet18 was substantially more robust under the difficult regimes, particularly S.

This result is a **historical matched reference comparison**. It does not authorize changing the current V2 champions.

---

## 5. Historical robustness and uncertainty

The final historical ResNet18 results were:

| Regime | Macro F1 | Source-video 95% CI | Balanced accuracy | MCC |
|---|---:|---:|---:|---:|
| O | 0.994969 | [0.986128, 1.000000] | 0.994828 | 0.989989 |
| S | 0.958387 | [0.909802, 0.993352] | 0.958202 | 0.916780 |
| T | 0.909140 | [0.865657, 0.946739] | 0.908163 | 0.833399 |
| ST | 0.877246 | [0.747988, 0.972281] | 0.867925 | 0.781163 |

Source-video bootstrap intervals are primary because individual frames from one video are correlated.

ST has the widest uncertainty because its test population contains only **24 source videos and 121 frames**.

---

## 6. Failure analysis

Historical ResNet18 errors were:

| Regime | Errors | Error rate | FP | FN | High-confidence error share ≥ 0.90 |
|---|---:|---:|---:|---:|---:|
| O | 3 | 0.0050 | 0 | 3 | 0.333 |
| S | 25 | 0.0410 | 12 | 13 | 0.320 |
| T | 27 | 0.0897 | 0 | 27 | 0.519 |
| ST | 14 | 0.1157 | 0 | 14 | 0.786 |

False negatives dominate the frozen historical error pattern. In T and ST, many errors are highly confident, so these are not simply threshold-adjacent mistakes.

The threshold was kept frozen at 0.50. Failure analysis was not allowed to trigger post-hoc threshold tuning.

---

## 7. Interpretability

### Grad-CAM

Grad-CAM was generated for **21 deterministic cases** chosen before visual inspection.

- 21/21 Grad-CAM maps were generated;
- 0 were degenerate;
- every FP32 attribution forward retained the frozen historical class decision.

The heatmaps are visible directly in `submission/notebooks/05_final_analysis.ipynb`.

Grad-CAM is treated as a diagnostic association with the predicted-class score, **not as causal evidence**.

### Intermediate activations

Four fixed ResNet18 stages were inspected:

- `layer1[-1]` → 56×56;
- `layer2[-1]` → 28×28;
- `layer3[-1]` → 14×14;
- `layer4[-1]` → 7×7.

Across 21 cases this produced **84 non-degenerate activation maps**. Mean absolute activation was used across channels, avoiding manual channel cherry-picking.

These maps are class-agnostic feature-strength diagnostics and should not be assigned semantic meaning without additional evidence.

---

## 8. Historical RGB/grayscale branch

The completed Scratch CNN RGB-versus-grayscale branch did not satisfy the full predefined meaningful-effect gate. It is therefore treated as closed secondary evidence and does not justify new ResNet18 grayscale experiments.

---

## 9. Reproducibility improvements in the submission package

The cleaned notebooks correct the major portability problems of the original working notebooks:

- no `C:\Users\...` paths;
- no stored W&B credentials;
- no hard-coded W&B entity;
- no dependency on original sweep IDs for report rendering;
- no accidental sweep creation on notebook execution;
- no stale T/ST winner identity;
- no failed debug cells in the logical narrative;
- verified results and figures are committed as snapshots;
- Grad-CAM heatmaps are displayed inline;
- the historical/V2 authority boundary is explicit.

The notebooks can therefore be reviewed in report mode immediately, while W&B remains available as an opt-in integration.

---

## 10. Current V2 status

The current `main` branch V2 rebuild is scientifically authoritative.

Frozen V2 scratch champions:

| Regime | Champion | F1 |
|---|---:|---:|
| O | C14 | 0.9983 |
| S | C18 | 0.8196 |
| T | C06 | 0.8800 |
| ST | C07 | 0.7040 |

The fixed ResNet18 V2 comparator does not provide statistically decisive paired evidence for replacing those champions.

Historical and V2 values must therefore remain separated.

---

## 11. Technical conclusion

The most defensible technical conclusion is not that one architecture universally “wins.” It is that **evaluation design dominates interpretation**.

A model that appears effectively perfect under a source-joint nominal split can lose substantial performance under unseen-source and later-frame conditions. For MonReader, source-aware and temporal evaluation are therefore mandatory before any production claim.

The historical notebooks demonstrate how that insight was discovered. The V2 rebuild turns the insight into a stricter prospective protocol.
