# MonReader progressive temporal study — final results (3 November 2026)

**Status:** Completed: 8/8 new GPU/W&B runs, 4 temporal (T) and 4 source-plus-temporal (ST). This document closes the **supplementary progressive ResNet18 study** performed in `updated_04_generalization_and_stress_tests.ipynb`; it does **not** replace the accepted historical O/S/T/ST benchmarks which used its own frozen model-selection protocol.

## Design and provenance

- Data: 2,989 labelled `flip`/`notflip` image frames, reconstructed into 117 source recordings.
- The observed order is reconstructed from filenames/frame numbers, **not verified capture timestamps**. A future-frame suffix means a later position in that order, not a known elapsed time or an actual future event forecast.
- Coverage checkpoints: 25%, 50%, 75%, 90% of each source's chronological frame positions. The observed portion is partitioned into fitting and validation frames; later frames are reserved for evaluation.
- T: the model trains on earlier frames of familiar sources and evaluates their later frames.
- ST: the model trains/validates only on **known** source recordings and evaluates later frames from the **24 source recordings frozen as unseen in the original ST split**. None of those 24 sources enters ST training or validation. The historical ST split excluded 489 early context frames and used 121 late test frames; the new progressive ST evaluation suffix can include those earlier context frames **for scoring only**. Therefore its scores must never be relabelled as the original 121-frame ST result.
- An ImageNet-pretrained ResNet18 is trained at 25% in each regime, then updated at 50%, 75%, and 90%; the update stages are sequential **within T and within ST**, but T and ST are separate models. Early stopping and validation-selected checkpoints were permitted in this supplementary protocol.
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, CUDA-enabled PyTorch, batch size 4, cached 224×224 RGB images. W&B project: [monreader-progressive-temporal](https://forge.coreweave.com/wandb/nickthegreek/monreader-progressive-temporal).
- Source of the numbers below: the **completed local Notebook 04 GPU execution log and its generated result tables**, supplied 3 November 2026. Each run logged training/validation progress and a final held-out summary to W&B.

## Final measured results

All Macro F1 and recall values are reported on the 0–1 scale; no rows are estimated or extrapolated.

| Run | Regime | Observed coverage | Train | Validation | Future scored | Future sources | Future Macro F1 | Future flip recall | Future false negatives | Future false positives | Matched final-10% Macro F1 | Matched final-10% flip recall |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | T | 25% | 579 | 119 | 2,291 | 117 | 0.739814 | 0.572711 | 476 | 98 | 0.338274 | 0.070588 |
| 2 | T | 50% | 1,208 | 271 | 1,510 | 117 | 0.707684 | 0.483085 | 382 | 31 | 0.365469 | 0.052941 |
| 3 | T | 75% | 1,796 | 394 | 799 | 117 | 0.698346 | 0.473008 | 205 | 20 | 0.504572 | 0.211765 |
| 4 | T | 90% | 2,182 | 480 | 327 | 117 | 0.730368 | 0.529412 | 80 | 5 | 0.730368 | 0.529412 |
| 5 | ST | 25% | 460 | 95 | 467 | 24 | 0.661980 | 0.356098 | 132 | 0 | 0.346535 | 0.000000 |
| 6 | ST | 50% | 962 | 216 | 309 | 24 | 0.695531 | 0.467153 | 73 | 12 | 0.297872 | 0.000000 |
| 7 | ST | 75% | 1,431 | 313 | 164 | 24 | 0.628785 | 0.356164 | 47 | 6 | 0.445689 | 0.129032 |
| 8 | ST | 90% | 1,737 | 381 | 66 | 24 | 0.653066 | 0.419355 | 18 | 3 | 0.653066 | 0.419355 |

The **matched final-10% cohort** contains the same 327 frames for every T checkpoint and the same 66 frames for every ST checkpoint. Those fixed within-regime cohorts are suitable for comparison across coverage, unlike each run's full remaining-future suffix, whose size and composition change with coverage.

### Fixed-cohort coverage comparison

| Coverage | T matched Macro F1 | T matched flip recall | ST matched Macro F1 | ST matched flip recall |
|---:|---:|---:|---:|---:|
| 25% | 0.338274 | 0.070588 | 0.346535 | 0.000000 |
| 50% | 0.365469 | 0.052941 | 0.297872 | 0.000000 |
| 75% | 0.504572 | 0.211765 | 0.445689 | 0.129032 |
| 90% | 0.730368 | 0.529412 | 0.653066 | 0.419355 |

The fixed-cohort T Macro F1 increased from 0.338274 to 0.730368 (+0.392094 absolute); ST increased from 0.346535 to 0.653066 (+0.306531 absolute). The trajectory is **not monotonic**: T flip recall and ST Macro F1 both decline at 50% before improving. These are observational within-run differences in retrospectively supervised checkpoints, **not** an estimate of a real-time online learning benefit.

### Horizon-window evidence and support restrictions

Reportable three-frame position windows were required to contain **at least 20 flip frames, at least 20 notflip frames, and at least 8 source recordings for each class**. Unsupported windows are recorded as unreportable, *not* set to zero.

- T 25%: Macro F1 drops from **0.965608** in positions 1–3 to **0.466276** in positions 16–18 and **0.461729** in positions 19–21; only qualifying windows are interpreted.
- ST 25%: Macro F1 drops from **0.832168** in positions 1–3 to **0.425362** in positions 13–15. Positions 16–18 onward do not satisfy the reporting support criteria at that checkpoint.
- ST 90%: just positions **1–3** are reportable (64 frames, Macro F1 **0.646966**); the remaining two frames cannot support a separate horizon estimate.

This descriptive deterioration with future-frame distance is compatible with distribution changes within recordings, but it does **not** establish chronological concept drift, elapsed-time effects, or causality.

## Scientific interpretation

1. **Validation versus future generalization:** validation Macro F1 reached 1.0000 during both initial 25% runs, yet their full future suffix scores were 0.7398 (T) and 0.6620 (ST). Near-perfect validation is not a reliable substitute for later-frame evaluation.
2. **Safety-relevant error mode:** missed page flips remain material. At the matched final-10% ST checkpoint (90% coverage), flip recall is **0.419355** and **18 true flip frames are false negatives** (31 actual flip frames in that final suffix); T at 90% misses **80 flip frames** out of 170 actual flips.
3. **A coverage-dependent population trap:** overall-future Macro F1 is not monotonically higher at greater coverage; however, the future suffix itself shrinks and changes. Causal claims about more training data, or direct cross-coverage performance rankings on unequal suffixes, would be unjustified.
4. **Source generalization remains challenging:** ST is evaluated on 24 frozen unseen recordings while T uses familiar sources. Their evaluation populations differ, so a T/ST numeric gap must not be attributed uniquely to recording identity or temporal drift.
5. **Operational scope:** this is frame classification under an offline supervised experimental protocol. It does not prove page-turn event forecasting, realtime inference, automatic acquisition of new ground-truth labels, continual learning in deployment, or readiness for unattended use.

**Final recommendation:** Treat the progressive ResNet18 results as a **completed supplementary proof of principle** and retain the separately accepted historical results unchanged. The observed false negatives and uncertainty in the small ST suffix prevent a deployment-ready claim. The evidence supports follow-up evaluation on independent recordings with timestamped event annotations and source-level uncertainty analysis if future product work is pursued; no additional training is required to close this particular experiment.

## Evidence and reproducibility locations

The local `generated/outputs/progressive_temporal/` directory contains `overall_results.csv`, `horizon_results.csv`, per-regime results and figures, and per-coverage model/checkpoint outputs. Existing incompatible legacy stage directories were renamed and preserved; no prior checkpoint was deleted. W&B numeric logs were written separately from the 120 historical qualifying trial records.

- [Run 1 — T / 25%](https://forge.coreweave.com/wandb/nickthegreek/monreader-progressive-temporal/runs/ef0bd109)
- [Run 2 — T / 50%](https://forge.coreweave.com/wandb/nickthegreek/monreader-progressive-temporal/runs/d3f7ebef)
- [Run 3 — T / 75%](https://forge.coreweave.com/wandb/nickthegreek/monreader-progressive-temporal/runs/be3d8e8b)
- [Run 4 — T / 90%](https://forge.coreweave.com/wandb/nickthegreek/monreader-progressive-temporal/runs/1f2d7368)
- [Run 5 — ST / 25%](https://forge.coreweave.com/wandb/nickthegreek/monreader-progressive-temporal/runs/482cb32b)
- [Run 6 — ST / 50%](https://forge.coreweave.com/wandb/nickthegreek/monreader-progressive-temporal/runs/e4c974c8)
- [Run 7 — ST / 75%](https://forge.coreweave.com/wandb/nickthegreek/monreader-progressive-temporal/runs/a64aec3c)
- [Run 8 — ST / 90%](https://forge.coreweave.com/wandb/nickthegreek/monreader-progressive-temporal/runs/4d6a8aa7)

**Protocol boundary:** These eight sequentially updated ResNet18 experiments are a distinct, completed offline study. They do not redefine the original historical 121-frame ST result, establish automatic online learning, or supersede the accepted historical model comparisons.