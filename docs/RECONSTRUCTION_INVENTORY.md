# MonReader — Scientific reconstruction inventory

> Working audit on `proofread`. This is a provenance and design record, not a claim that historical results have been reproduced.

## Protected evidence

- Original executed research notebooks, notebooks 01–05 (provided separately as HTML exports).
- Recorded held-out results: `submission/results/scratch_validation_test_metrics.csv`, `submission/results/resnet18_heldout_metrics.csv`, and `submission/results/wandb_sweep_completion.csv`.
- V2 dataset, split, and environment audit: `manifests/datasets/DATA-001_summary.json`, `manifests/splits/SPLIT_SUMMARY.json`, and `manifests/video_environment_map.csv`.
- The four published visual assets in `figures/` and the corresponding generation procedures.
- Original W&B Bayesian searches: 30 qualifying trials for each of O/S/T/ST. The recorded total attempts were 38, 37, 31, and 35 respectively.
- The separate preserved pre-V2 Git history (branch `historical/pre-v2-reset-2026-09-30`), original phase commits, and historical convergence experiment.

## Dataset facts grounded in the archived dataset audit

2,989 RGB images; 1,452 `flip` and 1,537 `notflip`; resolution 1080×1920; 117 reconstructed source videos, with 8–55 frames per source (median 29). The original archive includes a 2,392-image `training` folder and 597-image `testing` folder. The supplied test partition overlaps sources present in training and is not a source-safe test.

The source identifier includes the class and zero-padded video number. Numeric identifiers are reused across class directories; a matching number alone is not proof of the same physical recording. Frame order follows observed filenames, not confirmed elapsed-time stamps or dates.

## Critical split/provenance discrepancy

| Regime | Original research held-out samples | Later V2 manifest test samples |
|---|---:|---:|
| O | 597 | 597 |
| S | 610 | 770 |
| T | 301 | 624 |
| ST | 121 | 161 |

The refactored split constructor now preserves the supplied O testing partition and sorts source identifiers before seeded S/ST assignment. Its original held-out counts are independently checked against the archived manifest: O=597, S=610, T=301, ST=121. These checks establish the sample counts, not identical new model predictions. The later V2 environment-heldout split remains a separate historical design.

**Rule:** Original figures and historical scores are reference evidence under their original split definitions. Do not attribute them to the V2 split or a newly reconstructed split. Newly executed metrics must identify the exact manifest and method that generated them. Preserve both provenance families until the final author review decides which should feature in the README.

Recorded historical held-out Macro F1 (Scratch CNN / ResNet18):
- O: 0.9899 / 0.9950
- S: 0.4400 / 0.9584
- T: 0.8305 / 0.9091
- ST: 0.7906 / 0.8772

These are **historical observations only** and must not be hard-coded as fresh model outputs.

## Architecture and reproducibility findings

1. The prior draft Notebook 01 wrote approximately 43 KB of Python source into `generated/monreader_methods.py` via `%%writefile` for other notebooks to import. That implementation has now been extracted into tracked `src/` modules.
2. The previous scratch-CNN draft included an Optuna path. The revised optional full-rerun workflow uses W&B Bayesian optimization (30 *qualifying* trials per regime, validation Macro F1 selection); this expensive historical rerun is off by default, preserving the accepted submitted findings.
3. The previous one-stage ResNet18 draft differed from the earlier two-stage method. The revised optional comparator code uses head-only followed by full fine-tuning, but the accepted historical outcomes are read rather than retrained by default. A separately recorded V2 fixed comparator uses another recipe and must not be mixed with the original outcomes.
4. Existing charts and explanatory sections need consistent formatting and explicit distinction between validation and held-out measurements.
5. Notebook draft sections, including proposed T/ST work, must be written **before** the researcher executes them. No fake observations or fabricated conclusions.

## Locked incremental temporal proof of principle

- Two progressive ImageNet-pretrained ResNet18 sequences: one T (familiar sources), one ST (unseen sources withheld from training).
- Progressively add chronologically earlier labelled frames at 25%, 50%, 75%, and 90% of available sequence positions. At each checkpoint continue fine-tuning the same model with fixed, previously established parameters; no new sweeps.
- At every checkpoint measure (a) one overall Macro F1 on remaining future frames and (b) separate chronological frame-position windows, alongside flip recall, false negatives, and class/source supports.
- Compare the same final 10% positions where methodologically valid.
- Small windows with inadequate class or source support must be labelled insufficient evidence, not presented as precise scores.
- ST future position is not forecast decay since last seeing that test source; no ST test-source frame is incorporated in training.
- This is offline retrospective progressive model updating, not automatic online learning, and frame positions do not establish elapsed-time or day-to-day performance.

## Draft final reporting sequence

1. Dataset provenance, image properties, source/frame reconstruction, and O/S/T/ST splitting.
2. Scratch CNN architecture, W&B Bayesian trials, selected checkpoints, historical versus fresh performance.
3. ResNet18 training protocol, model comparison, and RGB/grayscale experiment.
4. Source-video uncertainty, failure analysis, robustness, and **fully drafted executable temporal experiment**.
5. Grad-CAM, intermediate activations, interpreted results, business relevance, limitations, and final synthesis.

## Release rules

Do not update final-result language in the README until the five updated notebooks have been executed on the original dataset and checked. Maintain `main` unchanged. Keep all development on `proofread`; independently back up the completed project before merging and deleting the remaining branches.

## Reconstruction checkpoint (pre-execution)

- Five root-level notebooks now use the `updated_` prefix and import tracked `src/` modules.
- Notebook 04 includes the drafted, runnable progressive T/ST analysis, while Notebook 05 includes the draft synthesis.
- Original reporting tables remain accessible as read-only archived references under `submission/results/`.
- Published figures have a layout-generation update; regenerated dataset-specific figure pixels require the original images and notebook execution.
- The source and notebook static check is supplied in `scripts/check_static.py`; CI results must be confirmed separately.

## Phase 8 completed code-review decisions (2026-10-31)

- The historical submitted results and figures remain the accepted reported findings; they are not claimed to be newly measured in the updated notebooks.
- An independently checked split constructor yields historical O=597, S=610, T=301, ST=121 test counts and ST=489 earlier unseen-context frames. The original source split contains **24** withheld sources, not the V2 ENV-03 **29**-source cohort.
- The new progressive ST experiment now reads the original **ST_split.csv** produced in Notebook 01 and checks all 2,989 sample IDs, 24 unseen sources and 489/121 historical context/test counts **before training**. The model never trains or validates on any of these 24 sources.
- New ST evaluation suffixes for 25/50/75% may cover a portion of the historically context-only 489 images, **as an entirely new evaluation**, not as historical ST testing or development. The accepted ST score remains confined to 121 later historical frames.
- O/S/T/ST model-selection tables are report-first by default, with expensive historic W&B sweeps and comparator reruns explicitly opt-in. Notebook 04 remains the only default **new model-training study**, with two progressive ResNet18 paths.
- The original static research figures remain published. New dataset-dependent figures will be rendered on actual image data in Jupyter; no unsupported plots or results have been fabricated.
- GitHub Actions static syntax and figure-path checks passed for commit `5d2721a79c154513b9686df022c0df87a2cb01a1` before this documentation update. This is **not** a full runtime validation of the new models.

**Next boundary:** The researcher needs a local clone, image archive extracted in the expected folder, appropriate PyTorch installation, and Jupyter execution. Review the new progressive outputs before making final scientific conclusions or changing README metrics. `main` is untouched.
