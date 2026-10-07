# Phase 12–14 — Local Authoritative Execution and Release Contract

**Decision date:** 2026-10-07  
**Status:** LOCKED EXECUTION CONTRACT

## Authority rule

All numerical results currently stored in GitHub are **tentative evidence**. They are retained for continuity, debugging, and comparison only.

The final authoritative MonReader results will come from one controlled manual execution in the user's local PC/Jupyter environment, with W&B logging integrated into that same execution.

The local run must use the already-frozen scientific methodology. Observed local metrics may replace tentative GitHub metrics, but they may not be used to change the protocol, candidate bank, split definitions, thresholds, epoch budget, preprocessing, or selection hierarchy.

If the local authoritative numbers differ from the tentative GitHub numbers, report the local values and preserve the discrepancy as reproducibility evidence. Do not average runs.

## Phase 12 — Local authoritative rerun + W&B

Run the complete frozen methodology in one local environment.

### 12.1 Environment receipt
Record:
- OS / hardware / CPU / GPU;
- Python version;
- package lock / exact versions;
- torch / torchvision / CUDA versions;
- deterministic flags and thread settings;
- git commit SHA;
- dataset archive size and SHA-256;
- candidate-bank SHA-256.

Use one environment for the full authoritative run unless a clearly documented hardware boundary is unavoidable.

### 12.2 Data and split verification
Reverify:
- 2,989 images;
- 117 videos;
- dataset SHA-256;
- master manifest;
- O/S/T/ST split membership and counts;
- no silent sample omission;
- all existing V2 split invariants.

Do not regenerate scientific split definitions from observed model results.

### 12.3 Baselines
Rerun the frozen baseline methodology and log receipts to W&B.

### 12.4 Scratch-CNN primary experiments
For O, S, T and ST:
- use the same frozen C01–C20 bank;
- train every candidate exactly 20 epochs;
- no validation loader;
- no early stopping;
- no Bayesian/adaptive search;
- final epoch checkpoint is primary;
- threshold 0.5;
- freeze all 20 candidate receipts before test evaluation;
- evaluate test exactly under the frozen protocol;
- select the regime champion by F1 -> balanced accuracy -> PR-AUC -> lower parameters -> candidate ID.

The local result table replaces the current tentative GitHub result table.

### 12.5 Cross-regime synthesis
Recompute from the local predictions:
- champion envelope for Best O / S / T / ST;
- fixed O-winning candidate across O/S/T/ST;
- frame bootstrap;
- video/group bootstrap;
- selection-conditioned uncertainty;
- all figures/tables.

Do not use a global average of O/S/T/ST as a model-selection criterion. Each regime remains a different robustness question.

### 12.6 Fixed ResNet18 comparator
Rerun the frozen fixed comparator recipe locally and log it separately from scratch champion selection.

### 12.7 Retained Phase 11 analyses
Only the locked retained analyses are rerun:
1. exact-duplicate structural audit;
2. Best-O fixed shortcut perturbation;
3. fixed resolution sensitivity;
4. full-corpus Rec.709 grayscale-domain ablation.

Do **not** restore:
- limb/hand/arm removal or reconstruction;
- pretrained limb segmentation;
- augmentation experiments;
- chroma/naturalistic-chroma counterfactuals;
- temporal/video-model extension for this project version.

For the grayscale-domain ablation, the current GitHub convention uses the newest fresh confirmation run rather than an arithmetic mean. The local rerun supersedes those tentative values.

### 12.8 W&B structure
Use the prior-project W&B pattern, adapted to MonReader:
- one project for the authoritative MonReader V2 execution;
- groups by phase/regime;
- deterministic run names;
- config JSON stored with each run;
- dataset/split/config/git hashes as run metadata;
- per-epoch training history;
- final test metrics;
- prediction tables/artifacts;
- confusion matrices and selected plots;
- champion tags;
- checkpoint artifacts where practical;
- explicit tags: tentative comparison source vs local-authoritative.

W&B is an observability/receipt layer. It must not introduce sweeps, adaptive search, or model-selection changes.

### 12.9 Supersession receipt
At the end of Phase 12, generate one machine-readable authority file mapping every final table/metric to:
- local run ID;
- W&B run ID;
- prediction artifact hash;
- checkpoint hash;
- split hash;
- candidate config hash;
- git commit;
- environment receipt.

This receipt formally marks the local outputs as authoritative and the current GitHub numbers as superseded tentative evidence.

## Phase 13 — Reproducibility and documentation

Using **only the authoritative local outputs**:
1. recompute metrics from saved predictions;
2. verify champion hierarchy programmatically;
3. verify all hashes and split counts;
4. verify 20 epochs / no validation / no early stopping from logs;
5. reconcile local files against W&B values;
6. run final error analysis;
7. update README, Technical Report, Executive Summary and Model Card to one consistent number set;
8. clearly separate primary results from retained post-hoc Phase 11 analyses;
9. document any local-vs-GitHub reproducibility differences without averaging them.

## Phase 14 — Public release

Before tagging:
- remove private backup receipts/links and local machine secrets;
- keep runnable code, frozen configs, manifests and authoritative result assets;
- keep only final public-facing secondary evidence;
- perform a clean-clone reproduction check;
- verify documentation consistency;
- tag v1.0 only after all gates pass.

## Current Phase 11 tentative grayscale values

Until Phase 12 supersedes them, the newest fresh strict confirmation run is used directly:

- O / C14: 0.9734513274
- S / C18: 0.7060606061
- T / C06: 0.8857644991
- ST / C07: 0.8372093023

No arithmetic averaging of the two GitHub grayscale runs is permitted.

## Final principle

**Methodology is frozen; execution evidence is replaceable.**

GitHub-hosted results are tentative. The single controlled local Jupyter/W&B execution is the final authority.
