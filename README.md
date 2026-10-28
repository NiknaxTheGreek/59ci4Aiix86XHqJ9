# MonReader — Flip / Not-Flip Image Classification

MonReader is a binary computer-vision project that classifies a single video frame as `flip` or `notflip`.

This repository now contains **two clearly separated analysis tracks**:

1. **Current V2 scientific rebuild — authoritative.**  
   This is the prospectively frozen train/test-only protocol on `main`.
2. **Historical notebook submission — reproducibility record.**  
   The five cleaned notebooks under [`submission/notebooks`](submission/notebooks) preserve the earlier O/S/T/ST workflow, W&B-based scratch-CNN tuning, ResNet18 comparison, stress testing, uncertainty, Grad-CAM, and final interpretation.

The historical notebooks are intentionally labelled **`HISTORICAL_NON_AUTHORITATIVE_PRE_V2`** so that their validation/Bayesian methodology cannot be mistaken for the current V2 protocol.

---

## Current V2 results — authoritative

### Frozen scratch-CNN champions

| Regime | Champion | F1 | Interpretation |
|---|---:|---:|---|
| O | C14 | **0.9983** | supplied source-joint benchmark |
| S | C18 | **0.8196** | held-out ENV-03 |
| T | C06 | **0.8800** | later frames of known videos |
| ST | C07 | **0.7040** | held-out ENV-03 + later frames |

### Fixed ImageNet ResNet18 comparator

| Regime | Scratch champion F1 | ResNet18 F1 | ΔF1 (ResNet − scratch) |
|---|---:|---:|---:|
| O | 0.9983 | 0.9931 | -0.0052 |
| S | 0.8196 | 0.7807 | -0.0390 |
| T | 0.8800 | 0.8807 | +0.0007 |
| ST | 0.7040 | 0.7196 | +0.0156 |

The existing V2 synthesis and comparator reports remain the source of truth for current conclusions.

---

## Historical submission notebooks

The cleaned historical notebook sequence is:

1. [`01_data_and_splits.ipynb`](submission/notebooks/01_data_and_splits.ipynb)  
   Dataset audit, source/video structure, and historical O/S/T/ST split logic.
2. [`02_cnn_selection.ipynb`](submission/notebooks/02_cnn_selection.ipynb)  
   Portable W&B integration, historical Bayesian scratch-CNN selection, corrected frozen winners, and generalization results.
3. [`03_model_comparison.ipynb`](submission/notebooks/03_model_comparison.ipynb)  
   Scratch CNN vs ResNet18 and the closed RGB/grayscale branch.
4. [`04_generalization_and_stress_tests.ipynb`](submission/notebooks/04_generalization_and_stress_tests.ipynb)  
   Robustness envelope, source-video bootstrap uncertainty, and failure-mode diagnostics.
5. [`05_final_analysis.ipynb`](submission/notebooks/05_final_analysis.ipynb)  
   Final synthesis, report figures, Grad-CAM heatmaps, activation summaries, conclusions, and limitations.

The successful numerical outputs are committed as CSV snapshots under [`submission/results`](submission/results). The completed figures — including the Grad-CAM regime heatmaps — are embedded directly inside the notebooks as Jupyter attachments, so GitHub renders the verified visual outputs without requiring a separate asset download, the original W&B account, or the local dataset.

---

## Historical results snapshot

Under the historical pre-V2 ResNet18 analysis:

| Regime | Macro F1 | Source-video 95% CI |
|---|---:|---:|
| O | **0.994969** | [0.986128, 1.000000] |
| S | **0.958387** | [0.909802, 0.993352] |
| T | **0.909140** | [0.865657, 0.946739] |
| ST | **0.877246** | [0.747988, 0.972281] |

These values are **historical**. They are retained to document the original experimental reasoning and must not replace the V2 metrics above.

---

## Reproducible report mode

The historical notebooks default to **report mode**. A reviewer can open them and inspect all committed tables and figures without:

- W&B credentials;
- the original 940 MB data archive;
- a specific Windows directory;
- a GPU;
- access to the original sweep IDs.

Install the notebook dependencies with:

```bash
python -m pip install -r submission/requirements.txt
```

If you have a local copy of the dataset, optionally set:

```bash
export MONREADER_DATA_ROOT=/path/to/monreader/data
```

On PowerShell:

```powershell
$env:MONREADER_DATA_ROOT="C:\path\to\monreader\data"
```

---

## Portable W&B integration

W&B is **opt-in** and configured entirely through environment variables.

```bash
MONREADER_ENABLE_WANDB=1
MONREADER_WANDB_PROJECT=monreader-historical-repro
MONREADER_WANDB_ENTITY=my-team       # optional
WANDB_API_KEY=...                    # or run `wandb login`
WANDB_MODE=online
```

No API key, username, entity, project owner, or machine-specific W&B credential is stored in the notebooks.

Notebook 2 contains the historical Bayesian sweep search space and uses `submission/src/repro.py` for reusable W&B setup. Creating a new sweep is additionally gated by:

```bash
MONREADER_CREATE_NEW_SWEEP=1
```

This prevents a reviewer from accidentally launching training merely by running the notebook.

---

## Repository layout

```text
submission/
├── notebooks/
│   ├── 01_data_and_splits.ipynb
│   ├── 02_cnn_selection.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_generalization_and_stress_tests.ipynb
│   └── 05_final_analysis.ipynb
├── results/                     # committed verified result snapshots
├── src/repro.py                 # portable paths + W&B helpers
├── requirements.txt
└── .env.example

reports/
├── TECHNICAL_REPORT.md
└── BUSINESS_RECOMMENDATIONS.md
```

---

## Key methodological lesson

Across both the historical work and the current V2 rebuild, the central lesson is the same:

> **Nominal frame-level performance is not enough. MonReader must be evaluated under source-aware and temporally aware conditions before any deployment claim is made.**

The O, S, T and ST regimes answer different generalization questions. They should not be treated as a single global model leaderboard.

---

## Reports

- [Technical report](reports/TECHNICAL_REPORT.md)
- [Business recommendations](reports/BUSINESS_RECOMMENDATIONS.md)
- Historical notebook package: [`submission/`](submission/)
- Current V2 synthesis: `reports/synthesis/CROSS_REGIME_SYNTHESIS.md`
- Current V2 ResNet18 comparator: `reports/resnet18/RESNET18_COMPARATOR_REPORT.md`
