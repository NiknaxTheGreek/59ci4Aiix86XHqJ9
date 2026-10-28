# Historical MonReader Notebook Submission

This directory contains the cleaned, submission-ready version of the original five-notebook MonReader analysis.

## Important status

These notebooks are **historical pre-V2** and are retained for reproducibility. The current V2 analysis on `main` is authoritative.

## What was fixed

- machine-specific Windows paths removed;
- W&B credentials/entity/project ownership externalized;
- W&B sweep creation made opt-in;
- stale/failed/debug cells removed from the narrative;
- corrected final scratch winner identities retained;
- verified tables committed as CSV snapshots;
- all successful notebook figures embedded directly in the cleaned notebooks for reliable GitHub rendering;
- Grad-CAM heatmaps displayed directly in Notebook 5;
- conclusions and limitations placed after the evidence they interpret;
- V2/historical authority boundary made explicit.

## Run in report mode

```bash
python -m pip install -r submission/requirements.txt
jupyter notebook
```

Open the notebooks in numerical order.

The result snapshots are committed and the completed figures are embedded directly in the notebooks, so no W&B account or raw dataset is required to review the completed work.

## Optional W&B mode

Copy `.env.example` into your own environment and set:

```bash
MONREADER_ENABLE_WANDB=1
MONREADER_WANDB_PROJECT=your-project
MONREADER_WANDB_ENTITY=your-entity
WANDB_API_KEY=...
```

Creating a new sweep is separately gated by `MONREADER_CREATE_NEW_SWEEP=1`.

Never commit `.env` or a W&B API key.
