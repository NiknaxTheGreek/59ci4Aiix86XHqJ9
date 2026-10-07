# Phase 11D deterministic confirmation

**Status: REVIEW_REQUIRED_DO_NOT_LOCK**

Confirmation reran the frozen O/C14, S/C18, T/C06 and ST/C07 grayscale-domain experiments with the same code, data, splits, configs, seeds, 20-epoch rule and threshold, while clamping the oneDNN CPU ISA to AVX2 to eliminate the hardware-dependent path identified in the S/C18 audit.

| Regime | F1 fresh | F1 published | Summary exact | Predictions exact | Checkpoint exact |
|---|---:|---:|:---:|:---:|:---:|
| O | 0.9734513274 | 0.9734513274 | YES | YES | YES |
| S | 0.7060606061 | 0.7787610619 | NO | NO | NO |
| T | 0.8857644991 | 0.9068965517 | NO | NO | NO |
| ST | 0.8372093023 | 0.8372093023 | NO | NO | NO |

No hyperparameter search, validation, early stopping, threshold tuning, candidate replacement, or primary-result reopening was performed.

This confirmation is secondary reproducibility evidence only.
