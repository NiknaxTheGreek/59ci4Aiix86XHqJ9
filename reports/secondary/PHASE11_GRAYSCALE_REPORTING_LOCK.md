# Phase 11D grayscale-domain reporting lock

**Status: LOCKED TENTATIVE — latest fresh-run convention**

Decision date: 2026-10-07.

The Phase 11 grayscale-domain experiment remains the retained color-domain ablation. The previous arithmetic two-run mean convention is retired. Until the local-PC authoritative rerun is completed, Phase 11 reports the **newest fresh strict confirmation run** directly.

Fresh confirmation run: `37502301919`.

| Regime | Frozen config | First published F1 | Latest fresh F1 | Current Phase-11 reporting F1 |
|---|---:|---:|---:|---:|
| O | C14 | 0.9734513274 | 0.9734513274 | **0.9734513274** |
| S | C18 | 0.7787610619 | 0.7060606061 | **0.7060606061** |
| T | C06 | 0.9068965517 | 0.8857644991 | **0.8857644991** |
| ST | C07 | 0.8372093023 | 0.8372093023 | **0.8372093023** |

## Reproducibility boundary

The fresh confirmation reused the same frozen scientific inputs: code, data, splits, configs, seeds, 20-epoch rule and threshold, with a constrained CPU execution contract. O reproduced exactly. S, T and ST did not reproduce byte-for-byte; therefore the current GitHub-hosted values are evidence, not the final authoritative project numbers.

No averaging is used.

## Local-authoritative supersession rule

All current GitHub execution results are **tentative pending the manual local-PC/Jupyter rerun**. Phase 12 will rerun the frozen methodology locally and integrate W&B. The local rerun's complete receipts, predictions and metrics will replace the current tentative reporting values. Methodology may not be changed in response to observed results.
