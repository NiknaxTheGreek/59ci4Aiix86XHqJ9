# Phase 11D — Full-corpus grayscale-domain ablation

**Full-corpus transform gate: PASS — 2,989/2,989 images transformed, 0 failures.**

Real MonReader ORIGINAL → GRAYSCALE examples were visually inspected and accepted by the user before this full run.

| Regime | Frozen config | RGB→RGB F1 | RGB→Gray F1 | Gray→Gray F1 | Δ Gray→Gray vs RGB→RGB |
|---|---:|---:|---:|---:|---:|
| O | C14 | 0.9983 | 0.6840 | 0.9735 | -0.0248 |
| S | C18 | 0.8196 | 0.6225 | 0.7788 | -0.0409 |
| T | C06 | 0.8800 | 0.6695 | 0.9069 | +0.0269 |
| ST | C07 | 0.7040 | 0.7109 | 0.8372 | +0.1332 |

## Video/group paired-bootstrap intervals

### O
- Gray→Gray minus RGB→RGB: -0.0248 (95% CI [-0.0447, -0.0092])
- RGB→Gray minus RGB→RGB: -0.3143 (95% CI [-0.4128, -0.2320])
- Gray→Gray minus RGB→Gray: +0.2895 (95% CI [+0.2014, +0.3874])

### S
- Gray→Gray minus RGB→RGB: -0.0409 (95% CI [-0.1537, +0.0636])
- RGB→Gray minus RGB→RGB: -0.1971 (95% CI [-0.3515, -0.0580])
- Gray→Gray minus RGB→Gray: +0.1562 (95% CI [+0.0426, +0.2879])

### T
- Gray→Gray minus RGB→RGB: +0.0269 (95% CI [-0.0002, +0.0546])
- RGB→Gray minus RGB→RGB: -0.2105 (95% CI [-0.3052, -0.1260])
- Gray→Gray minus RGB→Gray: +0.2374 (95% CI [+0.1538, +0.3295])

### ST
- Gray→Gray minus RGB→RGB: +0.1332 (95% CI [-0.0558, +0.3554])
- RGB→Gray minus RGB→RGB: +0.0069 (95% CI [-0.2184, +0.2391])
- Gray→Gray minus RGB→Gray: +0.1263 (95% CI [+0.0227, +0.2620])

## Interpretation boundary

This is a post-selection secondary analysis. It cannot replace or reopen the frozen primary O/S/T/ST champions.
It distinguishes abrupt test-time grayscale shift from performance after training adaptation to the grayscale domain.

