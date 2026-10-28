# MonReader Business Recommendations

## Executive recommendation

Do **not** make a production-readiness decision from the nominal O score alone.

The historical analysis shows near-perfect nominal performance but materially weaker performance under source and temporal stress. The current V2 rebuild reaches the same strategic conclusion from a stricter protocol: MonReader should be evaluated and piloted under source-aware and later-frame conditions before deployment.

The recommended business posture is therefore **controlled pilot, targeted data expansion, and failure-aware monitoring** rather than immediate broad deployment.

---

## 1. Treat unseen-source performance as the primary deployment risk

Historical O Macro F1 was approximately 0.995, but the source/temporal stress regimes were lower and less certain. The current V2 S/ST benchmarks are also materially harder than O.

### Recommendation

Before deployment, build a pilot dataset containing genuinely new recording sources, devices, backgrounds, lighting conditions, users and page-handling styles.

### Business reason

A nominal score that benefits from familiar sources can understate real-world support burden and customer-facing errors.

---

## 2. Prioritize false negatives

Historical failure analysis found false negatives to be the dominant error mode, especially under T and ST.

### Recommendation

Define product requirements around the operational cost of a missed flip:

- if a missed flip is expensive, use temporal smoothing or a review/retry path;
- monitor false-negative rate by source/session;
- do not optimize only for overall accuracy.

### Business reason

A system can have high aggregate accuracy while still failing on the exact events the product needs to detect.

---

## 3. Add temporal context before scaling

Single-frame classification discards information available in adjacent frames.

### Recommendation

For a production version, evaluate a lightweight temporal layer on top of the frozen frame classifier:

- rolling probability smoothing;
- hysteresis around state changes;
- short-window majority/median voting;
- explicit transition detection.

This should be tested prospectively and must not be tuned on the final deployment benchmark.

### Business reason

Page flips are temporal events. Using a short sequence can reduce isolated high-confidence errors without requiring a much larger model.

---

## 4. Expand the data where the model is weak, not where it is already saturated

The nominal O regime is already near ceiling.

### Recommendation

Spend annotation effort on:

- new source videos;
- later-frame portions of videos;
- difficult lighting;
- partial occlusion;
- hand/arm variation;
- camera viewpoint changes;
- repeated or ambiguous page motion.

### Business reason

More data from already-easy conditions is unlikely to produce the same return as data targeted at the stress regimes.

---

## 5. Keep model confidence as a monitoring signal, not a guarantee

The historical ST analysis contained a high share of confident errors.

### Recommendation

Log probability distributions and confidence by source/session in pilot deployments, but do not equate confidence with correctness.

Flag:

- sustained low-confidence windows;
- abrupt distribution shifts;
- sources with repeated confident misses.

### Business reason

High confidence can make silent failures more dangerous because they are less likely to be noticed without explicit monitoring.

---

## 6. Use interpretability for diagnostics, not product claims

Grad-CAM and intermediate activations are useful for investigating failure cases, but they do not prove why a prediction occurred.

### Recommendation

Use heatmaps internally for:

- debugging;
- dataset-quality audits;
- identifying suspicious recurring regions;
- selecting hypotheses for controlled ablations.

Do not market the model as “explainable” solely because Grad-CAM plots exist.

---

## 7. Preserve experiment governance

The project now has a stricter V2 protocol specifically because earlier adaptive analysis made it easy to over-interpret results.

### Recommendation

For every future production experiment:

- freeze the evaluation population before testing;
- freeze threshold/configuration before final benchmark access;
- record all candidate configurations;
- use source/video-level uncertainty;
- prohibit error-driven retuning after the final benchmark is opened.

### Business reason

This makes performance claims auditable and reduces the risk of a model that looks strong in a report but fails to reproduce.

---

## 8. Pilot success criteria

A reasonable next-stage pilot should require all of the following before expansion:

1. predeclared source-safe and temporal benchmarks;
2. confidence intervals at the source/video level;
3. acceptable false-negative rate;
4. stable performance across new sources;
5. no single source contributing a disproportionate share of failures;
6. documented rollback/fallback behaviour;
7. reproducible model and threshold provenance.

---

## 9. Recommended product roadmap

### Phase A — controlled validation
Use the current V2 champions and collect new source/session data without changing the final benchmark.

### Phase B — temporal decision layer
Test a simple sequence-aware post-processor on a development set.

### Phase C — targeted robustness improvements
Only after failure analysis identifies a repeatable mechanism, test controlled augmentations or additional data.

### Phase D — monitored pilot
Deploy to a limited set of users/sources with probability logging and source-level monitoring.

### Phase E — scale
Expand only if source-safe and temporal performance remains within predeclared limits.

---

## Final business conclusion

MonReader is promising, but the business value will come from **robust event detection across new sources**, not from maximizing a nominal frame-level score that is already close to 100%.

The best next investment is therefore not another round of unconstrained tuning. It is a controlled robustness pilot with new sources, explicit false-negative targets, temporal monitoring, and strict experiment governance.
