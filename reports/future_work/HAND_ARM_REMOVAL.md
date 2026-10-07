# Future work — hand/arm removal

Naturalistic hand/arm removal is **not part of the reported MonReader results** and is not used to construct any training, validation, or test dataset in the presented project.

## Why it was investigated

Hands and forearms are an inherent part of the MonReader image domain. In realistic deployment, MonReader should therefore be expected to receive images in which a hand is present; removing hands is **not a prerequisite for the task to function**.

The removal exercise was instead explored as a **secondary analysis** intended to probe what visual information the classifier may be using. The scientific motivation was to ask whether suppressing hand/arm information would materially change model behaviour and therefore help distinguish page-turning cues from possible hand-related shortcuts. It was never intended to redefine the canonical input domain.

## Why the approach was not retained

A pretrained human-parsing approach was selected because its reported arm-segmentation capability suggested a high likelihood of effective hand/arm localization. In practice, that promise did **not** translate into an acceptable end-to-end removal system. The overall pretrained removal approach failed decisively under visual inspection: although occluding regions could often be identified, reconstruction of the covered document area was frequently implausible or destructive. Repeated attempts to improve the masks, inpainting, compositing and reconstruction strategy did not resolve the underlying problem sufficiently for scientific use.

The central difficulty is that detecting the occluder and reconstructing what was hidden are fundamentally different problems. Even a strong segmentation model does not provide the true content that was covered by the hand.

## Key unresolved obstacles

- **No ground truth beneath real occlusions.** The supplied images do not reveal the true pixels hidden under a hand or forearm, so exact reconstruction quality cannot be verified directly.
- **Covered content could not be recovered reliably from related frames.** Nearby or visually similar images from the supplied dataset were investigated as potential sources of real page pixels, but the hidden content could not be extracted, aligned and transferred with sufficient completeness or consistency. Persistent occlusion, motion, viewpoint changes, page deformation and imperfect registration meant that same-video or similar-frame information was not a dependable substitute for ground truth.
- **Large missing regions.** Hand/arm occlusions can cover a substantial fraction of a frame, leaving generic inpainting severely underconstrained.
- **Structured document content.** Text, tables, diagrams, margins, ruled lines and page edges require semantic and geometric consistency that generic natural-image inpainting does not guarantee.
- **Hallucinated content risk.** A reconstruction can look superficially smooth while inventing incorrect characters, lines, shapes or document structure.
- **Residual-hand and boundary artifacts.** Incomplete masks or imperfect reconstruction can leave recognizable skin/hand structure, seams, blur or low-detail regions.
- **Tuning did not remove the core failure mode.** Adjusting mask extent, reconstruction method, temporal transfer and compositing could change individual outputs but did not yield a consistently acceptable corpus.
- **Evaluation without a reliable reference metric.** Pixel preservation outside the mask does not establish that reconstructed pixels inside the mask are visually or semantically correct. A convincing no-reference metric was not available to replace human inspection.
- **Potential classifier shortcuts.** Reconstruction artifacts themselves could become predictive features. A transformed corpus could therefore introduce a new shortcut while attempting to remove another one.
- **Video information is incomplete.** Nearby frames sometimes expose pixels that are occluded elsewhere, but the available frame sequences do not guarantee that every covered region is ever visible in a geometrically compatible view.

## Future direction

A future implementation would need a substantially stronger, document-aware and preferably temporally informed reconstruction method, together with independent residual-object checks, document-structure preservation checks and a prospectively defined QA protocol. It would also need a principled strategy for dealing with regions for which the supplied data contain no recoverable ground truth.

Any such work should remain a **secondary interpretability/robustness analysis** unless there is a separate deployment reason to alter the natural MonReader input domain.

No hand/arm-removal performance tables, transformed-corpus results or downstream classifier results are presented in this repository.
