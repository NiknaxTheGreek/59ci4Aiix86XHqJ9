# Future work — hand/arm removal

Naturalistic hand/arm removal is **not part of the reported MonReader results** and is not used to construct any training, validation, or test dataset in the presented project.

It remains a possible follow-on project because hands and forearms can obscure substantial portions of a page. A reliable solution would require more than detecting the occluder: it would also need to reconstruct document structure plausibly while avoiding the introduction of synthetic shortcuts.

Key unresolved obstacles are:

- **No ground truth beneath real occlusions.** The supplied images do not reveal the true pixels hidden under a hand or forearm, so exact reconstruction quality cannot be verified directly.
- **Large missing regions.** Occlusions can cover a substantial fraction of a frame, making generic image inpainting underconstrained.
- **Structured document content.** Text, tables, diagrams, margins and page edges require semantic and geometric consistency that generic natural-image inpainting does not guarantee.
- **Hallucinated content risk.** A visually smooth reconstruction can still invent incorrect characters, lines, shapes or page structure.
- **Residual-hand and boundary artifacts.** Incomplete masks or imperfect reconstruction can leave recognizable skin/hand structure, seams, blur or low-detail patches.
- **Evaluation without a reliable reference metric.** Pixel-preservation checks outside the mask do not establish that the reconstructed region is visually or semantically valid. Any future system would need independent residual-object checks, document-structure checks and a prospectively designed human QA protocol.
- **Potential classifier shortcuts.** Reconstruction artifacts themselves could become predictive features, so any cleaned-domain model would require strict leakage and artifact-sensitivity testing before scientific interpretation.
- **Video information is incomplete.** Nearby frames may expose some previously occluded pixels, but persistent occlusion, motion, viewpoint changes and imperfect registration prevent them from serving as complete ground truth.

A future implementation should therefore use a document-aware and preferably temporally informed reconstruction method, validate it independently of the model used to generate masks, and establish a prospective acceptance protocol before any transformed corpus is used for model training or reported analysis.

No hand/arm-removal performance results are presented in this repository.
