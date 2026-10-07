# Phase 11 finalization status

**Status: LOCKED FOR HANDOFF — current GitHub results are tentative pending the local authoritative rerun.**

Decision date: 2026-10-07.

Phase 11 is now closed as a GitHub-hosted exploratory phase. No further Phase 11 model tuning or nuisance-removal development is allowed before the local rerun.

## Retained Phase 11 analyses

### Exact-duplicate sensitivity
**CONFIRMED STRUCTURAL NULL.** The decoded-pixel audit found zero exact duplicate groups, so no duplicate-removal rerun is required.

### Best-O shortcut perturbation
**CONFIRMED SECONDARY RESULT.** The frozen C14 perturbation analysis is retained as a post-hoc sensitivity result. It is not a limb-removal/inpainting pipeline and does not change champion selection.

### Resolution sensitivity
**CONFIRMED SECONDARY RESULT.** Best-O/C14 sensitivity to half and quarter resolution is retained with its paired uncertainty analysis.

### Full-corpus grayscale-domain ablation
**CONFIRMED SECONDARY EXPERIMENT; NUMBERS TENTATIVE.** The successful 2,989-image Rec.709 grayscale-domain experiment is retained. The current reporting convention uses the newest fresh strict confirmation run `37502301919` rather than the former two-run arithmetic mean:

- O/C14: **0.9734513274**
- S/C18: **0.7060606061**
- T/C06: **0.8857644991**
- ST/C07: **0.8372093023**

O reproduced exactly. S/T/ST did not reproduce byte-for-byte on GitHub-hosted CPU runs, which is one reason the final authoritative execution will be performed in one controlled local environment.

## Removed from the active project

The following are deliberately removed from the active Phase 11 tree and are preserved only in the external pre-cleanup repository backup:

- all hand/arm/limb removal, segmentation, masking, reconstruction and inpainting development;
- all Project4 limb-removal bakeoffs and dev60 work;
- the pretrained hand/arm segmentation path;
- augmentation semantic/ablation experiments;
- chroma/naturalistic-chroma counterfactual experiments.

The dev60 limb-removal experiment failed the human visual requirement: masks could damage valid page content, additional hands could remain unmasked, and reconstruction was not sufficiently reliable to justify further project time.

## Temporal/video-model extension

**DEFERRED / WAIVED FOR THIS PROJECT VERSION.** The attempted GitHub workflow did not yield a retained scientific result. This optional exploratory extension is not required for Phase 11 closure and is removed from the active execution path.

## Phase 11 lock

Phase 11 is locked methodologically and dispositionally. Its retained GitHub results are **tentative evidence** only. They cannot trigger new GitHub tuning and they will not be treated as the final published numbers.

Phase 12 now becomes the authoritative execution stage: reproduce the complete frozen methodology in the user's local Jupyter/PC environment, capture full receipts, integrate W&B, and replace the tentative GitHub results with the local authoritative outputs.
