# Phase 11D — Full-corpus grayscale-domain ablation

## Status before execution

This is a post-selection secondary analysis. It cannot change the frozen primary champions:
O=C14, S=C18, T=C06, ST=C07.

## Scientific question

How much task-relevant information is lost when color is removed from the entire MonReader dataset, while keeping the model architecture, frozen regime memberships, training hyperparameters, epoch count, seed policy, threshold, and evaluation populations unchanged?

This is intentionally different from the earlier test-time grayscale sensitivity analysis. The earlier analysis trained on RGB and transformed only test images. Phase 11D trains and tests in the grayscale domain.

## Frozen image transform

Every one of the 2,989 authoritative source images is transformed at source resolution with Rec.709 luminance:

Y = 0.2126 R + 0.7152 G + 0.0722 B

Y is rounded/clipped to uint8 and repeated into three equal channels so the existing 3-channel ScratchCNN architecture is unchanged.

The transform is applied before the existing aspect-preserving 398x224 preprocessing.

## Full-corpus audit gate

Before any grayscale-domain training:

- verify the authoritative archive byte size and SHA-256;
- verify the canonical DATA-001 manifest SHA-256;
- account for exactly 2,989 unique sample IDs;
- transform all 2,989 images with zero decode/transform failures;
- verify output shape equals input shape for every image;
- verify all three output channels are identical for every image;
- write one audit row per sample with a SHA-256 of the transformed pixel array;
- produce deterministic ORIGINAL -> GRAYSCALE contact sheets spanning both labels, all four environments, and both supplied train/test sources.

The user visually inspected real MonReader ORIGINAL -> GRAYSCALE examples on 2026-10-04 and accepted the transform quality before this run.

## Fixed-config grayscale-domain runs

Exactly one already-frozen champion configuration is retrained per regime:

- O: C14
- S: C18
- T: C06
- ST: C07

No candidate search, validation, early stopping, threshold tuning, or hyperparameter adaptation is permitted.

Training remains exactly 20 epochs with the candidate's frozen optimizer, batch size, learning rate, dropout, architecture, weight decay, and seed from the fixed C01-C20 bank.

Each regime uses the unchanged frozen membership file and evaluates only its existing primary test role.

## Comparisons

For each regime report:

1. RGB-trained / RGB-tested frozen champion.
2. RGB-trained / grayscale-tested sensitivity result from Phase 11C.
3. Grayscale-trained / grayscale-tested Phase 11D result.

Use threshold 0.5 throughout.

Compute frame-level and canonical-video/group paired bootstrap intervals with 5,000 resamples for:

- grayscale-domain minus original RGB;
- test-time grayscale minus original RGB;
- grayscale-domain minus test-time grayscale.

## Interpretation boundary

This analysis may support statements about the information retained after removal of color and whether adaptation to a grayscale training domain recovers performance.

It must not be used to replace any frozen O/S/T/ST champion or to reopen primary model selection.
