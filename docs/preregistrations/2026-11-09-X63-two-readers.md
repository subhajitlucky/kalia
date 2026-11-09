# Pre-registration — X63: independent second reader

Registered: 2026-11-09, before the X63 GPU runs.

## Question

Does an **independent second reader** close the verifier–judge gap?

X60’s CNN captioner reads the selected 64×64 images back at **0.994**, while
the independent classifier judges them at **0.953**. The 4.1 pp gap is
verifier–judge disagreement that one captioner cannot see. A second reader
with a different image encoder should catch some of those false positives.

## Treatment

- **Reader 1:** the X58 CNN captioner (held-out 0.997).
- **Reader 2:** a from-scratch **ViT-style** image encoder — 8×8 patches,
  4 transformer blocks (256-d, 8 heads), 4 learned query tokens cross-attending
  to the 64 patches, projected to 4×512. Same frozen KALIA + LoRA decoder,
  trained on the same 64×64 seen combinations. No pretrained vision.
- **Selection:** sample N=8 candidates per target; score each with both
  readers’ teacher-forced CE of the target caption; select by
  `max(CE_cnn, CE_vit)` (both must agree). The independent 6-head classifier
  is the judge and is **not** used for selection.
- **Controls:** single-reader selection (CNN only); mean-CE ensemble; oracle
  any-of-8.

## Bars

| ID | prediction | threshold |
|---|---|---|
| **E-1** | ensemble-max selected classifier exact | ≥ 0.97 |
| **E-2** | ensemble gain over single reader | ≥ +1.5 pp |
| **E-3** | two independent readers; classifier not in selection | pass |
| **E-4** | from scratch; artifacts | pass |

## Decision rule

- **E-1/E-2 pass:** release **v0.12.0-two-readers** with the ensemble.
- **Any fail:** log the verifier boundary; no tuning after the result.
- Public base remains v0.3.1.

## Cost

≤ 2 GPU-h.
