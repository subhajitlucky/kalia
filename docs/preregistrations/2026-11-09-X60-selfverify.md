# Pre-registration — X60: self-verifying visual generation

Registered: 2026-11-09, before the X60 GPU run.

## Idea

KALIA becomes its own critic. For a target caption, sample **N=8** candidate
images from the diffusion, read each one back with the frozen captioner, score
it by the captioner’s teacher-forced cross-entropy of the target caption, and
keep the best candidate. No external reward model, no human labels, no
distillation — only our own captioner and diffusion.

This is the see→draw→see loop turned into **see→draw→judge→choose**.

## Why it matters

X58/X59 showed the 64×64 three-object diffusion is the bottleneck (0.695 at
3.75M, 0.758 at 8.7M generated exact). If the captioner can rank candidates
better than chance, self-verification can turn a sub-threshold generator into
a high-quality one at inference time. The 32×32 toy (0.948) should become
near-perfect.

## Design

- **64×64:** X59 8.7M UNetLarge, X58 captioner (held-out 0.997), X59 6-head
  classifier as the independent judge. 1,000 held-out three-object captions;
  N=8 candidates each.
- **32×32:** v1 diffusion, v2.1 captioner (held-out 0.992), from-scratch
  4-head classifier as the judge. 1,000 held-out two-object captions; N=8.
- **Verifier:** per-candidate teacher-forced CE of the target caption under the
  frozen captioner; select the minimum. The classifier is **not** used for
  selection.
- **Controls:** N=1 baseline; oracle (any candidate exact by the classifier);
  verifier selection rate (captioner readback exact).

## Bars

| ID | prediction | threshold |
|---|---|---|
| **S-1** | 64×64 selected classifier exact | ≥ 0.90 |
| **S-2** | 32×32 selected classifier exact | ≥ 0.98 |
| **S-3** | verifier gain over N=1 | ≥ +0.10 (64×64), ≥ +0.03 (32×32) |
| **S-4** | no external reward; from scratch; artifacts | pass |

## Decision rule

- **S-1…S-3 pass:** release **v0.10.0-selfverify** with the mechanism and a
  “8 candidates, the model’s choice” gallery.
- **Any fail:** log the verifier boundary; no tuning after the result.
- Public base remains v0.3.1.

## Cost

≤ 2 GPU-h.
