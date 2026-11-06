# Pre-registration — KALIA-Vision v1: two-object scenes

Registered: 2026-11-06, before the v1 GPU run.

## Goal

Extend the v0 image module from one shape to **two objects in one 32×32 image**:
a left object and a right object, each with its own shape and color. KALIA
v0.3.1 stays the frozen text encoder; the diffusion U-Net is trained from
scratch. This is a capability step, not a base-model change.

## Data

- 32×32 RGB, two shapes per image: one on the left, one on the right.
- Shapes: circle, square, triangle.
- Colors: red, green, blue, yellow.
- Captions: “a red circle on the left and a blue square on the right.”
- 50,000 train / 5,000 val / 5,000 test.
- Fully synthetic; no external images; no pretrained vision weights.

## Model

- Frozen KALIA v0.3.1 caption embedding (512-d).
- Conditional DDPM U-Net from scratch (same 3.75M architecture as v0).
- 25,000 steps, batch 128; DDIM 100-step sampling.

## Bars

| ID | prediction | threshold |
|---|---|---|
| **V1-1** | left shape | ≥ 0.80 |
| **V1-2** | left color | ≥ 0.80 |
| **V1-3** | right shape | ≥ 0.80 |
| **V1-4** | right color | ≥ 0.80 |
| **V1-5** | exact two-object scene | ≥ 0.50 |
| **V1-6** | from scratch | no pretrained vision weights; KALIA frozen |

A small classifier trained from scratch on real images is the diagnostic.

## Decision rule

- **V1-1…V1-5 pass:** release **v0.5.1-vision-two** capability update.
- **Any fail:** log the boundary; no release. Do not tune after the result.
- Audio remains the next window.

## Cost

Data CPU. Training ≤ 2 GPU-h; fits the current window.
