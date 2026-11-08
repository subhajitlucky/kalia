# Pre-registration — KALIA-Vision v2: image encoder + captioning

Registered: 2026-11-08, before the v2 GPU run.

## Goal

Give KALIA the **“see”** counterpart to the diffusion generator: a
from-scratch image encoder that feeds continuous image tokens into frozen
KALIA, which then captions the scene. This is a capability module, not a new
base model.

## Data

The existing synthetic two-object dataset (50k/5k/5k, 32×32, shapes/colors/
left-right positions). No external images, no pretrained vision weights.

## Model

- From-scratch CNN encoder → 4 continuous image tokens (512-d each).
- Frozen v0.3.1 KALIA; rank-16 LoRA on attention/MLP to use the image prefix.
- Caption decoder prompt: `Describe: `; loss on the caption tokens.
- 8 epochs, batch 64, AdamW; greedy caption at evaluation.

## Bars

| ID | prediction | threshold |
|---|---|---|
| **V2-1** | exact two-object caption | ≥ 0.80 |
| **V2-2** | each attribute (left/right shape/color) | ≥ 0.90 |
| **V2-3** | from scratch | no pretrained vision; KALIA frozen |
| **V2-4** | image is used | shuffled-prefix exact drops ≥ 0.20 |
| **V2-5** | artifacts | encoder+LoRA checkpoint, metrics, samples |

## Decision rule

- **V2-1…V2-4 pass:** release **v0.5.2-vision-see** capability update.
- **Any fail:** log the boundary; no tuning after the result.
- Next: cross-modal binding test (text-trained edit compiler applied to image
  captions/audio commands).

## Cost

≤ 3 GPU-h; fits the remaining window.
