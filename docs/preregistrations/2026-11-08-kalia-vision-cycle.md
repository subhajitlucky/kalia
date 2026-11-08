# Pre-registration — KALIA-Vision cycle: see → draw → see

Registered: 2026-11-08, before the cycle GPU run.

## Question

Do KALIA's **seeing** and **drawing** modules share a compositional visual
representation? Take a real synthetic image, caption it with v2.1, generate a
new image from that caption with v1, then caption the generated image. If the
caption survives the cycle, the two modules agree on shape/color/position.

No training; inference only. Both modules are from scratch; KALIA is frozen.

## Pipeline

1. Real 32×32 test image → v2.1 captioner → caption C1.
2. KALIA text encoder → 512-d embedding of C1.
3. v1 two-object diffusion → generated image.
4. v2.1 captioner → caption C2.
5. Compare C1 and C2.

## Bars

| ID | prediction | threshold |
|---|---|---|
| **X-1** | cycle caption exact | ≥ 0.70 |
| **X-2** | attribute consistency (each) | ≥ 0.80 |
| **X-3** | no training; both modules from scratch | pass |
| **X-4** | sample grid | pass |

## Decision rule

- **X-1/X-2 pass:** evidence of shared compositional visual binding across
  seeing and drawing; release **v0.5.4-vision-cycle**.
- **Fail:** boundary result; log it. Do not tune after the result.
- This is a cross-modal discovery test, not a benchmark chase.

## Cost

≤ 1 GPU-h.
