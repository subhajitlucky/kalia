# KALIA-Vision cycle — see → draw → see

Status: **null.** The captioner and diffusion modules each work on their own
distribution, but the generated image does not survive the round trip.

## Result

1,000 real test images; caption C1 → v1 diffusion image → caption C2.

| metric | result |
|---|---:|
| Cycle exact (C1 == C2) | **0.032** |
| Left shape | 0.476 |
| Left color | 0.472 |
| Right shape | 0.404 |
| Right color | 0.278 |
| X-1 exact ≥ 0.70 | **FAIL** |
| X-2 attrs ≥ 0.80 | **FAIL** |
| X-3 no training | pass |
| X-4 grid | pass |

## Interpretation

- The v1 classifier recognized generated images at **0.948** exact scene; the
  generated images are not random.
- The v2.1 captioner, trained on **real** synthetic images, does not read the
  generated distribution reliably. The loop fails at the captioner's input
  distribution, not at generation.
- This is a cross-modal **distribution gap**: seeing and drawing do not yet
  share a stable representation.

## Decision

No release. The next experiment is a **cycle-consistency fine-tune**: train the
captioner on diffusion-generated images with their known source captions, then
re-test the held-out cycle. If the loop closes, that is a real advance; if not,
the gap is a boundary at this scale.

## Artifacts

- `eval.json`, `cycle.log`.
- Pre-registration: `docs/preregistrations/2026-11-08-kalia-vision-cycle.md`
  (hash `5c0bc371…`).
- Kernel: `subhajitlucky/kalia-vision-cycle`.
