# KALIA-Vision v2 — image encoder + captioning

Status: **clean pass of V2-1…V2-5.** KALIA can now **see**: a from-scratch CNN
encoder feeds four continuous image tokens into frozen v0.3.1, and a LoRA
caption decoder describes the scene.

## Result

- Held-out exact caption: **1.000** (1,000 images).
- Blind decoder (no image): **0.006**.
- Shuffled-image control: **0.002**; image-use drop **0.998**.
- Left/right shape and color accuracy: **1.000** each.
- 2,328,832 trainable parameters; 8 epochs in **24.4 minutes**.
- Checkpoint sha `7e70bce3…`, 9.3MB.

| bar | result |
|---|---|
| **V2-1** exact caption ≥ 0.80 | **PASS (1.000)** |
| **V2-2** each attribute ≥ 0.90 | **PASS (1.000)** |
| **V2-3** no pretrained vision; KALIA frozen | **PASS** |
| **V2-4** shuffled-prefix drop ≥ 0.20 | **PASS (0.998)** |
| **V2-5** artifacts | **PASS** |

## What it does

Given a 32×32 two-object scene, KALIA produces the exact caption:
“a red circle on the left and a blue square on the right,” including both
objects’ shapes and colors. The image prefix is causally attended by the text
decoder; the blind and shuffled controls show the image is genuinely used.

## Caveat

The synthetic training set contains **all 144 two-object combinations**, so
this is a strong *seen-combination* captioner, not yet evidence of
compositional generalization. The next experiment holds out combinations and
tests whether the model composes unseen shape/color/position pairs.

## Artifacts

- `ckpt.pt` — 2.33M trainable encoder + LoRA state.
- `eval.json` — exact, blind, shuffled, attributes, samples.
- `train_manifest.json`, `kalia-vision-v2-train.log`,
  `kalia-vision-v2-eval.log`.
- Pre-registration: `docs/preregistrations/2026-11-08-kalia-vision-v2.md`
  (hash `067c63b1…`).
- Kernels: `subhajitlucky/kalia-vision-v2-train`,
  `kalia-vision-v2-eval`.
