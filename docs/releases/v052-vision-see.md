# v0.5.2-vision-see — KALIA-Vision v2

**Capability release — 2026-11-08.** KALIA now has a **“see”** module: a
from-scratch CNN image encoder feeds continuous image tokens into frozen
v0.3.1, and a LoRA caption decoder describes the scene. This complements the
diffusion module that already **draws**.

## Result

- Held-out exact caption: **1.000** (1,000 images).
- Blind decoder: **0.006**.
- Shuffled-image control: **0.002**; image-use drop **0.998**.
- Left/right shape and color: **1.000** each.
- 2,328,832 trainable parameters; 8 epochs in **24.4 minutes**.
- Checkpoint sha `7e70bce3…`, 9.3MB.

| bar | result |
|---|---|
| V2-1 exact caption ≥ 0.80 | **PASS (1.000)** |
| V2-2 each attribute ≥ 0.90 | **PASS (1.000)** |
| V2-3 no pretrained vision; KALIA frozen | **PASS** |
| V2-4 shuffled-prefix drop ≥ 0.20 | **PASS (0.998)** |
| V2-5 artifacts | **PASS** |

## What changed

v0/v1 could generate images from text. v2 can **read an image and describe
it**. The image prefix is attended by the text decoder; blind and shuffled
controls confirm the image is genuinely used.

## What is released

- `ckpt.pt` — 2.33M trainable encoder + LoRA state.
- `vision_captioner.py` — encoder, prefix injection, caption training/eval.
- `tools/kalia_vision_v2_*_kernel.py` — training and eval kernels.
- `docs/results/kalia-vision-v2/` — metrics, samples, logs, pre-registration.

## Caveat

Synthetic 32×32 two-object scenes, all 144 combinations present in training.
This is a seen-combination captioner, not yet evidence of compositional
generalization. The next experiment holds out combinations and tests unseen
shape/color/position pairs.

## Record

- Result: `docs/results/kalia-vision-v2/README.md`
- Pre-registration: `docs/preregistrations/2026-11-08-kalia-vision-v2.md`
- Licence: repository `LICENSE`.
