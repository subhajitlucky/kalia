# v0.5.1-vision-two — KALIA-Vision v1

**Capability update — 2026-11-06.** KALIA's image module now generates **two
objects in one 32×32 image**. The frozen v0.3.1 text encoder reads the caption;
a 3.75M-parameter conditional DDPM trained from scratch draws the scene.

This is a capability update, not a new base model.

## Result

5,000 held-out captions, from-scratch attribute classifier:

| bar | metric | result |
|---|---|---:|
| V1-1 | left shape | **0.9910** |
| V1-2 | left color | **0.9804** |
| V1-3 | right shape | **0.9932** |
| V1-4 | right color | **0.9756** |
| V1-5 | exact two-object scene | **0.9482** |
| V1-6 | no pretrained vision; KALIA frozen | pass |

- 25,000 DDPM steps, batch 128, **45.5 minutes**, 3.75M U-Net params.
- Checkpoint sha `ba2947e8…`, 15.0MB.
- DDIM 100-step sampling.
- Real-test classifier sanity: 0.9994 exact.

## What changed from v0.5.0-vision

v0 generated one shape. v1 generates **two objects**, each with its own shape
and color, on the left and right. Exact two-object scene accuracy is **0.948**.

## What is released

- `ckpt.pt` — the 3.75M-parameter conditional U-Net.
- `synthetic_vision_v1.py` — the two-object data generator.
- `vision_diffusion.py` — DDPM/U-Net and DDIM sampler.
- `tools/kalia_vision_v1_*_kernel.py` — data, training and eval kernels.
- `docs/results/kalia-vision-v1/` — metrics, grid, generated images, logs,
  pre-registration.

## Caveat

Synthetic 32×32 toy domain: three shapes, four colors, two fixed positions,
two objects. Not general text-to-image; not photorealism. Some samples are
blurry or merge objects.

## Next

**KALIA-Audio v0**: a from-scratch speech-command encoder feeding KALIA, then
TTS. Then image v2: more objects, 64×64, freer positions.

## Record

- Result: `docs/results/kalia-vision-v1/README.md`
- Pre-registration: `docs/preregistrations/2026-11-06-kalia-vision-v1.md`
- Licence: repository `LICENSE`.
