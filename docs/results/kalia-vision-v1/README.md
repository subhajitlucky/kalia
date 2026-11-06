# KALIA-Vision v1 — two-object scenes

Status: **clean pass of V1-1…V1-5.** KALIA v0.3.1 is the frozen text encoder; a
3.75M-parameter conditional DDPM generates 32×32 images with two objects.
Capability update, not a base-model change.

## Result

5,000 held-out captions; from-scratch attribute classifier (real-test sanity:
left shape 0.9996, left color 1.000, right shape 0.9998, right color 1.000,
exact 0.9994).

| bar | metric | result |
|---|---|---:|
| **V1-1** | left shape | **0.9910** |
| **V1-2** | left color | **0.9804** |
| **V1-3** | right shape | **0.9932** |
| **V1-4** | right color | **0.9756** |
| **V1-5** | exact two-object scene | **0.9482** |
| **V1-6** | from scratch; KALIA frozen | **pass** |

- 25,000 DDPM steps, batch 128, **45.5 minutes**, 3.75M U-Net params.
- Checkpoint sha `ba2947e8…`; 15.0MB.
- Text condition: mean-pooled KALIA caption embedding (512-d).
- DDIM 100-step sampling.

## What changed from v0

v0 generated one shape. v1 generates **two objects in one image**, each with
its own shape and color, on the left and right. Exact two-object scene accuracy
is **0.948**, not just per-attribute accuracy.

## Caveat

Still a synthetic toy domain: 32×32, three shapes, four colors, two fixed
positions, two objects. It is not a general text-to-image model and not
photorealism. Some samples are blurry or merge the two objects.

## Artifacts

- `eval.json` — V1-1…V1-5 and real-test sanity.
- `grid.png` — 64 held-out two-object samples.
- `generated_test.npy` — 5,000 generated test images.
- `ckpt.pt` — 3.75M U-Net.
- `train_manifest.json`, `kalia-vision-v1-train.log`,
  `kalia-vision-v1-eval.log`.
- Pre-registration: `docs/preregistrations/2026-11-06-kalia-vision-v1.md`
  (hash `c87bd710…`).
- Kernels: `subhajitlucky/kalia-vision-v1-data`,
  `kalia-vision-v1-train`, `kalia-vision-v1-eval`.
