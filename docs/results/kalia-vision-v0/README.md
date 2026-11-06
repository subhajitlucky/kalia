# KALIA-Vision v0 — text-to-image on synthetic shapes

Status: **clean pass of W-1…W-3.** KALIA v0.3.1 is the frozen text encoder; a
3.75M-parameter conditional DDPM draws 32×32 synthetic shapes. This is a
capability module, not a new base model.

## Result

5,000 held-out captions; attribute classifier trained from scratch on the real
synthetic images (real-test sanity: shape 1.000, color 1.000, position 0.957).

| bar | metric | result |
|---|---|---:|
| **W-1** | requested shape | **0.9472** |
| **W-2** | requested color | **0.9684** |
| **W-3** | requested position | **0.8524** |
| **W-4** | no pretrained vision weights; KALIA frozen | **pass** |
| **W-5** | sample grid | `grid.png` |

- Training: 20,000 DDPM steps, batch 128, 40.0 minutes, 3.75M U-Net params.
- Checkpoint sha `563ab937…`; 15.0MB.
- Text condition: mean-pooled KALIA caption embedding (512-d).
- Sampling: DDIM 100 steps.

## What it does

Given a caption such as “a red circle on the left” or “the square is blue and
on the right,” the diffusion module generates a 32×32 image with the requested
shape, color, and position. KALIA supplies the text embedding; the image module
is trained entirely from scratch on synthetic data.

## Caveat

This is a **synthetic toy domain**: 32×32, one shape per image, three shapes,
four colors, three positions. It is not a general text-to-image model and not
photorealism. Some samples are blurry. It is the first owned image-generation
module, not a claim of open-world vision.

## Artifacts

- `eval.json` — W-1…W-3 and real-test sanity.
- `grid.png` — 64 held-out samples.
- `generated_test.npy` — 5,000 generated test images.
- `ckpt.pt` — 3.75M-parameter U-Net.
- `train_manifest.json`, `kalia-vision-train.log`, `kalia-vision-eval.log`,
  `data.log`.
- Pre-registration: `docs/preregistrations/2026-11-06-kalia-vision-v0.md`
  (hash `0c04af74…`).
- Kernels: `subhajitlucky/kalia-vision-data`,
  `subhajitlucky/kalia-vision-train`,
  `subhajitlucky/kalia-vision-eval`.
