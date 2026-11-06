# v0.5.0-vision — KALIA-Vision v0

**Capability release — 2026-11-06.** KALIA's first image-generation module:
the frozen v0.3.1 language model reads a caption; a small conditional diffusion
model trained from scratch draws the requested 32×32 synthetic shape.

This is a capability module, not a new base model. The public base remains
v0.3.1.

## Result

5,000 held-out captions, attribute classifier trained from scratch:

| bar | metric | result |
|---|---|---:|
| W-1 | requested shape | **0.9472** |
| W-2 | requested color | **0.9684** |
| W-3 | requested position | **0.8524** |
| W-4 | no pretrained vision weights; KALIA frozen | pass |
| W-5 | sample grid | `grid.png` |

- 20,000 DDPM steps, batch 128, **40.0 minutes**, 3.75M U-Net params.
- Checkpoint sha `563ab937…`, 15.0MB.
- DDIM 100-step sampling.
- Real-test classifier sanity: shape 1.000, color 1.000, position 0.957.

## What is released

- `ckpt.pt` — the 3.75M-parameter conditional U-Net.
- `synthetic_vision.py` — the synthetic shapes dataset generator.
- `vision_diffusion.py` — DDPM/U-Net and DDIM sampler.
- `tools/kalia_vision_*_kernel.py` — data, training and eval kernels.
- `docs/results/kalia-vision-v0/` — metrics, sample grid, generated images,
  logs, pre-registration.

## Usage sketch

```python
from model import GPT, GPTConfig
from vision_diffusion import UNet, Diffusion
from synthetic_vision import ...
# frozen KALIA text encoder -> mean-pooled 512-d caption embedding
# conditional U-Net -> DDIM sample -> 32x32 RGB
```

## What this is not

- **Not a general text-to-image model.** It is a 32×32 synthetic-shapes toy
  domain: three shapes, four colors, three positions.
- **Not photorealism.** Some samples are blurry.
- **Not a new base model.** KALIA v0.3.1 is unchanged and frozen.
- **Not pretrained vision.** The U-Net is trained from scratch; no external
  images or vision weights.

## Next

**KALIA-Audio v0**: a from-scratch speech-command encoder feeding KALIA, then
a TTS module. The same pattern: a small owned modality encoder, the frozen
language core, and an honest narrow-domain evaluation.

## Record

- Result: `docs/results/kalia-vision-v0/README.md`
- Pre-registration: `docs/preregistrations/2026-11-06-kalia-vision-v0.md`
- Licence: repository `LICENSE`.
