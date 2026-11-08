# v0.5.3-vision-compose — KALIA-Vision v2.1

**Capability release — 2026-11-08.** The image captioner generalizes to
**unseen combinations** of seen shapes and colors, not just the training grid.

## Result

- 29 of 144 two-object combinations held out (20%, seed 2026).
- Train 39,853 images; val 4,015 seen; test 1,000 held-out + 1,000 seen.
- **Held-out exact: 0.9920**
- **Seen exact: 0.9970**
- **Generalization gap: 0.0050**
- Held-out attributes: left shape 1.000, left color 1.000, right shape 0.992,
  right color 1.000.
- Blind decoder: 0.000; shuffled image: 0.003.
- 2,328,832 trainable params; 8 epochs in **20.3 minutes**.
- Checkpoint sha `e16cb9cd…`, 9.3MB.

| bar | result |
|---|---|
| C-1 held-out exact ≥ 0.50 | **PASS (0.992)** |
| C-2 held-out attributes ≥ 0.80 | **PASS** |
| C-3 seen exact ≥ 0.90 | **PASS (0.997)** |
| C-4 gap reported | **PASS (0.005)** |
| C-5 from scratch; KALIA frozen | **PASS** |

## What changed

v2.0 was trained on all 144 combinations. v2.1 holds out 29 combinations,
trains on the rest, and tests the unseen pairings. The captioner still reaches
0.992 exact — evidence of compositional binding in this synthetic grid.

## What is released

- `ckpt.pt` — 2.33M encoder + LoRA state.
- `split.json` — the held-out combinations.
- `eval.json` — held-out/seen, attributes, blind/shuffled.
- `docs/results/kalia-vision-v2.1/`.
- Kernels: `kalia-vision-v2-split-train`, `kalia-vision-v2-split-eval`.

## Caveat

Synthetic 32×32 two-object scenes: three shapes, four colors, two positions.
This is compositional generalization on that grid, not open-world vision.

## Record

- Result: `docs/results/kalia-vision-v2.1/README.md`
- Pre-registration: `docs/preregistrations/2026-11-08-kalia-vision-v2.1-compose.md`
- Licence: repository `LICENSE`.
