# KALIA-Vision v2.1 — compositional split

Status: **clean pass of C-1…C-5.** The captioner generalizes to unseen
combinations of seen shapes and colors, rather than memorizing the 144-combination
grid.

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
| **C-1** held-out exact ≥ 0.50 | **PASS (0.992)** |
| **C-2** held-out attributes ≥ 0.80 | **PASS (0.992–1.000)** |
| **C-3** seen exact ≥ 0.90 | **PASS (0.997)** |
| **C-4** gap reported | **PASS (0.005)** |
| **C-5** from scratch; KALIA frozen | **PASS** |

## What this means

The v2 captioner saw all 144 combinations; v2.1 held 29 out. On those unseen
pairings it still captions almost perfectly. That is evidence that the small
image encoder + KALIA decoder **binds shape and color compositionally** in this
synthetic grid, not just a lookup table of training combinations.

This is a capability-level discovery, not a general-vision claim: the grid is
32×32, two objects, three shapes, four colors, two positions.

## Artifacts

- `ckpt.pt` — 2.33M encoder + LoRA state.
- `eval.json` — held-out/seen, attributes, blind/shuffled.
- `split.json` — the 29 held-out combinations.
- `train_manifest.json`, train/eval logs.
- Pre-registration: `docs/preregistrations/2026-11-08-kalia-vision-v2.1-compose.md`
  (hash `5e5e4975…`).
- Kernels: `subhajitlucky/kalia-vision-v2-split-train`,
  `kalia-vision-v2-split-eval`.
