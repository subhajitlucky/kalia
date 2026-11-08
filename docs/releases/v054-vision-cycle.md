# v0.5.4-vision-cycle — KALIA sees, draws, and reads back

**Capability release — 2026-11-08.** The see→draw→see loop closes on unseen
shape/color combinations. The earlier null was an inference-embedding mismatch,
not a compositional failure.

## Result

A/B with the same v1 diffusion, v2.1 captioner, real images and held-out split;
only the caption-embedding function changes.

| condition | split | loop exact | loop attributes | source readback | classifier exact |
|---|---|---:|---|---:|---:|
| **bidirectional** (v1 training) | held-out | **0.930** | 0.967–0.993 | **0.930** | **0.936** |
| bidirectional | seen | 0.949 | 0.978–0.997 | 0.955 | 0.954 |
| causal (old cycle) | held-out | **0.000** | 0.240–0.538 | 0.004 | 0.000 |
| causal | seen | 0.042 | 0.280–0.502 | 0.062 | 0.048 |

- Real caption exact: held-out **0.992**, seen **0.997** (captioner frozen).
- Cosine(bidirectional, causal): **0.692** mean, 0.018 std.
- 1,000 held-out + 1,000 seen for the corrected condition; 500 + 500 for the
  causal control. No training; KALIA, the captioner and the diffusion are
  frozen.
- E-1…E-7 pass; the causal control is materially lower (0.000 vs 0.930).

## What changed

The v1 diffusion was trained with a **bidirectional masked mean** over the
caption tokens. The earlier cycle kernels fed it a **causal mean**. The two
embeddings have cosine ~0.69, and the causal one is far outside the diffusion's
training distribution: the independent classifier cannot recognize a single
causal-conditioned sample. The previous 0.032 and 0.005 cycle results were
measuring that mismatch.

`vision_conditioning.py` now provides the canonical training-time embedding
(`mode="bidir"`) and keeps `mode="causal"` as the documented control.

## What is released

- `vision_conditioning.py` — corrected caption-conditioning module.
- `tools/kalia_x55_embedding_kernel.py` — the A/B evaluation kernel.
- `docs/results/x55-embedding-fix/` — metrics, log, generated images, samples.
- Correction notes on the earlier cycle and X53 results.

## Caveat

Synthetic 32×32 two-object scenes: three shapes, four colors, two positions,
29 of 144 combinations held out. This is compositional binding on that grid,
not open-world vision. The loop exact is 0.930, not 1.000; the remaining gap is
generation quality, not a compositional failure.

## Record

- Result: `docs/results/x55-embedding-fix/README.md`
- Pre-registration: `docs/preregistrations/2026-11-08-X55-embedding-fix.md`
  (hash `e5261fcf…`).
- Public base remains **v0.3.1**; no Hugging Face base release, because the
  standard-benchmark +10% gate is not met.
- Licence: repository `LICENSE`.
