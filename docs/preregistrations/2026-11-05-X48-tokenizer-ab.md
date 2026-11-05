# Pre-registration — X48: tokenizer A/B (GPT-2 BPE vs K-Tok 32k)

Registered: 2026-11-05, before the X48 GPU run.

## Question

Does KALIA's own 32k tokenizer improve the standard 9-task suite at matched
compute, or is GPT-2 BPE still better? This is the first model-level test of
K-Tok v0.

## Design

- **Screen scale:** 30M parameters (6 layers, 384 embd, 6 heads, context 512).
- **Arms:** GPT-2 BPE (`vocab_size=50257`) vs K-Tok 32k (`vocab_size=32000`).
- **Same body:** both arms use the identical transformer body. The K-Tok arm
  has ~7M fewer tied-embedding parameters at this width; that is part of the
  tokenizer's efficiency claim, not hidden.
- **Same text:** 312M GPT-2 tokens decoded to text; GPT-2 arm uses the original
  token IDs, K-Tok arm re-tokenizes the same text. Same optimizer steps,
  batch size and context.
- **Schedule:** 4,770 steps, micro 16 × grad accum 8 × 512 ≈ 312M tokens;
  AdamW, LR 6e-4, warmup 40, cosine, 2 seeds (1338/1339).
- **Evaluation:** the standard 9-task suite (ARC-Easy, ARC-Challenge, BoolQ,
  HellaSwag, LAMBADA, OpenBookQA, PIQA, SciQ, WinoGrande) at 2,000 samples
  where available, plus bits-per-byte on the held-out text.

## Bars

| ID | prediction | threshold |
|---|---|---|
| **H-1** | K-Tok beats GPT-2 on the 9-task average | ≥ +1.0 pp, both seeds |
| **H-2** | no task regresses | no task beyond 2σ |
| **H-3** | tokenizer efficiency | K-Tok bpB ≤ GPT-2 bpB |
| **H-4** | lossless | round-trip true |
| **H-5** | parameter budget reported | total and tied-embedding counts |

The screen is not a release gate. A +1.0 pp average at 30M is a signal to train
the 58M model with the winner; a null means GPT-2 BPE stays and K-Tok is
parked.

## Decision rule

- **H-1 and H-2 pass:** train the 58M base with K-Tok 32k next; pre-register
  the 58M run separately.
- **H-1 fails:** keep GPT-2 BPE; do not spend a 58M run on the tokenizer.
- **H-3/H-4 fail:** tokenizer bug; stop and fix before any model claim.
- No tuning of the tokenizer or schedule after the result.

## Cost

Four 30M arms × ~5 GPU-h + 9-task eval: **≤ 28 GPU-h**. Fits the 30-hour
window. Data preparation is CPU-only.

## Artifacts

- `docs/results/x48-tokenizer-ab/` — data manifest, configs, logs, per-arm
  results, benchmark rows.
- `tools/kalia_x48_data_kernel.py`, `tools/kalia_x48_train_kernel.py`.
- This file.
