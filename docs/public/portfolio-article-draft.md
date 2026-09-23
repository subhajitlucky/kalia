# Building KALIA: a 58M language model from scratch on free GPUs (DRAFT)

> Fill in final numbers at release. Structure: hook → why → how → evidence →
> failures → what is next → links.

## Hook

I trained a language model from scratch — random weights, my own data pipeline,
my own training code — on free-tier Kaggle GPUs with zero dollars of compute
spend. It writes short coherent stories. Every weight in it exists nowhere else.

## Why from scratch when fine-tuning exists

Fine-tuning is forking someone else's brain. I wanted a model where I could
account for every byte: the corpus mixture, the tokenizer, the architecture,
the optimizer, the exact training run. That decision costs capability — this is
a 58M model, not an assistant — and buys an auditable artifact.

## How it was built (the short version)

- **Data**: v0.1.2 trained on TinyStories + FineWeb-Edu (dedup), 2.46B tokens,
  tokenized to uint16 shards. The v0.2.0 corpus adds Cosmopedia and
  permissively licensed Python (60/20/15/5 mixture, 2.4B tokens).
- **Architecture**: decoder-only transformer, 10 layers, 512 dim, RMSNorm,
  SwiGLU, RoPE, QK-Norm, soft-capped logits, tied embeddings. Looping (weight
  sharing for extra effective depth) and GQA are under ablation.
- **Training loop**: fp16 + DDP across two T4s, resumable sessions with
  checkpoints on the HuggingFace Hub, validation loss and bits-per-byte logged.
- **Optimizer**: Muon (Newton–Schulz orthogonalization) on hidden matrices,
  AdamW for embeddings/head/norms. Muon+ (col-row normalization) won the
  micro-ablation but below the promotion threshold, so it stays an option.

## Evidence (numbers, not vibes)

Micro-ablations first (30M params, identical tokens, identical seed):

| Arm | Val loss @ step 700 |
|---|---|
| AdamW | 3.8041 |
| Muon | 3.5937 |
| Muon + QK-Norm + soft-cap | **3.5103** |

Muon+ vs plain Muon at matched tokens: 3.4941 vs 3.5091 (better on 7 of 7
checkpoints) — below the pre-set 0.02-nat promotion threshold, so the shipped
model uses plain Muon; the threshold exists to stop wishful promotions, and it
was enforced even against a promising result. A learning-rate sweep
(0.015/0.02/0.03/0.06) picked 0.02 as optimal; no further tuning has headroom.

Full-scale, equal-token comparison: the Muon-based model overtook the AdamW
baseline's *final* loss with ~23% fewer tokens, and at equal steps kept a
persistent ~0.15 nat advantage. Final numbers: 2.4366 held-out loss / 0.8184
bits-per-byte (deterministic, 819k tokens), and zero-shot benchmarks put it near
125M-class models on PIQA (61.4%) and ARC-Easy (45.8%) with 2× fewer parameters
and ~160× fewer training tokens. The validation curve plateaued for 1,000 steps,
then training was stopped at 73% of the schedule when the weekly quota ran out —
the plateau is documented, not hidden.

The entry–exit asymmetry ("Abhimanyu gap"): asked to model reversed text, the
same model needs **6.28 extra nats** (forward 3.18 vs reverse 9.46; random
would be ~10.8). It can enter fluent text but cannot exit. A pre-registered
experiment (chunk-preserving reversal training) tests whether that gap closes.

Looping economics: a weight-shared two-pass variant costs only ~1.35× the
wall-clock time of the single-pass model, not 2×, because the reused weights
stay cache-hot between passes.

## Failures worth reading

- A DDP resume race where both workers wrote the same checkpoint file → fixed
  with a single-downloader + atomic swap + barrier.
- A silent "successful" run where a failed subprocess was not checked → fixed
  by asserting exit codes.
- A no-license-metadata code dataset → replaced by permissive-license filtering.
- Ablations wasting ~1GB of checkpoint writes every 20 minutes on Kaggle's
  shared disk → interval checkpoints disabled for ablation runs.
- A monitoring error where I misreported a 44-minute run as ">5 hours" → fixed
  by reading kernel run-start times, and the correction is in the journal.
- A validation plateau that looked like saturation, then broke when the cosine
  decay arrived — a reminder to wait for the schedule before concluding.
- RL self-play (R-Zero style) rejected on evidence: it collapses at small scale
  (the smallest tested model peaked at iteration 1). Documented as a negative
  result instead of shipped as a buzzword.

## What is next

- **v0.2.0**: new data lineage (compliance-clean v2b corpus) plus only those
  changes that pass pre-registered promotion rules (architecture ablation and
  X16 reversal, decided in advance in `docs/preregistrations/`).
- Post-training: instruction tuning with a continual-learning recipe.
- Deployment: quantization + a local chat interface; the model already runs on
  a laptop CPU for generation.

## Links

- Code + full engineering journal: [GitHub]
- Weights + model card: [HuggingFace]
- Training notebooks + ablation results: [Kaggle]
