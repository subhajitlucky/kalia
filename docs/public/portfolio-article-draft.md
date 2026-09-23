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

- **Data**: TinyStories + FineWeb-Edu (dedup) + Cosmopedia + permissively
  licensed Python, ~2.4B tokens, tokenized to uint16 shards.
- **Architecture**: decoder-only transformer, 10 layers, 512 dim, RMSNorm,
  SwiGLU, RoPE, QK-Norm, soft-capped logits, tied embeddings.
- **Training loop**: fp16 + DDP across two T4s, resumable sessions with
  checkpoints on the HuggingFace Hub, validation loss logged per run.
- **Optimizer**: Muon with post-polar col-row normalization (Muon+) on hidden
  matrices, AdamW for embeddings/head/norms.

## Evidence (numbers, not vibes)

Micro-ablations first (30M params, identical tokens, identical seed):

| Arm | Val loss @ step 700 |
|---|---|
| AdamW | 3.8041 |
| Muon | 3.5937 |
| Muon + QK-Norm + soft-cap | **3.5103** |

Full-scale, equal-token comparison: the Muon-based model overtook the AdamW
baseline's *final* loss with ~23% fewer tokens, and at equal steps kept a
persistent ~0.15 nat advantage.

## Failures worth reading

- A DDP resume race where both workers wrote the same checkpoint file → fixed
  with a single-downloader + atomic swap + barrier.
- A silent "successful" run where a failed subprocess was not checked → fixed
  by asserting exit codes.
- A no-license-metadata code dataset → replaced by permissive-license filtering.
- RL self-play (R-Zero style) rejected on evidence: it collapses at small scale
  (the smallest tested model peaked at iteration 1). Documented as a negative
  result instead of shipped as a buzzword.

## What is next

- v0.2.0: looped/recurrent depth, richer data, WSD schedule, EMA readout.
- Post-training: instruction tuning with a continual-learning recipe.
- Deployment: quantization + a local chat interface; the model already runs on
  a laptop CPU for generation.

## Links

- Code + full engineering journal: [GitHub]
- Weights + model card: [HuggingFace]
- Training notebooks + ablation results: [Kaggle]
