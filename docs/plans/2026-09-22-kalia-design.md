# KALIA — A From-Scratch ~58M-Parameter Language Model

Status: approved
Date: 2026-09-22

## Summary

KALIA is a decoder-only transformer language model (~58M parameters) trained **from
scratch** — random initialization, no fine-tuning, no pretrained base — on ~2.5B tokens
of English text using free Kaggle GPU quota (2×T4, ~30 GPU-hours/week). The project spans
architecture configuration, corpus mixture, tokenizer pipeline, training run, weights,
name/brand, and a serving layer (API + chat UI) built on top of the trained model.

## Context & Constraints

- No local GPU: training happens exclusively on free-tier cloud compute; the local machine
  is used only for code editing, CPU smoke tests, and inference of the finished model.
- Budget: $0. All compute comes from free tiers.
- Compute: Kaggle free tier — 2×T4 (32GB VRAM total), ~30 GPU-hours/week (weekly reset),
  9–12h maximum GPU session length, 20GB persistent storage, unlimited CPU-notebook quota.
- Name "KALIA" verified available on HuggingFace (no user, org, or model collision with an
  exact match as of 2026-09-22). Planned org: `kalia`.
- Ownership stance: training from scratch produces weights that exist nowhere else. The
  nanoGPT-style skeleton and the 2017 transformer paper are treated as public reference
  material; the KALIA codebase is written fresh for this project.

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Platform | Kaggle | Documented 30 GPU-h/week quota, 32GB total VRAM, background "commit" runs, 20GB persistent storage, no credit card |
| Training mode | From scratch | Genuine ownership; full learning value |
| Model size | ~58M params ("KALIA-M") | Best capability/compute tradeoff at this budget |
| Data | TinyStories (~500M tokens) + FineWeb-Edu sample (~2B tokens) | TinyStories drives fluency at tiny scale; FineWeb-Edu adds world knowledge |
| Tokenizer | GPT-2 BPE, 50,257 vocab (`tiktoken`) | Off-the-shelf; avoids a separate tokenizer-training stage |
| Architecture | Pre-norm decoder-only transformer: RMSNorm, SwiGLU, RoPE, weight tying, no biases | Modern, proven, modest code complexity |
| Precision | fp16 with AMP | T4 supports fp16 (not bf16) |
| Optimization | AdamW (β 0.9/0.95, wd 0.1), grad clip 1.0, cosine schedule with warmup | Standard, stable |
| Multi-session | Checkpoint to private HF Hub repo every ~30 min and at session end; auto-resume | Kaggle sessions are finite; no work is lost |
| Tokenization | One-off Kaggle **CPU** notebook (CPU quota is unlimited) → uint16 `.bin` shards → private Kaggle Dataset | Heavy text processing never touches the local machine |

## Architecture Spec (KALIA-M)

| Field | Value |
|---|---|
| Layers (n_layer) | 10 |
| Heads (n_head) | 8 |
| Embedding dim (n_embd) | 512 |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Context length | 1,024 tokens |
| FFN | SwiGLU (hidden ≈ 4/3 × 4 × n_embd) |
| Normalization | RMSNorm, pre-norm |
| Positional encoding | RoPE (fallback: learned absolute embeddings) |
| Biases | None |
| Dropout | 0.0 (single-epoch-scale training) |
| Weight tying | Token embedding ↔ LM head |
| Init | Normal, std 0.02 |
| Total params | ≈58M |

## Data Pipeline

1. **Sources**
   - `roneneldan/TinyStories` (English short stories, ~500M GPT-2 tokens)
   - `HuggingFaceFW/fineweb-edu` sample slice (target ~2B GPT-2 tokens)
2. **Tokenize once** in `prepare.py`, run on a Kaggle CPU notebook: stream → tokenize with
   `tiktoken` → append to `train.bin` / `val.bin` as `uint16` (vocab fits in 16 bits).
3. **Validation split**: fixed held-out slices — 5M tokens from TinyStories, 5M from
   FineWeb-Edu, concatenated as `val.bin`.
4. **Upload** the `.bin` files as a private Kaggle Dataset (`kalia-tokens`) attached to the
   training notebook. Target size: ~5GB for 2.5B tokens.
5. **Sampling during training**: random contiguous windows of context length from `train.bin`.
   Optional later refinement: weighted domain sampling (e.g., 30% TinyStories / 70%
   FineWeb-Edu per batch).

## Training Loop

| Setting | Value |
|---|---|
| Tokens per step (effective) | ~0.5M (micro-batch × grad-accum × seq len × devices) |
| Total tokens | ~2.5B |
| Total steps | ~5,000 |
| Optimizer | AdamW, lr 6e-4, β (0.9, 0.95), weight decay 0.1 |
| Schedule | Linear warmup (~500 steps) → cosine decay to 10% of peak |
| Grad clip | 1.0 |
| Precision | fp16 + GradScaler |
| Parallelism | DDP across 2×T4 via `torchrun` |
| Estimated wall time | ~25–45 GPU-hours → ~1–2 weeks of weekly quota, 30–45h |
| Checkpoints | Every ~30 min and at session end: model, optimizer state, `last_step`, config |
| Logging | stdout loss + tokens/sec, CSV log, sample generations every ~500 steps |

### Session runbook (per Kaggle GPU session)

1. Attach `kalia-tokens` dataset; clone/pull the `kalia` repo.
2. Install deps.
3. `python train.py --resume` → pulls latest checkpoint from HF Hub (first run starts from
   random init).
4. Train until ~30 min remain in the session.
5. Push checkpoint + CSV log to HF Hub; verify upload; stop.
6. Next session repeats from step 1 — zero wasted quota.

## Evaluation

- **Primary**: validation loss on `val.bin` (expect ≈3.2–3.6 at this scale).
- **Qualitative**: generate continuations for a fixed prompt set every ~500 steps
  ("Once upon a time...", facts, etc.) and compare against previous checkpoints.
- **Sanity**: no NaNs, stable loss curve, checkpoint resume reproduces step count.

## Repository Layout

```
kalia/
  README.md                     # what KALIA is, glossary, run instructions
  LICENSE                       # MIT (code); model card states weights license
  requirements.txt
  configs/
    kalia-m.yaml                # architecture + training config
  model.py                      # RMSNorm, SwiGLU, RoPE, attention, GPT
  prepare.py                    # tokenize sources -> train.bin/val.bin
  train.py                      # DDP, AMP, resume, checkpoint/push loop
  sample.py                     # generate from a checkpoint
  notebooks/
    kalia-prep.ipynb            # Kaggle CPU: tokenize + upload dataset
    kalia-train.ipynb           # Kaggle GPU: train/resume/push
  docs/
    plans/2026-09-22-kalia-design.md
```

## Roadmap

- **v0 — pretrain (this document)**: KALIA-M pretrained text generator, runnable locally on
  CPU after export.
- **v1 — SFT**: small supervised instruction-tuning stage so KALIA can follow prompts.
- **v2 — local deployment**: quantized/GGUF export, CLI + FastAPI inference service, chat UI
  (the project's application layer), optional Ollama packaging.
- **v3 — scale-up (stretch)**: repeat the same pipeline at 125M+ if quota allows.

## Non-Goals

- Instruction following or chat behavior at v0.
- Custom tokenizer training, multilingual data, or swapping architectures.
- Competing with modern large models — KALIA-M targets coherent, fluent English at a tiny
  scale, and complete end-to-end ownership.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| RoPE implementation bugs | Unit-test shapes/invariants; fallback to learned positional embeddings (one-line config switch) |
| Kaggle session interrupts mid-run | Frequent HF checkpoints; resume logic; validation of uploaded checkpoint |
| Dataset tokenization fails on Kaggle limits | Shard tokenization; stream with `datasets` in streaming mode; CPU quota is unlimited |
| Loss plateaus / overfit | Track val loss; adjust mixture and schedule; reduce epochs |
| HF Hub free limits | Private repos are free; checkpoint retention policy (keep latest 2) |

## Glossary

See `README.md` for the full glossary (weights, attention, RoPE, loss, AdamW, checkpoints,
quantization, etc.).
