# KALIA

A from-scratch ~58M-parameter language model — trained on free Kaggle GPUs with zero
fine-tuning, zero pretrained weights, and zero budget.

KALIA is a decoder-only transformer trained from **random initialization** on ~2.5B tokens
of English text (TinyStories + FineWeb-Edu). Every weight in KALIA is learned by this
project's own training run; nothing is borrowed from another model.

## Spec

| Field | Value |
|---|---|
| Parameters | 57,854,976 |
| Layers | 10 |
| Heads | 8 |
| Embedding dim | 512 |
| Context length | 1,024 tokens |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Architecture | Pre-norm decoder-only: RMSNorm, SwiGLU, RoPE, weight tying, no biases |
| Training data | TinyStories (~500M tokens) + FineWeb-Edu (~2B tokens) |
| Training hardware | Kaggle free tier: 2× NVIDIA T4, ~30 GPU-hours/week |
| Precision | fp16 with gradient scaling |
| Optimizer | AdamW, cosine schedule with warmup |
| Total steps | ~4,770 (~0.5M tokens per step) |

## Repository layout

```
kalia/
  model.py          # RMSNorm, SwiGLU, RoPE, attention, GPT
  data.py           # memory-mapped token dataset + batching
  prepare.py        # tokenize text sources into uint16 .bin shards
  train.py          # resumable, time-budgeted, DDP-capable training loop
  sample.py         # generate text from a checkpoint
  configs/          # kalia-m.yaml (the real model), smoke.yaml (tiny CPU test)
  notebooks/        # Kaggle: kalia-prep.ipynb (CPU), kalia-train.ipynb (GPU T4x2)
  tests/            # pytest suite (20 tests)
  docs/plans/       # design doc + implementation plan
```

## Glossary (for developers new to ML)

| Term | Plain meaning | Developer analogy |
|---|---|---|
| Weights / parameters | The learned numbers — all of the model's knowledge | Config object written by training instead of by hand |
| Architecture | Blueprint for how the numbers are wired together | Schema + code structure |
| Transformer / GPT | The 2017 public design; a next-token predictor | A public spec (like HTTP) that everyone implements |
| Layer / heads | Processing stages / parallel attention lanes | Middleware chain / worker pool |
| Attention | Every token searches earlier tokens and mixes in relevant info | A JOIN across token positions |
| Embedding | Turns a token ID into a vector of numbers | `Map<tokenId, number[]>` that gets learned |
| Tokenizer / BPE / vocab | Text → chunk IDs; BPE learns common chunks | A compiler: string → int[] |
| Context window | How many tokens the model can see at once | Max payload size |
| RMSNorm | Keeps numbers healthy between stages | Input-validation middleware |
| SwiGLU | Gated feed-forward "thinking" block | Feature transform with a data-controlled volume knob |
| RoPE | Encodes token order by rotating vectors | Continuous timestamps for positions |
| Random init | Starts knowing nothing | Empty database |
| Training | Show text, predict next token, nudge dials | The learning loop |
| Loss | "How wrong" score; random guessing ≈ 10.8 | Error rate |
| Backprop | Computes blame for every dial | `git blame` with auto-fix instructions |
| AdamW | The dial-nudging rulebook | Update policy |
| Learning rate / warmup / cosine | Nudge size; start gentle, cool down | Ramp-up + graceful shutdown |
| Batch / gradient accumulation / step | Examples per update / one update | Batched requests / one deploy |
| fp16 AMP | Half-size numbers → ~2× faster | Compressed transport |
| DDP | Two GPUs learn together and sync | Two instances behind a load balancer |
| Checkpoint | Saved state of all dials + optimizer | Database dump |
| Overfit | Memorizing instead of understanding | Hardcoding test answers |
| Fine-tune | Adapt a pretrained model (KALIA does NOT do this) | Fork + patch |
| SFT | Stage 2: teach instruction following | API docs vs conversational onboarding |
| Quantization / GGUF / Ollama | Shrink weights to run locally | Minifying assets / Docker for models |
| Kaggle | Free GPU notebooks with weekly quotas | CI runners with GPUs |
| HF Hub | Hosting for models and checkpoints | npm registry for models |

## Local quickstart

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v          # full suite, ~8 seconds on CPU
```

Smoke-train the tiny test model on CPU (proves the whole pipeline works):

```bash
python -c "
import numpy as np, pathlib
p = pathlib.Path('data'); p.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
np.array(rng.integers(0, 256, 8192), dtype=np.uint16).tofile(p/'train.bin')
np.array(rng.integers(0, 256, 2048), dtype=np.uint16).tofile(p/'val.bin')
"
python train.py --config configs/smoke.yaml --data-dir data --out-dir out
```

Generate from any checkpoint:

```bash
python sample.py --ckpt out/ckpt.pt --prompt "Once upon a time"
```

## Kaggle runbook (training KALIA for real)

**One-time setup**

1. Create a private dataset `kalia-tokens` — run `notebooks/kalia-prep.ipynb` on Kaggle
   (CPU, Internet on), then `Save Version` → `Output` → `Create Dataset` (private).
2. Create a HuggingFace **write** token (huggingface.co/settings/tokens) and a private
   model repo, e.g. `yourname/kalia-m`.
3. Kaggle: verify your phone (Settings → Phone Verification) to unlock GPU quota.
4. Kaggle: add your `HF_TOKEN` under Add-ons → Secrets.
5. Push this repository to GitHub (private is fine — the notebooks clone it with a
   `GH_TOKEN` Kaggle secret).
6. Kaggle secrets: add `HF_TOKEN` (HuggingFace write token) and `GH_TOKEN` (GitHub token
   with read-only access to this repo). Toggle both **on** for each notebook.
7. Edit `HUB_REPO` in `notebooks/kalia-train.ipynb` to your HuggingFace repo id
   (e.g. `kalia-lm/kalia-m`).

**Every training session**

1. Open `notebooks/kalia-train.ipynb`, set Accelerator to **GPU T4 x2**, attach the
   `kalia-tokens` dataset, Internet on.
2. Run all cells. The first run starts from random weights; every later run resumes from
   the latest checkpoint — no progress is ever lost.

**Multi-session mechanics:** `train.py` saves a checkpoint (weights + optimizer + step
counter) to your HF repo every 30 minutes and at the end of a session. The next session
pulls the latest checkpoint and continues. Kaggle gives ~30 GPU-hours per week, so KALIA
needs roughly 1–2 weeks of weekly quota to finish.

**Monitoring:** watch `logs/train_log.csv` in your HF repo. Loss starts near 10.8 (random
guessing) and should fall toward ~3.2–3.6. Sample generations print every 500 steps.

## Version history

- **v0.1.0** — baseline: AdamW, RoPE, SwiGLU, RMSNorm, fp16, T4×2.
- **v0.1.1** — Muon optimizer for hidden weight matrices (AdamW keeps
  embeddings/head/norms). Micro-ablation at equal tokens (30M params, 50M
  tokens): **3.5937 vs 3.8041** val loss, a **−0.21** win.
- **v0.1.2** — QK-Norm + logit soft-capping (τ = 30) on top of Muon.
  Same ablation: **3.5103** val loss, **−0.29** vs baseline. Current best recipe.
  First coherent generations at step 1738 (loss 3.22): see
  `docs/samples/2026-09-23-first-words.md`.

## Roadmap

- **v0 — pretrain** (this repository): a from-scratch text generator.
- **v1 — SFT**: a small instruction-tuning stage so KALIA can follow prompts.
- **v2 — local deployment**: quantization/GGUF export, FastAPI inference service, chat UI.
- **v3 — scale-up**: reuse the same pipeline at 125M+ params.

## License

Code: MIT (see `LICENSE`). Model weights: released at the maintainers' discretion.
