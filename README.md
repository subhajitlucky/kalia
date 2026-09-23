# KALIA

> अथ शब्दानुशासनम्
>
> *"Now begins the discipline of words."* — the opening invocation of Pāṇini's
> Aṣṭādhyāyī, the oldest surviving systematic textbook of language.

A from-scratch ~58M-parameter language model — trained on free Kaggle GPUs with zero
fine-tuning, zero pretrained weights, and zero budget.

Model weights: [kalia-lm/kalia-v012](https://huggingface.co/kalia-lm/kalia-v012).

KALIA is a decoder-only transformer trained from **random initialization** on a
2.46B-token English corpus; 1.82B tokens were seen before the weekly quota stopped
training at step 3,478 of 4,770. Every weight in KALIA is learned by this project's
own training run; nothing is borrowed from another model. The whole engineering
record — decisions, incidents, pre-registered experiments — ships with the code.

## Spec

| Field | Value |
|---|---|
| Parameters | 57,856,256 |
| Layers | 10 |
| Heads | 8 |
| Embedding dim | 512 |
| Context length | 1,024 tokens |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Architecture | Pre-norm decoder-only: RMSNorm, SwiGLU, RoPE, QK-Norm, logit soft-cap (τ=30), weight tying, no biases |
| Training data | v0.1.x: TinyStories (~500M tokens) + FineWeb-Edu (~2B tokens) |
| Training hardware | Kaggle free tier: 2× NVIDIA T4, ~30 GPU-hours/week |
| Precision | fp16 with gradient scaling, DDP across 2×T4 |
| Optimizer | Muon (Newton–Schulz) on hidden matrices, AdamW for embeddings/head/norms (Muon+ validated at micro scale, below promotion threshold) |
| Schedule | Cosine with warmup; 4,770 steps (~0.5M tokens per step) |

## Measured results

Micro-ablations (30M params, equal tokens, equal seed, step-700 validation loss):

| Arm | Val loss |
|---|---|
| AdamW | 3.8041 |
| Muon | 3.5937 |
| Muon + QK-Norm + soft-cap | **3.5103** |

- Muon+ vs plain Muon at matched tokens: **3.4941 vs 3.5091** (better on 7/7 checkpoints) — below the pre-set 0.02-nat promotion threshold, so v0.1.2 ships **plain Muon**.
- LR sweep picked 0.02 as optimal (0.015: 3.4943 · 0.02: 3.4941 · 0.03: 3.5027 · 0.06: 3.5380).
- Full-scale: the Muon model overtook the AdamW baseline's *final* loss with **~23%
  fewer tokens** (3.2214 @ step 1730 vs 3.2702 @ step 2250).
- At step 3,478: held-out loss 2.4366 (deterministic, 819k tokens), **0.8184 bits-per-byte**.
- Zero-shot benchmarks (lm-eval, 0-shot, 500 samples): **PIQA 61.4%** · ARC-Easy 45.8% ·
  HellaSwag 36.8% (acc_norm) · WinoGrande 50.2% · LAMBADA 23.0% acc / ppl 194 — a
  storyteller's profile (near-125M-class on PIQA/ARC-Easy despite 2× fewer params).
- Entry–exit asymmetry ("Abhimanyu gap"): **6.06 nats** (forward 3.23 vs reversed-text
  9.29; random ≈ 10.8) — the model can enter text but not exit it. A pre-registered
  experiment (chunk-preserving reversal training, X16) tests whether that closes.

v0.1.2 is still training; final numbers are filled in at release. Full evaluation
reports live in `docs/eval/`.

## Repository layout

```
kalia/
  model.py              # RMSNorm, SwiGLU, RoPE, attention, QK-Norm, GPT
  data.py               # memory-mapped token dataset, batching, reversal transform
  prepare.py            # tokenize text sources into uint16 .bin shards (license filter)
  mix_bins.py           # blend shards into a training mixture
  train.py              # resumable, time-budgeted, DDP-capable training loop
  ablate.py             # sequential micro-ablation runner (+ Abhimanyu-gap readout)
  optim.py              # Muon / Muon+ / AdamW hybrid optimizer
  sample.py             # generate text from a checkpoint
  eval_probes.py        # frozen 27-prompt probe suite + held-out sentence loss
  eval_reversibility.py # entry-exit asymmetry (Abhimanyu gap)
  eval_entities.py      # entity-consistency report
  compare_models.py     # side-by-side generation comparison
  configs/              # kalia-m.yaml (the real model) + micro-* ablation configs
  notebooks/            # Kaggle: prep, mix, train, ablate, reproduce
  tests/                # pytest suite (64 tests)
  docs/journal/         # dated engineering journal (decisions, incidents, fixes)
  docs/DECISIONS.md     # numbered decision log
  docs/research/        # technique surveys with honest prior-art notes
  docs/preregistrations/# hash-anchored experiment pre-registrations + LEDGER
  docs/public/          # model card, article drafts, publish checklist
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
| QK-Norm | Normalizes attention queries/keys — stabilizes training | Schema validation on a hot path |
| SwiGLU | Gated feed-forward "thinking" block | Feature transform with a data-controlled volume knob |
| RoPE | Encodes token order by rotating vectors | Continuous timestamps for positions |
| Logit soft-cap | Squashes extreme output scores | Rate limiting the response |
| Random init | Starts knowing nothing | Empty database |
| Training | Show text, predict next token, nudge dials | The learning loop |
| Loss | "How wrong" score; random guessing ≈ 10.8 | Error rate |
| bpB (bits-per-byte) | Loss converted to compression of raw text | gzip ratio, but learned |
| Ablation | Re-train with one change to measure its effect | A/B test with everything else fixed |
| Backprop | Computes blame for every dial | `git blame` with auto-fix instructions |
| AdamW | The dial-nudging rulebook | Update policy |
| Muon / Muon+ | A newer dial-nudging rulebook for weight matrices | A faster scheduler for a specific workload |
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
python -m pytest tests/ -v          # full suite, ~16 seconds on CPU
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

Run the evaluation suite on a checkpoint:

```bash
python eval_probes.py --ckpt out/ckpt.pt --out out/eval/report.md
python eval_reversibility.py --ckpt out/ckpt.pt --out out/eval/reversibility.md
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
7. Edit `HUB_REPO` in `notebooks/kalia-train-v012.ipynb` to your HuggingFace repo id
   (e.g. `kalia-lm/kalia-v012`).

**Every training session**

1. Open `notebooks/kalia-train-v012.ipynb`, set Accelerator to **GPU T4 x2**, attach the
   `kalia-tokens` dataset, Internet on.
2. Run all cells. The first run starts from random weights; every later run resumes from
   the latest checkpoint — no progress is ever lost.

**Multi-session mechanics:** `train.py` saves a checkpoint (weights + optimizer + step
counter) to your HF repo every 30 minutes and at the end of a session. The next session
pulls the latest checkpoint and continues. Kaggle gives ~30 GPU-hours per week, so KALIA
needs roughly 1–2 weeks of weekly quota to finish.

**Monitoring:** watch `logs/train_log.csv` in your HF repo. Loss starts near 10.8 (random
guessing) and should fall toward ~3.1. Sample generations print every 500 steps.

## Version history

- **v0.1.0** — baseline: AdamW, RoPE, SwiGLU, RMSNorm, fp16, T4×2. Final val 3.2702.
- **v0.1.1** — Muon optimizer for hidden weight matrices (AdamW keeps
  embeddings/head/norms). Micro-ablation at equal tokens (30M params, 50M
  tokens): **3.5937 vs 3.8041** val loss, a **−0.21** win.
- **v0.1.2** — QK-Norm + logit soft-capping (τ = 30) on top of plain Muon at LR 0.02.
  Same ablation: **3.5103** val loss, **−0.29** vs baseline; LR 0.02 frozen.
  In full-scale training: 0.9415 bpB at step 3,478, Abhimanyu gap 6.06 nats.
  Stopped at 73% of the cosine schedule (quota) with the plateau documented.
  First coherent generations at step 1738: see `docs/samples/`.
- **v0.1.3+** — nothing: patch numbers stay inside a recipe family. The next
  release is **v0.2.0** (see roadmap).

## Roadmap

- **v0.2.0** — new data lineage: the compliance-clean v2 corpus (FineWeb-Edu-dedup,
  TinyStories, Cosmopedia, permissively licensed Python), plus only those changes
  that pass pre-registered promotion rules (architecture ablation; reversal
  training). Rules are hashed in `docs/preregistrations/LEDGER.md` before results
  exist.
- **v1 — SFT**: a small instruction-tuning stage so KALIA can follow prompts.
- **v2 — local deployment**: quantization/GGUF export, FastAPI inference service, chat UI.
- **v3 — scale-up**: reuse the same pipeline at 125M+ params.

## License

Code: MIT (see `LICENSE`). Model weights: Apache-2.0 at release. Training data is
not redistributed; sources are attributed in the model card (TinyStories:
CDLA-Sharing-1.0; FineWeb-Edu / Cosmopedia: ODC-By; code: permissive licenses only).
