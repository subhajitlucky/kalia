---
license: apache-2.0
language:
- en
library_name: pytorch
tags:
- text-generation
- small-language-model
- from-scratch
- muon
- qk-norm
- continual-learning
datasets:
- roneneldan/TinyStories
- HuggingFaceTB/smollm-corpus
- sedthh/gutenberg_english
pipeline_tag: text-generation
model-index:
- name: KALIA v0.2.0
  results:
  - task:
      type: text-generation
    dataset:
      name: PIQA
      type: piqa
    metrics:
    - name: Accuracy
      type: acc
      value: 63.8
  - task:
      type: text-generation
    dataset:
      name: ARC Easy
      type: arc_easy
    metrics:
    - name: Accuracy
      type: acc
      value: 42.0
  - task:
      type: text-generation
    dataset:
      name: HellaSwag
      type: hellaswag
    metrics:
    - name: Accuracy (normalized)
      type: acc_norm
      value: 39.8
  - task:
      type: text-generation
    dataset:
      name: WinoGrande
      type: winogrande
    metrics:
    - name: Accuracy
      type: acc
      value: 50.8
  - task:
      type: text-generation
    dataset:
      name: LAMBADA
      type: lambada_openai
    metrics:
    - name: Accuracy
      type: acc
      value: 20.8
---

# KALIA v0.2.0

A 57,856,256-parameter language model trained from **random initialization** on free
Kaggle GPUs. No pretrained weights, no fine-tuning, no distillation, no budget.

**This is an experiment release, not a promotion.** It improves held-out loss on its
pre-registered primary metric and it **fails its own pre-registered stability rule on two
of five benchmarks**. The thresholds were not moved after the results existed. The previous
release, [`kalia-v012`](https://huggingface.co/kalia-lm/kalia-v012), remains the recommended
checkpoint.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("kalia-lm/kalia-v020")
model = AutoModelForCausalLM.from_pretrained(
    "kalia-lm/kalia-v020", trust_remote_code=True
)
ids = tok("Once upon a time", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=80)[0]))
```

## What changed from v0.1.2

Only the training data. The architecture, optimizer, schedule and seed are identical, and
the run completed its full 4,770-step schedule where v0.1.2 stopped at 3,478 on quota.

The corpus was rebuilt shard by shard for licence compliance. That rebuild replaced the
held-out set, so **both** versions were re-measured on the new set before anything was
compared — the earlier 2.4366 figure is v0.1.2 on the old set and is never tabulated beside
these.

## Results

**Primary metric — deterministic held-out loss**, 100 batches × 8 × 1024 = 819,200 tokens,
seed 1234, the same call that produced v0.1.2's reference:

| | v0.1.2 | **v0.2.0** |
|---|---|---|
| val loss | 3.0533 | **2.8248** |
| bits per byte | 0.9992 | **0.9244** |

**0.2285 nats better, 4.6× the pre-registered 0.05-nat bar.** Both thresholds existed as a
hashed file before the run finished.

**Zero-shot benchmarks** (lm-evaluation-harness 0.4.9, 0-shot, 500 samples):

| task | v0.1.2 | v0.2.0 | Δ | σ | stability rule (≤1.0pp) |
|---|---|---|---|---|---|
| PIQA | 61.4 | 63.8 | +2.40 | 1.1 | pass |
| HellaSwag | 36.8 | 39.8 | +3.00 | 1.5 | pass |
| WinoGrande | 50.2 | 50.8 | +0.60 | — | pass (inside noise) |
| LAMBADA | 23.0 | 20.8 | **−2.20** | 1.2 | **fail** |
| ARC-Easy | 45.8 | 42.0 | **−3.80** | 1.7 | **fail** |

### Read the error bars

At 500 samples these benchmarks carry a standard error of roughly **2 points**. **Three of
the five movements are smaller than that**, so they cannot be called improvements. The
honest summary is *a clear improvement in held-out loss, no measurable change in most
zero-shot accuracy, and a real but modest regression on the two tasks that require
knowledge breadth and long-range recall.*

Two things follow, and they are both uncomfortable:

- **Our own accuracy thresholds were under-powered.** An earlier pre-registered bar of
  0.5 pp sat ~4.5× below the measurement's standard error, so it passed on noise alone
  (decision D45). It is recorded rather than quietly corrected.
- **The interim measurement was actively misleading.** At step 3,470 four of five tasks
  appeared to regress. Three of those reversed by the final checkpoint. Any conclusion about
  this corpus has to come from the final weights.

### Why LAMBADA moved down, and why that is not a mystery

A document-length probe run on the real mixture before training finished measured that only
**17.7% of training tokens sit inside a document at least as long as the 1,024-token
context**, and that TinyStories — a fifth of the mixture — supplies **0.03%** of them.
LAMBADA asks a model to hold a discourse and recall its final word, which is exactly what
that corpus cannot teach. The probe also found a candidate replacement source at 99.3%.

Whether the code-share change caused the ARC-Easy regression is **not established**. It is
the open question the next version has to answer.

## Architecture

| Field | Value |
|---|---|
| Parameters | 57,856,256 |
| Layers / heads | 10 / 8 |
| Embedding dim | 512 |
| Context length | 1,024 |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Blocks | Pre-norm, RMSNorm, SwiGLU, RoPE, QK-Norm, logit soft-cap (τ=30), tied embeddings, no biases |
| Optimizer | Muon (Newton–Schulz) on hidden matrices, AdamW for embeddings/head/norms |
| Schedule | Cosine, warmup 500, 4,770 steps |

Training: 2,500,853,760 tokens, 6.35h in the final session, 2× NVIDIA T4 via DDP, ~30
GPU-hours/week of free quota.

## Training data

Mixture as built, 2.4B training tokens + 10M held out:

| Dataset | Share | License | Docs ≥ context |
|---|---|---|---|
| `HuggingFaceTB/smollm-corpus` (FineWeb-Edu dedup) | 60% | ODC-By-1.0 | 26.2% |
| `roneneldan/TinyStories` | 20% | CDLA-Sharing-1.0 | 0.03% |
| `HuggingFaceTB/cosmopedia` v2 | 15% | Apache-2.0 | 13.2% |
| Python, per-file filtered | 5% | 7 permissive licences | — |

**This release's code slice is licence-clean by construction.** The previous release's was
not: it was built without a licence check, and a 20,000-file sample measured 41.7% of
characters under copyleft licences. That is disclosed on the
[v0.1.2 card](https://huggingface.co/kalia-lm/kalia-v012) rather than quietly fixed, and
those bytes are never redistributed.

Datasets are **not** redistributed here. Every source is public, so the reproducible
artifact is the recipe — mix config, per-source manifests, `prepare.py` and the licence
filter — which regenerates the corpus exactly. CDLA-Sharing §3.5 places Results (the
weights) beyond any obligation, and the sharing licence grants the right to train outright;
the three attribution conditions for releasing the bytes themselves are satisfiable and
documented in the repository.

## Limitations

- **Small and domain-narrow.** Coherent short-form English; children's stories and
  educational text. Not a general assistant, and there is no instruction tuning yet.
- **Two of five benchmarks regressed against v0.1.2** (ARC-Easy, LAMBADA) while loss
  improved. Cause not established.
- **Long-range recall is weak, and measurably so** — see the LAMBADA note above.
- **Training plateaus.** Validation loss is flat across the final 1,500 of 4,770 steps
  (spread 0.065 nats over 7 evals). The last 31% of training produced no measurable gain.
  Fixing that is the point of the next version.
- **Not continual yet.** This is a completed from-scratch base. v0.3.0 is planned as the
  first continual update of it — new data, learning rate re-warmed and re-decayed, old data
  replayed, no restart.
- May produce inaccurate or biased text. Not for production decisions.

## Reproducibility and provenance

- Weights exported from the step-4,770 training checkpoint and **verified by re-scoring the
  published `model.safetensors`** under the registered protocol; the export is only valid
  if it reproduces 2.8248. The tensors are bitwise-identical to the checkpoint.
- The published `config.json` is derived from the checkpoint's own configuration, not
  transcribed.
- `logs/train_log.csv` (477 rows, steps 10–4770) and `logs/val_log.csv` (19 rows) are
  included.
- Every decision, incident and pre-registration behind these numbers is in the
  [repository](https://github.com/subhajitlucky/kalia), with
  [`docs/SYNTHESIS.md`](https://github.com/subhajitlucky/kalia/blob/main/docs/SYNTHESIS.md)
  as the single-document summary. Pre-registration hashes verify with
  `python tools/verify_prereg_ledger.py`.

## Citation

```bibtex
@misc{kalia2026v020,
  author = {Subhajit Pradhan},
  title  = {KALIA v0.2.0: a 58M from-scratch language model, and the measurement errors that came with it},
  year   = {2026},
  note   = {Held-out loss improves 0.2285 nats over v0.1.2; two of five zero-shot benchmarks regress. Pre-registered, and reported as measured.}
}
```

## License

Code: MIT. Model weights: Apache-2.0. See [`LICENSE`](./LICENSE).
