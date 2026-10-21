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
- HuggingFaceTB/cosmopedia
- codeparrot/codeparrot-clean
pipeline_tag: text-generation
---

# KALIA v0.2.0

A 57,856,256-parameter language model trained from **random initialization** on
free Kaggle GPUs. No pretrained weights, no fine-tuning, no distillation, no
budget.

This is an **experiment release, not a promotion**. It improves held-out loss on
its pre-registered primary metric, and it **fails its own pre-registered
stability rule at the registered 500 samples on two of five benchmarks**. At the
2026-10-21 2,000-sample re-measurement, only **LAMBADA** remains outside the 1.0
percentage-point allowance (−1.25 points, z = −0.92). The ARC-Easy regression
did not replicate (+0.45), and no movement reaches 2σ. The thresholds were not
moved after the results existed. Under the registered rule, the previous
release, [kalia-v012](https://huggingface.co/kalia-lm/kalia-v012), remains the
recommended checkpoint.

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

Only the training data. The architecture, optimizer, schedule and seed are
identical, and the run completed its full 4,770-step schedule where v0.1.2
stopped at 3,478 on quota.

The corpus was rebuilt shard by shard for licence compliance. That rebuild
replaced the held-out set, so **both** versions were re-measured on the new set
before anything was compared — the earlier 2.4366 figure is v0.1.2 on the old
set and is never placed beside these numbers.

## Results

### Held-out loss

The primary metric is deterministic held-out loss: 100 batches × 8 sequences ×
1,024 tokens = 819,200 tokens, seed 1234, the same call that produced v0.1.2's
reference.

v0.1.2 scored **3.0533** val loss and **0.9992** bits-per-byte. v0.2.0 scored
**2.8248** and **0.9244**.

That is **0.2285 nats better — 4.6× the pre-registered 0.05-nat bar**. Both
thresholds existed as a hashed file before the run finished.

![Held-out loss comparison, v0.1.2 vs v0.2.0](charts/held_out_loss.png)

### Zero-shot accuracy

Measured with lm-evaluation-harness 0.4.9, 0-shot, at **2,000 samples** on
2026-10-21. Both checkpoints were scored in one CPU session. PIQA used its full
1,838 validation items and WinoGrande its full 1,267; the other three tasks used
2,000. Standard errors at this size are about 1.0–1.4 percentage points.

v0.1.2 → v0.2.0:

- **PIQA:** 59.63 → 61.37 (+1.74 points, z = 1.08)
- **HellaSwag:** 35.60 → 35.60 (0.00, z = 0.00)
- **WinoGrande:** 50.28 → 50.75 (+0.47, z = 0.24)
- **LAMBADA:** 25.15 → 23.90 (**−1.25 points, z = −0.92**)
- **ARC-Easy:** 45.95 → 46.40 (+0.45, z = 0.29)

![Zero-shot accuracy at 2,000 samples](charts/zero_shot_accuracy.png)

**No movement reaches 2σ.** LAMBADA is the only task outside the registered 1.0
percentage-point allowance, by 0.25 points, and its own z is −0.92. The honest
summary is a clear improvement in held-out loss, no individually significant
change in zero-shot accuracy, and one edge-of-the-rule regression on the task
that needs long-range recall.

![Accuracy deltas with 95% confidence intervals](charts/accuracy_deltas.png)

The original 500-sample evaluation remains in the record. Its ARC-Easy −3.80
and HellaSwag +3.00 did not survive the larger sample; the 500-sample standard
error was about 2 points (D45). The ARC-Easy point was a measurement artifact,
not a property of the corpus.

![The 500-sample deltas did not survive 2,000 samples](charts/correction_500_vs_2k.png)

### Why LAMBADA moved down

A document-length probe run on the real mixture before training finished
measured that only **17.7% of training tokens sit inside a document at least as
long as the 1,024-token context**, and that TinyStories — a fifth of the mixture
— supplies **0.03%** of them. LAMBADA asks a model to hold a discourse and
recall its final word, which is exactly what that corpus cannot teach. The probe
also found a candidate replacement source at 99.3%. The mixture remains the open
lever, but it is not an accuracy catastrophe.

## Architecture

- **Parameters:** 57,856,256.
- **Layers and heads:** 10 layers, 8 heads.
- **Embedding dimension:** 512.
- **Context length:** 1,024 tokens.
- **Vocabulary:** 50,257 (GPT-2 BPE).
- **Blocks:** pre-norm decoder-only, RMSNorm, SwiGLU, RoPE, QK-Norm, logit
  soft-cap (τ = 30), tied embeddings, no biases.
- **Optimizer:** Muon (Newton–Schulz) on hidden matrices, AdamW on embeddings,
  head and norms.
- **Schedule:** cosine, 500 warmup steps, 4,770 total steps.

Training used 2,500,853,760 tokens, 6.35 hours in the final session, on 2×
NVIDIA T4 via DDP, inside the free tier's roughly 30 GPU-hours per week.

## Training data

The mixture is 2.4B training tokens plus 10M held out:

- **60% FineWeb-Edu dedup** from `HuggingFaceTB/smollm-corpus`, under ODC-By-1.0.
  26.2% of its documents are at least one context long.
- **20% TinyStories** from `roneneldan/TinyStories`, under CDLA-Sharing-1.0.
  0.03% of its documents are at least one context long.
- **15% Cosmopedia v2** from `HuggingFaceTB/cosmopedia`, under Apache-2.0. 13.2%
  of its documents are at least one context long.
- **5% Python**, per-file filtered to seven permissive licences (MIT,
  Apache-2.0, BSD-2/3, ISC, Unlicense, CC0).

This release's code slice is licence-clean by construction. The previous
release's was not: it was built without a licence check, and a 20,000-file
sample measured 41.7% of characters under copyleft licences. That is disclosed
on the [v0.1.2 card](https://huggingface.co/kalia-lm/kalia-v012) rather than
quietly fixed, and those bytes are never redistributed.

### Released corpus shards

This repository publishes the five shards the continual-update experiments are
scored against: the four equalised per-source shards and the pinned probe set.
The full 2.4B-token training mixture is **not** redistributed. Each source shard
is 124,500,691 tokens, equalised because the mixture sampler requires
equal-length shards. Each skips its source's proportional validation head
(6M, 2M, 1.5M and 0.5M tokens respectively), so the canonical held-out set
cannot appear in them.

- **`corpus/fineweb.bin`** — 124,500,691 tokens of FineWeb-Edu dedup from the
  SmolLM corpus. ODC-By-1.0; credit `HuggingFaceTB/smollm-corpus`.
- **`corpus/tinystories.bin`** — 124,500,691 tokens of TinyStories.
  CDLA-Sharing-1.0. The licence terms are unmodified; the bytes are tokenized
  uint16 with an EOS separator between documents, not the raw source text.
  Credit `roneneldan/TinyStories`.
- **`corpus/cosmopedia.bin`** — 124,500,691 tokens of Cosmopedia v2 via the
  SmolLM corpus. Apache-2.0 content card; the ODC-By container is also credited.
- **`corpus/python.bin`** — 124,500,691 tokens of Python, filtered per file to
  seven permissive licences by `prepare.py::is_permissive_license`.
- **`corpus/probe_val.bin`** — 1,000,000 tokens: the frozen first million tokens of the v0.2.0 **training** stream. It is deliberately not held-out data and is not the canonical validation set. Per source, as above.
- **`corpus/manifest.json`** — source paths, ratios and per-shard SHA-256.

Per-file licences and attribution links are in
[`corpus/NOTICE.md`](https://huggingface.co/kalia-lm/kalia-v020/blob/main/corpus/NOTICE.md)
(the repository's `docs/legal/corpus-shard-notice.md`).

The bytes are the licence-filtered build — code from `kalia-prep-v2b`, text
sources unchanged — so **no copyleft text is redistributed**. The recipe that
regenerates the full corpus, including the mix config, `prepare.py` and the
licence filter, is in the
[repository](https://github.com/subhajitlucky/kalia). CDLA-Sharing §3.5 places
Results (the weights) beyond any obligation, and the sharing licence grants the
right to train outright.

## Limitations

- **Small and domain-narrow.** Coherent short-form English; children's stories
  and educational text. Not a general assistant, and there is no instruction
  tuning yet.
- **At 2,000 samples, one of five benchmarks is outside the 1.0-point
  allowance:** LAMBADA, at −1.25 points (z = −0.92). Loss improved
  substantially, and no movement reaches 2σ.
- **Long-range recall is weak, and measurably so.** See the LAMBADA note above.
- **Training plateaus.** Validation loss is flat across the final 1,500 of 4,770
  steps (spread 0.065 nats over seven evaluations). The last 31% of training
  produced no measurable gain. Fixing that is the point of the next version.
- **Not continual yet.** This is a completed from-scratch base. v0.3.0 is
  planned as the first continual update of it: new data, learning rate re-warmed
  and re-decayed, old data replayed, no restart.
- May produce inaccurate or biased text. Not for production decisions.

## Reproducibility and provenance

- Weights were exported from the step-4,770 training checkpoint and **verified
  by re-scoring the published `model.safetensors`** under the registered
  protocol. The export is only valid if it reproduces 2.8248. The tensors are
  bitwise-identical to the checkpoint.
- The published `config.json` is derived from the checkpoint's own
  configuration, not transcribed.
- `logs/train_log.csv` (477 rows, steps 10–4770) and `logs/val_log.csv` (19 rows)
  are included.
- The `corpus/` shards and probe set are hash-pinned: every published shard's
  SHA-256 matches `corpus/manifest.json` and the repository's
  `docs/legal/val-shard-hashes.json`.
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
  note   = {Held-out loss improves 0.2285 nats over v0.1.2. At 2,000 samples, one of five zero-shot benchmarks (LAMBADA) is outside the 1.0 pp allowance; the 500-sample ARC-Easy regression did not replicate and no movement reaches 2 sigma. Pre-registered, and reported as measured.}
}
```

## License

Code: MIT. Model weights: Apache-2.0. See [`LICENSE`](./LICENSE).
