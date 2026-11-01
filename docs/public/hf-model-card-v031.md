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
- weight-merge
datasets:
- roneneldan/TinyStories
- HuggingFaceTB/smollm-corpus
- HuggingFaceTB/cosmopedia
- codeparrot/codeparrot-clean
pipeline_tag: text-generation
---

# KALIA v0.3.1

A **57,856,256-parameter** language model, released as a **weight-space merge**
of v0.2.0 and v0.3.0. It is the same architecture and training recipe as v0.3.0;
no training, no distillation and no new data were used for this release.

The merge is:

```text
theta_v031 = 0.30 * theta_v020 + 0.70 * theta_v030
```

applied tensor by tensor. v0.3.0 is a continuation of v0.2.0, so the two
checkpoints lie on a connected path and their average is a valid model soup.
The step counter, token counter, optimizer state and config are taken from
v0.3.0.

**This is not a new architecture.** It recovers part of a checkpoint-selection
regression in v0.3.0. The v0.3.0 second epoch's best in-training 100-batch val
was **2.7255 at step 6,500**; the final checkpoint at step 6,770 scored
**2.7743**. The intermediate checkpoint was overwritten. The merge reaches
**2.7579** without training.

## Results

Primary metric: deterministic held-out loss, 100 batches × 8 sequences × 1,024
tokens = **819,200 tokens**, seed 1234.

| Model | Val loss | bpB |
|---|---:|---:|
| v0.2.0 | 2.8248 | 0.9244 |
| v0.3.0 | 2.7743 | 0.9070 |
| **v0.3.1** | **2.7579** | **0.9016** |

v0.3.1 is **−0.0164 nats** vs v0.3.0 and **−0.0669 nats** vs v0.2.0.

### Zero-shot accuracy at 2,000 samples

Measured with lm-evaluation-harness 0.4.9, 0-shot, 2,000 samples per task.
Standard errors are about 1.0–1.4 percentage points.

| Task | v0.2.0 | v0.3.0 | **v0.3.1** | Δ vs v0.3.0 | z |
|---|---:|---:|---:|---:|---:|
| ARC-Easy | 46.40 | 47.55 | **48.00** | +0.45 | +0.28 |
| HellaSwag | 35.60 | 37.20 | **37.65** | +0.45 | +0.29 |
| LAMBADA | 23.90 | 26.30 | **27.00** | +0.70 | +0.50 |
| PIQA | 61.37 | 61.59 | **61.97** | +0.38 | +0.24 |
| WinoGrande | 50.75 | 51.30 | **52.25** | +0.95 | +0.48 |

All five tasks improve; no task regresses by >2σ. The 2k gains are modest and
none reaches 2σ.

### CL-0 retention probe

| Model | Probe loss |
|---|---:|
| v0.2.0 | 2.798875 |
| v0.3.0 | 2.756648 |
| **v0.3.1** | **2.740949** |

The merge improves the retention probe by **0.0157** vs v0.3.0; the val gain is
not bought with forgetting.

## Architecture

| Field | Value |
|---|---|
| Parameters | 57,856,256 |
| Architecture | Pre-norm decoder-only: RMSNorm, SwiGLU, RoPE, QK-Norm, logit soft-cap (τ = 30), tied embeddings, no biases |
| Layers / heads / dim | 10 / 8 / 512 |
| Context | 1,024 tokens |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Optimizer (base run) | Muon (Newton–Schulz) on hidden matrices; AdamW on embeddings, head and norms |
| Merge | 30% v0.2.0 + 70% v0.3.0, per tensor |

## Training data (v0.3.0 base)

The base model is v0.3.0: 2,000 update steps on the v0.2.0 2.4B corpus, no
replay, re-warm cosine, seed 1401. Total tokens seen: **3.55B** (1.42 epochs).

| Source | Share | Licence |
|---|---:|---|
| FineWeb-Edu dedup (`HuggingFaceTB/smollm-corpus`) | 60% | ODC-By-1.0 |
| TinyStories (`roneneldan/TinyStories`) | 20% | CDLA-Sharing-1.0 |
| Cosmopedia v2 (`HuggingFaceTB/cosmopedia`) | 15% | Apache-2.0 |
| Python (`codeparrot/codeparrot-clean`) | 5% | Mixed; per-file permissive filter |

## Provenance

Source checkpoints:

- v0.2.0: `924ad52b27017bb823e925df6695d1becdef032b772ae8436eb0b74bdbac6718`
- v0.3.0: `80b064f9a99e6ed5372e7fa46021574a9b21298b822fd2c581357831cc77537e`

Merged checkpoint (`best_merged.pt`):

- sha256 `f59e059ce673d15601dc637940c713731fa77ca874202804395847ed9d415382`
- weight sha256 `19f68b01fdc27d5910223dc624b1bb9f33f99993643f0169481d2abd0c604db0`
- step 6,770; 3,549,429,760 tokens; val 2.7579 / bpB 0.9016

The export re-scores the written `model.safetensors` under the same protocol;
the published artifact is the one that was measured.

## Limitations

- **Weight-space soup, not an architecture result.** There is no new mechanism.
- **Small 2k gains.** The val and CL-0 gains are the strong parts; the benchmark
  deltas are within ~1σ–1.75σ.
- **58M parameters, 1,024 context.** This is a small model; it is not a
  frontier LLM and does not claim to be.
- **No instruction tuning.** It is a base model.
- **English web/stories/code mixture.** Other languages and domains are out of
  scope.
- **The merge does not fix the root cause.** The real fix is saving the best
  checkpoint during training; the merge is a zero-cost recovery.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("kalia-lm/kalia-v031")
model = AutoModelForCausalLM.from_pretrained(
    "kalia-lm/kalia-v031", trust_remote_code=True
)
ids = tok("Once upon a time", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=80)[0]))
```

## Older releases

- [kalia-v012](https://huggingface.co/kalia-lm/kalia-v012) — 3.0533 val.
- [kalia-v020](https://huggingface.co/kalia-lm/kalia-v020) — 2.8248 val.

Their model cards are frozen historical records and are not modified by this
release. v0.3.1 is a new repository.

## Licence

Apache-2.0 for the code and weights. Dataset licences are listed above; the
Python subset keeps a per-file permissive filter. See the repository `LICENSE`.
