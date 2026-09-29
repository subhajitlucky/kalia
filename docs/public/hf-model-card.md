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
- stories
datasets:
- roneneldan/TinyStories
- HuggingFaceTB/smollm-corpus
pipeline_tag: text-generation
model-index:
- name: KALIA v0.1.2
  results:
  - task:
      type: text-generation
    dataset:
      name: PIQA
      type: piqa
    metrics:
    - name: Accuracy
      type: acc
      value: 61.4
  - task:
      type: text-generation
    dataset:
      name: ARC Easy
      type: arc_easy
    metrics:
    - name: Accuracy
      type: acc
      value: 45.8
  - task:
      type: text-generation
    dataset:
      name: HellaSwag
      type: hellaswag
    metrics:
    - name: Accuracy (normalized)
      type: acc_norm
      value: 36.8
  - task:
      type: text-generation
    dataset:
      name: WinoGrande
      type: winogrande
    metrics:
    - name: Accuracy
      type: acc
      value: 50.2
  - task:
      type: text-generation
    dataset:
      name: LAMBADA OpenAI
      type: lambada_openai
    metrics:
    - name: Accuracy
      type: acc
      value: 23.0
---

# KALIA v0.1.2

> अथ शब्दानुशासनम् — *"Now begins the discipline of words."*
> (opening invocation of Pāṇini's Aṣṭādhyāyī)

A 57.9M-parameter decoder-only language model trained **from scratch** — no
pretrained weights, no distillation, no fine-tuning — on free-tier Kaggle T4
GPUs. Every weight was learned by this project's own training run.

## Model details

| Field | Value |
|---|---|
| Parameters | 57,856,256 |
| Architecture | Pre-norm decoder-only: RMSNorm, SwiGLU, RoPE, QK-Norm, logit soft-cap (τ=30), weight tying |
| Layers / heads / dim | 10 / 8 / 512 |
| Context | 1,024 tokens |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Optimizer | Muon (Newton–Schulz) on hidden matrices; AdamW for embeddings/head/1D |
| Precision | fp16 with gradient scaling, DDP across 2× NVIDIA T4 |
| Training data | TinyStories (~500M tokens) + FineWeb-Edu dedup (~2B tokens) |
| Tokens seen | 1.82B (3,478 of 4,770 planned steps — see "Stopping") |

## Usage

This repository hosts the raw training checkpoint
(`checkpoints/ckpt.pt` — a dict with `model`, `optimizer`, `step`, `tokens`,
`config`). Generate with the project's own code:

```bash
git clone https://github.com/subhajitlucky/kalia
cd kalia
pip install torch tiktoken pyyaml
python - <<'EOF'
from huggingface_hub import hf_hub_download
path = hf_hub_download("kalia-lm/kalia-v012", "checkpoints/ckpt.pt")
print(path)
EOF
python sample.py --ckpt <path-from-above> --prompt "Once upon a time"
```

Runs on CPU (~230MB in fp16 weights).

## Evaluation

| Metric | Value |
|---|---|
| Validation loss — deterministic 100-batch eval (819,200 tokens, seed 1234), measured on the **original** `kalia-prep` val set | **2.4366** |
| bits-per-byte (validation, original val set) | **0.8184** |
| Validation loss — same weights, same protocol, on the compliance-rebuilt `kalia-prep-v2b` val set (canonical from v0.2.0 onward) | **3.0533** |
| bits-per-byte (v2b val set) | **0.9992** |
| Probe held-out loss (20 fixed sentences) | 3.2303 |
| bits-per-byte (probe) | 0.9415 |
| Abhimanyu gap (reversed-text NLL − forward NLL) | 6.06 nats (forward 3.23 vs reverse 9.29; random ≈ 10.8) |

> The two validation rows use the same weights and the same 819,200-token protocol on two
> different held-out sets. The corpus was rebuilt for licence compliance, which replaced a
> source slice and made the held-out set measurably harder (+0.62 nats for identical
> weights). Numbers from the two sets are not comparable; the v2b row is the reference for
> every later version. See decision D43.

### Zero-shot benchmarks (lm-evaluation-harness, 0-shot, 500 samples)

| Task | Score | Chance |
|---|---|---|
| PIQA (acc) | **61.4%** | 50% |
| ARC-Easy (acc) | **45.8%** | 25% |
| HellaSwag (acc_norm) | 36.8% | 25% |
| WinoGrande (acc) | 50.2% | 50% |
| LAMBADA (acc / perplexity) | 23.0% / 193.6 | — |

For scale context: third-party leaderboard tables list OPT-125M (125M params, 300B
tokens) at PIQA 63.0 / ARC-Easy 43.5 / HellaSwag 29.2; harness versions differ, so
treat this as context, not a head-to-head. KALIA is a storyteller, not a knowledge
model — its LAMBADA weakness reflects a corpus without long-form narrative cloze.
Standard errors are ±2.2pp at this sample size.

Micro-ablations at 30M params / equal tokens / same seed: AdamW 3.8041 →
Muon 3.5937 → Muon + QK-Norm + soft-cap **3.5103**. The full-scale Muon model
overtook the AdamW baseline's final loss with **~23% fewer tokens**.

The "Abhimanyu gap" is the project's entry–exit asymmetry metric: the model can
enter fluent text but cannot exit it (process it in reverse). The pre-registered
experiment aimed at closing it has since run and **failed**: training on 50%
chunk-preserving reversal left the gap at 5.25 nats against a 5.11-nat control —
worse, on both seeds, on both metrics (X16, decision D42). Reversing chunk
*order* does not teach token-level entry, so no gap claim is carried forward and
v0.2.0 ships the same recipe with a compliance-rebuilt corpus.

## Stopping

Training stopped at step 3,478 (73% of the cosine schedule) when the weekly GPU
quota was exhausted. Validation had been flat within ±0.1 (eval noise) for
1,000 steps and a deterministic probe set showed no further gain, so the stop
was declared final under the project's pre-registered stopping discipline
rather than resumed. The plateau is documented in the repository journal.

## Limitations

Stated as measured, not as marketing. The noise floor matters here: at 500
samples these benchmarks have standard errors of roughly 2.0–2.2 points, so
differences smaller than about 2 points are not real.

- **Small model.** Coherent short-form English, not a general assistant.
- **Domain-narrow.** Children's stories and educational web text.
- **Entity consistency degrades over long outputs** (names drift).
- **No instruction following.** Post-training is a planned stage, not a
  completed one.
- **Two of five zero-shot benchmarks regressed against v0.1.2** when the corpus
  was rebuilt for licence compliance: ARC-Easy 45.8 → 42.0 and LAMBADA 23.0 →
  20.8. Held-out loss over the same rebuild improved by 0.2285 nats, so loss and
  accuracy moved in opposite directions. Three of the five tasks moved by less
  than their own ~2 pp standard error and are not separable at all. Whether the
  code-share change caused the regression is not established; it is the open
  question the next version has to answer.
- **Long-range recall is weak, and measurably so.** Only 17.7% of training tokens
  sit inside a document at least as long as the 1,024-token context, and
  TinyStories — 20% of the mixture — supplies 0.01% of them. LAMBADA, which
  requires holding a discourse and recalling its end, is the task this corpus is
  worst equipped for, and it is one of the two that regressed.
- **Training runs to a plateau.** Validation loss is flat across the final
  1,500 of 4,770 steps. The last 31% of training produced no measurable gain.
- **May produce inaccurate or biased text.** Not for production decisions.

## Training data attribution

Mixture weights, as built (`kalia-prep-v2`): FineWeb-Edu 60%, TinyStories 20%,
Cosmopedia 15%, code 5%, of 2.4B training tokens.

| Dataset | Share | License |
|---|---|---|
| HuggingFaceTB/smollm-corpus (FineWeb-Edu dedup) | 60% | ODC-By-1.0 |
| roneneldan/TinyStories | 20% | CDLA-Sharing-1.0 |
| HuggingFaceTB/cosmopedia | 15% | Apache-2.0 |
| bigcode/codeparrot-clean (Python) | 5% | **per-file, mixed — see below** |
| GPT-2 BPE tokenizer (openai/tiktoken) | — | MIT |

### Disclosure: the code slice of *this* checkpoint is not licence-clean

The 5% code slice of this checkpoint was built **without** a licence filter.
Sampling 20,000 files of the source corpus (197M characters) measures **41.7% of
characters under non-permissive licences**, of which GPL-family terms are 39.5%
of all files (GPL-3.0, AGPL-3.0, GPL-2.0, LGPL). Applying that share, this
checkpoint trained on roughly **50M copyleft-licensed code tokens, ≈2% of the
training set**. The figure is an estimate: it samples the upstream corpus
rather than our shard, and uses character share as a proxy for token share.

**No copyleft obligation is attached to these weights.** This release
redistributes weights only, not data. Training on copyleft text does not make
the weights a derivative work — the prevailing view, though not settled in every
jurisdiction. The obligation would attach to redistributing the copyleft text
itself, which this release does not do and will not.

The successor checkpoint, **v0.2.0**, rebuilds the code slice with a permissive
licence filter and is licence-clean by construction. The process failure and its
measurement are recorded as incident I16; the filter's late arrival is the root
cause.

Datasets are **not** redistributed. Code: MIT. Weights: Apache-2.0.

## Citation

```bibtex
@misc{kalia2026,
  title        = {KALIA: a 58M-parameter language model trained
                  from scratch on free GPUs},
  author       = {Pradhan, Subhajit},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/kalia-lm/kalia-v012}},
  note         = {v0.1.2; stopped at step 3,478 of 4,770 of the
                  cosine schedule; all logs, decisions, and
                  pre-registrations are public}
}
```

## Links

- Paper page (full record, replayable training console):
  [subhajitpradhan.vercel.app/kalia](https://subhajitpradhan.vercel.app/kalia)
- Code, journal, decisions, and hash-anchored pre-registrations:
  [github.com/subhajitlucky/kalia](https://github.com/subhajitlucky/kalia)

## Disclaimer

A personal research project. Not affiliated with any government scheme,
company, or other project using a similar name.
