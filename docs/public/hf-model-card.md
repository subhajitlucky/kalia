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
- HuggingFaceTB/cosmopedia
- codeparrot/codeparrot-clean
pipeline_tag: text-generation
---

# KALIA v0.1.2

> अथ शब्दानुशासनम् — *"Now begins the discipline of words."*
> (opening invocation of Pāṇini's Aṣṭādhyāyī)

A 57.9M-parameter decoder-only language model trained **from scratch** — no
pretrained weights, no distillation, no fine-tuning — on free-tier Kaggle T4
GPUs. Every weight was learned by this project's own training run.

## Model

The model has **57,856,256 parameters**. It is a pre-norm decoder-only
transformer with RMSNorm, SwiGLU, RoPE, QK-Norm, a logit soft-cap at τ = 30, and
tied embeddings. It has 10 layers, 8 heads, an embedding dimension of 512, a
context length of 1,024 tokens, and a 50,257-token GPT-2 BPE vocabulary.

Training used Muon (Newton–Schulz) on hidden matrices and AdamW on embeddings,
head and one-dimensional parameters, in fp16 with gradient scaling and DDP
across 2× NVIDIA T4. The model saw 1.82B tokens — 3,478 of 4,770 planned steps
— and stopped on quota; see “Stopping”.

## Usage

This repository hosts the raw training checkpoint
(`checkpoints/ckpt.pt`, a dict with `model`, `optimizer`, `step`, `tokens` and
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

It runs on CPU; the weights are about 230MB in fp16.

## Evaluation

The deterministic validation loss uses 100 batches × 8 sequences × 1,024 tokens
= 819,200 tokens, seed 1234.

On the **original** `kalia-prep` validation set, v0.1.2 scores **2.4366** val
loss and **0.8184** bits-per-byte. On the compliance-rebuilt `kalia-prep-v2b`
validation set, the same weights score **3.0533** and **0.9992**. The rebuild
replaced a source slice and made the held-out set measurably harder: identical
weights move by +0.62 nats across the two sets. Numbers from the two sets are
not comparable, and the v2b figure is the reference for later versions. See
decision D43.

The 20-sentence probe scores **3.2303** held-out loss and **0.9415**
bits-per-byte. The Abhimanyu gap — reverse-token NLL minus forward NLL — is
**6.06 nats** (forward 3.23, reverse 9.29; random is about 10.8).

### Zero-shot accuracy

Measured at release with lm-evaluation-harness, 0-shot, at 500 samples. The
standard error at this size is about 2 points (D45), so the scores below are
estimates, not precise rankings.

- **PIQA (acc):** 61.4% against 50% chance.
- **ARC-Easy (acc):** 45.8% against 25% chance.
- **HellaSwag (acc_norm):** 36.8% against 25% chance.
- **WinoGrande (acc):** 50.2% against 50% chance.
- **LAMBADA (acc / perplexity):** 23.0% / 193.6; no meaningful chance baseline.

![v0.1.2 zero-shot accuracy against chance](charts/v012_zero_shot_vs_chance.png)

For scale context, third-party leaderboard tables list OPT-125M — 125M
parameters, 300B tokens — at PIQA 63.0, ARC-Easy 43.5 and HellaSwag 29.2.
Harness versions differ, so treat that as context, not a head-to-head. KALIA is
a storyteller, not a knowledge model; its LAMBADA weakness reflects a corpus
without long-form narrative cloze.

The micro-ablations at 30M parameters, equal tokens and seed 1337 were AdamW
3.8041 → Muon 3.5937 → Muon + QK-Norm + soft-cap **3.5103**. The full-scale Muon
model overtook the AdamW baseline's final loss with **~23% fewer tokens**.

The Abhimanyu gap is the project's entry–exit asymmetry metric: the model can
enter fluent text but cannot exit it — process it in reverse. The
pre-registered experiment aimed at closing it has run and **failed**. Training
on 50% chunk-preserving reversal left the gap at 5.25 nats against a 5.11-nat
control: worse on both seeds and both metrics (X16, decision D42). Reversing
chunk *order* does not teach token-level entry, so no gap claim is carried
forward.

## Stopping

Training stopped at step 3,478 — 73% of the cosine schedule — when the weekly
GPU quota was exhausted. Validation had been flat within ±0.1 (eval noise) for
1,000 steps, and a deterministic probe set showed no further gain, so the stop
was declared final under the project's pre-registered stopping discipline rather
than resumed. The plateau is documented in the repository journal.

## Limitations

Stated as measured, not as marketing. At 500 samples these benchmarks have
standard errors of roughly 2.0–2.2 points.

- **Small model.** Coherent short-form English, not a general assistant.
- **Domain-narrow.** Children's stories and educational web text.
- **Entity consistency degrades over long outputs.** Names drift.
- **No instruction following.** Post-training is a planned stage, not a
  completed one.
- **Long-range recall is weak, and measurably so.** Only 17.7% of training
  tokens sit inside a document at least as long as the 1,024-token context, and
  TinyStories — 20% of the mixture — supplies 0.01% of them. LAMBADA, which
  requires holding a discourse and recalling its end, is the task this corpus is
  worst equipped for.
- **Training runs to a plateau.** Validation loss is flat across the final
  1,500 of 4,770 steps. The last 31% of training produced no measurable gain.
- **May produce inaccurate or biased text.** Not for production decisions.

## Training data

The mixture weights as built (`kalia-prep-v2`) are FineWeb-Edu 60%, TinyStories
20%, Cosmopedia 15% and code 5%, over 2.4B training tokens.

- **FineWeb-Edu dedup** from `HuggingFaceTB/smollm-corpus`, 60%, under
  ODC-By-1.0.
- **TinyStories** from `roneneldan/TinyStories`, 20%, under CDLA-Sharing-1.0.
- **Cosmopedia** from `HuggingFaceTB/cosmopedia`, 15%, under Apache-2.0.
- **Python** from `codeparrot/codeparrot-clean`, 5%, per-file mixed licences —
  see below.
- **GPT-2 BPE tokenizer** from `openai/tiktoken`, MIT.

![Training mixture: 60 / 20 / 15 / 5](charts/v012_data_mixture.png)

### Disclosure: the code slice of this checkpoint is not licence-clean

The 5% code slice of this checkpoint was built **without** a licence filter.
Sampling 20,000 files of the source corpus (197M characters) measures **41.7% of
characters under non-permissive licences**, of which GPL-family terms are 39.5%
of all files (GPL-3.0, AGPL-3.0, GPL-2.0, LGPL). Applying that share, this
checkpoint trained on roughly **50M copyleft-licensed code tokens, about 2% of
the training set**. The figure is an estimate: it samples the upstream corpus
rather than our shard, and uses character share as a proxy for token share.

**No copyleft obligation is attached to these weights.** This release
redistributes weights only, not data. Training on copyleft text does not make
the weights a derivative work — the prevailing view, though not settled in every
jurisdiction. The obligation would attach to redistributing the copyleft text
itself, which this release does not do and will not.

Datasets are **not** redistributed. Code is MIT; weights are Apache-2.0.

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

- Paper page, full record and replayable training console:
  [subhajitpradhan.vercel.app/kalia](https://subhajitpradhan.vercel.app/kalia)
- Code, journal, decisions and hash-anchored pre-registrations:
  [github.com/subhajitlucky/kalia](https://github.com/subhajitlucky/kalia)

## Disclaimer

A personal research project. Not affiliated with any government scheme,
company, or other project using a similar name.
