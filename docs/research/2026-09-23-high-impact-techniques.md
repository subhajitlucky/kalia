# High-Impact Techniques, Ethical Boundaries, and the KALIA-EM+TTT Target

Status: active research track
Date: 2026-09-23
Sources: SOAP (ICLR 2025), KL-Shampoo (2025), NVIDIA "SOAP, Muon, and Beyond"
(2026), Muon+ (arXiv 2602.21545), MobileLLM-R1 (ICLR 2026), Scaling
Data-Constrained LMs (NeurIPS 2023), Titans (NeurIPS 2025), TTT-E2E (NVIDIA,
2026), TTCD (2026).

## 0. Evidence rule

No known method multiplies learning efficiency by 100×. Such a claim would
require either measurement error or misrepresentation. The largest *measured*
efficiency multipliers are 1.5×–8×, and they are mostly data-side:

| Lever | Measured effect | Source |
|---|---|---|
| Data curation (quality + mixture) | match Qwen3-0.6B with 11.7% of its tokens ≈ **8×** | MobileLLM-R1 |
| Code mixed into English pretraining | **2× effective tokens** on NLP tasks | Muennighoff et al. |
| SOAP / KL-Shampoo optimizers | ~40% fewer iterations than AdamW | SOAP, KL-Shampoo |
| Muon+ | up to 37% faster to target loss | arXiv 2602.21545 |

Any future claim in this repository must carry a measurement and a source, or be
labelled as a hypothesis.

## 1. Three high-impact techniques (real, powerful, rarely adopted)

### F1. Test-time training — the model learns while it reads
Titans (NeurIPS 2025) and TTT-E2E (NVIDIA 2026) update a subset of weights at
inference: context is *compressed into weights* rather than cached. Results:
Titans outperforms GPT-4 on BABILong despite far fewer parameters; TTT-E2E
holds constant inference latency to 2M tokens with no loss scaling wall.
In-place variants (In-Place TTT, 2026; IP-TTCD, 2026) reuse the MLP
down-projection matrix as "fast weights" — a ~20-line architectural change on a
standard transformer.

Why it is rarely adopted: it breaks the assumption that weights freeze after
training. Why it fits KALIA: a 58M storyteller that remembers the characters and
facts of the story it is reading, *while* reading.

### F2. External memory / retrieval
Knowledge lives in an index, not in weights. A 58M model with retrieval answers
facts far beyond its parameter count. Zero training cost. Standard practice at
scale, rarely combined with tiny from-scratch models. Plan: for the local KALIA
app (v2 deployment), not for pretraining.

### F3. Self-teaching — EMA-teacher self-distillation
The model distils its own smoothed copy (EMA teacher) into itself during
pretraining. No external teacher, so the lineage stays 100% ours. Established in
semi-supervised learning (Mean Teacher, 2017); rarely validated at pretraining
scale — which makes it a legitimate originality candidate (R3/O5).

## 2. What we refuse (no exceptions)

- Training on benchmark/test data or otherwise contaminating evaluation.
- Reporting numbers without the config, seed, and commit that produced them.
- Presenting borrowed capability as our own (this is why D21 rejected
  distillation).
A technique that compromises the measurement is not a technique — it
invalidates the result and the entire project along with it.

## 3. Our invention target — KALIA-EM+TTT

**Claim**: entity-memory slots + in-place test-time fast weights, combined at
58M parameters, produce a story model whose character/fact consistency across
long texts exceeds models several times its size *without* borrowing any
external weights.

Novelty position: entity memory exists (Memorizing Transformers, 2022), test-time
fast weights exist (In-Place TTT, 2026), but their combination at sub-100M scale
for story consistency is, to our knowledge, unpublished. If it works, it is
ours — with a measured, reproducible comparison.

Evaluation plan (no leakage):
- Held-out multi-thousand-token stories, unseen at training time.
- Metric: character-name and attribute consistency across the story (a
  scripted probe set we build once and freeze), plus val loss/bpB.
- Baselines: KALIA v0.1.2 (no TTT), KALIA + TTT, KALIA + EM, KALIA + EM + TTT.

## 4. Experiment queue

| ID | Experiment | Cost | Status |
|---|---|---|---|
| X7 | In-place TTT micro-ablation (fast weights = MLP down-projection, updated over chunks) | ~2h GPU | queued |
| X8 | EMA-teacher self-distillation micro-ablation (= R3/O5) | ~2h GPU | queued |
| X9 | Entity-memory slots ablation (= O3, now validated as part of the invention target) | ~2h GPU | queued |
| X10 | Retrieval layer for local KALIA app | CPU only | planned (deployment phase) |

X7–X9 all run at micro scale first under the standard promotion rule (≥2 seeds
for promotion, bpB + val loss reported).
