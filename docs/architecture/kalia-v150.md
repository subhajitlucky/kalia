# KALIA v150 — from-scratch 149M dense decoder

Status: **architecture decision, 2026-11-10.** No v150 checkpoint exists yet.
The public base remains **v0.3.1** (57,856,256 params, held-out 2.7579, 9-task
mean 0.4258). This document is a specification and a pre-registration input,
not a result.

## 1. Decision

Build **one dense decoder**:

| field | value |
|---|---|
| parameters | **149,316,224** |
| layers | **34** |
| width | **576** |
| attention | **GQA, 9 query heads / 3 KV heads**, head_dim 64 |
| FFN | **SwiGLU, hidden 1536** |
| norm | **RMSNorm**, pre-norm |
| position | **RoPE, theta 100,000** |
| vocab | **50,257**, GPT-2 byte-level BPE, **tied embeddings** |
| context | **8,192** (staged: 2k → 4k → 8k) |
| stabilizers | **QK-Norm** + **logit soft-cap 30** |
| optimizer | **Muon** on hidden matrices, AdamW on embeddings/norms |
| data | **1T tokens**, multi-stage, English-first |
| post-training | capability adapters first; RLVR only as a documented probe |

This is **not** a depth-expanded v0.3.1. It is random-initialized and trained
from scratch on our own corpus.

## 2. Why this shape

### 2.1 The 100–200M evidence

| model | params | layers | width | heads / KV | FFN | vocab | ctx | tokens |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| GPT-2 124M | 124M | 12 | 768 | 12 / 12 | 3072 | 50k | 1k | ~10B |
| OPT-125M | 125M | 12 | 768 | 12 / 12 | 3072 | 50k | 2k | 180B |
| Pythia-160M | 160M | 12 | 768 | 12 / 12 | 3072 | 50k | 2k | 300B |
| Cerebras-GPT-111M | 111M | 10 | 768 | 12 / 12 | 3072 | 50k | 2k | 2.2B |
| SmolLM-135M | 135M | 30 | 576 | 9 / 3 | 1536 | 49k | 2k | 600B |
| **SmolLM2-135M** | 135M | 30 | 576 | **9 / 3** | 1536 | 49k | **8k** | **2T** |
| MobileLLM-125M | 125M | 30 | 576 | 9 / 3 | 1536 | 32k | 2k | 1T |
| **MobileLLM-R1-140M** | 140M | 15 | 576 | **9 / 3** | **2048** | **128k** | 4k/32k | **4T** |
| Mamba-130M | 130M | 24 | 768 | SSM | — | 50k | — | 300B |
| **KALIA v150** | **149M** | **34** | **576** | **9 / 3** | **1536** | **50k** | **8k** | **1T** |

The evidence is consistent:

- **Deep and thin beats wide and shallow at this scale.** The three strongest
  125–140M models use 576 width and 15–30 layers, not 768 × 12.
- **GQA 9/3 is the small-model standard.** It cuts KV cache 3× versus MHA with
  no measured quality loss at this scale.
- **FFN 1536 (4× width) is the proven default.** MobileLLM-R1 spends more on
  FFN (2048) and less on depth (15), but also uses a 128k vocab that consumes
  most of its parameter budget.
- **The leaders are overtrained.** 1T–4T tokens, not 2.4B. The data budget, not
  the shape, is the dominant gap between a compute-optimal 150M and a
  competitive 150M.
- **50k vocab is the efficient choice.** A 128k vocab at 150M would spend
  ~73.7M parameters on embeddings alone, leaving ~76M for the transformer.
  That is a reasoning/code trade, not a general-purpose base trade.

### 2.2 Why 34 layers and not 30

30 × 576 is the proven SmolLM2/MobileLLM shape at 135M. 34 × 576 is the same
family at 149M: it adds 13% parameters in depth, where the small-model evidence
says depth is productive. It is a small, evidence-based extension, not a new
architecture.

### 2.3 Why not 512 × 38

512 × 38 is the exact v0.3.1 model scaled. It is deep and narrow, uses **MHA
8/8** (3× larger KV cache), **1408 FFN**, and **1k context**. It is a clean
scale control, but it is not the best 150M. It also inherits the 58M model's
interface instead of choosing the best shape.

### 2.4 Why not 640 × 24 or 768 × 16

640 × 26 with GQA 10/5 and FFN 1728 is ~150M and is a better phone shape on
paper (fewer layers, faster decode). But 576 × 30 is the only shape with
multiple independent 100–200M validations, and 576 keeps the width close to the
existing 4×512 multimodal interface. 768 × 16 is too shallow for the evidence.
A 640 × 26 proxy is a **shape-sweep arm**, not the baseline.

### 2.5 Why not MoE, MLA, or a hybrid baseline

- **MoE** needs enough active parameters to amortize routing. At 150M total it
  is overhead, not capacity, and it complicates phone deployment.
- **MLA** compresses KV cache for 32k–1M context. At 8k, GQA 9/3 is enough.
- **Gated DeltaNet + attention** is promising for long context and phone KV
  cache, but there is no 150M-scale quality proof at 1T tokens. It is the first
  architecture arm, not the baseline.

The large-model ideas worth taking at 150M are **training** ideas, not
parameter-count ideas: Muon, QK-Norm, QK-Clip, MTP, multi-stage data, and
careful annealing.

## 3. Parameter budget

| component | parameters | share |
|---|---:|---:|
| token embedding (tied with LM head) | 28,948,032 | 19.4% |
| 34 transformer blocks | 120,367,616 | 80.6% |
| final RMSNorm | 576 | 0.0004% |
| **total** | **149,316,224** | **100%** |

Per block:

| component | parameters |
|---|---:|
| RMSNorm 1 | 576 |
| QKV projection (9 + 2×3 heads × 64) | 552,960 |
| QK-Norm (2 × 64) | 128 |
| attention output projection | 331,776 |
| RMSNorm 2 | 576 |
| SwiGLU (gate + up + down) | 2,654,208 |
| **per block** | **3,540,224** |

KV cache, bf16:

- **25.5 KiB per token** (2 × 3 KV heads × 64 × 34 layers × 2 bytes).
- **51 MiB at 2k context**, **204 MiB at 8k**.
- MHA 9/9 would be 3× larger: **612 MiB at 8k**.

Compute:

- forward: **298.6 MFLOPs/token** (2N);
- training: **895.9 MFLOPs/token** (6N);
- 1T tokens: **8.96 × 10²⁰ FLOPs** (0.896 ZFLOPs).

## 4. Components and provenance

Every component is implemented from scratch. Published ideas are cited;
no external weights, no distillation, no borrowed inference.

| component | inspiration | status | role at 150M |
|---|---|---|---|
| RMSNorm | Zhang & Sennrich 2019 | in `model.py` | standard pre-norm |
| SwiGLU | Shazeer 2020 | in `model.py` | FFN |
| RoPE | Su et al. 2021 | in `model.py` | position |
| GQA 9/3 | Ainslie et al. 2023; SmolLM2 | in `model.py` (`n_kv_head`) | 3× smaller KV cache |
| QK-Norm | Dehghani et al. 2023; Qwen3 | in `model.py` (`qk_norm`) | attention stability |
| logit soft-cap 30 | Gemma 2 | in `model.py` (`logit_softcap`) | output stability |
| tied embeddings | Press & Wolf 2017; SmolLM2 | in `model.py` | saves 29M params |
| Muon | Jordan et al. 2024; Moonlight | in `optim.py` | hidden-matrix optimizer |
| QK-Clip / MuonClip | Kimi K2 (2025) | **to implement** | attention-logit bound |
| head-wise Muon | DeepSeek-V4.1-Flash (2026) | **to implement** | per-head preconditioner |
| MTP | DeepSeek-V3 (2024) | **to implement** | training-only arm |
| multi-stage data | SmolLM2 (2025) | **to build** | 1T-token curriculum |
| Engram | Cheng et al. 2026; DeepSeek-V4.1-Flash | **arm only** | conditional memory |
| Gated DeltaNet | Yang et al. 2024; Qwen3-Next | **arm only** | long-context hybrid |
| RLVR / GRPO | DeepSeek-R1; MiMo | **probe only** | negative at 135M in our record |

### 4.1 Correction to the earlier chat spec

The earlier chat listed MTP, head-wise Muon, QK-Clip, and RLVR as if they were
part of the baseline. They are not. The evidence says:

- **MTP** is not used during backbone pretraining in DeepSeek-V4.1-Flash; it is
  a training-only ablation at 150M until a 20B-token probe shows a gain.
- **Head-wise Muon + QK-Clip** are stabilizers for large models with exploding
  attention logits. At 150M, QK-Norm already bounds them. They are an ablation,
  not a default.
- **RLVR** was **negative** in our own 135M probe: GRPO reduced GSM8K exact
  match from 1.82% to 1.59% (L20-Edu-135M, arXiv 2606.22189). It is a
  documented post-training probe, not a pretraining ingredient.

This is the difference between "the best-sounding spec" and "the best
defensible spec."

## 5. Training recipe

### 5.1 Baseline

- Random init, normal(0, 0.02); residual output projections scaled by
  `0.02 / sqrt(2 × n_layer)`.
- **Muon** on hidden 2D matrices: LR 0.02, momentum 0.95, weight decay 0.0,
  ns_steps 5.
- **AdamW** on embeddings, norms, and any 1D parameters: LR 6e-4, betas
  (0.9, 0.95), weight decay 0.1.
- Gradient clip 1.0.
- **WSD schedule**: 2,000 warmup steps, stable phase to 80% of tokens, linear
  decay to 10% over the final 20%. WSD is chosen over cosine because the data
  mixture changes at stage boundaries and the run may be extended.
- **Batch**: 2,097,152 tokens/step (micro 2 × accum 16 × 8,192 × 8 GPUs).
  About 476,838 steps for 1T tokens; stage step counts are computed from tokens,
  not fixed in advance.
- **Precision**: bf16 preferred. `train.py` currently uses fp16 autocast +
  GradScaler; a bf16 path is a prerequisite for the full run.
- **Checkpoint**: every 30 minutes, keep last 3 + best; 149M fp32 weights are
  ~596 MB, optimizer state ~2–3 GB.

### 5.2 Staged context and data

| stage | tokens | context | mixture emphasis |
|---|---:|---:|---|
| A | 300B | 2k | broad web, code, math; high LR |
| B | 400B | 4k | more code/math; long documents |
| C | 250B | 8k | long-context and code/math |
| D | 50B | 8k | anneal: highest-quality web, math, code, reference |

The 2k → 4k → 8k staging is a compute and data-efficiency choice, not a
different architecture. RoPE theta 100k covers 8k; a later 32k extension would
use YaRN or a theta increase and is a separate arm.

### 5.3 What is deliberately not in the baseline

- No MoE, MLA, or hybrid attention.
- No native multimodal pretraining; the vision/audio adapters are re-fit after
  the text base.
- No 1M context.
- No teacher-generated synthetic text.
- No RLVR in pretraining.
- No depth expansion of v0.3.1.

## 6. Data plan

### 6.1 Tokenizer

GPT-2 byte-level BPE, 50,257 tokens, tied embeddings. Reasons:

- clean comparison with v0.3.1 and the existing 4×512 multimodal interface;
- 50k is the efficient vocabulary at 150M;
- a 32k mobile tokenizer and a 128k reasoning/code tokenizer are later arms,
  not the general base.

### 6.2 Sources (English-first, permissive)

| source | share | license | use |
|---|---:|---|---|
| FineWeb-Edu | 62% | ODC-BY | high-quality educational web |
| DCLM-Baseline | 15% | CC-BY-4.0 | broad filtered web |
| The Stack v2 (permissive files only) | 15% | per-file, permissive | code |
| OpenWebMath + FineMath + procedural | 6% | ODC-BY + ODC-BY + ours | math |
| Wikipedia + public-domain books | 2% | CC-BY-SA / public domain | reference |

Licenses are checked per source before tokenization. The Stack v2 is filtered to
permissive per-file licenses; no file with an unclear license enters the run.
The math/procedural shard is generated by rules with verifiable answers, not by
a teacher model.

**No Cosmopedia or teacher-generated text in the clean run.** Cosmopedia is
model-generated; using it would blur the no-distillation claim. If it is used
at all, it is a disclosed arm, never the clean baseline.

Procedural math/code data is generated by rules with verifiable answers, not by
a teacher model.

### 6.3 Cleaning and decontamination

- Exact line dedup, URL dedup, MinHash LSH near-dedup (5-gram, Jaccard 0.8).
- Quality filters: language ID, repetition, boilerplate, toxicity/PII where
  required by the license.
- Benchmark decontamination: 13-gram overlap removal against the 9-task suite
  and any reported eval set.
- License audit per source; keep a manifest of URL, license, and filter
  version. The repo already has `audit_licences.py` and `docs/legal/`.
- Target **1.2T raw tokens** to yield **1T unique training tokens**.

### 6.4 Packing

Pack documents with EOS and a document-level attention mask so a token cannot
attend across document boundaries. This matters for code and math, where
cross-document leakage can inflate training loss quality.

### 6.5 How much is actually available on Hugging Face

| dataset | tokens | license | role |
|---|---:|---|---|
| FineWeb-Edu | **1.3T** | ODC-BY | primary web |
| DCLM-Baseline | **3.8T** | CC-BY-4.0 | primary web |
| FineWeb | 18.5T | ODC-BY | fallback / scale |
| FineWeb-2 | 3.34T | ODC-BY | multilingual arm |
| Dolma | 3T | ODC-BY | mix / fallback |
| RedPajama-v2 | 30T | check per source | fallback / scale; text hosted by Together, HF repo has metadata/subsets |
| The Stack v2 (filtered) | **~0.9T** | per-file permissive | code |
| OpenWebMath | 14.7B | ODC-BY | math |
| FineMath | 34B (3+) / 54B | ODC-BY | math |
| The Pile | 300B | other | not for public release |

These overlap: FineWeb, FineWeb-Edu, DCLM, Dolma, RedPajama and the Pile all
draw from Common Crawl. After deduplication, the usable **unique high-quality
pool is roughly 5–8T tokens**, not the raw multi-tens-of-trillions sum. That is
still **5–8× the 1T run** and **1.5–2× a 4T run**.

For the planned 1T mix:

- 620B FineWeb-Edu = 48% of that dataset;
- 150B DCLM = 4% of that dataset;
- 150B code = 17% of the filtered Stack v2;
- 60B math = OpenWebMath 14.7B + FineMath 34B + ~11B procedural;
- 20B reference = Wikipedia + public-domain books + CC-BY textbooks.

The 1T run uses less than half of FineWeb-Edu and a small fraction of DCLM, so
**2T–4T is a compute/budget decision, not a data-availability decision.** The
reference shard is the tightest: if public-domain books fall short, shift 1% to
web and use a 63/15/15/6/1 mix.

Storage and download:

- 1T tokens as uint16 = **2TB** tokenized;
- 1.2T raw tokens ≈ **4.8TB** text at 4 bytes/token;
- FineWeb-Edu full is **6.2TB** on HF; DCLM-Baseline is **7.2TB**;
- working space with raw shards, tokenized bins and checkpoints: **10–15TB**;
- at 1 Gbps, 6TB is ~13 hours; at 100 MB/s, ~17 hours;
- tokenization + dedup + decontamination: roughly **1–3 days on 32–64 cores**,
  depending on the pipeline.

So the answer is: **Hugging Face can supply 1T tokens easily, 2T comfortably,
and 4T with careful dedup and mixing. The real limit is GPU compute, not data.**

## 7. Compute plan

1T tokens at 6ND is **0.896 ZFLOPs**. Real throughput depends on hardware and
MFU:

| hardware | MFU | wall time | GPU-hours |
|---|---:|---:|---:|
| 8×A100 40GB | 30% | ~332 h (13.8 d) | ~2,656 |
| 8×A100 40GB | 40% | ~249 h (10.4 d) | ~1,992 |
| 8×H100 80GB | 40% | ~79 h (3.3 d) | ~632 |
| 8×H100 80GB | 50% | ~63 h (2.6 d) | ~506 |

Add ~20% for eval, checkpointing, and restarts. **Kaggle's ~30 GPU-h/week cannot
train 1T tokens**; it can run a 5–10B-token pilot. The full run needs TRC or
cloud credit. This is the real constraint, not the architecture.

Phases:

| phase | tokens | purpose | approximate cost |
|---|---:|---|---:|
| P0 shape smoke | 1B | param count, throughput, stability | 2–4 GPU-h |
| P1 recipe ablation | 20B | MTP vs no MTP; Muon vs head-wise Muon | 40–60 GPU-h |
| P2 full pretrain | 1T | the model | 2,000–2,700 A100 GPU-h |
| P3 anneal | 50B | high-quality decay | 100–150 GPU-h |
| P4 adapters/eval | — | captioner, diffusion, edit, audio | 50–100 GPU-h |

## 8. Arms

| arm | change | decision rule |
|---|---|---|
| **A0 baseline** | 576×34, Muon, no MTP | the control |
| **A1 MTP** | +1-token auxiliary head | use only if 20B probe gives ≥1% val gain |
| **A2 head-wise Muon + QK-Clip** | per-head Q/K preconditioner, logit clip | use only if it improves stability or throughput at 20B |
| **A3 Engram** | 50M–100M conditional n-gram memory | capability pack; not part of the 149M base |
| **A4 Gated DeltaNet hybrid** | replace 1/4–1/3 attention layers | long-context arm; 32k+ only |
| **A5 2T–4T tokens** | extend the run | the biggest lever if compute allows |
| **A6 640×26 shape** | 1B-token proxy | only if it beats 576×34 at equal tokens |
| **A7 RLVR** | GRPO on verifiable math/code | documented probe; prior at 135M is negative |

The baseline is the only run that can become the public base. Arms are
pre-registered separately and cannot be mixed into the baseline after the fact.

## 9. What would actually be better

This is the honest ceiling, not a sales pitch.

1. **More tokens.** 2T–4T curated tokens is a larger expected gain than any
   architecture change at 150M. SmolLM2 used 2T; MobileLLM-R1 used 4T. If the
   1T cap moves, extend the run before touching the shape.
2. **Better data.** FineWeb-Edu/DCLM quality, code/math balance, dedup, and
   decontamination matter more than another normalization layer.
3. **A real shape search.** 576×34 is the best evidence-based prior, not a
   proven optimum. A 1B-token sweep over width/depth/FFN/vocab/context would
   turn the prior into a measurement.
4. **A hybrid long-context arm.** Gated DeltaNet + attention could beat dense
   attention at 32k+ and cut phone KV cache. At 8k general text, it is unproven.
5. **A larger model.** A 1B–3B dense or a 3B-total/0.5B-active MoE would beat
   any 150M on capability. It would not be phone-friendly. That is the real
   trade, not a trick inside the 150M.

At 150M and 1T tokens, the realistic target is **SmolLM2-135M-class**, not
frontier reasoning. SmolLM2-135M reports HellaSwag 42.1, ARC avg 43.9, PIQA
68.4, MMLU 31.5, OpenBookQA 34.6, WinoGrande 51.3. Beating those would be a
strong result for a from-scratch model on constrained compute. Matching them
would already be a real advance over v0.3.1.

## 10. Release gates

The public base stays **v0.3.1** until all of these pass:

- **G1 standard 9-task mean ≥ 0.468** (+10% over 0.4258).
- **G2 common held-out loss**: report on the v0.3.1 held-out set and on a new
  v150 held-out set; do not compare losses across different tokenizers/data
  without the common set.
- **G3 capability regression**: captioner held-out ≥ 0.99; see→draw→see ≥ 0.90;
  self-verify ≥ 0.95; constraints ≥ 0.98; edit compiler within 2 pp; audio
  within 2 pp.
- **G4 provenance**: no external weights, no distillation, no teacher-generated
  text; license manifest and data card complete.
- **G5 reproducibility**: config, commit, data manifest, and eval script
  anchored in `docs/preregistrations/LEDGER.md`.

No release if any gate fails. A negative result is logged, not tuned away.

## 11. Open questions

1. Is 1T the hard cap, or can the run extend to 2T–4T if compute is available?
2. Do we have the storage and CPU budget for 1.2T raw tokens (~2–4 TB
   tokenized plus raw shards)?
3. Which compute provider clears first: TRC, cloud credit, or a sponsored
   cluster?
4. Does the 4×512 vision interface need a 512→576 projection, or do we re-fit
   the captioner decoder at 576?
5. Is a 32k mobile tokenizer worth a separate arm, or does it fragment the
   capability packs?

## 12. Provenance statement

> KALIA v150 is a 149,316,224-parameter decoder trained from random
> initialization on a permissively licensed, decontaminated corpus. Its
> architecture uses published components—RMSNorm, SwiGLU, RoPE, GQA, QK-Norm,
> logit soft-cap, Muon—implemented from scratch and cited. It uses no external
> model weights, no distillation, no teacher-generated text, and no borrowed
> inference. The v0.3.1 58M model remains the public base until v150 clears the
> registered gates.

## References

- Ben Allal et al., *SmolLM2: When Smol Goes Big*, arXiv:2502.02737, 2025.
- Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for
  On-Device Use Cases*, arXiv:2402.14905, 2024.
- Zhao et al., *MobileLLM-R1: Exploring the Limits of Sub-Billion Language
  Model Reasoners with Open Training Recipes*, arXiv:2509.24945, 2025.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints*, arXiv:2305.13245, 2023.
- Shazeer, *GLU Variants Improve Transformer*, arXiv:2002.05202, 2020.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*,
  arXiv:2104.09864, 2021.
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, arXiv:1910.07467,
  2019.
- Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters*,
  arXiv:2302.05442, 2023 (QK-Norm).
- Jordan et al., *Muon: An Optimizer for Hidden Layers in Neural Networks*,
  2024.
- Kimi Team, *Kimi K2: Open Agentic Intelligence*, arXiv:2507.20534, 2025
  (MuonClip / QK-Clip).
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437, 2024 (MTP).
- DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context
  Intelligence*, arXiv:2606.19348, 2026 (mHC, Muon, long context).
- DeepSeek-AI, *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache
  Compression*, arXiv:2609.19969, 2026 (CED, CSA2, Engram, head-wise Muon).
- Qwen Team, *Qwen3 Technical Report*, arXiv:2505.09388, 2025 (QK-Norm, GQA).
- Qwen Team, *Qwen3-Next-80B-A3B* model card, 2025 (Gated DeltaNet hybrid).
- GLM Team, *GLM-4.5: Agentic, Reasoning, and Coding Foundation Models*,
  arXiv:2508.06471, 2025.
- Xiaomi LLM-Core, *MiMo: Unlocking the Reasoning Potential of Language Model*,
  arXiv:2505.07608, 2025.
- L20-Edu-135M, arXiv:2606.22189, 2026 (negative RLVR at 135M).
- Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*,
  arXiv:2412.06464, 2024.
