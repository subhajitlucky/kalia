# KALIA: a 58M language model trained from scratch on free GPUs — the complete build log

> अथ शब्दानुशासनम् — *"Now begins the discipline of words."*

In two days, using only free-tier Kaggle GPUs (30 hours/week, 2× NVIDIA T4), I
trained a 58M-parameter language model from **random initialization** to
coherent story generation — no pretrained weights, no distillation, no
fine-tuning, zero dollars of compute. This is the complete record: every
experiment, every number, every decision, every mistake.

**Headline results**

| | |
|---|---|
| Model | 57,856,256 params, 10 layers × 512 dim, ctx 1024, GPT-2 BPE |
| Training | 1.82B tokens seen; stopped at step 3,478 of 4,770 (quota; plateau documented) |
| Held-out loss | **2.4366** / **0.8184 bits-per-byte** (deterministic, 819k tokens) |
| Zero-shot | PIQA **61.4%** · ARC-Easy 45.8% · HellaSwag 36.8% · WinoGrande 50.2% |
| vs baseline | overtook the AdamW baseline's *final* loss with **~23% fewer tokens** |
| Compute | ~20 GPU-hours total (free tier) |

- Code + full engineering journal: https://github.com/subhajitlucky/kalia
- Weights + model card: https://huggingface.co/kalia-lm/kalia-v012

---

## 1. The goal, and why from scratch

Fine-tuning forks someone else's brain. I wanted a model where I could account
for every byte: the corpus mixture, the tokenizer, the architecture, the
optimizer, the exact run. That costs capability — this is a storyteller, not an
assistant — and buys an auditable artifact. Every weight in KALIA exists
nowhere else.

The constraint was equally deliberate: **free compute only**. Kaggle's free
tier gives 30 GPU-hours per week on 2×T4 (32GB total), with background
"commit" runs and persistent notebook output. The whole system had to survive
that: 8.5-hour session caps, weekly quota, resumable checkpoints, and no
secrets in API-triggered runs.

The second goal shaped everything: **make the process verifiable**. Every
experiment was pre-registered before it ran, with a SHA-256 hash in a public
ledger. Every decision is numbered. Every incident is written down, including
the embarrassing ones.

## 2. Day 1 — build the machine, then measure

**Platform decision, by evidence.** Kaggle over Colab/Lightning: documented
30 h/week, 2×T4, commit runs, 20GB persistent output, no credit card. This
decision was made once and never revisited.

**The codebase** (13 tasks, test-driven, 20 tests at first green build):
a decoder-only transformer (RMSNorm, SwiGLU, RoPE, weight tying), a
memory-mapped token dataset, a resumable DDP trainer with HuggingFace
checkpoint sync, a sampler, and the Kaggle notebooks.

**The data** was tokenized on free CPU (no quota cost): TinyStories
(474M tokens) + FineWeb-Edu (2.0B) = **2.46B train + 10M val tokens**, GPT-2
BPE, uint16 shards.

**Micro-ablations before any expensive run.** Each arm: 30M params, 50M
tokens, identical seed, ~30 minutes. Step-700 validation loss:

| Arm | Val loss |
|---|---|
| AdamW | 3.8041 |
| Muon | 3.5937 (−0.21) |
| Muon + QK-Norm + logit soft-cap | **3.5103** (−0.29) |

The ordering held at every checkpoint — not noise. Only then did the two
full-scale runs launch (58M params, 2.5B tokens planned). After Day 1:
the AdamW baseline at 3.2702 (step 2,250) and the new recipe at 3.2214
(step 1,738) — **already past the baseline's final loss with 23% fewer
tokens**.

## 3. Day 2 — discipline under pressure

**A promising result was rejected.** Muon+ (one post-polar normalization step)
beat plain Muon on 7 of 7 checkpoints — but by −0.015 nats, below the 0.02
promotion threshold we set in advance. It did not ship. The threshold exists
to stop wishful promotion, and it was enforced against our own favorite.

**LR sweep:** 0.015 → 3.4943 · **0.02 → 3.4941** · 0.03 → 3.5027 ·
0.06 → 3.5380. 0.02 confirmed; further tuning has no headroom.

**Architecture ablation** (763 steps, same data/seed):

| Arm | Params | Val @ 700 |
|---|---|---|
| control (Muon+QK-Norm) | 29,921,280 | **3.4924** |
| looped depth (2× passes, shared weights) | 29,921,280 | 3.5012 |
| thin & deep (10L × 320) | 28,787,840 | 3.6969 |
| grouped-query attention | 28,741,632 | 3.5031 |

Nothing beat control by the pre-set margin, so the architecture stayed.
Depth-thinning hurt (−0.2); looping and GQA were quality-neutral — costs
without benefits at this scale. One finding survived: the looped model ran
**1.35× slower per step, not 2×**, because reused weights stay cache-hot.

**The training run told an honest story.** Validation loss plateaued for
1,000 steps (2.53–2.63) while training loss kept falling. Then the cosine
decay bit: 2.3986 at step 3,000. Then the weekly quota ran out at step 3,478
of 4,770 (73%). A deterministic high-precision eval (100 batches, 819,200
tokens) settled the question the noisy evals couldn't: **2.4366 loss /
0.8184 bpB** — no regression, the swings were noise. Under our pre-registered
stopping rule, the plateau plus quota exhaustion made the stop final.

**Evaluation beyond loss.** A frozen 27-prompt probe suite, a held-out
sentence set, and two custom metrics:

- Probe held-out loss **3.2303** / bpB 0.9415
- **Abhimanyu gap: 6.06 nats** — reversed-text NLL (9.29) minus forward NLL
  (3.23). The model can enter fluent text but cannot exit it. Random would be
  ~10.8. This motivated a pre-registered experiment (chunk-preserving reversal
  training, "X16") that runs next.
- Zero-shot benchmarks via lm-evaluation-harness, 0-shot, 500 samples:
  **PIQA 61.4%** · ARC-Easy 45.8% · HellaSwag 36.8% (norm) · WinoGrande 50.2%
  · LAMBADA 23.0% acc / ppl 194. For context, third-party tables list
  OPT-125M (2× the parameters, ~160× the training tokens) at PIQA 63.0 /
  ARC-Easy 43.5 / HellaSwag 29.2 — a fair scale reference, not a head-to-head.

**Compliance-clean data for v0.2.0.** The v2 corpus (2.4B tokens: FineWeb-Edu
60 / TinyStories 20 / Cosmopedia 15 / permissively-licensed Python 5) was
rebuilt with a per-file license filter after discovering the first build's code
shard had no license metadata. Nothing unclear-licensed will ever train a
public model.

## 4. Every decision (D1–D41)

| # | Decision | Outcome |
|---|---|---|
| D1 | Kaggle as the platform (30 h/week, 2×T4) | held |
| D2 | From-scratch only; no fine-tuning | held |
| D3 | ~58M params, 10L×512d, ctx 1024 | held |
| D4 | TinyStories + FineWeb-Edu corpus | held |
| D5 | Incremental version ladder, one change per version | held |
| D6 | Micro-ablation before every full run | held |
| D7 | Pre-register experiments with hashed predictions | held |
| D8 | Journal every session, 3-source research protocol | held |
| D9 | micro-batch 8 × accum 32 (OOM fix at equal effective batch) | held |
| D10 | Browser-started runs only (API runs can't read secrets) | held |
| D11 | Freeze v0.1.0 after one session as the equal-token baseline | held |
| D12 | Micro-ladder before full runs (30M/50M tokens ≈ 30 min/screen) | held |
| D13 | Journal everything; three independent sources per claim | held |
| D14 | Implement Muon+ (E1) | validated, below threshold |
| D15 | E1 (−0.015) below 0.02 threshold → do not auto-promote; test LR interaction | enforced |
| D16 | Fix resume race: single downloader + atomic swap + barrier | fixed |
| D17 | Expert reprioritization: LR sweeps first, recipe lock, bpB, data mix | held |
| D18 | Finish v0.1.2 instead of abandoning at 71% | held |
| D19 | v2 mixture: 60/20/15/5 (FineWeb-Edu/TinyStories/Cosmopedia/code) | built |
| D20 | "Punch above weight": token efficiency, data quality before scale | held |
| D21 | **Distillation rejected** — KALIA stays pure from-scratch | closed |
| D22 | Adopt test-time-training / self-teaching research track | queued |
| D23 | Invention target: entity memory + in-place TTT at 58M | queued |
| D24 | Looped depth promoted to front of queue | tested → rejected |
| D25 | **RL self-play and DSA/MoE rejected at this scale, on evidence** | closed |
| D26 | Continual-learning recipe for post-training stage | planned |
| D27 | Publishing policy: minimum footprint, honest only, no data redistribution | enforced |
| D28 | Rebuild v2 code shard with license filter (`prep-v2b`) | done |
| D29 | Evaluation harness (probe set, bpB, comparison tool) | done |
| D30 | Reasoning-model policy: public methods, our own weights, no RLVR yet | held |
| D31 | Freeze Muon LR at 0.02; sweep complete | closed |
| D32 | Three ancient-text-inspired designs (Kautilya mixture, Utsarga schedule, Apoha-ELECTRA) | queued |
| D33 | Four more (Jata-patha reversal = X16, Samyama loop scaling, Rasa conditioning, Anekantavada soup) | queued |
| D34 | Three Veda-derived designs (self-knowledge head, format conditioning, procedural subset) | queued |
| D35 | Gita compilations; prior art cited (guna controller adapted, not invented) | queued |
| D36 | Entity-consistency harness; **narrowed** the entity-memory novelty claim (prior art exists) | active |
| D37 | Chakravyuha dismissal corrected; Abhimanyu-gap metric built; X16 redesigned | active |
| D38 | Hash-anchored pre-registration (J1) and pre-registered stopping (J2) adopted | active |
| D39 | Strategic audit: finish → X16 → publish → v0.2.0; freeze experiment queue | held |
| D40 | Quota-exhaustion response: X16 deferred to reset; publish assets now | superseded |
| D41 | **Stop v0.1.2 at step 3,478** (converged within noise; quota saved for v0.2.0) | done |

## 5. Every incident (I1–I12)

| # | Incident | Root cause → fix |
|---|---|---|
| I1 | API-run couldn't read secrets | Secrets are per-notebook UI only → browser-started runs + fail-fast test |
| I2 | Config file missing in first GPU run | Dataset upload skips subdirectories → flatten configs, assert existence |
| I3 | OOM at step 1 | fp32 logits > T4 memory → micro-batch 8×32 + expandable segments |
| I4 | Failed subprocess marked COMPLETE | `!` doesn't fail cells → `subprocess.run` + assert returncode |
| I5 | Repo hygiene (env-specific wording, mixed authorship) | Normalized docs and commit identity |
| I6 | Resume crash (`EOFError`) | Both DDP ranks wrote/read the same file → single downloader, atomic swap, barrier |
| I7 | Latent crash on older checkpoints | Optimizer `load_state_dict` dropped new keys → defaults restored + regression test |
| I8 | Log history lost across sessions | Resume didn't pull logs → pull-and-append on resume |
| I9 | Mix step failed after 2h tokenization | Relative paths after `cd` → absolute paths; mix-only kernel reuses shards |
| I10 | Wrong dataset mount path + stale code | `/kaggle/input/datasets/<owner>/<slug>/` → corrected globs, re-versioned dataset |
| I11 | Misreported a 44-min run as ">5h" | Wall-clock guessing instead of run-start times → use `lastRunTime`; correction published |
| I12 | Weekly GPU quota exhausted before the planned experiment | Quota accounting; the UI's "24h remaining" was misleading → defer to reset, re-plan |

## 6. The mistakes that stayed in

- **Muon+ not promoted** despite winning 7/7 checkpoints — the threshold was
  set first, applied second.
- **Architecture search returned nothing** — looped depth and GQA were
  quality-neutral; we published the negatives instead of hiding them.
- **A documentation error was caught late**: our own docs described v0.1.2 as
  "Muon+" for hours; the config proved it was plain Muon. Every public text
  was corrected, and the hashed v0.2.0 pre-registration got a registered
  amendment rather than a silent edit.
- **The plateau** — 1,000 steps of no movement, documented instead of
  massaged, then resolved honestly by a deterministic eval.

## 7. What's next

- **X16** (pre-registered, hash-anchored): does chunk-preserving reversal
  training close the 6-nat Abhimanyu gap without hurting forward loss?
- **v0.2.0**: new data lineage (compliance-clean v2 corpus) plus only changes
  that pass pre-registered promotion rules — the rules were hashed *before*
  the results existed.
- Post-training, quantization, and a local chat demo are on the roadmap.

## 8. Reproduce it

```bash
git clone https://github.com/subhajitlucky/kalia && cd kalia
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -m pytest tests/ -v          # 64 tests
```

Data prep, training, and evaluation run on free Kaggle notebooks; the repo's
`notebooks/` folder contains every one of them. Pre-registration hashes are in
`docs/preregistrations/LEDGER.md`. The full journal, all 41 decisions, and all
12 incidents are in `docs/`.

*KALIA is a personal research project. Not affiliated with any government
scheme, company, or other project using a similar name.*
