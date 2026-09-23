# Technique Backlog (research queue)

Sources checked 2026-09-23. Each item: what, evidence, expected gain at our
scale, risk, status. Protocol: micro-ablation (30M params, 50M tokens, step-700
val loss) against the current best recipe (Muon + QK-Norm + soft-cap = 3.5103),
then a full run only for winners.

## Punch-above-weight queue (see docs/research/2026-09-23-punching-above-weight.md)

| ID | Experiment | Cost | Status |
|---|---|---|---|
| X1 | Deep-thin vs wide (12L×384d vs 6L×512d) | 1.5h GPU | queued |
| X2 | GQA (2 KV heads) vs MHA | 1.5h GPU | queued |
| X3 | Block sharing (repeat each block) | 1.5h GPU | queued |
| X4 | Code share 5% → 10% in mixture | 1h GPU | queued |
| X5 | WSD LR retune | 1h GPU | queued |
| X6 | Distillation probe from SmolLM2-1.7B | 3–4h GPU | decision pending |

## Optimizer tier (strongest evidence at our exact scale)

| ID | Technique | Source | Evidence | Expected | Risk | Status |
|---|---|---|---|---|---|---|
| E0 | **Learning-rate sweep for Muon+** (0.015 / 0.02 / 0.03 / 0.06) | D17 protocol | LR dominates algorithmic tricks | High | None | 0.02 done (3.4941), 0.03 running (E1b), 0.015/0.06 queued |
| E1 | **Muon+**: one normalization step after polar orthogonalization | arXiv 2602.21545 | Beats Muon, NorMuon, AdaMuon, Turbo-Muon across GPT/LLaMA 60M–7B; up to 37% pretraining speedup; zero extra state | High | Very low (one line) | **validated** (−0.015 nats, 7/7 checkpoints, matches paper's 60M effect) |
| E1b | Muon+ at higher Muon LR (paper's main lever is LR tolerance) | arXiv 2602.21545 | Muon+ stays stable where Muon degrades; our ablation used a fixed lr=0.02 | Unknown, likely + | Low | queued |
| E2 | **Polar Express** coefficients for Muon iterations | Amsel et al. 2025 (arXiv 2505.16932) | Better polar approximation; complementary to E1 | Medium | Low | queued |
| E3 | **Cautious weight decay** | Chen et al. 2025 | +1.3% in speedrun at small scale; also successful in nanochat | Medium | Low | queued |
| E4 | **Checkpoint EMA** during decay phase | IMU-1 (arXiv 2602.02522) | Consistent final-loss boost for free | Medium | Very low | queued |
| E5 | **NorMuon** (neuron-wise normalization) | Li et al. 2025 (arXiv 2510.05491) | +11% over Muon at 1.1B; but newer U-NorMuon/Aurora claim better | Medium | Medium (extra state) | queued after E1 |
| E6 | **cubic5** orthogonalization (cheaper NS) | arXiv 2606.00371 | Parity with quintic-5 on GPT-2 Small; 10 vs 15 matmuls | Speed only | Medium | later |

## Architecture tier (IMU-1 stack for small models)

| ID | Technique | Source | Evidence | Expected | Risk | Status |
|---|---|---|---|---|---|---|
| E7 | **Value residual connections** | Zhou et al. 2024; IMU-1 | Better gradient flow; part of 5.2% combined gain | Medium | Low | queued |
| E8 | **Per-head attention gating** | Qiu et al. 2025; IMU-1 | Mitigates attention sinks; improves expressivity | Medium | Low | queued |
| E9 | **LayerNorm scaling** | IMU-1 | Addresses depth-related pathologies | Small | Very low | queued |
| E10 | **ReLU² vs SwiGLU** | modded-nanogpt | Cheaper kernel, neutral-to-positive loss | Small | Very low | queued |

## Schedule / data tier

| ID | Technique | Source | Evidence | Expected | Risk | Status |
|---|---|---|---|---|---|---|
| E11 | **WSD schedule** + batch-size warmup | Kimi K2, GLM-4.5, IMU-1 | Flexible decay timing; enables data-mix changes mid-run | Medium | Low | queued |
| E12 | **Story→facts curriculum** | Xiaomi MiMo 3-stage mixture | Reasoning density + staged mixtures help | Medium | Low | planned (v0.1.4) |
| E13 | **End-of-training data annealing** | Kimi K2 (400B-token anneal) | Gains at fixed budget | Medium | Low | queued |

## RSI / self-improvement tier (honest assessment included)

| ID | Technique | Source | Evidence | Expected at 58M | Status |
|---|---|---|---|---|---|
| R1 | **Self-training loop**: model generates stories → filter (dedup, repetition, grammar heuristics, loss-under-EMA-teacher) → continue training on filtered self-data | STaR/self-distillation lineage | Works at scale when filters are strong; untested at 58M stories | Uncertain — could help fluency, risks mode collapse | probe after v0.1.2 finishes |
| R2 | **Tiny RLVR probe** (verifiable format/arithmetic tasks) | L20-Edu-135M | **Negative**: GRPO reduced GSM8K accuracy at 135M (1.82% → 1.59%) | Likely negative; run once for the record | queued as science |
| R3 | **EMA-teacher self-distillation during pretraining** (student = model, teacher = EMA of itself) | Mean Teacher (Tarvainen & Valpola 2017) applied to LM pretraining — rarely validated at our scale | Not established | Could add a free consistency signal | originality candidate |
| R4 | Automated experiment agent loop (agent proposes config → Kaggle run → parse → next) | practitioner automation | Already partially manual; formalize | High value for iteration speed | planned |

## Originality candidates (v0.2.0)

| ID | Idea | Novelty angle | Verdict path |
|---|---|---|---|
| O1 | Recurrent depth (share a block, apply twice) | Universal Transformers exist; no production LLM ships them | 45-min micro-ablation |
| O2 | Planner-encoder (context → plan tokens → decoder cross-attends) | Decoder-only dominates; built-in planner is rare | 45-min micro-ablation |
| O3 | Entity memory slots for story characters | Domain-specific; measurable on story data | 45-min micro-ablation |
| O4 | Bidirectional prediction (prev + next + 2-ahead aux loss) | Rarely combined at small scale | 45-min micro-ablation |
| O5 | EMA-teacher self-distillation (see R3) | Rare at pretraining scale | 45-min micro-ablation |

## Negative results so far (kept for the record)

- None yet from our own runs. Literature: RLVR at 135M (L20-Edu) degraded
  GSM8K accuracy; logit soft-cap alone was insufficient for Muon instability at
  53B (Kimi needed QK-Clip; we use QK-Norm + soft-cap at 58M).

## Harness improvements needed

1. `ablate.py`: accept `--arms a,b,c` and `--steps N` so new arms run without
   editing code.
2. `configs/`: one micro config per experiment (E1–E13) as they are scheduled.
3. Journal entry per experiment with raw numbers and the config used.
