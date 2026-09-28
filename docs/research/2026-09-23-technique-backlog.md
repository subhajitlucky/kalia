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
| E0 | **Learning-rate sweep for Muon+** (0.015 / 0.02 / 0.03 / 0.06) | D17 protocol | LR dominates algorithmic tricks | High | None | **complete**: 3.4943 / **3.4941** / 3.5027 / 3.5380 → 0.02 optimal |
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
| R2 | **Tiny RLVR probe** (verifiable format/arithmetic tasks) | L20-Edu-135M | **Negative**: GRPO reduced GSM8K accuracy at 135M (1.82% → 1.59%) | Likely negative; run once for the record | queued as a documented negative-result probe |
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

- **X16 (ours).** Chunk-preserving reversal training at 30M did not close the
  Abhimanyu gap: 5.2465 vs 5.1107 control, forward val +0.083. Both
  pre-registered bars missed on both seeds. See D42. The gap is architectural
  (5.1 nats at 30M, 6.06 at 58M), so the next attempt must target attention, not
  data — see `2026-09-28-why-27b-beats-397b.md`.
- **I14 (ours, process).** The published v0.1.2 training log silently began at
  step 1740 for two days: log-restore-on-resume landed one session after the
  lossy session had already started. Recovered from hub commit history.
- Literature: RLVR at 135M (L20-Edu) degraded GSM8K accuracy; logit soft-cap
  alone was insufficient for Muon instability at 53B (Kimi needed QK-Clip; we
  use QK-Norm + soft-cap at 58M); MoE upcycling loses to its own dense base at
  small scale (`moe-upcycle`: 13.46 vs 9.27 ppl), and the upcycling scaling law
  says from-scratch wins when no pretrained model exists.

## Added 2026-09-28 (from the 27B-vs-397B study)

| ID | Technique | Source | Evidence | Expected at 58M | Status |
|---|---|---|---|---|---|
| D1 | **Three-stage corpus** (general → reasoning-dense → long-form fiction) | Qwen3 tech report | S1 30T general, S2 5T STEM/code/reasoning with accelerated LR decay, S3 32K long-context | High | **v0.2.1 candidate** |
| D2 | **Instance-level data mixture** via small-proxy ablations | Qwen3 tech report | Replaces domain-level mixing; annotated 30T+ tokens on educational value / field / safety | High | adopt as screening method |
| D3 | **Best-of-N with self-consistency** (own log-prob as verifier) | Snell et al. 2024; Hassid et al. 2024 | 1B + inference scaling beats 405B without it; 13B×5 samples beats one 70B by up to 15% | High (inference-time) | **v0.3.0 candidate** |
| D4 | **Complete the LR schedule** (`min_lr_ratio` 0.1 → ~0.01) | Qwen "accelerated LR decay"; our own D41 | Our plateau was measured inside an unfinished cosine — weak evidence of convergence | Medium | **v0.2.2 candidate** |
| D5 | **Benchmark alongside loss in every screen** | Qwen3.8 tech report | "Loss and downstream accuracy do not always move together"; +1.58 accuracy at 0.002 loss gap; loss optimum ≠ accuracy optimum | Medium (method) | **adopt now** |
| D6 | **Gated Residual / 4-branch residual stream** | Qwen3.8; IMU-1 | +1.58 avg accuracy over pre-norm; data-dependent read/write adds +1.98 more | Low–Medium | 45-min micro-ablation |
| D7 | **Reverse-pass-aware attention** (bidirectionally-gated or reverse-bias) | our own gap analysis | Gap is architectural; X16 ruled out the data route | Unknown — honest bet | 45-min micro-ablation, pre-registered |
| D8 | **n-gram embedding tables** | Qwen3.8; DeepMind | Scales params at near-zero per-token FLOPs | Low | **parked** — changes output distribution, breaks the canonical val set |
| — | MoE upcycling | Komatsuzaki 2023; Liew et al. | Loses to dense base at small scale; from-scratch preferred when no base exists | Negative | **rejected, evidence recorded** |
| — | Long CoT + GRPO | Qwen3 | Needs ≥1.5B; 0.5B students show catastrophic forgetting | Negative at 58M | **rejected — scale floor, and no teacher** |

## Harness improvements needed

1. `ablate.py`: accept `--arms a,b,c` and `--steps N` so new arms run without
   editing code.
2. `configs/`: one micro config per experiment (E1–E13) as they are scheduled.
3. Journal entry per experiment with raw numbers and the config used.
