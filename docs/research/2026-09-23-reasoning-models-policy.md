# Reasoning Models (o1/R1): Adoption Policy and Scale Constraints

Status: policy record
Date: 2026-09-23
Sources: DeepSeek-R1 report (2025, MIT), OpenAI o1 system card (2024),
L20-Edu-135M RLVR negative result (arXiv 2606.22189, 2026), R-Zero
(NeurIPS 2025), MobileLLM-R1 (ICLR 2026).

## What o1/R1 actually are

They are not a new architecture. They use the same decoder-only transformer as
their base models. "Reasoning" comes from **post-training**: long
chain-of-thought supervised fine-tuning, reinforcement learning with verifiable
rewards (GRPO-family), rejection-sampling SFT, then further RL. Their papers and
code are public.

## Policy: methods are adopted, weights are not

| Action | Policy |
|---|---|
| Implementing a published method (GRPO, RLVR, long-CoT SFT, rejection sampling) with our own code, data and weights, citing the source | Permitted — standard research practice |
| Starting from someone else's trained weights (including R1/o1 or open reasoning models) | Prohibited — borrowed capability contradicts the project's core claim (D21) |
| Claiming external results as ours | Prohibited |

## Why R1-style RLVR is not used at KALIA's scale (58M)

1. **Documented failure at small scale.** L20-Edu-135M: direct GRPO on GSM8K
   reduced exact-match accuracy (1.82% → 1.59%).
2. **Self-play collapse.** R-Zero: the smallest model tested (0.6B) peaked after
   one iteration and then degraded; pseudo-label quality fell 79% → 63%.
3. **Capacity threshold.** MobileLLM-R1-140M required ~4.2T curated tokens plus
   a full cold-start + RL pipeline to reach weak but measurable reasoning
   (GSM8K 16.3% base). At 58M with 2.4B tokens the capacity is not there.
4. **No verifiable reward in our domain.** KALIA's target is story-like English;
   reasoning-style RL requires automatically checkable answers, which do not
   exist for narrative quality.

## What KALIA adopts from the reasoning lineage instead

- **Cold-start SFT** with curated instruction traces (planned post-training
  stage, v1).
- **Rejection sampling** for building small, high-quality SFT sets.
- **Continual-learning composition** (replay + self-distillation + importance
  regularization + merged LoRA) to avoid forgetting (D26).
- **Test-time compute** via in-place fast weights (TTT) and entity memory —
  our KALIA-EM+TTT target (D23).

## Revisit condition

R1-style RLVR may be revisited if two conditions hold: (a) KALIA scales beyond
~300M parameters, and (b) a task domain with automatic verification becomes
part of the plan (e.g., arithmetic or structured output). Until then, the
evidence says the compute is better spent on data quality and architecture.
