# Adoptable Frontier — Loops, KDA, Sparse Attention, MoE, RSI, Continual Learning

Status: active research record
Date: 2026-09-23
Sources: Ouro/LoopLM (arXiv 2510.25741, 2025), THOUGHTS (ICLR 2025),
recurrent-depth generalization (arXiv 2604.07822, 2026), STARS (2026),
Kimi Linear/KDA (arXiv 2510.26692, 2025), DeepSeek-V3.2 DSA (2025),
R-Zero (NeurIPS 2025), Absolute Zero (NeurIPS 2025), continual-learning
composition study (arXiv 2609.06986, 2026), Any-SSR (2025),
Titans (NeurIPS 2025), TTT-E2E (NVIDIA 2026).

## Verdict table

| Technique | Evidence | Fits KALIA 58M? | Verdict |
|---|---|---|---|
| **Looped / recurrent-depth transformers (LoopLM, Ouro)** | Ouro 1.4B/2.6B match up to **12B** standard models; gains are in knowledge *manipulation* (composition, multi-hop), not storage. Accuracy scales as log(effective depth). Depth extrapolation at inference time works. Known limits: overthinking + instability beyond trained depth (STARS fixes some) | **Yes** — same parameters, more effective depth; cheap to test at micro scale | **ADOPT — promote to front of queue (X11)** |
| Immediate block sharing (MobileLLM-LS) | +1.1% at 125M, zero extra parameters | Yes | Already queued (X3); simplest special case of looping |
| **KDA / Kimi Linear hybrid attention** | 3 KDA : 1 full attention beats full attention at 48B; −75% KV cache; 6× decode speed at 1M context. Needs chunkwise DPLR kernels (FLA library) | Only pays off at long context; heavy implementation for us | **DEFER** to the long-story/deployment phase |
| **DSA (DeepSeek Sparse Attention)** | Lightning indexer + top-k gather; built for agentic prefill at huge context | No — our context is 1K; sparsity pays nothing | **REJECT** |
| **MoE at small scale** | Needs scale to pay off; instability; MoEUT shows it can combine with recurrence | Not at 58M | **REJECT for now**; revisit at 300M+ |
| **RSI self-play (R-Zero, Absolute Zero)** | Gains at 4B. **R-Zero: smallest model (0.6B) peaked at iteration 1 then degraded; pseudo-label accuracy fell 79% → 63%.** AZR needs a code-execution verifier | No — collapse is documented at scales 10× ours | **REJECT at 58M**; recorded as a negative result |
| **Continual learning composition (anchors + merged LoRA)** | Best composition (data replay + self-distillation + importance reg + merged LoRA) raised 100-task retention from 1.2% → 34.9% (**28×**) | Post-training phase (SFT/continual) | **ADOPT for v1 post-training** |
| Any-SSR analytic routing | RLS-based, non-forgetting task routing | Post-training, later | Note for v2+ |
| Titans / TTT (test-time memory) | Small models beat GPT-4 on BABILong; constant latency to 2M tokens | Yes | Already X7 (invention target) |
| Self-teaching (EMA teacher) | Semi-supervised lineage; rarely validated at pretraining scale | Yes | Already X8 |

## Why looping is the biggest lever we have been ignoring

Our 58M model has 10 layers — very shallow by modern small-model standards
(MobileLLM recommends 30+ layers at 125M; Ouro loops 4 shared sub-stacks).
Looping decouples **effective depth** from **parameter count**:

- 5 unique layers looped 2× = effective depth 10 at half the parameters
- 10 unique layers looped 2× = effective depth 20 at our current parameter count

Ouro's results say the gain is in *composition/manipulation* — exactly what
"KALIA feels smart" means for stories (keeping track of characters, events,
causal chains) — while pure perplexity/memorization may be slightly worse than
a non-looped model of equal *effective* parameters. That trade is good for us:
we care about output quality and reasoning-like coherence, not just val loss.

Design rules from the literature: use sandwich normalization for stability;
train recurrence depth ≥ 4 to unlock inference-time depth extrapolation;
entropy-regularize the exit gate only after the fixed-loop version works;
expect overthinking if looping far beyond trained depth.

## Revised experiment queue

| ID | Experiment | Cost | Status |
|---|---|---|---|
| X11 | **Looped depth**: 5 unique layers × 2 loops vs 10 unique layers (equal effective depth); then 10×2 vs 10×1 | 2× 1.5h GPU | **front of queue** |
| X3 | Immediate block sharing (MobileLLM-LS, the L=2, adjacent-loop case) | 1.5h GPU | queued |
| X1, X2, X4, X5 | as previously queued | ~5h GPU | queued |
| X7, X8, X9 | TTT / self-teaching / entity memory | ~6h GPU | queued |
| X12 | Continual-learning anchors on top of SFT (post-v0.2.0) | TBD | planned |

## Negative results recorded (for honesty and future reference)

1. RLVR self-play (R-Zero paradigm) collapses at small scale; the smallest
   tested model (0.6B) peaked after the first iteration. A 58M model would not
   survive the loop. Our self-improvement path is therefore: self-teaching
   (EMA teacher) + test-time adaptation (TTT) + retrieval — never RL self-play
   at this size.
2. DSA and MoE add complexity that cannot pay off at 1K context / 58M
   parameters. Rejected on evidence, not on taste.
