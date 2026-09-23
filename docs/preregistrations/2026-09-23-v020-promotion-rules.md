# Pre-registration — v0.2.0 Composition and Promotion Rules

Registered: 2026-09-23, before the architecture-ablation verdict and before any
X16 result exists. Related: D39 (strategic audit), X16 pre-registration.

## Purpose

Fix, in advance, exactly what earns a place in v0.2.0 and what does not. No
post-hoc promotion: if a result misses its rule below, it is reported as a
negative result and excluded.

## Fixed by decision, not by result

- **Data**: the compliance-clean v2b corpus only (60% FineWeb-Edu-dedup /
  20% TinyStories / 15% Cosmopedia / 5% permissive-license Python; 2.4B train
  tokens). No further mixture changes without a new pre-registration.
- **Recipe base**: the frozen v0.1.2 recipe (Muon+ col_row at LR 0.02, QK-Norm,
  logit soft-cap 30, cosine schedule, fp16 + DDP, micro-batch 8 × accum 32 ×
  ctx 1024 × 2 GPUs, max_steps 4770). Promotions below are the only changes.
- **Evaluation protocol**: held-out sentence loss + bpB, 27-prompt probe suite,
  Abhimanyu gap, entity-consistency report; all reported against the final
  v0.1.2 checkpoint. Same seeds where applicable.

## Promotion rules (decided now)

1. **Architecture (kalia-ablate arms `micro-loop2`, `micro-deepthin`,
   `micro-gqa` vs control `micro-muonplus-qk`, 763 steps, step-700 val):**
   a variant enters v0.2.0 iff its final val ≤ control − 0.02 nats.
   If two or more qualify, the lowest final val wins; exact ties are broken by
   lower measured wall-clock cost per step. Variants within ±0.02 of control do
   not enter (parsimony).
2. **Reversal training (X16):** enters v0.2.0 as a data transform
   (prob 0.5, chunks 4–16) iff, on the seed-pair mean, **P2 holds**
   (treatment Abhimanyu gap < control gap − 0.10 nats) **and P3 holds**
   (forward validation loss within +0.02 of control). If only P2 holds, the
   transform is recorded as a promising negative and retested in a later cycle
   — it does not enter v0.2.0.
3. **Everything else is frozen out** until v0.2.0 has results: no MoE, no
   distillation, no RL, no new optimizer/schedule changes, no additional
   architecture ideas.

## Budget and stop rules

- v0.2.0 target: 2.5B tokens (full 4770 steps at 524,288 tokens/step), about
  two 8.5-hour sessions; starts only after the weekly quota reset and after the
  recipe freeze (rules above resolved).
- Pre-registered stopping: if validation loss does not improve across 3
  consecutive evals (750 steps) after step 3000, training may stop early and
  the final checkpoint is released with the plateau documented.

## Reporting

Results are reported regardless of outcome, including failures and deviations
from this plan (with reasons). If X16 or an architecture variant fails its
rule, the negative result is published alongside the v0.2.0 release.
