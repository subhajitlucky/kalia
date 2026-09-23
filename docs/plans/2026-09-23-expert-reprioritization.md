# Expert Reprioritization — Efficient Compute Plan

Status: active (supersedes the ordering in the v0.1.x ladder where they conflict)
Date: 2026-09-23
Author note: written as the solo-decision-taker assessment, adopted as the plan.

## Frank assessment

**Kept without hesitation:** from-scratch training; free-tier resumable harness
with HF checkpoints; micro-ablation gate before full runs; journaling every
incident; reproducibility (control arm matched across independent runs).

**Changed, ranked by impact:**

1. **Learning-rate sweep comes first.** We compared optimizers at fixed LRs.
   LR dominates any algorithmic trick; a 4-point sweep costs ~2 GPU-hours and
   routinely beats kernel-level changes. Optimizers are now compared at their
   own best LR, not one shared value.
2. **No full baseline runs before the recipe is locked.** The v0.1.0 AdamW run
   consumed ~8.5 GPU-hours (~30% of a week) for a model we abandoned. Full runs
   require: micro-ablation win + LR known + data frozen.
3. **Stricter evaluation.** Shuffled held-out set; **bits-per-byte (bpB)** as the
   headline metric (tokenizer-independent); two seeds before promotion; no
   fp16/fp32 mismatch between train and eval numbers.
4. **Data quality before architecture.** TinyStories is a fluency crutch.
   Target mixture for the next model: **~55% FineWeb-Edu, ~20% TinyStories
   (early-stage curriculum), ~15% Cosmopedia-style synthetic educational,
   ~10% code+math**. This sets the model's ceiling more than any optimizer.
5. **Stop by target loss, not step count.** WSD schedule plus an explicit
   `target_val_loss`: when validation reaches the target, run a short decay
   phase and stop. Saves GPU hours mid-run.

**Novelty bet:** EMA-teacher self-distillation (model teaches its own smoothed
copy) with a proper control — the one cheap, underexplored idea at 58M.

## Execution order and budget

Assumes ~20 GPU-hours remaining this week (30 h/week quota).

| Phase | Work | Compute | When |
|---|---|---|---|
| P1 | E1b: Muon vs Muon+ at lr 0.03 | ~1 h GPU | running |
| P2 | LR sweep completion: Muon+ at 0.015 and 0.06 (0.02 and 0.03 known) | ~1 h GPU | right after P1 |
| P3 | Implement WSD + target-loss stop + checkpoint EMA + bpB eval (+ tests) | 0 GPU | parallel with P1/P2 |
| P4 | Build v2 data mixture (new prep config, new dataset) | ~1 h Kaggle CPU (free) | parallel |
| P5 | Finish v0.1.2 to 4,770 steps (complete the 71%-done model) | ~8 h GPU (2 sessions) | this week |
| P6 | v0.2.0 full run: best recipe + v2 data + WSD/EMA/target-stop | ~15–18 h GPU | next week's quota |
| P7 | EMA-teacher self-distillation micro-ablation (novelty bet) | ~1 h GPU | with P6 |

Rationale for P5: v0.1.2 is ~71% through after session 3 and gives a complete,
shippable model this week. Abandoning it now would repeat the v0.1.0 mistake in
reverse. P3/P4 cost no GPU quota and unlock P6 for next week's reset.

## Evaluation protocol (new standard)

- **Metric**: bits-per-byte on a shuffled held-out set. Report alongside val CE.
- **Promotion rule**: an experiment is promoted only if it beats the current
  recipe by more than seed noise, evaluated with ≥2 seeds at micro scale.
- **Record**: every run logs config hash, seed, steps, tokens, bpB and the
  git commit of the code used.

## Rules going forward

1. LR/hyperparameters are tuned before algorithms are compared.
2. No full run starts until its recipe passed micro-ablations.
3. Full runs target a loss, not a step count.
4. Every incident, result and decision is journaled the day it happens.
