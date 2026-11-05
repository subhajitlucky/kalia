# Pre-registration — X41: Edit-Compile-Then-Continue (ECC)

Registered: 2026-11-05, before the X41 compiler training run. **This is a
capability experiment, not a new attention/architecture mechanism.** The
mechanism is a learned edit compiler; the scientific claim is that edit
application, not continuation, is the compositional bottleneck.

## Why this exists

The X40 gold-edit test gave the 30M models the post-intervention prefix. On the
clean 2×2 benchmark the hard cells stayed at chance/zero, so the project closed
X40 and concluded that the edit was not the missing piece. The 58M public model
has not been scored on the clean 2×2. An exploratory 400-record probe (not used
as evidence or to set bars) suggested a different picture at 58M: normal S-P
~0.20, gold-edit S-P ~0.57, with `number+place` 0.0 → 0.65 and `rename+swap`
~0.27 → 0.64. If that survives the full 1,600-record Kaggle CPU baseline, the
bottleneck is **edit application**, not the reader's ability to continue the
edited world.

The official baseline is the full Kaggle CPU kernel
`subhajitlucky/kalia-x41-baseline`, not the exploratory probe.

## Mechanism

Two passes, one frozen reader:

1. **Compiler** — a LoRA adapter on v0.3.1 (rank 16, attention + MLP linear
   layers only; base frozen) maps
   `instruction + factual prefix` → `post-intervention prefix`.
2. **Reader** — the unchanged public v0.3.1 scores the candidate continuations
   conditioned on the compiler's edited prefix, exactly as the gold-edit mode.

The compiler is trained on **single interventions** and on **non-test pair
types** (`rename+place`, `swap+place`, `place+negate`). The four benchmark pair
types (`rename+swap`, `swap+negate`, `number+place`, `rename+negate`) are
**held out entirely**. Names are disjoint across train/dev/test. This is a
held-out operation-combination test, not a memorisation test.

### Prior art

Counterfactual story rewriting (Qin et al. 2019; Huang et al. 2021), scratchpads
and chain-of-thought (Nye et al. 2021; Wei et al. 2022; self-notes 2023),
instruction-based editing, and two-stage generate-then-verify systems. The
specific composition — a learned intervention compiler plus a frozen reader on
the clean 2×2 — is a **B+/D application**, not an A-class new primitive. The
claim is a measured capability advancement on a specific topic, which is the
allowed release gate.

## Data

- Train templates: X40 `TRAIN_TEMPLATES` plus three extra templates
  (`carry_place`, `give_place`, `count_place`) that add pair coverage without
  using a test pair type.
- Train names: X40 `NAMES_TRAIN`.
- Dev names: X40 `NAMES_EVAL` (disjoint).
- Test: the clean 2×2 benchmark, seed 2026, 400 records per cell, names
  `NAMES_BENCH` (disjoint from both).
- Arm B (primary): singles + non-test pairs.
- Arm A (secondary): singles only. This is the strict composition test.

## Stage 0 — feasibility pilot

A 2,000-example Arm B subset, 200 optimizer steps, rank 16. Stop if dev
single-operation exact match is < 0.50. The pilot is engineering, not the
registered test; it cannot be used to claim success.

## Stage 1 — registered test

| ID | prediction | threshold |
|---|---|---|
| **C-1** | compiler learns single edits on held-out names | dev EM ≥ 0.90, both seeds |
| **C-2** | `number+place` becomes compositional | end-to-end ≥ 0.40, both seeds |
| **C-3** | `rename+swap` becomes compositional | end-to-end ≥ 0.45, both seeds |
| **C-4** | S-P aggregate | ≥ 0.40, both seeds |
| **C-5** | U-P aggregate (unseen templates) | ≥ 0.30, both seeds, secondary |
| **C-6** | reader unchanged | base checksum matches; no base gradients |
| **C-7** | no hidden negation collapse | report `swap+negate`, `rename+negate` by operation |

Chance is 0.25 for pair cells. The primary bars are the two hard operations,
not the aggregate; beating the negation pairs alone is not sufficient.

## Decision rule

- **C-2 and C-3 pass on both seeds:** the specific advancement is established.
  Record a GitHub release candidate with the LoRA adapter and the benchmark;
  the public base model remains v0.3.1.
- **Only C-4 passes:** partial; report, but do not claim the hard-cell
  advancement.
- **C-2 or C-3 fails on either seed:** log the null; no tuning of LR, rank,
  steps or data after the result. A different recipe is a new registration.
- **Stage 0 fails:** stop before Stage 1; log the pilot null.

## Cost

Stage 0 ≤ 0.3 GPU-h. Stage 1: two Arm B seeds + one Arm A seed, each ≤ 0.5
GPU-h, plus end-to-end eval ≤ 1 GPU-h. Cap: **4 GPU-h**. The baseline is CPU.

## Artifacts

- `docs/results/x41-edit-compiler/` — baseline, compiler logs, adapter,
  end-to-end summaries/rows, kernel logs.
- `docs/preregistrations/2026-11-05-X41-edit-compiler.md` (this file).
- Code: `edit_compiler.py`, `tools/kalia_x41_compiler_kernel.py`.
