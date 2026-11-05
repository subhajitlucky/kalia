# Pre-registration — X43: Template-Held-Out Pair Training (THPT)

Registered: 2026-11-05, before the X43 GPU run.

## Context

X41's compiler is a partial advancement. It learns single edits at 0.993–0.997
exact match, recovers `number+place` at the gold ceiling (**0.694** both
seeds), and lifts S-P to **0.428/0.410** and U-P to **0.568/0.635**. The
registered X41 rule still fails because `rename+swap` is **0.354/0.483**: one
seed misses the 0.45 bar.

X42 tried to repair the cell with a reader-guided ensemble of the two X41
edits. On the held-out test half, the selected rule (MinEntropy) reached
`rename+swap` **0.4286**, below 0.45; the oracle selection was capped by the
better single seed at 0.4898. The two seeds' errors are not complementary
enough. X42 is a null.

The remaining hypothesis is **data coverage, not architecture**: the compiler
has seen rename and swap singly, and has seen rename/swap paired with place,
but has never applied rename then swap. X43 teaches the exact pair types on
**disjoint templates**, then tests on the benchmark templates. This is a
template-transfer test, not the stricter held-out-operation-combination test
of X41. The X41 partial result remains the stricter claim.

## Data

- Base: the X41 Arm B data (singles + non-test pairs on all templates).
- Add **all four test pair types** (`rename+swap`, `swap+negate`,
  `number+place`, `rename+negate`) only on the three extra templates:
  - `carry_place`: rename+swap, rename+negate, swap+negate.
  - `give_place`: rename+swap, rename+negate, swap+negate.
  - `count_place`: number+place.
- Never add a test pair to `TRAIN_TEMPLATES` (`give_keep`, `hide_find`,
  `put_count`, `meet_tell`) or `EVAL_TEMPLATES` (`tell_secret`, `ask_help`,
  `pack_box`). The benchmark templates remain held out for pair supervision.
- Names: `NAMES_TRAIN` for train, `NAMES_EVAL` for dev, `NAMES_BENCH` for test.
- Dev: the Arm B dev plus test pairs on the extra templates, for checkpoint
  selection only.

## Treatment

Same compiler as X41: rank-16 LoRA on v0.3.1, base frozen, 1,413,120 trainable
parameters, 2,200 steps, two seeds (1401/1402). No architecture change.

## Bars

| ID | prediction | threshold |
|---|---|---|
| **D-1** | singles exact match on held-out names | ≥ 0.90, both seeds |
| **D-2** | `number+place` | ≥ 0.40, both seeds |
| **D-3** | `rename+swap` | ≥ 0.45, both seeds |
| **D-4** | S-P aggregate | ≥ 0.45, both seeds |
| **D-5** | U-P aggregate (unseen templates) | ≥ 0.35, both seeds |
| **D-6** | reader unchanged | base sha matches; no base gradients |

Also report the gold-edit ceiling and the X41 Arm B numbers as the comparison.

## Decision rule

- **D-1…D-6 pass:** the template-held-out pair curriculum is a specific
  advancement. Record a release candidate with the two adapters and the
  benchmark; the public base model remains v0.3.1.
- **D-3 fails:** log the null; the pair-composition training did not transfer
  across templates. No tuning after the result.
- **D-4 or D-5 fails but D-2/D-3 pass:** partial; report, do not claim the
  full advancement.

## Cost

Two seeds, 2,200 steps, plus end-to-end eval: **≤ 2 GPU-h**. Remaining window
is ~11 GPU-h.

## Artifacts

- `docs/results/x43-thpt/` — summary, train/dev logs, adapters, generated
  prefixes, end-to-end rows.
- `tools/kalia_x43_thpt_kernel.py`.
- This file.
