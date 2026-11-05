# X41 — Edit-Compile-Then-Continue

Status: **partial advancement.** The compiler solves `number+place` at the
gold-edit ceiling and lifts the pair cells, but `rename+swap` is seed-unstable
and misses the registered 0.45 bar on one seed. X43 is a registered follow-up
(template-held-out pair training). The public model remains v0.3.1.

## Baseline (v0.3.1, 58M, full 1,600 records)

Kaggle CPU kernel `subhajitlucky/kalia-x41-baseline`. Base sha
`bbd8a036…`, benchmark sha `aad1b321…`, 400 records per cell, chance 0.25
for pair cells and 0.50 for single cells.

| cell | normal | gold-edit | B_1401 | B_1402 | A_1401 |
|---|---:|---:|---:|---:|---:|
| **S-S** seen / single | 0.4500 | 0.7025 | 0.7025 | 0.7025 | 0.7025 |
| **S-P** seen / pair | **0.1700** | **0.5125** | **0.4275** | **0.4100** | **0.4075** |
| **U-S** unseen / single | 0.3150 | 0.8775 | 0.8600 | 0.8250 | 0.7850 |
| **U-P** unseen / pair | 0.0925 | 0.7825 | 0.5675 | 0.6350 | 0.6325 |

By operation:

| operation | n | normal | gold | B_1401 | B_1402 | A_1401 |
|---|---:|---:|---:|---:|---:|---:|
| `number+place` | 124 | **0.0000** | **0.6935** | **0.6935** | **0.6935** | **0.6935** |
| `rename+swap` | 294 | 0.1871 | 0.5612 | **0.3537** | **0.4830** | 0.3844 |
| `swap+negate` | 169 | 0.1124 | 0.5444 | 0.6095 | 0.4911 | 0.5444 |
| `rename+negate` | 213 | 0.1455 | 0.8216 | 0.4930 | 0.5023 | 0.5869 |
| `number` | 90 | 0.6000 | 0.9889 | 0.9889 | 0.9889 | 0.9889 |
| `place` | 123 | 0.1870 | 0.8130 | 0.8130 | 0.8130 | 0.8130 |
| `rename` | 216 | 0.2870 | 0.9074 | 0.9028 | 0.8981 | 0.8241 |
| `swap` | 214 | 0.4393 | 0.5047 | 0.4766 | 0.4159 | 0.4159 |
| `negate` | 157 | 0.4650 | 0.8854 | 0.8854 | 0.8854 | 0.8854 |

- **Arm B** trains on singles + non-test pair types; the four benchmark pair
  types are held out. **Arm A** trains on singles only.
- Compiler exact match on the full 2×2: B_1401 **0.4263**, B_1402 **0.4000**,
  A_1401 **0.3981**. Singles-only dev exact match: **0.9966 / 0.9920**.
- LoRA: rank 16, **1,413,120** trainable parameters, base frozen. Two seeds
  for Arm B; one seed for Arm A.
- End-to-end rows and generated prefixes are in `compiler/`; the eval kernel
  is `subhajitlucky/kalia-x41-eval`.

## Registered bars

| ID | prediction | result |
|---|---|---|
| **C-1** | singles dev EM ≥ 0.90, both seeds | **PASS** (0.9966 / 0.9920) |
| **C-2** | `number+place` ≥ 0.40, both seeds | **PASS** (0.6935 / 0.6935) |
| **C-3** | `rename+swap` ≥ 0.45, both seeds | **FAIL** (0.3537 / 0.4830) |
| **C-4** | S-P ≥ 0.40, both seeds | **PASS** (0.4275 / 0.4100) |
| **C-5** | U-P ≥ 0.30, both seeds | **PASS** (0.5675 / 0.6350) |
| **C-6** | reader unchanged | **PASS** |

**Verdict: partial advancement, not a full registered pass.** The strict
decision rule requires C-2 and C-3 on both seeds; C-3 misses on B_1401. We do
not cherry-pick B_1402.

## What the result says

1. **The 58M reader is competent in the edited world.** Normal S-P 0.170 →
   gold-edit 0.5125; `number+place` 0.000 → 0.694; `rename+swap` 0.187 →
   0.561. The bottleneck is edit application, not continuation.
2. **A 1.4M-parameter adapter can build the edited world for the number/place
   composition.** `number+place` goes from **0.000 to 0.6935** on both seeds,
   exactly the gold-edit ceiling. This is the clean positive result.
3. **`rename+swap` is the remaining failure.** The compiler applies the pair
   inconsistently; the seed spread is large (0.354 vs 0.483). The two seeds'
   errors are not complementary enough for a reader-guided selector to repair
   (X42: test `rename+swap` 0.4286, oracle capped at 0.4898).
4. **The singles-only compiler also composes partially.** Arm A, with no pair
   supervision, reaches S-P 0.4075 and `number+place` 0.6935. The pair data
   helps S-P but is not the whole story.
5. **This is a system-level capability result, not a new attention or
   tokenizer mechanism.** It does not change the public model's standard
   benchmark; it adds an experimental edit-compiler adapter.

## Artifacts

- `baseline/` — normal/gold summaries, per-record rows, bench audit and bench.
- `compiler/` — LoRA adapters, train/dev logs, generated prefixes,
  end-to-end summaries and rows, full kernel log.
- Pre-registration: `docs/preregistrations/2026-11-05-X41-edit-compiler.md`
  (hash `d3d5dc53…`).
- X42 null: `docs/results/x42-reader-ensemble/`.
- X43 follow-up: `docs/preregistrations/2026-11-05-X43-template-pairs.md`.
