# Pre-registration — X44 fresh benchmark + X45 depth

Registered: 2026-11-05, before the fresh-benchmark run. CPU/light GPU.

## Why this exists

X43 passed the registered 2×2 bars, but that benchmark has been used across
X41–X43. The next question is whether the compiler learned the editing rule or
the benchmark. X44 evaluates on **new names and new sentence templates** never
used in training or prior evals. X45 then asks whether the same compiler can
compose **three edits at once**, which it never saw in training.

This is a capability validation, not the user's standard five-task +10% HF
release gate. The public base remains v0.3.1.

## Fresh benchmark

New names (disjoint from `NAMES_TRAIN`, `NAMES_EVAL`, `NAMES_BENCH`):

`Anya, Bodhi, Cyrus, Dara, Eshan, Farah, Gita, Hiro, Indra, Jaya, Kiran, Lila2,
Mira, Naveen, Oona, Pavel, Rhea, Sami, Veer, Zoya`

New templates (none in `TRAIN_TEMPLATES`, `EVAL_TEMPLATES`, or
`EXTRA_TEMPLATES`):

1. `bake_share` — bake for B, B shares with C (rename, swap, negate).
2. `find_return` — find an object, B returns it (rename, swap, place, negate).
3. `count_pack` — pack N objects into a surface (number, place).
4. `promise_tell` — promise a gift, B tells C (rename, swap, negate).

Cells:

- **F-S** — fresh template, single op, 2-way, 400 records.
- **F-P** — fresh template, pair op, 4-way, 400 records.
- **F-T** — fresh template, triple op, 8-way, 400 records.

The same fresh name pool is used in every cell. Candidate sets are audited for
duplicates; the correct candidate is unique.

## Evaluation

- v0.3.1 normal and gold-edit (paired baseline).
- X43 C_1401 and C_1402 adapters, generated prefixes, frozen reader.
- No training in X44. X45 first tests X43 zero-shot on 3-op.

## Bars

| ID | prediction | threshold |
|---|---|---|
| **F-1** | X44 fresh pair S-P | ≥ 0.40, both seeds |
| **F-2** | fresh `number+place` | ≥ 0.40, both seeds |
| **F-3** | fresh `rename+swap` | ≥ 0.45, both seeds |
| **F-4** | no runaway compiler output | fallback < 0.01 |
| **F-5** | reader unchanged | base sha matches |
| **F-6** | X45 zero-shot triple | ≥ 0.30 (chance 0.125), both seeds |

If F-1…F-3 pass, X43 is validated on a fresh benchmark (v0.4.1 candidate).
If F-6 passes, the compiler has depth extrapolation (v0.5.0 candidate).
If F-6 fails, train one 3-op compiler on the X43 extra templates only and test
on fresh F-T (X45b); bars: fresh F-T ≥0.30 and ≥ +1–2 pp over the zero-shot
compiler. X45b is a separate pre-registration if needed.

## Decision rule

- **F-1…F-3 pass:** release a validation note for v0.4.1; the base stays
  v0.3.1.
- **F-6 pass:** release v0.5.0 depth note.
- **Any fail:** log the boundary; no tuning after the result. The X43 release
  remains valid for its registered benchmark, but no broader claim.
- No HF base-model release: the standard five-task +10% gate is separate and
  not met by this adapter work.

## Cost

Fresh benchmark generation: CPU. Eval: ≤1 GPU-h. X45b training, if needed:
≤1.5 GPU-h. Cap: 2.5 GPU-h.

## Artifacts

- `docs/results/x44-fresh/` — benchmark audit, baseline, X43 summaries/rows.
- `fresh_benchmark.py`, `tools/kalia_x44_eval_kernel.py`.
- This file.
