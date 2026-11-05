# Pre-registration — X46: broad-template compiler

Registered: 2026-11-05, before the X46 GPU run.

## Context

X44 showed the X43 compiler only partially generalizes to fresh templates:
F-P **0.085 → 0.523/0.493**, `number+place` **0.455 both**, but fresh
`rename+swap` **0.406/0.416** (bar 0.45) and fresh 3-op **0.283/0.365** (bar
0.30, one seed). Compiler exact match on fresh templates fell to
**0.185/0.124**.

The hypothesis is data diversity, not architecture: the compiler has seen only
seven templates. X46 trains the same rank-16 LoRA on **17 templates** (the
seven X43 templates plus ten new broad templates), with singles, pairs and
triples. The four X44 test templates remain completely held out.

## Data

- X43 templates: `give_keep`, `hide_find`, `put_count`, `meet_tell`,
  `carry_place`, `give_place`, `count_place`.
- Ten new broad templates: `mail_send`, `borrow_back`, `plant_count`,
  `paint_hang`, `cook_serve`, `carry_deliver`, `write_read`, `find_hide`,
  `pack_ship`, `promise_give`.
- All 1-, 2- and 3-op combinations supported by each template; ~300 examples
  per combination, `NAMES_TRAIN`.
- Dev: 80 per combination, `NAMES_EVAL`.
- Test: the untouched X44 fresh benchmark (`b69bcd61…`), four templates and
  twenty names never seen in training.

## Treatment

Same compiler as X43/X44: rank-16 LoRA on v0.3.1, base frozen, 1,413,120
trainable parameters, 3,000 steps, seeds 1401/1402.

## Bars (X44 fresh benchmark)

| ID | prediction | threshold |
|---|---|---|
| **G-1** | fresh F-P | ≥ 0.45, both seeds |
| **G-2** | fresh `rename+swap` | ≥ 0.45, both seeds |
| **G-3** | fresh `number+place` | ≥ 0.40, both seeds |
| **G-4** | fresh 3-op depth | ≥ 0.30, both seeds |
| **G-5** | compiler exact match on fresh | ≥ 0.25, both seeds (diagnostic) |
| **G-6** | reader unchanged / no runaway output | base sha matches; fallback < 0.01 |

## Decision rule

- **G-1…G-4 pass:** the broad-template compiler is a robust OOD capability;
  record a v0.4.1 validation. The public base remains v0.3.1.
- **G-2 or G-4 fails:** the fresh-template boundary is final for this
  architecture/data scale; no further template-tuning. Log the null and stop.
- No tuning after the result. No HF base-model release: the standard five-task
  +10% gate is separate and not addressed by this adapter.

## Cost

≤ 2 GPU-h. Remaining window ~10 GPU-h.

## Artifacts

- `docs/results/x46-broad/` — summary, adapters, logs, generated prefixes,
  end-to-end rows.
- `broad_templates.py`, `tools/kalia_x46_broad_kernel.py`.
- This file.
