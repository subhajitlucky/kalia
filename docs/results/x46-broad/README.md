# X46 — broad-template compiler: robust OOD capability

Status: **clean pass of G-1…G-6.** The fresh-template boundary found by X44 is
closed by broader template training. This is the robust capability release
(v0.4.1); the public base remains v0.3.1.

## Method

The same rank-16 LoRA compiler on frozen v0.3.1, trained on **17 templates**:
the seven X43 templates plus ten new broad templates (`mail_send`,
`borrow_back`, `plant_count`, `paint_hang`, `cook_serve`, `carry_deliver`,
`write_read`, `find_hide`, `pack_ship`, `promise_give`). Every supported 1-,
2- and 3-op combination is generated. The four X44 test templates are **never
seen**.

- Train: ~40k examples, `NAMES_TRAIN`.
- Dev: `NAMES_EVAL`, disjoint.
- Test: the untouched X44 fresh benchmark (`b69bcd61…`), new names and new
  templates.
- 3,000 steps, two seeds, 1,413,120 trainable parameters, base frozen.
- Cost: **0.96 GPU-h**.

## Result (X44 fresh benchmark)

| cell | v0.3.1 normal | gold-edit | BROAD_1401 | BROAD_1402 |
|---|---:|---:|---:|---:|
| **F-S** | 0.3875 | 0.7125 | **0.7475** | **0.7525** |
| **F-P** | **0.0850** | 0.5725 | **0.6825** | **0.6850** |
| **F-T** | **0.0350** | 0.3925 | **0.6150** | **0.5575** |

Hard operations:

| operation | normal | X43 | BROAD_1401 | BROAD_1402 |
|---|---:|---:|---:|---:|
| `number+place` | — | 0.4545 | **0.4545** | **0.4545** |
| `rename+swap` | — | 0.406/0.416 | **0.7228** | **0.6634** |
| `rename+place` | — | 0.576/0.727 | 0.6364 | 0.6364 |
| `swap+place` | — | 0.576/0.545 | 0.5455 | 0.5455 |
| `rename+negate` | — | 0.830/0.760 | 0.8700 | 0.8700 |
| `swap+negate` | — | 0.320/0.220 | 0.6600 | 0.6600 |

Compiler exact match on fresh templates: **0.3550 / 0.3783** (X43:
0.185/0.124). Fallback: **0.0000** both.

## Registered bars

| ID | prediction | result |
|---|---|---|
| **G-1** | fresh F-P ≥ 0.45, both | **PASS** (0.6825 / 0.6850) |
| **G-2** | fresh `rename+swap` ≥ 0.45, both | **PASS** (0.7228 / 0.6634) |
| **G-3** | fresh `number+place` ≥ 0.40, both | **PASS** (0.4545 / 0.4545) |
| **G-4** | fresh 3-op ≥ 0.30, both | **PASS** (0.6150 / 0.5575) |
| **G-5** | compiler EM ≥ 0.25, both | **PASS** (0.3550 / 0.3783) |
| **G-6** | reader unchanged / no runaway | **PASS** |

## What changed from X43/X44

| metric | X43 on X44 fresh | X46 on X44 fresh |
|---|---:|---:|
| F-P | 0.5225 / 0.4925 | **0.6825 / 0.6850** |
| `rename+swap` | 0.4059 / 0.4158 | **0.7228 / 0.6634** |
| F-T (3-op) | 0.2825 / 0.3650 | **0.6150 / 0.5575** |
| compiler EM | 0.1850 / 0.1242 | **0.3550 / 0.3783** |

Broad template diversity fixed the boundary. The result is not a benchmark
artifact: the X44 templates and names were never in training.

## Caveat

This is a **capability adapter**, not a new base model. The standard five-task
zero-shot suite is unchanged. The U-P/F-P results above the gold-edit
interface are a reader-system effect: the compiler generates reader-friendly
prefixes that are not identical to the gold edit. The claim is end-to-end
compositional editing, not a more faithful world model.

## Artifacts

- `x46_summary.json` — full bars, cells, ops, manifest.
- `lora_broad_seed1401.pt`, `lora_broad_seed1402.pt` — adapters only.
- `train_broad_seed*.csv`, `dev_broad_seed*.csv`.
- `generated_BROAD_*.jsonl`, `end2end_BROAD_*.json`,
  `end2end_BROAD_*_rows.jsonl`.
- `kalia-x46-broad.log`, `bench_audit.json`.
- Pre-registration: `docs/preregistrations/2026-11-05-X46-broad-compiler.md`
  (hash `ff6393be…`).
