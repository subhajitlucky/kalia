# X43 — Template-Held-Out Pair Training

Status: **clean pass of the registered bars (D-1…D-6).** This is the
capability release candidate: a learned edit compiler on the frozen v0.3.1
base. The public base model is unchanged.

## Method

The X41 compiler was trained on singles and non-test pair types; it never
applied `rename` then `swap`, and that cell was seed-unstable. X43 teaches all
four test pair types on the three **extra templates** (`carry_place`,
`give_place`, `count_place`), never on the benchmark templates
(`give_keep`, `hide_find`, `put_count`, `meet_tell`, `tell_secret`, `ask_help`,
`pack_box`). The benchmark templates remain held out for pair supervision.

Same compiler as X41: rank-16 LoRA on v0.3.1, **1,413,120** trainable
parameters, base frozen, 2,200 steps, seeds 1401/1402. The reader is the
unchanged public v0.3.1.

## Result (full 1,600-record 2×2 benchmark)

| cell | normal | gold-edit | C_1401 | C_1402 |
|---|---:|---:|---:|---:|
| **S-S** seen / single | 0.4500 | 0.7025 | 0.7025 | 0.7025 |
| **S-P** seen / pair | **0.1700** | **0.5125** | **0.4800** | **0.4925** |
| **U-S** unseen / single | 0.3150 | 0.8775 | 0.8200 | 0.8150 |
| **U-P** unseen / pair | **0.0925** | **0.7825** | **0.8600** | **0.8725** |

By operation:

| operation | n | normal | gold | C_1401 | C_1402 |
|---|---:|---:|---:|---:|---:|
| `number+place` | 124 | **0.0000** | 0.6935 | **0.6935** | **0.6935** |
| `rename+swap` | 294 | **0.1871** | 0.5612 | **0.5918** | **0.6463** |
| `swap+negate` | 169 | 0.1124 | 0.5444 | 0.5976 | 0.5621 |
| `rename+negate` | 213 | 0.1455 | 0.8216 | 0.8216 | 0.8216 |
| `number` | 90 | 0.6000 | 0.9889 | 0.9889 | 0.9889 |
| `place` | 123 | 0.1870 | 0.8130 | 0.8130 | 0.8130 |
| `rename` | 216 | 0.2870 | 0.9074 | 0.8981 | 0.8935 |
| `swap` | 214 | 0.4393 | 0.5047 | 0.4065 | 0.4019 |
| `negate` | 157 | 0.4650 | 0.8854 | 0.8854 | 0.8854 |

- Compiler exact match on the full benchmark: **0.5469 / 0.5438**.
- Singles-only dev exact match: **0.9920 / 0.9977**.
- Fallback to the factual prefix: **0.0000** both seeds.
- Overall end-to-end: **0.7156 / 0.7206** (gold-edit overall 0.7188).

## Registered bars

| ID | prediction | result |
|---|---|---|
| **D-1** | singles EM ≥ 0.90, both | **PASS** (0.9920 / 0.9977) |
| **D-2** | `number+place` ≥ 0.40, both | **PASS** (0.6935 / 0.6935) |
| **D-3** | `rename+swap` ≥ 0.45, both | **PASS** (0.5918 / 0.6463) |
| **D-4** | S-P ≥ 0.45, both | **PASS** (0.4800 / 0.4925) |
| **D-5** | U-P ≥ 0.35, both | **PASS** (0.8600 / 0.8725) |
| **D-6** | reader unchanged | **PASS** |

## What changed from X41

| metric | X41 Arm B | X43 |
|---|---:|---:|
| S-P | 0.4275 / 0.4100 | **0.4800 / 0.4925** |
| `number+place` | 0.6935 / 0.6935 | **0.6935 / 0.6935** |
| `rename+swap` | 0.3537 / 0.4830 | **0.5918 / 0.6463** |
| U-P | 0.5675 / 0.6350 | **0.8600 / 0.8725** |
| compiler EM | 0.4263 / 0.4000 | **0.5469 / 0.5438** |

The pair supervision on disjoint templates closes the `rename+swap` gap and
lifts the unseen-template cell. The `number+place` result is unchanged at the
gold ceiling.

## Caveat

This is a **template-held-out pair-transfer** test: the pair types appear in
training on the extra templates, but the benchmark templates never receive
pair supervision. It is **not** the stricter held-out-operation-combination
test of X41; X41 remains the stricter, partial result. The U-P result above
the gold-edit interface (0.860/0.873 vs 0.783) is a system effect: the
compiler's generated prefixes are reader-friendly but not identical to the
gold edited prefix (EM ≈ 0.545). It is not evidence that the generated world
is more faithful.

## Artifacts

- `x43_summary.json` — full bars, cells, operations, compiler exact match.
- `lora_armC_seed1401.pt`, `lora_armC_seed1402.pt` — adapters only (~5.5 MB
  each); the base is public v0.3.1.
- `train_armC_seed*.csv`, `dev_armC_seed*.csv` — training/dev curves.
- `generated_C_*.jsonl`, `end2end_C_*.json`, `end2end_C_*_rows.jsonl` —
  generated prefixes and per-record scores.
- `kalia-x43-thpt.log` — kernel log.
- Pre-registration: `docs/preregistrations/2026-11-05-X43-template-pairs.md`
  (hash `7cbce00f…`).
