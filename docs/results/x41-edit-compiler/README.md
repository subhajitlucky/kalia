# X41 — Edit-Compile-Then-Continue

Status: **baseline complete; compiler pilot/training running.** This directory
holds the clean 2×2 baseline and, when the GPU job finishes, the learned
edit-compiler result.

## Baseline (v0.3.1, 58M, full 1,600 records)

Kaggle CPU kernel `subhajitlucky/kalia-x41-baseline`. Base sha
`bbd8a036…`, benchmark sha `aad1b321…`, 400 records per cell, chance 0.25
for pair cells and 0.50 for single cells.

| cell | normal (instruction+prefix) | gold-edit (post-edit prefix) |
|---|---:|---:|
| **S-S** seen / single | 0.4500 | 0.7025 |
| **S-P** seen / pair | **0.1700** | **0.5125** |
| **U-S** unseen / single | 0.3150 | 0.8775 |
| **U-P** unseen / pair | 0.0925 | 0.7825 |

By operation:

| operation | n | normal | gold-edit |
|---|---:|---:|---:|
| `number+place` | 124 | **0.0000** | **0.6935** |
| `rename+swap` | 294 | **0.1871** | **0.5612** |
| `swap+negate` | 169 | 0.1124 | 0.5444 |
| `rename+negate` | 213 | 0.1455 | 0.8216 |
| `number` | 90 | 0.6000 | 0.9889 |
| `place` | 123 | 0.1870 | 0.8130 |
| `rename` | 216 | 0.2870 | 0.9074 |
| `swap` | 214 | 0.4393 | 0.5047 |
| `negate` | 157 | 0.4650 | 0.8854 |

## What the baseline says

1. **The 58M reader is competent in the edited world.** Given the
   post-intervention prefix, S-P rises from 0.170 to **0.5125**, and the two
   hard operations rise from 0.000/0.187 to **0.694/0.561**.
2. **The failure is edit application, not continuation.** The same reader,
   same candidates, same scoring rule: only the prefix changes.
3. **The 58M model is not the 30M X40 model.** The 30M gold-edit S-P was
   ~0.29–0.33 and `number+place` was 0.0 in 5/6 arms. At 58M the edited-world
   continuation is much stronger. This is a scale/quality lever, not a new
   mechanism.
4. **The compiler must recover the gold prefix.** The gold-edit table is the
   ceiling for the two-pass system; the learned compiler's end-to-end score
   cannot exceed it.

## Artifacts

- `x41_baseline_summary.json` — full normal/gold summaries.
- `x41_base_normal.json`, `x41_base_gold.json` — per-mode cells/ops.
- `x41_base_normal_rows.jsonl`, `x41_base_gold_rows.jsonl` — 1,600 rows each.
- `x41-bench_audit.json` — benchmark cells, names, token lengths, sha.
- `kalia-x41-baseline.log` — kernel log.

The compiler result, if it clears the registered bars, will be added here.
The public model remains v0.3.1; this is a capability experiment, not a new
attention or tokenizer mechanism.
