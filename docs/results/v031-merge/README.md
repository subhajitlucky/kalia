# v0.3.1 merge — a zero-GPU win

Run 2026-11-01 on `kalia-v031-merge` (Kaggle CPU; **no GPU**). The v0.3.0
second epoch's best in-training 100-batch val was **2.7255 at step 6,500**; the
final checkpoint at 6,770 scored **2.7743**. The intermediate checkpoint was
overwritten (one `ckpt.pt` every 30 minutes, no EMA).

The merge tests whether a weighted average of v0.2.0 and v0.3.0 — a valid
operation because v0.3.0 is a continuation of v0.2.0 — recovers some of that
0.049-nat gap.

## The result

| v0.2.0 weight | v0.3.0 weight | val |
|---:|---:|---:|
| 0.0 | 1.0 | 2.7743 |
| 0.1 | 0.9 | 2.7653 |
| 0.2 | 0.8 | 2.7601 |
| **0.3** | **0.7** | **2.7579** |

The w0.0 run reproduces the registered 2.7743 exactly. The best blend is
**30% v0.2.0 / 70% v0.3.0**, **−0.0164 nats** vs v0.3.0. The trend is
diminishing but still improving at 0.3; 0.4 was not run.

## The 2k benchmark

| task | v0.2.0 | v0.3.0 | **v0.3.1** | Δ vs v0.3.0 | z |
|---|---:|---:|---:|---:|---:|
| ARC-Easy | 46.40 | 47.55 | **48.00** | +0.45 | +0.28 |
| HellaSwag | 35.60 | 37.20 | **37.65** | +0.45 | +0.29 |
| LAMBADA | 23.90 | 26.30 | **27.00** | +0.70 | +0.50 |
| PIQA | 61.37 | 61.59 | **61.97** | +0.38 | +0.24 |
| WinoGrande | 50.75 | 51.30 | **52.25** | +0.95 | +0.48 |

All five tasks improve, none regresses by >2σ, and every task is above v0.2.0.
The gains are modest at 2k (z 0.24–0.50); the val and CL-0 are the strong parts.

## CL-0

| model | probe loss | drift vs v0.3.0 |
|---|---:|---:|
| v0.3.0 | 2.756648 | — |
| **v0.3.1** | **2.740949** | **−0.015699** |

The merge does not trade retention for the val gain; it improves the probe.

## Release gate

- val **2.7579 < 2.7743** — pass.
- 2k: no task regresses by >2σ — pass.
- CL-0 **2.740949 ≤ 2.763648** (v0.3.0 + 0.0070) — pass.
- **`release_candidate = true`.**

## The checkpoint bug

The merge kernel scored the best checkpoint, copied it to
`/kaggle/working/best_merged.pt`, then deleted it in cleanup: the `best` row and
the `rows[best_name]` entry were the same dict, so updating the path and then
unlinking it removed the copy. **The numbers are valid** — the summary, 2k and
CL-0 were produced from the real weights — but the binary is gone.

The exact blend is deterministic: **0.3 × v0.2.0 + 0.7 × v0.3.0**. A
materialise kernel rebuilds it from the hash-pinned source checkpoints and
re-scores the 100-batch val; the release uses that checkpoint only if the val
reproduces **2.7579**. Source hashes:

- v0.2.0 `924ad52b27017bb823e925df6695d1becdef032b772ae8436eb0b74bdbac6718`
- v0.3.0 `80b064f9a99e6ed5372e7fa46021574a9b21298b822fd2c581357831cc77537e`
- val.bin `625f8e5fdeae0f37b519fd6b898fd0c482ee6e7643899e0b4fe44eec64f464bc`

## What this is and is not

- **It is a weight-space soup**, not an architecture result. It recovers part of
  the final-checkpoint regression by averaging a continuation's endpoints.
- **It is not a substitute for a best-checkpoint callback.** The real fix is to
  save the checkpoint at the best val; the merge is a zero-cost recovery.
- **It is not the data experiment.** The TinyStories-heavy mix (20%) is the next
  data lever; it waits.
- The 2k gains are small; a 5k re-score would confirm them before a public
  benchmark claim.

## Next

1. Materialise and re-score the w0.3 checkpoint (CPU); if val = 2.7579, it is
   the release artifact.
2. Export v0.3.1 to safetensors and re-score the written file (the v0.2.0
   release gate); update the model card.
3. Optional: w0.4 / w0.5, or a lower-LR best-checkpoint continuation. The merge
   already gives the release candidate; the continuation is a further bet.
4. Data mix: 10% TinyStories / 70% FineWeb-Edu / 15% Cosmopedia / 5% code,
   only after the release path is settled.

## Artifacts

- `v031_merge_summary.json` — the four val rows, 2k rows, CL-0, gates.
- `v031_merge_manifest.json` — source/val/bench hashes.
- `bench_v031_2k.json`, `cl0_v031.jsonl`, `val-w0.3.md`.
- `kalia-v031-merge.log`.
- `docs/research/2026-11-01-v031-merge-plan.md` — positives, negatives,
  verdict written before the run.
