# v0.3.0 second epoch — release candidate

> **v0.3.1 note (2026-11-01):** the 30% v0.2.0 / 70% v0.3.0 weight merge is the
> new release candidate (val **2.7579**, all five 2k tasks positive, CL-0
> **2.740949**). See `docs/results/v031-merge/`.

Registered: `docs/preregistrations/2026-10-27-v030-second-epoch.md`
(sha `7375a595…`, ledger-verified). Training kernel `kalia-v030-epoch2`,
code `b103d5d`, **~10 GPU-h**. Final eval kernel `kalia-v030-final-eval`,
code `0f1a370`, CPU, zero GPU quota.

The 2,000-step re-warm cosine continued v0.2.0 on the same 2.4B corpus, no
replay, seed 1401. The final checkpoint is step 6,770.

## Verdict

**All three registered bars pass. `release_candidate = true`.**

| bar | required | measured | verdict |
|---|---|---|---|
| **E-1** canonical 100-batch val | ≤ 2.8048 | **2.7743** (bpB **0.9070**) | **PASS** |
| **E-2** 2,000-sample benchmark | no task worse than −2σ | all five tasks improved | **PASS** |
| **E-3** CL-0 probe | ≤ 2.805875 (+0.0070) | **2.756648**, drift **−0.042227** | **PASS** |

E-1 is **0.0505 nats** better than v0.2.0's 2.8248. E-3 is not merely inside
the allowance: the probe loss is **0.0422 lower** than v0.2.0, so this update
did not trade retention for language quality on the CL-0 probe.

## E-2 — the 2,000-sample benchmark

| task | v0.2.0 | v0.3.0 | Δ pp | z | regression |
|---|---:|---:|---:|---:|---|
| ARC-Easy | 46.40 | 47.55 | +1.15 | +0.73 | no |
| HellaSwag | 35.60 | 37.20 | +1.60 | +1.05 | no |
| LAMBADA | 23.90 | 26.30 | +2.40 | +1.75 | no |
| PIQA | 61.37 | 61.59 | +0.22 | +0.14 | no |
| WinoGrande | 50.75 | 51.30 | +0.55 | +0.28 | no |

The comparison uses the archived v0.2.0 2,000-sample baseline
(`bench_v020_2k.json`). The registered 500-sample verdict is not rewritten;
this is a new, higher-resolution observation.

## What this is and is not

- **It is a model win**: more tokens and a re-warm cosine produced a
  checkpoint that is better on val, better on all five benchmarks, and better
  on the CL-0 probe than v0.2.0.
- **It is not an architecture result.** The architecture is unchanged. The
  next-gen search continues separately; v0.3.0 is the baseline it must beat.
- The first final-eval kernel printed the val then died on a markdown parse
  bug (`**Val loss: x**` vs `Val loss: **x**`); the fixed kernel is `0f1a370`.
  No number changed between the two launches.

## Artifacts

- `final/v030_final.json` — E1/E2/E3, benchmark rows, release-candidate flag.
- `final/bench_v030_2k.json` — the raw 2,000-sample results.
- `final/cl0_v030.jsonl` — the CL-0 probe ledger row.
- `final/eval_val.md` — the canonical val output.
- `final/run_manifest.json` — checkpoint/val/baseline/probe hashes.
- `final/kalia-v030-final-eval.log` — the CPU eval log.
- `val_log.csv` — the in-training val curve; the last point 2.8208 at 6,700,
  the canonical final 2.7743.
- `train_log.csv` — the 2,000 update steps.
- `run_manifest.json` — the training code revision, data/checkpoint hashes and
  schedule.
- `v030-epoch2.yaml` — the built config.
- `kalia-v030-epoch2.log` — the training kernel log.
