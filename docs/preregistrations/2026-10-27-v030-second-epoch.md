# v0.3.0 — the second epoch: train v0.2.0 longer, with a re-warm

Registered: 2026-10-27, before any GPU time. Related: D41, D48, D55, D59, D60,
X35.

## Why this exists

v0.2.0 is the current model. Since it finished, every mechanism track has been
tested and closed: architecture (D48), replay/re-warm (M0), adaptive mixture
(X26/X28), TTT (X25), document masking (X17), entity decoding and the linear
selector (X34/D59), and the static code mixture (X35/D60). None produced a
better model.

The model is undertrained. v0.2.0 is 57.9M parameters and saw **2.5B tokens —
about 43 tokens per parameter**. Small models are normally trained at thousands
of tokens per parameter. The last 31% of v0.2.0's cosine bought nothing, but a
500-step re-warm update already improved canonical val from **2.8248 to 2.7903**
(X35 control). The cheapest thing that can improve the model is **more tokens**,
not another mechanism.

This is the first actual model-training run since v0.2.0. It is a model
improvement, not a mechanism claim.

## Design

- **Resume** from the v0.2.0 step-4,770 checkpoint.
- **Data:** the full 2.4B `kalia-prep-v2b` `train.bin`; the same corpus v0.2.0
  trained on. No replay, no new source, no mixture change.
- **Budget:** **2,000 update steps** (1.05B tokens). Total tokens seen: 3.55B
  (~61 tokens/parameter, 1.42 epochs).
- **Schedule:** a fresh cosine over the update — `rewarm: true`, 200-step
  warmup, peak LR 6e-4, floor 6e-5. This is the v0.3.0 re-warm recipe, at a
  larger budget than the 500-step screen that worked.
- **Seed:** 1401 (fresh data order; the weights are fixed).
- **Monitoring:** 100-batch eval every 100 steps during training; the primary
  val is a separate **100-batch eval on the final checkpoint**, the canonical
  protocol, not a training-log point.
- **Cost:** ~2,000 steps × ~18s ≈ **10 GPU-h**; the run stops cleanly at 11h
  (`--max-minutes 660`) and publishes the checkpoint/logs.

## Why 2,000 steps, not a full 4,770-step epoch

A full epoch is ~22 GPU-h, essentially the whole remaining window, and leaves no
budget for evaluation or a second decision. 2,000 steps is 4× the 500-step update
that improved val, fits one session, and leaves ~12h for the 2k benchmarks, CL-0,
and a possible continuation. If the val curve is still improving at 2,000 steps,
a second stage can be registered with the remaining quota; this registration does
not authorise one.

## Pre-registered bars

| ID | bar | threshold |
|---|---|---|
| **E-1** | The second epoch improves canonical val | final 100-batch val **≤ 2.8048** (a 0.02-nat improvement over v0.2.0's 2.8248) |
| **E-1b** | It also beats the 500-step screen point | report final val against the X35 control 2.7903 (a 20-batch screen point, not a bar) |
| **E-2** | No benchmark regression | at 2,000 samples, no task regresses by **more than 2σ** versus v0.2.0 |
| **E-3** | Retention is not traded away | CL-0 probe loss not worse than v0.2.0 by more than **0.0070** (the measured static-update spread) |
| **E-4** | Cost is recorded | GPU-hours and tokens reported |

The primary is E-1. E-2 and E-3 are the release guards: a val gain that costs a
benchmark or retention is not a clean model improvement.

## Decision rule (fixed now)

- **E-1, E-2 and E-3 pass** → v0.3.0 is a **release candidate**. Evaluate at
  2,000 samples, write the card, and publish as an experiment release. It is not
  a promotion of v0.2.0 on the old registered stability rule; that rule is about
  the v0.2.0 corpus comparison and is not retroactively changed.
- **E-1 passes, E-2 or E-3 fails** → report the trade-off. The checkpoint may be
  released as an experiment, but not as a clean improvement.
- **E-1 fails** → the second epoch did not help. Keep v0.2.0, stop, and do not
  tune the schedule or budget to force a win.
- **If the run stops at the 11h cap before 2,000 steps** → resume from the
  published checkpoint in a second kernel; the step count is reported, not
  rounded up.

## What this does not establish

- Not a two-seed claim. This is a model-improvement run; the release is an
  experiment. A claim that "more epochs help at 58M" would need a paired seed and
  is not made here.
- Not a corpus change. It is the same 2.4B tokens, seen a second time.
- Not a resolution of the loss/accuracy divergence. The 2k benchmark is the
  guard, not a promise that val gains become accuracy gains.
- Not a new architecture. The model block is byte-identical to v0.2.0.

## Deviations

Any change to the data, budget, schedule, seed, bars or decision rule after this
commit requires a hash-registered amendment. Results are reported regardless of
outcome.

## Artifacts (to be archived)

- `docs/results/v030-epoch2/` — final checkpoint, val log, final 100-batch eval,
  2k benchmark, CL-0 ledger, run manifest.
- `tools/kalia_v030_epoch2_kernel.py` — the training runner.
- `kaggle/kernel-metadata-v030-epoch2.json`.
