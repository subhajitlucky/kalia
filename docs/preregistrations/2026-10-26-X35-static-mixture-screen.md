# Pre-registration — X35: does the static source mixture move the 58M continual update?

Registered: 2026-10-26, before any GPU time. Related: D19, D56, D59, X26, X28,
X29, v0.3.0.

## Why this exists

The only large positive in the record is **data**: the compliance-rebuilt corpus
moved held-out loss by **0.169 nats at matched tokens** and 0.2285 at release.
Every architecture arm is closed (D48), replay/re-warm failed its gate (M0), the
adaptive mixture drifts to the easy sources and costs **+0.0183 nats** (X26/X28),
and the linear selector did not replicate (X34/D59). D56 names the mixture as the
open lever; D19 planned **55/20/15/10** (FineWeb / TinyStories / Cosmopedia /
Python) and it was never run.

X29's 94% “delivery path” effect was confounded: the no-replay arm sampled the
2.4B corpus at ~0.2 passes, while the static arm sampled the 498M equalised
shards at ~1.05 passes. A static mixture screen holds the token budget, schedule,
seed and shards fixed and moves **only the sampling weights**. That is the
cheapest clean test of the data lever the project has.

The 58M from-scratch ruler is unusable (0.2892), but the continual-update ruler
is **0.0015–0.0070** (X29). A ~0.01-nat mixture effect is readable in this
regime at 500 steps.

## Design

**Model.** v0.2.0 at step 4,770. Architecture unchanged: 10 layers × 512,
QK-Norm + soft-cap as trained. Base config
`configs/v030-rewarm-s1401.yaml` (500 update steps, warmup 50, LR 6e-4,
`rewarm: true`, `update_budget: 500`). The only changes per arm are the seed and
the source weights.

**Data.** The four published equalised per-source shards from
`kalia-lm/kalia-v020` — `fineweb`, `tinystories`, `cosmopedia`, `python`;
124,500,691 tokens each, 498,002,764 total. No replay (`replay_prob: 0`), matching
the X28 static arm. Validation is the canonical `kalia-prep-v2b` `val.bin`.

**Arms.** TinyStories (20%) and Cosmopedia (15%) are fixed; the FineWeb share
absorbs the change. All arms sum to 100%.

| arm | FineWeb | TinyStories | Cosmopedia | Python |
|---|---|---|---|---|
| `control` | 0.60 | 0.20 | 0.15 | **0.05** |
| `code10` | 0.55 | 0.20 | 0.15 | **0.10** |
| `code25` | 0.40 | 0.20 | 0.15 | **0.25** |

`code10` is the D19 target and the primary test. `code25` is a dose-response:
it tells us whether the 5-point shift is too small to move anything, and it
brackets the direction. `code25` is exploratory and cannot be adopted on its own.

**Stages.**

- **Stage 1 screen** — seed 1401, three arms, 500 steps. Kernel
  `subhajitlucky/kalia-x35-mix-screen`.
- **Stage 2 replication** — seed 1402, `control` and `code10` only, 500 steps,
  and only if S-1 passes. Kernel `subhajitlucky/kalia-x35-mix-replicate`.
- **CL-0** — a CPU probe on the Stage 2 checkpoints after the replication. No
  GPU quota.

**Primary metric.** Final val loss at step 5,270 (500 update steps) on the
canonical `val.bin`, from `out/<arm>/val_log.csv`.

**Secondary metrics.** CL-0 probe loss on the G2-pinned probe shard (retention);
`per_source_log.csv` held-out loss (gaming); GPU-hours (cost).

## Pre-registered predictions

| ID | Prediction | Threshold |
|---|---|---|
| **S-1** | The D19 target beats the v0.2.0 mixture at seed 1401 | `code10 − control ≤ −0.010` nats |
| **S-2** | The effect survives a second seed | if S-1 passes: `code10 − control ≤ −0.010` at seed 1402, two-seed mean `≤ −0.010`, both seeds the same sign |
| **S-3** | The code dose-response is reported | no bar; if `code25 − control > +0.050` or the arm diverges, record the boundary |
| **S-4** | The gain is not paid for by retention | `code10` CL-0 probe loss not worse than control by more than **0.0070** (the measured static seed spread) |
| **S-5** | Gaming is reported | code source held-out loss improves; no other source regresses by more than 0.020; report the four-source table |

S-4 is deliberately a retention bar, not a val bar: a mixture that improves
`val.bin` by fitting the training distribution harder while losing the CL-0
probe is the dessert failure in a new costume.

## Decision rule (fixed now)

- **S-1 fails** → no code-share signal at 500 steps. Do not run Stage 2. Report
  the screen and close the static-mixture track at this budget. Do not tune the
  weights until something passes; that is searching for a result.
- **S-1 passes, S-2 fails** → seed-sensitive. No adoption; report.
- **S-1 and S-2 pass, S-4 passes** → `code10` is a candidate for a 1,000-step /
  v0.4 test. **Not a release claim.**
- **S-1/S-2 pass, S-4 fails** → the val gain is at a retention cost. Report the
  trade-off, no adoption.
- **S-3/S-5** are reported regardless of the other bars.
- If any arm produces no val log, or an arm diverges, record it. No verdict from
  a partial pair.

## Cost and stopping rules

Stage 1: 3 arms × ~2.4h ≈ **7.2 GPU-h**. Stage 2: 2 arms × ~2.4h ≈ **4.8 GPU-h**.
CL-0 CPU: ~0.5h. Total ≈ **12.5 GPU-h**, against 29.36h remaining. This uses
about 42% of the weekly quota; it is a real training screen, not a probe.

- If Stage 1 exceeds 10 GPU-h, stop after Stage 1 and bank Stage 2 for the next
  window.
- If an arm's final val is outside 2.70–3.00, or the control is not within
  0.02 of the archived X28 static anchor at the same step, stop and inspect the
  data path before reading any contrast.
- A session failure loses the arms in that session; rerun the stage. The code
  and data hashes make the rerun comparable.

## What this does not establish

- Not a release. 500 steps, not v0.2.0's 4,770.
- Not a new corpus. The shards are the v0.2.0 equalised per-source shards, so
  this is a **sampling-weight** effect, not a new-source effect.
- Not the long-form data test. PG-19/Gutenberg is a separate experiment: the
  probe measured document lengths but no training shard exists yet.
- Not a claim that code share helps v0.4. A pass buys the 1,000-step/v0.4 test;
  it does not establish the model card.

## Deviations

Any change to the arms, weights, seeds, steps, data shards, thresholds or
decision rule after this commit requires a hash-registered amendment. Results
are reported regardless of outcome.

## Artifacts (to be archived)

- `docs/results/x35/` — screen and replication summaries, val logs, per-source
  logs, CL-0 ledger, run manifests.
- `tools/kalia_x35_mix_kernel.py` — the screen/replicate runner.
- `kaggle/kernel-metadata-x35-mix-screen.json`,
  `kaggle/kernel-metadata-x35-mix-replicate.json`.
