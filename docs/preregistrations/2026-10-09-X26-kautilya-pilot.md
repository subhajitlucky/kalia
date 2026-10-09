# X26 — Kautilya adaptive mixture, first real run (paired pilot)

**Status:** registered and launching 2026-10-09, before any GPU time was spent.
**Depends on:** v0.2.0 checkpoint (public), the four published source shards
(`corpus/*.bin`), the CL-0 ledger (Step 0).
**Relation to v0.3.0:** the registered Step-4 arm was cancelled by Step 1's stop
rule. This pilot re-opens it as a **paired screen**, not as a verdict.

## Why this can be read at all, when Step 1 stopped the sequence

Step 1 measured the 58M **independent-seed** spread (0.2892, confirmed at
0.2912) and the registered stop rule fired: micro-budget comparisons across
seeds are unreadable at this scale. A continual update of the *same checkpoint*
is a different design: same weights, same seed, same data order, same schedule —
the only difference is whether the Kautilya policy runs. The comparison is
**paired by construction**, not across seeds.

What has *not* been measured is the noise of a paired update: two identical
arms resumed twice could still diverge through nondeterminism we have not
bounded. Therefore:

- this run is a **screen**; its difference is reported, no promotion follows
  from it alone;
- a verdict requires a same-config rerun (to bound paired noise) and one fresh
  seed (1402), both registered here as the follow-ups;
- the registered expectation is **null or small**, consistent with this
  project's history; a large effect is as suspicious as it is interesting.

## Design

Two arms, both resumed from v0.2.0 at step 4770, both seed 1401, both
`rewarm: true`, both 1000 update steps (`update_budget`), identical schedule
and initial weights (60/20/15/5):

- **static** — `adaptive_every: 0`; the same code path, policy disabled.
- **adaptive** — `adaptive_every: 100`; weights re-estimated every 100 steps
  from per-source held-out loss (sama/dana/bheda/danda).

Data: the four equalised published shards, sampled at runtime; no offline
re-blend. The static arm is the registered control for the Kautilya mechanism.

## Measurement

- Primary: held-out val loss on the canonical `val.bin` (prep-v2b), logged
  every 100 steps.
- Secondary: per-source held-out losses and the applied strategy per source
  (`per_source_log.csv`), which is the policy's own audit trail.
- Retention: CL-0 probe (`tools/forgetting_probe.py`) on both final
  checkpoints, scored against the Step-0 ledger.

## Cost and stopping

~5 GPU-hours (2 arms x 1000 steps on 2x T4 DDP), plus CPU evaluation. No early
stop; the session quota is the only ceiling.

## What this is not

- Not a capability claim; not a paper result on its own.
- Not the v0.3.0 sequence resumed; Steps 2-3 stay cancelled.
- Not a re-run of anything published: this is the first real execution of the
  Kautilya policy on real data.
