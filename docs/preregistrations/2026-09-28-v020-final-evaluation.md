# Pre-registration — v0.2.0 Final Evaluation and Claim Boundaries

Registered: 2026-09-28, while session 2 is running (step 1,746 → ~3,492) and
before the step-4,770 result exists. Related: D38 (J1 hash-anchoring, J2
pre-registered stopping), D41, D42, D43, and
`docs/preregistrations/2026-09-23-v020-promotion-rules.md`.

## Why this document exists

The v0.2.0 model will exist; the claims about it do not yet. This fixes the
yardstick, the metrics, the thresholds, and — most importantly — what may and may
not be said about the outcome, *including the claims that are forbidden*. D42
showed what an un-pre-registered story looks like under pressure; D43 showed what
a silently redefined yardstick looks like. Both are cheap to prevent in advance
and expensive to fix afterwards.

## The yardstick (fixed by D43, not chosen here)

- **Canonical held-out set**: `kalia-prep-v2b`'s `val.bin`, 10,000,000 tokens.
- **Primary protocol**: `eval_val.py --batches 100 --batch-size 8 --seed 1234`
  = 819,200 tokens, deterministic, bytes/token measured the same way (4.4086 for
  this set).
- **Fixed reference — v0.1.2 on this exact protocol: 3.0533 loss / 0.9992 bpB.**
  The published 2.4366 / 0.8184 belongs to the *old* val set and is not a
  comparator for anything in v0.2.0.
- Any v0.2.0 number computed on the old val set, or at a different batch count,
  is not comparable and will be reported separately with the protocol labelled.

## Primary criterion (fixed now, before the number exists)

| ID | Criterion | Threshold |
|---|---|---|
| P-A | **Success** — v0.2.0 matches v0.1.2 on the same yardstick | val loss ≤ **3.0533** |
| P-B | **Improvement** — a real gain, above eval noise | val loss ≤ **3.0033** (≥ 0.05 better) |
| P-C | **Regression** — a real loss | val loss > **3.1033** (> 0.05 worse) |

Between 3.0033 and 3.1033 the result is "matched within noise". P-C is
registered symmetrically with P-B on purpose: a regression is reported as one,
with no re-runs, seed-shopping, or protocol changes used to escape it.

## Secondary metrics (reported under fixed settings)

Zero-shot via `lm-evaluation-harness`, 0-shot, 500 samples — PIQA, ARC-Easy,
HellaSwag, WinoGrande, LAMBADA. v0.1.2 references: 61.4 / 45.8 / 36.8 / 50.2 /
23.0.

- **S-A**: no task may regress by more than **1.0 pp** against those references.
- A task improving by more than 1.0 pp is reported as improved — with no claim
  beyond the measurement itself.
- LAMBADA accuracy and perplexity both reported (it is the long-range task where
  v0.1.2 was weakest at 23.0% / 193.6).

**Abhimanyu gap**: forward and reverse NLL on the fixed probe set, plus the
20-sentence probe held-out loss. v0.1.2 reference: gap **6.06** nats (forward
3.23, reverse 9.29), probe loss 3.2303, probe bpB 0.9415.

**Two-point curve**: the step-3,492 checkpoint (session 2's endpoint) is
evaluated under the identical primary protocol, so v0.2.0 has a same-yardstick
curve rather than a single terminal number.

## No gap claim is registered

X16 (D42) established that chunk-preserving reversal training does not close the
Abhimanyu gap, and v0.2.0 ships no reversal transform. Therefore **no
gap-reduction or gap-improvement claim is pre-registered for v0.2.0.** The gap
is measured and published as a measurement. If it moves, that is recorded as an
observation with no causal attribution; a claim requires its own pre-registration.

## Stopping (J2)

The session-3 decision applies the pre-registered J2 criterion to v0.2.0's own
val curve — a within-version comparison, therefore unaffected by the val-set
rebuild: if the deterministic val loss has not improved by ≥ 0.05 nats across the
preceding 1,000 steps, the run stops and the current checkpoint is final. The
decision and its evidence are recorded whichever way it goes.

## Forbidden claims (registered in advance)

1. **That compliance-cleaning improved the model.** The corpus changed and no
   v0.1.2 run exists on the v2b data, so every v0.2.0-vs-v0.1.2 quality
   difference is confounded with the corpus change and will be described that
   way.
2. **Tabulating 2.4366 beside any v0.2.0 number** without naming the val set on
   each row.
3. **Attributing any Abhimanyu-gap movement to the corpus rebuild.**
4. **Headlining the training-loop val curve.** It is a 20-batch in-training
   estimate and is noisy by design; the deterministic 100-batch number is the
   headline, and the training curve is published as context only.

## Deviations

Any change to a metric, threshold, protocol, or claim boundary after this commit
requires a hash-registered amendment — as with Amendment 1 to the promotion rules
— stating what changed and why. Results are reported regardless of outcome,
including P-C.
