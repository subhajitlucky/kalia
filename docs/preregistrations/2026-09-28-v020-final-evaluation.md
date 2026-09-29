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


## Results — read on 2026-09-29

Both registered metrics, on the final step-4,770 checkpoint, on the canonical v2b
held-out set.

### P-A / P-B / P-C — deterministic protocol

`eval_val.py --batches 100 --batch-size 8 --seed 1234`, 819,200 tokens, the same
call that produced v0.1.2's 3.0533.

| | v0.1.2 | v0.2.0 | Δ |
|---|---|---|---|
| **val loss** | 3.0533 | **2.8248** | **−0.2285** |
| bits per byte | 0.9992 | **0.9244** | −0.0748 |

- **P-A PASS** — 2.8248 ≤ 3.0533.
- **P-B PASS** — 2.8248 ≤ 3.0033. An improvement of **0.2285 nats, 4.6× the
  registered 0.05 bar.** This is a real gain in the primary metric.
- **P-C not triggered.**

The in-training `eval_steps 50` figure was 2.8832 at step 4750. It is *not* the
headline and was not used to decide anything; the deterministic number is the one
compared to the reference.

### S-A — the five zero-shot tasks

| task | v0.1.2 | v0.2.0 | Δ | stderr | σ | S-A (≤1.0pp) |
|---|---|---|---|---|---|---|
| PIQA | 61.4 | 63.8 | **+2.40** | 2.15 | 1.1 | ok |
| HellaSwag | 36.8 | 39.8 | **+3.00** | 1.99–2.21 | 1.5 | ok |
| WinoGrande | 50.2 | 50.8 | +0.60 | 2.24 | 0.3 | ok |
| LAMBADA | 23.0 | 20.8 | **−2.20** | 1.82 | 1.2 | **FAIL** |
| ARC-Easy | 45.8 | 42.0 | **−3.80** | 2.21–2.23 | 1.7 | **FAIL** |

- **S-A FAIL**, on two of five tasks.

### What the interim measurement would have told us, and why it must not be used

At step 3,470 the same five tasks read: PIQA −0.2, ARC-Easy −4.6, HellaSwag +1.8,
WinoGrande −0.8, LAMBADA −4.6 — four of five regressing, which reads as a
catastrophic corpus regression. The final checkpoint reversed three of those
(PIQA −0.2 → +2.40, WinoGrande −0.8 → +0.60, HellaSwag +1.8 → +3.00) and left
ARC-Easy negative. **The interim benchmark was actively misleading**, and any
conclusion about the corpus has to come from the final checkpoint.

### The noise floor, applied to this result rather than used to escape it

D45 established that these benchmarks carry ~2.0–2.2 pp standard errors at
`limit=500`. The registered S-A allowance of 1.0 pp sits below that, so S-A cannot
in principle distinguish a 1 pp regression from a coin flip.

That does not rescue the result and is not used to. Read against the measured
noise: **three of the five tasks are not separable from v0.1.2 at all** — PIQA,
HellaSwag and WinoGrande all move by less than their standard error. The two
failures are ARC-Easy at ~1.7σ and LAMBADA at ~1.2σ; neither is overwhelming on
its own, but they point the same direction and are not independent of the
underlying cause.

So the honest summary is: **a clear, well-registered improvement in held-out
loss, no measurable change in most zero-shot accuracy, and a real but modest
signal of regression on the two tasks that depend most on knowledge breadth
(ARC-Easy) and long-range recall (LAMBADA).** The thresholds stand as registered
and the verdict is FAIL on S-A.

### One thing the LAMBADA regression explains

The long-form probe measured the mixture before this run finished: only **17.7%
of training tokens sit inside a document at least as long as the 1,024-token
context**, and TinyStories — 20% of the mixture — supplies **0.01%** of them.
LAMBADA is precisely the task that requires holding a discourse and recalling its
end, so a −2.20 there is the predicted direction for this corpus rather than a
mystery. The same probe found a source at 99.3% (Gutenberg, MIT-declared), which
makes the remedy measured rather than speculative.

Whether the code-share swap caused the ARC-Easy regression is **not** established.
It is the prime suspect and it is the open question that v0.3.0 has to answer.

## Deviations

Any change to a metric, threshold, protocol, or claim boundary after this commit
requires a hash-registered amendment — as with Amendment 1 to the promotion rules
— stating what changed and why. Results are reported regardless of outcome,
including P-C.
