# Pre-registration — X21: is the static-gate result real?

Registered: 2026-09-29, before any GPU time is spent. Related: X18, X19, X20,
D45, D47.

## Why this exists

X20's `micro-staticgate` scored **4.6304** against control's 4.7662, the gated
arm's 4.7226, and the branch-norm-only arm's 4.7680 — clearing its pre-registered
bar of 4.7544 by 0.1240. A gate that never reads its input beat one that does by
0.0922 nats, with 73,728 fewer parameters.

**That is 3.1× the largest effect any architecture arm has produced in this
project**, and every arm so far (X18, X19, X20) is single-seed. The previous best,
X18's 0.0436, was already shown not to belong to the mechanism it was credited
to. A larger effect from a *simpler* component has two live explanations — real, or
a single-seed artifact — and one run cannot distinguish them.

D45 exists because a threshold below the measurement's noise passes on nothing.
The same scepticism applies here with more force, not less, precisely because the
number is large.

## Design

**Paired, interleaved, four arms.** Control and treatment at the same two fresh
seeds, run in the same kernel so they share a machine and a session — the
discipline the modded-nanogpt records use, where a result of this size was
certified with 18 treatment runs interleaved with 9 control runs and a one-sided
t-test.

| Arm | Config | Seed |
|---|---|---|
| control | `micro-base` | 1338 |
| **staticgate** | `micro-staticgate` | 1338 |
| control | `micro-base` | 1339 |
| **staticgate** | `micro-staticgate` | 1339 |

Seed 1337 (the original pair) is retained as a third observation of each arm but
is **not** part of the test, because it is the run that generated the hypothesis
and including it would be circular. Four fresh arms, two per cell, paired by seed.

## Pre-registered predictions

| ID | Prediction | Threshold |
|---|---|---|
| **R-1** | The effect replicates at fresh seeds | mean paired Δ (staticgate − control) ≤ **−0.050**, i.e. at least half the 0.1358 first observed |
| **R-2** | It is not one lucky seed | **both** individual seed deltas ≤ −0.050 |
| **R-3** | The static gate is not a trivial constant | channel-wise std of the static gate > 1e-3, closing the gap X20 left open (H-2 unverified) |
| **R-4** | The static arm still beats the gated arm | mean staticgate ≤ mean gated (4.7226) |

R-2 is the one that matters. R-1 alone can be met by one seed going wild, and the
whole point of a replication is to refuse that.

## Honest priors, recorded before the run

- **Most likely: the effect is real but smaller than 0.1358.** A 500-step,
  30M-parameter, single-seed measurement that large is usually partly seed luck.
  Anything in the 0.05–0.10 range would still be the largest architecture effect
  we have and still worth shipping.
- **The genuine risk of a null is that X20 was right to be doubted.** We have
  three consecutive arms in this line and the last one produced a result too good
  to be typical. Prior on R-2 passing: genuinely uncertain, and the reason it is
  worth 0.8 hours of expiring quota.
- **Two seeds is the minimum honest test, not a strong one.** The modded-nanogpt
  standard is closer to twenty runs. If R-1 and R-2 pass, the result is
  "replicated at two seeds" and must be described that way, never as confirmed.
- Control at 1338/1339 also gives us, incidentally, the seed spread of the
  *control*, which we have never measured and which every other arm's delta in
  this project has been silently assuming.

## Decision rule (fixed now)

- **R-1 and R-2 both pass** → the static per-channel modulation is a real effect
  at our scale. Carry it into v0.3.0, described as **replicated at two seeds**,
  and report that the published data-dependent gate is dominated by a strictly
  simpler variant of itself.
- **R-1 passes, R-2 fails** → one seed is carrying it. Report as unreplicated and
  do not ship. This is the outcome the whole design exists to detect.
- **R-1 fails** → X20 was a single-seed artifact. Report that plainly, close the
  gated-residual line entirely, and record it as the third architecture arm to
  produce an effect that did not survive scrutiny.
- **R-3 fails** → the adapter collapsed toward a constant and the mechanism story
  is wrong even if the loss number holds. Report both halves.

## Cost

Four arms, 500 steps, 30M parameters. X20's single arm cost about 0.4 hours of
GPU; four is roughly 0.8 hours against the ~1.77 remaining before the 2026-10-03
reset. It fits with a small margin, which is why the design is four arms and not
six.

## Deviations

Any change to arms, seeds, steps, thresholds, or metrics after this commit requires
a hash-registered amendment. Results are reported regardless of outcome.

---

## Result (appended 2026-09-29, after the run)

Kernel `subhajitlucky/kalia-x21-replication` v2, COMPLETE. All four arms 500/500
steps, ~32k tok/s, no failures.

| seed | control | staticgate | Δ (static − control) | R-2 |
|---|---|---|---|---|
| 1338 | 4.8769 | 4.9623 | **+0.0854** | FAIL |
| 1339 | 4.8908 | 4.9087 | **+0.0179** | FAIL |
| **mean** | 4.8839 | 4.9355 | **+0.0516** | — |

- **R-1 FAIL** — mean paired Δ is **+0.0516**, i.e. the static gate was *worse*
  than control, against a registered bar of ≤ −0.050.
- **R-2 FAIL** — both individual deltas are positive. Not one seed is close.
- **R-4 FAIL** — static mean 4.9355 against the gated arm's 4.7226.
- **R-3 not measured.** The kernel did not emit the static gate's channel-wise
  std, so H-2's gap is still open. It is moot for the decision: R-1 and R-2 failed
  on the loss itself, and a gate can be a non-trivial constant in the wrong
  direction.

**X20 was a single-seed artifact, and the sign reversed.** The 0.1358-nat gain did
not merely shrink under replication — it inverted, at both fresh seeds, by
roughly a third to two-thirds of the original magnitude.

### The baseline variance, which is the actual finding

| config | control val loss |
|---|---|
| X18, seed 1337 | 4.7662 |
| X19, seed 1337 | 4.7680 |
| X20, seed 1337 | 4.7662 |
| X21, seed 1338 | 4.8769 |
| X21, seed 1339 | 4.8908 |

Two **independent sessions** at seed 1337 agree to **0.0018**. So the session
contributes almost nothing, and the **~0.11-nat** gap to the fresh pair is a
**seed** effect, not a machine or data effect.

But within the fresh pair the two seeds differ by only **0.0139**. A 0.11 gap
between 1337 and 1338 against 0.0139 between 1338 and 1339 says seed-to-seed
loss differences are not exchangeable: some seeds are much better than others.

Consequences, stated plainly:

1. The control is **not** seed-invariant. Every single-seed delta in this project
   was computed against an unmeasured baseline spread.
2. That spread (~0.11) **exceeds X18's entire −0.0436 "effect"** and X19's
   −0.0018 by an order of magnitude. Both were noise being read as signal.
3. X20's −0.1358 was the only delta above the spread, and X21 refutes it directly
   by sign flip. Its apparent 3.1× margin over X18 was a margin over noise.
4. **The architecture-ablation line has produced zero effects that survive
   scrutiny.** Three arms, three single-seed results, none reproducible.
