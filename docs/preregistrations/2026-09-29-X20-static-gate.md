# Pre-registration — X20: was it ever a gate?

Registered: 2026-09-29, before the arm is implemented, before any GPU time is
spent on it. Related: X18, X19, X19 Amendment 1, D46.

## The question X19 created

X19 removed the gate from Gated Residual and the gain vanished. `micro-branchnorm`
scored **4.7680** against control's 4.7662 and gated's 4.7226, missing the
pre-registered G-1 bar of 4.7544 by 0.0136. The branch normalisation contributes
nothing.

But G-0 had already measured the gate as **non-responsive to its input** — a
standard deviation of 5e-06 across documents against a mean of 0.019. Those two
results are both correct and they do not sit together: a component cannot be
irrelevant to its input and load-bearing at the same time unless it is doing
something other than what it is named.

The candidate explanation: with `g ≈ 0.019` the expression

```
pre = concat( RMSNorm(x_i) * sigmoid(w2 · silu(w1 · x)) )
```

is a **learned, static, per-channel modulation** of the normalised stream. G-0
already found the gate is not constant — channel-wise spread is 2.96e-04, about
1.5% relative. It simply does not vary with the input. What we built and measured
is therefore plausibly a **rank-32 learned low-rank adapter on the residual path**,
and the "gating" is vestigial.

## The arm

| Arm | Config | Change | Params |
|---|---|---|---|
| control | `micro-base` | none | 29,920,512 |
| (X18) gated | `micro-gated` | branch norm + gate that reads `x` | 30,072,768 |
| (X19) branchnorm | `micro-branchnorm` | branch norm, no gate | 29,922,816 |
| **static** | `micro-staticgate` | branch norm + a gate with **no** input | **29,996,736** |

The static gate replaces `w1: Linear(dim, rank)` with a learned constant of size
`rank`, keeping `w2: Linear(rank, dim)` and the same `bias = −4.0` start, so the
arm begins at the same near-closed magnitude as the gated one and is otherwise
identical. It is a **strictly smaller** module than the gated arm by 76,032
parameters, which is the point: if the smaller one matches, the input path is not
where the work is.

## Pre-registered predictions

| ID | Prediction | Threshold |
|---|---|---|
| **H-1** | The static adapter reproduces the gain, so the gate was never a gate | `micro-staticgate` val loss ≤ **4.7544** (the same G-1 bar, reused) |
| **H-2** | The learned static gate is not itself trivial | channel-wise std of the gate > 1e-3, i.e. it is not collapsing to a constant |
| **H-3** | The static arm lands strictly between the two known arms | control < static ≤ gated, i.e. 4.7226 ≤ static ≤ 4.7662 + 0.005 |

## Honest priors, recorded before the run

- **Most likely: H-1 passes.** The reasoning is that if a component's learned
  per-channel pattern is what matters, then removing only the input dependence
  should leave the gain intact. Nothing in G-0 or G-1 contradicts this, and the
  static arm is a strict subset of the gated arm's capacity.
- **The real risk is the opposite:** that the gated arm's gain needs the *product*
  of both the input path and the low-rank structure together, in which case H-1
  fails and the gated mechanism is real after all — just not for the reason its
  name suggests. That would be the more interesting result.
- **A smaller module is not a controlled comparison in the strict sense.** It is
  fewer parameters doing the same job, which is a *better* thing if it works, but
  it also means a null is not fully conclusive about the gated arm. Recorded now
  so it cannot be argued later in whichever direction is convenient.
- Single seed, 30M parameters, 500 steps — the same screening window as X18/X19,
  with the same single-seed caveat those runs carry.

## Decision rule (fixed now)

- **H-1 passes** → the data-dependent read contributes nothing; the finding is
  published as *"the mechanism did not transfer, the low-rank residual adapter
  did"*, and the static adapter is the version worth keeping.
- **H-1 fails** → the gate needs its input after all, G-0's measurement is
  re-opened as wrong, and the gated arm stands as a genuine mechanism. X18 and X19
  are then a story about a cheap diagnostic (mean and dispersion) being able to
  falsify a borrowed claim, which is still worth publishing.
- **H-2 fails** → the adapter learned a near-constant, i.e. the whole module
  collapsed toward a no-op and the −0.0436 is a seed artifact. Report as a null
  and stop.

## Cost

One arm, 500 steps, 30M parameters — the same footprint as X19, which took about
0.4 hours of GPU. Roughly 20–30 minutes of the quota remaining before the
2026-10-03 reset.

## Deviations

Any change to arms, steps, seed, thresholds, or metrics after this commit requires
a hash-registered amendment. Results are reported regardless of outcome.

## Results — read on 2026-09-29

One arm, 500 steps, seed 1337, same corpus and lineage as X18/X19.

| arm | params | val loss | vs control |
|---|---|---|---|
| control (`micro-base`) | 29,920,512 | 4.7662 | — |
| branchnorm (`micro-branchnorm`, X19) | 29,922,816 | 4.7680 | +0.0018 |
| gated (`micro-gated`, X18) | 30,072,768 | 4.7226 | −0.0436 |
| **staticgate** (`micro-staticgate`) | **29,999,040** | **4.6304** | **−0.1358** |

`micro-staticgate: 100=5.9919 200=5.3744 300=5.0679 400=4.9119 500=4.6304`

| ID | Threshold | Measured | Verdict |
|---|---|---|---|
| **H-1** | ≤ 4.7544 | **4.6304** | **PASS**, by 0.1240 |
| **H-2** | channel-wise std > 1e-3 | not emitted by the kernel | **unverified** |
| **H-3** | 4.7226 ≤ static ≤ 4.7712 | 4.6304 | **FAIL** — below the gated arm |

G-2 re-verified on the trained checkpoint: **29,999,040 parameters exactly**, zero
input-path tensors. Amendment 1's corrected figure is confirmed against the
constructed model.

### What this shows

**A gate that never reads its input beats one that does, by 0.0922 nats, with
73,728 fewer parameters.** And it beats control by 0.1358 — **3.1× the entire
gated-arm gain** — from a mechanism that is not in the published technique.

The data-dependent read is not merely inert at our scale (G-0). It is
**actively harmful**: removing the input path makes the same module substantially
better. Whatever Gated Residual is doing, it is not the thing it is named for, and
the part that helps is a static per-channel modulation that the paper does not
describe.

**H-3 failed in the favourable direction.** It was registered as "lands strictly
between the known arms" and the result lands below both. Recorded, not argued away.

### The honest caveat, and why this is not yet a result we can use

**0.1358 is 3.1× the largest effect any architecture arm has produced in this
project.** The previous best, X18's 0.0436, was already shown to be unattributable
to its stated mechanism. An effect three times larger, from a component strictly
*simpler* than the thing it beats, has two available explanations and this run
cannot distinguish them:

1. It is real and large.
2. It is a single-seed artifact.

Every X18/X19/X20 arm is single-seed. We rejected a 0.0436 effect on precisely
that basis, and the pre-registration of this document recorded that a null would
have been the *more* interesting result — which is itself a sign the prior was
weak. D45 exists because we learned the hard way that a threshold below the
measurement's noise passes on nothing; the same scepticism applies here with more
force, not less, precisely because the number is large.

**So H-1 is reported as passing and nothing is promoted.** Replication is
pre-registered separately, paired against control at fresh seeds, which is the
discipline the modded-nanogpt records use for a result of this size: interleaved
unseeded repeats of arm and control on the same machine, with a stated test.

**H-2 remains unverified** — the kernel did not emit the channel-wise dispersion of
the static gate, so it is not known whether the adapter learned a non-trivial
pattern or collapsed toward a constant. That is a real gap in the evidence for the
mechanism and is closed by the replication.

