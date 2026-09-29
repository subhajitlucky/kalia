# Pre-registration — X18: Two 2026 Frontier Changes (NoPE, Gated Residual)

Registered: 2026-09-28, before any run of this experiment exists.
Related: D5 (screen on benchmarks, not loss alone), X17 (doc-mask),
`2026-09-28-why-27b-beats-397b.md`, `2026-09-28-open-source-landscape-and-transfers.md`.

## Why these two

Both come from the frontier survey, and both are cheap enough to screen honestly.
Neither is a large architectural gamble: they are the two changes with published
evidence that survive a cost check at 58M.

- **NoPE** — SmolLM3 (3B): drop rotary embeddings on every 4th layer. Their
  ablations on 3B/100B tokens found long-context gains at no short-context cost.
  The property we like: it *removes* machinery and is exactly parameter-neutral
  (verified: 29,920,512 with and without at micro dims), so a difference in
  result cannot be explained by capacity.
- **Gated Residual** — Qwen3.8-Flash-Next: normalise residual branches
  independently and read them through a data-dependent elementwise gate. Their
  reported split: widening the stream +1.58 accuracy, data-dependent read a
  further +1.98, while the *loss* gap was 0.002.

**Scope, stated before the run:** we implement only the **gating half** of
Gated Residual, on the existing width. The widening (four full-width branches) is
the expensive part and is not tested here. This arm therefore tests the
*mechanism*, not the capacity increase, and is named `gated` accordingly.

## Arms

| Arm | Config | Change | Params |
|---|---|---|---|
| **control** | `micro-base` | none | 29,920,512 |
| **nope** | `micro-nope` (`nope_interval: 4`) | RoPE dropped on layers 4 and 8 | 29,920,512 (identical) |
| **gated** | `micro-gated` (`gated_residual: true`) | gated residual read | 30,072,768 (+0.5%) |

Same 30M architecture otherwise, same seed 1337, same data, same 500 steps, same
LR schedule. One variable per arm, three arms total.

## Pre-registered predictions

| ID | Prediction | Threshold |
|---|---|---|
| **F-1** | NoPE matches or beats control on val loss | val loss ≤ control |
| **F-2** | NoPE costs no forward quality while helping long-range | forward val within +0.01, LAMBADA ≥ control |
| **F-3** | Gated residual improves accuracy by more than it improves loss | val loss ≤ control **and** ≥0.5 pp better on a downstream task |
| **F-4** | The gate actually opens during training | mean gate value at the end > 0.05 |

**F-3 is the important one.** It is the only prediction here that tests a claim
about *our own methodology* rather than a technique: Qwen reported a 0.002 loss
gap alongside a multi-point accuracy gap. If we reproduce that pattern at 30M, it
is direct evidence that D5 (select on benchmarks, not loss) was right — and it
would be the first result in this project that is about how to do science on a
58M model rather than about the model.

## Honest priors, recorded before the run

- **NoPE is the safer bet.** It is parameter-neutral and validated at 3B, but
  3B → 30M is a 100× scale drop and the ablation window here is 500 steps.
- **Gated residual is the more interesting bet and the likelier to be noise.** At
  30M with a rank-32 gate added to a 384-dim stream, the mechanism may simply be
  too small to matter. A null here is uninformative about Qwen's result at 125B.
- **Both arms will lose to control on absolute loss** if anything: the gate starts
  near-closed (mean 0.018) and must learn to open it, which costs early steps.
  We therefore judge on the *final* loss, not the curve shape.

## Decision rule (fixed now)

- **F-1 passes** → NoPE is a candidate for v0.3.0; it is free and safe.
- **F-3 passes** → the loss/accuracy divergence is reproduced at our scale, which
  is itself a publishable methodological result, independent of whether the
  mechanism ships.
- **Both fail** → published as a negative result, with the caveat above that a
  null at 30M says little about 125B.

## Cost

~1.5h GPU for three arms at 30M / 500 steps. Scheduled **before** session 3 of
v0.2.0 so it uses the free batch-GPU slot that session 2's completion released,
rather than delaying the main run. Total remaining quota after both: ~2.3h spare.

## Deviations

Any change to arms, steps, seed, thresholds, or metrics after this commit
requires a hash-registered amendment. Results are reported regardless of outcome.

## Results — read on 2026-09-28 after the run completed

`kalia-x18-ablate`, 3 arms, 500 steps, seed 1337, all from the same run.

| Arm | 100 | 200 | 300 | 400 | **500 (final)** | Δ vs control |
|---|---|---|---|---|---|---|
| `micro-base` (control) | 6.1428 | 5.2673 | 5.0237 | 4.8660 | **4.7662** | — |
| `micro-nope` | 6.1664 | 5.2825 | 5.0384 | 4.8797 | **4.7772** | **+0.0110** |
| `micro-gated` | 5.8664 | 5.2978 | 5.0448 | 4.7805 | **4.7226** | **−0.0436** |

### Against the registered predictions

| ID | Threshold | Measured | Verdict |
|---|---|---|---|
| **F-1** | NoPE val loss ≤ control | 4.7772 vs 4.7662 | **FAIL** — worse than control |
| **F-2** | forward val within +0.01, LAMBADA ≥ control | +0.0110; LAMBADA not measured | **FAIL** on the forward half (+0.0010 outside tolerance), LAMBADA half unmeasured |
| **F-3** | val loss ≤ control **and** ≥0.5 pp on a downstream task | 4.7226 ≤ 4.7662 ✓; accuracy not measured | **INCONCLUSIVE** — the accuracy half was never run |
| **F-4** | mean gate value > 0.05 | **0.0192** | **FAIL** — the gate never opened |

**F-4 is the finding that matters, and it invalidates the arm's own premise.**
Measured with `gate_probe.py` on real text: every one of the six gated blocks
sits at mean gate **0.0192**, against a designed start of 0.0180 and a
threshold of 0.05. The gate is inert. The checkpoint weights show why — `w2.bias`
moved from −4.0000 to −3.9342 in 500 steps, a change of 0.066 on a scale where
opening the gate usefully would require moving it by ~1 nat or more.

So **the −0.0436 nats is not attributable to the mechanism this arm was
registered to test.** Gated Residual's claim is that a *data-dependent* read is
where the gain comes from (Qwen: +1.98 accuracy on top of widening). Here the
gate is a near-constant 0.019 and carries almost no information. What the arm
actually varies is the **branch structure**: four slices of the stream, each
RMSNorm'd independently, concatenated back to full width before the sublayers.
That normalisation change is plausibly worth the loss improvement on its own,
and the +0.5% of parameters from the gate may be buying nothing at all.

**A design note in the code is wrong, and it is why.** Two separate problems,
both confirmed by reading the code and the trained weights:

1. *Near-closed is not the identity.* `GatedResidual.__init__` says the module
   starts "almost closed so the block begins as a faithful copy of the pre-norm
   residual path." A near-closed gate does **not** approximate the identity:
   `pre = b * g` with `g ≈ 0.019` is a near-zero tensor, not a copy of `x`. What
   rescues it is that `norm1` (RMSNorm) is scale-invariant and renormalises the
   stream back to unit scale — so the block runs on a branch-normalised
   re-expression of `x` whose direction the gate barely modulates. The near-closed
   init is not a safe no-op; it is a constant.
2. *The deliberate init never survived anyway.* `GatedResidual.__init__` sets
   `nn.init.zeros_(self.w2.weight)`, but `GPT.__init__` then calls
   `self.apply(self._init_weights)`, which re-runs `nn.init.normal_` over every
   `nn.Linear` and overwrites it — the trained `w2.weight` has mean magnitude
   0.0167, not 0. The mechanism's documented starting point is dead code.
   `test_gate_weight_init_is_overwritten_by_global_reinit` pins this down so it
   cannot be quietly repaired later without someone noticing that X18's
   measurement basis changed.

The near-closed behaviour survives only by accident: because `sigmoid` is
flattening near its floor, the random `w2` barely moves the mean (0.01798
measured at init against a 0.01799 design intent). The mechanism happens to start
closed. It was never actually designed to.

### Consequences

- **NoPE is rejected** (F-1). Its one favourable property survives — it is
  parameter-neutral and free — but at 30M / 500 steps it costs 0.011 nats. Note
  the honest prior recorded above: SmolLM3 validated NoPE at 3B, and 3B → 30M
  is a 100× scale drop, so this null says little about 3B.
- **Nothing is promoted to v0.3.0.** The decision rule promotes on F-1 or F-3;
  F-1 failed, and F-3 cannot be adjudicated without the accuracy half.
- **F-3 is completed by measurement, not by argument.** A CPU benchmark on the
  `micro-base` and `micro-gated` checkpoints is required to close the
  pre-registration properly. That is completing a registered criterion, not
  amending it, so no amendment is needed.
- **A better next experiment is now obvious and is not X18's question.** Isolate
  the two halves: an arm with the branch normalisation and the gate pinned to 1
  (no gate at all, no extra parameters). If that arm reproduces −0.0436, the
  gate is dead weight and the finding is a cheap normalisation win worth keeping
  in the residual path; if it does not, the gate is doing something the F-4
  measurement did not detect. Registered separately as X19.

### Caveats recorded against the result

- **Single seed.** X16 used two seeds and the preregistration ledger treats
  seed-paired results as the standard. 0.0436 is 4.4× the 0.010 bar, so a
  single-seed result at that margin is probably real, but it is not the
  two-seed evidence X16 was held to.
- **500 steps on 30M parameters** is a screening window, not a training run.
  Every conclusion here is "at 30M over 500 steps", which is what was
  registered.
- The `--reversibility` flag was not passed to `ablate.py`, so the Abhimanyu
  gap was not measured. It was not a registered criterion for F-1..F-4 and no
  conclusion here depends on it, but the notebook's own summary header
  advertised it, so the header overpromised.

## F-3 and F-4 completed — and the accuracy threshold was below our own noise

A CPU kernel (`kalia-x18-close`) supplied the two missing measurements. Both are
now closed.

**F-4, on 100 real 512-token validation batches** (not the 20 short probe
sentences used for the first estimate): mean gate **0.01917**, per-block
0.01916–0.01919, against a 0.01799 floor and a 0.05 threshold. The first estimate
(0.0192) was right. **F-4 FAIL**, confirmed on real validation text.

**F-3, both halves now measured:**

| task | control | gated | Δ |
|---|---|---|---|
| PIQA | 51.8 | 51.6 | −0.20 |
| ARC-Easy | 29.4 | 29.4 | +0.00 |
| HellaSwag | 27.0 | 27.8 | +0.80 |
| WinoGrande | 51.8 | 53.0 | **+1.20** |
| LAMBADA | 0.0 | 0.0 | +0.00 |

Val loss half passed (4.7226 ≤ 4.7662). Best accuracy delta **+1.20 pp**, so F-3
**passes on its registered terms** (≥0.5 pp).

### But the threshold is below the noise floor, so that pass means nothing

The standard errors on these measurements, at `limit=500`:

| task | stderr |
|---|---|
| PIQA | 2.24 pp |
| ARC-Easy | 2.04 pp |
| HellaSwag | 1.99–2.01 pp |
| WinoGrande | 2.24 pp |

**F-3's registered 0.5 pp threshold sits ~4.5× below the measurement's own
standard error.** The observed +1.20 pp on WinoGrande is **0.54σ** — the expected
size of noise, and indistinguishable from it. A 0.5 pp bar at this sample size
passes roughly half the time by construction.

So F-3 "passes" in the letter and carries no information. Two things follow, and
the first is uncomfortable:

1. **The pre-registration is at fault, not the mechanism.** The threshold was
   chosen as a round number without ever being compared against the benchmark's
   standard error at `limit=500`. This is recorded against X18 honestly. It is
   the same class of error as a threshold set without reference to the metric's
   resolution — and we have now made it twice, in the accuracy direction.
2. **This contaminates the v0.2.0 promotion rule.** S-A allows no task to
   regress by more than **1.0 pp** — also below the ~2.0–2.2 pp noise floor at
   `limit=500`. S-A therefore cannot distinguish a real 1 pp regression from a
   coin flip. The interim measurement's ARC-Easy −4.6 and LAMBADA −4.6 *are*
   outside noise and stand as real regressions; PIQA −0.2 and WinoGrande −0.8
   are inside it and do not.

**The registered thresholds are not changed.** S-A is evaluated on its 1.0 pp
terms exactly as written, and the noise floor is reported *alongside* the result
rather than used as an escape hatch — the same handling P-C was given. What
changes is every *future* pre-registration: accuracy thresholds must be stated
either above the measured standard error or with enough samples to bring it
below the bar. See D45.

### Where X18 actually leaves us

Both signals that could have supported the mechanism are now measured, and
neither supports it: the gate is inert (F-4), and the accuracy "gain" is noise
(F-3). The −0.0436 nats loss improvement remains real and remains unattributed.
X19 exists precisely to attribute it, and its G-1 threshold is stated in nats
(0.010), which is unaffected by this problem — the benchmark half of X18 is the
part that was under-powered.


