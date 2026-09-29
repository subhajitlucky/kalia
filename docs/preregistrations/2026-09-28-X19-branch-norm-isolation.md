# Pre-registration — X19: which half of the gated arm earned the −0.0436?

Registered: 2026-09-28, before any run of this experiment exists and before any
GPU time is spent on it. Related: X18, I15, D5, D42,
`2026-09-28-v020-final-evaluation.md`.

## The question

X18's `micro-gated` arm finished **0.0436 nats ahead of control** — 4.4× the
registered 0.010 bar, the largest effect any architecture arm has produced in
this project. It is also unattributable. F-4 measured the gate at a mean of
**0.0192** against a designed floor of 0.0180 and a bar of 0.05, so the
data-dependent read that Gated Residual actually claims was never operating.

That leaves two candidate explanations for the gain, and they imply opposite
actions:

1. **The branch structure earns it.** `GatedResidual` splits the residual stream
   into four slices, RMSNorm's each independently, and concatenates them back to
   full width before the sublayers. Because RMSNorm is scale-invariant, a
   near-closed gate makes this reduce to *per-quarter renormalisation of the
   residual stream* before attention. That is a real architectural change and a
   plausible source of a genuine loss gain.
2. **The gate earns it, and F-4's statistic was the wrong one.** A gate can sit
   at a low mean and still be strongly data-dependent. Mean is a first moment;
   it says nothing about variance across channels or across documents. A gate
   that is 0.019 on average but ranges over 0.001–0.5 by channel is doing
   substantial work while looking "closed" on average.

If (1), the gate is 152,256 dead parameters (+0.5%) and should be deleted. If
(2), F-4's interpretation was wrong and the mechanism is real. These lead to
opposite conclusions, so the experiment has to distinguish them rather than assume.

## Arms

| Arm | Config | Change | Params |
|---|---|---|---|
| control | `micro-base` | none | 29,920,512 |
| (already run) gated | `micro-gated` | branch norm + gate, `w2` init overwritten to N(0,0.02) | 30,072,768 |
| **branchnorm** | `micro-branchnorm` | branch norm, **no gate** | **29,920,512** |

`branchnorm` applies the four-way independent normalisation and concatenation and
then multiplies by a constant 1.0. It has *no* gate parameters at all, so if it
matches `gated`, the gate's entire +0.5% is dead weight.

Control and gated are **not re-run**: they are X18's existing arms, same seed
1337, same data, same 500 steps, same commit, and re-running them would spend
GPU to reproduce numbers we already have and whose logs are published. X19 runs
one new arm. That is the entire experiment.

## Pre-registered predictions

| ID | Prediction | Threshold |
|---|---|---|
| **G-0** | The gate is not constant, so F-4's mean understates it | Report gate **std across channels** and **std across documents**, not just the mean. No pass/fail — this is a measurement that corrects a possible error in our own F-4 interpretation, and it is registered so we cannot quietly pick the flattering statistic afterwards |
| **G-1** | The branch normalisation earns the gain | `micro-branchnorm` val loss ≤ midpoint(control, gated) + 0.010, i.e. it captures **≥ half** the 0.0436 |
| **G-2** | The branch norm is free | `micro-branchnorm` params **== 29,920,512**, exactly equal to control |
| **G-3** | The residual stream is genuinely rescaled | Per-quarter RMS after branch norm is ~1.0 and differs from the control's per-quarter RMS by more than 1% |

G-2 and G-3 are correctness checks on the implementation, not findings. They are
registered anyway because every prior version of this experiment's mechanism has
turned out to be not doing what its code comment said (I15, and the
`nn.init.zeros_` overwrite in X18), and a silently-misconfigured arm would
produce a clean, confident, wrong number.

## Honest priors, recorded before the run

- **Most likely: G-1 passes.** F-4's measurement is direct — 0.0192 on all six
  blocks, uniform — and `w2.bias` moved 0.066 in 500 steps. The gate barely
  moved. The branch structure is the only part of the arm with a plausible
  mechanism.
- **The real uncertainty is G-0, and it cuts against me.** I concluded from a
  mean that the gate is inert. If the gate turns out to have high channel-wise
  variance, that conclusion is wrong and X18's result stands as a genuine
  reproduction of Qwen's claim. I am registering G-0 precisely because I do not
  trust my own first-moment argument, and the cheapest way to find out is to
  measure the second moment on a checkpoint we already have — no GPU needed.
- **Micro context may distort this.** At `context_len 512` with documents ~200
  tokens, the residual-stream statistics at 30M are not the 1024-token, 58M
  statistics we would ship. The X17 pre-registration records the same caveat and
  it applies here.
- **Single seed, as with X18.** If G-1 passes by a wide margin that is
  informative; if it lands within ~0.01 of the midpoint it is a coin flip and
  must be reported as a null, not rounded into a win.

## Decision rule (fixed now)

- **G-1 passes and G-2 holds** → the branch normalisation is a real, free win.
  Promote the *branch norm*, drop the gate, and carry the normalisation into
  v0.3.0 as a 0-parameter change. Report the gate as measured dead weight.
- **G-1 fails** → the gain lives in the gate after all, which means F-4's
  interpretation was wrong. Re-open the X18 conclusion and treat the mechanism as
  unproven rather than disproven.
- **G-0 shows high gate dispersion** → F-4's threshold was a bad statistic and
  the pre-registration itself is partly at fault. Record that against X18
  honestly; the prediction was badly chosen, not the mechanism.
- **Any of G-2/G-3 fail** → the arm is misconfigured and its loss number means
  nothing. Report that, not the loss.

## Cost

One arm, 500 steps, 30M parameters. X18's three arms cost well under one GPU-hour
in total, so this is roughly 20–30 minutes of quota. Scheduled **after** the
v0.2.0 final evaluation is read, and within the ~2.58h of GPU quota left before
the 2026-10-03 reset.

## G-0 measured on 2026-09-28, before any GPU was spent

Run on X18's existing `micro-gated` checkpoint (30M, 500 steps, seed 1337) — no
new training, no quota. The new `gate_stats` in `gate_probe.py` reports the
second moments alongside the mean, and its two new tests build a known-constant
gate and a known-data-dependent gate to confirm the statistic separates them
before it is trusted on a real checkpoint.

| quantity | value | reading |
|---|---|---|
| mean gate | 0.019172 | floor is 0.017986; F-4 bar 0.05 |
| max std **across channels** | 2.96e-04 | 1.5% relative — a near-uniform reweighting |
| max std **across inputs** | **5.09e-06** | **the gate does not respond to its input** |

Per-block input spread runs 1.0e-06 to 5.1e-06 across all six blocks.

**G-0's answer: F-4's reading stands, and it survives the challenge.** I
registered this specifically because I did not trust my own inference from a
first moment — a gate can be low on average and strongly variable, in which case
F-4's threshold was the wrong statistic and the mechanism was fine. The
dispersion measurement rules that out. The gate is not data-dependent at any
useful scale: 5e-06 of variation in response to input, against a mean of 0.019,
is a constant, not a gate. The small channel spread is a fixed per-channel bias,
not per-input modulation.

So the mechanism Gated Residual actually claims — a *data-dependent* read, worth
+1.98 accuracy in Qwen's split — **is not operating in our arm at all.** What
remains is the branch structure, and X19's G-1 becomes the decisive test rather
than a formality: with the gate established inert in both moments, the −0.0436
either survives without it or it does not exist.

Recorded before the run, as required. The prior recorded in this document — "the
real uncertainty is G-0, and it cuts against me" — resolved against me.


## Deviations

Any change to arms, steps, seed, thresholds, or metrics after this commit
requires a hash-registered amendment. Results are reported regardless of outcome.
