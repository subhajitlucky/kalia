# X27 — Reverse-Coordinated Causal Attention (RCAA), registered under the D48 protocol

**Status:** registered 2026-10-09, before any RCAA code exists in the repository.
**Depends on:** the design (`docs/plans/2026-09-28-v030-reverse-causal-attention-design.md`),
the Abhimanyu-gap metric (`eval_reversibility.py`), D48 (paired seeds), and the
X26 pilot (concurrent; unrelated mechanism).
**Replaces for this arm:** the design document's original absolute thresholds,
which predate D48 and the measured seed spread.

## Why this is screenable when other architecture arms are not

The Abhimanyu gap — reversed-text loss minus forward loss — is **5.1 nats at
30M** and **6.06 nats at 58M**. The measured seed spread is 0.11 nats (30M
loss) and 0.2892 (58M, confirmed at 0.2912). Every architecture arm that died
in D48 chased effects **smaller** than the spread; the gap is 20–50x larger.
That is what makes this arm legitimate where X18–X20 were not.

But the gap's *own* seed spread has never been measured, so it is measured
first, and every threshold below is 1x that spread — the D48 rule, applied
before the fact rather than after.

## Step 0 — the control's gap spread (prerequisite, ~1.5 GPU-h)

Three control arms at fresh seeds **1401/1402/1403**, micro budget (30M
parameters, 50M tokens, 500 steps — the standard screen), each with the
`--reversibility` readout. Output: the max pairwise gap spread across seeds.

**Stopping rule:** if the gap spread exceeds **0.50 nats**, the metric is not
screenable at this budget; stop and report that, do not run Step 1.

These three arms are also Step 1's controls: no separate control run.

## Step 1 — the paired micro screen (~1.5 GPU-h)

Two RCAA arms at the same seeds (1401, 1402), same data, same tokens, paired
against Step 0's controls per seed.

| ID | Prediction | Passes if |
|---|---|---|
| R-P1 | RCAA reduces the gap | gap <= control gap − 1x measured spread, at both seeds |
| R-P2 | forward quality does not collapse | forward val <= control + 1x measured spread |
| R-P3 | the gate opens (diagnostic, not a bar) | mean gate > 0.05 with >=1 layer > 0.10 |
| R-P4 | a real task moves, not just the metric | LAMBADA >= +2.0 pp **and** >= 2 sigma at the sample size actually run (the v030-am1 MDE rule; at 500 samples LAMBADA's sigma is ~1.8 pp, so this means >= 3.6 pp or more samples) |

**Falsification, registered now:** if R-P3 fails (gate stays closed), the
hypothesis is cleanly falsified regardless of the gap — the mechanism was never
switched on, and that is a wiring result, not a mechanism result. If the gate
opens and the gap still does not move, that is the more interesting negative:
the model uses the reverse view for something other than reversal. Both are
reportable; neither permits a promotion claim.

## Step 2 — full scale (registered, budget deferred)

Only if Step 1 passes. 58M RCAA vs the v0.2.0-recipe control, paired at >= 2
fresh seeds, full schedule. Estimated >= 13 GPU-h per seed pair; scheduled
across quota weeks, not committed by this document.

## Design frozen by this document

- Split the channel dim in half per RCAA layer; forward stream is standard
  causal MHA with RoPE; reverse stream flips the sequence, applies its own
  causal MHA with its own qkv/proj weights, flips back; concatenate and
  project. Reverse layers: **every other layer (1, 3, 5, 7, 9)**.
- The gate starts almost closed (bias -4, sigmoid ~0.018), so step-0 RCAA is a
  faithful approximation of the control architecture; a null is interpretable.
- QK-Norm and logit soft-cap stay on (the design's first-line stabilisers).
- Parameter budget at 58M scale: **<= 62M** (+7%), half-width heads in the
  reverse stream only.
- No distillation, no teacher, no borrowed weights (D2/D21 hold).

## What we expect

A skeptical prior, stated before the run: X16 showed the data transform does
not close the gap, and Hymba-1.5B is the standing example of novel structure
losing to a plain transformer with better training. The expected outcome is
R-P1 failing with the gate partially open — which would say the deficit is
deeper than a single reverse pass. The gate telemetry is why even that result
is worth the quota: it separates "never wired" from "wired and insufficient",
and no previous arm in this project could make that distinction.

## Cost

Step 0 + Step 1: five micro arms, ~2.5–3 GPU-h total. CPU evaluation free.
Step 2 deferred to its own budget decision.
