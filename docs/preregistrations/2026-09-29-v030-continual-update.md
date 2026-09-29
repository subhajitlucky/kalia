# v0.3.0 — the first continual update, with the Kautilya mixture arm

**Status:** registered, not run
**Date registered:** 2026-09-29
**Supersedes:** nothing. **Depends on:** v0.2.0 (public, https://huggingface.co/kalia-lm/kalia-v020)
**Blocked until:** GPU quota resets 2026-10-03T00:00:00 (0.61h remaining, insufficient for any arm below)

---

## Why this document exists

We are short on GPU, we have a quota rather than a budget, and the last thing we
nearly shipped was a single-seed result that inverted under replication (X20/D48).
This registration therefore fixes more than usual in advance: thresholds, seeds,
arms, stopping rules, and the specific things that would make us abandon the
project's headline.

It also registers a **negative expectation**. See "What we expect to find."

## The one finding this is built on

D48 measured something we had never measured: **the control is not seed-invariant.**
Seed 1337 gives control val loss 4.7662 (two independent sessions, 0.0018 apart);
seeds 1338/1339 give 4.8769/4.8908. That is **~0.11 nats** of baseline spread at
30M micro scale.

Two consequences govern everything below:

1. **A single-seed delta is unreadable.** Every architecture arm in X18–X20 was
   compared against an unmeasured baseline. X18's −0.0436 and X19's −0.0018 are
   both smaller than the spread and are now classified as nulls.
2. **We do not know the spread at 58M.** The 0.11 figure is a 30M measurement. Any
   full-scale threshold set before measuring the 58M spread is a guess.

So **Step 1 below is a baseline measurement, not an experiment.** It has no
hypothesis and no threshold; it exists so that Steps 2–4 can be interpreted.

## What 0.3.0 is, precisely

**Not a new architecture.** Muon-embedded RMSNorm, RoPE, QK-Norm, logit
soft-capping, SwiGLU — all unchanged from v0.2.0. D48 closed the architecture
line and nothing here reopens it.

**New training schedule.** LR re-warm from the decayed floor back toward peak,
then re-decay over the update budget. This is Ibrahim et al.,
*"Simple and Scalable Strategies to Continually Pre-train Large Language Models"*
(arXiv 2403.08763) — the one paper in our design whose abstract we have read in
full. Their claim, verbatim from the abstract: LR re-warming, LR re-decaying and
replay of previous data "is sufficient to match the performance of fully
re-training from scratch," at 405M and 10B.

**New data.** The long-form chunked corpus (`prep_longform.py`, probed: 99.27% of
`sedthh/gutenberg_english` documents ≥1024 tokens, median document 104,719 tokens
requiring sentence-aware chunking) plus a replay shard of v0.2.0's corpus.

**One arm of our own.** The Kautilya adaptive mixture, from
`docs/research/2026-09-23-ancient-texts-algorithms.md` (A1).

## The honest scale mismatch

The recipe we are adopting is demonstrated at **405M and 10B**. We are 58M — an
order of magnitude below the smallest validated scale. No replay ratio has been
published at our size. **We are not adopting a ratio; we are measuring two.**

An earlier version of this design cited MIITA (arXiv 2607.22556) for the claim
that replay degrades as backbones shrink, with a 0.6B floor. **That was wrong and
is retracted (D49).** MIITA is a memory-based inference-time method that works
"without backbone updates." The sweep below now stands on the correct and weaker
ground: nothing verified covers 58M, so nothing can be adopted.

---

## The four steps

### Step 0 — CL-0 forgetting baseline (prerequisite, ~0.1h)

`tools/forgetting_probe.py` **had never been run.** It is now executed end-to-end
on a CPU checkpoint as part of `test_forgetting_probe.py` (6 tests), which is how
a real bug was found before spending any GPU: the probe's bits-per-byte field
divided by a hardcoded **4.4086** bytes-per-token, while our corpus measures about
**3.38**. Every bpB the forgetting ledger had ever reported was ~30% low, and
that ledger is the artifact Steps 2–4 are scored against. bpB is now measured with
`eval_val.bytes_per_token` and the divisor is recorded next to the value, so the
number is auditable. The probe has still never seen a real checkpoint — that is
what Step 0 on Kaggle is for.

Without it we have no measure of whether a continual update preserved anything,
which would make the whole exercise uninterpretable. This is the cheapest
high-value thing in the plan and it is a hard prerequisite.

**Registered expectation:** the probe will show measurable forgetting on held-out
val after any update, because that is what continual updates do. If it shows
*none*, the probe is broken and we stop and fix the probe.

### Step 1 — Baseline spread at 58M (not an experiment, ~0.6h)

Two control arms at fresh seeds (1401, 1402), full v0.2.0 recipe, 500 steps,
paired in one kernel.

**Output:** the 58M control's own seed spread. **No threshold. No verdict. This
number sets the thresholds for Step 3.** If it is larger than ~0.10 nats, every
threshold below is judged against that and not against its own number.

We are spending a sixth of the monthly quota to avoid the mistake that cost us
X18, X19 and X20.

### Step 2 — Re-warm vs no re-warm (the borrowed recipe, ~0.6h)

Two arms, same fresh seed (1401), same data, same tokens:

- **A (control):** continue from v0.2.0 at the current decayed LR. No re-warm.
- **B (treatment):** re-warm to peak, then re-decay. Ibrahim et al.'s recipe.

This is the only place in 0.3.0 where we are testing someone else's claim rather
than our own. **If B does not beat A by more than the Step 1 spread, we report
that the recipe does not transfer to 58M** and say so in the model card. That is a
publishable result and we will not treat it as a failure.

### Step 3 — Replay ratio sweep (the open question, ~0.7h)

Three arms, same fresh seed (1401), re-warm schedule fixed by Step 2's outcome:

- 0% replay (control), 10% replay, 40% replay

**Our prior is genuinely split.** The 2026-09-23 backlog carries tree-replay
ratios from larger models; nothing is validated here. This is a measurement, not
a confirmation.

### Step 4 — Kautilya adaptive mixture (our own arm, ~0.7h)

**Source principle** (Arthashastra, resource doctrine): allocate to provinces by
strategic return, responding with sama (keep), dana (upsample), bheda
(downsample), danda (freeze). **Mechanism:** re-estimate the four sources' weights
every N steps from marginal loss improvement.

Two arms, same fresh seed (1401), same total tokens:
- **static:** the v0.2.0 mixture, 60/20/15/5 (FineWeb-Edu / TinyStories /
  Cosmopedia v2 / Python)
- **adaptive:** weights re-estimated online

**Known implementation cost, registered now so it is not a surprise:** the
current data path **pre-blends shards offline** via `mix_bins.py` into a single
`.bin`, and `data.py` memory-maps one file. Runtime per-source weighting does not
exist. It requires a multi-shard loader with per-source sampling and a way to
attribute per-source val loss.

**Status update, 2026-09-29 (after registration, before any run).** `mixture.py`
now implements `SourceMixtureDataset` (per-row source sampling at runtime weights,
same `context_len`/`__len__`/`get_batch` interface as `TokenDataset` so
`train.py` is unchanged), `per_source_loss` (held-out loss per source — *not*
training loss, which is confounded by how much of that source was just sampled),
and `update_weights_from_signal` implementing sama/dana/bheda/danda. 21 CPU tests
in `test_mixture.py`, deterministic.

Two design bugs were found and fixed while testing it, both recorded because both
would have produced a plausible, meaningless number:

1. **A source at the mean loss was labelled `dana`, not `sama`.** With `<=`,
   identical losses across sources labelled every source as "improving" — a policy
   reporting progress while having none, which normalises to a no-op. Now strict.
2. **Floor-without-ceiling has no restoring force.** After 500 updates against a
   persistently-better source, weights reached 0.97/0.01/0.01/0.01. That is not a
   mixture; it is a single-source run wearing a mixture's name. A naive
   clamp-then-renormalise did not fix it (0.8 ceiling produced 0.87, because
   dividing by a sum below 1 pushes entries back up) and iterating it made the
   bounds *infeasible* (0.8 + 3×0.01 = 0.83 < 1). Replaced with a proper
   Euclidean projection onto {sum=1, lo≤v≤hi} by bisection on a uniform shift,
   with the postcondition asserted rather than assumed, and infeasible bounds
   raising instead of silently violating.

**Status update 2, 2026-09-29.** The arm is wired and tested. `train.py` takes
`--sources / --source-bins / --source-weights`, re-estimates weights every
`adaptive_every` steps from held-out per-source loss, and writes
`per_source_log.csv` (per-source loss, resulting weight, and the strategy applied
to each source) so the policy's decisions are auditable rather than inferred.
`adaptive_every: 0` gives the static control from the same code path.
`configs/kalia-kautilya.yaml` is diff-locked to `kalia-v020.yaml` outside an
explicit allow-list, so the only difference between the two arms is whether the
policy runs.

Three bugs found while wiring, each of which would have produced a plausible
number rather than an error: `per_source_loss` called the model with the wrong
signature (KALIA's `GPT.forward` takes `targets` and returns `(logits, loss)`);
the base `TokenDataset` was built unconditionally so a per-source run still
required a `train.bin` it never read; and `validate_weights` referenced a
non-existent local `seed`. The first was only caught because an integration test
runs the real `train.py` on CPU.

**Falsification also exposed a gap the unit tests missed:** replacing
`per_source_loss` with a constant left every test green, because the toy models
were uniform enough to make a constant indistinguishable from the truth. The
policy would have run on a signal that was not there. Now asserted directly.

**Status update 3, 2026-09-29 — the re-warm mechanism was broken, and the
notebooks found it.** Five execution notebooks now exist
(`kalia-v030-cl0`, `-baseline`, `-rewarm`, `-replay`, `-kautilya`), one per
registered step so a stopping-rule trigger costs one kernel and not the sequence.
Seven arm configs are written, diff-locked to v0.2.0 so each pair differs in
exactly the key its claim depends on.

Building them surfaced a defect that would have voided Step 2 without producing
any error. **`train.py --resume` continues the source run's cosine schedule.**
Resuming v0.2.0 at step 4770 of 4770 means `base_lr_scale` returns
`min_lr_ratio` for every remaining step, so the re-warm arm would have been
byte-identical to its control, run the recipe at LR multiplier 0.100, and
reported "the recipe does not transfer at 58M" — a null for a mechanism that was
never switched on. Measured: 0.100 flat for the whole update, versus
0.02 → 0.97 → 0.10 once rebased.

Fixing it took three attempts and two of the intermediate states were worse than
the original:

1. Rebasing the schedule **unconditionally** also re-warmed the no-re-warm
   control. Both arms peaked at 0.00060 — two identical arms and a meaningless
   null.
2. Gating the step budget on `rewarm` gave the control a budget of 40 against a
   `start_step` of 60, so it trained **zero** steps while the treatment trained
   40. One arm moved, one did not.
3. Making the budget origin-relative for *any* resumed run broke the smoke test
   that asserts `--max-steps` means an absolute step ceiling, and would have
   silently changed how every already-recorded resume behaves.

The resolution keeps both meanings distinct: `--max-steps` stays absolute, and a
new config key `update_budget` means "N more steps", set on the v0.3.0 arms.
`rewarm` now defaults to **False**, because a plain `--resume` must not change
the schedule of runs whose results are already published.

Verified end-to-end by running both arms: same 40 update steps, re-warm peaking
at 0.00060 against the control's 0.00006 — a 10× difference in the learning rate
the treatment actually trains at. 12 tests in `test_rewarm_schedule.py`, including
four that run real resumes; falsified by re-introducing the unconditional rebase.

**What is still not done:** nothing here has run on real data or on a GPU. It is
code-complete and unit-tested, which is a strictly weaker claim than "validated".
Step 1's variance baseline remains the prerequisite, and it is the number every
later threshold is expressed as a multiple of.

---

## Thresholds, fixed now

| Step | Claim | Passes if | Fails if |
|---|---|---|---|
| 0 | probe works | forgetting measurable | probe shows nothing |
| 1 | — | — | spread > 0.15 → re-derive all thresholds |
| 2 | re-warm helps (Ibrahim) | Δ ≤ −1× spread | Δ within ±spread → **no transfer** |
| 3 | ratio matters | best ratio beats 0% by > spread | all within spread → ratio insensitive here |
| 4 | adaptive beats static | Δ ≤ −1× spread | within spread → static mixture is fine |

**Every "passes" bar is 1× the measured Step 1 spread, not an absolute number.**
This is the D48 lesson encoded. Where an absolute bar is also registered, it is
marked and must not be quietly replaced.

**Secondary metric for Step 4:** if adaptive and static tie on loss, the tie-break
is whether the adaptive arm's per-source losses are more balanced. A mechanism
that equalises sources without improving the total has still learned something —
but we will say so plainly rather than call it a win.

## Stopping rules

1. **If Step 1's spread exceeds 0.15 nats**, stop and re-derive every threshold.
   At that noise level this project cannot measure 58M-scale changes in a
   micro-budget experiment, and continuing would generate more X20s.
2. **If Step 2 fails**, do not run Step 4. An unverified LR schedule makes an
   adaptive mixture result uninterpretable — the mixture's effect would be
   confounded with a schedule we do not trust.
3. **No step runs from a single seed.** If quota forces a one-seed run, it is
   recorded as a *screen*, not a result, and the registered verdict stays "unrun."
4. **If all steps return nulls**, v0.3.0 is still released, labelled as a null
   with the measured baseline, the retracted citation, and the protocol. A
   documented null at 58M with a variance baseline is worth more than an
   unreplicated win.

## What we expect to find

Stated before the run, so it cannot be edited afterwards.

**Our expectation is a null, or a small effect that does not clear the spread.**
Reasons:

- The one large win in this project came from data (−0.2285 nats). Architecture
  produced nothing across seven arms.
- The recipe is validated an order of magnitude above us.
- The control spread at 30M was 0.11 nats. We have no reason to think 58M is
  quieter, and the plateau at step 3,250 suggests this model is in a regime where
  changes are hard to move.

**We are registering this because a pre-registration that only permits success is
a marketing document.** If 0.3.0 comes back null, that is the expected outcome
having been correctly anticipated, not a failure.

## What this is not

- **Not a capability claim.** We do not expect a 10x improvement and have
  registered that. Attention is ~15% of KALIA's per-token FLOPs at context 1024
  (4·T·d·L with d=512, L=10), so no attention-layer change can deliver an order
  of magnitude at our scale. The lever is tokens-per-GPU-hour.
- **Not the Sanskrit programme.** One arm (Kautilya) is ours. The other two
  designs from that scan — Panini precedence, Nyaya apoha — remain unimplemented
  and unmeasured. Scanning ancient texts produced three designs in six days and
  shipped zero, which is a real failure of that process and is recorded as such.
- **Not the active-forgetting experiment.** The embedding is 44.4% of parameters
  (8.2× a whole transformer layer, verified from `configs/kalia-v020.yaml`:
  50257 × 512 = 25.7M of 57.9M) and NeurIPS 2023's active forgetting fits that
  shape in three lines. **It is deferred to 0.4** because shipping an architecture
  change one commit after D48 declared architecture effects unmeasurable would
  not be defensible, however good the structural argument is.
- **Not distillation.** No teacher, no teacher outputs, at any point.

## Budget

| Step | GPU-hours | Cumulative |
|---|---|---|
| 0 | 0.1 | 0.1 |
| 1 | 0.6 | 0.7 |
| 2 | 0.6 | 1.3 |
| 3 | 0.7 | 2.0 |
| 4 | 0.7 (+ engineering) | 2.7 |
| Kautilya loader dev | CPU only | — |

~2.7 GPU-hours against a 30h monthly quota. Step 4's implementation is the
schedule risk, not the GPU.

## What would make us stop the whole programme

- Step 1 spread > 0.15 nats **and** a second measurement confirming it → 58M is
  below our measurement floor at this budget, and we should stop publishing
  micro-scale architecture and data claims entirely.
- The Kautilya loader cannot be made correct and tested without a result we would
  trust → drop it rather than ship an unmeasured mechanism.
