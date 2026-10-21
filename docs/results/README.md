# Run results

Machine-readable record of every experiment in this project: what ran, on which
kernel, how many parameters, and the validation loss with its full curve.

**Regenerate `experiments.json` with `python tools/collect_results.py`. Verify with `--check`.**

`experiments.json` is generated from the raw logs archived in `logs/` and is not
hand-maintained — `test_results_record.py` fails if the committed file stops
matching its logs, so the two cannot drift. **This README is hand-written around
it.** An earlier version of this paragraph claimed the generator produced both,
which it never did: `collect_results.py` writes only the JSON.

## The benchmark sample size

The per-release numbers recorded here were measured at **`limit=500`**, where
lm-eval's standard error is **1.99–2.24 pp** (D45). Read them with that error bar
attached — several do not resolve. They are left exactly as they ran.

From 2026-10-18 `eval_bench.py` defaults to **2,000** samples for new runs, and
warns loudly below 1,000 rather than passing silently. The first 2,000-sample
release comparison ran on 2026-10-21: **v0.1.2 vs v0.2.0, both checkpoints, one
CPU session**, in `docs/results/bench-pair-2k/` (narrative:
`docs/eval/2026-10-21-v020-bench-pair-2k.md`). It changes the two headline
readings: the ARC-Easy regression (−3.80) becomes **+0.45**, and the HellaSwag
gain (+3.00) becomes **0.00**. LAMBADA remains the only task outside the 1.0 pp
allowance at **−1.25**, and no movement reaches 2σ. The frozen 500-sample
registration is not edited; the 2,000-sample result is the higher-resolution
observation.

## Reading it

```json
{
  "experiment": "x21",
  "kernel": "subhajitlucky/kalia-x21-replication",
  "arms": [
    {
      "arm": "micro-staticgate-s1338",
      "params": 29999040,
      "val_loss_final": 4.9623,
      "val_curve": {"100": 6.0207, "200": 5.5024, "...": 0, "500": 4.9623},
      "delta_vs_control": 0.0854,
      "source": "docs/results/logs/x21.log"
    }
  ]
}
```

`source` is the part to read first:

| `source` | meaning |
|---|---|
| `docs/results/logs/<exp>.log` | parsed from an archived run log |
| `prereg:<exp>` | transcribed from a pre-registration's prose — **not** parsed from a run |

`val_curve: null` with `truncated: true` means the number is real but its curve
was lost or never in the archived window. `kaggle_live_log.py` returns a rolling
window, so older arms fall out of it. Five of the eleven arms are in that state:
X18's three (its window closed before the summary printed) and the X19/X20
controls (never in their surviving windows).

## The results

| experiment | arm | params | val loss | Δ vs control | curve |
|---|---|---|---|---|---|
| X18 | `micro-base` | 29,920,512 | 4.7662 | — | lost |
| X18 | `micro-gated` | 30,072,768 | 4.7226 | −0.0436 | lost |
| X18 | `micro-nope` | 29,920,512 | 4.7772 | +0.0110 | lost |
| X19 | `micro-branchnorm` | 29,922,816 | 4.7680 | +0.0018 | full |
| X20 | `micro-staticgate` | 29,999,040 | 4.6304 | −0.1358 | full |
| **X21** | `micro-base-s1338` | 29,920,512 | 4.8769 | — | **full** |
| **X21** | `micro-staticgate-s1338` | 29,999,040 | 4.9623 | **+0.0854** | **full** |
| **X21** | `micro-base-s1339` | 29,920,512 | 4.8908 | — | **full** |
| **X21** | `micro-staticgate-s1339` | 29,999,040 | 4.9087 | **+0.0179** | **full** |

X19, X20 and X21 have complete curves. The X19/X20 treatment curves were in the
archived logs all along; the collector's first curve regex did not match their
colon format and recorded them as lost until 2026-10-17.

## X31 — QK-Norm/soft-cap isolation (2026-10-21)

D8's bundled −0.0834 (QK-Norm + logit soft-cap, seed 1337) re-tested as a 2×2
at seeds 1338/1339, 763 steps, one kernel. Archive:
`docs/results/x31/`; registration:
`2026-10-21-X31-qk-softcap-isolation.md`.

| condition | seed 1338 | seed 1339 | mean Δ vs control |
|---|---|---|---|
| control | 4.3356 | 4.3148 | — |
| **qk** | **4.2435** | **4.2382** | **−0.0844** |
| cap | 4.3485 | 4.3217 | **+0.0099** |
| both | 4.2517 | 4.2405 | −0.0791 |

**The gain is QK-Norm, not the bundle.** QK alone reproduces the original
−0.0834 at both seeds; the soft-cap alone is +0.0099 (both seeds positive) and
does not add to QK. Q-1, Q-1b, Q-2a and Q-3 pass; Q-2b fails. Future configs
keep `qk_norm` and drop `logit_softcap`; v0.2.0 is not retro-changed. See D55.

## v0.3.0 Step 1 — the 58M control spread (2026-10-08)

Not an experiment: a baseline measurement, run twice in two sessions. Three
fresh-seed control arms, 500 steps each, v0.2.0 recipe.

| arm | session 1 | session 2 | Δ |
|---|---|---|---|
| `v030-ctl-s1401` | 3.8045 | 3.8086 | +0.0041 |
| `v030-ctl-s1402` | 3.7388 | 3.7367 | −0.0021 |
| `v030-ctl-s1403` | 4.0280 | 4.0279 | −0.0001 |
| **spread** | **0.2892** | **0.2912** | **+0.0020** |

Session reproducibility is ≤ 0.0041 nats; the 0.29-nat spread is a **seed
effect**, not machine noise. At 30M the same protocol measured 0.1139, so the
58M ruler is 2.5x shakier. Both conditions of the pre-registered programme-stop
clause held (spread > 0.15, second measurement confirming), so **Steps 2–4 were
not run** and no further micro-scale architecture or data claims will be
published at 58M.

Raw artifacts: `step1_spread.json`, `step1_spread_rerun.json`, and the six
`logs/v030-baseline-*-val_log.csv` files. `experiments.json` does not yet
include these rows — the collector extension is follow-up work.

## Tier 1 probe — retrieval utilization and best-of-N (2026-10-08)

The 58M model, three probes, 64/48 items, deterministic, bootstrap CIs over
items (paired where the comparison is paired). Harness: `retrieval_probe.py`;
raw record: `retrieval-probe.json`.

**Can it read a fact it was just given?** Synthetic facts with invented family
names (unknown by construction); the answer is one of 8 single-token places,
chance = 0.125.

| condition | top-1 | 95% CI | MRR |
|---|---|---|---|
| none | 0.000 | [0.000, 0.000] | 0.035 |
| BM25 (true passage in top-3 100% of the time) | 0.297 | [0.188, 0.406] | 0.475 |
| oracle (the true sentence prepended) | **0.828** | [0.734, 0.922] | 0.880 |

At 58M the model **can** copy an answer that is sitting one sentence away
(0.828 against 0.125 chance). The utilization failure the 2026 literature finds
for sub-7B instruction models does not transfer to verbatim copying — but as
soon as two confusable facts sit in the context, accuracy drops to 0.297 even
though retrieval never missed.

**Does context help or hurt on real text?** Mean NLL of 16 held-out tokens,
paired per window: relevant context −0.1075 nats [−0.179, −0.046]; irrelevant
context **+0.1222 nats** [+0.077, +0.171]. Both intervals exclude zero:
relevant context helps, and an unrelated passage measurably distracts.

**Does best-of-N help?** On the oracle condition, 8 samples per item:

| selector | top-1 | 95% CI |
|---|---|---|
| greedy | 0.828 | [0.734, 0.922] |
| majority | 0.672 | [0.562, 0.797] |
| own log-prob, sel(k=1) | 0.828 | [0.734, 0.922] |
| own log-prob, sel(k=2) | 0.703 | [0.594, 0.812] |
| own log-prob, sel(k=4) | 0.547 | [0.422, 0.672] |
| coverage (ceiling for any selector) | 0.953 | [0.891, 1.000] |

Scoring the whole 4-token continuation selects **worse than greedy** (−0.281
[−0.406, −0.172]): a fluent wrong continuation out-scores a correct first token
that continues awkwardly. The fix — score only the decision position
(`score_tokens=1`) — removes the harm but adds nothing: sel(k=1) − greedy =
**+0.000 [0.000, 0.000]**. At 58M the best first token in the sampled pool *is*
the greedy token. Reaching the 0.953 coverage ceiling needs a selector with
information the model's own log-probs do not carry — a SCATR-style head trained
on labels — and that number is what it has to beat.

## Corpus contamination gate — 13-gram scan vs the five benchmarks (2026-10-08)

`corpus_gates.py` scans tokenized shards for **exact** 13-gram matches against
the five scored evaluation sets; every hash hit is verified token-for-token, so
no collision can appear as a finding. Run on `train.bin` (2.4B tokens) and
`val.bin` (10M tokens):

| task | verified windows (train) | approx. regions |
|---|---|---|
| PIQA | 1,429 | ~111 |
| HellaSwag | 725 | ~167 |
| LAMBADA | 368 | ~191 |
| ARC-Easy | 377 | ~170 |
| WinoGrande | 0 | 0 |
| val.bin (all tasks) | 4 | ~3 |

2,903 windows in 2.41B tokens is ~1.2 per million — small but real: the corpus
carries wikiHow-style instruction text (PIQA/HellaSwag), book prose (LAMBADA),
and science phrasing (ARC-Easy) that the benchmarks also use. Honest card line:
those four scores may be marginally optimistic, bounded by a share well under
0.001% of tokens; WinoGrande is unaffected. Raw record: `corpus-gates.json`.

**The first run of this gate reported 63,491,977 PIQA hits — all of them
whitespace.** 120 PIQA rows carry copied web text with long space runs, and a
13-space window matches Python indentation millions of times. The absurd count
was the bug report; patterns now require >=6 distinct tokens. A second artifact
(ARC's `choices` is a dict; stringifying it produced Python-repr patterns that
matched quiz-like code 601 times) was caught the same way. A gate that reports
63M hits is broken, and the count says so before any human reads the matches.

## X26 — the Kautilya pilot: the policy ate dessert (2026-10-09)

The first real run of the Kautilya adaptive mixture: two paired arms resumed
from v0.2.0 at step 4770, seed 1401, 1000 update steps each, differing only in
whether the policy ran (`adaptive_every` 0 vs 100). Artifacts in
`x26-pilot/` (the static arm's val log is reconstructed from the archived v1
run log; its session failed after completing and Kaggle publishes no outputs
for failed runs).

**Held-out val, identical protocol:**

| step | static | adaptive |
|---|---|---|
| 4800 | 2.7660 | 2.7660 |
| 4900 | 2.6783 | 2.6801 |
| 5000 | 2.9829 | 2.9786 |
| 5100 | 2.7749 | 2.7784 |
| 5300 | 2.8570 | 2.8596 |
| 5500 | 2.8423 | 2.8543 |
| **5700** | **2.8200** | **2.8377** |

The arms track each other to 0.002 while the policy has barely moved, then
diverge as the mixture drifts: adaptive ends **+0.0177 nats worse**.

**The diary (per_source_log.csv):** ten decisions, monotonic throughout —
FineWeb-Edu 60% → **22.7%**, TinyStories 20% → **56.3%**, Python 5% → 14.1%,
Cosmopedia 15% → 7.0%. The policy downsampled the hardest source every single
time and upsampled the two easiest.

**The mechanism-level finding.** The adaptive arm's *training* loss fell from
~2.63 to **1.79** while held-out loss on the original distribution did not
improve. It lowered the number it was optimizing by changing the distribution
it was measured on — feeding itself what was already easy ("eating dessert").
The val delta itself is small and unverified (this is a registered screen; a
same-config rerun to bound paired noise plus a fresh seed is required before
any verdict), but the drift and the train/val divergence are robust: they are
properties of the policy, not of a noisy evaluation.

Next arm, already implemented: the trend-based policy (improving/flat/
worsening/collapsed instead of level-vs-mean), which should not reward a
source for merely being easy.

## Entity-verified decoding — first measurement (2026-10-11)

20 frozen Gutenberg probe stories (`eval/ttt_probe_stories.json`), prompt 768
tokens, 256 generated, 8 samples at temperature 0.8 plus a greedy draw. The
same 8 samples were ranked three ways; retention and span via `eval_entities.py`.
Raw record: `entity-decode.json`.

| selector | retention | paired delta vs greedy (95% CI) | mean span | mean logprob |
|---|---|---|---|---|
| greedy | 0.030 | — | 0.050 | −0.38 |
| log-probability pick | 0.035 | +0.006 [−0.012, +0.023] | 0.055 | −1.42 |
| **entity pick (rule)** | **0.091** | **+0.061 [+0.039, +0.086]** | 0.074 | −2.23 |
| reference prose | 0.098 | — | 0.067 | — |

Paired bootstrap over stories, 5,000 resamples. The entity rule **triples**
retention over greedy and **beats the log-probability selector by 2.6x** —
whose own gain is again indistinguishable from zero. The model's entity-picked
continuations keep their characters about as well as the real Gutenberg
continuation does (0.091 vs 0.098). The rule picks differently from the
log-probability selector in 18 of 20 stories.

Two honest caveats, recorded with the result:

1. **The metric rewards keeping names, not quality.** The reference is not a
   ceiling for retention — the model beats it on several stories — because
   real prose moves on while the model echoes. What is claimed is a
   consistency improvement on the registered failure metric, nothing more.
2. **The selection trades fluency-likelihood for retention.** Entity picks
   average −2.23 mean log-prob against greedy's −0.38. They are still draws
   from the model's own 0.8-temperature pool, but the rule systematically
   prefers less-likely text; whether that costs story quality is unmeasured
   and is the next question.

**Correction (2026-10-21).** The +0.061 above is circular: the selector
maximises retention and the kernel reports retention. The holdout test in
`docs/results/entity-holdout/` selects on the first half of each continuation
and scores the second half the selector never saw. The honest advantage over
greedy is **+0.0108, 95% CI [−0.0073, +0.0318] — not distinguishable from
zero**. The full-text 0.0907 reproduces (the selector and metric see the same
text), but it no longer supports a 3× claim or the memory-only rung. What
survives is coverage: the oracle reaches 0.0618 on the held-out half against
greedy's 0.0170, so the right continuation is in the pool; the rule is not
finding it. The 2026-10-11 measurement stands as measured, with this correction
attached.

## X28 — the Kautilya replication: the dessert is real, and the machine is exact (2026-10-11)

Three arms, seed 1401, 1000 update steps. The static arm reran the X26 control;
the level arm reran the X26 policy exactly. Artifacts in `x28-trio/`.

**Run-to-run reproducibility (same arm, two independent sessions):**

| step | static v1 | static rerun | delta |
|---|---|---|---|
| 5000 | 2.9829 | 2.9822 | −0.0007 |
| 5600 | 2.8439 | 2.8407 | −0.0032 |
| **5700** | **2.8200** | **2.8201** | **+0.0001** |

Per-checkpoint agreement is −0.0032 to +0.0019; the final points differ by
**0.0001 nats**. The machine is not the variable, and the paired-noise bound
X26 deferred is now measured.

**The policy replicates.** The level arm ended at **2.8384** against X26's
adaptive arm at **2.8377** — a **+0.0007** difference — and its diary is
identical to the X26 diary to four decimals (final weights 0.2269 / 0.5625,
the same monotonic drift to the easy sources).

**The policy's cost is real and readable:** level − static(rerun) =
**+0.0183 nats**, against X26's +0.0177. The delta grows with the drift
(+0.0001 early, +0.0183 by the end) — the mechanism's signature. Against the
measured session noise (~0.001–0.003 per checkpoint), the cost is ~10–180x
larger.

**Registered expectation check:** the capability outcome is small but not
null, and it is negative for the policy. A fresh-seed (1402) replication is
still required by the X28 pre-registration before any claim ships.

**The trend arm — the drift is slowed, not stopped, and the cost tracks the
drift.** Final held-out val: static 2.8201 < **trend 2.8277** < level 2.8384,
and the mixture drift orders exactly the same way:

| arm | FineWeb (hard) | TinyStories (easy) | val vs static |
|---|---|---|---|
| static | 60% (fixed) | 20% (fixed) | — |
| trend | 41.4% | 41.4% | +0.0076 |
| level | 22.7% | 56.3% | +0.0183 |

The trend rule (judge by improvement, not level) **halved the drift and
halved the cost** — but did not stop it, falsifying the registered mechanism
prediction that its weights would stay near 60/20/15/5. The recorded
hypothesis for why: both rules share the same feedback loop — a source with
more exposure shows lower and improving loss (familiarity, not learning), and
improvement earns more exposure. `per_source_loss` draws windows from the same
shards the sampler trains on, so the signal cannot separate the two. A frozen
per-source probe set, never trained on, is the fix the next iteration would
need; not queued yet.

The dose-response is the useful part: three arms, three drift magnitudes,
three costs, ordered — evidence that the drift causes the cost rather than
merely accompanying it.

## X29 — patha replay, all four arms complete (2026-10-13)

The first test of Vedic recitation orderings (krama / jata / ghana) as replay
schedules: four arms, one continual update each (4770 → 5770, seed 1401, 10%
replay of the same 256 windows), identical except presentation order of the
replay stream. Null or small was the registered expectation. Artifacts in
`x29/`.

**Val (secondary metric), final step 5700 — a monotone staircase in structure
depth:** uniform (control) **2.8076** < krama **2.8314** < jata **2.8510** <
ghana **2.8752**; the X28 no-replay static arm ended 2.8201 at the same step.
The ordering holds at every checkpoint from 5,200 on (exceptions: 5,000 and
5,100, krama's mid-run spike region), with three near-equal rungs — +0.0238,
+0.0196, +0.0242 — every one far above the ≤0.003 per-checkpoint session-noise
bound, total span +0.0676. Uniform also sits below no-replay static: plain
shuffled replay helps, and each added layer of recitation structure (adjacent
pairs → two-word braid → nested triples) costs about 0.02 nats.

**Retention (primary metric), CL-0 ledger on the G2-pinned probe shard:** the
self-check re-scored v0.2.0 to **0.000000–0.000001** of the recorded
2026-10-07 baseline across two sessions — the instrument is exact. All four
arms are monotone on the same rungs as val: uniform **−0.0237** (better than
the pretrain's own probe score), krama **−0.0007** (at the baseline), jata
**+0.0181**, ghana **+0.0446** (the largest CL-0 drift this project has
measured). The two instruments even agree on **rung sizes** — val
+0.0238/+0.0196/+0.0242 against retention +0.0230/+0.0188/+0.0265 — three
~0.02 steps each, from two different evaluation sets and protocols. First
arm-level retention movement this project has measured; no claim (n=1, second
seed required).

One incident for the record: the probe's first three versions scored the
wrong probe file (`val_forget.bin` instead of the G2-pinned
`corpus/probe_val.bin`) — the built-in self-check refused with a +0.1107
discrepancy, which is exactly the fake-regression a silent substitution
would have produced on every arm. The same producer/consumer mismatch is
already pinned by `test_publish_contract.py`; this was its third sighting,
fixed in `d220918`.

Also on the record: both jata failures (and one uniform failure) before this
run were **our bug, not the platform's** — `PathaReplayDataset` concatenated
CPU replay rows with device-side new rows, which passes on CPU and dies on
the first GPU batch (rank 1 crashes, rank 0's allreduce times out). Fixed by
assembling on CPU and moving once (`d1055c5`), with a contract test that
spies on the device argument. This is the second bug of this shape (x26's
`_probs` was the first): local CPU tests cannot falsify device contracts,
and the test suite now pins this one where it can.



## What the record says

**X20's −0.1358 inverted.** At two fresh seeds the static gate is *worse* than
control, by +0.0854 and +0.0179. R-1, R-2 and R-4 all fail.

**The control is not seed-invariant.** `micro-base` scores 4.7662 at seed 1337
(two independent sessions, 0.0018 apart) and 4.8769/4.8908 at seeds 1338/1339 —
about **0.11 nats** higher. Two sessions agreeing to 0.0018 means the machine is
not the variable, so this is a seed effect.

That 0.11 **exceeds X18's entire −0.0436 and X19's −0.0018** by an order of
magnitude. Every single-seed delta in this project was read against a baseline
that was never measured. `test_the_control_baseline_spread_is_recorded_not_just_asserted`
holds that number in place: changing it requires editing a test whose docstring
says why.

## Caveat worth stating

X18's three arms and the X19/X20 controls are `prereg:`-sourced. Those numbers
live in markdown tables, not in data. They are the numbers D48 reclassified as
nulls, and the honest position is that the refutation is *stronger* than the
record: X21 refutes the mechanism directly, and the missing baseline would only
strengthen that conclusion, not weaken it.

Re-running X18 with an archived log would close this. It costs ~0.4 GPU-hours and
this project no longer spends them on architecture arms (D48).
