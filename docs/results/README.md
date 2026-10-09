# Run results

Machine-readable record of every experiment in this project: what ran, on which
kernel, how many parameters, and the validation loss with its full curve.

**Regenerate with `python tools/collect_results.py`. Verify with `--check`.**

Generated from the raw logs archived in `logs/`. It is not hand-maintained —
`test_results_record.py` fails if the committed file stops matching its logs, so
the two cannot drift.

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
was lost. `kaggle_live_log.py` returns a rolling window, so older arms fall out
of it. Seven of the eleven arms are in that state.

## The results

| experiment | arm | params | val loss | Δ vs control | curve |
|---|---|---|---|---|---|
| X18 | `micro-base` | 29,920,512 | 4.7662 | — | lost |
| X18 | `micro-gated` | 30,072,768 | 4.7226 | −0.0436 | lost |
| X18 | `micro-nope` | 29,920,512 | 4.7772 | +0.0110 | lost |
| X19 | `micro-branchnorm` | 29,922,816 | 4.7680 | +0.0018 | lost |
| X20 | `micro-staticgate` | 29,999,040 | 4.6304 | −0.1358 | lost |
| **X21** | `micro-base-s1338` | 29,920,512 | 4.8769 | — | **full** |
| **X21** | `micro-staticgate-s1338` | 29,999,040 | 4.9623 | **+0.0854** | **full** |
| **X21** | `micro-base-s1339` | 29,920,512 | 4.8908 | — | **full** |
| **X21** | `micro-staticgate-s1339` | 29,999,040 | 4.9087 | **+0.0179** | **full** |

Only X21 has complete curves, because it is the only run whose log was archived
whole.

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
