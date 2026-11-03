# X40 — intervention pretraining at 30M: closed null

**Verdict: X40 is closed at 30M. I-1 and I-3 fail. No 58M, no 134M. No eval
tuning, no pair-weight rescue, no composition-set rescue.**

Run: `subhajitlucky/kalia-x40-ip`, kernel version 2, revision
`2153d6f4de3e50706ade6a55386fd08dd2252bba`, recorded
`2026-11-02T09:05:32Z`. Registration:
`docs/preregistrations/2026-11-01-X40-intervention-pretraining.md`
(hash `5e42b784…`).

The v1 kernel died at the first workspace arm on an fp16 `masked_fill(-1e9)`
overflow; it produced no eval and is not used here. v2 is the screen.

## Registered bars

| ID | prediction | threshold | 1338 | 1339 | verdict |
|---|---|---|---:|---:|---|
| **I-1** | IP does not cost language quality | `IP val − control ≤ +0.005` | **+0.2040** | **+0.1689** | **FAIL** |
| **I-2** | seen interventions improve | `IP seen − control ≥ +0.10` | +0.2022 | +0.1867 | PASS |
| **I-3** | unseen compositions beat pairs-only | `IP unseen − pairs ≥ +0.10` | **+0.0283** | **0.0000** | **FAIL** |
| **I-4** | retention does not regress | `IP retention − control ≥ −0.005` | −0.00128 | −0.00036 | PASS (floor) |
| **I-5** | workspace alive and used | entropy > 2 bits, rank > 4, probe gain > 0.10 | 3.794 / 12.283 / 0.889 | 3.763 / 11.960 / 0.889 | PASS |
| **I-6** | pair audit clean | no template/name overlap | clean | clean | PASS |
| **I-7** | ruler reported | control seed spread | val 0.0113 / seen 0.0244 / unseen 0.0100 | — | reported |

`advance = false`. The registered rule is explicit: **I-1 fails → no scale;
I-3 fails → close this implementation.**

## Arms

| arm | seed | val | seen (2-way) | unseen (4-way) |
|---|---:|---:|---:|---:|
| control | 1338 | 4.1495 | 0.3333 | 0.2533 |
| control | 1339 | 4.1608 | 0.3578 | 0.2633 |
| adapter | 1338 | 3.8914 | 0.3311 | 0.2250 |
| adapter | 1339 | 3.9018 | 0.3600 | 0.2550 |
| pairs | 1338 | 4.2730 | 0.3422 | 0.2367 |
| pairs | 1339 | 4.2831 | 0.5400 | 0.3017 |
| workspace | 1338 | 4.3671 | 0.4422 | 0.2600 |
| workspace | 1339 | 4.3373 | 0.6067 | 0.2733 |
| **IP** | 1338 | **4.3535** | **0.5356** | **0.2650** |
| **IP** | 1339 | **4.3297** | **0.5444** | **0.3017** |

Chance is 0.50 for the 2-way seen set and 0.25 for the 4-way unseen set. The IP
seen accuracy is only **3.6–4.4 pp above chance**; the unseen accuracy is
**1.5–5.2 pp above chance**. The control is *below* 2-way chance because a
plain LM prefers the factual continuation; the IP gain is partly a correction
of that bias, not a strong intervention skill.

Parameter match: control/pairs 29,921,280; adapter 30,322,944; workspace
30,323,088; IP 30,759,952. Adapter vs workspace differs by 144 parameters.

## The deltas that decide

| comparison | 1338 | 1339 |
|---|---:|---:|
| IP val − control | +0.2040 | +0.1689 |
| pairs val − control | +0.1235 | +0.1223 |
| workspace val − control | +0.2176 | +0.1765 |
| adapter val − control | **−0.2581** | **−0.2590** |
| IP seen − control | +0.2022 | +0.1867 |
| IP unseen − pairs | +0.0283 | 0.0000 |
| workspace unseen − pairs | +0.0233 | −0.0283 |
| pairs unseen − control | −0.0167 | +0.0383 |

The language tax is **not only the state loss**:

- pairs-only already costs **+0.123/+0.122** nats vs control;
- the event workspace adds **+0.094/+0.054** over pairs;
- the state/diversity terms **slightly reduce** the tax vs the workspace arm
  (**−0.014/−0.008**);
- total IP tax **+0.204/+0.169**.

The pair data and the workspace architecture are the costs; the auxiliary
state term is not the culprit. The total is still 34–41× the 0.005 allowance,
and the pair data alone is 24× over the bar.

The **adapter** is the largest val gain in the screen: a parameter-matched
output MLP, trained on the corpus only, improves val by **0.26 nats**. The
workspace at the same parameter budget is **0.44–0.48 nats worse than the
adapter** (and **0.18–0.22 nats worse than control**). This is a
capacity/placement signal: at 30M the extra parameters are better spent at the
output than in the event workspace. It is not the X40 mechanism, and it is not
a reason to scale X40.

## Unseen compositions, by operation

| composition | pairs 1338 | IP 1338 | pairs 1339 | IP 1339 |
|---|---:|---:|---:|---:|
| rename+swap | 0.1600 | 0.2267 | 0.1800 | 0.2867 |
| swap+negate | 0.4267 | 0.4533 | 0.5067 | 0.4867 |
| rename+negate | 0.3600 | 0.3800 | 0.4133 | 0.4333 |
| number+place | **0.0000** | **0.0000** | **0.1067** | **0.0000** |

The aggregate I-3 is dominated by **number+place**, which is 0.0 in 9 of 10
arms. The `pack_box` eval template ("packed … → the bag holds …") uses a causal
relation that never appears in training; the model never predicts the double
change. That is a **template/lexical confound**, not a pure composition test.

Excluding it, the remaining three compositions still do not give a consistent
+0.10:

- rename+swap: **+0.067 / +0.107** (one seed clears, one does not);
- swap+negate: **+0.027 / −0.020**;
- rename+negate: **+0.020 / +0.020**.

The null survives the confound. What the confound changes is the **size** of
the composition claim, not the verdict: this screen did not measure composition
cleanly, and it still failed.

## What the null says

1. **The objective learns a seen-intervention bias, not a causal state.** IP
   moves 2-way seen accuracy from below chance to ~0.54, but the gain does not
   transfer to unseen compositions. Against the pair-only baseline the seen
   gain is **+0.193 on seed 1338 but +0.004 on seed 1339**; the `pairs` arm
   itself is unstable (0.342 vs 0.540). The I-2 pass is vs control, not vs the
   data baseline.
2. **The workspace is alive but not the lever.** Entropy 3.76–3.79 bits,
   effective rank 11.96–12.28, probe gain 0.889. The instruction type is
   linearly decodable; the language model does not use the workspace for
   composition.
3. **Pairs-only is not the lever either.** It barely moves unseen
   (+0.011 average) and its seen accuracy is unstable across seeds
   (0.342 vs 0.540). Adding counterfactual text is not enough.
4. **The objective is not free.** The pair data and the event workspace each
   add a language-quality cost; the state/diversity terms slightly reduce the
   workspace's tax but not enough to matter. The total is 34–41× the registered
   allowance. Scaling it would carry that tax to 58M and 134M.
5. **The retention bar is vacuous here.** The X36 greedy retention is at the
   floor (0.000–0.001) for every arm; I-4 passes but does not measure
   retention. The swap sensitivity is small (0.001–0.004) with no consistent
   pattern.
6. **The composition benchmark needs a cleaner design.** A future composition
   test should hold the causal relation fixed across train/eval and vary only
   the operation composition. `number+place` conflates the two.

## Post-hoc 2×2 benchmark (2026-11-03)

The clean benchmark separates template novelty from composition: seen/unseen
sentence type × single/pair change, one new name pool held constant. It is in
`docs/results/x40-bench/README.md`.

The result **does not rescue X40; it explains it**:

- On the cleanest cell (seen template, pair change only), every arm is at
  chance: control 0.240/0.240, pairs 0.253/0.290, IP 0.275/0.228 (chance 0.25).
- `number+place` is **0.0** for every model, even on the seen `put_count`
  template; `rename+swap` is at chance for every model.
- The above-chance pair cells are negation-marked (`swap+negate`,
  `rename+negate`), where the **control** is already at 0.32–0.33. That is a
  “did not”/new-name heuristic, not a learned composition.
- The IP does not improve the hard cells over pairs or control.

The composition failure survives the benchmark fix. The aggregate S-P at
chance is not just `number+place`: `rename+swap` is at chance too. Future
composition claims must report the S-P by-operation table, not just an
aggregate.

The gold-edit test (D70) gives the model the post-intervention prefix; S-P
rises only to 0.29–0.33, and the hard cells still fail (`number+place` 0.0 in
5/6 arms, `rename+swap` at chance). The edit is not the missing piece.

## Decision

**Close X40.** The registered rule says stop on I-1 or I-3; both fail, on both
seeds. No 58M, no 134M. The GPU budget is not spent on a mechanism that costs
language quality and does not beat the data baseline on the capability it was
designed for.

This is a **mechanism-level null at 30M**, not a seed accident:

- IP seen gain is consistent (+0.20/+0.19) but does not transfer;
- IP unseen gain over pairs is within seed noise (+0.028/0.000);
- the workspace is alive, so the null is not a dead-module artifact;
- the parameter-matched adapter beats the workspace on val.

What is **not** claimed: that intervention prediction is impossible, that
causal world models do not work, or that a different implementation/weight
schedule could not help. The claim is narrower and sufficient: **this
auxiliary objective, at this scale, with these weights, fails the
pre-registered capability and language-quality bars.** A different objective
is a new pre-registration, not a rescue of X40.

## Artifacts

- `x40_summary.json` — full bars, deltas, workspace stats, by-operation rows.
- `x40_seen_unseen_rows.jsonl` — 10,500 per-record rows (10 models × 1,050).
- `run_manifest.json` — code revision, hashes, pair audit, configs.
- `pair_audit.json` — 16,000 train / 1,050 eval; disjoint templates and names.
- `configs/` — the ten run configs.
- `val/` — the ten canonical 100-batch val reports.
- `x40.log` — the kernel log (v2).

Kernel: `https://www.kaggle.com/code/subhajitlucky/kalia-x40-ip`.
