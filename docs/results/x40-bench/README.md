# X40 2×2 benchmark — the failure is composition, not the test

**Verdict: the clean benchmark does not rescue X40. It sharpens the null.**
The hard compositions — `rename+swap` and `number+place` — are at chance or
zero for every arm, including the control, even on seen templates with no new
causal relation. The above-chance pair cells are negation-marked, where the
control is already above chance. X40 stays closed.

Run: `subhajitlucky/kalia-x40-bench` (CPU), scoring the archived X40 v2
checkpoints (`control`, `pairs`, `IP` at seeds 1338/1339). Benchmark seed 2026,
400 records per cell, 1,600 records / 2,400 candidates. This is a **post-hoc
diagnostic**, not a re-score of I-1…I-7.

## Design

The X40 eval mixed two variables: a new **combination** of interventions and a
new **sentence type / causal relation**. The 2×2 separates them:

|  | single change (seen) | pair change (unseen) |
|---|---|---|
| **seen sentence type** | S-S | **S-P** |
| **unseen sentence type** | U-S | U-P |

- **S-S** — train template, one intervention. The easy baseline.
- **S-P** — train template and causal relation, two interventions. The real
  composition test: nothing is new except the pair.
- **U-S** — unseen template, one intervention. Template/lexical generalization.
- **U-P** — both at once (the old X40 aggregate).

All cells use the same new name pool, so name novelty is constant. Degenerate
compositions are rejected (`give_keep` + swap+negate collapses to the factual
suffix). Chance is 0.50 for single cells, 0.25 for pair cells.

## The 2×2

| cell | control 1338 | control 1339 | pairs 1338 | pairs 1339 | IP 1338 | IP 1339 |
|---|---:|---:|---:|---:|---:|---:|
| **S-S** | 0.4375 | 0.4300 | 0.6350 | 0.7200 | **0.7050** | **0.7200** |
| **S-P** | 0.2400 | 0.2400 | 0.2525 | 0.2900 | 0.2750 | 0.2275 |
| **U-S** | 0.3650 | 0.4000 | 0.3700 | 0.5425 | **0.5400** | **0.5550** |
| **U-P** | 0.3300 | 0.3450 | 0.3325 | 0.3425 | 0.3275 | 0.3275 |

- The single-op cells (S-S, U-S) show the X40 seen gain: pairs/IP learn
  `number`/`place` and negation; the control is below chance.
- **S-P is at chance for every arm.** The aggregate says composition fails even
  with a familiar template and causal relation.
- **U-P is above chance for every arm**, but this is a mixed cell; the by-op
  breakdown below shows it is driven by negation-marked pairs, not the hard
  compositions.

## S-P by operation — the decisive table

Chance = 0.25.

| operation | control 1338 / 1339 | pairs 1338 / 1339 | IP 1338 / 1339 |
|---|---:|---:|---:|
| `number+place` | **0.000 / 0.000** | **0.000 / 0.000** | **0.000 / 0.000** |
| `rename+swap` | 0.239 / 0.179 | 0.246 / 0.269 | 0.239 / 0.246 |
| `swap+negate` | 0.236 / 0.326 | 0.292 / 0.348 | 0.382 / 0.270 |
| `rename+negate` | 0.323 / 0.323 | 0.316 / 0.368 | 0.331 / 0.256 |

- **`number+place` is 0.0 for every model**, even on the seen `put_count`
  template. The model saw `number` and `place` singly; it cannot combine them.
- **`rename+swap` is at chance** for every model. It is the name-only role
  composition with no lexical marker.
- **`swap+negate` / `rename+negate` are above chance**, but the **control is
  already above chance** (0.32–0.33). The gain is a “did not” / new-name
  heuristic, not a learned composition. Pairs/IP do not consistently improve
  it; seed 1339 IP is at or below control on both.
- The S-P aggregate of ~0.25 is not just `number+place`: `rename+swap` is at
  chance too, and the two negation pairs are only ~0.3–0.38 with no IP
  advantage.

## Gold-edit interface test (2026-11-03)

The benchmark normally gives the model `instruction + factual prefix`. The
gold-edit mode gives it the **post-intervention prefix** instead. Same
candidates, same scoring. The question: can the model continue correctly once
the edit has already been applied?

| cell | control normal → edited | pairs normal → edited | IP normal → edited |
|---|---|---|---|
| S-S | .4375 → .4250 / .4300 → .4350 | .6350 → .6575 / .7200 → .5600 | .7050 → .6350 / .7200 → .6725 |
| **S-P** | **.2400 → .2925 / .2400 → .2950** | **.2525 → .2550 / .2900 → .2600** | **.2750 → .3200 / .2275 → .3250** |
| U-S | .3650 → .4450 / .4000 → .4525 | .3700 → .4350 / .5425 → .5075 | .5400 → .5775 / .5550 → .6125 |
| U-P | .3300 → .4650 / .3450 → .4200 | .3325 → .4225 / .3425 → .4000 | .3275 → .3950 / .3275 → .4275 |

The aggregate S-P rises by ~0.05, but the S-P by-operation table shows where:

| S-P operation | control normal → edited | pairs normal → edited | IP normal → edited |
|---|---|---|---|
| `number+place` | 0.000 → 0.000 / 0.000 → 0.000 | 0.000 → 0.000 / 0.000 → 0.000 | 0.000 → **0.318** / 0.000 → 0.000 |
| `rename+swap` | 0.239 → 0.254 / 0.179 → 0.216 | 0.246 → 0.239 / 0.269 → 0.261 | 0.239 → 0.239 / 0.246 → 0.269 |
| `swap+negate` | 0.236 → 0.180 / 0.326 → 0.225 | 0.292 → 0.157 / 0.348 → 0.247 | 0.382 → 0.303 / 0.270 → 0.348 |
| `rename+negate` | 0.323 → **0.504** / 0.323 → **0.519** | 0.316 → 0.421 / 0.368 → 0.353 | 0.331 → 0.414 / 0.256 → **0.474** |

The gold edit helps **`rename+negate`** (especially the control), but:

- **`number+place` is still 0.0** in 5 of 6 arms (the lone IP-s1338 0.318 is
  one seed and does not replicate);
- **`rename+swap` is still at chance** in every arm;
- `swap+negate` does not improve consistently and is often worse.

**Interpretation.** The model can continue some negation-marked pairs from the
edited world, but it cannot continue the two hard compositions even when the
edit is given. The bottleneck is **not just applying the edit**; it is the
role/attribute composition itself. A small edit interface would help the
negation cells the control already handles, not the cells the mechanism was
meant to unlock.

The IP’s gold-edit S-P is +0.065 over pairs on both seeds (0.320/0.325 vs
0.255/0.260), which is consistent but small; the control is better on
`rename+negate` (0.504/0.519 vs 0.414/0.474), so the IP advantage is not a
general composition ability. This does not justify an edit-interface bet.

## What this means

1. **The X40 null is real, not a template artifact.** On the cleanest cell
   (S-P), the model does not compose even when the sentence type and causal
   relation are familiar. The 30M model’s composition failure survives the
   benchmark fix. The gold-edit test does not change this: S-P rises only to
   ~0.29–0.33, and `number+place` and `rename+swap` remain at 0.0/chance.
2. **The mechanism’s gains are in the wrong place.** The IP/pairs gains are on
   single operations (`number`, `place`, negation) and the easy negation
   pairs. They do not improve `rename+swap` or `number+place`.
3. **The workspace/state loss is not the lever.** On S-P, IP is not better than
   pairs; on the hard operations, it is at or below control. This is consistent
   with X40: the workspace is alive but not used for composition.
4. **Aggregate accuracy hides the effect.** A 4-way composition score can look
   “at chance” because one hard operation is 0.0 while another is 0.4. Future
   composition benchmarks must report **by operation**, not just by cell.
5. **The negation pairs are a heuristic trap.** The control already scores
   ~0.32–0.33 on `rename+negate`/`swap+negate`; a mechanism that “beats
   chance” on the aggregate may only be exploiting the “did not” marker.

## Caveats

- **Post-hoc.** This does not reopen the registered X40 verdict; it explains it.
- **30M, 763 steps, two seeds.** No scaling claim. The checkpoints are the
  archived X40 v2 models, not a new training run.
- **U-P is still mixed.** It is useful as “both new,” but the S-P cell is the
  clean composition test.
- **The candidate sets are controlled 2-/4-way choices.** The negation cells
  contain a lexical marker; the high control score is consistent with a
  heuristic, not necessarily composition.
- **One benchmark seed.** The 400 records per cell give ~±0.04–0.05 standard
  error on a cell; the by-op cells are smaller (44–160 records).
- **The gold-edit prompt is zero-shot.** None of the models were trained to
  continue from the edited prefix. The test is an interface probe, not a
  fine-tuned upper bound; a model trained on edited prefixes could score
  higher. The hard-cell failure is still informative because continuing from
  the edited world should be easier than applying the edit.

## Decision

- **X40 stays closed.** No 58M, no 134M. The clean benchmark confirms the
  failure is composition, not the unfair template.
- **The benchmark is the reusable asset.** It separates template novelty from
  composition and exposes the hard operations. Any future composition claim
  should use this design and report the S-P by-operation table.
- **A future mechanism must target `rename+swap` and `number+place`.** Beating
  the negation pairs or the single-op cells is not enough; the control already
  handles those. The gold-edit test shows that giving the model the edited
  prefix is not enough either; the missing piece is binding/composition, not
  edit application. That is a new pre-registration, not a rescue of X40.

## Artifacts

- `x40_bench_summary.json` — the 6 checkpoints, 2×2, by-operation, checkpoint
  hashes, benchmark audit.
- `x40_bench_rows.jsonl` — 9,600 per-record rows for the instruction-prefix
  mode (6 models × 1,600).
- `x40_bench_summary_gold_edit.json` / `x40_bench_rows_gold_edit.jsonl` — the
  two-mode run (28,800 candidate scores).
- `bench_audit.json` — cells, counts, names, lengths.
- `x40-bench.log` / `x40-bench-gold-edit.log` — the v2 and v3 CPU kernel logs.
- Kernel: `https://www.kaggle.com/code/subhajitlucky/kalia-x40-bench`.
