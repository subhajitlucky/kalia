# What survives a control?

## A negative-results record for small-LM mechanisms (X18–X40)

Status: **draft** for a workshop paper or technical report. Not submitted.
This is a synthesis of the archived runs, not a new experiment.

---

## Abstract

Between September and November 2026 we ran a pre-registered sequence of
mechanism experiments on a 30M micro model and a 58M model: gated residuals,
branch norms, static gates, test-time training, adaptive data mixtures, replay
ordering, QK-Norm/soft-cap isolation, entity circuits, linear readouts, code
mixtures, entity memory, a Pāṇinian rule grammar, a counterfactual residual
bottleneck, and intervention pretraining. Every bet was parameter-matched,
seed-replicated and registered before GPU time. **No mechanism survived its
pre-registered capability bar.** The 58M model that ships (v0.3.1) is the same
architecture as v0.1.2; its gains come from data, a full schedule, a second
epoch and a weight merge, not from a new mechanism.

The contribution of this record is not a win. It is a map of what fails, why
it fails, and what a credible mechanism claim would have to clear: a
parameter-matched control, a data-only baseline, a clean composition test, a
language-quality guard, a live-and-used module check, and a replication ruler.
We release the 2×2 composition benchmark that makes the last of these testable.

## 1. Why keep a negative record

Mechanism papers are selected for wins. The failures are scattered across
appendices, blog posts and abandoned branches. That makes the field's base rate
invisible and lets the same plausible add-on be re-tried under a new name.

This programme kept the opposite record. Each bet has:

- a pre-registration with fixed bars, written before GPU time;
- a parameter-matched control (and, where relevant, a data-only baseline);
- two seeds at 30M, two at 58M;
- a canonical 100-batch val, not a moving training-loop number;
- an archive of configs, logs, checkpoints and per-record eval rows;
- a rule that a failed bar closes the mechanism; no eval tuning, no scale.

The ledger verifies 33 pre-registrations. The archive holds 35 runs and 260
files. Three early runs are recorded as gaps rather than reconstructed.

## 2. Protocol

**30M screen.** 6 layers × 384 width, 6 heads, context 512, 763 steps, seeds
1338/1339. It is a screen, not a scaling result. The purpose is to kill a
mechanism cheaply before a 58M run.

**58M confirmation.** 10 layers × 512, 8 heads, context 1,024, 2,000 steps,
seeds 1401/1402, conditional on the 30M screen.

**Controls.** At minimum: the base model, a parameter-matched adapter or
widened block, and the data-only arm (the same text without the mechanism). A
win must beat the *data* baseline, not only the base model.

**Measurement.** Canonical 100-batch val at seed 1234; bpB; task benchmarks;
retention and swap probes; module liveness (entropy, effective rank, linear
probe). Seed spread is reported as a ruler, not hidden.

**Decision rule.** A failed capability bar closes the mechanism. A failed
language-quality guard closes it even if the capability moves. No post-hoc
eval changes, no weight rescues, no "promising trend" scale-ups.

## 3. What was tried

| run | bet | decisive result | outcome |
|---|---|---|---|
| X18 | NoPE + gated residual | the gate measured inert | null |
| X19 | branch-norm isolation | the gain lives in the gate, not the norm | null |
| X20 | static per-channel gate | largest effect measured | not promoted |
| X21 | X20 replication at fresh seeds | effect inverted | null |
| X25 | test-time training | closed null across a 40× range | null |
| X26/X28 | adaptive Kautilya mixture | "dessert": small, not the meal | not promoted |
| X29 | Patha replay ordering | no ordering beat uniform | null |
| X31 | QK-Norm / soft-cap isolation | QK-Norm carries the gain; soft-cap does not | architecture decision |
| X32 | entity-selection circuit | heads carry entity mass; ablation hurts | analysis |
| X33/X34 | linear midpoint readout | probe gain; seed-1007 replication fails the bar | null |
| X35 | static code mixture | −0.0007 / +0.0058; S-1 fails | null |
| X36 | entity memory | retention passes; val +0.076 vs matched control; M-2 fails | no scale |
| X37 | Pāṇinian rule grammar | S-1/S-2 fail; rules used uniformly | no scale |
| X38 | counterfactual residual bottleneck | C-1/C-1b pass; C-2/C-3/C-5 fail; stable stream rank ~1.5 | no scale |
| X39 | causal event attribution | registered; side paper, not run as the main bet | side |
| X40 | intervention pretraining | I-1 and I-3 fail; workspace alive, composition at chance | closed |

The pattern is not "nothing works." It is that **small add-ons reliably buy a
small language-model gain or a liveness signal, and rarely buy a capability
that survives the data and parameter controls.**

## 4. Four recurring failure modes

### 4.1 The language tax

Every module that adds a path to the residual stream pays a val cost. X36's
memory cost +0.076 nats against its matched control and failed M-2. X40's
intervention workspace cost +0.204/+0.169 against control; the pair data alone
cost +0.123/+0.122, the workspace added +0.094/+0.054 over pairs, and the
auxiliary state term slightly reduced the tax (−0.014/−0.008).

The sharpest control result: in X40, a **parameter-matched output adapter**
trained on the corpus only improved val by **0.26 nats**, while the event
workspace at the same budget was **0.44–0.48 nats worse than the adapter**.
Extra parameters at the output beat the mechanism. A new module must justify
its placement, not just its parameter count.

### 4.2 Liveness is not capability

X40's workspace passed every liveness check: 3.76–3.79 bits slot entropy,
effective rank 11.96–12.28, intervention probe 0.889. It encoded the
instruction. It did not improve unseen composition. X38's stable stream passed
its invariance bars and collapsed to rank ~1.5; X37's rules were alive but used
uniformly. **A module can be working and still not be the lever.** Liveness is
a debugging gate, not a result.

### 4.3 Data-only gains and the control's heuristics

X40's `pairs` arm trained on the same counterfactual text as ordinary LM data,
with no workspace and no state head. On the seen 2-way task it reached
0.342/0.540; the IP reached 0.536/0.544. On seed 1339 the data-only baseline
matched the mechanism. The pair data, not the workspace, carried most of the
seen gain.

The control also has heuristics that look like composition. In the clean
benchmark, the plain control scored 0.32–0.33 on `rename+negate` and
`swap+negate`, above the 0.25 chance level, because the negation marker ("did
not") and the new names are partially discriminative. A mechanism that
"beats chance" on an aggregate may only be exploiting the same marker.

### 4.4 Aggregates hide operation-level failure

X40's unseen aggregate was 0.265/0.302 against 0.25 chance. The clean 2×2
benchmark showed why that number is not a composition result:

| S-P operation | control | pairs | IP |
|---|---:|---:|---:|
| `number+place` | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| `rename+swap` | 0.239 / 0.179 | 0.246 / 0.269 | 0.239 / 0.246 |
| `swap+negate` | 0.236 / 0.326 | 0.292 / 0.348 | 0.382 / 0.270 |
| `rename+negate` | 0.323 / 0.323 | 0.316 / 0.368 | 0.331 / 0.256 |

The aggregate at chance is the sum of a 0.0 operation, a chance operation, and
two negation-marked operations the control already handles. **Report by
operation.** An aggregate over heterogeneous operations is not a capability
measure.

## 5. The 2×2 composition benchmark

X40's original eval mixed two variables: a new operation pair and a new
sentence type/causal relation (the `pack_box` "packed → the bag holds"
template). The clean benchmark separates them:

|  | single change (seen) | pair change (unseen) |
|---|---|---|
| **seen sentence type** | S-S | **S-P** |
| **unseen sentence type** | U-S | U-P |

All cells use one new name pool, so name novelty is constant. Degenerate
compositions are rejected. S-P is the clean composition cell: the sentence
type and causal relation were seen; only the pair is new.

The result (chance 0.50 single, 0.25 pair):

| cell | control 1338/1339 | pairs 1338/1339 | IP 1338/1339 |
|---|---:|---:|---:|
| S-S | 0.438 / 0.430 | 0.635 / 0.720 | 0.705 / 0.720 |
| **S-P** | **0.240 / 0.240** | **0.253 / 0.290** | **0.275 / 0.228** |
| U-S | 0.365 / 0.400 | 0.370 / 0.543 | 0.540 / 0.555 |
| U-P | 0.330 / 0.345 | 0.333 / 0.343 | 0.328 / 0.328 |

The composition failure is real, not a template artifact. The hard cases are
`rename+swap` (role binding without a lexical marker) and `number+place` (two
independent attribute changes). The negation pairs are not the target: the
control already handles them.

### The gold-edit interface test

One question remains: is the model failing to **apply** the intervention, or
to **continue** the intervened world? Giving the model the post-intervention
prefix (the gold edit) and scoring the same candidates separates the two.

S-P aggregate, normal → edited (chance 0.25):

| model | 1338 | 1339 |
|---|---:|---:|
| control | 0.240 → 0.293 | 0.240 → 0.295 |
| pairs | 0.253 → 0.255 | 0.290 → 0.260 |
| IP | 0.275 → 0.320 | 0.228 → 0.325 |

The gain is concentrated in `rename+negate` (control 0.323 → 0.504/0.519; IP
0.331/0.256 → 0.414/0.474). The hard cells do not move: `number+place` is
0.0 in 5/6 arms (one IP seed 0.318), and `rename+swap` remains at chance.
`swap+negate` is mixed and often worse.

The model can continue some negation-marked pairs from the edited world, but
it cannot continue the two hard compositions even when the edit is given. The
bottleneck is not just edit application; it is role/attribute composition. The
gold-edit prompt is zero-shot and out of distribution, so a fine-tuned
interface could score higher; the hard-cell failure is still informative
because continuing from the edited world should be the easier task.

The benchmark is released as `docs/results/x40-bench/` with a generator, an
audit, per-record rows, a scoring script and the gold-edit results. It is a
diagnostic, not a leaderboard; its value is the S-P by-operation table.

## 6. What would count as evidence

A mechanism claim at this scale should clear all of the following:

1. **Capability.** Beat the data-only baseline on the **S-P hard operations**,
   on both seeds, with a margin larger than the control seed spread.
2. **No language tax.** Val vs the base model within the pre-registered guard
   (for X40, +0.005).
3. **Used, not just alive.** The module must change the LM's behavior in the
   target cell; entropy, rank and a linear probe are necessary but not
   sufficient.
4. **Parameter and placement controls.** A matched output adapter and a
   matched wider block are the baselines; the mechanism must beat them.
5. **Operation-level reporting.** No aggregate over hard and easy
   compositions.
6. **Scale check.** The gain must not shrink at 58M/134M; a 30M screen is not a
   scaling result.
7. **Pre-registration and a null log.** The bar is fixed before GPU time; a
   failure closes the mechanism.

None of X18–X40 cleared this set. The two closest were X36's retention (which
passed but cost val and failed the matched control) and X40's seen-intervention
gain (which was mostly the data-only baseline and did not transfer).

## 7. Limitations

- The models are 30M and 58M; the training runs are short (763–4,770 steps).
  This is a screen for small-scale mechanisms, not a claim about frontier
  scale.
- The tasks are synthetic and the candidate sets are controlled 2-/4-way
  choices. The negation cells contain a lexical marker; the high control score
  is consistent with a heuristic.
- Two seeds per cell. The benchmark has 400 records per cell; the by-operation
  cells are smaller (44–160 records).
- The negative record is selected by what was tried. It does not prove that a
  mechanism cannot work; it proves that these implementations did not clear
  these bars.
- The 58M model is a data/schedule/merge result. The report does not claim a
  new architecture.

## 8. Conclusion

The shipped model is real: 57,856,256 parameters, val 2.7579, bpB 0.9016,
public at `kalia-lm/kalia-v031`. The mechanism search has no winner.

The useful output of the programme is the **measurement machine and the map**:
pre-registration, matched controls, seed rulers, archived nulls, and a clean
composition benchmark that separates template novelty from composition and
exposes the hard operations.

The next bet should target **binding**: `rename+swap` and `number+place`. It
should not be another generic workspace, another auxiliary state loss, or a
bigger version of X40. It should be a new pre-registration with the S-P table
as the primary bar. If no such mechanism exists, the honest move is to ship
v0.3.1 and stop the architecture search.

## Artifacts

- `docs/results/RUNS.md` — 35 runs, 260 files, 3 recorded gaps.
- `docs/preregistrations/LEDGER.md` — 33 verified registrations.
- `docs/results/x40/README.md` — the X40 null and I-1…I-7.
- `docs/results/x40-bench/README.md` — the 2×2 benchmark and S-P by-operation.
- `docs/DECISIONS.md` — D55–D69, including the X40 close.
- `docs/journal/` — the dated record, including the two startup failures.
- `https://huggingface.co/kalia-lm/kalia-v031` — the public model.
