# Novelty audit — what was new, what was already tested, and what it cost

Date: 2026-11-03. Scope: the mechanism experiments X16–X40 plus the supporting
runs. Purpose: classify each experiment against prior art and against the GPU
spent, so the next test is chosen on evidence rather than another novelty
claim.

**Limitations.** The search was a focused web sweep, not a systematic review.
The search quota was exhausted mid-audit; X37/X38 are classified
conservatively. GPU hours are exact where the decision record states them and
estimated from the run/step budget otherwise. The estimates are labelled.

## The table

Classes:

- **A — new composition/interface.** No exact prior match for this combination;
  the components are known.
- **B — known mechanism, new setting.** A published mechanism tested here; not
  a new experiment in the field.
- **C — known mechanism, replication/ablation.** No novelty; a control.
- **D — new evaluation/benchmark.** Not a mechanism.
- **E — training/data/merge.** Known recipe; may still produce the model.

| run | mechanism | closest prior art | class | GPU-h (est.) |
|---|---|---|---|---:|
| X16 | chunk-preserving reversal training | reverse LMs (Ledom 2025; reverse LM 2024) | B | 1.7 |
| X18 | NoPE + gated residual | NoPE 2023/24; Qwen3.8 gated residual 2026 | B | 1.2 |
| X19 | branch norm | branch-normalised gated residual | C | 1.2 |
| X20 | static per-channel gate | LayerScale / static gated residuals | B | 2.0 |
| X21 | X20 replication | — | C | 2.0 |
| X25 | test-time training | TTT 2020/2024; TTT-LM 2025 | B | 3.0 |
| X26 | adaptive mixture pilot | DoReMi 2023; online data mixing 2023 | B | 4.0 |
| X28 | Kautilya trio | data mixing laws / online mixing | B | 8.0 |
| X29 | Patha replay ladder | continual pretraining + replay (Ibrahim 2024) | B | 25.0 |
| X17 | intra-document masking | packed sequences / document masks | C | 13.0 |
| X31 | QK-Norm / soft-cap isolation | QK-Norm 2020/2023; Gemma 2 soft-cap 2024 | C | 3.8 |
| X32 | entity-selection circuit | entity-tracking circuit (2024) | C | 1.5 |
| X33 | linear midpoint readout | linear probing/steering | C | 1.5 |
| X34 | X33 replication | — | C | 0.64 |
| X35 | static code mixture | data mixing / code pretraining | B | 7.71 |
| X36 | entity-slot memory + copy head | memory tokens (2020), RMT, entity memory | B | 5.4 |
| X37 | SUTRA low-rank rule grammar | MoE / low-rank experts (MoRE 2025); Pāṇinian framing new | B+ | 14.0 |
| X38 | stable/volatile residual split (CRB) | counterfactual representation, low-rank decomposition; specific split new | A− | 6.7 |
| X39 | causal event attribution | causal attribution; **not run** | — | 0 |
| X40 | intervention next-state + workspace | causal world models, NextLat; specific composition new | A− | 8.5 |
| X40 bench | clean 2×2 composition benchmark | no exact prior match | D | 0 |
| v0.3.1 merge | weight soup | model soups / checkpoint averaging | E | 0 (CPU) |
| v0.3.0 re-warm | continual pretraining | Ibrahim et al. 2024 | E | ~12 |
| v0.2.0 data mix | data mixture | data mixing laws | E | ~30 |
| v0.1.2 | QK-Norm + soft-cap base | QK-Norm / Gemma 2 | E | ~20 |

## The totals

**Mechanism experiments only (X16–X40, excluding base training):**

- **Known / replication / ablation (B/C): ~84 GPU-h.**
- **Genuinely new composition (A−: X38, X40): ~15 GPU-h.**
- **Novel framing of a known mechanism (X37, B+): ~14 GPU-h.**
- **New evaluation (X40 bench): 0 GPU.**
- **Total: ~113 GPU-h.**

If X37 is counted as a new composition, the new share is ~29 GPU-h (26%); if
not, it is ~15 GPU-h (13%). **Roughly three quarters of the mechanism GPU went
to mechanisms that already exist in the literature.**

**Including the base model training (~60 GPU-h), the programme spent on the
order of 170 GPU-h: ~85–100 on known mechanisms/replications, ~15–30 on
genuinely new compositions, and ~60 on the training that produced v0.3.1.**

The model that shipped came from **known** data, schedule and weight-merging
recipes, not from a new mechanism. That is the central finding.

## What was genuinely new

- **X38 CRB** — the stable/volatile split of each block's residual delta is a
  new composition. The components (low-rank decomposition, counterfactual
  invariance) are known. It failed C-2/C-3/C-5; the stable stream collapsed to
  rank ~1.5.
- **X40 intervention pretraining** — an auxiliary intervention next-state
  objective with a workspace and held-out compositions. The broad mechanism is
  an adaptation of causal world models / NextLat; the composition is new. It
  failed I-1 and I-3.
- **X40 2×2 benchmark** — genuinely new evaluation. It separated template
  novelty from composition and exposed the operation-level failure. It cost
  CPU, not GPU.
- **X37 SUTRA** — the Pāṇinian framing is new; the mechanism is a sparse
  low-rank rule mixture, which is known. Call it a novel framing, not a new
  primitive.

## What was not new

X16, X18–X21, X25–X29, X31–X36, X17: reverse LMs, NoPE, gated/static
residuals, test-time training, adaptive data mixtures, replay, QK-Norm,
soft-cap, entity-tracking circuits, linear probes, code mixtures, and memory
tokens all have prior art. They were **local replications or ablations**, not
new mechanisms.

Some of those replications were still worth running:

- **X31** decided that QK-Norm carries the gain and the soft-cap does not.
- **X36/X38** produced the evidence that a live module can cost val and not
  transfer.
- **X29/X35** produced data-mixture/replay nulls.
- **X32–X34** produced the entity-circuit/readout analysis.

But they were not novel, and the audit should not call them novel.

## Positive points

1. **v0.3.1 is real**: 57,856,256 parameters, val 2.7579, bpB 0.9016, public.
2. **The negative record is real**: 35 runs, 260 files, 33 registrations, a
   pre-registered ledger, seed rulers, and archived nulls.
3. **The benchmark is new and reusable**: the 2×2 design and the
   operation-level table expose failures that aggregate accuracy hides.
4. **Some replications paid for themselves**: X31's architecture decision, X36's
   val-tax evidence, X40's control showing the data-only baseline matched the
   mechanism on seen interventions.
5. **The process caught two bugs before wrong results**: the edited-prefix leak
   and the fp16 workspace crash.

## Negative points

1. **~84 GPU-h on known mechanisms**; the novelty claims were often
   adaptations. The fail list is long because the programme kept re-testing
   known ideas under new names.
2. **The genuine new compositions (~15–29 GPU-h) also failed.** Novelty did not
   buy a win.
3. **The model improvement came from known recipes** (data, schedule, merge),
   not a mechanism.
4. **Underpowered accuracy bars**: X18's 0.5 pp bar sat ~4.5× below the
   measured standard error; some GPU bought noise (D45).
5. **The val tax was predictable and repeated**: X36 +0.076, X40 +0.204/+0.169.
   The parameter-matched output adapter beat the mechanism by 0.44–0.48 nats
   in X40.
6. **The X40 template confound** (number+place) was discovered after the run;
   a CPU check before GPU would have caught it.
7. **The "never" claim was not prior-art-checked before GPU in every case.**
   EBAS was withdrawn before GPU only because of the second search; that should
   be the default gate, not a late correction.

## The gate for any next test

No GPU is spent until all five are written down:

1. **Novelty.** Prior-art memo with class A/B/C/D. No “never” claim without a
   search. B/C runs are allowed only as deliberate controls or ablations, and
   are labelled as such.
2. **Power.** The bar is above the measured seed/accuracy noise floor. State
   the standard error and the number of samples (D45).
3. **Val tax.** If the mechanism adds a path, budget the expected val cost and
   include a parameter-matched output adapter as the control.
4. **Benchmark.** For composition/accuracy, use the 2×2 S-P by-operation table,
   not an aggregate. Run it on a CPU checkpoint before GPU.
5. **Cost/stop.** Estimate GPU-hours and expected value; define the kill bar at
   30M. No scale without a pass; no eval tuning; no rescue.

## What I would do now

The audit does **not** support another speculative “never” mechanism. The
only evidence-backed path to beat v0.3.1 on val is the **output-adapter
fine-tune**: known mechanism, 6–8 GPU-h, val bar 2.7579, 2k regression guard.
It is not novel, but it is the one positive signal the programme has.

If the goal is a paper, the benchmark + negative record is the stronger
artifact and costs no GPU. If the goal is another mechanism, the honest answer
is that the current queue has no A-class candidate with a positive prior; the
next idea needs a new prior-art memo, not another run.
