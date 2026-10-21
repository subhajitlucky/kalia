# Programme direction — the goal, the four metrics, and the ladder

**Status:** draft for approval, 2026-10-17. **Amended 2026-10-18** after M0's gate
was read on a clean contrast: M0 is measured rather than projected, metric R's
gate turned out to be green by construction, and the ladder's ordering is
contested below with evidence. **Amended 2026-10-21** after the entity-holdout
test: the recorded +0.061 entity-decode win is circular, and the honest
held-out-half result is +0.011 [−0.007, +0.032]. The goal, the four protocols
and the non-goals are unchanged.
**Supersedes:** nothing. Does not touch the v0.3.0 registration; its stopped
steps stay stopped.

---

## The goal, in one sentence

**Become the reference for honest, cheap continual learning: a sub-billion model
whose updates from use are measured on four curves — acquisition, retention,
cost, and gaming — with the failure modes published, not hidden.**

What that is *not*: frontier-model parity. KALIA is 57.9M parameters trained on
free quota at 43 tokens/parameter (D20). Aiming at "Claude-class" from here is a
guaranteed failure by construction, and an implicit target that only generates
bad decisions. The project already owns assets no frontier lab has a reason to
build:

- the **CL-0 retention ledger** (`forgetting_probe.py` + the G2-pinned
  `corpus/probe_val.bin`), whose self-check reproduces a five-day-old baseline
  to 1e-6;
- the **dessert-drift finding** (X26/X28): a policy that reads a self-referential
  signal games its own metric — the exact failure mode of every
  "learns from usage" system, demonstrated with a dose-response ladder;
- the **replay ladder** (X29): plain shuffled replay helps retention, structure
  costs in proportion to depth — measured on two instruments;
- the **TTT null** (X25) and the **entity-decode coverage result** — the oracle
  finds 0.0618 on a held-out half against greedy's 0.0170, so the right
  continuation is in the pool; the recorded +0.061 full-text "3×" is circular
  and did not survive the holdout test (`docs/results/entity-holdout/`);
- the pre-registration + incident apparatus that makes every one of the above
  auditable.

## The four metrics (exact protocols, fixed before the first usage loop)

Any update cycle is scored on all four. A single number is not a result.

### A — Acquisition

*Did the update teach the model anything from the new data?*

- **Instrument:** a frozen acquisition shard, `corpus/acq_v1.bin`, built by
  `prepare.py` from the curated usage corpus with its own held-out split
  (10%), hashes recorded in `docs/legal/val-shard-hashes.json` at build time.
- **Measurement:** the registered deterministic protocol —
  `eval_val.py --batches 100 --batch-size 8 --seed 1234` — on the acquisition
  held-out split, plus a 200-item frozen cloze set of facts/entities drawn from
  that split (`eval/acq_prompts.json`, hashed).
- **Report:** ΔA = pre-update − post-update, on both instruments, each with its
  measured spread (see the ruler rule below).
- **Never:** training loss on the update stream. That is the dessert's food.

### R — Retention

*Did the update destroy what the model already knew?*

- **Instrument:** CL-0 on the G2-pinned probe shard, exactly as the X29 ledger
  runs it (sha `1664f71a…`, baseline row seeded, v0.2.0 self-check before any
  arm is believed).
- **Report:** drift vs the current pretrain baseline, per checkpoint, on the
  ledger. Session noise bound: ≤0.003 per checkpoint (measured, X28).
- **Gate:** no update ships whose R drift exceeds the update's own measured
  spread (below), regardless of ΔA.

  **Amended 2026-10-18 — as written this gate is green by construction whenever
  nothing is forgetting, and that is exactly what happened.** In the
  `x29-replication` cohort every arm *improved* on the probe: −0.0237, −0.0222,
  −0.1122, −0.1161. A rule of the form "drift must not exceed the spread" passes
  all four and discriminates nothing. R is a signed change in loss on a fixed
  training slice; below zero it measures **how much extra fitting happened**, not
  how much was retained. Read it as a comparative test — *any arm whose R is
  worse than its paired control's by more than the spread is rejected* — and treat
  "what a positive drift would mean" as open, because in this cohort no arm ever
  produced one.

### C — Cost

*What does a unit of knowledge cost?*

- **Record:** GPU-hours (kernel wall time × GPUs, as Kaggle charges), tokens,
  wall time, and acquisition-per-GPU-hour — one ledger row per update, appended
  to the forgetting ledger so cost and retention live in the same artifact.
- **Purpose:** the ladder competes on knowledge per quota-hour, not on knowing
  more per update at any price. Memory-only updates cost ≈0 GPU-h and are the
  benchmark every weight update has to beat.

### G — Gaming

*Is the update eating dessert?*

- **Instrument:** per-source training loss (the `per_source_log.csv` path)
  against **frozen per-source probe loss** on shards the update never trains on.
- **Signature to detect:** training loss falls while frozen-probe loss is flat
  or rises, *and* source weights drift monotonically toward the easy sources.
  X26 measured all three; the trend policy (X28) halved them but did not stop
  them.
- **Rules:** every signal a policy consumes comes from a frozen probe, never
  from the training stream (the X28 lesson; frozen-probe weighting is standard
  practice — DoReMi/ODM — and is claimed as instrumentation, not novelty).
  Any update with a positive frozen-probe cost and monotone drift is rejected
  regardless of ΔA.

### The ruler rule (D48, applied to updates)

Before any ΔA or ΔR is interpreted, **run the same update twice at two seeds**
from the same checkpoint and record the update's own spread. No delta below
that spread is a result. This is v0.3.0 Step 1's lesson institutionalised: the
ruler is measured before it is used, and its cost (~2 arms) is part of every
update budget.

## The ladder

| Milestone | What | Gate |
|---|---|---|
| **M0 — v0.2.1** (read 2026-10-18) | **Gate not passed.** The seed-1402 pair plus a no-replay control ran. No export, no publish. Detail below. | Replication reproduces direction on both instruments — **it does not**: val and CL-0 disagree on the replay contrast |
| **M1 — usage-loop v0: memory only** (1–2 weeks, ~0 GPU) | Retrieval + entity-consistent decoding around the frozen model. Users get a model that remembers across a session; no weights move. Measure entity/name retention vs greedy (the existing protocol). | Memory lift is real and costs ~0 GPU-h — this is the number C that weight updates must beat |
| **M2 — usage-loop v1: gated weight updates** (2–6 weeks, ≤4 GPU-h/cycle) | Curated interactions → periodic continual update (rewarm + uniform replay, the v0.2.1 recipe) → scored on A, R, C, G with the ruler rule. Publish the first acquisition-vs-retention-vs-cost curves at sub-billion scale. | Three consecutive cycles with ΔA above spread and R within it; any G rejection triggers a rebuild, not a retry |
| **M3 — the artifact** | The failure atlas (dessert, replay ladder, TTT null, wrong-instrument class) + the four curves + the harness and ledger format. A methodology reference for cheap continual learning that does not exist in this form. | Independent reproduction of one cycle from the repo alone |

**M1's evidence changed on 2026-10-21.** The entity-decode +0.061 is circular;
the held-out-half test is **+0.011 [−0.007, +0.032]**, not distinguishable from
zero at 20 stories. The coverage result (oracle 0.0618 vs greedy 0.0170) still
says the right continuation is in the pool, but the entity rule does not
reliably retrieve it. M1 must be scoped as a **selector/coverage** problem, not
a proven memory lift, and its 0-GPU-hour claim no longer holds for the completed
measurement: the CPU protocol could not finish 20 stories.

## M0 as measured (2026-10-18)

The gate was written as *"uniform 10% replay + rewarm beats no-replay beyond the
spread"*, to be read on both instruments. A clean contrast — same runner, same
2.4B corpus, same budget, replay the only variable — gives:

| instrument | replay vs no-replay | standing against the ruler |
|---|---|---|
| CL-0 retention | **+0.0056 / +0.0055 worse at two seeds** | 3.6× the measured 0.0015 spread |
| canonical val | +0.0012 / −0.0020 | sign-unstable, inside the 0.0018 spread |

And the gap the 1402 pair was originally read against — uniform minus the X28
`static` arm, 0.0884 and 0.0939 on CL-0 — decomposes into **6% replay and 94%
delivery path**. The path is the corpus and the number of passes over it (498M
tokens ~1.05 times, against 2.4B ~0.2 times), not the replay.

**So the gate fails, and it fails informatively.** Weight updates did not beat a
no-update baseline on the instrument the registration named as primary; on the
capability instrument the difference is not resolvable.

### And the stop that preceded it used the wrong ruler

v0.3.0's Steps 2–4 were cancelled on 2026-10-08 when Step 1 measured a **0.2892**
seed spread against a registered 0.15 threshold. That measurement was of arms
trained **from scratch**. The arms it stopped are **continual updates from a
shared checkpoint**, whose spread this project measured for the first time on
2026-10-18: **0.0015–0.0070**, some 40–200× smaller. The work then resumed under
X26/X28/X29 registrations, which was the right instinct, but the false stop is
recorded nowhere in the v0.3.0 registration.

Both facts point the same way for this plan: **the ruler must match the regime
being measured**, and a threshold inherited from a different one is the D45
failure in a new costume.

## Contested: the ladder's ordering

Recorded rather than resolved, because it is a judgement about what to bet on and
it is the owner's.

The evidence to date is lopsided. Measured positive: a corpus change (**−0.2285
nats**, the largest effect this project has ever produced) and the entity-decode
**coverage** result (the right continuation is in the pool). The recorded
**+0.061 "3×" is circular**; the held-out-half test is **+0.011
[−0.007, +0.032]**, so the rule is not the win. Measured null or worse:
test-time training (closed null across a 40× range), the adaptive mixture
(+0.018 worse, and it ate dessert), the replay ladder (orderings that cost
proportionally to their structure), reverse-aware attention (withdrawn pre-GPU),
and now replay itself (6% of an effect that was 94% corpus).

The ladder as drafted puts **M2 — gated weight updates** second, and that is the
mechanism with the least evidence the project owns. **M1 — memory only** contains
the one inference-time result that measured a win, at zero GPU cost, and the plan's
own kill criterion says to prefer it if it matches weight updates on ΔA. That
criterion can be satisfied by evidence this project already has, without waiting
for M2.

The suggested change is to promote the memory-only rung and demote weight updates
to a rung that must beat a **zero-GPU-hour baseline** before it runs again. That
is a reordering, not a new goal: the four metrics, the ruler rule and the
falsification criteria all still apply.

## What changes immediately (from the 2026-10-17 review)

1. QK-Norm + soft-cap re-test, 2 seeds, ablated (~2–3 GPU-h) — the one shipped
   ingredient whose micro delta (−0.083) sits below the measured 30M spread
   (≈0.11) and was never replicated. Either the card keeps it with evidence or
   the recipe gets simpler.
2. Benchmarks at `limit=2000` for every release claim (halves σ; costs CPU
   only).
3. WSD / flexible-decay implemented (recommended 2026-09-23, never landed;
   v0.2.0 paid 6.3 h for a flat tail it exists to avoid).
4. Runner kernels clone the public repo at a pinned commit (or record the code
   revision), killing the I10/I23/X29-v1 stale-bundle class; run records carry
   code revision + dataset versions.
5. Training checkpoint sync moves to a repo that stays private; release repos
   carry final artifacts only (≈59 GB of LFS history today).

## Non-goals

- Frontier parity, chatbot behaviour, instruction tuning for its own sake.
- Per-user weight updates (privacy, drift, and X26/X28 already show the
  mechanism's failure mode).
- Any update trained on a signal that also trains the model (self-reference
  ban).
- Publishing usage data beyond the licence conditions in the matrix.

## Falsification and kill criteria

- If the update's own spread (the ruler) exceeds any acquisition effect at 58M,
  the weight-update track closes — memory-only becomes the product, and the
  four-curve null is still the contribution.
- If memory-only matches weight updates on ΔA at ≈0 GPU-h, prefer memory.
- If gaming signatures appear despite frozen probes, stop and rebuild the
  instrument before another cycle runs (the D48 precedent: an unmeasured ruler
  invalidates everything read against it).
