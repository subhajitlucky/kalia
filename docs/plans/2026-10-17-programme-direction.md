# Programme direction — the goal, the four metrics, and the ladder

**Status:** draft for approval, 2026-10-17
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
- the **TTT null** (X25) and the **entity-decode win** (+0.061 retention, 3×
  greedy);
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
| **M0 — v0.2.1 ships** (this window) | The seed-1402 replicas decide whether uniform 10% replay + rewarm beats no-replay beyond the spread. If yes: export, evaluate (benchmarks at `limit=2000` — CPU is quota-free), publish, card carries the 1402 outcome, the seed-1337 caveat, and the QK/soft-cap evidence note. | Replication reproduces direction on both instruments |
| **M1 — usage-loop v0: memory only** (1–2 weeks, ~0 GPU) | Retrieval + entity-consistent decoding around the frozen model. Users get a model that remembers across a session; no weights move. Measure entity/name retention vs greedy (the existing protocol). | Memory lift is real and costs ~0 GPU-h — this is the number C that weight updates must beat |
| **M2 — usage-loop v1: gated weight updates** (2–6 weeks, ≤4 GPU-h/cycle) | Curated interactions → periodic continual update (rewarm + uniform replay, the v0.2.1 recipe) → scored on A, R, C, G with the ruler rule. Publish the first acquisition-vs-retention-vs-cost curves at sub-billion scale. | Three consecutive cycles with ΔA above spread and R within it; any G rejection triggers a rebuild, not a retry |
| **M3 — the artifact** | The failure atlas (dessert, replay ladder, TTT null, wrong-instrument class) + the four curves + the harness and ledger format. A methodology reference for cheap continual learning that does not exist in this form. | Independent reproduction of one cycle from the repo alone |

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
