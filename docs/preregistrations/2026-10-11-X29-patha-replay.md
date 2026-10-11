# X29 — Patha replay: Vedic recitation orderings as continual-pretraining replay schedules

**Status:** registered 2026-10-11, before any run.
**Depends on:** the continual-update infrastructure (replay shard, CL-0 ledger),
the X28 trio (running; supplies the run-to-run noise bound), and the Vedic
recitation scholarship on the pāṭha methods (chunking and forward/backward
chaining as an oral-memory technology).

## Why this, and what is actually new

Replay scheduling is a crowded 2026 field — Spaced Repetition Training
(SuperMemo-2), FOREVER, Replay-on-Demand, MSSR — and **every one of them
schedules from psychology: Ebbinghaus curves, SM-2 intervals, model update
magnitudes.** The Vedic pāṭhas are a different lineage: a 3,000-year-old,
field-tested set of *structured rehearsal orderings* designed to preserve a
large corpus by memory alone. Nobody has tested them as replay schedules for
language-model training. (Nearest prior art found 2026-10-11: an education
literature connecting pāṭhas to memory science, and a personal-memory app
using the same views for retrieval storage — neither trains a model.)

X16 tested jata-patha as a **text transform** (reversing chunk order) and
failed. This arm changes the **schedule**, not the data: which replay windows
return, and in what order.

## The four arms

One continual update of v0.2.0 per arm, seed 1401, `rewarm: true`, 1000 update
steps, replay fixed at 10% of every batch. Identical in everything except the
replay presentation order:

- **uniform (control):** replay windows drawn in a seeded shuffled cycle —
  the same set as the others, no structure.
- **krama:** overlapping forward pairs, `12, 23, 34, ...` — each interior
  window returns adjacent to its neighbour.
- **jata:** forward-reverse-forward, `12, 21, 12, 23, 32, 23, ...`
- **ghana:** nested triples, `12, 21, 123, 321, 123, ...`

Fairness constraints, registered: all arms draw replay from the **same 256
seeded base windows** of the same shard, at the same 10% budget, so the total
replay rows are equal by construction; only adjacency and repetition structure
within the replay stream differ.

## What is measured

- **Retention:** the CL-0 ledger (`tools/forgetting_probe.py`) on each final
  checkpoint, drift from best-ever. This is the metric replay exists to move.
- **Capability:** canonical held-out val (the same 100-step protocol).
- Both read against the X28 static rerun's measured run-to-run agreement
  (~0.0006 nats at step 4900; final number when X28 completes).

## Registered expectation

Null or small. The patha structure is a hypothesis about rehearsal, not a
measured effect at this scale, and the X16 precedent says data-side changes
have failed here before. The value of this arm is that it is the first test of
recitation-derived schedules; a clean null is a publishable observation about
which memory technologies transfer to gradient-based learners.

## Budget and sequencing

Four arms x ~4.7 GPU-h ≈ 19h — more than the remaining window. Operational
priority if quota is short, registered now so it is not chosen after seeing
results: **uniform and jata first** (the most distinct contrast), then ghana,
then krama. A 500-step screen (≈ 9h for all four) is an acceptable substitute
and must be labelled as a screen, not a result.

## Stopping

Session quota. No promotion from this run alone; any winner is replicated at
a second seed before a claim.
