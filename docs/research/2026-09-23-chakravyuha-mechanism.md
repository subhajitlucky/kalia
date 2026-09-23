# Chakravyuha as a Mechanism — Correcting the Earlier Dismissal

Status: audit corrected; one deliverable built; X16 redesigned
Date: 2026-09-23
Sources: Padmavyuha/Chakravyuha literature (Mahabharata; "Dynamics of an
Invincible Troop", INDECS 2021); TCS Codevita "Chakravyuha Problem"; Dutta 2025
(RL multi-layered defense inspired by Chakravyuh); Gupta (chaos/fuzzy
cybersecurity model); reversal-curse and reverse-training literature (Berglund
et al. 2023; Golovneva et al. 2024; reverse modeling, arXiv 2410.09817;
reversal invariance, arXiv 2511.00341).

## What the formation actually encodes

- Seven concentric tiers, breached in **sequential order**, forming a labyrinth.
- The formation is **dynamic**: after the intruder is inside, the commander
  rearranges roadblocks and creates dead ends. The escape route "can only be
  devised within the Chakravyuha while it is in action" — there is no
  predefined exit rule.
- **Abhimanyu knew the entry, not the exit** (he overheard only the penetration
  technique before birth). He breached six tiers and died inside.

The strategic content is not the geometry: it is **asymmetric entry/exit
knowledge under a closing system**, and the cultural usage ("you have fallen
into a chakravyuha") names exactly that pathology.

## Prior art (honest)

- The Chakravyuha is literally a competitive-programming problem (spiral
  traversal, TCS Codevita).
- Multi-layer defense inspired by it is published with RL results (Dutta 2025:
  mean time to breach 52 episodes, ~82% trap efficiency) and with chaos/fuzzy
  models (Gupta). "Chakravyuha as defense" is taken.

## Compilations

### C1 — Abhimanyu gap (built): entry-exit asymmetry measurement

`eval_reversibility.py`: forward NLL vs reversed-token NLL on held-out text,
token-weighted. A large gap means the model can enter fluent text but cannot
exit by processing it backwards — the computational Abhimanyu condition.
Related measures: reversal curse; forward-minus-reverse loss as a data-quality
signal (arXiv 2410.09817).

### C2 — Chakravyuha reversal training (redesign of X16)

Original X16 (Vedic Jata-patha) proposed a reverse-prediction head. The
reversal-curse literature shows the stronger, validated form is **data-level
chunk-preserving reversal**: token reversal, word reversal, entity-preserving
reversal, and random-segment reversal; entity-preserving and random-segment
variants work best. In the data-bound regime it **doubles effective tokens**
and improves standard benchmarks without changing the architecture.
KALIA is data-bound; we are a direct fit.

Revised X16 arms: control vs random-segment reversal augmentation (50% of
sequences reversed in chunks of random length), evaluated on the Abhimanyu gap
and validation loss.

### C3 — Rule: know the exit before you enter

Formal project rule (extends the publish checklist): every experiment and
system must specify its exit/rollback criteria *before* it starts — stopping
condition, revert path, and what "trapped" would look like. Abhimanyu's death
is the case study.

## What remains discarded

The spiral geometry itself (a formation is not a mechanism) and the specific
RL-defense results (already published). The asymmetry insight is what we keep.
