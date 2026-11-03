---
name: llm-godmode
description: Persistent LLM-research idea search and experiment gate. Uses the failure ledger and prior-art probes to find an A-class experiment or prove saturation; kills B+/C novelty claims and prevents GPU spend on known mechanisms. Use when the user asks for a new LLM mechanism, attention variant, benchmark, or experiment, or says "llm godmode".
disable-model-invocation: true
---

# LLM godmode

This is the LLM-research specialization of `godmode`. It does not promise a
new mechanism. It promises that **no GPU is spent on a known one**, and that
the search keeps going after the obvious ideas die.

## What success means

Success is one of:

1. an **A-class** candidate — a new problem or composition with no exact prior
   match — plus a cheap test; or
2. a **documented saturation map** plus the cheapest **D-class** benchmark that
   makes the gap measurable.

A B+ is not success. A null on a B+ is a process failure, not a result.

## Inputs

- `docs/research/2026-11-03-novelty-audit.md` — what was already tested and
  what it cost.
- `docs/research/llm-idea-ledger.md` — every candidate, its closest prior art,
  class, status, and next action.
- `.cursor/skills/godmode/lenses/llm-research.md` — the experiment gate.
- `.cursor/skills/godmode/scripts/prior_art.py` — OpenAlex + arXiv probe.

## The loop — keep searching, not keep buying nulls

1. **Read the ledger.** Cross out every candidate that failed or is B+/C.
2. **Pick one search axis:** problem, data, objective, architecture, evaluation,
   scale/regime, or transfer from another field.
3. **Generate 8–10 claims** on that axis. Use the prompts below.
4. **Probe twice** for each claim with different phrasings. Read the closest
   three abstracts. Classify A/B+/C/D.
5. **Kill B+/C for novelty.** If D, ask whether the benchmark is cheap and
   worth doing first. If A, write the one-page pre-registration.
6. **If no A, change axis.** After three axes without an A, declare saturation
   and choose the cheapest D benchmark or stop. Do not “try harder” by running
   a B+ on GPU.

## Candidate prompts

- What does the failure ledger say the model cannot do? (binding, order,
  conservation, feasibility, ambiguity, calibration, long-horizon state.)
- Invert the task: what if the model must **refuse**, **ask**, **verify**,
  **simulate**, **teach**, or **undo**?
- Transfer: how do databases, compilers, control theory, law, biology, or
  history handle the same problem?
- Evaluation-first: what measurement would make a mechanism visible? Is the
  benchmark itself the contribution?
- Regime: what changes at 1,024 context/58M vs 128k/7B? What is the smallest
  model where the capability could appear?
- Data: what data is missing — impossible edits, ambiguous edits, multi-party
  stories, temporal chains, conservation violations?
- Scale: is the claim a scaling curve rather than a mechanism?

## The classes

| class | meaning | action |
|---|---|---|
| **A** | new problem/composition; no exact prior match | pre-register; CPU probe; 30M screen |
| **B+** | known mechanism, new setting | no GPU as a novelty bet; label it |
| **C** | replication/ablation | control only |
| **D** | new benchmark/evaluation | CPU/cheap; do it first |

If a reviewer would say “this is X applied to Y,” it is B+.

## The search history is the asset

The audit already maps the dead ends: reverse LMs, NoPE, gated/static residuals,
test-time training, adaptive mixtures, replay, QK-Norm/soft-cap, entity
circuits, linear probes, code mixtures, memory tokens, rule MoE, CRB,
intervention pretraining, order-contrastive training, attention mediation,
edit algebra, reversibility, locality, and commutators. Do not re-enter those
rooms.

The remaining new work is usually a **D**: a benchmark or measurement that
makes the next mechanism testable. That is a legitimate success. The 2×2
benchmark is the proof.

## The no-GPU rule

- B+/C: no GPU. Write it down and move on.
- D: CPU or existing checkpoints first.
- A: pre-register, CPU probe, 30M screen; 58M only if the 30M passes.
- If the answer to “what do we learn if this fails?” is “that this known
  mechanism also fails here,” do not run it.

## The final test

> After the search, can I name the closest three papers and say exactly why
> this is not the same experiment?

If not, it is not A. Keep searching CPU-side; do not buy the same null twice.
