---
name: godmode
description: Prior-art-first research protocol for ML experiments. Classifies novelty, checks the literature twice, computes power and val-tax budgets, defines a kill bar, and refuses known-mechanism GPU spend. Use when designing or approving a new experiment, checking whether an idea is already tested, deciding whether to spend GPU, or when the user asks for a new mechanism, attention variant, or "never tried" idea.
disable-model-invocation: true
---

# Godmode

Godmode is not a new module, a bigger model, or a promise. It is a **process
for making the next GPU-hour informative**.

The failure mode it exists to prevent: a plausible mechanism is renamed,
tested on a small model, fails, and the programme learns nothing because the
mechanism was already in the literature. A long fail list is not proof of
ambition; it is often proof of unasked prior-art questions.

## The one rule

> If the only reason to run it is that it is new, don't.
> If the only reason is that it might improve the metric and it is already
> known, call it engineering, not research.

## The gate — no GPU until all seven are answered

1. **Claim.** What is the new *problem* or *composition*, stated in one
   sentence? "We add X" is not a claim.
2. **Closest three.** What are the three closest papers? Read the abstracts,
   not the titles.
3. **Class.** A, B+, C, or D (below). If B+/C, it is a control or an
   application, not a new mechanism.
4. **Power.** What is the measured seed/accuracy noise floor? Is the bar at
   least ~2 standard errors, or are there enough samples to resolve it?
5. **Val tax.** If the mechanism adds a path, what is the expected val cost,
   and what is the parameter-matched output-adapter control?
6. **Benchmark.** What is the operation-level table? Can a CPU probe answer the
   question before any training?
7. **Kill.** What is the GPU budget, the 30M kill bar, and what will we learn
   if it fails? If the answer is "nothing new," do not run it.

## Novelty classes

| class | meaning | allowed GPU |
|---|---|---|
| **A** | new problem or new composition; no exact prior match | pre-register, 30M screen |
| **B+** | known mechanism, new setting/application | only as a deliberately labelled application, with the prior art cited |
| **C** | replication or ablation | only as a control, never as a novelty claim |
| **D** | new benchmark/evaluation | do it first; it is usually CPU-cheap |

A reviewer saying "this is X applied to Y" is the definition of **B+**, not A.

## Prior-art protocol — twice, independently

1. Run `scripts/prior_art.py "<claim>"` and a second phrasing of the claim.
2. Search exact phrases, not only keywords. Check arXiv, OpenAlex, and the web.
3. Read the closest three abstracts. Write one line each: *why this is not
   the same experiment*.
4. Date the search and record the queries. "Never tested" is a claim with a
   date and a search trail, not a feeling.
5. If the closest paper already trains on the same data, predicts the same
   target, or evaluates the same split, it is **C**, not A.

## Power protocol

- Use loss when the sample is small; accuracy at 500 examples has ~2 pp
  standard error.
- Report seed spread as a ruler. A bar inside the spread is a coin flip.
- Prefer 2 seeds at 30M and a bar above the observed spread.
- Do not "add samples until it passes"; fix the sample size before the run.

## Val-tax protocol

- Any new module/path pays a language-quality cost until proven otherwise.
- Include a parameter-matched output adapter; in the X40 screen the adapter
  beat the mechanism by 0.44–0.48 nats.
- If val regresses beyond the pre-registered guard, close even if the target
  capability moves.

## Benchmark protocol

- Report **by operation**, not aggregate. A 4-way aggregate can hide a 0.0
  operation and a chance operation.
- Separate template/lexical novelty from composition: seen/unseen sentence ×
  single/pair change.
- Use a gold-edit/interface probe to separate "apply the edit" from
  "continue the edited world."
- A new benchmark is a legitimate **D** result; do not disguise it as a
  mechanism.

## Run order

```text
problem
  -> prior art (twice)
  -> CPU probe / existing checkpoints
  -> pre-registration (hash, bars, seeds, configs)
  -> 30M screen (2 seeds)
  -> 58M only if pass
  -> 134M only if 58M passes
```

No scale without a pass. No eval tuning. No rescue.

## Red flags

- "It's just a small module." The val tax is not small until measured.
- "We'll tune the weight." That is eval tuning unless fixed before the run.
- "The trend is promising." A trend inside seed noise is not a result.
- "It's never been tried." Without a dated search trail, it is a guess.
- "The aggregate improved." Show the operation-level table.
- "One seed showed it." Report the spread; do not promote.
- "The data baseline is close." Then the mechanism is not the lever.
- "A bigger model will fix it." Only if the 30M screen shows a signal.

## Logging

- `prior-art.md`: queries, dates, closest three, class.
- `prereg.md`: claim, arms, controls, bars, seeds, cost, kill rule; hash it.
- `result/README.md`: positive **and** negative; operation-level rows.
- Update the run registry and the decision log.
- A null is a deliverable. A null that was already in the literature is a
  process failure, not a result.

## Worked example — OIT/IOAM

Claim: "ordered intervention sequences + attention mediation are new."

Search result: order prediction/contrastive pretraining (COCO-LM 2021,
ALBERT SOP), multi-step instruction following (MUFFIN, ProcBench), order
sensitivity/compositional generalization (SCAN/COGS), and causal attention
mediation/head gating (2020–2025) are prior art. No exact match for the
combined ordered-intervention benchmark, but a reviewer could call it an
application.

Verdict: **OIT = B+** (known order-contrastive training, new application);
**IOAM = D** (new benchmark/problem, not a new mechanism). Run IOAM as a cheap
diagnostic if the benchmark question matters; do not call OIT a new mechanism.

## The final test

Before approving GPU, ask:

> If this fails, what do we learn that the literature does not already know?

If the answer is "that this known mechanism also fails here," do not run it.
If the answer is "a new problem is measurable, and the result changes the next
decision," run the cheap version and log it.

Godmode is not certainty. It is the refusal to buy the same null twice.
