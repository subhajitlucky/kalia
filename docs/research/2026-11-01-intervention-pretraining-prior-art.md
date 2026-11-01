# Intervention Pretraining — prior-art check

Status: active research record
Date: 2026-11-01
Question: is an **auxiliary pretraining objective** that trains a latent event
workspace in a small LM to predict the next state under do-interventions, and
tests **unseen compositions**, something that has not been released?

Method: web search plus arXiv search for the exact phrase, causal world models,
counterfactual next-state prediction, world-model fine-tuning, causal
representation learning, and event prediction. Focused check, not a systematic
review.

## Verdict

**No exact match was found.** The nearest work builds **external causal world
models** or **fine-tunes** LLMs for precondition/effect prediction. The candidate
novelty is narrower:

> To our knowledge, no prior work makes intervention next-state prediction an
> **auxiliary pretraining objective** inside a small LM, with a learned event
> workspace, and evaluates it on **unseen compositions** at 58M/134M.

The claim is provisional. It is a new composition, not a new primitive.

## Closest prior art

| work | what it does | why it is not X40 |
|---|---|---|
| **Language Agents Meet Causality — Bridging LLMs and Causal World Models** (ICLR 2025) | learns a causal world model from causal representations and exposes it to an LLM for planning; predicts intervention effects | an **external** world model + LLM interface, not an auxiliary pretraining loss inside the LM; no event workspace trained by next-state prediction; no unseen-composition pretraining test |
| **Making Large Language Models into World Models with Precondition and Effect Knowledge** (COLING 2025) | fine-tunes separate LLMs to predict action preconditions and effects | post-training/fine-tuning for planning, not a pretraining objective; no latent event workspace; no small-LM scaling curve |
| **Causal Transformer for Estimating Counterfactual Outcomes** (ICML 2022) | transformer for longitudinal treatment-effect estimation | medical/time-series outcomes, not language pretraining; no event workspace or LM next-token path |
| **CausaLM** (2021) | counterfactual language representations for classifier explanations | BERT classification explanations, not generative LM pretraining; no intervention next-state objective |
| **SPARTAN** (NeurIPS 2025) | sparse transformer world model for object-centric environments | RL/world-model setting, not text/LM pretraining; state tokens are objects, not language |
| **Causal Distillation** (2022) | causal abstraction between teacher and student | distillation; the user is against distillation; different objective |
| **Counterfactual-Augmented Representation Learning based Event Prediction** (ICME 2025) | counterfactual augmentation for event graphs | event prediction, not an LM pretraining objective |
| **Verbalizable Representations Form a Global Workspace in LMs** (2026) | a global workspace in LM representations | analysis of an existing workspace; not a causal intervention objective |

## Exact-phrase searches

- **"intervention pretraining"** — no exact match.
- **"counterfactual next-state"** + language model — causal/RL world models, not
  an LM pretraining objective.
- **"event workspace"** + intervention — no LM pretraining match.
- **"causal world model"** + language model — external simulators and planning.
- **"counterfactual next-state prediction"** — RL/decision-transformer work, not
  text pretraining.

## What this means for the paper

- The paper cannot claim a new **primitive**. Causal world models and
  intervention prediction are established.
- It can claim a new **training interface**: intervention next-state prediction
  as an **auxiliary pretraining loss**, with a latent event workspace, at
  58M/134M, evaluated on unseen compositions.
- The introduction must cite the causal-world-model/LLM line, the
  world-model fine-tuning line, and causal representation learning.
- The key baselines are a **pairs-only** arm (counterfactual text as extra LM
  data) and a **parameter-matched adapter**; if the event workspace does not beat
  pairs-only on unseen compositions, the workspace is not the lever.

## What would falsify the claim

- A paper that adds intervention next-state prediction as an auxiliary
  pretraining loss with a latent workspace and an unseen-composition test.
- A paper that trains an event workspace inside an LM with do-interventions at
  small scale and reports a scaling curve.
- A paper that calls the objective "intervention pretraining" and tests
  compositional counterfactuals.

Search again before submission. If a closer paper appears, the work becomes a
replication/extension; still useful, but a different claim.

## The honest sentence

We are not inventing causal world models. We are testing whether **making
intervention prediction part of pretraining**, through a small event workspace,
produces a capability that survives unseen compositions — and whether it scales.
