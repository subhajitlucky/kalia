# The Bhagavad Gita as Algorithmic Inspiration — Fourth Pass

Status: research queue
Date: 2026-09-23
Sources: Bhagavad Gita (2.47-51, 3.20-25, 3.35, 14, 18), Antahkarana-Net
(gnua-controller architecture, 2025; validated on Split-CIFAR/MNIST), Sattvic AI
framework (NPRC Journal, 2026), svadharmic decision theory (Wiese, 2025),
Darshana LLM framework (2026; stage-gated SFT validated), nishkama-karma
decision analysis (Sophia, 2020).

## The Gita's core concepts and what they compile to

| Gita concept | Meaning | Compiles to |
|---|---|---|
| **Gunas** (sattva/rajas/tamas) | Three modes of nature: clarity, activity, inertia | A **mode controller** for training dynamics: explore (rajas) / consolidate (sattva) / decay-and-prune (tamas) |
| **Samatva** (2.48, equanimity) | Evenness, balance | **Adaptive balance of multiple objectives** (aux-loss weighting that equalizes relative progress) |
| **Nishkama karma** (2.47, action without attachment to fruits) | Do the work, not the reward | **Target-loss stopping** — stop when quality is reached, not when the fruit (step count) arrives |
| **Svadharma** (3.35, own duty) | Better one's own duty imperfectly than another's well | **Per-domain specialization** (adapters/LoRA per task; two-stage routing: identity first, consequences second) |
| **Buddhi yoga** (2.50, skill in action) | Disciplined understanding in action | Journaling/decision review — already practice (this document) |

## Prior art (stated plainly — this is the honest part)

- **Antahkarana-Net** (2025) already implements a **GunaController** mapping a
  (sattva, rajas, tamas) vector to plasticity, exploration, consolidation and
  pruning, wrapped around an EWC-based continual-learning memory. Validated on
  vision benchmarks (forgetting cut ~60–80×, gated accuracy 0.978). It also
  includes a calibrated abstention gate (pramana) — conceptually adjacent to our
  Nasadiya head (X20) — and a reward-invariant monitor (turiya).
- The **Sattvic AI framework** (2026) applies dharma/nishkama/gunas as an
  ethics-and-governance layer (Dharma Statement, Guna Impact Assessment).
- **Svadharmic decision theory** formalizes the Gita's choice structure as a
  Rational Shortlist Method: the svadharma filter first, consequences second.
- The **Darshana framework** (2026) maps six schools to LLM layers and
  validates stage-gated SFT (60–62% win rates against reverse curricula).

Consequence: a guna-mode controller is **not novel**. Adopting it is adaptation
of validated prior art, and must be cited as such. What is genuinely new to test
is its application to *from-scratch LM pretraining at 58M on free GPUs* with our
specific measured signals.

## Queue

| ID | Experiment | Mechanism | Cost | Priority |
|---|---|---|---|---|
| X23 | Guna mode controller for pretraining | Three modes switched by measured dynamics (loss trend, gradient-norm variance, validation delta): rajas = higher LR on fresh data; sattva = balanced mid-training; tamas = decay + EMA consolidation | 2× 1.5h GPU | medium (adapted prior art) |
| X24 | Samatva aux-loss balancing | Adaptive λ for auxiliary heads (MTP/Nasadiya/ELECTRA) equalizing relative improvement rates, vs fixed λ | 2× 1.5h GPU | when aux heads land |
| — | Nishkama stopping | Already implemented (target-loss stop, P3) | — | done |
| — | Svadharma adapters | Already planned (continual-learning composition, D26) | — | planned |
| — | Process impact statement | Sattvic AI's Guna Impact Assessment adapted as a short "training process impact" paragraph in the model card (energy, data ethics, stability) | docs only | planned |

## What we deliberately do not take

- Nishkama-karma game theory (prisoner's dilemma cooperation) — multi-agent
  reward design, not applicable to single-model pretraining.
- Dharma-as-identity prompts for LLMs (2026 frameworks) — an inference-time
  prompting pattern, not a pretraining mechanism; out of scope for KALIA.
