# CEA prior-art check — is a learned edit algebra new?

Status: active research record
Date: 2026-11-01
Question: is a **learned algebra of edit operators on LM hidden states**, with
involution/commutation/cycle constraints and a counterfactual next-token
objective, something that has never been released?

Method: web search plus arXiv full-text search for the exact phrases and the
nearest concepts. This is a focused check, not a systematic review.

## Verdict

**No exact match was found. The closest prior art is function-vector algebra.**
The claim must be narrower than "we invented an algebra of edits."

> To our knowledge, no prior work learns a set of edit operators on LM hidden
> states, constrains them with involution/commutation/cycle rules, trains them
> by counterfactual next-token prediction, and tests **unseen compositions** of
> those operators.

The new part is the **constraints plus the unseen-composition test**, not the
idea of composing activation vectors.

## Closest prior art

| work | what it does | why it is not CEA |
|---|---|---|
| **Function Vectors in Large Language Models** (Todd et al., ICLR 2024) | extracts activation vectors from in-context examples; function vectors compose by addition/subtraction to form new task vectors (a parallelogram in activation space) | vectors are **extracted**, not learned operators; composition is additive, not an algebra with involution/commutation; no counterfactual next-token objective; no unseen-composition test with edit operators |
| **Editing Models with Task Arithmetic** (Ilharco et al., ICLR 2023) | weight-space task vectors; negation, addition, scaling, analogies | parameter space, not hidden-state edit operators; no algebra constraints; not counterfactual next-token |
| **In-Context Algebra** (ICLR 2026) | studies algebraic structure emerging in in-context learning: commutative copying, identity and cancellation facts | studies emergent behaviour, not a learned operator module; no edit types, no counterfactual pairs |
| **Orthogonal Representation Editing** (2026) | hidden-space edit vectors with orthogonal constraints for batch knowledge editing | knowledge editing, not composable edits; no involution/commutation/cycle; no counterfactual next-token |
| **Representation editing / activation steering** (Turner et al.; Zou et al.) | adds steering vectors to hidden states to control behaviour | single directions, no learned operator algebra, no composition test |
| **Edit-based sequence models** (Levenshtein Transformer; EDUCATED) | discrete token insertion/deletion/replacement for editing text | discrete text edits, not hidden-state operators; no algebra |
| **Model editing** (ROME, MEMIT, AlphaEdit) | factual knowledge updates via weight/hidden-state edits | factual editing, not a composable edit algebra |
| **Concept algebra / word2vec arithmetic** | vector arithmetic over semantic concepts | static embeddings or semantic directions, not learned edit operators |

## Exact-phrase searches

- **"Causal Edit Algebra"** — no match.
- **"edit algebra"** on arXiv — 2 results: source-code algebra (ICSME 2026)
  and reversible diffusion ODE solvers (ICML 2026 workshop). Neither is an LM
  hidden-state edit algebra.
- **"counterfactual next-token"** on arXiv — 0 results.
- **"edit operators" + hidden state / language model** — representation editing
  and knowledge editing, not a learned composable algebra.
- **"function vector algebra"** — the closest: activation vectors compose by
  addition/subtraction; task vectors in weight space compose similarly.

## What this means for the paper

- We cannot claim a new **primitive**. Function-vector algebra already shows
  that activation vectors compose.
- We can claim a new **composition and a new test**: learned operators with
  algebraic constraints, trained by counterfactual next-token prediction, and
  evaluated on **unseen compositions** (rename∘swap, swap∘negate,
  number∘place) that never appear in training.
- The paper must include an **unstructured-operator** baseline and a
  **vector-addition** baseline. If the algebra does not beat them on unseen
  compositions, the constraints are not the lever.
- The introduction must cite function vectors, task arithmetic, in-context
  algebra, representation editing and edit-based models. The novelty paragraph
  should be one sentence.

## What would falsify the claim

- A paper that learns edit operators on LM hidden states with algebraic
  constraints and tests unseen compositions.
- A paper that extends function-vector algebra with involution/commutation and
  counterfactual next-token training.
- A paper on "counterfactual next-token prediction" with learned operators.

Search again before submission. If a closer paper appears, the work becomes a
replication/extension; still useful, but a different claim.

## The honest sentence

We are not inventing the idea of an algebra of representations. We are
inventing a **constrained, learned edit algebra and the composition test that
can show whether it generalizes**. That is a mechanism-level paper candidate,
not a new generation of LLM by itself.
