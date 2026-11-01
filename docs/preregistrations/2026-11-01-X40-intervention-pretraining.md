# Pre-registration — X40: does intervention pretraining teach a small LM a causal world model?

Registered: 2026-11-01, before any GPU time on X40. Related: D65, X36, X37,
X38, X39, and `docs/research/2026-11-01-intervention-pretraining-prior-art.md`.

## Why this exists

The shipped v0.3.1 is the same architecture as v0.1.2. Training and weight
merging improved the score, but no architecture mechanism survived a control.
X36's memory kept names and cost language quality; X37's rule grammar used all
rules uniformly; X38's counterfactual bottleneck collapsed to rank ~1.5.

X40 changes the **objective**, not the block. The question:

> Can an auxiliary **intervention next-state** loss, carried by a small event
> workspace, teach a 58M LM a causal world model that generalizes to **unseen
> compositions** of interventions?

The claim is a mechanism-level paper candidate, not a new architecture.

## Prior art and the narrow claim

The closest work is external: causal world models exposed to LLMs for planning
(ICLR 2025), and LLMs fine-tuned to predict action preconditions/effects
(COLING 2025). Causal Transformers estimate treatment effects; CausaLM learns
counterfactual representations for classification; SPARTAN learns object-centric
world models.

**No exact match** was found for intervention next-state prediction as an
**auxiliary pretraining objective** inside a small LM, with a learned event
workspace, tested on unseen compositions. The claim is provisional:

> To our knowledge, the first auxiliary intervention-pretraining objective with
> a latent event workspace, evaluated for unseen-composition generalization at
> 58M and 134M.

## The mechanism

The base LM is the standard KALIA transformer. Two optional pieces are added:

1. **EventWorkspace** — `K = 16` slots of width 64. At chunk ends, a write gate
   updates the slots from the chunk mean and the current hidden state. A
   cross-attention reads the slots into the current chunk. The read projection is
   zero-initialised, so the model starts as its control.
2. **InterventionHead** — a small MLP that takes the workspace state and an
   intervention embedding and predicts the **next hidden state** of the edited
   continuation (a detached target from the model's own forward on the edited
   suffix). The head is auxiliary; the LM head still predicts tokens.

The training loss is:

```text
L = L_lm(corpus) + L_lm(prefix + intervention + edited suffix)
    + lambda * L_state(InterventionHead(workspace, intervention), target_state)
    + lambda_div * L_diversity(workspace)
```

The **diversity loss** prevents the rank-1 collapse that killed X38: the slots
must stay distinct and the intervention head must use them. `lambda` and
`lambda_div` are fixed before the run.

### Inputs and targets

Each counterfactual example has:

```text
instruction: "Intervention: swap."
prefix:      "Maya gave Leo a red book. Leo put it on the table."
edited:      "Leo gave Maya a red book. Maya put it on the table."
```

The model sees the instruction + prefix and must predict the **edited
continuation**. The target state is the model's own hidden state at the start of
the edited suffix, detached. The event workspace must carry who has the book,
not just the surface words.

Intervention types: **rename, swap, negate, number, place**. The **unseen
compositions** are rename∘swap, swap∘negate, number∘place, rename∘negate.
No composition pair is in training.

## Data

- **Real corpus:** the v2b 2.4B corpus, unchanged.
- **Counterfactual pairs:** `intervention_pairs.py`, a deterministic generator
  with causal chains, disjoint train/eval templates and names, and the held-out
  compositions. The generator writes an audit manifest with counts and hashes.
- **Evaluation:** a held-out synthetic counterfactual set (2-way seen edits,
  4-way unseen compositions), the X37 swap probe on the 20 stories, and the X36
  short-context retention probe.

## Arms

30M micro (6 × 384, context 512, 763 steps, seeds 1338/1339):

| arm | pairs data | workspace | intervention head/loss |
|---|---|---|---|
| control | no | no | no |
| adapter | no | parameter-matched adapter | no |
| pairs | yes | no | no |
| workspace | yes | yes | no |
| **IP** | yes | yes | **yes** |

The `pairs` arm is the data-augmentation baseline: the same instruction + prefix
+ edited suffix, trained as ordinary LM text. If `IP` does not beat `pairs` on
unseen compositions, the workspace/objective is not the lever.

58M confirmation (10 × 512, context 1,024, 2,000 steps, seeds 1401/1402):
`control`, `pairs`, `IP` only.

134M scaling point (12 × 768, context 1,024, 1,500 steps, seeds 1401/1402):
`control`, `pairs`, `IP` only. Conditional on 58M.

## Metrics

1. **Canonical 100-batch val** (`eval_val.py`, seed 1234).
2. **Seen-intervention accuracy** — 2-way choice between the correct edited
   continuation and the factual continuation.
3. **Unseen-composition accuracy** — 4-way choice over the correct composition
   and three counterfactual alternatives.
4. **Entity retention** — the X36 short-context greedy protocol, secondary.
5. **Workspace usage** — slot entropy, effective rank, and a linear probe for
   the intervention type. A dead workspace is a null.
6. **Swap sensitivity** — the X37 protocol.
7. **Pair audit** — no train/eval template or name overlap.
8. **Ruler** — control seed spread.

## Bars

| ID | Prediction | Threshold |
|---|---|---|
| **I-1** | The IP loss does not cost language quality | `IP val − control val ≤ +0.005`, both seeds |
| **I-2** | The model learns seen interventions | `IP seen accuracy − control ≥ +0.10`, both seeds |
| **I-3** | The model generalizes to unseen compositions | `IP unseen accuracy − pairs unseen accuracy ≥ +0.10`, both seeds |
| **I-4** | Retention does not regress | `IP retention − control ≥ −0.005`, reported |
| **I-5** | The workspace is alive and used | slot entropy > **2.0 bits**, effective rank > **4**, intervention probe > chance + 0.10, both seeds |
| **I-6** | The pair audit is clean | no template/name overlap; hashes match |
| **I-7** | The ruler is reported | control seed spread on val/seen/unseen |

A win requires **I-1, I-2, I-3 and I-5**.

## Decision rule (fixed now)

- **30M: I-1, I-2, I-3, I-5 pass** → run the 58M confirmation.
- **58M: I-1 and I-3 pass in the same direction** → run the 134M scaling point.
- **134M: the I-3 gain does not shrink** (still ≥ +0.05 over `pairs`) → the
  objective is a candidate; write the paper.
- **I-1 fails** → the intervention loss costs language quality; no scale.
- **I-3 fails at 30M** → the workspace/objective does not generalize to unseen
  compositions; close this implementation, do not tune the eval.
- **I-5 fails** → the workspace collapsed; fix the mechanism before reading I-3.
- **`pairs` ≈ `IP` on I-3** → the gain is data augmentation, not the workspace;
  report and do not scale the workspace.

## What this does not establish

- Not a 58M or 134M result yet; the 30M micro is a screen.
- Not a new architecture: the base block is unchanged.
- Not a general causal-reasoning claim; the pairs are synthetic and the test is
  a controlled 2-way/4-way continuation choice.
- Not a frontier-adoption claim. If it scales, it is a mechanism a lab can test.
- Not "never invented": causal world models and intervention prediction are
  prior art; the new part is the auxiliary pretraining interface and the
  unseen-composition test.

## Cost and stopping

- 30M: 5 arms × 2 seeds × 763 steps ≈ **8–10 GPU-h**.
- 58M: 3 arms × 2 seeds × 2,000 steps ≈ **25–30 GPU-h** (one to two windows).
- 134M: 3 arms × 2 seeds × 1,500 steps ≈ **40 GPU-h** (two windows).
- The registration does not expire. If the 30M screen fails I-1 or I-3, stop;
  no 58M, no 134M. If the workspace collapses, fix the mechanism, not the eval.

## Artifacts (to be archived)

- `docs/results/x40/` — summary, configs, val logs, seen/unseen rows, workspace
  stats, run manifests, pair audit.
- `intervention_pairs.py` — the causal-chain pair generator.
- `tools/kalia_x40_ip_kernel.py`.
- `kaggle/kernel-metadata-x40-ip.json`.
- `test_intervention_pairs.py`, `test_intervention_pretraining.py`.
