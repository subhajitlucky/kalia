# Pre-registration — X39: does a learned edit algebra generalize to unseen compositions?

Registered: 2026-11-01, before any GPU time on X39. Related: D65, X38,
`docs/research/2026-11-01-edit-algebra-prior-art.md`.

## Why this exists

X38 failed because its pairwise loss had no structural reason to keep more than
one direction: the stable stream collapsed to rank ~1.5 and became an edit
detector. The lesson is not "counterfactual training is bad." The lesson is
that a pairwise distance can be satisfied by a single knob.

X39 asks a different question:

> Can a model **learn a set of edit operators**, constrain them with algebraic
> rules, and **compose them on edits it has never seen**?

This is a mechanism-level question, not a re-test of counterfactual invariance.
The closest prior art is function-vector algebra: activation vectors already
compose by addition/subtraction. X39's candidate novelty is a **learned,
constraint-abiding algebra of edit operators** plus an **unseen-composition
test**. The paper claim is narrow: the constraints and the composition test are
the contribution, not the idea of an algebra of representations.

## The mechanism

A set of `K = 6` edit operators acts on the final hidden state of a prefix:

```text
E_k(z) = z + U_k V_k z          # low-rank, rank 16, gated residual
```

The operator is applied to the prefix's last hidden state; the base model then
decodes the edited suffix with the transformed state as the last prefix state.
The standard LM path on the corpus is unchanged; the CEA path is auxiliary.

The operators are trained on **individual edits only**:

| operator | example |
|---|---|
| rename | "Maya gave Leo" → "Nora gave Sam" |
| synonym | "a red book" → "a crimson book" |
| swap | "Maya gave Leo" → "Leo gave Maya" |
| negate | "Maya gave Leo" → "Maya did not give Leo" |
| number | "two books" → "three books" |
| place | "on the table" → "under the table" |

The **unseen compositions** are held out:

```text
rename ∘ swap      swap ∘ negate      number ∘ place      rename ∘ negate
```

No composition pair is in training. At evaluation the model applies the two
operators in sequence to the original prefix state and decodes the composed
suffix. A single rank-1 detector cannot do this: it has one knob, not a
composition.

## The algebra losses

- **Involution:** `E_swap(E_swap(z)) ≈ z`; `E_negate(E_negate(z)) ≈ z`.
- **Commutation:** `E_rename(E_swap(z)) ≈ E_swap(E_rename(z))`.
- **Cycle:** `E_num_up(E_num_down(z)) ≈ z`; `E_place_on(E_place_under(z)) ≈ z`.
- **Diversity:** pairwise cosine of the operator deltas < 0.9; effective rank
  of the operator matrix > K/2.

The suffix LM loss is the content objective. The algebra losses are auxiliary.

## Design

- **Base:** `configs/micro-muon.yaml`, 6 layers × 384 × 6, context 512, Muon,
  QK-Norm, no soft-cap, 763 steps, seeds **1338/1339**.
- **Arms:**

| arm | operators | algebra | suffix loss |
|---|---|---|---|
| `control` | no | no | yes (original prefix) |
| `vector` | mean activation differences (function-vector-like) | no | no training |
| `unstructured` | 6 learned operators | no | yes |
| `algebra` | 6 learned operators | involution + commutation + cycle + diversity | yes |

The `vector` arm is the function-vector baseline: extract the mean hidden-state
difference for each edit from the training pairs, add the two mean vectors for
a composition. The `unstructured` arm has the same operator parameters as
`algebra` but no constraints. The key comparison is **algebra vs unstructured**
on unseen compositions.

- **Data:** `edit_pairs.py` generates prefix/suffix pairs from the existing
  template/name pools, with disjoint train/eval templates and names. The
  composition pairs are generated at eval from held-out templates/names.
- **Training:** 763 steps, batch 16 × grad-accum 8, LR 6e-4, Muon; identical
  across arms except the flags. The suffix loss is applied on the CEA batches;
  the standard LM loss is unchanged.

## Metrics

1. **Canonical 100-batch val** (`eval_val.py`, seed 1234).
2. **Seen-edit suffix NLL** on held-out templates/names.
3. **Unseen-composition suffix accuracy** on the four held-out compositions.
4. **Algebra error:** normalised involution and commutation error.
5. **Operator diversity:** effective rank and pairwise cosine.
6. **Swap sensitivity:** the X37 protocol on the 20 stories.
7. **Paraphrase invariance:** the held-out invariant pairs.
8. **Entity retention:** greedy, secondary.
9. **Pair audit:** no train/eval template or name overlap.

## Bars

| ID | Prediction | Threshold |
|---|---|---|
| **E-1** | The CEA path does not cost language quality | `algebra val − control val ≤ +0.005`, both seeds |
| **E-2** | The operators help the seen edits | `algebra seen NLL ≤ control seen NLL`, both seeds |
| **E-3** | The algebra generalizes to unseen compositions | `algebra composition accuracy ≥ unstructured + 0.10`, both seeds |
| **E-4** | The algebra rules hold | involution error < **0.05**, commutation error < **0.05**, both seeds |
| **E-5** | The operators are distinct | effective rank > **3**, pairwise cosine < **0.9**, both seeds |
| **E-6** | The model is more sensitive to a role swap | `algebra swap − control swap ≥ +0.01`, both seeds |
| **E-7** | The model is more invariant to a paraphrase | `algebra paraphrase ≤ control − 0.01`, both seeds |
| **E-8** | The pair audit is clean | no template/name overlap; hashes match |

A win requires **E-1, E-2, E-3, E-4 and E-5**.

## Decision rule (fixed now)

- **E-1, E-2, E-3, E-4, E-5 pass** → the edit algebra is a candidate. Register
  the 58M confirmation (control vs algebra, 2 seeds) when quota allows.
- **E-1 passes, E-3 fails** → the algebra does not generalize to unseen
  compositions; close this implementation or change the operator
  parameterisation, not the eval.
- **E-1 fails** → the CEA gradient costs language quality; no scale.
- **E-4/E-5 fail** → the operators collapse or are not distinct; fix the
  mechanism before reading E-3.
- **`vector` ≈ `algebra` on E-3** → the constraints are not the lever; report
  that additive function vectors already compose and do not scale the algebra.
- **`unstructured` ≈ `algebra` on E-3** → the learned operators do the work but
  the algebra does not; report and reconsider the claim.

## What this does not establish

- Not a 58M result; the 30M micro is a screen.
- Not a new primitive: function-vector algebra (Todd et al., ICLR 2024), task
  arithmetic (Ilharco et al., ICLR 2023), in-context algebra (ICLR 2026),
  representation editing and edit-based models are prior art.
- Not a capability claim; the pairs are synthetic and the context is 512.
- Not "a new generation of LLM"; it is a mechanism-level paper candidate.

## Cost and stopping

4 arms × 2 seeds × 763 steps at 30M ≈ **8–10 GPU-h**, plus the 100-batch val
and the composition probes. The current window has ~23h, but the implementation
is not built; the run waits for the 2026-11-07 window if necessary. The
registration does not expire. If E-3 fails, do not tune the eval; close the
implementation.

## Artifacts (to be archived)

- `docs/results/x39/` — summary, configs, val logs, composition rows, operator
  statistics, run manifests, pair audit.
- `edit_pairs.py` — prefix/suffix pair generator.
- `tools/kalia_x39_edit_algebra_kernel.py`.
- `kaggle/kernel-metadata-x39-edit-algebra.json`.
- `test_edit_pairs.py`, `test_edit_algebra.py`.
