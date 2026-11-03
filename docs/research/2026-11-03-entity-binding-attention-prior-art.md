# Entity-Binding Attention Supervision — prior-art check

Date: 2026-11-03. Status: proposal, not registered, no GPU spent.
Question: is an **entity-level counterfactual attention-alignment loss**,
derived from symbolic intervention pairs and added to autoregressive LM
pretraining, something that has not been released?

Method: focused web search for attention supervision with counterfactuals,
entity binding, attention knockout, circuit-guided steering, next-latent
prediction, gated adapters, and compositional curricula. Not a systematic
review.

## Verdict — corrected after the second sweep

**No exact match in the narrow LM/entity-binding application, but the broad
mechanism is not new.** Counterfactual/causal attention supervision already
exists in adjacent domains:

- **offline RL** — UCF, *Unifying Counterfactual Data Augmentation and
  Architectural Inductive Biases in Offline RL* (ICLR 2026 submission):
  supervises attention weights to focus on causal parents using counterfactual
  data;
- **graph attention** — CSA, *Causal-Based Supervision of Attention in Graph
  Neural Networks* (2023): counterfactual attention regularization;
- **text classification** — *Attention Supervision with Counterfactuals for
  Text Classification* (EMNLP 2020).

The remaining gap is the **application**: entity binding in a decoder-only LM,
targets derived from **symbolic intervention pairs**, held-out composition
evaluation at 30M/58M. That is an **adaptation/extension**, not a new
primitive.

**Do not claim “never experimented.”** The defensible claim is:

> We apply counterfactual attention supervision to entity binding in a small
> decoder-only LM, with targets derived from symbolic intervention pairs, and
> evaluate held-out compositions.

The benchmark and the small-scale result are the contribution; the mechanism
is borrowed.

## Closest prior art

| work | what it does | why it is not this |
|---|---|---|
| **Attention Supervision with Counterfactuals for Text Classification** (EMNLP 2020) | human attention supervision on counterfactual text for classification | classification, human annotations, no LM pretraining, no entity binding or held-out compositions |
| **UCF: Unifying Counterfactual Data Augmentation and Architectural Inductive Biases in Offline RL** (ICLR 2026 submission) | causal attention supervision: attention weights are pushed to the causal parents using counterfactual data | offline RL, not a decoder-only LM; no entity binding or held-out composition |
| **CSA: Causal-Based Supervision of Attention in Graph Neural Networks** (2023) | counterfactual attention regularization for graph attention | graph classification, not LM entity binding |
| **Counterfactual Simulation Training** (2026) / **Counterfactual Sensitivity Regularization** (2025) | counterfactual training for faithful CoT/reasoning | reasoning traces, not entity attention; no held-out composition |
| **LUKE / Mention Memory / entity-tracking attention** (2019–2026) | entity-aware attention and mention-memory architectures | architectures, not a pair-derived attention loss |
| **Mixing Mechanisms: How LMs Retrieve Bound Entities In-Context** (ICLR 2026) | attention-knockout analysis of lexical/positional/reflexive binding | analysis only; no training loss |
| **A retrieval conditioned rebinding circuit for dynamic entity tracking** (EMNLP 2026) | identifies and patches a rebinding circuit | circuit analysis; no pretraining objective |
| **Next-Latent Prediction Transformers** (2025) | auxiliary latent next-state prediction | latent state, not entity attention; X40 tested a related next-state objective and failed |
| **Next Concept Prediction** (2026) | continuous-concept auxiliary prediction | concept-level, not entity-attention binding; no intervention pairs |
| **Uncertainty-Gated Adapter** (2025, visual AR) / gated LoRA (2025) | conditional adapters | output/capacity mechanisms, not attention binding; close prior art for the adapter fallback |
| **Compositional curricula** (2025–2026) | train on skill/subtask curricula, test held-out compositions | data-side; no entity-attention target; X40's data-only arm did not compose |

## The mechanism

**Entity-Binding Attention Supervision (EBAS).**

- Data: the X40 intervention pairs (`instruction + factual prefix` → edited
  suffix). The generator knows the entity mapping (A→A2, B→B2, C→C2) and the
  operation.
- Target: for a suffix token that is an entity name, the model's attention from
  that token to the **corresponding entity mention** in the prompt should be
  high:
  - `rename`: the new name in the instruction and/or the old name in the
    prefix (the entity being renamed);
  - `swap`: the prefix mention of the role-swapped entity;
  - `negate`/`number`/`place`: no entity target; exclude from the attention
    loss (or target the operation token separately).
- Loss: KL between the model's attention distribution (one chosen layer,
  averaged heads) at the suffix entity token and a target distribution over
  prompt token positions. No new parameters; a training-only hook. Weight
  fixed before the run (e.g., 0.05).
- The auxiliary term is applied only to the pair batch; the corpus LM loss is
  unchanged. The `pairs` arm is the data-only baseline.

## Why it might work

X40's failure was not a dead workspace; it was that the workspace/state loss did
not make the model **bind** the edited entity to the suffix. The gold-edit test
shows the base cannot continue the hard cells even when the edited world is
given. EBAS supplies the missing signal directly: attend to the entity that the
instruction moved, not the factual one. It adds no module, so there is no new
capacity to pay for; only the auxiliary gradient can cost val.

## The test

30M screen: `pairs` vs `pairs + EBAS` vs `pairs + shuffled-target EBAS`
(control for the extra gradient), seeds 1338/1339, 763 steps.

Bars:

- **E-1** `EBAS val − pairs val ≤ +0.005` (no language tax).
- **E-2** S-P `rename+swap` accuracy `EBAS − pairs ≥ +0.10`, both seeds.
- **E-3** S-P `swap+negate`/`rename+negate` not regress by >0.03.
- **E-4** S-S/U-S not regress by >0.03.
- **E-5** attention mass on the target mentions increases vs pairs
  (mechanism check).
- **E-6** shuffled-target control does not show the same gain.

A pass at 30M (`E-1`, `E-2`, `E-5`) justifies a 58M confirmation; a fail closes
the idea. No eval tuning.

**Cost:** 3 arms × 2 seeds × 763 steps ≈ 6–8 GPU-hours at 30M, plus CPU eval.
No 58M until it passes.

## Honest risk

- **The broad mechanism is prior art in adjacent domains.** EBAS is an
  adaptation to decoder-only LM entity binding, not a new primitive. A reviewer
  can reasonably call it “causal attention supervision applied to LMs.”
- This targets the **composition benchmark**, not val. If “better than v0.3.1”
  means only val/bpB, the only positive signal is the **output adapter**
  (X40's parameter-matched control improved 30M val by 0.26 nats); a 58M
  output-adapter fine-tune is the safe 6–8h experiment, but it is a capacity
  tweak, not a new mechanism.
- The attention target is derived from a synthetic generator; it may teach a
  shortcut (attend to the name token) without the role binding.
- Attention supervision can hurt calibration/val; E-1 is the guard.
- The prior is low. Most auxiliary objectives fail at this scale, and X40's
  next-state objective did not transfer. EBAS is a different signal, not a
  guaranteed win.

## The honest sentence

We are not inventing attention supervision, counterfactual supervision, or
entity binding. We are testing whether **applying** counterfactual
attention supervision to entity binding in a small decoder-only LM, with
targets derived from symbolic intervention pairs, teaches held-out composition
without a language tax. The mechanism is borrowed; the benchmark and the
small-scale result are the new part.

## Second sweep note

The first memo said “no exact match.” The second sweep found counterfactual
attention supervision in offline RL (UCF, ICLR 2026 submission), graph
attention (CSA, 2023) and classification (EMNLP 2020), plus counterfactual
training for reasoning (CST/CSR, 2025–2026) and entity-aware attention
architectures (LUKE, Mention Memory, entity-tracking). The correct conclusion
is **no exact match for this application, but the mechanism is not new**. The
“never” claim was withdrawn before any GPU time.
