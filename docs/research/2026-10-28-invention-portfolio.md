# Invention portfolio — mechanisms that could make a 58M model interesting

Status: active research record
Date: 2026-10-28
Context: the goal is **not** a frontier model. It is a novel mechanism that
gives a 58M model a surprising, falsifiable, publishable property. This is a
candidate portfolio, not a set of results. Prior-art checks are pending for
every entry; no novelty is claimed yet.

## The filter

A candidate is worth GPU time only if:

1. It changes a **mechanism**, not a hyperparameter.
2. It has a **behavioral probe** that can fail, not only val loss.
3. It can be **parameter-matched** against a control.
4. It is implementable at 58M in less than a day of engineering.
5. Its paper claim is not "we scaled a known idea."

Scores below are 0–3 for novelty, testability at 58M, expected effect size,
implementation cost (3 = cheap), and paper potential.

## Portfolio

| # | mechanism | one-line idea | N | T | E | C | P | total | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | **Counterfactual Residual Bottleneck** (CRB) | split the residual stream into a stable causal subspace and a volatile surface subspace; train the stable one to be invariant to meaning-preserving edits and sensitive to meaning-changing ones | 2 | 3 | 2 | 2 | 3 | **12** | **first bet** |
| 2 | **Relational Algebra Attention** (RAA) | a small set of relation matrices shared across layers, composed by a learned algebra, so relations generalize to unseen compositions | 3 | 2 | 2 | 1 | 3 | **11** | high-risk second |
| 3 | **Predictive-Coding Depth** (PCD) | each block predicts the next block's state; the prediction error is fed back as a modulation signal | 2 | 2 | 2 | 2 | 2 | **10** | practical third |
| 4 | Latent Plan Bottleneck (LPB) | predict a K-token latent plan before each token | 1 | 3 | 2 | 2 | 2 | 10 | multi-token-prediction adjacent |
| 5 | Surprise-Gated Global Memory (SGGM) | sparse global memory written only on high-surprise tokens | 1 | 3 | 2 | 2 | 2 | 10 | Titans-adjacent; X36 warns of val cost |
| 6 | Self-Correcting Decoder (SCD) | draft, critic, revise in a second pass | 1 | 3 | 2 | 2 | 2 | 10 | inference cost; not an architecture change |
| 7 | Entropy-Budgeted Depth (EBD) | per-token depth with a fixed compute budget and a specialization loss | 1 | 3 | 2 | 2 | 2 | 10 | MoR/MoD-adjacent |
| 8 | Causal Chunking Hierarchy (CCH) | learn chunk boundaries, then hierarchical attention within/between chunks | 1 | 2 | 3 | 1 | 2 | 9 | complex at 512 context |
| 9 | Type-Constrained Composition (TCC) | typed functions plus a differentiable type checker | 3 | 1 | 2 | 0 | 3 | 9 | too complex for this window |
| 10 | Associative Residual Slots (ARS) | a few slots updated by attention and read by every layer | 1 | 3 | 2 | 3 | 1 | 10 | memory-token adjacent |

## First bet — Counterfactual Residual Bottleneck (CRB)

### Why this one

X36 proved the repo can make a memory that **keeps names** — and showed the
price: +0.076 nats of language quality because the workspace stole capacity
from the LM path. SUTRA proved the repo can make a **live codebook** — and
showed the price: all 64 rules used uniformly, because nothing forced them to
specialize.

CRB targets both failures at once, without a separate memory or a discrete
router. It gives the model a structural reason to separate **what the sentence
means** from **how it is phrased**. The repo already has the probes: the name
swap, entity retention, and CL-0.

### Mechanism

At each block, split the residual stream into two learned subspaces:

```text
x  =  s + v
s  =  P_s x        # stable, low-rank (e.g. 64 of 384 dims)
v  =  P_v x        # volatile, full-rank
```

The block reads `x` as usual. The output is written back with separate gates
for `s` and `v`. A rank/sparsity penalty keeps `s` small, so it cannot simply
duplicate the whole stream.

The training signal has two halves:

- **Invariance:** for a meaning-preserving edit (consistent entity rename,
  synonym substitution), the final stable state must not change:
  `||s(x) − s(x')||²` is minimized.
- **Sensitivity:** for a meaning-changing edit (subject/object swap, number
  change, negation), the stable state must move past a margin:
  `max(0, m − ||s(x) − s(x'')||²)` is minimized.

The volatile subspace has no consistency loss. It carries the surface form.

### Why it might produce a real result

The model cannot solve the counterfactual objective by copying names, because
the stable state is explicitly rewarded for ignoring a consistent rename. It
cannot ignore a subject/object swap, because the stable state is rewarded for
moving. That is a structural pressure toward the exact behavior the swap test
measures — and it is a pressure SUTRA never had.

It is also parameter-cheap: the split is linear, and the control is the same
model without the split and the loss. The comparison is parameter-matched.

### Test at 58M

Stage 1 (30M screen, 4 arms × 2 seeds, 763 steps, ~8 GPU-h):

| arm | split | counterfactual loss |
|---|---|---|
| control | no | no |
| split-only | yes | no |
| loss-only | no | yes |
| CRB | yes | yes |

Stage 2 (58M confirmation, control vs CRB × 2 seeds, 2,000 steps, ~16–20 GPU-h)
only if Stage 1 clears the bars.

Bars, fixed before the run:

- **C-1** val ≤ control + 0.005 (no language-modelling tax).
- **C-2** swap sensitivity ≥ control + 0.005 (more sensitive to the causal edit).
- **C-3** paraphrase invariance: mean stable-state distance under a
  meaning-preserving edit is lower than control.
- **C-4** entity retention ≥ control − 0.005.
- **C-5** stable subspace is alive: effective rank > 8, stable share < 0.5.
  A collapsed split is a null, not a win.

Ablations: split-only, loss-only, random split. The full CRB must beat all
three to attribute the effect to the interaction.

### Paper claim, if it works

> A counterfactual residual bottleneck improves systematic sensitivity in a
> 58M language model without a language-modelling tax.

The paper would include the 30M screen, the 58M confirmation, the four-arm
ablation, the existing entity/swap/CL-0 probes, and a 134M replication if the
budget allows. That is a workshop paper or an arXiv preprint, not a frontier
claim. The repo's pre-registration and negative results are the credibility.

### Risks

- The counterfactual pairs may be noisy or gameable; the pair generator must
  be audited before the run.
- The stable subspace may collapse; C-5 catches it.
- The invariance loss may cost val; C-1 catches it.
- Prior art: causal representation learning, disentangled representations,
  counterfactual data augmentation, invariant risk minimization. The
  architectural split inside an LM is the candidate novelty; a prior-art check
  must happen before any paper claim.

## Second bet — Relational Algebra Attention (RAA)

A small set of relation matrices `R_r` is shared across layers. Each attention
head computes `qᵀ A k` with `A = Σ_r w_r(x) R_r`, where `w_r` is content
dependent. A small composition network maps relation embeddings across layers,
and an auxiliary consistency loss trains composition on a synthetic relational
task. The hypothesis: a shared algebra learns relations that generalize to
unseen compositions, which a dense MLP cannot.

Test: synthetic kinship/ordering composition + LM val; bars on composition
accuracy, val, and relation diversity. Cost is higher and the implementation is
riskier than CRB. This is the most "next-gen" candidate and the most likely to
be a prior-art rediscovery; check before building.

## Third bet — Predictive-Coding Depth (PCD)

Each block predicts the next block's hidden state; the prediction error is fed
back as a modulation signal, with an auxiliary loss on the error. The
hypothesis: depth-wise predictive coding improves sample efficiency and makes
deeper layers useful at 58M. The probe is depth-wise error versus token
difficulty. This is the most practical candidate, but auxiliary losses often do
not survive a val-loss control at this scale.

## What would make it a paper

1. **A clean mechanism** that can be described in one paragraph and implemented
   in one block.
2. **A behavioral win** that is not just val loss: swap sensitivity, retention,
   composition, or depth extrapolation.
3. **A parameter-matched control** and at least two seeds.
4. **Ablations** that isolate the mechanism from the extra parameters/compute.
5. **A scaling check** at 58M and, if possible, 134M.
6. **A limitations section** that says what the 58M result does not prove.

The repo already has 1–4. The missing pieces are 5 and the paper itself.

## Recommendation

Run CRB first. It is the cheapest candidate with a direct behavioral probe, it
targets two measured failures (X36's val cost, SUTRA's uniform router), and it
has a paper claim that is modest enough to be credible. If CRB fails C-1 or
C-2, move to RAA; if RAA is too costly, PCD.

Do not run another 30M rule-grammar or generic memory micro before CRB. The
portfolio exists to stop us from tuning the last idea instead of testing a new
one.
