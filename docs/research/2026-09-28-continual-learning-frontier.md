# Continual Learning: What the 2026 Literature Says We Should Build

Compiled 2026-09-28 while X18 runs. Purpose: the user asked what frontier work
should change in the continual-learning design. This records what the literature
actually supports, what it warns against, and the one finding that invalidates an
earlier decision of ours.

## The headline: continual *pretraining* has a known recipe, and we were not using it

**"Rules of thumb for continual pre-training"** (arXiv 2403.08763) is the paper we
should have been building against. It establishes, at 405M and 10B scale, that a
simple combination matches full retraining from scratch:

1. **Re-warm the learning rate** from its decayed floor back toward the original
   peak, then **re-decay** it over the new budget
2. **Replay** a fraction of the old data
3. They test 1%, 5%, 10% and 50% replay and find **similar final validation loss
   across the range** — so the ratio is not the delicate knob the folklore
   suggests

Two consequences for us:

- **Audit finding C is now answered, and the answer is uncomfortable.** Our config
  has `min_lr_ratio: 0.1` and every full run in this project's history has ended
  with the LR still 3.5x above zero. The rules of thumb say the *worst* option for
  continual adaptation is sitting at a small constant LR: *"re-warming and
  re-decaying the learning rate increases both adaptation and forgetting"* — i.e.
  not re-warming costs adaptation. Our daily loop currently does not re-warm at
  all, because there is no daily loop yet. That is the first thing to build.
- **Their stated knob:** *"Decreasing the schedule's maximum learning rate can help
  reduce forgetting, whereas increasing it can improve adaptation."* Directly
  tunable for a stability/plasticity dial in our daily cycle.

**"Beyond Cosine Decay"** (L4LA 2026, MLR v330) adds the piece our cadence
specifically needs: an **infinite LR schedule** beats repeated cosine annealing
for continual pre-training *and is not bound to a fixed token budget*. A daily
update has no natural token budget — we run until the session's time limit — so a
schedule that assumes a fixed horizon is the wrong shape. The two papers agree
infinite schedules are promising and specifically flag re-warm as the thing that
causes forgetting.

## The finding that invalidates our CL-1 design

**MIITA (arXiv 2607.22556) measured that replay-based continual learning degrades
sharply as the backbone shrinks.** Their words: replay methods "rely on large
parameter capacity to absorb and retain knowledge," and CT0/FOREVER "drop
substantially" when the backbone shrinks. Their smallest backbone is **0.6B**.
KALIA is **0.058B** — a tenth of that.

**Our CL-1 default of 10% replay is therefore not the validated number.** The
1–2% figures come from 0.6B–13B backbones; MIT's tree-replay study found 0.4:1
needed to stay within 8.2% at 7B. **Nobody has published a replay ratio validated
at 58M.** We are extrapolating downward by an order of magnitude, and the one
paper that measured the direction of that slope says it goes against us.

This does not kill replay — FOREVER's mechanism (replay more frequently when
parameter updates are large, since identical step counts produce varying amounts
of forgetting) is step-size-dependent, and our update magnitudes are far larger
relative to the model. But the ratio must be *measured at our scale*, not adopted.
It also argues the first experiment should be a **replay-ratio sweep**, not a
single 10% arm.

## The idea that fits KALIA's specific shape

**Active forgetting** (NeurIPS 2023, "Improving Language Plasticity via
Pretraining with Active Forgetting") resets the token embedding layer every K
updates during pretraining. The result is counterintuitive and useful: models
trained this way are *more plastic*, adapt to new languages far faster, and reach
92% of full adaptation performance in 5% of the steps.

This is unusually well-matched to us, and the reason is a measurement we already
have: **the embedding matrix is 44.5% of our parameters** — 25.7M of 57.9M, eight
times an entire transformer layer. A 58M model with a 44.5% embedding share is
unusual, and this technique converts that structure into plasticity. Every other
method in this document wants to *protect* weights; this one deliberately throws
part of the model away on a schedule to keep it able to learn.

Nobody has tried it on a from-scratch Muon model at this scale, in continual
*pretraining* rather than adaptation. That is a cheap, genuinely open experiment
that uses a property of our architecture rather than borrowing someone else's
trick.

## What the orthogonal-projection family offers, and why we are not rushing it

Five papers converge on the same move: decompose weights, protect the
high-singular-value subspace, learn only in the orthogonal remainder.

| Method | Idea | Our read |
|---|---|---|
| OSFT (ICLR 2026) | Weight-SVD, project updates orthogonally, constant memory | Best-scaled of the family (7B). Still a *fine-tuning* method. |
| OPLoRA | Double-sided projection on LoRA updates | LoRA-specific; we ruled LoRA out (adds params, loses to full FT with adequate replay) |
| NESS | Learn *in* the null space, parameterised directly | Elegant, but image classification |
| ROGO | Relax orthogonality into a "relaxing space" to allow forward transfer | Fixed network, no buffers — closest to our constraints |
| GORP | Joint full + low-rank in one subspace | — |
| **Muon-OGD** (2026) | **Spectral-norm-aware projection using Muon's operator geometry** | **Most interesting: it is the only one built for Muon** |

**Muon-OGD is the one to watch.** Projection methods are formulated in Euclidean /
Frobenius geometry, and its argument is that Muon's orthogonalized updates carry a
*spectral-norm* interpretation, so Frobenius projection is the wrong geometry for
matrix-valued weights. We train with Muon. Nobody has combined this with
from-scratch pretraining rather than post-hoc fine-tuning, or tested it at 58M.

The family's shared weakness is that all of them are evaluated on *task
benchmarks* with multiple discrete tasks, where forgetting is easy to define. Our
setting is a single distribution that drifts — harder to measure and not what
these papers optimised for.

## The architecture alternative worth naming

**Learning, Fast and Slow (FST)** (arXiv 2605.12484) splits the learner in two:
parameters are *slow* weights, an optimised prompt/context is *fast* weights.
Reported: up to 3x more sample-efficient than parameter-only RL, **70% less KL
drift from the base model**, less forgetting, and — the part that matters most
for continual learning — **better plasticity on the *next* task** than
parameter-only training. Parameter-only RL "stalls" as domains change.

For a 58M model this is genuinely attractive: the fast learner needs no
parameters at all, so a small model gets an adaptation mechanism that larger
models need adapters for. Cost: it changes the inference signature (context
becomes part of the model state), which is a bigger claim to publish.

## Recommendations, in order

1. **Re-warm + re-decay for every daily update.** Canonical, cheap, and we are
   currently not doing it. Our existing cosine stops at `min_lr_ratio 0.1`; a
   daily cycle should warm back toward peak and decay again. Implement with an
   infinite-decay schedule so it does not assume a fixed token budget.
2. **Measure the replay ratio instead of adopting one.** Sweep 10% / 40% at 58M
   with the CL-0 probe as the metric. The literature gives us a reason to think
   10% may be too low, which is exactly why it needs a number.
3. **Add KL-to-base to the CL-0 ledger** (arXiv 2606.27634). It is a free
   early-warning signal, holds under reversed task order, and is precisely the
   metric FST and Muon-OGD optimise against — so it is the common measure for
   comparing all of these.
4. **Then, and only then, embedding-reset plasticity** (active forgetting). This
   is the one experiment that is ours rather than borrowed, and it is cheap
   because the mechanism is 3 lines.
5. **Muon-OGD** if 1–4 land, as a proper pre-registered bet. It is the only
   frontier CL method built around our optimizer.
6. **FST** is a v0.4 question, not v0.3: it changes what the artifact *is*.

## The honest summary

Most of the field is aimed at *fine-tuning* a pretrained model across discrete
task sequences, and at backbones of 0.6B–13B. **We are doing continual
pretraining of a 0.058B model on a drifting single distribution.** That is off the
main road in both directions: too small for the methods to have been validated at,
and a different problem shape from what they optimise.

The transferable core is small and reliable — **re-warm the LR, decay again, keep
replay, measure KL drift.** The interesting core is that we have a specific
architectural property (44.5% of parameters in one embedding matrix) that an
existing plasticity technique fits unusually well, and that no one has tried it
here.
