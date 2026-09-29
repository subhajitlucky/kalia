# Continual Learning: What the 2026 Literature Says We Should Build

Compiled 2026-09-28 while X18 runs. Purpose: the user asked what frontier work
should change in the continual-learning design. This records what the literature
actually supports, what it warns against, and the one finding that invalidates an
earlier decision of ours.

## The headline: continual *pretraining* has a known recipe, and we were not using it

**"Simple and Scalable Strategies to Continually Pre-train Large Language Models"**
(Ibrahim et al., arXiv 2403.08763, submitted 2024-03-13) is the paper we should
have been building against. Its abstract states that a simple combination of
learning-rate re-warming, re-decaying, and replay "is sufficient to match the
performance of fully re-training from scratch," demonstrated at **405M and 10B**
across English→English and English→German distribution shifts. The three moves:

1. **Re-warm the learning rate** from its decayed floor back toward the original
   peak, then **re-decay** it over the new budget
2. **Replay** a fraction of the old data
3. They also propose alternatives to the cosine schedule that "help circumvent
   forgetting induced by LR re-warming" and are **"not bound to a fixed token
   budget"** — which is what makes the recipe applicable to a time-boxed daily
   session rather than a fixed-horizon run

**Verified vs unverified.** Every claim above is from the arXiv abstract, checked
2026-09-29. Two things in this section's earlier draft are **not** in the
abstract and are not being relied on: that the paper tests 1%/5%/10%/50% replay
and finds similar loss across the range, and the framing of a "worst option" for
continual adaptation. The replay-ratio evidence would have to come from the full
text, which we have not read. `test_citations.py` records this.

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

> **RETRACTED 2026-09-29 (D49).** This section previously read: *"MIITA
> (arXiv 2607.22556) measured that replay-based continual learning degrades sharply
> as the backbone shrinks. Their words: replay methods 'rely on large parameter
> capacity to absorb and retain knowledge,' and CT0/FOREVER 'drop substantially'
> when the backbone shrinks. Their smallest backbone is 0.6B."*
>
> **MIITA's abstract contains none of that.** Checked 2026-09-29. The paper is
> *"MIITA: Memory-Induced Inference-Time Adaptation for Continual Learning with
> Small Language Models"* (Li et al., submitted 2026-05-20), and it is a
> **memory-based, inference-time** method: it stores correction-direction
> prototypes with semantic anchors, retrieves them at inference using semantic and
> uncertainty cues, and applies them through **gated temporary hidden-state
> adaptation** — explicitly "without backbone updates, prompt extensions, or
> test-time backpropagation."
>
> It is not a replay paper at all. Every claim we attributed to it — the
> parameter-capacity quotation, the CT0/FOREVER degradation, the 0.6B floor, and
> the conclusion that the replay-versus-size slope "goes against us" — was written
> from memory and is **not supported by the paper.** The 0.6B number in particular
> appears to have been inferred from the words "small language models" in the
> title.
>
> **The conclusion we drew is retracted. The decision is not.** See below.

**What still holds, and why the sweep decision survives the retraction.**

The retracted section argued that 10% replay is not a validated number at 58M, and
that the one paper measuring the slope of replay quality against backbone size
says it goes against us. The *evidence* for the second half is gone. The *first*
half is not, because it never depended on MIITA:

- Ibrahim et al. demonstrate their recipe at **405M and 10B**. Neither is within an
  order of magnitude of 58M.
- The specific 1%/5%/10%/50% sweep we relied on is not in the abstract and is now
  unverified.
- So **no ratio we could adopt has a verified source at our scale.** That was the
  argument for sweeping rather than adopting, and the argument is unchanged — it
  now rests on the absence of applicable evidence rather than on a citation that
  turned out to be wrong.

**We are left with a weaker but honest position:** the recipe is well-supported at
405M+, and the ratio at 58M is unmeasured because nobody has measured it. That is
an argument for running the sweep. It is not an argument about which direction the
slope goes, and we no longer claim to know that.

An earlier draft also cited FOREVER's step-size-dependent replay frequency, MIT's
tree-replay study (0.4:1 to stay within 8.2% at 7B), and the 1–2% figures. **None
of those carry a verified citation in our records** and they are dropped here
rather than restated. The ratio must be measured at our scale, not adopted — and
the honest reason is simply that we have no applicable source for it.

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

**Learning, Fast and Slow (FST)** (Tiwari et al., arXiv 2605.12484, submitted
2026-05-12 — *verified*) splits the learner in two: parameters are *slow* weights,
an optimised context is *fast* weights. From the abstract, verbatim: up to **3x
more sample-efficient** than parameter-only RL across reasoning tasks; **up to 70%
less KL divergence** from the base LLM; and critically for us, the reduced drift
**preserves plasticity — FST-trained models adapt more effectively to a subsequent
task**, with parameter-only RL **"stalling"** as domains change on the fly. That
last claim is the whole reason this paper matters to v0.4.

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
   with the CL-0 probe as the metric. The reason to sweep is now the *absence* of
   applicable evidence rather than a claim about slope direction (D49) — the
   recipe is demonstrated at 405M and 10B, and nothing we have verified covers
   58M.
3. **Adopt the checkpoint-level protocol and reference-set diagnostics** from
   **"Continual Learning for Sequential Personalization of Small Language Models:
   A Stability Monitoring Analysis"** (Paula, Kupssinskü & Barros, arXiv
   2606.27634, submitted 2026-06-26 — *verified*). They save a checkpoint after
   each adaptation stage and evaluate on current tasks,
   previously-seen tasks, and a **fixed reference set**, showing that lightweight
   reference-set diagnostics reveal instability "including cases where task-level
   metrics alone hide harmful adaptation." That last clause is the argument: our
   single-distribution setting has no task boundary, so a reference set is the
   only way a harmful update becomes visible. Note their v2 corrected a
   next-token-selection bug and recomputed KL/entropy/margin — a reminder that
   these diagnostics are not free of implementation error.
4. **Then, and only then, embedding-reset plasticity** (active forgetting). This
   is the one experiment that is ours rather than borrowed, and it is cheap
   because the mechanism is 3 lines.
5. **Muon-OGD** if 1–4 land, as a proper pre-registered bet. It is the only
   frontier CL method built around our optimizer.
6. **FST** is a v0.4 question, not v0.3: it changes what the artifact *is*.

## The honest summary

Most of the field is aimed at *fine-tuning* a pretrained model across discrete
task sequences, and the two papers we verified are at 405M and 10B (Ibrahim) or
explicitly about small models (MIITA, but by memory rather than replay). **We are
doing continual pretraining of a 0.058B model on a drifting single
distribution.** That is off the main road in both directions: an order of
magnitude below the scale where the recipe was demonstrated, and a different
problem shape from what the methods optimise.

The transferable core is small and reliable — **re-warm the LR, decay again, keep
replay, measure KL drift.** The interesting core is that we have a specific
architectural property (44.5% of parameters in one embedding matrix) that an
existing plasticity technique fits unusually well, and that no one has tried it
here.
