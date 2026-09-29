# Citation verification record

Every arXiv ID in `docs/research/` was checked against arxiv.org on **2026-09-29**.
This file records what was confirmed, what was wrong, and what remains unverified.

Machine-readable form and the test that enforces it: `test_citations.py`.

## What was wrong

**1. An invented title.** The CL document cited arXiv 2403.08763 as *"Rules of
thumb for continual pre-training."* **That title does not exist.** The paper is
**"Simple and Scalable Strategies to Continually Pre-train Large Language Models"**
(Ibrahim, Thérien, Gupta, Richter, Anthony, Lesort, Belilovsky & Rish). The ID was
correct; the title was written from memory.

**2. A load-bearing claim attributed to the wrong paper.** The same document
stated that MIITA (arXiv 2607.22556) *"measured that replay-based continual
learning degrades sharply as the backbone shrinks,"* quoting "rely on large
parameter capacity to absorb and retain knowledge," reporting CT0/FOREVER dropping
"substantially," and giving a 0.6B smallest backbone.

**MIITA's abstract contains none of it.** The paper is *"MIITA: Memory-Induced
Inference-Time Adaptation for Continual Learning with Small Language Models"*
(Li et al.), and it is a **memory-based, inference-time** method: correction-
direction prototypes retrieved at inference and applied through **gated temporary
hidden-state adaptation**, explicitly "without backbone updates, prompt
extensions, or test-time backpropagation." It is not a replay paper. The 0.6B
figure appears to have been inferred from the words "small language models" in
the title.

**3. Two smaller fabrications** in the same document: a claim that the stability
paper's diagnostic "holds under reversed task order," and that KL-to-base is
"free." Neither is in that abstract. A "worst option for continual adaptation"
framing, and the specific 1%/5%/10%/50% replay sweep with "similar final validation
loss across the range," are also absent from the Ibrahim abstract.

Also dropped: unattributed references to FOREVER's step-size-dependent replay
frequency and MIT's tree-replay study (0.4:1 at 7B).

## What survives

The recipe v0.3.0 implements is **real and correctly identified**. Ibrahim et
al.'s abstract states that LR re-warming, LR re-decaying and replay of previous
data "is sufficient to match the performance of fully re-training from scratch,"
at **405M and 10B**, on English→English and English→German shifts — and proposes
schedule alternatives "not bound to a fixed token budget," which is what makes it
apply to a time-boxed session.

FST (arXiv 2605.12484) checks out on every claim we made: 3x sample-efficiency,
up to 70% less KL divergence, preserved plasticity, and parameter-only RL
"stalling" as domains change.

The stability-monitoring paper (arXiv 2606.27634) genuinely proposes
checkpoint-per-stage evaluation against a fixed reference set, and genuinely
reports that this "reveal[s] instability including cases where task-level metrics
alone hide harmful adaptation." Its v2 corrected a next-token-selection bug and
recomputed KL/entropy/margin.

**The decision is unchanged, for a weaker reason.** The replay-ratio sweep was
argued partly on MIITA saying the slope goes against us. That evidence is gone. It
survives on the remaining fact: the recipe is demonstrated at 405M and 10B, nothing
verified covers 58M, so no ratio can be adopted — only measured.

## Verified titles

| ID | Title | Level |
|---|---|---|
| 2403.08763 | Simple and Scalable Strategies to Continually Pre-train Large Language Models | abstract read |
| 2605.12484 | Learning, Fast and Slow: Towards LLMs That Adapt Continually | abstract read |
| 2607.22556 | MIITA: Memory-Induced Inference-Time Adaptation for Continual Learning with Small Language Models | abstract read |
| 2606.27634 | Continual Learning for Sequential Personalization of Small Language Models: A Stability Monitoring Analysis | abstract read |
| 2510.25741 | Scaling Latent Reasoning via Looped Language Models | title only |
| 2510.26692 | Kimi Linear: An Expressive, Efficient Attention Architecture | title only |
| 2604.07822 | Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers | title only |
| 2609.06986 | Continual Learning Mechanisms Compose for Long-Horizon Memorization | title only |
| 2604.04937 | Pramana: Fine-Tuning Large Language Models for Epistemic Reasoning through Navya-Nyaya | title only |
| 2410.09817 | Reverse Modeling in Large Language Models | title only |
| 2511.00341 | Reversal Invariance in Autoregressive Language Models | title only |
| 2602.21545 | MUON+: Towards More Effective Muon via One Additional Normalization Step for LLM Pre-training | title only |
| 2606.22189 | L20-Edu-135M: An Auditable Single-GPU Study of Data-Efficient Small Language Modeling | title only |
| 2505.16932 | The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm | title only |
| 2510.05491 | NorMuon: Making Muon more efficient and scalable | title only |
| 2602.02522 | IMU-1: Sample-Efficient Pre-training of Small Language Models | title only |
| 2606.00371 | How Much Orthogonalization Does Muon Need? | title only |

**"Title only" means exactly that: HTTP 200 plus the `citation_title` metadata
field.** The numeric claims attributed to these thirteen in the 2026-09-23 docs
were not re-read from their abstracts. That distinction is not pedantry — MIITA
resolved and still was not the paper we said it was. A test
(`test_id_verified_entries_are_a_weaker_claim_than_the_rest`) keeps the two
levels from being conflated.

Also unverified and therefore not relied on: the Active Forgetting paper
(NeurIPS 2023) and the orthogonal-projection family (OSFT, OPLoRA, NESS, ROGO,
GORP, Muon-OGD) named in the CL document. They are marked as direction-setting,
not as evidence.

## What would have caught this

A test asserting the verified title appears in any document citing the ID. It
took five attempts to build one that fails on a real error and passes on correct
prose — four of those attempts were bad tests, not bad documents:

1. Blacklisting the one invented string we happened to write. A *different*
   invented title passed.
2. Scanning a window around each ID for any wrong quoted title. Flagged correct
   titles of *other* papers in the same file.
3. Narrowing the window to ±500 characters. Failed on the FST title, written
   600 characters before its citation.
4. Widening to 2000. Failed on MIITA, whose title sits in a retraction block
   1.4k characters from the citation.
5. Matching a 3-word prefix as a fallback for abbreviated titles. Let a genuinely
   wrong title through, because "continual learning with fast" collides with
   another paper's title in the same file.

The working version matches the **full** title, punctuation-insensitive, anywhere
in a document that cites the ID, with abbreviated forms declared explicitly per
entry. A doc citing a bare ID in a table is exempt, since it never claimed a
title there.

The deeper lesson matches the rest of this project: an unverified check is worse
than no check, because it produces a passing result. Two of the five attempts
above would have reported "verified" while verifying nothing.
