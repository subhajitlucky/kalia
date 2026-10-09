# 2026-10-08 — Frontier re-check: what is still worth trying after the seed-spread finding

**Status:** research note, written after Step 1 confirmed the 58M micro-ruler is
unusable (spread 0.2892/0.2912, D48 rules applied). The question this note
answers: given that micro-scale claims are out, which 2026 techniques are worth
anything to KALIA — and which change our current plan?

**Verification levels are marked per claim**, following `CITATIONS.md`:
`[abstract]` = read in full on arXiv today; `[full text]` = read in the paper's
published HTML, not the abstract; `[venue]` = cited by venue and year, no arXiv
ID claimed.

---

## 1. The finding that changes the plan: retrieval at small scale

**Can Small Language Models Use What They Retrieve? An Empirical Study of
Retrieval Utilization Across Model Scale** — arXiv 2603.11513, Pandey et al.,
2026-03-12 `[abstract]`

Five model sizes (360M–8B), three families, four conditions (none / BM25 / dense
E5 / oracle-guaranteed). Results `[abstract]`:

- ≤7B models fail to extract the correct answer **85–100% of the time on
  questions they cannot answer alone, even with oracle retrieval**.
- Adding any retrieval context **destroys 42–100% of answers the model previously
  knew** — a distraction effect driven by the *presence* of context, not its
  quality.
- The dominant failure mode is irrelevant generation: the model ignores the
  provided context entirely.
- Conclusion: below 7B, the bottleneck is **utilization, not retrieval quality**;
  standard-prompting RAG is a **net negative**.

KALIA is 58M, ~100x below the smallest model tested. The "lookup helper" as
originally pitched — prepend retrieved passages, let the model read them — is
therefore **expected to fail, and possibly to hurt**. The honest consequence:
measure it (companion harness `retrieval_probe.py`), and if a lookup feature
ships, ship it as a *tool* (passages surfaced to the user; the model never has to
interpret them) rather than as model capability.

**Measured 2026-10-08** (`docs/results/retrieval-probe.json`, findings in the
results README): oracle copy reaches 0.828 [0.734, 0.922] against 0.125 chance —
the utilization failure does *not* transfer to verbatim copying at 58M — but
BM25 top-3 collapses to 0.297 despite 100% retrieval; irrelevant context costs
+0.1222 nats [+0.077, +0.171]; own-log-prob best-of-N underperforms greedy
by 0.281 [−0.406, −0.172] against a 0.953 coverage ceiling; scoring only the
decision position removes the harm but exactly ties greedy (+0.000), so at 58M
the selector, not the sampling, is the missing piece. The measurement rewrote
the conclusion in both directions: the model can read, and our selector is
worse than doing nothing.

Supporting work, cited by venue only `[venue]`: MiniRAG (ACL 2026) found SLM-RAG
needs graph/entity indexing to work at all; CALI (Zenodo, 2026) found dense
embeddings add nothing over lexical + a cross-encoder reranker — and a
cross-encoder is a borrowed model, which D2 rules out.

## 2. Test-time selection: our log-prob scorer is the weak baseline

- **SCATR: Simple Calibrated Test-Time Ranking** — arXiv 2604.16535, Shyamal
  et al., 2026-04-16 `[abstract]`: token log-probability confidence heuristics
  "often perform substantially worse" than a lightweight scorer trained on the
  base model's **own hidden representations**; SCATR beats confidence baselines
  by up to **9%** with **8000x fewer trainable parameters than LoRA** `[abstract]`.
  Fully compatible with from-scratch: it is a small head on KALIA's own features
  plus a few hundred labeled calibration items.
- **Parallel Test-Time Scaling with Multi-Sequence Verifiers** — arXiv
  2603.03417, Kim et al. `[abstract]`: score candidates *jointly* instead of in
  isolation; +6% best-of-64 relative; early stopping reaches baseline accuracy at
  **less than half the latency** `[abstract]`.
- **Best-of-Majority** (ICLR 2026) `[venue]`: proves plain majority voting and
  plain BoN *degrade* as the sampling budget grows; frequency-filter-then-rank is
  minimax optimal.
- Best-of-N TR-2026-02 (Zenodo) `[venue]`: on Qwen2.5-0.5B/GSM8K, selection
  contributes +20.2 points (45.3 → 66.5) — but only where answers are
  **verifiable**. Open-ended generation has no such oracle; selection quality
  must be measured on verifiable probes, never asserted for stories.

Consequence for `best_of_n.py`: it is the *weak-selector baseline* the 2026
literature measures against. It should only ever claim gains on probes with
objective answers, and the upgrade path is a SCATR-style head, not better
prompting.

## 3. Training and data: two genuine levers

- **Squeezing More from Limited Data with Recursive Transformers** — arXiv
  2608.26973, Gülbahar, Edman & Fraser, 2026-08-27 `[abstract]`: standard
  Transformers "scale down poorly to this setting, because **embeddings consume
  a large fraction of the parameter budget** and per-token computation is tied to
  representational capacity"; **recursive weight sharing + factorized
  embeddings** outperform standard Transformers at 10M and 100M word budgets and
  are competitive with BabyLM 2025 winners `[abstract]`. This is a structural
  description of KALIA's own defect: 44.4% of its parameters are the embedding,
  8.2x a whole transformer layer. The strongest architecture candidate for a v0.4
  **full rebuild** — and testable only at full scale, since micro-differences at
  58M are exactly what Step 1 showed cannot be read.
- **It's All Training: A Fully Synthetic Single-Stage Recipe for LLMs** — arXiv
  2609.37891, Langlais et al., NeurIPS 2026 `[abstract]`: SYNTH, a public
  synthetic corpus from 58,698 Wikipedia articles, trains a 56M model (Monad)
  and 0.3–0.6B models competitively with similarly-sized open baselines "despite
  10–140x fewer training tokens" `[abstract]`. Full text reports Monad sees on
  the order of 180B tokens and posts strong multiple-choice numbers `[full text]`.
  **Lineage caveat:** SYNTH is back-translated with a 12B external model. Using
  it would import borrowed generation, which conflicts with this project's
  identity (D2/D21 spirit). The clean variant is self-generated data (SSR-style,
  see §4), which needs its own registration before it is anything.
- **L20-Edu-135M** — arXiv 2606.22189 `[title-verified only]`: a single-L20-GPU
  134.5M training study and the closest published counterpart to KALIA's budget.
  Cited here for orientation only; its numeric claims were not re-read, so none
  are relied on. Its full text is worth consulting for its **data gates**
  (cross-source near-dedup, segment dedup, benchmark-contamination removal)
  before the next corpus build `[full text, search excerpt]`. **Run
  2026-10-08** on KALIA's own corpus: 2,903 exact 13-gram windows across the
  five scored benchmarks (~1.2 per million tokens; WinoGrande zero) — small,
  real, and recorded in `docs/results/corpus-gates.json` with the false-positive
  story that found the gate's own degeneracy bug.

## 4. Continual learning: if the line ever resumes

- **Beyond Static Models: An Evolving Framework for Continual Learning in Large
  Language Models across Training Stages** — arXiv 2603.12658, Chen et al.,
  2026-03-13 `[abstract]`: current survey of rehearsal / regularization /
  architecture CL for LLMs across pretraining, fine-tuning and alignment.
  (Note: an earlier draft circulated under a different title; the arXiv title is
  the one recorded in `CITATIONS.md`.)
- **FOREVER** (ACL 2026) `[venue]`: schedule replay by **parameter-update
  magnitude** — model time, not step counts — with Ebbinghaus-style intervals;
  consistent gains at 0.6B–13B.
- **Revisiting Replay and Gradient Alignment for Continual Pre-Training**
  (L4LA 2026, PMLR v330) `[venue]`: replay + gradient alignment (MER-style)
  remain stable at 100B-token scale; small replay rates are worth more than
  model size, but scaling the model is more compute-efficient than heavy replay.

The v0.3.0 sequence already stopped by its own pre-registered rule, so none of
this is scheduled. If continual updates are ever revisited, these supersede the
schedule choices that were never run.

## 5. Ranked implications for KALIA

1. **Measure retrieval utilization and best-of-N before building either into the
   product** — `retrieval_probe.py` implements the 2603.11513 protocol adapted
   to a 58M base LM (none/BM25/oracle; fact-utilization and distraction arms;
   best-of-N against verifiable cloze). Free, CPU, and a real result either way.
2. **Corpus data gates** (dedup, segment dedup, benchmark-contamination scan)
   before any v0.4 corpus build — pure hygiene, CPU only.
3. **One full-scale v0.4 bet, data-first**: multi-epoch (<=4 is near-fresh per
   this project's own research) + code share + staged curriculum + WSD/infinite
   schedule. Full-scale only, paired seeds only.
4. **The architecture exception, if taken, is RecursiveGPT-shaped**: factorized
   embeddings + recursive depth, targeting the 44.4% embedding share. One paired
   full-scale experiment, D48 rules (>=2 seeds, spread reported first).
5. **Do not**: RLVR at this scale, distillation (D21), borrowed encoders or
   rerankers (D2), Naive RAG prompting without the utilization measurement.

## Sources

Verified today (abstract read, titles and claims recorded in
`test_citations.py` and `CITATIONS.md`): 2603.11513, 2608.26973, 2609.37891,
2604.16535, 2603.03417, 2603.12658. Venue-only citations carry no arXiv ID and
are not relied on for numeric claims.
