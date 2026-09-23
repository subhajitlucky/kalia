# The Mahabharata Audit — Character Networks, Entity Memory, and Prior Art

Status: research record; one deliverable built
Date: 2026-09-23
Sources: Gultepe & Mathangi, Heritage 2023 (character social network analysis);
ACL 2016 computational analysis of the Mahabharata (co-occurrence networks,
sentiment/emotion arcs across 18 parvas); ICCC 2026 (knowledge-graph-guided
long-form story generation and entity-retention metrics); Papalampidi & Lapata
2022 (dynamic entity memory + auxiliary entity loss for narrative generation).

## The text (facts)

- ~100,000 verses / ~1.8M words in 18 books (parvas): the longest epic poem.
- Structural features: nested narration (Vyasa → Vaisampayana → Ugrashravas →
  …), multi-perspective retellings of the same events, and a large cast whose
  relationships have been studied quantitatively.
- Computational studies: character co-occurrence networks recover the
  protagonist/antagonist communities; predicted character-to-character links at
  F ≈ 0.81; trained network metrics show the narrative's
  stability–instability–stability arc.

## Deliverable built from this audit: entity-consistency evaluation

`eval_entities.py` (with tests): character extraction by a deterministic
capitalization heuristic, then

- **retention**: fraction of prompt characters that persist in the continuation
- **max-span ratio**: how long a character remains referenced across the text
  (metric family from Papalampidi & Lapata 2022)
- **co-occurrence edges**: the character-network view (Gultepe & Mathangi 2023)

This is the missing evaluation layer for the KALIA-EM+TTT invention target: we
can now measure whether entity memory and test-time fast weights actually
improve character consistency.

## Prior-art correction (important)

**Papalampidi & Lapata (2022)** already introduced a dynamic entity memory
augmented with an auxiliary entity loss for narrative generation, with
published coherence/consistency metrics. The knowledge-graph story pipeline
(ICCC 2026) further shows graph conditioning improves character retention.

Therefore the entity-memory component of KALIA-EM+TTT is **not novel**. The
honest residual claim is narrower: the *combination* of entity memory with
in-place test-time fast weights, at 58M parameters, trained from scratch on
free-tier compute, evaluated with a frozen entity-consistency harness.

## Extractions and queue

| ID | Extraction | Status |
|---|---|---|
| M1 | Entity-consistency evaluation (retention, max-span, network edges) | **built** |
| M2 | KALIA-EM+TTT invention target | retained, prior art cited, claim narrowed |
| M3 | Nested-narration (story-within-story) objective | backlog (needs synthetic nested data) |
| M4 | Sentiment-arc metric for generated stories | backlog |
| M5 | Action → instruction → reflection curriculum | aligns with A2 (no new experiment) |

## What we deliberately do not take

- The dice game, Karna's armour, the Vyasa–Ganesha pace-control story: no
  compilable mechanism was found; recorded as discarded metaphors rather than
  dressed up as algorithms.
- **Chakravyuha: dismissal corrected.** The spiral geometry is still not a
  mechanism, but the entry-exit asymmetry is (see
  `docs/research/2026-09-23-chakravyuha-mechanism.md`): it produced the
  Abhimanyu-gap metric and redesigned X16 as chunk-preserving reversal
  training.
