# The vikṛti matrix is incomplete — and what survives the prior-art check (2026-10-12)

**Status:** research note, written the day the first two pāṭha arms finished
(X29: uniform 2.8076 / −0.0237 retention; jata 2.8510 / +0.0181) and ghana was
still training. Purpose: after the first arm pair, (1) establish what the
canonical Vedic rehearsal set actually contains, (2) check four candidate
"second ideas" against the 2025–2026 literature before any of them gets quota,
and (3) say honestly which are new and which are not.

Verification levels follow `CITATIONS.md`. Numbers attached to a legacy source
are marked **[tradition]** (recitation manuals and encyclopedic digests of the
tradition, quoted as such — these are not NLP results and must never be read
as measurements).

---

## 1. The finding: eight vikṛtis, two tested

The X29 pre-registration describes the pāṭha methods as "krama / jata /
ghana". The tradition's own enumeration is larger, and it is canonical —
repeated verbatim across recitation manuals and encyclopedic digests:

> jaṭā mālā śikhā rekhā dhvajo daṇḍo ratho ghanaḥ |
> aṣṭau vikṛtayaḥ proktāḥ kramapūrvā maharṣibhiḥ ||

"Jata, mala, shikha, rekha, dhvaja, danda, ratha, ghana — these eight vikṛtis
are declared, krama coming first." **[tradition]** Sources: the Kanchi
Shankaracharya's *Methods of Chanting* discourse; the *Patha* encyclopedic
digest; the *Vikṛtivallī* verse quoted in a 2025 education paper on the oral
tradition; a Sanskrit literary history quoting the same verse with a
commentary fragment.

Three facts matter for us:

1. **Krama is the gateway, not one of the eight.** The eight vikṛtis are
   defined *by the reversal of word order*; krama has no reversal. X29 runs
   krama as a mid-complexity arm, which is consistent with the tradition's own
   grading (simpler than the vikṛtis, more structured than padas),
   but the registered "four arms" phrase undercounts the canonical set by
   five.
2. **The tradition states a dependency structure among the methods.** The
   commentary fragment in the literary history says: jata and danda are the
   two *principal* (mukhya) vikṛtis; **shikha derives from jata; mala, rekha,
   dhvaja and ratha derive from danda; ghana derives from both** **[tradition]**.
   That is a partial order on rehearsal structures, authored by the tradition
   itself — and it is a natural ordering for an arm roster: test the two
   principal structures, then their derivations.
3. **The word-level patterns documented with concrete sequences:** jata
   (`1 2 2 1 1 2`), shikha (triples: `1 2 ~ 2 1 ~ 1 2 3`), danda (nested
   blocks with growing depth), ghana (nested triples), and dhvaja — described
   in one manual's table as binding the first pair to the **last** pair
   (`1 2 ~ 8 ...`, the table truncated in our source). Mala, rekha and ratha
   appear in every enumeration but their exact patterns **differ between the
   sources we found** (garland-with-pauses vs pairwise; nested reversal chain
   vs block reversal; moving interleave vs rotated descent).

**Consequence, stated honestly:** the eight names are canonical, but the
precise recitation sequences must be **pinned to a single documented source
per arm and registered before the arm runs** — exactly as X29's implementation
did for jata and ghana. We are not free-styling "ancient patterns"; we are
transcribing one school's documented sequence and labelling it as that.

### Why this is still the right thing to run

- It is the **first known test** of any of these orderings as replay
  schedules — the X29 prior-art check (no ML work on pāṭha schedules) is not
  disturbed by the wider set.
- The untested six are not cosmetic variants; they differ on axes that matter
  to a rehearsal schedule: **adjacency distance** (jata/krama: adjacent pairs;
  ratha/dhvaja: distant pairs), **nesting depth** (shikha → danda → ghana),
  and uniquely, **beginning–end binding** — dhvaja pairs the first words with
  the last, which is precisely the structure our retention and entity probes
  could detect.
- Cost is known: ~4.7 GPU-h per arm on the existing X29 harness. Six arms is
  ~28h — two to three quota windows, stageable one arm at a time.

### Registered-style hypotheses (to be fixed in the X30 pre-registration)

| arm | structure | hypothesis (to freeze before running) |
|---|---|---|
| dhvaja | first pair ↔ last pair binding | the only ordering that rehearses the head–tail relation; expect the largest retention gain on long-range/entity probes, and **no** val gain |
| danda | nested blocks, growing depth | principal form; expect val cost between jata and ghana, retention intermediate |
| rekha | nested reversal chain | deepest nesting, longest distance; expect ghana-like harm on val |
| shikha | forward triples | mildest of the derived set; expect smallest effect |
| mala | pairwise with pause structure | closest to krama-jata neighbourhood; expect near-uniform on retention |
| ratha | moving interleave | distance without nesting; separates "distance" from "depth" |

The matrix earns its quota only if we commit to these readings before the
first arm — the X29 pair already showed effects at ~0.04 nats, an order of
magnitude above the measured noise, so per-structure predictions are
falsifiable at this scale.

---

## 2. Prior-art checks on the second ideas

### 2a. Held-out "witness" data-mixing — NOT a new mechanism

The dessert result (X26/X28: a mixture policy fed by loss on the same shards
it trains on drifts to the easy sources, cost +0.018, dose-response across
three arms) suggested a "witness" variant: judge each source by a **frozen
probe set never trained on**. Prior-art check, 2026-10-12: that is *standard
practice* in the data-mixing literature, and the field is crowded —
DoReMi (Xie et al., 2023), ODM (Albalak et al., 2023), Data Mixing Laws
(Ye et al., ICLR 2025), TiKMiX (arXiv 2508.17677), FASTMIX (ICLR 2026),
DUET (2025). Several of them explicitly warn about validation overfitting or
add entropy regularization for exactly this reason (FASTMIX, ICLR 2026).

**Verdict:** the witness arm is worth running as the **internal control** for
the dessert — does the drift vanish when the feedback signal is uncontaminated,
at 58M, with everything else frozen? — but it is *not* a novel mechanism and
must never be framed as one. The novel part is the measurement the project
already owns: a small-scale, replicated, dose-response **demonstration of what
contamination does**, in a regime where the effect is legible and cheap.

### 2b. Prosody/chandas auxiliary objective — weak novelty, wrong corpus

Meter-as-auxiliary-task is established in poetry NLP and TTS: Deep-speare
(ACL 2018) trains joint language+meter+rhyme objectives and finds vanilla LMs
implicitly capture iambic pentameter; metrical tagging corpora train
stress/meter predictors (arXiv 2102.08858); prosody-augmented objectives
appear in constrained poetry generation (2026) and phoneme-level BERT for
TTS. Nothing we found uses **Vedic chandas/mātrā structure as a
self-supervised auxiliary during general-domain pretraining** — but our corpus
is TinyStories and FineWeb-Edu **prose**, where meter does not exist. The
residual-novelty question ("does prosodic supervision transfer to prose
modeling?") has no plausible transfer channel to justify ~19h of quota.

**Verdict:** NONE for now, and record the reason. Revisit only if the corpus
ever includes verse.

### 2c. Head–tail consistency as a decoding rule — crowded space, useful probe

Mīmāṃsā's interpretive principle *upakrama-upasamhāra* (the beginning and the
conclusion of a text must agree) maps onto selecting continuations whose
**ending still references its head's entities**. The decoding literature
already owns this neighborhood: Look-back decoding (arXiv 2305.13477) tracks
KL divergence to the prefix to prevent topic drift; reflection-window decoding
(arXiv 2502.03678) revises spans; contrastive/adaptive search penalizes
degeneration — coherence selection is one of the most crowded corners of
decoding research.

**Verdict:** a *cheap probe addition*, not a headline: extend the entity
harness with a head–tail entity-overlap metric and score the existing pools
with it (CPU, free). If the entity rule's advantage (0.030 → 0.091) is
concentrated in stories where head–tail overlap survives, that is one
sentence of mechanism evidence for the entity-decode result. Prior art must
be cited alongside; no novelty claim.

### 2d. Piṅgala's naṣṭa/uddiṣṭa as an algorithm-learning probe — parked

The Chandahśāstra's chapter-8 algorithms (prastāra enumeration, naṣṭa =
index→pattern, uddiṣṭa = pattern→index, implemented as recursive descent;
documented in the UoHyd recursion papers and the FPSAC 2022 talk) are a
2,200-year-old, exactly-specified recursive algorithm family, and no work we
found trains sequence models on the *naṣṭa/uddiṣṭa* task specifically.
It is a clean little question ("can a 58M LM learn a pre-Leibnizian recursive
algorithm from examples?") but it exercises integer algorithmic reasoning,
not the language/continual-learning axis this project is built on.

**Verdict:** parked, recorded, not queued. If ever run, it belongs with a
grokking-style CPU/GPU probe budget (~1h), not the main line.

---

## 3. What changes in the queue

1. **X30 registered as the vikṛti completion** — not "more arms", but the
   canonical eight minus ghana: dhvaja, danda, rekha, shikha, mala, ratha,
   each with its structure-specific prediction frozen before it runs, exact
   source-pinned patterns in the pre-registration, one arm per quota window.
   dhvaja first (head–tail binding is the most testable new structure).
2. **Entity-harness extension (CPU, free):** head–tail entity overlap as an
   additional, prior-art-cited probe metric for the existing decode pools.
3. **The dessert's second seed remains the standing gate** — the witness arm,
   if run, is a control for it, not a replacement.

## Sources used in this note (for the record)

Tradition: Kanchi Shankaracharya, *Methods of Chanting* (kamakoti.org and the
hinduonline.co digest); *Patha* encyclopedic digest, incl. the
`jaṭā mālā śikhā …` verse; Bhagwanti, *Preservation of the Veds* (aṣṭa-vikṛti
list and pattern tables); *Glimpses of History of Sanskrit Literature*
(wisdomlib, with the Sanskrit verse and the mukhya/derivation fragment);
Besra, *Memory Techniques in the Vedic Oral Tradition* (2025). Modern
scholarship: TiKMiX (arXiv 2508.17677); Look-back Decoding (arXiv 2305.13477);
Metrical Tagging in the Wild (arXiv 2102.08858); plus DoReMi, ODM, Data
Mixing Laws, FASTMIX, DUET, Deep-speare (ACL 2018), de la Rosa et al.
(Springer, 2021), reflection-window decoding, and the UoHyd/FPSAC Piṅgala
papers cited by name and venue.
