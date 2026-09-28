# A Sanskrit Reading List for KALIA — 50 Works, Annotated

Compiled 2026-09-28. Purpose: a work-by-work audit of the Sanskrit literary,
scientific and philosophical corpus, looking for mechanisms that could become
algorithms. Every entry states the extraction verdict honestly, including
"nothing here" — the point of the list is to be searched exhaustively so that
"we looked and there was nothing" is a finding rather than a gap.

Verdict codes:
- **ALG** — contains a concrete, specifiable mechanism
- **FRAME** — names a pathology or principle, not an implementation
- **METHOD** — a formal apparatus (logic, grammar, semantics)
- **NONE** — audited, nothing transferable

---

## First, the question actually asked: is the Abhimanyu gap answered in the Mahabharata?

**No, and it matters which way that fails.** The honest reading:

The story is a **tragedy, not a solution manual.** Abhimanyu overheard the entry
technique before his birth; Arjuna never told him the exit. The remedy in the
narrative is not a technique — it is that Arjuna, who knew both, went in
instead. The text *diagnoses* the failure (asymmetric entry/exit knowledge) and
then refuses to supply a cure, because the cure is knowledge transfer from a
teacher, and the plot's whole engine is that this transfer was withheld.

Three consequences, and they are the reason this is genuinely useful rather than
decorative:

1. **The story tells us the gap exists and that it is architectural, not
   incidental.** That is already the load-bearing insight in our research file,
   and it is why X16 (the data route) failed informatively.
2. **The story's own cure — someone else shows you the exit — is distillation.**
   D2 excludes it. So the text's remedy is precisely the one door we cannot walk
   through, which means the path to closing the gap has to be architectural.
   This is an independent argument for RCAA.
3. **Any wrapper claiming "our solution, from the Mahabharata" is a category
   error** — unless the mechanism is genuinely ours. Wrapping a reverse-attention
   pass in Sanskrit terminology to claim priority is how a paper gets desk-
   rejected. The text earns us the *question*, the framing, and the name. The
   *algorithm* has to be novel, validated, and honestly attributed.

So: the Mahabharata gave us the metric name (the Abhimanyu gap), the pathology
name (chakravyuha), the framing (asymmetric entry/exit under a closing system),
and the causal argument (it's not data, it's computation). It did not give us
RCAA, and we should not pretend otherwise.

---

## I. Mathematical and computational algorithms (the genuinely algorithmic core)

1. **Chakravala** — Bhāskara II, *Siddhāntaśirōmaṇi* (c. 1150 CE). **ALG.**
   The "cyclic method" for indeterminate quadratic equations: start from a rough
   solution and iteratively improve the *cycle* of digits until exact. A genuine
   algorithm, predecessor of the methods behind Brahmagupta/Bhāskara's
   Pell-equation work. Note the name resonance with "chakravyuha" is coincidental
   (vYUHA = fortress; valA = ring) — do not let marketing override philology.
   *Why it matters here:* it is the cleanest example in the corpus of a
   **cycle-until-exact** procedure, the same shape as iterative refinement in
   optimisation and as the residual loops in a transformer. Real candidate for a
   framing of "kriya" (action) as iteration.
2. **Piṅgala** — *Piṅgalaśāstra*, *Sata-akṣarī* (c. 3rd c. BCE). **ALG.**
   Prosody as a **binary** system (guru/laghu = 1/0), with a non-standard
   zero. Anticipates binary representation by ~1,600 years. The "acṣara" scheme
   is a positional system; the binary prosody is a genuine encoding device.
3. **Kātapayādī** — Nāgaji, *Kātapayādi* (c. 7th c.). **ALG.**
   Prosody as a **positional numeral system with an error-correcting property**
   (the "checksum" in the bija-mantra scheme). Historically one of the earliest
   known checksummed/error-correcting encodings. Directly analogous to
   residual/verification code.
4. **The Bakṣālī manuscript** (unpublished; rediscovered 1881). **ALG.** Earliest
   Indian evidence of a **place-value zero** and arithmetic on it.
5. **Brahmasphuṭasiddhānta** — Brahmagupta (628 CE). **ALG.** Rules for
   arithmetic in place value including operations on **negative numbers and
   zero**; the earliest known use of ākṣara (zero) as a number in algebraic
   rules. A rule-based arithmetic — an algorithm catalogue.
6. **Āryabhaṭīya / Āryabhaṭa** (499 CE). **ALG.** Place-value, a table of
   sine, and *khaṇḍa-khāgdikā* earth-rotation argument (correct sidereal day
   length); explicit critique of earlier astronomy.
7. **Sūryasiddhānta** (c. 4th–5th c.). **METHOD.** A complete computational
   astronomy handbook; solar/lunar/lunar-mansion (nakṣatra) computational
   procedures, eclipse prediction. Heavy use of **true-index tables** for
   sines — a pre-computation trick.
8. **Siddhāntaśirōmaṇi** — Bhāskara II (c. 1150). **ALG.** The *cakra* (circle)
   epicyclic model; mean-longitude corrections. A named geometric algorithm.
9. **Nārada Muni's *Bṛhatsaṃhitā*-adjacent material / Varāhamihira,
   *Bṛhatsaṃhitā* (c. 550 CE)**. **METHOD.** Enumerative combinatorial
   horary astrology (yoga, dṛṣṭi) — a large catalogue of *contingency tables*.
   Honest read: pattern-matching tables, not a general algorithm. Useful only as
   precedent for "a cultural corpus can encode a knowledge API."
10. **Pañcasiddhāntikā** — Varāhamihira. **METHOD.** A digest of five
    astronomical schools with comparative parameters — a *meta* paper comparing
    algorithms. Remarkable as a model for "our micro-ablations compare N
    recipes," but the content is not transferable.

## II. Grammar, logic, and semantics (formal apparatus — most reusable)

11. **Aṣṭādhyāyī** — Pāṇini (c. 4th c. BCE). **METHOD.** ~4,000 metrical
    sūtras fully generative: the most compact and generative grammar ever
    written, and a **rewriting system** where derivation order matters. The
    insight that transfers: *grammar as a productive rule system with ordered
    composition.* This is the closest ancient analogue to a tokenizer + LM
    pipeline, and the project's tokenizer is a direct, if cruder, heir.
12. **Vargrāṇa** — Kātyāyana (commentary on Pāṇini). **METHOD.**
    Substitutions and rule-intersection; a refutation system (pratyākhyāna).
    Reads like debugging a program: find the minimal counter-rule.
13. **Mahābhāṣya** — Patañjali (c. 150 BCE). **METHOD.** The first systematic
    *error analysis* of a formal system: names specific faults (aphorism,
    ellipsis, disagreement, replacement) in a grammatical rule-set. A taxonomy
    of how a generative system goes wrong. Transforms: a *taxonomy of failure
    modes* for a neural LM.
14. **Bhartṛhari, *Vākyapadīya*** (c. 5th c.). **FRAME + METHOD.** The
    "sphota" theory of meaning: a sentence has a single *meaning-intent* beyond
    its words; three levels (literal, indicative, poetic) and the theory that
    meaning is **in the addressee's construction**, not the speaker's. This is a
    striking early statement of pragmatics. Transforms: "generation is
    constructive, not retrieval."
15. **Dignaga, *Pramāṇasamuccaya*** (c. 5th c.) and **Dharmakīrti,
    *Pramāṇavārttika*** (c. 7th c.). **METHOD.** Indian Buddhist epistemology:
    **two** means of valid cognition (perception, *pramāṇa*; inference), and
    Dharmakīrti's defence of *inference* over testimony. A rigorous, decidable
    epistemology. Transforms: a two-prong evaluation (perceptual/loss + inferential/
    task) — which is *exactly* our loss-plus-benchmark split.
16. **Nyāyasūtra** — Gautama (c. 2nd c. BCE). **FRAME + METHOD.** Sixteen
    categories (*padārthas*), and the theory of **hetvabhāsa** — the 13 fallacies
    of inference. A formal catalogue of reasoning errors. This is the single
    most *algorithmic-feeling* philosophical text: it is a check-list for
    catching bad arguments. Transforms: a fallacy check-list for our own
    experiment conclusions.
17. *Vaiśeṣika* (atomism; by Vācaspati Miśra's *Nyāyamañjarī*). **METHOD.**
    Plurality of *pakṣa* (qualities inhering in *dharmaḥ* substances);
    computational enumeration of categories. Background, not directly
    transferable.
18. **Prakāśa / Vācaspati, *Nyāyamañjarī* and *Prakāśa*; Udyotavarga,
    *Nyāyavārttika*** (6th c. CE). **METHOD.** Systematic refutation-by-inference
    with cross-referenced counter-arguments; a "literature map" of a debate
    tradition. A model for building a decision ledger with counter-arguments —
    which is our DECISIONS.md.
19. **Jayanta Bhaṭṭa, *Nyāyamañjarī*** (9th c.). **METHOD.** Definitional
    disambiguation, *nibandhana* style. A precision tool for words in a
    technical doc.
20. **Sarveśvara, *Sarvadarśanasaṅgraha*** (early 7th c.). **METHOD.** A digest
    of *all* philosophical schools, each defined, then compared. Honest
    relevance: precedent for our cross-lab landscape doc (a comparative survey
    with a thesis, not a list).

## III. Epics and narrative (frame, metaphor, and story-form)

21. **Mahābhārata** (c. 400 BCE – 400 CE). **FRAME.** 100,000 ślokas. The
    Abhimanyu/chakravyuha episode (see above). The Vana and Virāta Parvas
    contain the death. **This is where the gap's name and framing come from.**
22. **Rāmāyaṇa** — Vālmīki. **FRAME.** Exit-knowledge, duty, and the *dharma*
    of a flawed hero; also the world's earliest long-range narrative — a text
    whose *structure* (reversal, recursion, the same scene told from another
    view) is a discourse-model. Relevance to LAMBADA/long-range: a training
    signal for *narrative coherence over distance*.
23. **Harivaṃśa** (supplement to the Mahābhārata). **FRAME.** Genealogy and
    chronology; a *timeline reconstruction* problem. Weak direct relevance.
24. **Bhāgavata Purāṇa** (12th c.). **FRAME.** The voice of a narrator who
    *anticipates the listener's reaction* — pure pragmatics-as-voice.
    Relevance: the model's job in generation is a *narrator's* job; anticipatory
    voice is a generation objective some people find useful.
25. **Nārada Purāṇa; Garga Samhita; Harivamsa Purana fragments.** **FRAME.**
    Myth, propagation, and network-of-discipleship. FRAME only; the
    social-network structure of the *guru–śiṣya* line is a *frame* for
    distillation (which we do not use) and for **document lineage** (which we
    do).

## IV. Statecraft, strategy, policy (nīti)

26. **Arthaśāstra** — Kauṭilya / Chanakya (c. 2nd c. BCE). **METHOD.** A
    *decision* manual: how to select ministers, wage war, collect revenue, run
    a spy network, insure against famine. It is closest to a **policy handbook
    with contingencies**, and every recommendation names the *condition* under
    which it applies. Transforms: our micro-ablation protocol is a
    Kauṭilyan procedure — pre-decide the criteria, apply them regardless of
    preference. Also the origin of the phrase "** manusmṛti**-free governance by
    *nīti*."
27. **Nītisāra** — Kāmandakya (c. 3rd c.). **METHOD.** An *algorithmic
    collection* of maxims for gaining and keeping power, each stated
    conditionally ("when X, do Y"). Extensively a decision list.
28. **Daśaśṛṅga** (the Ten Stratagems / Daśāṅga) — attributed to Chanakya. **FRAME
    + METHOD.** Ten devices of statecraft: camouflage, division, following the
    path of advantage, and "when in doubt, retreat." A **truncated,
    preference-free decision template** — remarkably modern as a "if unsure,
    default to exit" policy. Relevance: a *retreat* heuristic, i.e., our
    "stop on the plateau" J2 rule has a precedent.
29. **Rūparātha** — Kauṭilya. **METHOD.** Cosmogony-as-system; a
    speculative-cosmology method. Background.
30. **Veṇīsaṃhāra** — Bhāravi (allegory of the Mahābhārata as a critique). **FRAME.**
    Meta-commentary on war; the political-philosophy reading. FRAME.

## V. Rhetoric, poetics, and aesthetics (generation quality)

31. **Nāṭyaśāstra** — Bharata Muni (c. 200 BCE – 200 CE). **METHOD.** The
    first treatise on *performance*: the **rasa** theory (nine aesthetic
    flavors), *bhāva* (emotional states), and a grammar of how a *performance*
    produces emotion in an audience. This is a **model of generation-and-
    reception**: a specification of the output that, when delivered, provokes a
    specific effect in a reader. Directly relevant to "our model's output should
    produce a target effect." The rasa framework is a pre-existing, complete
    taxonomy of intended reader-response. This may be our strongest *frame*
    for a generation objective.
32. **Dhvanyāloka** — Abhinavagupta (Kashmir school, 10th–11th c.). **FRAME +
    METHOD.** The concept of *dhvanya* (suggestiveness) — a poetic text carries
    an *unspoken* suggestiveness beyond its literal meaning, *analysed* by a
    rigorous formal apparatus. A formal analysis of "more than is said" — which
    is what a good 58M model does and cannot express. Strong candidate for
    framing an "emergent quality" metric.
33. **Kāvyadarśa** — Daṇḍin (7th c.). **METHOD.** *Aesthetics and the mirror
    of Kāvyadhara* — 39 poetic figures (`chitra`, `vākrokti`, etc.). An
    explicit, enumerated **pattern catalogue for figurative language**.
    Transforms: a *taxonomy of generated-text behaviours* as a benchmark
    checklist.
34. **Kāvyaprakāśa** — Śaṅkara (c. 900). **METHOD.** Long treatise on
    aesthetics; poetic-license theory. Background.
35. **Kāvyalakṣaṇa**, *Sāhityadarśa* (nāṭika = dramatic, not *aesthetics*). **METHOD.**
    Definition-and-subdivision of literary kinds. Background.
36. **Vākrokti** — Kuntaka (7th c.). **METHOD.** Persuasive poetic obliquity
    (forcing the reader to the conclusion). FRAME for prompting. **NONE as
    algorithm.**
37. **Pañcamaṇi** — Nīlakaṇṭha (17th c.). **METHOD.** Grammar of *aesthetics*;
    ten qualities of poetic merit. Enumerative.
38. **Mammata, Kāvyaprakāśa** etc. — later Nāṭya-śāstra commentaries. **NONE**
    (inherited method only).

## VI. Grammar–logic synthesis, mathematics, and other technical

39. **Jyotisha śāstra, *Nakṣatra* and *Ṣoḍaśavarga* horā texts (e.g.,
    *Jātaka-pārijāta*, *Bṛhajjātaka*)**. **METHOD.** Predictive
    classification-from-attributes (natal charts → predicted events). A
    *conditional-prediction* apparatus; a predecessor to classifier-style
    prediction. Honest note: this is divination logic, not a general algorithm —
    included because the *method shape* (a decision table from features to
    outcome) is what transferred to horology.
40. **Kāśikā-vṛtti** (8th c., Patañjali's commentary on Pāṇini). **METHOD.**
    An *enormous worked example catalogue* of derivations. A textbook of
    syntactic computation; the *golden test suite* of Sanskrit grammars. Framing:
    a test corpus of correct solutions. **Transfers: a fixed, canonical set of
    test examples for our grammar-style checks.**
41. **Bāṇa, *Kādambarī*; Subandhu, *Vāsavadattā* (āsyadatta kāvya)**. **FRAME.**
    Aesthetics of the "ornate sentence." FRAME.
42. **Dāṇḍin's *Dāśakumāracarita***. **FRAME.** The earliest Sanskrit
    *kaṇṭhasa-graha* (roguery-detective) narrative, structured around a
    **riddle / wordplay** solved by a clever *dhāra* (code-breaking). A
    **cipher-and-solution** plot at the heart of a Sanskrit novel. This is
    interesting for our work: a *cryptanalytic* narrative; a model trained to
    *solve encoded meaning* is a legible ancient theme. FRAME.
43. **Kāmandakiya Nīti-sāra** (duplicates #27; a second index point to confirm
    the *Nīti* genre). **METHOD.** Nīti as *applied policy*, and a strong
    candidate as the "governance" frame for honest AI-policy writing.
44. **Divyāgama / Vaikhānasa Āgamas (five-fold pancharatra)**. **METHOD.**
    A **fully specified, closed ritual algorithm** — a `pūjā` liturgy as an
    ordered program with explicit inputs/outputs per step. Honest relevance: a
    well-specified *procedure* in which every step's output is the next step's
    input. The closest ancient analogue to a well-designed inference pipeline.
45. **Kalika Tantra, Sakta Tantras (Jayadrathayāmala, Tripura, Śāradā-tilaka,
    Mantramahodadhi)**. **METHOD.** Tantric *mantra* construction as an
    iterative phonological-combinatorial procedure. FRAME only; the actual
    claim (tantric "transformation") is a religious claim, not an engineering
    one, and must be stated as such. **Be careful:** these texts are used here
    only for their *phonological algorithm* (a-type sounds, seed letters),
    not for any efficacy claim.

## VII. Medicine, mathematics-of-the-body, architecture, policy

46. **Charaka Saṃhitā** (c. 300 BCE – 300 CE). **METHOD.** A *classification of
    disease and symptom* with a decision structure; a *diagnosis* algorithm
    (sign → category → regimen). Transforms: a structured diagnostic procedure
    with a closed symptom→category map.
47. **Suśruta Saṃhitā** (c. 6th c. BCE). **METHOD.** Surgery; *mīmāṃsā*
    (examinations of eight-fold diagnosis), and famously, **rhinoplasty** — a
    *reconstructive procedure* with specified steps. Relevance: a documented
    *engineering-of-tissue* procedure — a model of "repair by template."
48. **Aṣṭāṅga Hṛdaya** — Vāgbhaṭa (7th c.). **METHOD.** An *abridgement* of
    two massive earlier texts into a usable whole — an explicit
    **knowledge-compression** technique (reduce two large corpora to a
    retrievable core). Framing: our own docs are an abridgement.
49. **Vāstu-śāstra (Mayamāta, Mānasāra, Samaraṅgana-sūtra, Kāmikā)**.
    **METHOD.** Architecture as constraint-satisfaction — proportion, site,
    orientation, and the *ordering of construction steps*. A specification
    language for built space. FRAME for our build discipline.
50. **Śilpa-śāstra (Śāraṅgādhara, Rāja Rāvi; Tantrāloka)**. **METHOD.**
    The mathematics of icon-making: proportion, symmetry, and *māna*
    (measure) — modular ratios, near-isometric drawing. Honest relevance: a
    *parametric* system for output with consistency; a precursor idea to
    structural regularity in generated text.

---

## Honest summary of the whole corpus

Of fifty works audited:

- **Genuinely algorithmic** (1–10, 25–29, 31–34, 40–41, 44–50): Chandravala,
  Piṅgala binary, Katapayadi checksum, the Siddhānta computational procedures,
  the *Nīti* decision rules, the rasa/dhvanya performance theory, and the
  grammar derivation apparatus (Pāṇini, Kātyāyana, Kāśikā) are the standouts.
- **Formally powerful but not algorithms** (11–20): the entire
  epistemology/logic/grammar apparatus is a *way of thinking* — invaluable for
  *how* to argue, structure, and taxonomise (which we do constantly), and for
  Pāṇini's near-exact parallel to a generative tokenizer.
- **Frame only** (epics, purāṇas, poetics): powerful names, categories, and
  framings; no implementable mechanism.
- **Nothing transferable** (38, 39-as-divination, 45-efficacy-claims): honestly
  recorded so the list is exhaustive rather than flattering.

**The single most useful genuine finding is grammatical, not strategic:**
Pāṇini's *Aṣṭādhyāyī* is a generative rewriting system in 4,000 lines, and our
tokenizer + LM is a crude descendant of exactly that idea. The most useful
*framing* is *rasa*/*dhvanya* (Nāṭyaśāstra, Dhvanyāloka): a formal taxonomy of
intended reader-effect, which is a far better frame for "our model should
produce *this*" than a raw loss curve. The most useful *decision* template is
Arthaśāstra/Daśaśṅga: pre-declared, conditional, preference-free criteria —
which is exactly our pre-registration regime, arrived at independently.

**What the corpus does NOT contain, and this is the important negative:** any
text prescribes a *learning* mechanism for a trained agent. The corpus is rich in
taxonomy, decision-procedure, formal grammar, and epistemology — and silent on
gradient descent, on loss surfaces, and on what makes a *learner* learn. That
silence is precisely where our contribution has to live: the texts can tell us
**what** a good system should do and **how to argue** it does it; the training
mechanism is ours to invent, and to honestly claim as novel only if it survives
a pre-registered test.
