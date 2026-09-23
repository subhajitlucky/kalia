# The Vedas as Algorithmic Inspiration — Third Pass

Status: research queue
Date: 2026-09-23
Sources: Rigveda dating and structure (Jamison & Brereton 2014; Witzel),
Nasadiya Sukta (RV 10.129) scholarship, four-Samhita structure literature
(Rig/Yajur/Sama/Atharva), selective prediction and calibration literature
(Geifman & El-Yaniv 2017; ConfidNet 2019; calibrated selective classification
2024).

## The texts (facts, for the record)

- **Oldest**: the **Rigveda Samhita** — the oldest extant Indic text,
  composed c. 1500–1200 BCE (books 2–9), 1,028 hymns / 10,600 verses in ten
  mandalas. The other three Samhitas date to c. 1200–900 BCE.
- **The four Vedas** (the foundation of Sanatana Dharma, each with a distinct
  function):
  1. **Rigveda** — verse hymns for recitation
  2. **Yajurveda** — prose formulas spoken while performing ritual actions
  3. **Samaveda** — the same verses set to melody; 1,549 stanzas of which all
     but 78 come from the Rigveda; one verse maps to many chants (gānas)
  4. **Atharvaveda** — practical charms, medicine and daily life; added later
- The first three form the *trayi vidya* (triple knowledge); the Rigveda is
  the source text, the others derive from and transform it.
- **Structural observation**: the Rigveda is deliberately ordered (hymns
  grouped by deity with decreasing counts; metres ordered jagati/tristubh →
  anustubh/gayatri).
- **Nasadiya Sukta** (RV 10.129): the creation hymn that ends in radical
  skepticism — "Who truly knows?" Possibly the earliest recorded statement of
  principled epistemic humility.

## Compilations

### H1 — Nasadiya head (epistemic humility as an auxiliary objective)

- **Source principle.** RV 10.129 refuses to assert what cannot be known and
  makes the limits of knowledge explicit.
- **Mechanism.** Auxiliary self-supervised head that predicts, per token,
  whether the model's own next-token prediction will be correct (trained
  against the actual correctness signal available during pretraining). At
  inference, its output is a calibrated "should I trust this continuation"
  signal.
- **Nearest modern relative.** Selective prediction / auxiliary confidence
  models (ConfidNet); calibration literature. Applying a token-level
  self-knowledge head during *pretraining* of a 58M LM is the new twist.
- **Test.** X20: control vs +Nasadiya head (λ = 0.2), equal tokens, two seeds;
  measure validation loss and selective accuracy at fixed coverage.

### H2 — "One verse, many chants" (format-conditioned generation)

- **Source principle.** The Samaveda takes a fixed set of verses and maps each
  to many melodic renderings — same content, many forms, systematically
  labelled.
- **Mechanism.** Our Cosmopedia data already carries a `format` label
  (textbook, story, wikiHow, …). Prepend a format tag to each sequence during
  training and measure whether tags steer the style of generations.
- **Nearest modern relative.** Controllable generation / style tokens.
  Free at our scale because the labels already exist in the dataset.
- **Test.** X21: control vs format-tagged training arm; probe generation under
  each tag. Requires one prep variant run (CPU).

### H3 — Yajurveda procedural subset (formulas attached to actions)

- **Source principle.** The Yajurveda is prose bound to action; its Black/White
  recensions differ in whether explanation is interleaved with or separated
  from the procedure.
- **Mechanism.** Upweight procedural text (Cosmopedia wikiHow-format subset)
  in the mixture and compare procedural-text proficiency.
- **Test.** X22: mixture arm with procedural share raised ≈5% → 10%; reuses the
  same prep variant as X21.

### H4 — Rigveda ordering (backlog only)

- Systematic ordering by decreasing complexity (deity groups, hymn lengths,
  metres). A complexity-ordered curriculum within each source is a known idea
  with mixed evidence; recorded here, not queued.

## Queue

| ID | Experiment | Cost | Priority |
|---|---|---|---|
| X20 | Nasadiya self-knowledge head | 2× 1.5h GPU | highest |
| X21 | Format-conditioned generation (Samaveda) | prep (CPU) + 2× 1.5h GPU | medium |
| X22 | Procedural subset mixture (Yajurveda) | shares prep with X21 | medium |
