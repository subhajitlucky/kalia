# Ancient Indian Texts as Algorithmic Inspiration — Three Testable Designs

Status: research queue
Date: 2026-09-23
Sources: Arthashastra (mechanism design, monitoring frictions), Panini's
Ashtadhyayi computational-grammar literature (Kiparsky, Scharf, Sridhar &
Srinivasa), Nyaya/Apoha semantics (Dignaga, Dharmakirti; Navya-Nyaya
computational work, arXiv 2604.04937), Apoha-vs-error-signal discussion.

## Honest framing

Ancient texts are not cryptography: they contain *principles of organization*,
not ML algorithms. The value is in the formulations — precedence orders,
resource doctrines, exclusion-based meaning — which can be formalized into
trainable mechanisms and tested under our standard protocol. Each design below
states its nearest modern relative and what is actually new in the combination.

## A1 — Kautilya adaptive mixture ("four strategies" bandit scheduling)

- **Source principle.** Arthashastra's resource doctrine: allocate to
  provinces by strategic return; respond with four strategies — sama
  (conciliation/keep), dana (gift/upsample), bheda (division/downsample),
  danda (punishment/freeze).
- **Mechanism.** Data sources are sampled with weights re-estimated every N
  steps from each source's marginal loss improvement: sources that keep
  improving are upsampled (dana), stagnant ones downsampled (bheda),
  degrading ones frozen until they recover (danda).
- **Nearest modern relative.** Online data-mixing and bandit data selection;
  RegMix/DoReMi are offline mixture optimization.
- **New in combination.** Online, loss-trend-driven reallocation among four
  heterogeneous sources at 58M scale; no proxy model needed.
- **Test.** X13: static 60/20/15/5 mixture vs adaptive reallocation, equal
  tokens, two seeds. Needs per-source shards in the data loader.

## A2 — Utsarga-Apavada schedule (general rule, then exceptions)

- **Source principle.** Panini's blocking: a special rule (apavada) overrides
  a general rule (utsarga); later rules are invisible to earlier ones
  (asiddhatva), so the system is ordered rather than blended.
- **Mechanism.** Two-phase curriculum with a *loss-triggered*, not
  step-triggered, transition: phase 1 trains the general distribution
  (stories + bulk educational web); when validation loss crosses a threshold,
  phase 2 introduces "exception" data (synthetic textbook formats, code) with
  precedence weights that dominate the mixture.
- **Nearest modern relative.** Curriculum learning; SmolLM2's annealing
  upsampling of math/code. Triggered-by-loss phase transitions are the
  variation.
- **Test.** X14: static mixture vs two-phase schedule with the switch at a
  fixed validation loss. Cheap: two bins + a phase flag.

## A3 — Apoha-ELECTRA (meaning by exclusion, with self-generated negatives)

- **Source principle.** Dignaga's apoha: the meaning of a term is constituted
  by the exclusion of what it is not; validity comes from the joint absence
  (vyatireka) of the term where the referent is absent.
- **Mechanism.** An auxiliary binary head classifies every token as original
  or replaced. Replaced tokens are sampled from the model's *own* current
  distribution, so the negatives evolve with training ("self-generated
  apoha").
- **Nearest modern relative.** ELECTRA replaced-token detection — one of the
  few *proven* sample-efficiency methods at small model scale; typically the
  negatives come from a fixed small generator rather than the model itself.
- **New in combination.** Self-generated, training-time-adaptive negatives at
  58M, with an explicit exclusion objective.
- **Test.** X15: control vs control + detection head (loss = LM + λ·detection),
  equal tokens, two seeds. This is the strongest of the three because the
  underlying mechanism has established small-scale wins.

## Queue

| ID | Experiment | Cost | Priority |
|---|---|---|---|
| X13 | Kautilya adaptive mixture | 2× 1.5h GPU | after architecture session |
| X14 | Utsarga-Apavada loss-triggered curriculum | 2× 1.5h GPU | after X13 |
| X15 | Apoha-ELECTRA detection head | 2× 1.5h GPU | **highest of the three** |
