# More Ancient Texts as Algorithmic Inspiration — Second Pass

Status: research queue
Date: 2026-09-23
Sources: Vedic patha recitation methods (Vedic chant literature; memory-chunking
analysis), Patanjali's Yoga Sutras (8 limbs; samyama stages), Natyashastra rasa
theory (Bh. 6), Jain anekantavada/syadvada, and computational work that maps
Indian epistemology to LLM engineering (Darshana framework, 2026).

## Honest framing (repeated deliberately)

A principle becomes an algorithm only when it yields a mechanism with a
measurable prediction. Everything below is stated as: source principle →
mechanism → nearest modern relative → test. Metaphors that could not be
compiled were discarded (e.g. "rasa as cosmic flavor").

## G1 — Jata-patha chained objective (forward + reverse chaining)

- **Source principle.** Vedic recitation preserves texts through redundancy in
  *both directions*: krama-patha chains forward (12, 23, 34 …); jata-patha
  recites forward-reverse-forward (12 21 12, 23 32 23 …); ghana-patha adds a
  bell-shaped triple. Multi-order recitation cross-checks the text with itself.
- **Mechanism.** Add an auxiliary "reverse chaining" loss: in addition to
  predicting the next token, the model predicts the *previous* token at each
  position (a second, reverse-direction head). Forward + reverse chaining =
  jata-patha as a training objective.
- **Nearest modern relative.** Multi-token prediction (forward only) and
  backward-LM auxiliary objectives; rarely combined at 58M for English.
- **Test.** X16: control vs control + reverse-chaining loss (λ = 0.3), equal
  tokens, two seeds. Note: English is not word-order-free like Sanskrit mantra
  recitation, so the auxiliary loss must be a *prediction* task, never input
  reversal — a lesson read directly from the source texts' own caution.

## G2 — Samyama adaptive depth (progressive refinement at inference)

- **Source principle.** Yoga Sutras' inner limbs form a progression: dharana
  (focus) → dhyana (sustained) → samadhi (absorption). Deeper engagement is
  applied where it is needed, not uniformly.
- **Mechanism.** The looped model (X11) already supports multiple recurrent
  passes at inference. Samyama scaling: evaluate the same checkpoint at 1, 2,
  and 4 recurrent loops and use more passes only for harder inputs.
- **Nearest modern relative.** Test-time compute scaling / recurrent-depth
  extrapolation; our addition is measuring the curve on our own story model at
  58M.
- **Test.** X17: inference-time loop scaling on the finished loop2 checkpoint.
  Near-zero cost (evaluation only). Implementation: `eval_probes.py --n-loops`.

## G3 — Rasa-conditioned generation (Natyashastra)

- **Source principle.** Bharata's rasa theory: eight stable sentiments
  (sringara, hasya, karuna, raudra, vira, bhayanka, bibhatsa, adbhuta) arise
  from a combination of determinants, consequents and transitory states.
- **Mechanism.** Weak-label stories with a lexicon-based rasa tagger (built
  once, deterministic), prepend a rasa token to training sequences, and evaluate
  whether conditioning steers the emotional tone of generations.
- **Nearest modern relative.** Controllable text generation / style tokens;
  absent from small from-scratch story models.
- **Test.** X18: control vs rasa-conditioned arm, plus a probe measuring
  emotional-tone shift under each condition. Labeling is CPU-only.

## G4 — Anekantavada model soup (many-sidedness)

- **Source principle.** Jain anekantavada: no single viewpoint is complete;
  syadvada formalizes conditional, multi-perspective assertions.
- **Mechanism.** Train two runs with different data order/seed and average
  their weights (model soup), evaluating whether the averaged model beats each
  individual view.
- **Nearest modern relative.** Model soups / weight averaging — known effect;
  our contribution would be a cheap two-seed test at 58M.
- **Test.** X19: least novel of the four; queued last.

## Queue

| ID | Experiment | Cost | Priority |
|---|---|---|---|
| X16 | Jata-patha reverse-chaining auxiliary loss | 2× 1.5h GPU | highest of the four |
| X17 | Samyama inference-time loop scaling | eval only | after X11 completes |
| X18 | Rasa-conditioned generation | labeling CPU + 2× 1.5h GPU | medium |
| X19 | Anekantavada two-seed soup | 2 full runs | low |
