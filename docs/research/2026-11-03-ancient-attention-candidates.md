# Ancient-text candidates for a new attention/tokenizer

Date: 2026-11-03. Status: proposal, no GPU. Lens: source principle → formal
structure → mechanism → nearest modern relative → test. If the mapping is only
metaphor, discard it.

The repo already mined Jāta-pāṭha (X16), Sāma-format conditioning (X21),
Nasadiya self-knowledge (X20), Rasa conditioning (X18), Anekāntavāda soup
(X19), and Pāṇinian rules (X37). The candidates below are **not** those.

## Candidate 1 — Saptabhaṅgī Attention (SBA)

**Source.** Jain syādvāda / saptabhaṅgī: a predication is made from a
perspective, with seven modes — *is*, *is not*, *is and is not*,
*inexpressible*, *is + inexpressible*, *is not + inexpressible*, and
*is + is not + inexpressible*. The point is not seven truth values in a
truth-table; it is **perspectival qualification**: the same relation can be
affirmed, denied, both, or left open, under a stated standpoint.

**Formal structure.** For a query-key pair, the model computes a 7-way
perspective distribution `π(q,k)` and a per-perspective attention bias
`b_m(q,k)`. The attention logit is

```text
logit(q,k) = base(q,k) + log Σ_m π_m(q,k) * exp(b_m(q,k) / τ)
```

with a learned router for `π` and per-head, per-mode biases. At
initialisation the biases are zero and the router is uniform, so the model
starts exactly as its control. No new KV cache; a small router and 7 biases per
head.

**Why it is not just seven heads.** Heads are parallel feature channels; the
seven modes are **logical qualifications of the same relation**, mixed by a
perspective router. The mechanism can represent “attend to this token as
affirmed from one standpoint, denied from another” without duplicating the
whole head. The test is whether the router uses the modes on a
perspective/negation task.

**Nearest modern relatives.** Multi-view/multi-head attention, fuzzy/soft
attention masks, negation-aware models, and mixture-of-experts routing. The
syādvāda concept has been used for **XAI framing** (2026), not, to our
knowledge, as a seven-perspective attention mechanism.

**Verdict.** **A−/B+** — a new composition with a specific logical structure;
the components are known. It is not a new primitive, and it must not be called
“never tested” without the pair of prior-art probes.

**Benefit for KALIA.** A targeted capability for **perspective and negation**:
who believes what, what is denied, what is uncertain. X40 showed the model uses
a “did not” heuristic; SBA gives it a structured way to represent the
qualification instead of a surface marker. It will not improve val.

## Candidate 2 — Pratyāhāra Set Tokenizer (PST)

**Source.** Pāṇini’s pratyāhāra: a compact name denotes a **set** of phonemes
via marker letters. The formal structure is a set-valued symbol with a
composition rule.

**Mechanism.** A tokenizer whose tokens can denote **sets of subword units**;
attention can address a set with one query, and a decoder expands it. For
English this is a vocabulary/sequence-length experiment, not a Sanskrit
transliteration.

**Nearest modern relatives.** BPE/WordPiece, byte-level tokenization,
multi-token prediction, vocabulary compression, set-valued tokens.

**Verdict.** **B+** — the set-valued idea is a new framing, but tokenization is
crowded and the English benefit is unproven. Not the first bet.

## Candidate 3 — Nyāya Five-Member Verifier (NFV)

**Source.** Nyāya syllogism: pratijñā (thesis), hetu (reason), udāharaṇa
(example), upanaya (application), nigamana (conclusion).

**Mechanism.** A five-stage auxiliary reasoning head that must fill each member
and a verifier that checks the conclusion against the reason/example. It is a
structured chain-of-thought, not a new attention.

**Nearest modern relatives.** Chain-of-thought, process reward models,
verifiers, structured reasoning.

**Verdict.** **B+**. Useful as a reasoning benchmark, not a new attention.

## Candidate 4 — Pramāṇa Evidence Router (PER)

**Source.** Mīmāṃsā’s pramāṇas: perception, inference, comparison, testimony,
postulation, non-apprehension.

**Mechanism.** A six-source evidence router at each layer; the model selects a
pramāṇa per token/claim and cites it.

**Nearest modern relatives.** Retrieval-augmented generation, tool routing,
mixture-of-experts, claim verification.

**Verdict.** **B+**. A nice interface, not a new primitive.

## Recommendation

**Test the D first: a Perspective & Negation Benchmark (PNB).**

Before building SBA, check whether the capability gap is real:

- stories with characters who **believe / deny / are unsure** about a fact;
- questions answered from a named perspective;
- negation consistency (A believes P; B believes not-P; what does each say?);
- uncertainty (“unknown” is a valid answer).

Models: v0.3.1, GPT-2 small, Pythia-160M, Qwen2.5-0.5B (public checkpoints).
Cost: CPU/light GPU. No training.

If the models already handle PNB, SBA is unnecessary. If they fail, SBA gets a
30M screen:

- arms: control, pairs/negation data, SBA;
- 2 seeds, 763 steps, ~3–5 GPU-h;
- bars: val ≤ control + 0.005; PNB perspective accuracy ≥ +0.10; negation
  consistency ≥ +0.10; trained cells not regress;
- kill on val tax or no PNB gain.

This is the godmode sequence: **D benchmark first, A− attention only if the gap
is real.**

## Honest sentence

The ancient text supplies a **formal constraint** — perspectival qualification
— not a brand name. SBA is a new composition, not a proven mechanism. If PNB
shows no gap, the correct result is to stop, not to build the metaphor.
