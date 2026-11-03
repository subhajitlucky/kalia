# LLM idea ledger

Every candidate considered for KALIA's next experiment, its closest prior art,
its novelty class, and what happened. Read this before proposing a new run.

Classes: **A** new problem/composition; **B+** known mechanism, new setting;
**C** replication/ablation; **D** new benchmark/evaluation.

## Tested mechanisms

| candidate | prior art | class | status |
|---|---|---|---|
| X16 reversal training | reverse LMs (Ledom 2025; reverse LM 2024) | B | null |
| X18 NoPE + gated residual | NoPE 2023/24; Qwen3.8 gated residual 2026 | B | null |
| X19 branch norm | branch-normalised gated residual | C | null |
| X20 static gate | LayerScale/static gates | B | replicated, inverted |
| X21 replication | — | C | null |
| X25 test-time training | TTT 2020/2024 | B | null |
| X26/X28 adaptive mixture | DoReMi 2023; online mixing | B | null |
| X29 replay ordering | Ibrahim et al. 2024 | B | null |
| X31 QK-Norm/soft-cap | QK-Norm 2023; Gemma 2 2024 | C | architecture decision |
| X32 entity circuit | entity-tracking circuit 2024 | C | analysis |
| X33/X34 linear readout | linear probing/steering | C | null |
| X35 code mixture | data mixing | B | null |
| X36 entity memory | memory tokens 2020; RMT | B | null |
| X37 SUTRA | low-rank MoE (MoRE 2025); Pāṇinian framing new | B+ | null |
| X38 CRB | counterfactual representation; low-rank split new | A− | null |
| X39 CEA | causal attribution | — | not run |
| X40 intervention pretraining | causal world models; NextLat | A− | closed |

## Candidates after the audit

| candidate | closest prior art | class | status |
|---|---|---|---|
| OIT — order-contrastive intervention training | COCO-LM 2021; ALBERT SOP 2019; SCAN/COGS; MUFFIN/ProcBench; order sensitivity 2024/2026 | B+ | do not run as new |
| IOAM — ordered-intervention + attention mediation benchmark | causal mediation 2020–2023; activation steering via causal mediation 2026; attention gating 2025 | D | new as a benchmark; cheap |
| EBAS — entity-binding attention supervision | counterfactual attention supervision 2020/2023; causal attention supervision in RL/GNN 2023/2026; binding circuits 2024–2026 | B+ | do not run as new |
| ICC — intervention composition curriculum | skill composition 2024; compositional curricula 2025–2026; dataset cartography 2023 | B+ | do not run as new |
| Attention atlas + steering | BertViz 2019; head taxonomy/induction heads; attention knockout/steering 2023–2026 | D/B+ | artifact, not a mechanism |
| Edit algebra — identity/inverse/commutator/conservation | Composable Interventions 2024; ALAM 2026; “Algebraic Consistency Alone Does Not Certify…” 2026; Commutator Memory 2026 | B+/D | concept taken; benchmark application only |
| Impossible/infeasible interventions | “Recognizing Limits” 2025; code impossibility 2026; VLA infeasibility 2026 | D | new narrative application; close prior art |
| **Cross-model 2×2 composition evaluation** | composition benchmarks (SCAN/COGS); no exact 2×2 intervention match found | **D** | **next cheap experiment** |

## The verdict

No A-class new mechanism was found in the intervention/attention space. The
only defensible new work is **D**: a benchmark or measurement that makes the
next mechanism testable. The 2×2 benchmark is the first; the cross-model
evaluation is the next.

## Next D experiment — cross-model 2×2

**Question:** is the composition failure specific to KALIA, or universal across
small LMs? Does scale or instruction tuning move the S-P hard cells?

- Models: KALIA v0.3.1 (58M), GPT-2 small/medium, Pythia-70M/160M/410M,
  Qwen2.5-0.5B (if available).
- Data: the existing 2×2 benchmark, instruction and gold-edit modes.
- Metrics: S-P by operation; `number+place`, `rename+swap`,
  `rename+negate`, `swap+negate`.
- Cost: CPU/light GPU, no training.
- Novelty: D (new evaluation; no exact 2×2 intervention match found).
- Benefit: benchmark credibility and a scale map; **not** a better 031.
- If scale fixes the hard cells, that is a target for a future mechanism; if
  it does not, the benchmark is the paper.

## Search axes not yet exhausted

- **Feasibility:** impossible edits, ambiguous references, missing entities.
- **Conservation:** what must not change under an edit (counts, ownership,
  location, identity).
- **Temporal:** edits in a timeline and downstream effects.
- **Multi-party:** knowledge/belief edits in multi-agent stories.
- **Calibration:** does the model know when an edit is unsupported?
- **Cross-model/scale:** the D experiment above.
- **Transfer:** edit rules across languages, domains, and formats.

Each axis still needs two prior-art probes before it becomes an experiment.


## Ancient-text candidates (2026-11-03)

| candidate | formal idea | closest prior art | class | status |
|---|---|---|---|---|
| **Saptabhaṅgī Attention (SBA)** | seven-perspective attention: is / is not / both / inexpressible / combinations, mixed by a router; zero-init control | Multi-Perspective Reasoning Transformers (2021); NAST negation-aware training (2026); NLMs (2023); syādvāda XAI (2026) | **B+** | **not A; do not run as new** |
| Pratyāhāra Set Tokenizer (PST) | set-valued tokens with a composition rule | BPE/WordPiece; multi-token prediction; vocabulary compression | B+ | not first |
| Nyāya Five-Member Verifier (NFV) | five-stage thesis/reason/example/application/conclusion head | CoT, process reward models, verifiers | B+ | benchmark only |
| Pramāṇa Evidence Router (PER) | six-source evidence routing per claim | RAG, tool routing, MoE, claim verification | B+ | interface |

**Confirmed: PNB is D/B+, not a new problem.** Closest prior art: ReFract
(2026, perspective-awareness benchmark), SYMBOLICTOM (2023, multi-character
belief tracker), ToMBench / OpenToM / CogToM (2024–2026), and negation-
consistency benchmarks (2023–2026). The exact combination may be new, but it is
a theory-of-mind + negation application.

Under the “everything new” rule, **do not run SBA or PNB as a new mechanism**.
If a labelled B+ application is acceptable, PNB is cheap; otherwise keep
searching CPU-side. The ancient-text lens produced a new composition, not a new
primitive.
