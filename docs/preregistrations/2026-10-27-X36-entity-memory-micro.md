# Pre-registration — X36: does a self-supervised entity memory improve retention?

Registered: 2026-10-27, before any GPU time on X36. Related: D23, D36, X32,
X33, X34, D59, D60.

## Why this exists

Every cheap selector failed. The model can generate the right continuation —
oracle 0.0794 against greedy 0.0170 on the 58M entity holdout — but the
information is not usable at generation time. X33 found a linear midpoint
signal, X34 failed to replicate it, and the linear-selector track closed.

The architectural hypothesis is different: instead of probing a frozen model
for a signal, **train the model to carry a small, self-supervised entity memory
and copy from it**. The model learns what to remember from a recurrence
objective, not from a post-hoc probe.

This is the first stage. It tests the memory/copy mechanism at the output head,
with the trunk unchanged and a parameter-matched control. If it wins, the read
moves into the trunk (Stage 2); if it loses, the mechanism is closed before we
build the deeper version.

## The mechanism

`model.py` now carries four optional pieces, all off by default:

1. **EntityWorkspace** — 32 slots of width `d`, with learned keys and a learned
   initial value. At every 64-token chunk boundary, a write gate sees the
   chunk's mean hidden state, the mean recurrence probability and the mean
   surprise (the model's own NLL), and updates each slot. The read is a
   cross-attention from the current chunk to the slot values, with a
   zero-initialised output projection. The workspace is a bottleneck, not a
   second KV cache.
2. **RecurrenceHead** — a linear head on the hidden state that predicts whether
   the same token id will appear again within 64 tokens. The label is
   self-supervised from the text; no parser. The loss is applied to token ids
   above 1000 (a fixed content mask; the 1,000 most frequent ids are excluded),
   and its output gates the workspace write.
3. **CopyHead** — a causal attention from the current hidden state to the input
   prefix hidden states, scattered onto the vocabulary ids of the prefix and
   added through a learned, initially closed gate. A name already seen can be
   copied instead of regenerated.
4. **OutputAdapter** — the control. A zero-initialised residual MLP with the
   same parameter count as the memory + copy + recurrence path, and no memory.

The trunk is unchanged: RMSNorm, SwiGLU, RoPE, QK-Norm, 6 layers × 384 × 6 at
the micro scale. The memory read is at the output, so the trunk is byte-for-byte
the control's trunk. Both arms start from the same base weights: the base is
initialised before any optional module is constructed.

The deeper version — memory cross-attention at two trunk layers, with a KV
cache across chunks — is **Stage 2**, and is not authorised by this
registration. It only runs if Stage 1 passes.

## Design

- **Base:** `configs/micro-qk-s1338.yaml`, 6 layers × 384 dim × 6 heads,
  context 512, Muon, QK-Norm, no soft-cap, 763 steps, seeds **1338/1339**.
- **Arms:**

| arm | what it adds | params |
|---|---|---:|
| `base` | nothing (the standard micro model) | 29,921,280 |
| `control` | parameter-matched output adapter, no memory | 30,707,712 |
| `copy` | copy head only | 30,216,577 |
| `memory` | workspace + copy + recurrence | 30,697,378 |

The memory/control difference is **0.03%**; the control exists so a memory win
cannot be a capacity win. The `copy` arm isolates the copy head from the
workspace; the `base` arm is the standard-model anchor.

- **Data:** the same micro training data and token budget as the other micro
  arms. No new corpus, no replay, no mixture change.
- **Training:** 763 steps, batch 16 × grad-accum 8, LR 6e-4, Muon; identical
  across arms except the four model flags and the seed.
- **Cost:** 8 arms × 763 steps. X31 ran 8 arms in 3.8 GPU-h; the memory path is
  heavier, so budget **~5–6 GPU-h** plus the entity probe. The quota after the
  running v0.3.0 epoch is ~21h.

## The entity protocol (micro adaptation)

The 58M protocol uses 768 prompt tokens and 256 generated; the micro context is
512. The micro protocol is a **short-context adaptation**, not a reproduction of
X34:

- 20 frozen Gutenberg stories, prompt truncated to **384 tokens**.
- 8 candidates × **128 generated tokens** at temperature 0.8 / top-k 200, plus
  one greedy draw.
- Split each candidate at **64 tokens**: the selector sees the first half; the
  metric is retention of the prompt's entities in the second half.
- Report per model: greedy retention, first-half pick, oracle, and the
  candidate pool's coverage.

The absolute retention is not comparable to the 58M numbers. The **paired
control/memory contrast** is the measurement.

## Pre-registered predictions

| ID | Prediction | Threshold |
|---|---|---|
| **M-1** | The memory model's own greedy continuation retains more entities than the control's | `memory − control ≥ max(control seed spread, +0.005)` on greedy retention, both seeds the same direction |
| **M-1b** | The candidate pool is not degraded | memory oracle not lower than control by more than 0.005; half-pick reported |
| **M-2** | The memory does not cost language quality | memory val loss − control val loss ≤ **+0.010** |
| **M-3** | The recurrence head learns the write signal | AUC > **0.60** on a held-out val batch (memory arm) |
| **M-4** | The copy head does not degenerate | copy/memory greedy distinct-2 not worse than base by more than 20%; copy accuracy on recurring names reported |
| **M-5** | The control ruler is reported | control seed spread on val and retention |

M-1 is the main test. A model that copies names but cannot keep the story's
entities has not solved the problem; a model that lowers val but loses
retention has not either.

## Decision rule (fixed now)

- **M-1, M-2 and M-3 pass** → the output-level memory/copy mechanism is a
  candidate. Register the 58M screen, then the Stage 2 trunk integration. No
  capability claim from the micro.
- **M-1 passes, M-2 fails** → the memory buys retention at a language cost;
  report the trade-off, no scale.
- **M-1 fails** → the mechanism does not improve retention at 30M; close this
  implementation. Do not tune the slot count, chunk size or gate until
  something passes; that is searching for a result.
- **M-3 fails** → the recurrence label/head is not learnable; fix the signal,
  not the model, and do not read M-1 as evidence about memory.
- **M-4 fails** → drop the copy head and test the workspace alone in a new
  registration; a copy head that repeats text is not a win.

## What this does not establish

- Not a 58M result; the micro is a screen.
- Not a full memory transformer; the read is at the output, the trunk is
  unchanged. Stage 2 is a separate registration.
- Not a claim of architectural novelty in the literature: entity memory, copy
  mechanisms, slot attention and contrastive predictive coding all exist. The
  novel combination is the self-supervised recurrence-gated workspace + copy
  head, measured on the project's held-out-half entity protocol at sub-100M.
- Not a capability claim even if the micro passes; the 58M replication is the
  next test.

## Prior art (cited, not claimed as new)

- Papalampidi & Lapata (2022), entity memory for story generation.
- Wu et al. (2022), Memorizing Transformers; Bulatov et al. (2022), Recurrent
  Memory Transformer.
- Gu et al. (2016), copy mechanisms; pointer networks.
- Locatello et al. (2020), Slot Attention.
- Oord et al. (2018), contrastive predictive coding (the recurrence objective is
  a discrete cousin, not the same method).
- D23/D36 in this repository already narrowed the KALIA-EM claim.

## Artifacts (to be archived)

- `docs/results/x36/` — the 8 training summaries, val logs, recurrence AUC,
  copy accuracy, entity retention, run manifests.
- `tools/kalia_x36_memory_kernel.py` — the training runner.
- `tools/kalia_x36_entity_kernel.py` — the micro entity probe.
- `kaggle/kernel-metadata-x36-memory.json`,
  `kaggle/kernel-metadata-x36-entity.json`.

---

## Amendment 1 (2026-10-27, before any GPU time)

The base config is built from `configs/micro-muon.yaml` with
`qk_norm: true` and `logit_softcap: 0.0`, because the `kalia-code-dev` snapshot
predates the committed `micro-qk-s1338.yaml` file. The X31 kernel built its QK
arm the same way. The arm is the same 6-layer × 384-dim × 6-head QK model, with
the same parameter counts; only the config source file differs. No arm, flag,
seed, bar or decision rule changes.

