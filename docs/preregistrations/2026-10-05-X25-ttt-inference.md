# Pre-registration — X25 in-place test-time training at 58M (Stage 1: implement + measure)

Registered: 2026-10-05, before any TTT code exists in this repo and before any
measurement with it. Stage 1 registers the mechanism and the measuring set. It
sets no improvement threshold — thresholds are set by amendment from the
measured baseline variance, the same two-stage pattern as v0.3.0's Step 1
spread. A threshold written before the first measurement would be a round
number below the noise floor, and X18 is the reason we no longer do that.

Related: D22 (high-impact techniques), D23 (KALIA-EM+TTT invention target),
D36 (entity-memory prior art narrows the novelty claim), D21 (no
distillation — TTT uses no teacher, so the lineage stays ours), D38
(pre-registration as identity).

## Prior art, stated plainly

Test-time training exists: Titans (NeurIPS 2025) and TTT-E2E (NVIDIA 2026)
update weights at inference; In-Place TTT / IP-TTCD (2026) reuse the MLP
down-projection as fast weights (~20-line change). Entity memory exists:
Memorizing Transformers (2022); Papalampidi & Lapata (2022) for narrative
entity tracking. What is, to our knowledge, unpublished is **their combination
at sub-100M scale for story consistency, from scratch, on free compute**.
X25 builds the TTT half only. The EM half (X9) and the combination stay queued
until X25 reports.

## The mechanism (frozen by this document)

`ttt.py` (new file, inference-only, no training involved):

1. Load the v0.2.0 checkpoint. Generate a continuation for a probe story with
   the existing sampler (temperature 0.8, top_k 200, fixed seed 7).
2. Every **256 generated tokens** (one chunk), take **1 gradient step** on that
   chunk's own next-token cross-entropy loss, updating **only the ten
   `mlp.down` matrices** (~7.2M params, ~12% of the model). Everything else
   stays frozen. Learning rate **1e-4**, no momentum, no schedule.
3. Continue generating with the updated weights. **Reset to the base
   checkpoint before every story** — fast weights are story-local memory, not
   permanent learning. A weight change that persists across stories is a
   training run wearing a TTT name, and this document forbids it.
4. Control arm: identical code path with learning rate **0** (zero-update
   control). If the control matches the treatment, the gain is scaffolding,
   not memory.

Constants above (256, 1 step, 1e-4, `mlp.down`, seed 7, temp 0.8 / top_k 200)
are fixed for Stage 1. Tuning them before measuring is searching for a
result; tuning happens, if at all, in Stage 2 under its own registration.

## The measuring set (frozen by this document)

20 Gutenberg passages, 2,000–4,000 tokens each, published as
`eval/ttt_probe_stories.json` with sha256 recorded in the ledger amendment.
Gutenberg is unseen by v0.2.0 (trained on FineWeb-Edu / TinyStories /
Cosmopedia / Python), so there is no train-test contamination by construction.
Each story is cut at a fixed point; the model continues for 1,024 tokens.

Metric, per story: `entity_report` retention (existing `eval_entities.py` —
prompt entities persisting in the continuation), TTT vs zero-update control,
paired by story. Guardrail: val loss on the canonical v2b set, TTT procedure
applied to val batches, must not regress beyond the Step-1 spread once known;
until then, reported without a bar.

## Falsification, registered now

- The LR=0 control must show no gain. If it does, the measurement is broken,
  not the mechanism validated.
- A shuffled-context variant (chunks in random order) must show no gain. If
  it does, the updates are doing something other than compressing narrative
  context.
- If per-story retention variance makes any reasonable threshold unreachable,
  Stage 2 reports "unmeasurable at this story count" and the story count —
  not the bar — is what changes.

## Cost and ordering

Implementation is CPU work. Measurement is inference-only GPU minutes, not
hours: 20 stories × 2 arms × ~1k tokens on a 58M model. It runs whenever GPU
is available and does not compete with Phase A training quota in any
meaningful amount. It must not borrow quota from Steps 0–2.

## What Stage 1 does NOT claim

No improvement threshold, no promotion, no model card line. Stage 1 output
is a baseline variance and a measured delta. Stage 2 (separate amendment)
sets the bar from that variance or declines to, in writing.
