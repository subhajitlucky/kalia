# KALIA v0.3.1 weight audit

Date: 2026-11-04. Method: downloaded `kalia-lm/kalia-v031/model.safetensors`,
loaded it into `GPT`, and ran weight/activation probes on eight held-out story
prompts (256 tokens each). This is a **measurement audit**, not native
weight-reading. I can inspect tensors and run probes; I cannot “see” semantics
in 57.8M numbers.

## What is healthy

- **No NaN/Inf** in any tensor.
- **No dead rows** in any attention/MLP matrix; no zero-norm tensors.
- **No rank collapse**: effective rank is 456–481 for QKV/MLP, 376–456 for
  attention output projections (out of 512). The X38 rank-1.5 collapse is not
  present in the base model.
- **MLP neurons all active**: 0 dead / 0 low-activity neurons at every layer.
- **Attention heads healthy**: entropy 2.2–4.0 bits, max attention 0.10–0.59;
  one previous-token head (L05.0, 0.58 previous-token mass); no collapsed or
  uniform-dead heads.
- **Embeddings healthy**: token norms 0.74–1.71, mean 1.14; no zero or outlier
  rows.
- **RMSNorm/QK-Norm weights**: all positive; no near-zero layer.

## Findings

### 1. Release packaging: duplicated tied embedding

The safetensors has **83,587,840 tensor parameters** and 93 tensors; the model
is **57,856,256** parameters with tied embeddings. The difference is exactly
50,257 × 512: `tok_emb.weight` and `lm_head.weight` are both stored and
identical. The file is 334 MB instead of ~231 MB.

- Not a model-quality bug; the model card’s parameter count is correct.
- A loader that does not implement tying will double-count the embedding.
- Fix/export note: drop `lm_head.weight` from the safetensors or document the
  duplicate. This is the clearest concrete issue.

### 2. Activation scale is high, but not a merge bug

Residual-stream norms (mean / max over the probe batch):

| layer | v0.1.2 | v0.2.0 | v0.3.1 |
|---|---:|---:|---:|
| 00 | 241 / 713 | 474 / 2,014 | 500 / 1,969 |
| 05 | 730 / 6,972 | 1,728 / 16,439 | 1,946 / 20,216 |
| 09 | 2,037 / 3,318 | 4,933 / 7,990 | 5,690 / 9,484 |

v0.3.1 is **2.3–2.8× v0.1.2**, but only **10–15% above v0.2.0**. The scale
comes from the v0.2.0 training regime, not the 0.3/0.7 merge. The merge adds a
small amount.

- Half-precision forward is finite: max hidden element **615**, no Inf/NaN.
- The high aggregate norms are a **quantization/outlier risk**, not a quality
  bug. They do not indicate a dead model or a merge failure.

### 3. Soft-cap is still on

The config has `logit_softcap = 30.0`. D55 says the soft-cap is not supported
for future configs; v0.3.1 kept it for continuity. This is a **config-level
known issue**, not a weight anomaly. A cheap diagnostic: re-score the same
checkpoint with soft-cap 0 vs 30; do not tune on val.

### 4. Greedy decoding loops

At `temperature=0.01, top_k=1`, some prompts loop or emit spaces (4-gram repeat
0.51–0.98). At the model card’s sampling setting (`temperature=0.8, top_k=50`),
repetition is mild (4-gram repeat 0.08–0.11). This is a **small-model decoding
behavior**, not a proven single weight defect. The blog’s Qwen case had a
specific loop direction; I do not see one from weights alone.

## What this means for weight-space programming

- There is **no catastrophic single defect** in the weights — no dead heads, no
  rank collapse, no overflow.
- The concrete targets are **packaging** (duplicate tied embedding),
  **config** (soft-cap), and **activation-scale/quantization** (outlier
  risk).
- The main capability failure — compositional binding — is **not visible as a
  dead unit or collapsed head**. It looks like a missing ability, not a single
  fixable loop. The blog’s Gemma control is the warning: weight editing removed
  an obstacle in Qwen, but did not add ability in Gemma.

If we do a weight-space experiment, it should be a **diagnostic**: fix the
duplicate export and test soft-cap 0; do not claim a new mechanism, and do not
steer edits with val/2k.
