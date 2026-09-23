# Punching Above Weight — How a 58M Model Can Overperform

Status: active strategy
Date: 2026-09-23
Sources: MobileLLM (ICML 2024), MobileLLM-R1 (ICLR 2026), SmolLM2 (2025),
Gemma 3 270M (2025), Scaling Data-Constrained LMs (NeurIPS 2023), RegMix (2024),
MiniLLM/GKD/OKD distillation literature (2024–2026).

## The hard truth first

Modern sub-1B models are trained on **thousands of tokens per parameter**:

| Model | Params | Tokens | Tokens/param |
|---|---|---|---|
| SmolLM2-135M | 135M | 2.0T | ~14,800 |
| SmolLM2-360M | 360M | 4.0T | ~11,100 |
| **Gemma 3 270M** | 268M | **6.0T** | ~22,400 |
| MobileLLM-R1-140M | 140M | 4.2T | ~30,000 |
| **KALIA v0.1.2 (current)** | 58M | 2.5B | **~43** |

We are 300–700× below the token-per-parameter regime of state-of-the-art small
models. **We cannot match a 1B general model with 18 GPU-hours/week.** Any claim
otherwise would be dishonest. What we *can* do is maximize capability per token
and per parameter, and be excellent inside our domain.

## Six levers, ranked by expected gain per cost

1. **Token efficiency of data (2× effective tokens for free).**
   Muennighoff et al. found mixing ~10% Python into English training gives a
   **2× increase in effective tokens even on natural-language tasks**. We
   already added 5% code in the v2 mixture; consider raising to 10%.
   Also: up to 4 epochs of repeated data is nearly as good as fresh data
   (half-life ~15 epochs) — a free multiplier when data runs out.

2. **Architecture at sub-billion scale.**
   MobileLLM: **depth beats width** (30–42 layers at 125M outperform 12-layer
   models), embedding sharing (have it), **GQA with ~4 KV heads at parity**
   while shrinking the model, and **immediate block-wise layer sharing**
   (repeat each block) adding ~1.1% accuracy at zero parameter cost.
   MobileLLM-R1-140M ships 15 layers × 576 dim × 3 KV heads.

3. **Data quality per token.** Cosmopedia-style synthetic textbooks; educational
   filtering; curriculum with high-quality math/code in the annealing phase
   (SmolLM2, MobileLLM-R1 both upweight high-quality data late). MobileLLM-R1
   matches Qwen3-0.6B with 11.7% of its tokens purely via curation.

4. **Recipe** (already largely adopted): Muon+ optimizer, QK-Norm, soft-cap,
   WSD + 20% decay, EMA, target-loss stop. SmolLM2's small models used
   **lr 3e-3** with WSD — much higher than our 0.06 Muon-equivalent; worth a
   sweep after the current one.

5. **Distillation — REJECTED (decision D21, 2026-09-23).**
   Technically the biggest lever, but it transfers a teacher's distribution into
   the student: borrowed capability. KALIA's identity is "every weight is ours,
   the recipe is auditable." The capability ceiling this implies is accepted
   knowingly. No KALIA-D will be built unless that decision is explicitly
   revisited and labelled.

6. **Quantization-aware training + deployment.** Gemma 3 270M ships QAT INT4
   checkpoints for laptop/phone inference. Applicable to KALIA's final stage
   for fast local use.

## Not now (low value at 58M / our budget)

- MoE (needs scale to pay off), MLA (for long-context KV cache), FP8 (no T4
  hardware support), vocabulary retraining (breaks weight tying and tokenizer
  continuity; revisit only for a from-scratch successor).

## Experiment queue (micro-ablation first)

| ID | Experiment | Hypothesis from | Cost | Status |
|---|---|---|---|---|
| X1 | Deep-thin: 12L×384d vs 6L×512d (equal params) | MobileLLM depth>width | 1.5h GPU | queued |
| X2 | GQA: 2 KV heads vs full MHA (equal params) | MobileLLM GQA parity | 1.5h GPU | queued |
| X3 | Block sharing: repeat each block twice | MobileLLM-LS +1.1% | 1.5h GPU | queued |
| X4 | Code ratio 5% → 10% in the mixture | 2× effective tokens (Muennighoff) | 1h GPU | queued |
| X5 | LR retune at WSD (post current sweep) | SmolLM2 small models lr 3e-3 | 1h GPU | queued |
| X6 | Distillation probe: 58M student from SmolLM2-1.7B teacher | MiniLLM/OKD | 3–4h GPU | **rejected (D21)** |

## Implications for v0.2.0

- Keep 58M params for now; adopt architecture winners from X1–X3.
- Raise code share to ~10% if X4 confirms.
- Add WSD schedule + EMA + target-loss stop (implemented).
- Two-epoch training of the v2 mix is *permissible* (≤4 epochs rule) and is the
  cheapest way to add tokens once the unique data is exhausted.
- Decide on KALIA-D (distillation) explicitly, and label it honestly if built.
