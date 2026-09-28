# Open-Source Landscape: Last Two Generations, and What Transfers to KALIA

Compiled 2026-09-28. Scope: the two most recent generations per organisation, and
the specific thing (if any) that transfers to a 58M-parameter model trained on
2× T4 free GPUs for ~30 GPU-hours a week.

**How to read the "adapt" column.** Four filters decide whether anything lands:

1. **Cost** — must fit 30 GPU-h/week, or be free at inference time only.
2. **Yardstick** — must not change the canonical held-out set (D43). Anything
   altering output distribution needs a re-baseline and is marked as such.
3. **D2** — from-scratch-only stays intact. No teacher weights, no distillation.
4. **Evidence** — must be a measured claim, not a launch number. Vendor
   self-reported benchmarks are labelled as such and do not count.

Priority codes: **P1** = candidate for the next version · **P2** = micro-ablation
queue · **P3** = read and cite, no action · **P4** = rejected, with reason.

---

## Tier 1 — small-model labs (our actual peer group)

### Hugging Face — SmolLM3 (3B), SmolLM2 (1.7B) · **P1**
The most useful document found in this entire search, because HF released the
*whole recipe* rather than the weights. Four transferable items:

- **WSD schedule (warmup-stable-decay)**: 2,000 warmup steps, constant, then
  linear decay to zero over the final 10%. Our config uses cosine with
  `min_lr_ratio: 0.1`, which never reaches zero — that is D4, and it is the
  single cheapest change available to us. Directly comparable to our D4.
- **NoPE**: remove rotary position embeddings from every 4th layer. Validated by
  their ablations on a 3B model against MHA. Freer than adding a mechanism.
- **Intra-document masking**: tokens from different documents in one packed
  sequence must not attend to each other (Llama 3 does this). **We pack
  sequences and do not mask across document boundaries** — this is a real
  correctness gap in our data pipeline, not a tuning knob. Highest-value item
  here.
- **Remove weight decay from embedding layers** (from OLMo 2, inherited by
  SmolLM3) for training stability.

Their stage structure: 0→8T stable, 8→10T introduce reasoning-dense, 10→11.1T
decay with math/code upsampled. That is our D1 in miniature — the shape
transfers even though the scale does not.

Also relevant: they found that upsampling long-context *data* (code repos,
books) did **not** further improve long-context benchmarks — NoPE plus longer
sequences was sufficient. That partly contradicts my earlier recommendation to
add fiction for LAMBADA, and it is a reason to test the data change and the
attention change separately rather than bundling them.

### Allen Institute for AI — OLMo 3 (32B, 7B), OLMo 2 (32B, 7B) · **P2**
The only lab releasing weights, data, code, logs, *and* intermediate
checkpoints. Their architectural choices worth noting: reordered-norm block,
QK-norm (we already have), 3:1 sliding-window/global attention mix, z-loss,
per-parameter weight-decay masking, skip-step optimizer.

Two transferable practices rather than tricks:
- **Per-parameter weight-decay masking** so embeddings can be excluded. We
  currently apply `weight_decay: 0.1` indiscriminately.
- **n-gram repetition filter** masking training instances with >32 repeated
  n-grams. Cheap data hygiene we do not currently do.

Their three-stage curriculum is the same shape as SmolLM3's and Qwen's: general
→ mid-training anneal on high-quality math/code/reasoning → long-context. Three
independent labs converging on the same structure is the strongest signal in
this document.

### Liquid AI — LFM-1.3B-Math, LFM2 series · **P3**
Notable negative result for us: *"pure reinforcement learning tends to be more
challenging"* for small models, and SFT from a strong teacher is what actually
works. That reinforces the D2 exclusion — RL is not a viable small-model path
without a teacher. They also report aggressive SFT hyperparameters working at
1.3B (lr 3e-4, 3 epochs, ~100B tokens) where gentler ones would not, because
large hard datasets need high LR to be absorbed. If we add a reasoning-dense
stage, this is the evidence for how hard to push it.

---

## Tier 2 — frontier labs (transfer is indirect but real)

### Alibaba Qwen — Qwen3.8 series, Qwen3.6-27B · **P1 (data) / P2 (arch)**
Two generations covered by `docs/research/2026-09-28-why-27b-beats-397b.md`.
Summary here: instance-level mixture optimisation via small-proxy ablations is
the transferable method (**D2** in that doc), and Gated Residual / 4-branch
residual stream is the architectural candidate (D6). Their own ablation found
widening the residual stream gave **+1.58 accuracy points at a 0.002 loss gap**
— which is the direct argument for D5 (screen on benchmarks, not loss alone).

### DeepSeek — V4 Preview / V4.1-Flash, V3.2 · **P3**
Multi-head Latent Attention and auxiliary-loss-free MoE load balancing. Both are
MoE-scale techniques with no dense-58M analogue. DeepSeek Sparse Attention is
the one idea that generalises, but it is a long-context mechanism and we are not
changing context length in v0.2.1. Recorded, not adopted.

### Mistral AI — Mistral Small 4, Large 3, Ministral 3 · **P3**
Sliding-window attention patterns and GQA. GQA specifically: SmolLM3's ablation
found GQA **matches MHA performance** while cutting KV cache size. At 58M our
inference is not the bottleneck, so there is no benefit here — but it is a free
non-regression if we ever add a second head configuration, and X2 in our backlog
already tests GQA. Close X2 with this citation rather than re-deriving it.

### Google DeepMind — Gemma 4, Gemma 3 · **P2**
Gemma 4 moved to **Apache 2.0**, dropping the Prohibited Use Policy — so their
data-curation recipes and documentation are now freely usable for a
compliance-clean project like ours. That is a licence-level change, not a
technique, and it matters for the compliance constraint we just enforced (v2b
rebuild). Sliding-window attention with 5:1 local/global interleaving is a
candidate for the attention work implied by our LAMBADA weakness.

### NVIDIA — Nemotron 3 Ultra (561B), Nemotron 3 Super (124B) · **P4 at our scale**
Hybrid Mamba-2 + Transformer + MoE. Mamba-style state-space layers are a genuine
architectural idea for long-range recall at linear cost, and the most credible
route to fixing LAMBADA without doubling context. **Rejected for now**: a
hybrid SSM/attention stack at 58M is a large architectural change with real
instability risk (Hymba-1.5B's 53.75% vs Qwen2.5-1.5B's 70% on GSM8K is the
warning — novel architecture plus weak training loses). If D7 (reverse-aware
attention) is pre-registered and fails, a *minimal* linear-attention layer is
the honest next hypothesis, not a full Mamba hybrid.

### Meta — Muse Glimmer (30B), Llama 4 · **P3**
The 30B Apache 2.0 release is small enough to read end-to-end, which makes it
the most tractable "how does a well-run small model do it" reference available.
Llama 4 is MoE-only and out of scale.

### OpenAI — gpt-oss-120b, gpt-oss-20b · **P4**
MoE with adjustable reasoning effort. The *reasoning-effort* dial is the
interesting transferable idea: it is test-time compute made explicit and
controllable, which is the same mechanism as our D3 (best-of-N with
self-consistency). Their weights are not a training path for us.

### Moonshot AI — Kimi K2.5, Kimi K2 · **P3**
1T/32B MoE. Their WSD adoption and long-context recipe are already captured via
SmolLM3. Nothing further.

### Zhipu AI — GLM 5.3, GLM-5 · **P3**
Uses DeepSeek Sparse Attention; 128K single-pass output. Out of scale.

### MiniMax — M3, M2.5 · **P3**
MiniMax Sparse Attention (KV-block selection) — same family as DSA, same
conclusion as DeepSeek. Not adopted.

### Thinking Machines Lab — Inkling (975B), Inkling Small · **P4**
Per Hugging Face's own analysis, Inkling is built on Chinese model artifacts.
Frontier scale, MoE. Nothing transfers.

### Poolside — Laguna S 2.1, XS 2.1 · **P3**
Code-specialised MoE (118B/8B). Relevant only as evidence that code-heavy
mixture emphasis is a current frontier priority, which supports D1's
reasoning-dense stage. No architectural transfer.

### Cohere — North Mini Code (30B) · **P3**
30B with a small active footprint and local-hardware latency claims. Worth
reading for inference-efficiency design if v0.3.0's test-time compute makes
latency a concern.

### Xiaomi — MiMo-V2.6-Pro/Flash · **P4**
Distill variants of Qwen base. No original technique published that we can use.

### Dots Studio — Dots3-Note Preview (280B/16B) · **P4**
MoE, agentic focus. No small-model technique published.

### inclusionAI — Ling 3.0 Flash · **P4**
Flash model; no recipe published.

### Tencent — Hy4 preview, Hunyuan · **P4**
No published technique we can use; huge download volume, no methodological
detail.

### StepFun — Step 3.7 Flash · **P4**
Same.

### AMD — Radeon-optimised open series · **P3 (tooling)**
Third most prolific open publisher of 2026. Relevant to us only in that
hardware-vendor-published models are tuned for specific hardware — a reminder
that our T4 constraint is a first-class design input, not an afterthought.

### Microsoft — Phi-4, Phi-4-mini · **P2**
Phi line's whole thesis is that data quality dominates parameter count at small
sizes — which is exactly our situation and exactly the D1 argument, arrived at
independently. Concrete: the "textbook-quality" synthetic data approach
(Cosmopedia, already 15% of our mix) is a Phi-line idea that we have been
running without knowing its provenance. Worth reading their data-methods writeup
before redesigning the mixture.

---

## Tier 3 — smaller and adjacent

### TII — Falcon 3, Falcon-H1R · **P3**
Hybrid Mamba-2/Transformer again, same conclusion as NVIDIA. Falcon 3's
technical report is unusually explicit about training-data composition per
domain.

### Nous Research — Hermes 4 · **P4**
Post-training of an existing base; no training technique published.

### 01.AI (Yi) — Yi-Lightning · **P4**
Efficient-attention claims without published ablation detail.

### Sakana AI — Evo 2, Transformer² · **P2 (concept)**
Evolutionary model merging: train small specialists, merge into a larger model.
Costs no additional training compute. This is the closest published analogue to
what a 58M project could do to increase effective capacity, and it is the one
idea on this list that is neither distillation nor MoE. If we ever train a
second arm (say v0.2.1 with a different mixture), **merging those two 58M
checkpoints is a zero-GPU-cost experiment** that could give a genuine capacity
increase. Filed as a candidate, not a plan.

### LightOn — Falcon-Reasoning, Alps · **P4**
No published technique.

### Kyutai — Kyutai-models (Moshi) · **P4**
Speech; out of scope.

### Prime Intellect — INTELLECT-3 · **P3 (negative)**
Their RL environments are for models far above 58M. Combined with Liquid's
finding that RL is hard for small models, this closes the RL door for us.

---

## What survived, ranked

| Rank | Item | From | Why it survives the four filters |
|---|---|---|---|
| 1 | **Intra-document masking** | SmolLM3 / Llama 3 | We pack sequences without masking; this is a correctness bug, not a tweak. Free. |
| 2 | **WSD schedule to zero LR** (D4) | SmolLM3, OLMo 3, Kimi K2 | Three labs converged; our cosine never reaches zero; directly addresses the weak J2 evidence in D41 |
| 3 | **Three-stage mixture** (D1) | Qwen, SmolLM3, OLMo 3 | Three independent convergences; attacks the binding constraint (underfitting) |
| 4 | **Screen on benchmarks, not loss alone** (D5) | Qwen3.8 | Our entire selection history optimises loss; documented case of +1.58 accuracy at 0.002 loss gap |
| 5 | **NoPE on every 4th layer** | SmolLM3 | Validated ablation, free, helps long-range which is our weakness |
| 6 | **n-gram repetition filter** | OLMo 3 | Cheap data hygiene against synthetic-corpus degeneracy |
| 7 | **Per-parameter weight-decay masking** | OLMo 3 | We decay embeddings indiscriminately |
| 8 | **Checkpoint merging** | Sakana AI | Zero GPU cost; only useful once a second arm exists |
| 9 | **Gated Residual 4-branch** (D6) | Qwen3.8 | Real but architectural; micro-ablation only |
| 10 | **Reverse-aware attention** (D7) | our own analysis | The honest novel bet; pre-register before running |

Rejected with reasons: MoE upcycling (loses to its own dense base at our scale),
MoE/SSM hybrids (instability risk at 58M, and the one datapoint we have is
negative), RL (scale floor plus no teacher), distillation (D2), n-gram embedding
tables (breaks the yardstick), and — newly — **bundling the fiction data with an
attention change**, because SmolLM3 found the data half of that story did not
work on its own. Test them apart.

## The uncomfortable finding

Of thirty organisations, **four** published a technique we should adopt
unconditionally, and all four are from the small-model tier. The frontier tier
published impressive results using compute we do not have, techniques that do
not apply below ~100B, or both. That is not a failure of the research — it is
the clearest available evidence that KALIA's constraint is not a handicap to be
engineered around. It is the actual research question, and the peer group to be
measured against is Doge-60M, Navdyut-60M, G1-nano and SmolLM3 at the small end,
not Qwen3.8.
