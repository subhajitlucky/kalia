# KALIA owned-stack roadmap (2026-11-05)

**Goal.** A self-contained KALIA research artifact: its own tokenizer, its own
attention, its own architecture and training. No pretrained weights, no
distillation, no copying another model's inference.

## What "own" means

- PyTorch is a framework, not a model. Standard mathematical components
  (RMSNorm, SwiGLU, attention) are published ideas; our implementations are
  ours.
- The borrowed parts are the **GPT-2 BPE tokenizer** and any pretrained
  checkpoint. We remove the tokenizer; we never use pretrained weights.
- Novelty claims require two prior-art checks. If a mechanism exists, we cite
  it and do not call it new.

## Versioning

The line stays **v0.x** until the user explicitly declares a checkpoint KALIA
v1. Working scheme:

- **v0.3.1** — current public base.
- **v0.4.x** — edit-compiler capability packs (v0.4.0 X43, v0.4.1 X46).
- **v0.5.x** — K-Tok tokenizer experiments.
- **v0.6.x** — K-Attn attention experiments.
- **v0.7.x** — combined from-scratch base (K-Tok + winning K-Attn).
- **v1.0** — only by explicit user declaration.

Model-size scaling is separate. Stay at 58M until the evidence says
capacity-limited.

## Benchmark

The standard small-model zero-shot suite: ARC-Easy, ARC-Challenge, BoolQ,
HellaSwag, LAMBADA, OpenBookQA, PIQA, SciQ, WinoGrande (2,000 samples where
available), plus WikiText perplexity. Stage 0 is the v0.3.1 baseline.

## K-Tok v0

- Decode a sample of the training corpus (GPT-2 BPE is lossless) to raw text.
- Train byte-level BPE at **32k / 50k / 64k** vocabularies, with a digit-aware
  variant.
- Measure bytes/token, tokens/word, round-trip, and domain compression
  (English, code, numbers, names) against GPT-2 BPE.
- Report the tied-embedding parameter budget: 50,257 tokens × 512 ≈ 25.7M
  parameters, nearly half of the 58M model. A 32k vocabulary frees ~9M
  parameters at the same total budget.
- Pick a vocabulary only after the measurements. A tokenizer change requires a
  from-scratch model; v0.3.1 remains public.

## K-Attn

- Two prior-art probes before GPU.
- Three candidate designs, not one.
- Matched micro-ablation at 30M/58M; pre-registered bars.
- If none beats the QK-Norm/RoPE control, keep the control and record the null.
- Candidate directions: role/binding-aware attention, content-addressable
  memory attention, new relative-position schemes. Names follow the evidence.

## Roadmap

1. Stage 0 benchmark baseline (CPU, running).
2. K-Tok v0 (CPU).
3. 30-hour data experiment at 58M: reasoning-heavy mixture vs matched control.
4. K-Attn design + micro-ablation (next window).
5. v0.7.x combined from-scratch 58M with K-Tok + winning K-Attn.
6. Scale to 134M only if v0.7.x is capacity-limited.

## Modalities

Voice and images are separate owned modules, not consequences of more text
training: ASR/TTS for voice, a diffusion or image-token model for images. The
58M language core can be the brain of that system.

## HF gate

No base-model release on Hugging Face until the standard suite improves and the
user declares v1. Capability packs remain GitHub releases.
