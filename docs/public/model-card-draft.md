# KALIA v0.1.2 — Model Card

> Final numbers filled 2026-09-23. Publish with the repository milestone.

## Model description

**KALIA** is a 57.9M-parameter decoder-only language model trained **from
scratch** on 1.82B tokens of English text. No pretrained weights, no
distillation, no fine-tuning of another model: every weight was learned by this
project's own training run on free-tier Kaggle T4 GPUs.

- Developed by: subhajitlucky (GitHub/Kaggle)
- Organization: `kalia-lm` (HuggingFace)
- Model type: decoder-only transformer (RMSNorm, SwiGLU, RoPE, QK-Norm, logit
  soft-capping, weight-tied embeddings)
- License (weights): Apache-2.0
- Repository: github.com/subhajitlucky/kalia

## Architecture

| Field | Value |
|---|---|
| Parameters | 57,856,256 |
| Layers / heads | 10 / 8 (512-dim) |
| Context | 1,024 tokens |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Optimizer | Muon (Newton–Schulz orthogonalization) on hidden matrices; AdamW for embeddings/head/1D |
| Precision | fp16 with gradient scaling, DDP across 2×T4 |

Muon+ (col-row normalization after the orthogonalization step) was validated in
a 30M-parameter micro-ablation at matched tokens (3.4941 vs 3.5091, better on
7/7 checkpoints) but fell below the project's pre-set 0.02-nat promotion
threshold, so v0.1.2 ships plain Muon. Muon+ remains an optional flag.

## Training

- Data: TinyStories (~500M tokens) + FineWeb-Edu (dedup, ~2B tokens) — GPT-2
  BPE, uint16 shards
- Schedule: cosine with 500-step warmup; 4,770 steps planned
- **Stopped at step 3,478 (73% of the schedule)** when the weekly GPU quota was
  exhausted. Validation loss had been flat within ±0.1 (eval noise) for 1,000
  steps, and a deterministic probe set showed no further gain, so the stop was
  declared final under the project's pre-registered stopping discipline rather
  than resumed.
- No EMA readout; no target-loss stopping (both are v0.2.0 candidates)
- Hardware: free-tier Kaggle (2× NVIDIA T4), resumable across sessions with
  checkpoints synced to the HuggingFace Hub

## Evaluation

| Metric | v0.1.0 baseline (AdamW) | v0.1.2 (Muon) |
|---|---|---|
| Final validation loss | 3.2702 @ 2,250 steps | **3.2214 @ 1,730 steps** (overtook the baseline's final loss with ~23% fewer tokens) |
| Last validation eval | — | 2.3986 @ 3,000 steps (50×8×1024-token batches; ±0.1 noise) |
| Probe held-out loss (20 fixed sentences) | — | **3.2303** @ step 3,478 |
| bits-per-byte (probe) | — | **0.9415** |
| Abhimanyu gap (reverse − forward NLL) | — | **6.057 nats** (forward 3.2303 vs reversed-text 9.2875; random ≈ 10.8) |

Ablations (30M-param micro-runs, equal tokens, same seed): AdamW 3.8041 →
Muon 3.5937 → Muon + QK-Norm + soft-cap 3.5103. LR sweep: 0.02 optimal. Full
logs, pre-registrations, and the incident journal are in the repository.

## Usage

```python
# generate text from a checkpoint (CPU works; model is ~230MB in fp16)
python sample.py --ckpt ckpt.pt --prompt "Once upon a time"
```

## Sample generations

See `docs/eval/2026-09-23-v012-step3478-probes.md` for 27 unedited generations
with the checkpoint step and loss each was generated from.

## Limitations

- Small model: coherent short-form English, not a general assistant.
- Domain-narrow: trained largely on children's stories and educational text.
- Entity consistency degrades over long outputs (e.g. names drift within a
  paragraph); no instruction-following in this version (planned post-training
  stage).
- Training stopped at 73% of the cosine schedule (quota; plateau documented).
- Outputs may be inaccurate or biased; not suitable for production decisions.

## Training data attribution

v0.1.2 used:

| Dataset | License | Link |
|---|---|---|
| TinyStories | CDLA-Sharing-1.0 | roneneldan/TinyStories |
| FineWeb-Edu (dedup, SmolLM corpus) | ODC-By-1.0 | HuggingFaceTB/smollm-corpus |
| GPT-2 BPE tokenizer | MIT | openai/tiktoken |

Planned for v0.2.0 (not used by this model): Cosmopedia v2 (ODC-By) and a
permissively licensed Python subset (per-file MIT/Apache/BSD/ISC/Unlicense/CC0).

Datasets are **not** redistributed with this model. This model card attributes
them as required by their licenses.

## Disclaimer

KALIA is a personal research project. It is not affiliated with any government
scheme, company, or other project using a similar name.
