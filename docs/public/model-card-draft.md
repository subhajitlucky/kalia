# KALIA — Model Card (DRAFT, publish after v0.1.2 completes)

> Placeholders in [brackets] are filled with final numbers at release.

## Model description

**KALIA** is a 58M-parameter decoder-only language model trained **from scratch**
on ~[2.4]B tokens of English text. No pretrained weights, no distillation, no
fine-tuning of another model: every weight was learned by this project's own
training run on free-tier Kaggle T4 GPUs.

- Developed by: [public name/handle]
- Organization: `kalia-lm` (HuggingFace)
- Model type: decoder-only transformer (RMSNorm, SwiGLU, RoPE, QK-Norm, logit
  soft-capping, weight-tied embeddings)
- License (weights): Apache-2.0
- Repository: [GitHub URL]

## Architecture

| Field | Value |
|---|---|
| Parameters | ~57.9M |
| Layers / heads | 10 / 8 (512-dim) |
| Context | 1,024 tokens |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Optimizer | Muon on hidden matrices (col-row normalized, Muon+), AdamW (embeddings/head/1D) |
| Precision | fp16 with gradient scaling, DDP across 2×T4 |

## Training

- Data: SmolLM-corpus FineWeb-Edu (dedup) 60%, TinyStories 20%, Cosmopedia v2
  15%, permissively-licensed Python 5% — tokenized with GPT-2 BPE
- Schedule: cosine with warmup; target-loss stopping with a short decay phase
- Checkpoint EMA applied for the final readout
- Hardware: free-tier Kaggle (2× NVIDIA T4), resumable across sessions with
  checkpoints synced to the HuggingFace Hub

## Evaluation

| Metric | Baseline (AdamW) | KALIA v0.1.2 (Muon+) |
|---|---|---|
| Validation loss @ equal tokens | [3.2x] | [3.1x] |
| Validation loss at baseline's stop | [3.2702 @ 2,250 steps] | [reached ~23% earlier] |
| bits-per-byte | [TBD] | [TBD] |

Ablations (30M-param micro-runs, equal tokens, same seed): AdamW [3.8041] →
Muon [3.5937] → Muon + QK-Norm + soft-cap [3.5103]. Full logs and the incident
journal are in the repository.

## Usage

```python
# generate text from a checkpoint (CPU works; model is ~230MB in fp16)
python sample.py --ckpt ckpt.pt --prompt "Once upon a time"
```

## Sample generations

See `docs/samples/` for dated, unedited samples with the checkpoint step and
loss each was generated from.

## Limitations

- Small model: coherent short-form English, not a general assistant.
- Domain-narrow: trained largely on children's stories and educational text.
- Entity consistency degrades over long outputs; no instruction-following in
  this version (planned post-training stage).
- Outputs may be inaccurate or biased; not suitable for production decisions.

## Training data attribution

| Dataset | License | Link |
|---|---|---|
| TinyStories | CDLA-Sharing-1.0 | roneneldan/TinyStories |
| FineWeb-Edu (dedup, SmolLM corpus) | ODC-By-1.0 | HuggingFaceTB/smollm-corpus |
| Cosmopedia v2 | ODC-By-1.0 | HuggingFaceTB/smollm-corpus |
| CodeParrot-Clean (permissive subset) | per-file, filtered to MIT/Apache/BSD/ISC/Unlicense/CC0 | codeparrot/codeparrot-clean |
| GPT-2 BPE tokenizer | MIT | openai/tiktoken |

Datasets are **not** redistributed with this model. This model card attributes
them as required by their licenses.

## Disclaimer

KALIA is a personal research project. It is not affiliated with any government
scheme, company, or other project using a similar name.
