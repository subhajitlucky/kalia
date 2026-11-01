# v0.3.1 release gate — PASSED

Run 2026-11-01 on Kaggle CPU. Zero GPU.

## The artifact

The release is the **30% v0.2.0 / 70% v0.3.0 weight merge**, rebuilt from the
hash-pinned source checkpoints:

- v0.2.0 `924ad52b27017bb823e925df6695d1becdef032b772ae8436eb0b74bdbac6718`
- v0.3.0 `80b064f9a99e6ed5372e7fa46021574a9b21298b822fd2c581357831cc77537e`
- merged checkpoint `f59e059ce673d15601dc637940c713731fa77ca874202804395847ed9d415382`

The merged checkpoint re-scores **2.7579 / bpB 0.9016** on the canonical
100-batch protocol (819,200 tokens, seed 1234).

## The export gate

`tools/export_hf_model.py --release v031` wrote **93 tensors** to
`model.safetensors` and then loaded **those written tensors** back through the
repo's own model and re-scored them:

```text
verification on the written tensors: val 2.7579 / bpB 0.9016 over 819,200 tokens
OK: the exported artifact reproduces the registered number.
```

- `model.safetensors`: **334,360,208 bytes**, sha256
  `bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468`.
- Config: 10 layers × 512, context 1,024, QK-Norm, soft-cap 30.
- 93 tensors, including the de-shared tied head.

## The upload

- Repository: **`kalia-lm/kalia-v031`** (private), 15 files.
- Card: `README.md` with three static charts
  (`charts/v031_held_out_loss.png`, `charts/v031_zero_shot_deltas.png`,
  `charts/v031_cl0.png`).
- Weights, config, generation config, `model.py`, `sample.py`, `LICENSE`, and
  the GPT-2 tokenizer files.
- **v0.1.2 and v0.2.0 model cards are untouched.** They remain frozen historical
  releases; v0.3.1 is a new repository.

## Results

| metric | v0.2.0 | v0.3.0 | **v0.3.1** |
|---|---:|---:|---:|
| val | 2.8248 | 2.7743 | **2.7579** |
| bpB | 0.9244 | 0.9070 | **0.9016** |
| CL-0 | 2.798875 | 2.756648 | **2.740949** |

2k deltas vs v0.2.0: ARC +1.60, HellaSwag +2.05, LAMBADA +3.10, PIQA +0.60,
WinoGrande +1.50 pp. No task regresses by >2σ. The gains are modest; the val and
CL-0 are the strong parts.

## Limitations

- Weight-space soup, not a new architecture.
- 58M parameters, 1,024 context; not a frontier model.
- 2k gains are within ~1σ–1.75σ.
- No instruction tuning; English web/stories/code mixture.
- The merge does not fix the root cause (best-checkpoint saving); it recovers
  part of a final-checkpoint regression at zero GPU.
