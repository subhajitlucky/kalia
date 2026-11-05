# X47 Stage 0 — v0.3.1 standard small-model suite

Status: **complete.** Kaggle CPU, 2,000 samples where available. This is the
baseline for the next base-model data experiment.

| task | v0.3.1 acc | stderr |
|---|---:|---:|
| ARC-Easy | 0.4800 | 0.0112 |
| ARC-Challenge | 0.2014 | 0.0117 |
| BoolQ | 0.5435 | 0.0111 |
| HellaSwag | 0.3235 | 0.0105 |
| LAMBADA | 0.2700 | 0.0099 |
| OpenBookQA | 0.1640 | 0.0166 |
| PIQA | 0.6197 | 0.0113 |
| SciQ | 0.7080 | 0.0144 |
| WinoGrande | 0.5225 | 0.0140 |
| **mean** | **0.4258** | — |

## Weak spots

The four weakest tasks are **OpenBookQA 0.164**, **ARC-Challenge 0.201**,
**LAMBADA 0.270**, **HellaSwag 0.324**. These are knowledge, multi-step
reasoning, and long-range recall — not edit application.

The strong tasks are SciQ, PIQA, BoolQ, WinoGrande, ARC-Easy. A data mixture
that improves the weak four without regressing the strong five is the target
for the 30-hour base experiment.

## Artifacts

- `kalia-x47-suite.log` — full kernel log and per-task metrics.
- Kernel: `subhajitlucky/kalia-x47-suite`.
