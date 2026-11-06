# X48 K-Tok 32k arm — seed 1338

Status: training complete; 9-task eval running.

- 4,770 steps, 30M parameters (6 layers, 384 embd, 6 heads, context 512).
- Final val loss: **3.3706** nats/token (step 4750).
- Checkpoint: 274,985,229 bytes, sha256 `07a12913d0033c87025479f93c659b5ba5a8b3091b46f157ea8d8e4300b3d4ec`.
- Data: `train_ktok.bin`, 309.3M tokens (same text as the GPT-2 arm).
- Training throughput: ~39k tokens/s vs ~26k for the GPT-2 arm; the K-Tok
  model is smaller (32k tied embedding).

## Val comparison caveat

Val loss per token is **not** directly comparable across tokenizers. The same
text has 10.0M GPT-2 tokens and 9.925M K-Tok tokens. Total val nats over the
same text: GPT-2 ~32.02M, K-Tok ~33.45M, so GPT-2 is better on val bpB by
about 4.5%. The benchmark suite is the deciding metric.

## Artifacts

- `train.log` — full training log.

## 9-task eval

| task | acc |
|---|---:|
| SciQ | 0.5800 |
| PIQA | 0.5653 |
| WinoGrande | 0.5241 |
| BoolQ | 0.4595 |
| ARC-Easy | 0.3840 |
| HellaSwag | 0.2995 |
| LAMBADA | 0.1660 |
| ARC-Challenge | 0.1596 |
| OpenBookQA | 0.1280 |
| **mean** | **0.3629** |

Compared with the GPT-2 arm (0.3657), K-Tok is **−0.28 pp** on the average.
LAMBADA is a real win (+3.85 pp), but OpenBookQA (−2.80) and ARC-Challenge
(−1.45) lose more. H-1 and H-3 fail.
