# X48 — tokenizer A/B: GPT-2 BPE vs K-Tok 32k

Status: **data ready; first 30M arm running.** Pre-registered H-1…H-5.

## Matched data

The same 312M GPT-2 tokens of the v2b training shard were used for both arms:

| arm | train tokens | val tokens | train bytes |
|---|---:|---:|---:|
| GPT-2 BPE | 312,000,000 | 10,000,000 | 624,000,000 |
| K-Tok 32k | 309,298,735 | 9,925,458 | 618,597,470 |

K-Tok compresses the same text into ~0.8% fewer tokens. The K-Tok arm is
therefore slightly smaller (32k tied embedding) at the same transformer body.

## Design

- 30M parameters: 6 layers, 384 embd, 6 heads, context 512.
- 4,770 steps × 16 micro × 8 accum × 512 ≈ 312M tokens.
- AdamW, LR 6e-4, warmup 40, cosine.
- Two seeds: 1338/1339.
- Evaluation: 9-task suite at 2k, separate CPU kernel.

## Bars

| ID | prediction | threshold |
|---|---|---|
| H-1 | K-Tok beats GPT-2 on the 9-task average | ≥ +1.0 pp, both seeds |
| H-2 | no task regresses | no task beyond 2σ |
| H-3 | tokenizer efficiency | K-Tok bpB ≤ GPT-2 bpB |
| H-4 | lossless | round-trip true |
| H-5 | parameter budget reported | total and tied embedding |

A +1.0 pp average at 30M is the signal to train the 58M base with the winner;
a null keeps GPT-2 BPE.

## Artifacts

- `data_manifest.json`, `kalia-x48-data.log`.
- Pre-registration: `docs/preregistrations/2026-11-05-X48-tokenizer-ab.md`
  (hash `0c7a054c…`).
- Kernels: `subhajitlucky/kalia-x48-data`, `subhajitlucky/kalia-x48-gpt2-s1338`.
