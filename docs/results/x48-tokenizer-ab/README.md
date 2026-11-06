# X48 — tokenizer A/B: GPT-2 BPE vs K-Tok 32k

Status: **complete — H-1 and H-3 fail. GPT-2 BPE stays; K-Tok is parked.**

## Design

- 30M parameters: 6 layers, 384 embd, 6 heads, context 512.
- 4,770 steps × 16 micro × 8 accum × 512 ≈ 312M tokens.
- Same underlying text: 312.0M GPT-2 tokens vs 309.3M K-Tok tokens.
- Seed 1338 completed for both arms; the screen stopped there because the
  first matched pair is already below the +1.0 pp bar and the val bpB is worse.
- Evaluation: 9-task suite at 2k where available.

## Result

| task | GPT-2 BPE | K-Tok 32k | Δ (pp) |
|---|---:|---:|---:|
| ARC-Easy | 0.3920 | 0.3840 | −0.80 |
| ARC-Challenge | 0.1741 | 0.1596 | −1.45 |
| BoolQ | 0.4690 | 0.4595 | −0.95 |
| HellaSwag | 0.3010 | 0.2995 | −0.15 |
| **LAMBADA** | 0.1275 | **0.1660** | **+3.85** |
| OpenBookQA | 0.1560 | 0.1280 | −2.80 |
| PIQA | 0.5686 | 0.5653 | −0.33 |
| SciQ | 0.5810 | 0.5800 | −0.10 |
| WinoGrande | 0.5217 | 0.5241 | +0.24 |
| **mean** | **0.3657** | **0.3629** | **−0.28** |

- **H-1 FAIL**: K-Tok does not beat GPT-2 by +1.0 pp.
- **H-2 PASS**: no task regresses beyond 2σ.
- **H-3 FAIL**: K-Tok total val nats are ~4.5% worse over the same text.
- **H-4/H-5 PASS**: both tokenizers lossless; parameter budget reported.

LAMBADA improves significantly (+3.85 pp, ~3.4σ), but OpenBookQA and
ARC-Challenge lose more. The 32k vocabulary is smaller and faster (39k vs
26k tok/s), but it is not a better benchmark tokenizer at 30M.

## Decision

**Keep GPT-2 BPE. Park K-Tok. Do not spend a 58M run on this tokenizer.**
The owned-tokenizer goal remains open for a different design later; this
32k byte-level BPE is not the win. The next 30-hour window goes to the
reasoning-heavy data experiment on the weak tasks.

## Artifacts

- `gpt2_1338/` — training log, eval log, bench, manifest.
- `ktok_1338/` — training log, eval log, bench.
- `data_manifest.json`, `kalia-x48-data.log`.
- Pre-registration: `docs/preregistrations/2026-11-05-X48-tokenizer-ab.md`
  (hash `0c7a054c…`).
