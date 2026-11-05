# K-Tok v0 — own tokenizer

Status: **complete.** Four KALIA byte-level BPE tokenizers trained on a
50M-token corpus sample. All are lossless. **K-Tok 32k is the candidate** for
the first from-scratch A/B.

## Why this exists

The model still borrowed GPT-2's 50,257-token BPE. At 58M parameters the tied
embedding is **25.73M parameters** — nearly half the model. Replacing it with a
32k KALIA vocabulary frees **9.35M parameters** at the same total budget, and
makes the stack self-contained.

## Training

- Corpus: 50M GPT-2 tokens decoded from the v2b training shard (lossless).
- Held-out: 2M tokens from `val.bin`.
- Vocabularies: 32k, 50k, 64k byte-level BPE; plus a digit-aware 32k variant.
- Special token: `<|endoftext|>`.
- CPU only, no GPU.

## Result

Bytes per token: **higher is better** (fewer tokens for the same text).

| tokenizer | vocab | English B/tok | code B/tok | numbers B/tok | tied emb params | vs GPT-2 | lossless |
|---|---:|---:|---:|---:|---:|---:|---|
| GPT-2 BPE | 50,257 | 4.319 | 2.230 | **2.478** | 25.73M | — | yes |
| **K-Tok 32k** | 32,000 | **4.380** | **3.162** | 2.152 | **16.38M** | **+9.35M** | **yes** |
| K-Tok 50k | 50,000 | 4.459 | 3.262 | 2.320 | 25.60M | +0.13M | yes |
| K-Tok 64k | 64,000 | 4.493 | 3.320 | 2.411 | 32.77M | −7.04M | yes |
| K-Tok 32k digits | 32,000 | 4.297 | 3.114 | 1.049 | 16.38M | +9.35M | yes |

- **32k beats GPT-2 on English and code compression** and frees 9.35M
  parameters.
- **GPT-2 still wins on numbers** (2.478 vs 2.152 B/tok). The digit-aware
  variant is much worse on numbers (1.049 B/tok): atomic digits cost tokens.
  It may help arithmetic, but it is not the general-text candidate.
- **50k/64k are not worth it**: tiny compression gains, no parameter savings;
  64k costs 7.04M extra embedding parameters.

## Round-trip correction

The first report marked every K-Tok tokenizer `roundtrip: false`. That was a
measurement bug: `tokenizer.decode()` defaults to `skip_special_tokens=True`,
which drops `<|endoftext|>`. The verification kernel reloads the saved
tokenizers and checks with `skip_special_tokens=False`:

- 32k, 32k-digits, 50k, 64k: **all round-trip true** on held-out text and on an
  explicit `<|endoftext|>` sample.
- Raw report: `ktok_report.json` (roundtrip flag is the buggy default).
- Corrected: `ktok_verify.json`.

## Caveats

- Compression is measured on a 2MB held-out sample, one code file, and a
  synthetic number set. It is not a model benchmark.
- A tokenizer change requires a from-scratch model; the 58M weights cannot be
  reused.
- The tokenizer's effect on the 9-task suite is unknown until the A/B.

## Decision

**K-Tok 32k is the first owned-tokenizer candidate.** Next: a matched
tokenizer A/B — GPT-2 BPE vs K-Tok 32k — at 30M (2 seeds), then the winning
recipe at 58M. The public base remains v0.3.1.

## Artifacts

- `ktok_32k.json`, `ktok_50k.json`, `ktok_64k.json`,
  `ktok_32k_digits.json` — tokenizer files.
- `ktok_report.json` — original metrics.
- `ktok_verify.json` — corrected round-trip.
- `kalia-ktok-v0.log`, `kalia-ktok-verify.log`.
- Kernels: `subhajitlucky/kalia-k-tok-v0`,
  `subhajitlucky/kalia-k-tok-verify`.
