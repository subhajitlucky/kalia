# Corpus shards — licences and attribution

Published with KALIA v0.2.0. Each file is the source text tokenized with GPT-2
BPE into uint16, with the project's EOS document separator inserted between
documents. Each source's proportional validation head (6M/2M/1.5M/0.5M tokens)
was skipped at build time, so the canonical held-out set cannot appear here.
`manifest.json` records token counts, mixture ratios and each shard's SHA-256.

| File | Tokens | Source | Licence | Attribution |
|---|---|---|---|---|
| `fineweb.bin` | 124,500,691 | FineWeb-Edu dedup via `HuggingFaceTB/smollm-corpus` | ODC-By-1.0 | `HuggingFaceTB/smollm-corpus` |
| `tinystories.bin` | 124,500,691 | `roneneldan/TinyStories` | CDLA-Sharing-1.0 | `roneneldan/TinyStories`; republished under the unmodified CDLA-Sharing-1.0 terms |
| `cosmopedia.bin` | 124,500,691 | Cosmopedia v2 via `HuggingFaceTB/smollm-corpus` | Apache-2.0 (content card, `HuggingFaceTB/cosmopedia`); container tagged ODC-By | `HuggingFaceTB/cosmopedia` and `HuggingFaceTB/smollm-corpus` |
| `python.bin` | 124,500,691 | `bigcode/codeparrot-clean`, per-file filtered | MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Unlicense, CC0-1.0 only | the respective per-file authors |
| `probe_val.bin` | 1,000,000 | held-out windows from the sources above | as above, per source | as above |

Licence references:

- ODC-By-1.0 — <https://opendatacommons.org/licenses/by/1-0/>
- CDLA-Sharing-1.0 — <https://cdla.dev/sharing-1-0/>
- Apache-2.0 — <https://www.apache.org/licenses/LICENSE-2.0>

The full 2.4B-token training mixture is not distributed. These shards are the
equalised per-source inputs the continual-update experiments are scored against;
the rationale and legal analysis are in the repository's
`docs/legal/training-data-licence-matrix.md`.
