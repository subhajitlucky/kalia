# X49 — reasoning-heavy data mixture

Status: **data ready; smoke test running.** Full 58M arms run in the 30-hour
window.

## Mixtures

Same source shards, same 524M training tokens, different ratios:

| source | control | treatment |
|---|---:|---:|
| FineWeb-Edu | 60% | **45%** |
| TinyStories | 20% | **5%** |
| Cosmopedia | 15% | **40%** |
| code (`stack_smol`) | 5% | **10%** |

- Control train sha `dfa712d9…`; treatment train sha `6cb98123…`.
- Both 1,048,000,000 bytes, 524M tokens; 10M val tokens.
- Data prep CPU only; no GPU.

## Design

Resume v0.3.1, 2,000 re-warm steps (524M tokens), Muon, LR 6e-4, warmup 200,
cosine, seed 1501 for both arms. The only difference is the mixture.

## Bars

- **K-1** treatment 9-task mean ≥ control + 1.0 pp.
- **K-2** no task regresses beyond 2σ.
- **K-3** weak-task average (OpenBookQA, ARC-C, LAMBADA, HellaSwag) ≥ +1.5 pp.
- **K-4** treatment val bpB ≤ control + 0.01.

A positive Stage 1 triggers a seed-1502 replication. A null parks the mixture.

## Artifacts

- `data_manifest.json`, `data.log`.
- Pre-registration: `docs/preregistrations/2026-11-06-X49-data-mix.md`
  (hash `3e86a1a3…`).
- Kernel: `subhajitlucky/kalia-x49-data`; smoke: `kalia-x49-smoke`.
