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

## 9-task result (both arms, 2k where available)

| task | control | treatment | Δ (pp) |
|---|---:|---:|---:|
| ARC-Easy | 0.4845 | 0.4845 | 0.00 |
| ARC-Challenge | 0.1997 | 0.1937 | −0.60 |
| **BoolQ** | 0.5115 | **0.5660** | **+5.45** |
| HellaSwag | 0.3280 | 0.3265 | −0.15 |
| LAMBADA | 0.2540 | 0.2595 | +0.55 |
| OpenBookQA | 0.1560 | 0.1540 | −0.20 |
| PIQA | 0.6153 | 0.6175 | +0.22 |
| SciQ | 0.7120 | 0.7140 | +0.20 |
| WinoGrande | 0.5170 | 0.5359 | +1.89 |
| **mean** | **0.4198** | **0.4280** | **+0.82** |

- **K-1 FAIL**: +0.82 pp < +1.0.
- **K-2 PASS**: no task regresses beyond 2σ.
- **K-3 FAIL**: weak-task average **−0.10 pp** < +1.5.
- **K-4 PASS**: shared control val 2.80249 → 2.80061 (−0.0019).

The mixture moved BoolQ and WinoGrande, not the targeted weak tasks. Under
the registered rule, the mixture is parked; no seed-1502 replication.
