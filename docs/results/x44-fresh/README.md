# X44/X45 — fresh benchmark and depth: boundary result

Status: **partial.** The compiler generalizes to fresh templates for pair
composition and `number+place`, but `rename+swap` falls below the registered
0.45 bar, and 3-op depth passes on only one seed. This is a boundary result,
not a new release.

## Design

New names and new sentence templates, disjoint from every training/eval set in
X40–X43. Three cells, 400 records each:

- **F-S** — single op, 2-way.
- **F-P** — pair op, 4-way.
- **F-T** — triple op, 8-way (chance 0.125).

Benchmark sha `b69bcd61…`. Eval kernel `subhajitlucky/kalia-x44-fresh-eval`,
0.24 GPU-h.

## Baseline vs X43

| cell | v0.3.1 normal | gold-edit | C_1401 | C_1402 |
|---|---:|---:|---:|---:|
| **F-S** | 0.3875 | 0.7125 | **0.7275** | **0.6950** |
| **F-P** | **0.0850** | 0.5725 | **0.5225** | **0.4925** |
| **F-T** | **0.0350** | 0.3925 | **0.2825** | **0.3650** |

Hard operations:

| operation | normal | gold | C_1401 | C_1402 |
|---|---:|---:|---:|---:|
| `number+place` | — | — | **0.4545** | **0.4545** |
| `rename+swap` | — | — | **0.4059** | **0.4158** |
| `rename+place` | — | — | 0.5758 | 0.7273 |
| `swap+place` | — | — | 0.5758 | 0.5455 |
| `rename+negate` | — | — | 0.8300 | 0.7600 |
| `swap+negate` | — | — | 0.3200 | 0.2200 |

Compiler exact match on fresh templates: **0.1850 / 0.1242** — much lower than
the 0.547/0.544 on the original 2×2. The reader still recovers many answers
end-to-end, but the compiler's edit generation is the weak link OOD.

## Registered bars

| ID | prediction | result |
|---|---|---|
| **F-1** | fresh pair S-P ≥ 0.40, both | **PASS** (0.5225 / 0.4925) |
| **F-2** | fresh `number+place` ≥ 0.40, both | **PASS** (0.4545 / 0.4545) |
| **F-3** | fresh `rename+swap` ≥ 0.45, both | **FAIL** (0.4059 / 0.4158) |
| **F-4** | fallback < 0.01 | **PASS** |
| **F-5** | reader unchanged | **PASS** |
| **F-6** | 3-op depth ≥ 0.30, both | **FAIL** (0.2825 / 0.3650) |

## What this means

1. **The pair-composition gain is real OOD.** On fresh templates, normal F-P is
   **0.085**; the compiler reaches **0.523/0.493** (gold 0.573). `number+place`
   clears 0.40 on both seeds. That is a genuine generalization result.
2. **`rename+swap` is the boundary.** The original 2×2 result (0.592/0.646)
   does not fully transfer to fresh templates (0.406/0.416). The cell is
   role-binding-heavy; the compiler's exact edit is often wrong even when the
   reader still gets the answer.
3. **3-op depth is partial.** One seed reaches 0.365 (close to the gold 0.3925);
   the other is 0.2825. Depth is promising but not stable at the registered
   bar.
4. **The X43 release remains valid for its registered benchmark**, but no
   v0.4.1/v0.5.0 broad claim is made. The release note's template-held-out
   caveat is now backed by a measured boundary.

## Next

If the next step is a robust OOD compiler, the recipe is broader template
diversity in training (X46), with X44 kept as the held-out test. If not, the
honest endpoint is: X43 is a benchmark-specific capability release; the
fresh-template boundary is the finding.

## Artifacts

- `x44_summary.json` — cells, ops, bars, benchmark manifest.
- `baseline_normal.json`, `baseline_gold.json`.
- `end2end_C_*.json`, `end2end_C_*_rows.jsonl`, `generated_C_*.jsonl`.
- `bench_audit.json`, `kalia-x44-eval.log`.
- Pre-registration: `docs/preregistrations/2026-11-05-X44-fresh-depth.md`
  (hash `45b516f2…`).
