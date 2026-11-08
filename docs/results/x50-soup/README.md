# X50 — 50/50 soup of v0.3.1 and X49 treatment

Status: **null.** The soup is a small mean gain but misses the registered
+1.0 pp and weak-task bars.

## Result

| task | v0.3.1 | soup | Δ (pp) |
|---|---:|---:|---:|
| ARC-Easy | 0.4800 | 0.4900 | +1.00 |
| ARC-Challenge | 0.2014 | 0.1997 | −0.17 |
| BoolQ | 0.5435 | 0.5390 | −0.45 |
| HellaSwag | 0.3235 | 0.3240 | +0.05 |
| LAMBADA | 0.2700 | 0.2660 | −0.40 |
| OpenBookQA | 0.1640 | 0.1640 | 0.00 |
| PIQA | 0.6197 | 0.6213 | +0.16 |
| SciQ | 0.7080 | 0.7200 | +1.20 |
| WinoGrande | 0.5225 | 0.5320 | +0.95 |
| **mean** | **0.4258** | **0.4284** | **+0.26** |

- **S-1 FAIL**: +0.26 pp < +1.0.
- **S-2 PASS**: no task beyond 2σ regression.
- **S-3 FAIL**: weak-task average **−0.13 pp**.
- **S-4 PASS**.

## Decision

No v0.3.2. No soup tuning. Public base remains v0.3.1. The next window is the
cross-modal binding discovery.

## Artifacts

- `eval.log`, `eval_summary.json`.
- Soup kernel `subhajitlucky/kalia-x50-soup`; checkpoint sha `03003451…`
  (334MB, held in the kernel output, not committed).
- Pre-registration: `docs/preregistrations/2026-11-08-X50-soup.md`
  (hash `5a1494e3…`).
