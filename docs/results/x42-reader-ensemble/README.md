# X42 — Reader-Guided Compiler Ensemble

Status: **null.** The frozen reader cannot reliably choose between the two X41
compiler edits on the unstable `rename+swap` cell. The X41 partial result
stands; the public model remains v0.3.1.

## What was tested

The two X41 Arm B compilers make different `rename+swap` errors
(0.354 vs 0.483). X42 asked whether the frozen reader's candidate distribution
can select the better edit without labels. The selection rule was chosen on a
deterministic dev half and evaluated once on the held-out test half
(`sha256(id + "x42") % 2`).

### Dev rule table (n = 771)

| rule | S-P | `rename+swap` |
|---|---:|---:|
| MaxProb | 0.4213 | 0.4082 |
| Margin | 0.4157 | 0.4150 |
| **MinEntropy (chosen)** | **0.4213** | **0.4218** |
| Agreement | 0.4213 | 0.4082 |
| MaxProbNoNoop | 0.4213 | 0.4082 |
| Oracle (not deployable) | 0.4326 | 0.4830 |

## Held-out test half (n = 829)

| metric | selected rule | B_1401 | B_1402 | Oracle |
|---|---:|---:|---:|---:|
| S-P | **0.4144** | 0.4234 | 0.4099 | 0.4279 |
| `number+place` | 0.6143 | 0.6143 | 0.6143 | — |
| `rename+swap` | **0.4286** | 0.3537 | **0.4898** | **0.4898** |
| U-P | 0.6186 | 0.5825 | 0.6598 | — |

Bars: **T-1 pass, T-2 fail, T-3 pass, T-4 pass, T-5/T-6 pass.**

The selected rule lands between the two seeds on `rename+swap` (0.4286 vs
0.3537/0.4898). The oracle selection is capped by the better single seed at
0.4898: the seeds' errors are not complementary enough for a verifier to
repair the cell. On the full 1,600 records, the selected rule reaches S-P
0.4175 and `rename+swap` 0.4252.

## Verdict

**X42 is a null.** Reader-guided edit selection does not fix the seed
instability. The failure is in the compiler's edit distribution, not in the
reader's ability to choose. A different selector is a new registration, not a
rescue. X43 tests the remaining data-coverage hypothesis.

## Artifacts

- `x42_summary.json` — dev rule table, test/full results, best single seed,
  oracle, bars.
- `x42_selections.jsonl` — per-record split, selected seed, correctness.
- `tools/x42_reader_ensemble.py` — pure JSON analysis; no model inference.
- Pre-registration: `docs/preregistrations/2026-11-05-X42-reader-ensemble.md`
  (hash `0746ac10…`).
