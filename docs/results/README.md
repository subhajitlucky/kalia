# Run results

Machine-readable record of every experiment in this project: what ran, on which
kernel, how many parameters, and the validation loss with its full curve.

**Regenerate with `python tools/collect_results.py`. Verify with `--check`.**

Generated from the raw logs archived in `logs/`. It is not hand-maintained —
`test_results_record.py` fails if the committed file stops matching its logs, so
the two cannot drift.

## Reading it

```json
{
  "experiment": "x21",
  "kernel": "subhajitlucky/kalia-x21-replication",
  "arms": [
    {
      "arm": "micro-staticgate-s1338",
      "params": 29999040,
      "val_loss_final": 4.9623,
      "val_curve": {"100": 6.0207, "200": 5.5024, "...": 0, "500": 4.9623},
      "delta_vs_control": 0.0854,
      "source": "docs/results/logs/x21.log"
    }
  ]
}
```

`source` is the part to read first:

| `source` | meaning |
|---|---|
| `docs/results/logs/<exp>.log` | parsed from an archived run log |
| `prereg:<exp>` | transcribed from a pre-registration's prose — **not** parsed from a run |

`val_curve: null` with `truncated: true` means the number is real but its curve
was lost. `kaggle_live_log.py` returns a rolling window, so older arms fall out
of it. Seven of the eleven arms are in that state.

## The results

| experiment | arm | params | val loss | Δ vs control | curve |
|---|---|---|---|---|---|
| X18 | `micro-base` | 29,920,512 | 4.7662 | — | lost |
| X18 | `micro-gated` | 30,072,768 | 4.7226 | −0.0436 | lost |
| X18 | `micro-nope` | 29,920,512 | 4.7772 | +0.0110 | lost |
| X19 | `micro-branchnorm` | 29,922,816 | 4.7680 | +0.0018 | lost |
| X20 | `micro-staticgate` | 29,999,040 | 4.6304 | −0.1358 | lost |
| **X21** | `micro-base-s1338` | 29,920,512 | 4.8769 | — | **full** |
| **X21** | `micro-staticgate-s1338` | 29,999,040 | 4.9623 | **+0.0854** | **full** |
| **X21** | `micro-base-s1339` | 29,920,512 | 4.8908 | — | **full** |
| **X21** | `micro-staticgate-s1339` | 29,999,040 | 4.9087 | **+0.0179** | **full** |

Only X21 has complete curves, because it is the only run whose log was archived
whole.

## What the record says

**X20's −0.1358 inverted.** At two fresh seeds the static gate is *worse* than
control, by +0.0854 and +0.0179. R-1, R-2 and R-4 all fail.

**The control is not seed-invariant.** `micro-base` scores 4.7662 at seed 1337
(two independent sessions, 0.0018 apart) and 4.8769/4.8908 at seeds 1338/1339 —
about **0.11 nats** higher. Two sessions agreeing to 0.0018 means the machine is
not the variable, so this is a seed effect.

That 0.11 **exceeds X18's entire −0.0436 and X19's −0.0018** by an order of
magnitude. Every single-seed delta in this project was read against a baseline
that was never measured. `test_the_control_baseline_spread_is_recorded_not_just_asserted`
holds that number in place: changing it requires editing a test whose docstring
says why.

## Caveat worth stating

X18's three arms and the X19/X20 controls are `prereg:`-sourced. Those numbers
live in markdown tables, not in data. They are the numbers D48 reclassified as
nulls, and the honest position is that the refutation is *stronger* than the
record: X21 refutes the mechanism directly, and the missing baseline would only
strengthen that conclusion, not weaken it.

Re-running X18 with an archived log would close this. It costs ~0.4 GPU-hours and
this project no longer spends them on architecture arms (D48).
