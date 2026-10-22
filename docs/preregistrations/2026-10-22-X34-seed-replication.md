# Pre-registration — X34: does the midpoint readout survive a fresh generation seed?

Registered: 2026-10-22, before any GPU time. Related: X33, D58.

## Why this exists

X33 found a linear midpoint signal: a ridge probe on layer-9 features from the
prompt plus the first 128 generated tokens beat greedy on the unseen second half
by **+0.0155** [+0.0005, +0.0321], with a shuffled-label control at +0.0036.
The lower bound is barely above zero. The first thing that could explain it is
the sampling draw: X33 used generation seed base **7**.

X34 changes **only** that seed, to **1007**, and repeats the identical protocol.
If the probe still beats greedy, the signal is not a property of one sampling
draw. If it does not, X33 was a seed artifact.

## Design

Same 20 stories, same v0.2.0 checkpoint, same 768 prompt tokens, 8 samples,
temperature 0.8, top-k 200, layer 9, features (mean first half + last token),
α = 10, leave-one-story-out, same baselines and shuffled-label control.

The only change is the generation seed: `1007 + story_index` instead of
`7 + story_index`. The probe, features, labels and split are unchanged.

## Pre-registered predictions

| ID | Prediction | Threshold |
|---|---|---|
| **R-1** | The probe still beats greedy | probe − greedy mean **≥ +0.010**, paired 95% CI excludes zero |
| **R-2** | The readout still beats log-probability | probe − log-prob mean **≥ 0** |
| **R-3** | The signal is not a label artifact | shuffled-label probe − greedy **< +0.010** |

## Decision rule (fixed now)

- **R-1, R-2 and R-3 pass** → the readout replicates across sampling seeds. X35
  (a fresh story set) is the remaining test before any selector claim.
- **R-1 fails** → X33 was a seed artifact. Report a null; do not proceed to a
  selector.
- **R-1 passes, R-3 fails** → the probe is reading an artifact; report a null.
- **R-2 fails** → the readout is not better than log-probability on this seed;
  report the mixed result and do not claim a selector.

## Cost

One GPU session, twenty stories. **1.5–2 GPU-hours**.

## Deviations

Any change to the seed, stories, layer, features, α or thresholds after this
commit requires a hash-registered amendment. Results are reported regardless of
outcome.

---

## Run status (appended 2026-10-22, before the registered run)

**v1 is not a result.** The first X34 launch reran generation seed **7**, not
1007: the stager's `pins` tuple only acted on `SEED` and `REPLAY`; `SEED_BASE`
was listed but never patched. The tell was exact equality with X33's numbers.
The stager now supports `--seed-base`, patches and reads back the value, and has a
test that asserts the runner consumes it.

**v2 is staged with `SEED_BASE = "1007"` but is blocked by the weekly GPU quota**
(30.00 hours reached). It will run after the reset. No v1 output is archived as
a replication.
