# X34 — the midpoint readout did not replicate on a second generation seed

Registered: `docs/preregistrations/2026-10-22-X34-seed-replication.md`
(sha `bb52e5f3…`, ledger-verified). Kernel
`subhajitlucky/kalia-x34-seed-s1007`, version 2, code `7381818`, one GPU session
(**0.64 GPU-h**).

X33 found a linear midpoint readout on layer-9 states: probe − greedy **+0.0155**
[+0.0005, +0.0321], with a shuffled-label control at +0.0036. Its lower bound was
barely above zero, so the first replication test was the sampling draw. X34
changed **only** the generation seed, from 7 to **1007**, and repeated the
identical protocol.

## Design

Same 20 frozen Gutenberg stories, same v0.2.0 checkpoint, same 768 prompt tokens,
8 candidates at temperature 0.8 / top-k 200, 256 generated tokens. For each
candidate the model saw only the prompt plus the first 128 tokens; the layer-9
block output was captured at the midpoint as the mean over the first-half
positions plus the last-token state (1,024 features). Ridge **α = 10**, leave-one-
story-out, features standardized on the training stories only. Same greedy,
log-probability, first-half, oracle and 20-shuffle controls.

The only moved variable is the seed. The invariants are exact:

| item | X33 | X34 |
|---|---|---|
| code fingerprint | `0444c15d…` | `0444c15d…` |
| checkpoint sha256 | `924ad52b…` | `924ad52b…` |
| stories sha256 | `a94b15f6…` | `a94b15f6…` |
| generation seed base | 7 | **1007** |

**v1 is not a result.** The first X34 launch reran seed 7 because the stager
silently ignored `SEED_BASE`; it returned X33's numbers exactly. The stager now
pins and reads back the value; v2 was staged at 1007, pushed as kernel version
2, and the pushed source was pulled back and checked before this run was read.
The shared runner's manifest at revision `7381818` did not include `SEED_BASE`;
`provenance.json` and the archived pulled source record the verified seed. The
runner now writes `seed_base` as well (provenance-only change, after this run).

## Results

| selector | mean second-half retention |
|---|---|
| greedy | 0.0170 |
| log-probability pick | 0.0114 |
| first-half entity pick | 0.0401 |
| **linear probe (midpoint, unseen story)** | **0.0299** |
| oracle (coverage ceiling) | 0.0794 |
| shuffled-label probe | 0.0238 |

| contrast | mean | 95% CI |
|---|---|---|
| **probe − greedy** | **+0.0129** | **[−0.0048, +0.0343]** |
| probe − first-half | −0.0103 | [−0.0350, +0.0100] |
| probe − log-probability | +0.0185 | [−0.0129, +0.0476] |
| shuffled probe − greedy | +0.0069 | — |

## Registered bars

| ID | bar | measured | verdict |
|---|---|---|---|
| **R-1** | probe − greedy ≥ +0.010, paired 95% CI excludes zero | **+0.0129**, CI [−0.0048, +0.0343] | **FAIL** — CI includes zero |
| **R-2** | probe − log-probability ≥ 0 | +0.0185 | **PASS** |
| **R-3** | shuffled-label probe − greedy < +0.010 | +0.0069 | **PASS** |

The runner prints `P1/P2/P3`, inherited from X33. Its `P2` is the stricter X33
bar (probe ≥ **both** the half-pick and log-probability), so it prints `P2: FAIL`
because the half-pick is 0.0401. Read against X34's registered **R-2**, which
names only log-probability, R-2 passes. Both readings are recorded; the
registered rule is the binding one.

**Registered verdict: R-1 fails. Report a null; do not proceed to a selector.**
The linear-selector track is closed.

## What the null means

The point estimate is **+0.0129** against X33's **+0.0155** — a move of −0.0026,
well inside X33's interval. The effect did not collapse; it is not resolvable on
a second seed at 20 stories. The pre-registration labels an R-1 failure "X33 was
a seed artifact"; the data support the narrower statement better: **the effect
did not replicate at the registered bar**. Either way there is no selector claim.

The substantive reversal is against the cheap rule. The first-half pick rose
from 0.0277 to **0.0401** while the probe stayed near 0.03, so probe − half
flipped from **+0.0047** to **−0.0103**. On this seed the first-half rule is the
better selector. The probe still beats log-probability (+0.0185), but that
interval also includes zero; X33's log-prob advantage (+0.0222, CI
[+0.0041, +0.0443]) does not survive either.

Coverage is unchanged as a fact about the candidate pool: the oracle reaches
**0.0794** against greedy's **0.0170**. The right continuation is in the pool.
Descriptively, the probe closes ~21% of that gap on this pool and the half-pick
~37%; X33's probe closed ~35% of its own gap. No cheap selector finds the right
continuation reliably.

## What this establishes

- **No seed-robust linear midpoint selector at 58M in this protocol.** The X33
  result is a single-seed readout observation that failed its registered
  replication; it must not be cited as a capability.
- **The first-half rule is the stronger cheap baseline on this pool**, and any
  future selector must beat it before claiming the readout adds information.
- **The coverage/selection gap remains.** The pool contains better continuations
  than greedy; the project still has no selector that finds them.

## What this does not establish

- It does not establish that the midpoint state contains no signal. The point
  estimate is stable and positive; the test is underpowered at 20 stories.
- It does not establish a better probe. No feature, layer or α change was tried,
  and none should be read into this run.
- It does not license X35 (a fresh story set) or a selective-accuracy curve.
  Both were contingent on R-1 passing; the registered decision is to stop.

## Next

Close the linear-selector track (D59). X35 is not run. If the question is ever
reopened, it needs a pre-registered larger-n design — 20 stories cannot resolve
a ~0.013 effect with this variance — and the first-half rule as the baseline to
beat. No more 20-story pilot is queued.

## Artifacts

- `x33_summary.json` — per-story rows and verdict (the shared runner's filename).
- `x33_features.npz` — features, labels, log-probs, candidate ids.
- `run_manifest.json` — code fingerprint, checkpoint hash, story hash, α, layer.
- `provenance.json` — the verified generation seed, kernel version and source hash.
- `kalia-x34-seed-s1007.py` — the pulled version 2 kernel source that ran.
- `kalia-x34-seed-s1007.log` — the kernel log.

The verdict is reproduced from the registered JSON; the decision rule is in the
pre-registration.
