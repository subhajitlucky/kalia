# X33 — the midpoint state linearly knows whether the entity will survive

Registered: `docs/preregistrations/2026-10-22-X33-linear-readout.md`
(sha `1deb9d28…`, ledger-verified). Kernel
`subhajitlucky/kalia-x33-linear-readout`, code `804bf5a`, one GPU session.

X32 found entity attention concentrated in layer-9 heads, but ablating them
**reduced** second-half retention. X33 asked whether the information is linearly
readable **before the second half exists**.

## Design

20 frozen stories, 768 prompt tokens, 8 candidates at temperature 0.8 / top-k
200. For each candidate, the model saw the prompt plus **only the first 128
generated tokens**. The layer-9 block output was captured at the midpoint:
1,024 features = mean over the first-half positions plus the last-token state.

A ridge probe (**α = 10**) was trained on 19 stories and tested on the held-out
story; leave-one-story-out over all 20. Features were standardized on the
training stories. The label — the candidate's second-half entity retention — was
never part of the input.

Baselines: greedy, log-probability, first-half entity pick, oracle, and a
20-shuffle shuffled-label probe.

## Results

| selector | mean second-half retention |
|---|---|
| greedy | 0.0170 |
| log-probability pick | 0.0103 |
| first-half entity pick | 0.0277 |
| **linear probe (midpoint, unseen story)** | **0.0324** |
| oracle (coverage ceiling) | 0.0618 |
| shuffled-label probe | 0.0205 |

| contrast | mean | 95% CI |
|---|---|---|
| **probe − greedy** | **+0.0155** | **[+0.0005, +0.0321]** |
| probe − first-half | +0.0047 | [−0.0106, +0.0217] |
| probe − log-probability | +0.0222 | [+0.0041, +0.0443] |
| shuffled probe − greedy | +0.0036 | — |

## Registered bars

| ID | bar | measured | verdict |
|---|---|---|---|
| **P-1** | probe − greedy ≥ +0.010, CI excludes zero | **+0.0155**, CI [+0.0005, +0.0321] | **PASS** |
| **P-2** | probe ≥ half-pick and log-prob | 0.0324 vs 0.0277 / 0.0103 | **PASS** |
| **P-3** | shuffled-label probe − greedy < +0.010 | +0.0036 | **PASS** |

**Registered verdict: a linear midpoint signal exists; replicate before a
selector claim.**

## What this establishes

**The model’s midpoint state contains a linearly decodable signal about future
entity persistence.** The probe is trained on other stories, sees only the first
half, and still selects a continuation whose unseen second half retains more
entities than greedy, log-probability, and the first-half rule.

It closes about **35% of the coverage gap** (0.0324 against greedy 0.0170, with
the oracle at 0.0618). The lower confidence bound is barely above zero, so this
is a promising signal, not a certified selector.

**It beats log-probability clearly** (+0.0222, CI excludes zero). **It beats the
first-half rule by +0.0047, but that interval includes zero** — the probe’s
advantage over the cheap entity rule is not yet resolvable at 20 stories.

**The shuffled-label control lands at +0.0036 over greedy**, below the registered
+0.010 bar. The probe’s +0.0155 is about four times that control mean, but the
control is not zero; the label distribution itself gives a small lift to any
selector.

## What this does not establish

- Not a capability claim. Twenty stories, one split, one seed of generation.
- Not better than the half-pick beyond noise.
- Not a first in the world: learned selection and verifier-guided decoding exist
  (GenSelect at 1.7B, Best-of-Evidence, small-model verifier work). The novel part
  is the **causal first-half linear readout at 58M** and the measured 35% closure
  of the coverage gap.
- The features are 1,024-dimensional and the training set is 152 rows; the ridge
  result is a readout, not proof that the representation is low-dimensional.

## Next

1. **Replicate on a second story set** — 20 different Gutenberg stories, same
   protocol, same α. A selector claim needs the sign and the bar to hold.
2. **Second generation seed** — regenerate the candidate pools to ensure the
   signal is not a property of one sampling draw.
3. If both hold, train the probe on more stories and measure the
   **selective-accuracy curve** (accuracy at 20/50/80% coverage), with abstention.
4. Only then call it a selector. Until then it is a readout result.

## Artifacts

- `x33_summary.json` — per-story rows and verdict.
- `x33_features.npz` — features, labels, log-probs, candidate ids.
- `run_manifest.json` — code fingerprint, checkpoint hash, story hash, α, layer.
- `kalia-x33-linear-readout.log` — the kernel log.

Reproduce the verdict from the registered JSON; the decision rule is in the
pre-registration.
