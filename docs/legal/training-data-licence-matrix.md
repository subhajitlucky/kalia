# Training-data licence matrix

Written 2026-09-28, prompted by I16 (41.7% of the code corpus measured as
copyleft) and by a follow-up question this document answers honestly: we
filtered *code* per-file because The Stack's licensing is genuinely mixed, and
took the *text* sources' dataset-level licences at face value. That asymmetry was
never examined. Now it has been.

**This is an engineering reading of licence texts, not legal advice.** Nothing
here is a substitute for counsel if we ever make a large data release. It is
good enough to decide what to build and what to refuse to publish.

## The corpus, source by source

| source | share | licence | on the Hub |
|---|---|---|---|
| `HuggingFaceTB/smollm-corpus` (FineWeb-Edu dedup) | 60% | ODC-By-1.0 | `odc-by` |
| `roneneldan/TinyStories` | 20% | CDLA-Sharing-1.0 | `cdla-sharing-1.0` |
| `HuggingFaceTB/cosmopedia` (v2, inside smollm-corpus) | 15% | Apache-2.0 | — |
| code (Python, permissive filter) | 5% | per-file, filtered to 7 permissive | — |
| code as built for **v0.1.2** | 5% | per-file, **unfiltered** — 41.7% copyleft | — |

Note what is *not* in the permissive list: only the code slice is. The three text
sources are outside it by design — they are single-licence datasets, so a
per-file filter has nothing to do.

## What each licence actually permits

### TinyStories — CDLA-Sharing-1.0

This is the one I expected to be the problem, and reading it changed my mind.

- **Training: explicitly permitted.** §1.2 defines "Computational Use" to include
  any computational analytical technique, and §2.1 grants a worldwide,
  non-exclusive, irrevocable right to Use the Data.
- **Publishing the weights: explicitly unrestricted.** §3.5 states the Agreement
  "imposes no obligations or restrictions on Your Use or Publication of
  **Results**", and §1.11 defines Results as "the outcomes or outputs that You
  obtain from Your Computational Use of Data." A trained model is the archetypal
  Result. This carve-out is the entire reason the licence is usable for model
  training at all.
- **Redistributing the TinyStories shard: permitted, but conditional.** §3.1
  requires publication under an unmodified copy of the Agreement with the
  attribution preserved, and §3.3 requires including the licence text or a
  hyperlink and forbids adding further restrictions.

**One clause deserves monitoring.** §1.11 says Results "shall not include more
than a *de minimis* portion of the Data on which the Computational Use is based."
That is a licence term, not just a privacy concern: in principle, a model that
memorised and could regurgitate its training data might exceed it. A 58M model
on 2.4B tokens is not the regime where that happens, but it is the reason a
memorisation probe belongs in the release checklist rather than being assumed
away. It is not on our roadmap and I am not proposing it now — it is a
consequence we would be exposed to, so it is written down.

### FineWeb-Edu — ODC-By-1.0

ODC-By restricts **publication of the database**, not derived outputs. Training
and publishing the weights carry no restriction. Redistributing the shard
requires attribution to the provider, which §3 of ODC-By obliges us to include.

### Cosmopedia — Apache-2.0

Permissive. Redistribution permitted with the licence and NOTICE retained.

### Code slice, v0.2.0 — per-file filtered

Only apache-2.0, mit, bsd-2-clause, bsd-3-clause, cc0-1.0, isc and unlicense
survive the filter. This slice is **ours to give**, and it is the most
interesting artifact to publish: it is the part with a real provenance story.

### Code slice, v0.1.2 — per-file unfiltered

41.7% of characters under GPL/AGPL/LGPL/MPL/EPL. Training on it is a separate
question from redistributing it, and on the prevailing view the resulting weights
are not a derivative work. **Redistributing that shard would attach genuine
copyleft obligations** — source availability, notice, and the share-alike
consequences of GPL-3.0 and AGPL-3.0. We do not, and will not, publish it.

## The asymmetry, stated plainly

**v0.2.0 is the licence-clean checkpoint and v0.1.2 is not.** The worse model on
the benchmarks is the one whose corpus we can actually redistribute. That is
uncomfortable and it is the fact, and it is a point in favour of the
compliance rebuild having been worth doing on its own terms, separate from any
loss or accuracy claim.

## What this permits us to do

| action | permitted? | conditions |
|---|---|---|
| Train on any of these sources | **yes** | all four grant Computational Use |
| Publish v0.1.2 or v0.2.0 weights | **yes** | CDLA §3.5 puts Results beyond restriction; ODC-By restricts databases, not outputs |
| Publish the journal, code, configs, logs, pre-registrations, licence audit | **yes** | our own work, MIT/Apache-2.0 as already stated |
| Redistribute the **v0.2.0** corpus | **yes, conditional** | republish the TinyStories slice under unmodified CDLA-Sharing-1.0 with attribution; attribute FineWeb-Edu; retain the Apache notice for Cosmopedia; ship the filtered code slice with its filter |
| Redistribute the **v0.1.2** corpus | **no** | would attach GPL/AGPL obligations; the copyleft slice is not ours to relicense |

So the earlier shorthand — "publish the recipe, not the bytes" — turns out to be
**conservative rather than necessary** for v0.2.0. The bytes are publishable if
we honour three attribution conditions that are cheap to satisfy and are in our
own repo already. The recipe is still the better artifact on reproducibility
grounds: the sources are public, so re-downloading beats redistributing, and a
4.8GB upload serves nobody. The corpus becomes an *option* for a future release,
not a blocked item.

What remains genuinely non-negotiable: **the v0.1.2 code shard is never
published**, and the card for v0.1.2 continues to disclose that its training
data contained copyleft and that we do not redistribute it.

## Gap worth closing

We have now read the licences of all four sources, which we had not. We still
have not recorded the **cosmopedia-v2 licence as declared inside
`smollm-corpus`** — the standalone `HuggingFaceTB/cosmopedia` is Apache-2.0, but
the v2 config lives in a different repository and the datasets-server reports an
empty `license` field for it. That is the same "mirror declares nothing"
situation that disqualified the pg19 mirrors, and it is the one remaining
unverified cell in this table. It should be resolved before any corpus
redistribution, and it does not affect the v0.2.0 release, which redistributes
nothing.

## v0.3.0 new artifacts (added 2026-10-05, pre-launch)

The continual update introduces three new artifact classes. None redistributes
third-party text beyond what v0.2.0 already covered, but each gets a row here
because I16 taught us that ordering gaps, not missing filters, are the defect.

| artifact | derivation | licence status |
|---|---|---|
| `corpus/probe_val.bin` (CL-0 probe shard) | slice of the v0.2.0 training stream | inherits v0.2.0 coverage; no new source |
| `corpus/<source>.bin` replay shards | same filtered sources as v0.2.0, repackaged per-source | inherits v0.2.0 coverage; no new source |
| long-form chunked corpus (`sedthh/gutenberg_english`, sentence-aware chunks) | **new source** | **unverified** — Project Gutenberg texts are public domain in the US, but the HF mirror's declared licence has not been read. Must be read before any redistribution; training use is not blocked by this, redistribution is. |

Rule going forward: no corpus artifact is consumed by a registered step until
it has a row in this table. That is the filter-gates-build wiring, stated as a
table invariant rather than a timestamp comparison.
