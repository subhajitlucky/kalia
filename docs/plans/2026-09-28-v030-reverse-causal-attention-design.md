# v0.3.0 — Reverse-Causal Attention: Design

Status: **design, pre-registration pending.** Written 2026-09-28 while v0.2.0
session 2 runs. Related: D42 (X16 negative), D43 (canonical val set), D5
(screen on benchmarks), `2026-09-28-why-27b-beats-397b.md`,
`2026-09-28-open-source-landscape-and-transfers.md`.

## The problem this is trying to solve

KALIA's defining measured weakness is the **Abhimanyu gap**: reversed-text loss
minus forward loss. 9.29 − 3.23 = **6.06 nats** at 58M. X16 confirmed it is not a
scale artifact — control micro-models at 30M already show **5.11 nats**. Random
token prediction is ≈10.8, so the model is barely better than chance at exiting a
sequence it could enter.

X16 attacked this with a *data* transform (train on 50% chunk-reversed text) and
it failed, correctly, on both seeds and both metrics. That failure is the useful
part: **if exposure to reversed text does not help, the deficit is in the
computation, not the training distribution.**

The structural cause is that causal attention can only ever aggregate the past.
Every mainstream decoder-only LM inherits this. It is not a bug in our
implementation — it is the design of the entire class, which is why 5B-parameter
models have the same 6-nat gap we do.

## The hypothesis

**A recurrent reverse pass makes reverse-order information architecturally
reachable rather than merely predicted.**

For each layer, instead of one causal attention stack, run two:

- a **forward causal pass** — standard, unchanged;
- a **reverse pass** — the same sequence processed in reverse order with its own
  causal mask, so that position *t* can attend to what follows it.

Fuse with a learned gate. The forward stream is initialised at zero
contribution, so the model is **functionally identical to the current
architecture at step 0** and has to *learn* whether the reverse stream is useful.
This matters: it means a null result is interpretable (the gate stayed closed
because the reverse view genuinely does not help) rather than ambiguous (did the
thing fail, or did the new thing never get wired in?).

Naming: **RCAA — Reverse-Coordinated Causal Attention.** The design is
independent; the name is a label, not a claim to antiquity.

### Why this and not the alternatives

- **Bidirectional prediction (next + 2-ahead aux loss)**, already in our backlog
  as O4: trains the model to predict forward-future tokens, which makes the
  forward pass stronger but gives no mechanism for *reverse* processing.
- **Compressive/recurrent memory** (TII Falcon, NVIDIA Nemotron): adds long-range
  capacity but keeps the causal asymmetry intact. Fixes LAMBADA, not the gap.
- **Reversal data augmentation** (X16): tested, failed.
- **RCAA** is the only option in this set that changes what the model can
  *compute*, not what it has *seen* — which is the one thing X16 proved is
  insufficient.

## Design

### ReverseRecurrentAttention

For input `x` (B, T, C):

1. `h_f, h_r = norm(x).chunk(2)` — split the channel dim into forward and
   reverse streams, each `n_embd // 2`. Halving width (rather than running two
   full-width stacks) keeps parameter count and FLOPs near the current model, so
   v0.3.0 is comparable to v0.2.0 at equal compute — the property that makes the
   comparison honest.
2. Forward stream: standard causal MHA with RoPE, `n_head` heads, head_dim
   `n_embd // 2 // n_head`. **Causal mask as today.**
3. Reverse stream: flip the sequence, apply the same causal MHA **with its own
   qkv/proj weights**, flip back. Positions get *reversed* RoPE so the rotary
   encoding still reflects the true index.
4. Concatenate `[h_f_out, h_r_out]` → `n_embd`, project.

The two streams see disjoint halves of the representation and never mix except
through the gate, so the forward path is not diluted.

### The gate

```
g = sigmoid(W_g(RMSNorm(x)))        # (B, T, n_embd), zero-init
out = g * proj([h_f, h_r])
```

Zero-init `W_g` → sigmoid(0) = 0.5, not zero, so instead initialise the **gate
bias** to a large negative value (e.g. −4) so the initial gate is ≈0.018: the
reverse stream is nearly silent at step 0 and the model starts as a faithful
approximation of v0.2.0's architecture.

The gate is per-channel and data-dependent, so the model can learn to open the
reverse view only where it carries information. Qwen's GatedNorm and Gated
Residual work found the same shape — a cheap elementwise gate after a norm —
worth 1.58–1.98 accuracy points, so the gate is doing real work in the frontier
too, not just in our design.

### Where it goes

Alternate: reverse stream in every **other** layer (layers 1, 3, 5, 7, 9), plain
causal in the rest. Reasons:

- parameter cost is roughly halved again versus all-layer;
- adjacent reverse layers risk the two streams learning redundant features
  (this is the Hymba lesson in miniature — novel structure without a clear job
  underperforms a plain block);
- gives 5 independent gates, so we can report *per-layer* gate openings. That is
  a genuinely new piece of evidence: which depths actually want reverse
  information?

### Parameter budget

Current: 57,856,256. v0.3.0 target: **≤ 62M** (+7%). Layers that carry a reverse
stream add one extra qkv + proj at half width. Held under budget by keeping
`n_embd` at 512 and accepting half-width heads (head_dim 32 instead of 64) in
the reverse stream only.

## What must be true, and what would falsify it

Pre-registered predictions (to be hashed before any run):

| ID | Prediction | Threshold |
|---|---|---|
| **R-P1** | RCAA reduces the Abhimanyu gap vs a same-recipe control | gap ≤ control − **0.50 nats** at matched steps |
| **R-P2** | RCAA does not cost forward quality | forward val loss ≤ control + **0.03** |
| **R-P3** | The gate actually opens | mean gate value > **0.05**, with ≥1 layer > 0.10 |
| **R-P4** | RCAA improves a real task, not just the metric it was designed for | LAMBADA ≥ control + **3 pp** |

R-P3 is the diagnostic that makes the rest interpretable. If the gate stays shut
and the gap does not move, the hypothesis is cleanly falsified: reverse-pass
information is not useful to this model, and we stop. If the gate opens and the
gap still does not move, that is a *more interesting* negative — the model uses
the reverse view for something other than reversal.

## The risk I cannot design away

A reverse pass is a second exponential-decay path per layer, which can
destabilise training. Mitigations, in order of preference:

1. **QK-Norm and logit soft-cap stay on** (already in the config) — they exist
   precisely to stop attention blow-ups.
2. If the first micro-run diverges, fall back to **detaching the reverse stream
   through the gate for the first 200 steps** (warmup), then release it.
3. If it still diverges, the hypothesis is untestable in this budget, and that
   is recorded as the result. I will not spend a full run discovering it.

The one datapoint I have about novel architecture at small scale is negative:
Hymba-1.5B scored 53.75% on GSM8K against Qwen2.5-1.5B's 70%. Novel structure
with weak training loses to a plain transformer with good training. So the
control arm in this experiment is not optional — it is the whole point.

## Cost and sequencing

| Stage | GPU | Purpose |
|---|---|---|
| S0: intra-document masking fix | **0h** | Correctness bug in our packing; measure independently first |
| S1: RCAA micro-ablation, 2 arms × 30M × 50M tokens | ~1.5h | Does it train at all? Gate behaviour |
| S2: RCAA micro-ablation, seeds | ~1h | Is the gap movement real or noise? |
| S3: full run at 58M-equivalent | ~20h | The actual result |
| **Total** | **~23h** | Fits one quota window with ~7h spare |

S0 must land **before** S1, and its effect reported separately. Bundling a
correctness fix with an architecture change would make both uninterpretable —
which is exactly the mistake v0.2.0 would have made had I bundled the fiction
mixture with an attention change.

## What ships if this fails

If R-P1 through R-P4 miss, the result is published as a negative with the gate
openings reported, and v0.3.1 falls back to the four verified wins: WSD to zero,
three-stage mixture, NoPE on alternating layers, benchmark-based screening.
Those are unglamorous and they work. The architecture bet is the risky 20% of the
effort, not the whole of it.

## Explicitly out of scope

Distillation, any teacher weights (D2 holds), MoE, SSM/Mamba hybrids, and
changing `context_len` — all of which would break either the from-scratch
constraint, the yardstick, or both.
