# X55 — text-conditioning embedding fix

Status: **clean pass of E-1…E-7.** The earlier see→draw→see null was an
inference-embedding mismatch, not a compositional gap. With the embedding the
v1 diffusion was trained with, the loop closes on held-out combinations at
**0.930 exact**.

## Result

A/B with the same v1 diffusion, v2.1 captioner, real test images and held-out
split; only the caption-embedding function changes.

| condition | split | loop exact | loop attributes | source readback | classifier exact |
|---|---|---:|---|---:|---:|
| **B — bidirectional** (v1 training) | held-out | **0.930** | 0.967–0.993 | **0.930** | **0.936** |
| B — bidirectional | seen | 0.949 | 0.978–0.997 | 0.955 | 0.954 |
| **C — causal** (old cycle) | held-out | **0.000** | 0.240–0.538 | 0.004 | 0.000 |
| C — causal | seen | 0.042 | 0.280–0.502 | 0.062 | 0.048 |

- Real caption exact: B held-out **0.992**, B seen **0.997** (sanity; captioner
  frozen).
- Cosine similarity between B and C: **mean 0.692**, std 0.018, min 0.647,
  max 0.736.
- B generated-image pixel stats are close to real: mean 0.72/0.71/0.69 vs real
  0.76/0.74/0.73; std 0.38/0.37/0.42 vs real 0.37/0.37/0.42.
- C generated images are shifted (green mean 0.83, std 0.18), consistent with
  an out-of-distribution condition.
- 1,000 held-out + 1,000 seen for B; 500 + 500 for C. No training; KALIA, the
  captioner and the diffusion are frozen.

| bar | result |
|---|---|
| **E-1** B held-out loop exact ≥ 0.70 | **PASS (0.930)** |
| **E-2** B held-out loop attributes ≥ 0.80 | **PASS (0.967–0.993)** |
| **E-3** B held-out source readback ≥ 0.70 | **PASS (0.930)** |
| **E-4** B held-out classifier exact ≥ 0.80 | **PASS (0.936)** |
| **E-5** B held-out real caption ≥ 0.90 | **PASS (0.992)** |
| **E-6** C materially lower | **PASS (0.000 vs 0.930)** |
| **E-7** from scratch; artifacts | **PASS** |

## Cause

The v1 diffusion was trained with

```python
h = base_text.encode(x, attn_mask=mask[:, None, None, :])
pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
```

a **bidirectional** masked mean. The earlier cycle kernels used

```python
h = base_text.encode(ids)
pooled = h.mean(1)
```

a **causal** mean. The two embeddings have cosine ~0.69, and the causal one is
far outside the diffusion's training distribution: the independent classifier
trained on real images cannot recognize a single causal-conditioned sample
(0.000 exact). The earlier cycle's 0.032 and X53's 0.005 were measuring this
mismatch, not the model's compositional ability.

## What this means

- KALIA's seeing and drawing modules **do** share a compositional
  representation in this synthetic grid. A real held-out scene is captioned,
  drawn from that caption, and captioned back at **0.930 exact**; the
  independent classifier reads the drawn scene at **0.936**.
- The previous "compositional gap" interpretation (D97/D98) is corrected. The
  X53 generated-image fine-tune was adapting to out-of-distribution images
  produced by the wrong conditioning, so its null is not evidence of a
  compositional boundary.
- The v2.1 captioner's compositional split result (held-out **0.992**) stands.

## Artifacts

- `metrics.json` — B/C metrics, cosine, pixel stats, samples.
- `x55.log` — run log.
- `bidir_held.npy`, `bidir_seen.npy`, `causal_held.npy`, `causal_seen.npy` —
  generated images (uint8, 32×32).
- Pre-registration: `docs/preregistrations/2026-11-08-X55-embedding-fix.md`
  (hash `e5261fcf…`).
- Kernel: `subhajitlucky/kalia-x55-embedding-fix`.
- Corrected embedding module: `vision_conditioning.py`.
