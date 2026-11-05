# V032 — frozen-base output-adapter fine-tune

Status: **val bar passed; 2k guard passed; val-only internal candidate.**
Engineering, not a new mechanism.

## Val result

| seed | val | base val | delta | pass |
|---|---:|---:|---:|---|
| 1401 | **2.7493** | 2.7580 | **−0.0087** | yes |
| 1402 | **2.7495** | 2.7580 | **−0.0085** | yes |

- Base val re-measured in the same kernel: **2.7580** (published 2.7579).
- Bar: adapter ≤ base − 0.005 both seeds. **Pass.**
- Frozen-weight checksum unchanged before/after, both seeds. **Pass.**
- Adapter: `512 → 1024 → 512`, **1,048,576** trainable params; no base updates.
- 2,000 steps, micro 8 × grad accum 8, LR 1e-3, cosine; 2 seeds.
- Base sha `bbd8a036…`; kernel `subhajitlucky/kalia-v032-adapter` v2,
  revision `c40b464`.

## 2k guard (V-5)

Paired in one CPU kernel: base v0.3.1 and both adapters, limit 2,000 per task.
The fresh base run reproduced the published v0.3.1 2k numbers exactly.

| task | base | 1401 | Δ 1401 | 1402 | Δ 1402 | 2σ | regression |
|---|---:|---:|---:|---:|---:|---:|---|
| ARC-Easy acc | 0.4800 | 0.4845 | +0.0045 | 0.4860 | +0.0060 | 0.0316 | no |
| PIQA acc | 0.6197 | 0.6175 | −0.0022 | 0.6186 | −0.0011 | 0.0321 | no |
| HellaSwag acc | 0.3235 | 0.3240 | +0.0005 | 0.3200 | −0.0035 | 0.0296 | no |
| LAMBADA acc | 0.2700 | 0.2615 | −0.0085 | 0.2615 | −0.0085 | 0.0280 | no |
| LAMBADA ppl | 122.88 | 127.72 | +4.83 | 129.14 | +6.25 | 30.9 | no |
| WinoGrande acc | 0.5225 | 0.5217 | −0.0008 | 0.5233 | +0.0008 | 0.0397 | no |

**V-5 passes:** no task regresses beyond 2σ. But the 2k result is **neutral, not
better**: ARC gains ~+0.5–0.6 pp, LAMBADA loses ~0.85 pp, PIQA loses
~0.1–0.2 pp, HellaSwag/WinoGrande are flat; LAMBADA perplexity is 4–6 points
higher, within its ~31-point 2σ.

## Verdict

- **V-1…V-5 pass.** The adapter is an internal **v0.3.2-val-only candidate on
  internal candidate.**
- It is a **small val improvement**, not a benchmark improvement. Public model
  remains **v0.3.1**; no Hugging Face push.
- It is engineering, not a new mechanism. It does not address the composition
  failure.

## Artifacts

- `v032_summary.json` — val result, hashes, seeds, frozen check.
- `v032_bench_summary.json` — paired 2k comparisons, V-5.
- `v032_base_2k.json`, `v032_seed1401_2k.json`, `v032_seed1402_2k.json`.
- `v032_train_log.csv`, `v032_val_log.csv` — training/val curves.
- `adapter_seed1401.pt`, `adapter_seed1402.pt` — adapter state dicts only
  (~4.2 MB each); the base is public v0.3.1.
- `v032.log`, `v032-bench.log` — kernel logs.
- Pre-registration: `docs/preregistrations/2026-11-04-V032-output-adapter.md`
  (hash `4ead53b6…`).
