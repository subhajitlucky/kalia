# X64 v150 smoke — passed

Status: **smoke passed, 2026-11-10.** This is a throughput/initialization
smoke, not the full 150M run.

## Result

| field | value |
|---|---|
| kernel | `subhajitlucky/kalia-x64-v150-smoke-2` |
| code commit | `7c0becbbab32b103791c28024e40ea7824995f56` |
| parameters | **147,803,904** |
| GPUs | 2×T4 16GB |
| init val | **2.7731** (bpB 0.9066) |
| init gate | **2.9079** (v0.3.1 2.7579 + 0.15) |
| final val | 2.7737 (bpB 0.9068) |
| steps / tokens | 22 / 11,534,336 |
| throughput | **6,531–6,913 tok/s** |
| max minutes | 30 |

The zero-init identity insertion preserves the v0.3.1 function: the expanded
147.8M model starts at **2.7731**, 0.0152 above the registered v0.3.1 number
(2.7579). That is inside eval noise and far below the 2.9079 gate. The previous
linear-interpolation method started at **8.3523** and failed.

The smoke ran only 22 steps because the 30-minute budget expired; final val is
therefore essentially unchanged (2.7737). The point was to measure init and
throughput, and both pass.

## What it means for the 120-hour run

At **6.7k tok/s**:

- 2.4B tokens = **~99.5 hours**;
- 4,770 steps × 524,288 tokens = 2.5B tokens = **~103.6 hours**;
- 120 hours leaves **~16 hours** for evaluation, checkpointing and restarts.

So the X64 full run **fits the 120-hour Kaggle budget**. The 150M depth-expand
is viable; no pivot to the 58M is needed on throughput grounds.

## Artifacts

- `train.log` — full Kaggle log.
- `run_manifest.json` is embedded in the log; the init checkpoint sha256 is
  `3e879d0b12ea97d0695e2b2ec8b4574880c8735cd943a0cbd2dc729e2c23e492`.
- Base checkpoint: `subhajitlucky/kalia-v031-ckpt` (v0.3.1 `model.safetensors`
  sha `bbd8a036…`, 334,360,208 bytes).
- Data: `kalia-prep-v2b` `train.bin` sha `313fc7ba…`, `val.bin` sha
  `625f8e5f…`.

## Next

Wait for the **Nov 14** Kaggle quota refresh, then run the registered X64 full
continued pretraining in 4 weeks of 7.5-hour sessions. The public base remains
**v0.3.1** until the 9-task gate and capability regression pass.
