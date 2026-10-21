# Benchmark pair 2k — raw artifacts

The 2,000-sample re-measurement of v0.1.2 vs v0.2.0 on the five zero-shot tasks.
Narrative and verdict: `docs/eval/2026-10-21-v020-bench-pair-2k.md`.

| file | what |
|---|---|
| `bench_v012_2k.json` | lm-eval results for v0.1.2 |
| `bench_v020_2k.json` | lm-eval results for v0.2.0 |
| `bench_pair_2k.json` | deltas, z, S-A flags |
| `run_manifest.json` | checkpoint hashes, code fingerprint, limit, tasks |
| `kalia-bench-pair-2k.log` | kernel console log |

Kernel `subhajitlucky/kalia-bench-pair-2k`, CPU, code `cdbcc3b`. PIQA used
1,838 validation items and WinoGrande 1,267 (their full sets); the other tasks
used 2,000. The 500-sample numbers remain in the record with their ~2 pp error;
this is a higher-resolution observation, not a retroactive change to the frozen
registration.
