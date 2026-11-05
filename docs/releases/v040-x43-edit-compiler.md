# v0.4.0-x43-edit-compiler

**Capability release — 2026-11-05.** A learned edit compiler for compositional
story interventions, built on the frozen public **v0.3.1** base. The base
model is unchanged; this release adds a modular adapter pack and the evaluation
that measures it.

## What it does

The compiler maps

```
instruction + factual prefix  →  post-intervention prefix
```

and the frozen v0.3.1 reader then scores candidate continuations in that
edited world. It turns an edit-application failure into a continuation task
the base model already handles.

## Result

Full 1,600-record 2×2 composition benchmark (400 per cell; pair chance 0.25):

| cell | v0.3.1 normal | gold-edit interface | X43 C_1401 | X43 C_1402 |
|---|---:|---:|---:|---:|
| **S-S** seen / single | 0.4500 | 0.7025 | 0.7025 | 0.7025 |
| **S-P** seen / pair | **0.1700** | 0.5125 | **0.4800** | **0.4925** |
| **U-S** unseen / single | 0.3150 | 0.8775 | 0.8200 | 0.8150 |
| **U-P** unseen / pair | **0.0925** | 0.7825 | **0.8600** | **0.8725** |

Hard operations:

| operation | v0.3.1 normal | X43 C_1401 | X43 C_1402 |
|---|---:|---:|---:|
| `number+place` | **0.0000** | **0.6935** | **0.6935** |
| `rename+swap` | **0.1871** | **0.5918** | **0.6463** |
| `rename+negate` | 0.1455 | 0.8216 | 0.8216 |
| `swap+negate` | 0.1124 | 0.5976 | 0.5621 |

Registered bars D-1…D-6 all pass. Compiler exact match on the benchmark is
**0.5469 / 0.5438**; singles-only dev exact match is **0.9920 / 0.9977**.
Adapters are rank-16 LoRA, **1,413,120** trainable parameters, base frozen.
End-to-end overall is **0.7156 / 0.7206** against the gold-edit interface's
0.7188.

## What is released

- `lora_armC_seed1401.pt`, `lora_armC_seed1402.pt` — two compiler adapters
  (~5.5 MB each).
- `edit_compiler.py` — data generation, LoRA loader, greedy compiler, and
  end-to-end scorer.
- `tools/kalia_x43_thpt_kernel.py` — the training/eval kernel.
- `docs/results/x43-thpt/` — summary, logs, generated prefixes, per-record
  rows, benchmark audit.
- Pre-registration and hash ledger entry.

The base checkpoint remains **kalia-v031** (`bbd8a036…`, 57,856,256 params,
val 2.7579, bpB 0.9016). The adapters do not modify it.

## Usage sketch

```python
from edit_compiler import apply_lora, load_lora, generate_edited_prefix

# base: the public v0.3.1 GPT, loaded as usual
apply_lora(base, rank=16, alpha=16)
load_lora(base, "lora_armC_seed1402.pt")

record = {"instruction": "Intervention: rename Bella to Eli; swap Eli and Tara.",
          "prefix": "Bella gave Tara the shiny book."}
edited = generate_edited_prefix(base, record, enc, device)
# score the candidate continuations conditioned on `edited`
```

The compiler is not a chat model; it is a prompt-to-edited-prefix adapter for
the KALIA reader.

## What this is not

- **Not a new base model.** The public base remains v0.3.1.
- **Not a standard-benchmark gain.** The five-task zero-shot suite is
  unchanged; the advancement is on the compositional-edit benchmark.
- **Not the stricter held-out-operation-combination claim.** X43 trains the
  four pair types on three disjoint templates and tests on the benchmark
  templates. The stricter X41 result (pair types held out entirely) is a
  partial advancement: `number+place` 0.000 → 0.694 both seeds, S-P
  0.428/0.410, but `rename+swap` 0.354/0.483 and one seed misses 0.45.
- **Not evidence that the generated world is more faithful.** The U-P result
  above the gold-prefix interface is a reader-system effect; generated
  prefixes are reader-friendly, not identical to the gold edit (EM ≈ 0.545).

## Record

- Result: `docs/results/x43-thpt/README.md`
- X41 partial result: `docs/results/x41-edit-compiler/README.md`
- X42 null: `docs/results/x42-reader-ensemble/README.md`
- Pre-registration: `docs/preregistrations/2026-11-05-X43-template-pairs.md`
- Licence: repository `LICENSE`.
