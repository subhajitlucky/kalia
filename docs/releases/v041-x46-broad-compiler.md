# v0.4.1 — X46 broad-template compiler

**Capability release — 2026-11-05.** A robust edit compiler for compositional
story interventions, built on the frozen public **v0.3.1** base. The base
model is unchanged; this release adds two adapters and a fresh-template
validation.

## What changed from v0.4.0

v0.4.0 passed on the 2×2 benchmark but only partially generalized to fresh
templates: fresh `rename+swap` was 0.406/0.416 and 3-op depth was
0.283/0.365. X46 trains the same rank-16 LoRA on **17 templates** (ten new
broad templates, all 1/2/3-op combinations) while holding the four X44 test
templates out.

## Result (fresh templates, never seen in training)

| cell | v0.3.1 normal | gold-edit | C_1401 | C_1402 |
|---|---:|---:|---:|---:|
| F-S | 0.3875 | 0.7125 | **0.7475** | **0.7525** |
| F-P | **0.0850** | 0.5725 | **0.6825** | **0.6850** |
| F-T (3-op) | **0.0350** | 0.3925 | **0.6150** | **0.5575** |

| operation | v0.3.1 normal | C_1401 | C_1402 |
|---|---:|---:|---:|
| `number+place` | — | **0.4545** | **0.4545** |
| `rename+swap` | — | **0.7228** | **0.6634** |
| `rename+negate` | — | 0.8700 | 0.8700 |
| `swap+negate` | — | 0.6600 | 0.6600 |

Registered bars G-1…G-6 all pass. Compiler exact match on fresh templates:
**0.3550 / 0.3783**; fallback 0.0000. Cost: **0.96 GPU-h** for two seeds.

## What is released

- `lora_broad_seed1401.pt`, `lora_broad_seed1402.pt` — rank-16 adapters,
  **1,413,120** trainable parameters each.
- `broad_templates.py` — the 17-template training data generator.
- `tools/kalia_x46_broad_kernel.py` — training/eval kernel.
- `docs/results/x46-broad/` — summary, logs, generated prefixes, rows.
- Pre-registration and hash ledger entry.

The base checkpoint remains **kalia-v031** (`bbd8a036…`, 57,856,256 params,
val 2.7579, bpB 0.9016). The adapters do not modify it.

## Usage sketch

```python
from edit_compiler import apply_lora, load_lora, generate_edited_prefix

apply_lora(base, rank=16, alpha=16)
load_lora(base, "lora_broad_seed1402.pt")

record = {"instruction": "Intervention: rename Bella to Eli; swap Eli and Tara.",
          "prefix": "Bella gave Tara the shiny book."}
edited = generate_edited_prefix(base, record, enc, device)
# score the candidate continuations conditioned on `edited`
```

The compiler is a prompt-to-edited-prefix adapter, not a chat model.

## What this is not

- **Not a new base model.** The public base remains v0.3.1.
- **Not a standard-benchmark gain.** The five-task zero-shot suite is
  unchanged; the advancement is compositional editing.
- **Not a faithfulness claim.** The generated prefixes are reader-friendly and
  not identical to the gold edit; the U-P/F-P results above the gold interface
  are a reader-system effect.
- **Not a held-out-operation-combination claim.** X46 trains all pair types on
  broad templates and tests on four fresh templates. X41 remains the stricter,
  partial held-out-combination result.

## Record

- Result: `docs/results/x46-broad/README.md`
- X44 boundary: `docs/results/x44-fresh/README.md`
- X43 release: `docs/releases/v040-x43-edit-compiler.md`
- Pre-registration: `docs/preregistrations/2026-11-05-X46-broad-compiler.md`
- Licence: repository `LICENSE`.
