"""Build the X18 F-3/F-4 completion notebook.

X18's registered predictions could not be adjudicated from the GPU run alone:
F-4 (does the gate open?) was never measured because the kernel passed no gate
diagnostic, and F-3's decisive half (">=0.5 pp on a downstream task") needs a
benchmark that was never run. Measured locally, F-4 fails clearly -- mean gate
0.0192 against a 0.05 threshold -- but on 20 short probe sentences, which is
thin evidence for a claim this consequential.

This kernel closes both, on CPU, at zero GPU quota cost, in parallel with the
v0.2.0 training session:

- **F-3**: the same five registered zero-shot tasks against `micro-base` and
  `micro-gated`, so the loss/accuracy question is answered with the accuracy
  number rather than argued about.
- **F-4**: mean gate value over 100 real 512-token validation batches from the
  canonical v2b val set, which is the measurement the prediction was written
  against ("mean gate value at the end"), replacing the 20-sentence estimate.

Completing a registered metric is not amending a pre-registration, so no
amendment hash is required: the arms, steps, seed, and thresholds are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "kalia-x18-close.ipynb"

CELLS = [
    (
        "markdown",
        "# KALIA X18 — close F-3 and F-4\n"
        "\n"
        "X18 ran three arms (control / NoPE / gated) and the GPU kernel measured only\n"
        "validation loss. Two registered predictions were therefore left unadjudicated:\n"
        "\n"
        "- **F-3** — *gated improves accuracy by more than it improves loss*: needs the\n"
        "  accuracy half, which was never run.\n"
        "- **F-4** — *the gate actually opens (mean > 0.05)*: needs a gate diagnostic,\n"
        "  which the kernel did not emit.\n"
        "\n"
        "F-1 and F-2 are already settled from validation loss alone and need nothing\n"
        "here. This kernel supplies the two missing measurements. CPU, so it costs no\n"
        "GPU quota and runs alongside the v0.2.0 session.",
    ),
    (
        "code",
        "# Same pin as the v0.2.0 bench kernel: lm-eval 0.4.9 imports\n"
        "# AutoModelForVision2Seq, removed in transformers v5.\n"
        '!pip install -q "lm-eval==0.4.9" "transformers<5" tiktoken pyyaml datasets\n'
        "import glob, importlib.metadata as md, shutil, sys\n"
        "for pkg in ('lm_eval', 'transformers', 'torch', 'datasets'):\n"
        "    try:\n"
        '        print(f"{pkg:14s} {md.version(pkg)}")\n'
        "    except Exception as exc:\n"
        '        print(f"{pkg:14s} MISSING ({exc})")',
    ),
    (
        "code",
        "root = sorted(glob.glob(\"/kaggle/input/datasets/subhajitlucky/kalia-code-dev\"))[0]\n"
        "for f in [\"eval_bench.py\", \"kalia_lm.py\", \"model.py\", \"data.py\",\n"
        "          \"eval_reversibility.py\", \"gate_probe.py\"]:\n"
        '    shutil.copy(f"{root}/{f}", "/kaggle/working/")\n'
        "    print(\"copied\", f)\n"
        "import os\n"
        'os.chdir("/kaggle/working")\n'
        "ckpts = sorted(glob.glob(\n"
        '    "/kaggle/input/notebooks/subhajitlucky/kalia-x18-ablate/out/*/ckpt.pt"))\n'
        'print("checkpoints:", *ckpts, sep="\\n  ")\n'
        'assert len(ckpts) == 3, f"expected 3 arms, found {len(ckpts)}"',
    ),
    (
        "markdown",
        "## F-4 first — it decides how F-3 should be read\n"
        "\n"
        "If the gate is inert, then any accuracy difference on the gated arm belongs\n"
        "to the branch-normalisation structure rather than to the data-dependent read\n"
        "that Gated Residual actually claims, and F-3's question changes shape. So\n"
        "measure the gate before spending CPU on the benchmark.",
    ),
    (
        "code",
        "gated = [c for c in ckpts if 'micro-gated' in c][0]\n"
        "print('gated checkpoint:', gated)\n"
        "import subprocess\n"
        "subprocess.run(\n"
        "    [sys.executable, 'gate_probe.py',\n"
        "     '--ckpt', gated,\n"
        "     '--json-out', '/kaggle/working/gate_f4_sentences.json'],\n"
        "    check=True,\n"
        ")",
    ),
    (
        "code",
        "# F-4 on the canonical validation set, not the 20 short probe sentences.\n"
        "# The prediction says \"mean gate value at the end\"; the end of training is\n"
        "# best estimated on real val text in full-length windows.\n"
        "import json\n"
        "import numpy as np\n"
        "import torch\n"
        "import gate_probe as gp\n"
        "from eval_reversibility import load_model\n"
        "\n"
        "val_bin = sorted(glob.glob(\n"
        '    "/kaggle/input/notebooks/subhajitlucky/kalia-prep-v2b/**/val.bin",\n'
        "    recursive=True))\n"
        "assert val_bin, \"attach the kalia-prep-v2b notebook output for val.bin\"\n"
        "print('val.bin:', val_bin[0])\n"
        "\n"
        "device = torch.device('cpu')\n"
        "model = load_model(gated, device)\n"
        "ctx = model.cfg.context_len\n"
        "print('context_len:', ctx)\n"
        "\n"
        "arr = np.memmap(val_bin[0], dtype=np.uint16, mode='r')\n"
        "print('val tokens:', len(arr))\n"
        "batches = []\n"
        "for i in range(100):\n"
        "    start = i * ctx\n"
        "    if start + ctx > len(arr):\n"
        "        break\n"
        "    ids = torch.tensor(arr[start:start + ctx], dtype=torch.long).unsqueeze(0)\n"
        "    batches.append(ids)\n"
        "print('batches:', len(batches), 'tokens each:', ctx)\n"
        "res = gp.gate_means(model, batches)\n"
        "res['batches'] = len(batches)\n"
        "res['init_mean'] = gp.INIT_GATE_MEAN\n"
        "res['delta'] = res['overall'] - gp.INIT_GATE_MEAN\n"
        "res['f4_threshold'] = gp.F4_THRESHOLD\n"
        "res['f4_passes'] = bool(res['overall'] > gp.F4_THRESHOLD)\n"
        "for i, v in enumerate(res['per_block']):\n"
        "    print(f'  block {i}: mean gate {v:.4f}')\n"
        "print(f\"  overall {res['overall']:.4f} | init {res['init_mean']:.4f} \"\n"
        "      f\"| delta {res['delta']:+.4f} | threshold {res['f4_threshold']} \"\n"
        "      f\"| F-4 {'PASS' if res['f4_passes'] else 'FAIL'}\")\n"
        "with open('/kaggle/working/gate_f4_val.json', 'w') as fh:\n"
        "    json.dump(res, fh, indent=2)",
    ),
    (
        "markdown",
        "## F-3 — does the accuracy gain exceed the loss gain?\n"
        "\n"
        "Registered threshold: gated must be at least control on val loss (it is, by\n"
        "0.0436) **and** at least 0.5 pp better on a downstream task. Same five tasks,\n"
        "0-shot, 500 samples, as every other number in this project.",
    ),
    (
        "code",
        "import json\n"
        "import subprocess\n"
        "for arm in ('micro-base', 'micro-gated'):\n"
        "    ck = [c for c in ckpts if arm in c][0]\n"
        "    print('=' * 64)\n"
        "    print('ARM:', arm)\n"
        "    subprocess.run(\n"
        "        [sys.executable, 'eval_bench.py',\n"
        "         '--ckpt', ck,\n"
        "         '--out', f'/kaggle/working/bench_{arm}.json',\n"
        "         '--limit', '500'],\n"
        "        check=True,\n"
        "    )",
    ),
    (
        "code",
        "base = json.load(open('/kaggle/working/bench_micro-base.json'))\n"
        "gated = json.load(open('/kaggle/working/bench_micro-gated.json'))\n"
        "\n"
        "def acc(res, task):\n"
        "    m = res[task]\n"
        "    for key in ('acc_norm,none', 'acc,none'):\n"
        "        if key in m:\n"
        "            return m[key] * 100\n"
        "    raise KeyError(task)\n"
        "\n"
        "tasks = sorted(set(base) & set(gated))\n"
        "print('F-3: gated minus control, in accuracy points')\n"
        "print(f\"{'task':24s} {'control':>9s} {'gated':>9s} {'delta':>8s}\")\n"
        "deltas = []\n"
        "for t in tasks:\n"
        "    b, g = acc(base, t), acc(gated, t)\n"
        "    deltas.append(g - b)\n"
        "    flag = '  <- >=0.5pp' if (g - b) >= 0.5 else ''\n"
        "    print(f'{t:24s} {b:9.1f} {g:9.1f} {g-b:+8.2f}{flag}')\n"
        "best = max(deltas)\n"
        "print(f\"\\nbest task delta {best:+.2f} pp | F-3 needs >= 0.5 pp \"\n"
        "      f\"-> {'PASS' if best >= 0.5 else 'FAIL'}\")\n"
        "print('val loss delta was -0.0436 nats (gated better)')",
    ),
    (
        "markdown",
        "Read the two verdicts together, not separately.\n"
        "\n"
        "- **F-4 fail + F-3 fail** → the gate is inert *and* buys no accuracy, so the\n"
        "  −0.0436 nats belongs to the branch normalisation. Keep the normalisation,\n"
        "  drop the gate, and register the isolation arm (X19).\n"
        "- **F-4 fail + F-3 pass** → the accuracy gain is real but is not coming from\n"
        "  the gate, which is a more interesting and more awkward result: a claim\n"
        "  borrowed from Qwen's 125B does not survive being reimplemented at 30M.\n"
        "- **F-4 pass** → the pre-registered F-3 question is answerable as written, and\n"
        "  the loss/accuracy divergence Qwen reported is reproduced at our scale.\n"
        "\n"
        "All outcomes are reported. The pre-registration's own caveat applies: a null at\n"
        "30M over 500 steps is weak evidence about 125B, and this is a screening run, not\n"
        "a training run.",
    ),
]


def main() -> None:
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
