"""Build the KALIA v0.2.0 benchmark notebook.

Audit fix 4: every public number we have quoted (PIQA 61.4, ARC-Easy 45.8, the
6.06-nat gap) is from v0.1.2. v0.2.0 has a loss number and nothing else, so a
reader has no way to know that. Benchmarks are CPU work, so this runs as a
**CPU kernel in parallel with the training session at zero GPU quota cost**,
rather than waiting for the weekly allowance or running anywhere local.

Usage:
    python tools/build_bench_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "kalia-bench-v020.ipynb"

CELLS = [
    (
        "markdown",
        "# KALIA bench v0.2.0 — the registered evaluation suite\n"
        "\n"
        "Runs the same five zero-shot tasks as v0.1.2 (ARC-Easy, PIQA, HellaSwag, LAMBADA,\n"
        "WinoGrande; 0-shot, 500 samples) against the **v0.2.0** checkpoint, so the two\n"
        "versions are finally comparable on the same suite.\n"
        "\n"
        "CPU kernel: no GPU quota. Sourced from the `kalia-train-v020` output; the code\n"
        "dataset must be the latest version, since it supplies `eval_bench.py` and the fix\n"
        "that makes `find_ckpt` prefer the attached training kernel.",
    ),
    (
        "code",
        "# lm-eval 0.4.9 imports transformers.AutoModelForVision2Seq, which was REMOVED in\n"
        "# transformers v5 -- and Kaggle's image now ships v5. Pinning transformers<5 as well\n"
        "# gives a pair that matches. kalia_lm.py does not import transformers at all (it uses\n"
        "# tiktoken + our own model.py), so the downgrade cannot affect our code.\n"
        '!pip install -q "lm-eval==0.4.9" "transformers<5" tiktoken pyyaml datasets\n'
        "import glob, importlib.metadata as md, shutil, sys\n"
        "print(sys.version)\n"
        "import importlib.metadata as md\n"
        "for pkg in ('lm_eval', 'transformers', 'torch', 'datasets'):\n"
        "    try:\n"
        '        print(f"{pkg:14s} {md.version(pkg)}")\n'
        "    except Exception as exc:\n"
        '        print(f"{pkg:14s} MISSING ({exc})")\n'
        "import transformers\n"
        'print("AutoModelForVision2Seq present:", hasattr(transformers, "AutoModelForVision2Seq"))\n'
        "print(''.join(sorted(glob.glob('/kaggle/input/notebooks/subhajitlucky/kalia-train-v020/*'))))",
    ),
    (
        "code",
        "code = sorted(glob.glob(\"/kaggle/input/datasets/subhajitlucky/kalia-code-dev\"))\n"
        "assert code, \"attach the kalia-code-dev dataset\"\n"
        "root = code[0]\n"
        "for f in [\"eval_bench.py\", \"kalia_lm.py\", \"model.py\", \"data.py\", \"eval_reversibility.py\"]:\n"
        "    src = f\"{root}/{f}\"\n"
        "    shutil.copy(src, \"/kaggle/working/\")\n"
        "    print(\"copied\", f)\n"
        "ckpts = sorted(glob.glob(\"/kaggle/input/notebooks/subhajitlucky/kalia-train-v020/**/ckpt.pt\", recursive=True))\n"
        "print(\"checkpoints found:\", ckpts)",
    ),
    (
        "code",
        "import subprocess\n"
        "subprocess.run(\n"
        '    [sys.executable, "eval_bench.py",\n'
        '     "--ckpt", ckpts[0],\n'
        '     "--out", "/kaggle/working/bench_v020.json",\n'
        '     "--limit", "500"],\n'
        "    check=True,\n"
        ")",
    ),

    (
        "code",
        "!cat /kaggle/working/bench_v020.json\n"
        "\n"
        "print(\"\\n=== comparison target: v0.1.2 (published, same suite) ===\")\n"
        "print(\"PIQA 61.4 | ARC-Easy 45.8 | HellaSwag 36.8 | WinoGrande 50.2 | LAMBADA 23.0\")",
    ),
    (
        "markdown",
        "Results are reported as measured, including any regression. Per the registered\n"
        "evaluation, the yardstick is the v2b val set and the comparator is v0.1.2 at 3.0533\n"
        "on that set; the pre-registered secondary metric S-A allows no task to regress by\n"
        "more than 1.0 pp.",
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
