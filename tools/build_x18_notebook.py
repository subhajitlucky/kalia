"""Build the X18 micro-ablation notebook (NoPE vs Gated Residual vs control).

Three arms, one variable each, 30M / 500 steps, seed 1337. Scheduled into the
batch-GPU slot that v0.2.0 session 2's completion freed, so it costs the main run
nothing except the ~1.5h it occupies.

Uses the compliance-clean v2b corpus (the same one v0.2.0 trains on), because
X17/X18 are only meaningful against the corpus whose structure we measured in
I15 -- the v1 prep corpus has a different document mixture.

Usage:
    python tools/build_x18_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "kalia-x18-ablate.ipynb"

CELLS = [
    (
        "markdown",
        "# X18 — two 2026 frontier changes, screened at 30M\n"
        "\n"
        "**NoPE** (SmolLM3): drop rotary embeddings on every 4th layer. Parameter-neutral by\n"
        "construction, so a difference cannot be attributed to capacity.\n"
        "\n"
        "**Gated Residual** (Qwen3.8): data-dependent elementwise gate on the residual path.\n"
        "Only the gating half is implemented, not the four-branch widening — the\n"
        "pre-registration says so, and the arm is named `gated` for that reason.\n"
        "\n"
        "Control vs the two, 500 steps, seed 1337, same data. Pre-registered at\n"
        "`docs/preregistrations/2026-09-28-X18-frontier-attention.md` (hash 4a4b…).",
    ),
    (
        "code",
        "import glob, os, shutil, subprocess, sys\n"
        "\n"
        "work = \"/kaggle/working/kalia\"\n"
        "if not os.path.exists(work):\n"
        '    hits = sorted(glob.glob("/kaggle/input/**/ablate.py", recursive=True))\n'
        '    assert hits, "attach the kalia-code-dev dataset (must be the NEW version)"\n'
        "    shutil.copytree(os.path.dirname(hits[0]), work)\n"
        "os.chdir(work)\n"
        "print(\"code from\", work)\n"
        "data = sorted(glob.glob(\"/kaggle/input/**/train.bin\", recursive=True))\n"
        'assert data, "train.bin not found - attach the kalia-prep-v2b output"\n'
        'print("data:", data[0])',
    ),
    (
        "code",
        'print(subprocess.run([sys.executable, "ablate.py",\n'
        '                       "--arms", "micro-base,micro-nope,micro-gated",\n'
        '                       "--steps", "500",\n'
        '                       "--data-dir", str(os.path.dirname(data[0])),\n'
        '                       "--out-root", "/kaggle/working/out"],\n'
        "                      capture_output=True, text=True).stdout[-4000:])",
    ),
    (
        "code",
        "import pandas as pd\n"
        "for arm in [\"micro-base\", \"micro-nope\", \"micro-gated\"]:\n"
        "    p = f\"/kaggle/working/out/{arm}/val_log.csv\"\n"
        "    try:\n"
        '        df = pd.read_csv(p)\n'
        '        print(f"\\n{arm}: " + "  ".join(f"{int(r.step)}={r.val_loss:.4f}" for r in df.itertuples()))\n'
        "    except Exception as exc:\n"
        '        print(arm, "->", exc)',
    ),
    (
        "markdown",
        "Read against the pre-registered thresholds:\n"
        "- **F-1** NoPE val loss ≤ control\n"
        "- **F-3** gated: val loss ≤ control **and** ≥0.5pp better on a downstream task\n"
        "- **F-4** the gate opens (mean > 0.05 at the end)\n"
        "\n"
        "Results are reported as measured. A null at 30M says little about 125B, and the\n"
        "pre-registration records that caveat in advance.",
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
