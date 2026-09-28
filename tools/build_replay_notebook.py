"""Build the KALIA replay-shard notebook from cells.

The other notebooks in this repo are committed directly; this one is generated so
the source of truth for the CL-1 pipeline stays a script that is unit-tested
(`make_replay_shard.py`) rather than a JSON file nobody can lint.

Usage:
    python tools/build_replay_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "kalia-replay.ipynb"

CELLS = [
    (
        "markdown",
        "# KALIA replay shard — carve the frozen corpus slice that makes continual learning work\n"
        "\n"
        "CL-1. Replay only protects the past if the replayed tokens are ones the daily run would\n"
        "otherwise *not* see. This kernel therefore does two things: it carves `replay.bin` from\n"
        "the v2b `train.bin`, and it writes `replay_manifest.json` recording the exact token\n"
        "ranges. Any future 'new data' feed must skip those ranges, or the replay is a no-op.\n"
        "\n"
        "Runs on **CPU only** (`enable_gpu: false`), so it consumes no GPU quota.\n"
        "\n"
        "Inputs: the output of `kalia-prep-v2b` (attached as a kernel source).",
    ),
    ("code", "!mkdir -p /kaggle/working/data\n!ls -lh /kaggle/input/notebooks/subhajitlucky/kalia-prep-v2b/data/"),
    (
        "code",
        "import glob\n"
        "import shutil\n"
        "import subprocess\n"
        "import sys\n"
        "\n"
        "def find(name):\n"
        '    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))\n'
        '    assert hits, f"{name} not found - attach the kalia-prep-v2b output"\n'
        "    return hits[0]\n"
        "\n"
        "train_bin = find(\"train.bin\")\n"
        "shutil.copy(\"make_replay_shard.py\", \"/kaggle/working/\")\n"
        "print(\"train.bin:\", train_bin)",
    ),
    (
        "code",
        "subprocess.run(\n"
        '    [sys.executable, "make_replay_shard.py",\n'
        '     "--train-bin", train_bin,\n'
        '     "--out-dir", "/kaggle/working/data",\n'
        '     "--fraction", "0.10",\n'
        '     "--blocks", "32",\n'
        '     "--seed", "1337"],\n'
        "    check=True,\n"
        ")",
    ),

    ("code", "!ls -lh /kaggle/working/data/\n!cat /kaggle/working/data/replay_manifest.json"),
    (
        "markdown",
        "Done. `replay.bin` plus `replay_manifest.json` are the kernel output.\n"
        "\n"
        "Attach this run's output to the training kernel together with a **new-data shard that\n"
        "excludes the manifest's ranges**, and train with `replay_prob: 0.10` and\n"
        "`--replay-bin <this output>/replay.bin`.\n"
        "\n"
        "Measure before and after with `tools/forgetting_probe.py` — that is the only way to know\n"
        "whether the replay did anything.",
    ),
]


def main() -> None:
    cells = []
    for kind, source in CELLS:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
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
