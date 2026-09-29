"""Build the X19 notebook: one new arm, reusing X18's control and gated results.

X19 runs a single arm. X18's `micro-base` (4.7662) and `micro-gated` (4.7226)
already exist from the same seed, the same data, the same step count and the
same code lineage, and their logs are published — re-running them would spend
GPU to reproduce numbers we have and would make X19's comparison *less* clean,
because a fresh run is a fresh draw rather than the exact arm G-1's threshold is
written against.

So the kernel trains `micro-branchnorm` only, and the comparison is done in the
notebook against the frozen X18 numbers. The threshold from the pre-registration
is a midpoint:

    G-1: branchnorm <= midpoint(control, gated) + 0.010
       = (4.7662 + 4.7226)/2 + 0.010 = 4.7544

G-2 and G-3 are construction-time checks that need no training and are asserted
locally in `test_branch_norm.py`; this kernel re-asserts G-2 from the trained
checkpoint so the arm that actually ran is the arm that was checked.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "kalia-x19-branchnorm.ipynb"

# Frozen from X18 (docs/preregistrations/2026-09-28-X18-frontier-attention.md).
CONTROL = 4.7662
GATED = 4.7226
G1_BAR = round((CONTROL + GATED) / 2 + 0.010, 4)  # 4.7544
EXPECTED_BRANCHNORM_PARAMS = 29_922_816

CELLS = [
    (
        "markdown",
        "# KALIA X19 — is it the gate, or the branch structure?\n"
        "\n"
        "X18's `micro-gated` arm beat control by **0.0436 nats**, 4.4× the registered\n"
        "bar — the largest effect any architecture arm has produced here. It was also\n"
        "unattributable, because the gate was measured as non-responsive to its input\n"
        "(G-0: std across inputs 5e-06 against a mean of 0.019).\n"
        "\n"
        "So this trains **one arm**: the same four-branch normalisation with the gate\n"
        "removed entirely.\n"
        "\n"
        "| arm | params | val loss | source |\n"
        "|---|---|---|---|\n"
        f"| control (`micro-base`) | 29,920,512 | {CONTROL} | X18, not re-run |\n"
        f"| gated (`micro-gated`) | 30,072,768 | {GATED} | X18, not re-run |\n"
        "| **branchnorm** (`micro-branchnorm`) | 29,922,816 | *this run* | X19 |\n"
        "\n"
        "Control and gated are **not re-run**: same seed, data, steps and lineage, and\n"
        "their logs are published. Re-running would cost quota to reproduce numbers we\n"
        "have, and would replace a fixed reference with a fresh draw.\n"
        "\n"
        "**G-1 bar: val loss ≤ 4.7544** (midpoint of control and gated, +0.010).\n"
        "\n"
        "If branchnorm clears that, the −0.0436 is the branch structure and the gate's\n"
        "149,952 parameters are being spent on a constant. If it does not, the gated\n"
        "arm's result was an artifact and the whole idea goes.",
    ),
    (
        "code",
        "import glob, os, shutil, subprocess, sys\n"
        "\n"
        "work = '/kaggle/working/kalia'\n"
        "if not os.path.exists(work):\n"
        '    hits = sorted(glob.glob("/kaggle/input/**/ablate.py", recursive=True))\n'
        '    assert hits, "attach the kalia-code-dev dataset (must be current: it carries configs/ and eval/)"\n'
        "    shutil.copytree(os.path.dirname(hits[0]), work)\n"
        "os.chdir(work)\n"
        "print('code from', work)\n"
        "print('branch_norm in model.py:', 'class BranchNorm' in open('model.py').read())\n"
        "data = sorted(glob.glob('/kaggle/input/**/train.bin', recursive=True))\n"
        "assert data, 'train.bin not found - attach the kalia-prep-v2b output'\n"
        "print('data:', data[0])\n"
        "cfg = open('configs/micro-branchnorm.yaml').read() if os.path.exists('configs/micro-branchnorm.yaml') else ''\n"
        "print('arm config present:', bool(cfg))",
    ),
    (
        "code",
        "print(subprocess.run([sys.executable, 'ablate.py',\n"
        "                       '--arms', 'micro-branchnorm',\n"
        "                       '--steps', '500',\n"
        "                       '--data-dir', str(os.path.dirname(data[0])),\n"
        "                       '--out-root', '/kaggle/working/out'],\n"
        "                      capture_output=True, text=True).stdout[-4000:])",
    ),
    (
        "code",
        "import pandas as pd\n"
        "p = '/kaggle/working/out/micro-branchnorm/val_log.csv'\n"
        "df = pd.read_csv(p)\n"
        "print('micro-branchnorm: ' + '  '.join(f\"{int(r.step)}={r.val_loss:.4f}\" for r in df.itertuples()))\n"
        "loss = float(df.iloc[-1].val_loss)\n"
        "print(f'\\nfinal val loss {loss:.4f}')",
    ),
    (
        "code",
        "# G-2 re-asserted on the checkpoint that actually trained, not on a\n"
        "# freshly constructed model. Amendment 1 corrected the expected count from\n"
        "# 29,920,512 to 29,922,816 (RMSNorm is affine, so the branch norm is not\n"
        "# free) and the gate's own share to 149,952.\n"
        "import torch\n"
        "ck = torch.load('/kaggle/working/out/micro-branchnorm/ckpt.pt',\n"
        "                 map_location='cpu', weights_only=False)\n"
        "sd = ck['model']\n"
        "# Count *parameters*, not state-dict entries. lm_head is tied to tok_emb, so\n"
        "# summing sd.values() counts the embedding twice: 29,922,816 + 19,298,688 =\n"
        "# 49,221,504, which is what version 1 of this kernel reported and called a\n"
        "# FAIL. The model was correct; the check was not.\n"
        "n_params = sum(v.numel() for k, v in sd.items() if k != 'lm_head.weight')\n"
        f"EXPECTED = {EXPECTED_BRANCHNORM_PARAMS}\n"
        "print(f'G-2: checkpoint params {n_params:,} | expected {EXPECTED:,} | '\n"
        "      + ('PASS' if n_params == EXPECTED else 'FAIL'))\n"
        "gates = [k for k in sd if '.gated.' in k]\n"
        "branches = [k for k in sd if '.branch.' in k]\n"
        "print(f'G-2: gate tensors present: {len(gates)} (expected 0 for this arm)')\n"
        "print(f'G-2: branch tensors present: {len(branches)}')\n"
        "assert not gates, 'this arm must carry no gate parameters'\n"
        "assert n_params == EXPECTED, f'parameter count {n_params} != {EXPECTED}'\n"
        "print('G-2 PASS')",
    ),
    (
        "code",
        f"CONTROL, GATED, BAR = {CONTROL}, {GATED}, {G1_BAR}\n"
        "print('=== X19 G-1 ===')\n"
        "print(f'  control      {CONTROL:.4f}')\n"
        "print(f'  gated        {GATED:.4f}   ({GATED-CONTROL:+.4f} vs control)')\n"
        "print(f'  branchnorm   {loss:.4f}   ({loss-CONTROL:+.4f} vs control, {loss-GATED:+.4f} vs gated)')\n"
        "print(f'  G-1 bar      {BAR:.4f}   (midpoint + 0.010)')\n"
        "print()\n"
        "if loss <= BAR:\n"
        "    print(f'  G-1 PASS  branchnorm captures >= half the gated arm gain')\n"
        "    print('  => the -0.0436 is the BRANCH STRUCTURE, not the gate.')\n"
        "    print('  => the gate spends 149,952 parameters on a constant (G-0).')\n"
        "    print('  => promote branch normalisation to v0.3.0; drop the gate.')\n"
        "else:\n"
        "    print(f'  G-1 FAIL  branchnorm is {loss-BAR:+.4f} outside the bar')\n"
        "    print('  => the gain is NOT explained by the branch structure alone,')\n"
        "    print('     so either the gate does something G-0 could not detect, or')\n"
        "    print('     X18s -0.0436 was a single-seed artifact. Both are negative')\n"
        "    print('     results and are reported as such.')\n"
        "print()\n"
        "print('Caveat carried from X18 and still binding here: single seed, 30M '\n"
        "      'parameters, 500 steps. A screening window, not a training run.')",
    ),
    (
        "markdown",
        "Read alongside X18, not on its own.\n"
        "\n"
        "- **G-1 pass** → X18's gain is attributed: branch normalisation, gate inert.\n"
        "  A free architectural change and a 149,952-parameter deletion.\n"
        "- **G-1 fail** → the branch structure does not explain the gain. Combined with\n"
        "  G-0's zero input-dependence, that leaves X18's −0.0436 most likely a\n"
        "  single-seed artifact, which would be the honest conclusion and would mean\n"
        "  the gate *and* the branch norm both go.\n"
        "\n"
        "Either way the registered thresholds are reported as written. G-1 is a\n"
        "loss threshold in nats, so unlike X18's accuracy bar it is not sitting below\n"
        "the measurement's own noise (see D45).",
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
    print(f"wrote {OUT} ({len(cells)} cells) | G-1 bar {G1_BAR}")


if __name__ == "__main__":
    main()
