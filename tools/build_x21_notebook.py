"""Build the X21 replication notebook: four arms, paired by seed, one kernel.

Sequential inside one kernel, not interleaved. They share a machine and a session,
which is what the pre-registration claimed, but calling it interleaved would
overstate it -- the modded-nanogpt standard alternated arm and control across
runs to cancel drift, and this does not. The difference is stated rather than
glossed, because the write-up will have to say how weak the replication is.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "kalia-x21-replication.ipynb"

# Frozen from the pre-registration. Not to be edited here.
SEEDS = (1338, 1339)
GATED = 4.7226          # X18, seed 1337
X20_STATICGATE = 4.6304  # X20, seed 1337
R1_BAR = -0.050         # mean paired delta must be at least this negative
R3_STD_FLOOR = 1e-3     # static gate channel dispersion

CELLS = [
    (
        "markdown",
        "# KALIA X21 \\u2014 is the static-gate result real?\n"
        "\n"
        "X20 scored **4.6304** against control's 4.7662, clearing its pre-registered bar\n"
        "by 0.1240. A gate that never reads its input beat one that does by 0.0922\n"
        "nats, with 73,728 fewer parameters. **That is 3.1\\u00d7 the largest effect any\n"
        "architecture arm has produced here**, and it rests on one seed \\u2014 as do all of\n"
        "X18, X19 and X20.\n"
        "\n"
        "Four arms: control and treatment at two fresh seeds, in one kernel so they share a\n"
        "machine and a session.\n"
        "\n"
        "| arm | seed |\n"
        "|---|---|\n"
        "| control | 1338 |\n"
        "| staticgate | 1338 |\n"
        "| control | 1339 |\n"
        "| staticgate | 1339 |\n"
        "\n"
        "Seed 1337 is **excluded**: it generated the hypothesis, and including it would be\n"
        "circular.\n"
        "\n"
        "- **R-1** mean paired delta \\u2264 \\u22120.050\n"
        "- **R-2** *both* individual deltas \\u2264 \\u22120.050 \\u2014 the one that matters, because\n"
        "  a mean can be carried by a single wild seed\n"
        "- **R-3** static gate channel std > 1e-03, closing the gap X20 left open\n"
        "- **R-4** staticgate still beats gated (4.7226)\n"
        "\n"
        "Sequential, not interleaved: they share a machine and a session, but do not\n"
        "alternate to cancel drift the way the modded-nanogpt records do. Said plainly here\n"
        "because the write-up will have to say how weak this is.",
    ),
    (
        "code",
        "import glob, os, shutil, sys\n"
        "\n"
        "work = '/kaggle/working/kalia'\n"
        "if not os.path.exists(work):\n"
        "    hits = sorted(glob.glob('/kaggle/input/**/ablate.py', recursive=True))\n"
        "    assert hits, 'attach the kalia-code-dev dataset'\n"
        "    shutil.copytree(os.path.dirname(hits[0]), work)\n"
        "os.chdir(work)\n"
        "print('code from', work)\n"
        "print('static_gate in model.py:', 'static_gate' in open('model.py').read())\n"
        "data = sorted(glob.glob('/kaggle/input/**/train.bin', recursive=True))\n"
        "assert data, 'train.bin not found - attach the kalia-prep-v2b output'\n"
        "print('data:', data[0])\n"
        "arms = ['micro-base-s1338', 'micro-staticgate-s1338',\n"
        "        'micro-base-s1339', 'micro-staticgate-s1339']\n"
        "for a in arms:\n"
        "    p = f'configs/{a}.yaml'\n"
        "    assert os.path.exists(p), f'missing {p}'\n"
        "    txt = open(p).read()\n"
        "    print(f'  {a:28s} seed={txt.split(\"seed:\")[1].split()[0]}'\n"
        "          f' static_gate={\"static_gate: true\" in txt}')",
    ),
    (
        "code",
        "print(subprocess.run([sys.executable, 'ablate.py',\n"
        "                       '--arms', ','.join(arms),\n"
        "                       '--steps', '500',\n"
        "                       '--data-dir', str(os.path.dirname(data[0])),\n"
        "                       '--out-root', '/kaggle/working/out'],\n"
        "                      capture_output=True, text=True).stdout[-5000:])",
    ),
    (
        "code",
        "import pandas as pd\n"
        "losses = {}\n"
        "for a in arms:\n"
        "    p = f'/kaggle/working/out/{a}/val_log.csv'\n"
        "    df = pd.read_csv(p)\n"
        "    line = '  '.join(f\"{int(r.step)}={r.val_loss:.4f}\" for r in df.itertuples())\n"
        "    print(f'{a:28s} {line}')\n"
        "    losses[a] = float(df.iloc[-1].val_loss)\n"
        "json.dump(losses, open('/kaggle/working/x21_losses.json', 'w'), indent=2)",
    ),
    (
        "code",
        f"SEEDS = {SEEDS}\n"
        f"GATED, X20_STATICGATE, R1_BAR = {GATED}, {X20_STATICGATE}, {R1_BAR}\n"
        "pairs = []\n"
        "print('=== paired deltas (staticgate - control) ===')\n"
        "print(f\"{'seed':>6s} {'control':>9s} {'staticgate':>11s} {'delta':>9s}  R-2\")\n"
        "for s in SEEDS:\n"
        "    c = losses[f'micro-base-s{s}']\n"
        "    g = losses[f'micro-staticgate-s{s}']\n"
        "    d = g - c\n"
        "    pairs.append(d)\n"
        "    print(f'{s:>6d} {c:9.4f} {g:11.4f} {d:+9.4f}  '\n"
        "          + ('PASS' if d <= R1_BAR else 'FAIL'))\n"
        "mean_d = sum(pairs) / len(pairs)\n"
        "r1 = mean_d <= R1_BAR\n"
        "r2 = all(d <= R1_BAR for d in pairs)\n"
        "print(f'\\nmean paired delta {mean_d:+.4f} | R-1 (mean <= {R1_BAR}) '\n"
        "      + ('PASS' if r1 else 'FAIL'))\n"
        "print(f'R-2 (every seed <= {R1_BAR}) ' + ('PASS' if r2 else 'FAIL'))\n"
        "ctl = [losses[f'micro-base-s{s}'] for s in SEEDS]\n"
        "print(f'\\ncontrol seed spread: {min(ctl):.4f} .. {max(ctl):.4f} '\n"
        "      f'= {max(ctl)-min(ctl):.4f} nats')\n"
        "print('  this is the first time we have measured it; every delta in this')\n"
        "print('  project has silently assumed the control is seed-invariant.')\n"
        "res = {'pairs': pairs, 'mean_delta': mean_d, 'r1': r1, 'r2': r2,\n"
        "       'control_spread': max(ctl) - min(ctl), 'losses': losses,\n"
        "       'x20_staticgate': X20_STATICGATE, 'gated': GATED}\n"
        "json.dump(res, open('/kaggle/working/x21_result.json', 'w'), indent=2)",
    ),
    (
        "code",
        "print('=== verdict, fixed in the pre-registration ===')\n"
        "if res['r1'] and res['r2']:\n"
        "    print('  REPLICATED. Report as \"replicated at two seeds\", never as confirmed.')\n"
        "    print(f\"  staticgate mean {sum(losses[f'micro-staticgate-s{s}'] for s in SEEDS)/2:.4f} \"\n"
        "          f\"vs gated {GATED}: R-4 \" + ('PASS' if sum(losses[f'micro-staticgate-s{s}'] for s in SEEDS)/2 <= GATED else 'FAIL'))\n"
        "    print('  => the published data-dependent gate is dominated by a strictly')\n"
        "    print('     simpler variant of itself at our scale.')\n"
        "elif res['r1'] and not res['r2']:\n"
        "    print('  NOT REPLICATED. One seed is carrying it. Do not ship.')\n"
        "elif not res['r1']:\n"
        "    print('  X20 WAS A SINGLE-SEED ARTIFACT. Report that plainly and close')\n"
        "    print('  the gated-residual line. This is the third architecture arm to')\n"
        "    print('  produce an effect that did not survive scrutiny.')\n"
        "print()\n"
        "print(f\"X20 reported {X20_STATICGATE:.4f} on seed 1337; this run's staticgate\")\n"
        "print('mean is', round(sum(losses[f'micro-staticgate-s{s}'] for s in SEEDS)/2, 4), '-')\n"
        "print('the gap between the first observation and the replication is itself')\n"
        "print('the finding, and it is why R-2 was registered separately.')",
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
    print(f"wrote {OUT} ({len(cells)} cells) | R-1 bar {R1_BAR}")


if __name__ == "__main__":
    main()
