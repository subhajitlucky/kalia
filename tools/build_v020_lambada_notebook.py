"""Build the LAMBADA completion kernel.

The v0.2.0 final evaluation returned four of five tasks. LAMBADA did not run:
`EleutherAI/lambada_openai` still carries a `lambada_openai.py` loading script,
and modern `datasets` refuses to execute dataset scripts, so lm-eval's task
definition -- which points at the repo by id -- failed to load. This is the same
failure as `deepmind/pg19` in the long-form probe, and it has now bitten the
registered evaluation itself.

The data is not missing. The repo carries six parquet files and the standard
5,153 examples with a single `text` column; only the loader path lm-eval requests
is dead.

So the fix is deliberately narrow: **patch the loader, not the measurement.**
`datasets.load_dataset` is wrapped so that a request for `lambada_openai`
resolves to the repo's own parquet files, and everything else is untouched. lm-eval
still does its own scoring, which is the whole point -- a reimplemented LAMBADA
scorer would produce a number that is not comparable with v0.1.2's published
23.0, and S-A would then be comparing two different measurements.

Without this task, S-A cannot be adjudicated at all: it is a rule about *no task*
regressing by more than 1.0 pp, and one of the five was never measured.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "kalia-v020-lambada.ipynb"

# v0.1.2, published, same task, same 0-shot/500-sample protocol.
V012_LAMBADA = 23.0
S_A_ALLOWANCE = 1.0

CELLS = [
    (
        "markdown",
        "# KALIA v0.2.0 — the missing fifth task\n"
        "\n"
        "The final evaluation returned PIQA, ARC-Easy, HellaSwag and WinoGrande.\n"
        "LAMBADA raised a loading-script error and was never measured, which leaves\n"
        "**S-A unadjudicable** — the rule is about *no* task regressing by more than\n"
        "1.0 pp, and one of the five has no number.\n"
        "\n"
        "The fix is narrow on purpose. The dataset is present (6 parquet files, the\n"
        "standard 5,153 examples, one `text` column); only lm-eval's requested loader\n"
        "path is dead, because the repo still carries a loading script that modern\n"
        "`datasets` refuses to execute. So `load_dataset` is wrapped to resolve that\n"
        "one repo to its own parquet files, and **lm-eval still does the scoring** — a\n"
        "hand-rolled LAMBADA scorer would give a number that is not comparable with\n"
        "v0.1.2's 23.0, and S-A would be comparing two different measurements.\n"
        "\n"
        "CPU, no quota.",
    ),
    (
        "code",
        '!pip install -q "lm-eval==0.4.9" "transformers<5" tiktoken pyyaml datasets\n'
        "import glob, os, shutil, sys\n"
        "root = sorted(glob.glob('/kaggle/input/datasets/subhajitlucky/kalia-code-dev'))[0]\n"
        "for f in ['eval_bench.py', 'kalia_lm.py', 'model.py', 'data.py',\n"
        "          'eval_reversibility.py', 'lambada_loader.py']:\n"
        "    shutil.copy(f'{root}/{f}', '/kaggle/working/')\n"
        "os.chdir('/kaggle/working')\n"
        "ckpt = sorted(glob.glob(\n"
        "    '/kaggle/input/notebooks/subhajitlucky/kalia-train-v020/**/ckpt.pt', recursive=True))\n"
        "assert ckpt, 'attach the kalia-train-v020 kernel output'\n"
        "CKPT = ckpt[0]\n"
        "print('checkpoint:', CKPT)",
    ),
    (
        "code",
        "# The loader fix lives in lambada_loader.py, in the repo, with tests -- not\n"
        "# inline in this notebook. Two versions of this kernel failed that way: v1\n"
        "# globbed all six parquet files (30,918 rows = six copies of the task), and\n"
        "# v2 filtered on '/en/' believing it selected one file when it matched two.\n"
        "# The module pins the config to what lm-eval 0.4.9 actually requests and\n"
        "# asserts the standard 5,153 rows before any scoring happens.\n"
        "from huggingface_hub import HfApi\n"
        "from lambada_loader import REPO, config_parquets, load_lambada\n"
        "\n"
        "files = HfApi().list_repo_files(REPO, repo_type='dataset')\n"
        "print('configs in repo:', sorted(config_parquets(files)))\n"
        "ds = load_lambada(files)\n"
        "print('rows:', len(ds), '| columns:', list(ds.column_names))\n"
        "print('sample:', repr(ds[0]['text'][:150]))",
    ),
    (
        "code",
        "import subprocess\n"
        "rc = subprocess.run(\n"
        "    [sys.executable, 'eval_bench.py',\n"
        "     '--ckpt', CKPT,\n"
        "     '--tasks', 'lambada_openai',\n"
        "     '--out', '/kaggle/working/bench_lambada.json',\n"
        "     '--limit', '500'],\n"
        "    capture_output=True, text=True,\n"
        ")\n"
        "print(rc.stdout[-3000:])\n"
        "if rc.returncode != 0:\n"
        "    print('STDERR:', rc.stderr[-4000:])\n"
        "# Version 4 ran eval_bench, it failed on a missing module, and the cell\n"
        "# carried on -- so the failure surfaced two cells later as a bare\n"
        "# FileNotFoundError on the results file, with the real cause buried in\n"
        "# captured output nobody printed. Fail here, where the cause is.\n"
        "assert rc.returncode == 0, f'eval_bench failed (exit {rc.returncode}); see STDERR above'\n"
        "import os\n"
        "assert os.path.exists('/kaggle/working/bench_lambada.json'), (\n"
        "    'eval_bench exited 0 but wrote no results file')\n"
        "print('results file written')",
    ),
    (
        "code",
        "import json\n"
        "res = json.load(open('/kaggle/working/bench_lambada.json'))\n"
        "print(json.dumps(res, indent=2))\n"
        "m = res['lambada_openai']\n"
        "got = m.get('acc,none', float('nan')) * 100\n"
        f"REF, ALLOW = {V012_LAMBADA}, {S_A_ALLOWANCE}\n"
        "d = got - REF\n"
        "print()\n"
        "print('=== S-A, fifth task ===')\n"
        "print(f'  v0.1.2  {REF:.1f}')\n"
        "print(f'  v0.2.0  {got:.1f}   ({d:+.2f} vs v0.1.2)')\n"
        "print(f\"  S-A: {'FAIL' if d < -ALLOW else 'ok'} (allowance {ALLOW} pp)\")\n"
        "print()\n"
        "print('=== S-A, all five ===')\n"
        "rows = [\n"
        "    '  PIQA        61.4 ->  63.8   +2.40  ok',\n"
        "    '  ARC-Easy    45.8 ->  42.0   -3.80  FAIL',\n"
        "    '  HellaSwag   36.8 ->  39.8   +3.00  ok',\n"
        "    '  WinoGrande  50.2 ->  50.8   +0.60  ok',\n"
        "]\n"
        "rows.append('  LAMBADA     %.1f ->  %.1f   %+.2f  %s' % (REF, got, d, 'FAIL' if d < -ALLOW else 'ok'))\n"
        "for line in rows:\n"
        "    print(line)\n"
        "print()\n"
        "print('Standard errors on these tasks at limit=500 are about 2.0-2.2 pp (D45), so read')\n"
        "print('the +/-0.6 as noise and the -3.80 as the one signal that is not.')",
    ),
    (
        "markdown",
        "Whatever LAMBADA shows, two things are already settled and should not be\n"
        "re-litigated when this lands:\n"
        "\n"
        "- **P-B passes.** Deterministic val loss **2.8248** against v0.1.2's 3.0533 on\n"
        "  the identical 100-batch protocol — a **+0.2285 nat** improvement, 4.6× the\n"
        "  registered 0.05 bar. bpB 0.9244 vs 0.9992.\n"
        "- **S-A fails on ARC-Easy** at −3.80 pp against a 1.0 pp allowance, which is\n"
        "  registered symmetrically with P-B and is reported as it stands.\n"
        "\n"
        "The interim step-3,470 benchmark showed four of five tasks regressing. The final\n"
        "checkpoint reversed that on three of them: PIQA −0.2 → +2.40, WinoGrande\n"
        "−0.8 → +0.60, HellaSwag +1.8 → +3.00. Only ARC-Easy stayed negative. Whatever\n"
        "conclusion is drawn about the corpus, it should be drawn from the final\n"
        "checkpoint, not the interim one.",
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
