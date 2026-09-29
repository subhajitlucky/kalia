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
        "for pkg in ('lm_eval', 'transformers', 'datasets'):\n"
        "    try:\n"
        "        print(f'{pkg:14s}', __import__('importlib.metadata', fromlist=['x']).version(pkg))\n"
        "    except Exception as exc:\n"
        "        print(f'{pkg:14s} MISSING ({exc})')",
    ),
    (
        "code",
        "root = sorted(glob.glob('/kaggle/input/datasets/subhajitlucky/kalia-code-dev'))[0]\n"
        "for f in ['eval_bench.py', 'kalia_lm.py', 'model.py', 'data.py']:\n"
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
        "# The fix: route ONLY EleutherAI/lambada_openai to its own parquet files.\n"
        "# Everything else is passed straight through, so no other task's\n"
        "# measurement can be affected by this patch.\n"
        "import datasets\n"
        "from huggingface_hub import HfApi\n"
        "\n"
        "REPO = 'EleutherAI/lambada_openai'\n"
        "# The repo ships one parquet file per language config (default/de/en/es/fr/it).\n"
        "# Loading them all concatenates six copies of the task: 5,153 x 6 = 30,918.\n"
        "# Version 1 of this kernel hit exactly that and the row-count assertion caught\n"
        "# it, so the config is selected rather than globbed.\n"
        "all_files = [f for f in HfApi().list_repo_files(REPO, repo_type='dataset')\n"
        "             if f.endswith('.parquet')]\n"
        "urls = [f'https://huggingface.co/datasets/{REPO}/resolve/main/{f}'\n"
        "        for f in all_files if '/en/' in f or '/default/' in f]\n"
        "assert len(urls) == 1, f'expected exactly one English LAMBADA file, got {urls}'\n"
        "print('parquet files in repo:', len(all_files), '| using:', urls[0].rsplit('/', 1)[-1])\n"
        "\n"
        "_orig_load = datasets.load_dataset\n"
        "\n"
        "def patched(path, *args, **kwargs):\n"
        "    if isinstance(path, str) and 'lambada_openai' in path:\n"
        "        print('  [patched] serving', REPO, 'from parquet')\n"
        "        kwargs.pop('name', None)\n"
        "        kwargs.pop('trust_remote_code', None)\n"
        "        return _orig_load('parquet', data_files=urls, split='train', **kwargs)\n"
        "    return _orig_load(path, *args, **kwargs)\n"
        "\n"
        "datasets.load_dataset = patched\n"
        "import lm_eval.tasks\n"
        "print('patch installed')",
    ),
    (
        "code",
        "ds = datasets.load_dataset(REPO)\n"
        "print('rows:', len(ds), '| columns:', list(ds.column_names))\n"
        "print('sample:', repr(ds[0]['text'][:160]))\n"
        "assert len(ds) == 5153, f'expected the standard 5,153 LAMBADA examples, got {len(ds)}'\n"
        "print('row count matches the standard task')",
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
        "    print('STDERR:', rc.stderr[-3000:])",
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
