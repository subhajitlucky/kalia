"""Build the v0.2.0 final-evaluation notebook.

v0.2.0 session 3 finished the full schedule (step 4770, 2.50B tokens, 6.33h,
477 contiguous train rows, 19 val rows). Its last training-time val_loss was
2.8832 at step 4750 -- but that is the in-training estimate at
`eval_steps 50`, which the pre-registration explicitly calls "noisy by design"
and "not the headline".

So neither registered number can be read off the training log. This kernel runs
the two that can, on CPU, at zero GPU quota:

- **P-A / P-B / P-C** via the registered primary protocol,
  `eval_val.py --batches 100 --batch-size 8 --seed 1234` (819,200 tokens), against
  the fixed v0.1.2 reference of 3.0533.
- **S-A** via the five zero-shot tasks, against v0.1.2's published numbers.

Both thresholds were hashed before any of these numbers existed, so the verdict
is read off the results rather than negotiated. The corpus changed underneath
this run (D43), which is exactly why the benchmark half matters: the
interim step-3,470 measurement had loss improving while four of five tasks
regressed, and that tension is unresolved until the final checkpoint is scored.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "kalia-v020-final-eval.ipynb"

CELLS = [
    (
        "markdown",
        "# KALIA v0.2.0 — the registered final evaluation\n"
        "\n"
        "Session 3 completed the full 4,770-step schedule. This kernel produces the two\n"
        "numbers the pre-registration actually asks for, on the final checkpoint:\n"
        "\n"
        "| ID | Question | Threshold |\n"
        "|---|---|---|\n"
        "| P-A | matches v0.1.2 on the same yardstick | val loss ≤ **3.0533** |\n"
        "| P-B | a real gain, above eval noise | val loss ≤ **3.0033** |\n"
        "| P-C | a real loss | val loss > **3.1033** |\n"
        "| S-A | no task regresses more than 1.0 pp | benchmark vs v0.1.2 |\n"
        "\n"
        "Protocol is fixed: `eval_val.py --batches 100 --batch-size 8 --seed 1234`,\n"
        "819,200 tokens, the same call that produced v0.1.2's 3.0533. P-C is\n"
        "registered symmetrically with P-B so a regression is reported as one, with no\n"
        "re-runs, seed-shopping, or protocol changes used to escape it.\n"
        "\n"
        "CPU kernel: no GPU quota consumed.",
    ),
    (
        "code",
        '# lm-eval 0.4.9 imports AutoModelForVision2Seq, removed in transformers v5.\n'
        '!pip install -q "lm-eval==0.4.9" "transformers<5" tiktoken pyyaml datasets\n'
        "import glob, importlib.metadata as md, os, shutil, sys\n"
        "for pkg in ('lm_eval', 'transformers', 'torch', 'datasets'):\n"
        "    try:\n"
        '        print(f"{pkg:14s} {md.version(pkg)}")\n'
        "    except Exception as exc:\n"
        '        print(f"{pkg:14s} MISSING ({exc})")',
    ),
    (
        "code",
        "root = sorted(glob.glob(\"/kaggle/input/datasets/subhajitlucky/kalia-code-dev\"))[0]\n"
        "import os\n"
        "for f in [\"eval_val.py\", \"eval_bench.py\", \"kalia_lm.py\", \"model.py\",\n"
        "          \"data.py\", \"eval_reversibility.py\"]:\n"
        "    shutil.copy(f\"{root}/{f}\", \"/kaggle/working/\")\n"
        "    print(\"copied\", f)\n"
        "os.makedirs(\"/kaggle/working/eval\", exist_ok=True)\n"
        "for f in glob.glob(f\"{root}/eval/*.json\"):\n"
        "    shutil.copy(f, \"/kaggle/working/eval/\")\n"
        "os.chdir(\"/kaggle/working\")\n"
        "ckpt = sorted(glob.glob(\n"
        '    "/kaggle/input/notebooks/subhajitlucky/kalia-train-v020/**/ckpt.pt",\n'
        "    recursive=True))\n"
        "print(\"checkpoint:\", ckpt)\n"
        'assert ckpt, "attach the kalia-train-v020 kernel output"\n'
        "CKPT = ckpt[0]\n"
        "val_bin = sorted(glob.glob(\n"
        '    "/kaggle/input/notebooks/subhajitlucky/kalia-prep-v2b/**/val.bin",\n'
        "    recursive=True))\n"
        "print(\"val.bin:\", val_bin)\n"
        'assert val_bin, "attach the kalia-prep-v2b kernel output"',
    ),
    (
        "markdown",
        "## P-A / P-B / P-C — the registered primary protocol\n"
        "\n"
        "100 batches of 8, seed 1234. The yardstick is the canonical v2b `val.bin`\n"
        "that v0.1.2 was re-baselined onto at 3.0533, because the corpus rebuild\n"
        "redefined the held-out set (D43).",
    ),
    (
        "code",
        "import subprocess\n"
        "rc = subprocess.run(\n"
        "    [sys.executable, 'eval_val.py',\n"
        "     '--ckpt', CKPT,\n"
        "     '--val-bin', val_bin[0],\n"
        "     '--batches', '100',\n"
        "     '--batch-size', '8',\n"
        "     '--seed', '1234',\n"
        "     '--out', '/kaggle/working/eval_final.md'],\n"
        "    capture_output=True, text=True,\n"
        ")\n"
        "print(rc.stdout[-2000:])\n"
        "if rc.returncode != 0:\n"
        "    print('STDERR:', rc.stderr[-3000:])\n"
        "assert rc.returncode == 0, 'deterministic eval failed'\n"
        "print(open('/kaggle/working/eval_final.md').read())",
    ),
    (
        "code",
        "# eval_val.py writes MARKDOWN, not JSON. Version 1 of this kernel assumed\n"
        "# JSON and died in json.load with a bare JSONDecodeError, taking the\n"
        "# benchmark half down with it -- the same 'one step fails and the kernel\n"
        "# stops' shape as the x18-close failures. So: parse explicitly, and assert\n"
        "# the parse produced a number rather than trusting a regex to have matched.\n"
        "import json, re\n"
        "md = open('/kaggle/working/eval_final.md').read()\n"
        "m_loss = re.search(r'Val loss:\\s*([0-9.]+)', md)\n"
        "m_bpb = re.search(r'bpB:\\s*([0-9.]+)', md)\n"
        "m_bpt = re.search(r'bytes/token:\\s*([0-9.]+)', md)\n"
        "m_tok = re.search(r'=\\s*([0-9,]+)\\s*tokens', md)\n"
        "assert m_loss, f'could not parse val loss from eval_final.md:\\n{md}'\n"
        "assert m_bpb, f'could not parse bpB from eval_final.md:\\n{md}'\n"
        "loss = float(m_loss.group(1))\n"
        "bpb = float(m_bpb.group(1))\n"
        "print('bytes/token', m_bpt.group(1) if m_bpt else '?',\n"
        "      '| tokens', m_tok.group(1) if m_tok else '?')\n"
        "\n"
        "REF, GOOD, BAD = 3.0533, 3.0033, 3.1033\n"
        "print()\n"
        "print('=== registered thresholds (v0.1.2 = 3.0533 on this exact protocol) ===')\n"
        "print(f'  v0.2.0 val loss {loss:.4f} | bpB {bpb:.4f}')\n"
        "if loss <= GOOD:\n"
        "    print(f'  P-B PASS  improvement of {REF-loss:+.4f} nats vs v0.1.2 (needs <= {GOOD})')\n"
        "    print('  P-A PASS  (P-B implies P-A)')\n"
        "    print('  P-C not triggered')\n"
        "elif loss <= REF:\n"
        "    print(f'  P-A PASS  matches v0.1.2 ({REF-loss:+.4f} nats), inside the noise band')\n"
        "    print('  P-B FAIL  not >= 0.05 better')\n"
        "    print('  P-C not triggered')\n"
        "elif loss <= BAD:\n"
        "    print(f'  P-A FAIL  {loss-REF:+.4f} nats worse than v0.1.2')\n"
        "    print('  matched within noise: between 3.0033 and 3.1033')\n"
        "else:\n"
        "    print(f'  P-C FAIL  regression of {loss-REF:+.4f} nats, reported as one')\n"
        "json.dump({'val_loss': loss, 'bpb': bpb, 'batches': 100,\n"
        "           'batch_size': 8, 'seed': 1234},\n"
        "          open('/kaggle/working/eval_final.json', 'w'), indent=2)",
    ),
    (
        "markdown",
        "## S-A — the five zero-shot tasks\n"
        "\n"
        "This is the half that the interim step-3,470 run left in tension: deterministic\n"
        "loss improved by 0.169 nats while four of five tasks fell, two of them by more\n"
        "than the 1.0 pp allowance. The corpus changed underneath that run, so the final\n"
        "checkpoint is the first one whose benchmark and loss can both be taken at face\n"
        "value against v0.1.2.",
    ),
    (
        "code",
        "subprocess.run(\n"
        "    [sys.executable, 'eval_bench.py',\n"
        "     '--ckpt', CKPT,\n"
        "     '--out', '/kaggle/working/bench_final.json',\n"
        "     '--limit', '500'],\n"
        "    check=True,\n"
        ")",
    ),
    (
        "code",
        "bench = json.load(open('/kaggle/working/bench_final.json'))\n"
        "\n"
        "# v0.1.2, published, same five tasks, same 0-shot/500-sample protocol.\n"
        "V012 = {'piqa': 61.4, 'arc_easy': 45.8, 'hellaswag': 36.8,\n"
        "        'lambada_openai': 23.0, 'winogrande': 50.2}\n"
        "\n"
        "def acc(res, task):\n"
        "    m = res[task]\n"
        "    for key in ('acc_norm,none', 'acc,none'):\n"
        "        if key in m:\n"
        "            return m[key] * 100\n"
        "    raise KeyError(f'{task}: no acc metric, keys={sorted(m)}')\n"
        "\n"
        "print(f\"{'task':18s} {'v0.1.2':>7s} {'v0.2.0':>7s} {'delta':>7s}  S-A\")\n"
        "fails = []\n"
        "for task, ref in V012.items():\n"
        "    got = acc(bench, task)\n"
        "    d = got - ref\n"
        "    bad = d < -1.0\n"
        "    if bad:\n"
        "        fails.append(task)\n"
        "    print(f\"{task:18s} {ref:7.1f} {got:7.1f} {d:+7.2f}  {'FAIL' if bad else 'ok'}\")\n"
        "\n"
        "print()\n"
        "if fails:\n"
        "    verdict = 'S-A: FAIL - ' + ', '.join(fails)\n"
        "else:\n"
        "    verdict = 'S-A: PASS - no task regressed more than 1.0 pp'\n"
        "print(verdict)",
    ),
    (
        "markdown",
        "Read the two verdicts together.\n"
        "\n"
        "- P-B **and** S-A pass → the compliance rebuild is a genuine, publishable\n"
        "  improvement and v0.2.0 ships.\n"
        "- P-B passes, S-A fails → a real loss improvement with a real accuracy cost.\n"
        "  That is a finding about the corpus, not a model defect, and it says the\n"
        "  5%→permissive-Python swap is not cost-free — the honest response is to report\n"
        "  it and stop treating the mixture as inert.\n"
        "- P-C → a regression. Registered symmetrically, so it is reported as one, with\n"
        "  no re-runs or protocol changes used to escape it.\n"
        "\n"
        "The val curve flattened after step 3,250 (2.874–2.940 across the last 1,500\n"
        "steps), the same J2 plateau that stopped v0.1.2 at 73% of its schedule. Worth\n"
        "recording either way: this run spent its last 31% of steps inside noise.",
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
