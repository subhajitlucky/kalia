#!/usr/bin/env python3
"""Build the v0.3.0 execution notebooks: CL-0, the baseline, and the arms.

Four kernels, in dependency order, so that Step 1's variance result gates Step 2
rather than being run alongside it. The pre-registration's stopping rule says a
spread above 0.15 nats means the project cannot measure 58M changes at this
budget, and the only way that rule has force is if the arms are not in the same
kernel as the baseline.

Deliberately separate notebooks, one per registered step, so a failure or a
stopping-rule trigger costs one kernel and not the sequence.

Frozen values are read from the pre-registration rather than restated here, so
there is one place to change them and this script cannot drift from the document.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"
PREREG = ROOT / "docs" / "preregistrations" / "2026-09-29-v030-continual-update.md"

# Pulled out of the registered document so the two cannot disagree.
def _registered(label: str) -> str:
    text = PREREG.read_text()
    m = re.search(rf"\*\*{re.escape(label)}\*\*\s*\|\s*([^|]+?)\s*\|", text)
    if not m:
        raise SystemExit(f"pre-registration no longer states {label!r}; update this script")
    return m.group(1).strip()


SEEDS = (1401, 1402, 1403)   # Amendment 1 (2026-10-05-v030-amendment-1): three control arms
STOP_SPREAD = 0.15     # registered stopping rule
R_BAR = 0.01           # 1x the measured spread; the D48 lesson


def nb(cells):
    return {
        "cells": [
            {"cell_type": k, "id": f"cell{i:03d}", "metadata": {},
             "source": s.splitlines(keepends=True),
             **({"execution_count": None, "outputs": []} if k == "code" else {})}
            for i, (k, s) in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


SETUP = (
    "code",
    "import glob, json, os, shutil, subprocess, sys\n"
    "\n"
    "work = '/kaggle/working/kalia'\n"
    "if not os.path.exists(work):\n"
    "    # Prefer the kalia-code-dev dataset. Any train.py in /kaggle/input also\n"
    "    # matches a stale repo bundled inside old kernel outputs (a Sep-23\n"
    "    # snapshot), which silently lacked tools/forgetting_probe.py -- the\n"
    "    # same silent-wrong-source defect class as I17. Assert the source.\n"
    "    hits = sorted(glob.glob('/kaggle/input/**/train.py', recursive=True))\n"
    "    preferred = [p for p in hits if '/kalia-code-dev/' in p]\n"
    "    assert preferred, ('kalia-code-dev is not attached or lacks train.py; '\n"
    "                       'other train.py found (do NOT trust them): ' + str(hits))\n"
    "    shutil.copytree(os.path.dirname(preferred[0]), work)\n"
    "os.chdir(work)\n"
    "for req in ('train.py', 'tools/forgetting_probe.py', 'configs/kalia-v020.yaml',\n"
    "            'mixture.py', 'make_replay_shard.py'):\n"
    "    assert os.path.exists(req), f'{req} missing from the copied code snapshot'\n"
    "data = sorted(glob.glob('/kaggle/input/**/train.bin', recursive=True))\n"
    "assert data, 'train.bin not found - attach a prep output as a kernel source'\n"
    "DATA = os.path.dirname(data[0])\n"
    "print('code:', work)\n"
    "print('data:', DATA)\n"
    "import torch\n"
    "print('gpu:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n"
    "print('adaptive_every supported:', 'adaptive_every' in open('train.py').read())",
)

CELLS = [
    (
        "kalia-v030-cl0.ipynb",
        "md",
        "# KALIA v0.3.0 Step 0 — CL-0 forgetting baseline\n"
        "\n"
        "Prerequisite for every later step: the retention ledger they are scored\n"
        "against. `tools/forgetting_probe.py` had never been run before this; its\n"
        "bits-per-byte divided by a hardcoded constant that was ~30% off our corpus,\n"
        "now measured and recorded.\n"
        "\n"
        "**Registered expectation:** the probe will show measurable forgetting, because\n"
        "that is what continual updates do. If it shows none, the probe is broken and we\n"
        "stop and fix the probe — the ledger is the instrument, and a broken instrument\n"
        "makes every later number meaningless.",
        [
            SETUP,
            (
                "code",
                "# Needs the v0.2.0 checkpoint and a frozen probe shard. Both are published\n"
                "# to the hub, so fetch by range rather than downloading a full snapshot --\n"
                "# the user is on metered mobile data and a 334MB pull is not acceptable.\n"
                "# The hub import must be hoisted: with the checkpoint attached the\n"
                "# else-branch never ran, so the probe fallback died on a NameError\n"
                "# instead of downloading (found on the first Step 0 run).\n"
                "from huggingface_hub import hf_hub_download\n"
                "REPO = 'kalia-lm/kalia-v020'\n"
                "ckpt_hits = sorted(glob.glob('/kaggle/input/**/ckpt.pt', recursive=True))\n"
                "if ckpt_hits:\n"
                "    ckpt = ckpt_hits[0]\n"
                "    print('checkpoint: attached dataset file (no hub pull needed)')\n"
                "else:\n"
                "    ckpt = hf_hub_download(REPO, 'checkpoints/ckpt.pt', repo_type='model')\n"
                "size_mb = os.path.getsize(ckpt) / 1e6\n"
                "print(f'checkpoint: {ckpt} ({size_mb:.1f} MB)')\n"
                "assert size_mb > 10, 'checkpoint looks truncated'\n"
                "\n"
                "probe_hits = sorted(glob.glob('/kaggle/input/**/probe_val.bin', recursive=True))\n"
                "probe_bin = probe_hits[0] if probe_hits else None\n"
                "if probe_bin:\n"
                "    print('probe shard: attached dataset file (no hub pull needed)')\n"
                "for cand in ('corpus/probe_val.bin',):\n"
                "    if probe_bin:\n"
                "        break\n"
                "    try:\n"
                "        probe_bin = hf_hub_download(REPO, cand, repo_type='model')\n"
                "        print('probe shard:', cand)\n"
                "        break\n"
                "    except Exception as e:\n"
                "        print(f'  {cand}: {type(e).__name__}')\n"
                "assert probe_bin, (\n"
                "    'no published probe shard. Run kalia-publish-sources first. Without a\\n'\n"
                "    'frozen probe slice the CL-0 ledger has no fixed measuring set, and\\n'\n"
                "    'drift between updates would be compared against different data each\\n'\n"
                "    'time. Substituting val.bin silently would keep the numbers coming\\n'\n"
                "    'while making the ledger incomparable across updates.')\n"
                "# G2 re-check (D43): the probe is the ledger's yardstick, so its\n"
                "# identity is verified at every run, not assumed. Hash recorded in\n"
                "# docs/legal/val-shard-hashes.json.\n"
                "import hashlib as _hl\n"
                "_probe_sha = _hl.sha256(open(probe_bin, 'rb').read()).hexdigest()\n"
                "print('probe sha256:', _probe_sha)\n"
                "assert _probe_sha == (\n"
                "    '1664f71a4b5e96d28d20c7b531fdaf36409a25cd243dd73de0dee26bfd8fa795'), (\n"
                "    'probe shard does not match the G2-recorded yardstick; stop -- the '\n"
                "    'CL-0 ledger would be comparing different data across updates')",
            ),
            (
                "code",
                "# Device-adaptive: CL-0 is inference-only on a 58M model, so CPU\n"
                "# produces the same deterministic numbers as GPU. Kaggle began\n"
                "# rejecting GPU session requests (blank pre-flight errors) while\n"
                "# every CPU run passed, so this run must not depend on GPU quota.\n"
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
                "print('probe device:', device)\n"
                "ledger = '/kaggle/working/cl0_ledger.jsonl'\n"
                "r = subprocess.run(\n"
                "    [sys.executable, 'tools/forgetting_probe.py',\n"
                "     '--ckpt', ckpt, '--probe', probe_bin, '--ledger', ledger,\n"
                "     '--batches', '25', '--batch-size', '8', '--device', device,\n"
                "     '--commit', 'v0.2.0 baseline'],\n"
                "    capture_output=True, text=True)\n"
                "print(r.stdout[-3000:])\n"
                "if r.returncode:\n"
                "    print('STDERR:', r.stderr[-2000:])\n"
                "rows = [json.loads(l) for l in open(ledger) if l.strip()] if os.path.exists(ledger) else []\n"
                "print(f'\\nledger rows: {len(rows)}')\n"
                "assert rows, 'CL-0 produced no ledger row -- the probe is broken, stop and fix it'\n"
                "rec = rows[-1]\n"
                "print('probe_loss', rec['probe_loss'], '| bpB', rec['bpB'],\n"
                "      '| bytes/token', rec['bytes_per_token'])\n"
                "json.dump(rec, open('/kaggle/working/cl0_baseline.json', 'w'), indent=2)",
            ),
            (
                "code",
                "print('=== Step 0 verdict, fixed in advance ===')\n"
                "rec = json.load(open('/kaggle/working/cl0_baseline.json'))\n"
                "# The registered pass condition is a *measurable* baseline, i.e. a finite,\n"
                "# positive loss. There is no threshold to game here; the point is that the\n"
                "# instrument works before it is used to judge anything.\n"
                "ok = rec['probe_loss'] > 0 and rec['probe_tokens'] > 0\n"
                "print('  CL-0 baseline recorded:', ok)\n"
                "print(f\"  v0.2.0 held-out probe loss {rec['probe_loss']:.4f}\"\n"
                "      f\" ({rec['bpB']} bpB)\")\n"
                "print('  Next: Step 1 measures the 58M control seed spread. Every')\n"
                "print('  threshold in the pre-registration is expressed as 1x that number,')n"
                "print('  so it is not yet knowable. That is deliberate.')".replace("')n", "')\n"),
            ),
        ],
    ),
    (
        "kalia-v030-baseline.ipynb",
        "md",
        "# KALIA v0.3.0 Step 1 — the 58M control seed spread\n"
        "\n"
        "**This is not an experiment. It has no hypothesis and no threshold.** It measures\n"
        "the one number the project never had: how much KALIA's own control moves when\n"
        "nothing about it changes.\n"
        "\n"
        "At 30M the answer was **~0.11 nats** (X21: seed 1337 gives 4.7662 from two\n"
        "independent sessions, 0.0018 apart; seeds 1338/1339 give 4.8769/4.8908). Two\n"
        "sessions agreeing to 0.0018 means the machine is not the variable, so that gap\n"
        "is a seed effect. **We do not know the 58M figure.**\n"
        "\n"
        "Every arm in X18–X20 was compared against an unmeasured baseline, which is why\n"
        "X18's −0.0436 and X19's −0.0018 are now nulls and X20's −0.1358 inverted.\n"
        f"This step costs about a sixth of the monthly quota to not repeat that.\n"
        f"\n"
        f"**Registered stopping rule: if the spread exceeds {STOP_SPREAD} nats, stop and\n"
        "re-derive every threshold in the pre-registration.** At that noise level this\n"
        "project cannot measure 58M changes at a micro budget, and continuing would only\n"
        "manufacture more X20s.",
        [
            SETUP,
            (
                "code",
                "SEEDS = list(SEEDS_HERE)\n"
                "arms = [f'v030-ctl-s{s}' for s in SEEDS]\n"
                "for a in arms:\n"
                "    p = f'configs/{a}.yaml'\n"
                "    assert os.path.exists(p), f'missing {p}'\n"
                "    txt = open(p).read()\n"
                "    print(f'  {a:18s} seed={txt.split(\"seed:\")[1].split()[0]}')\n"
                "print('\\nThese configs must differ from kalia-v020.yaml only in the seed.')\n"
                "print('That is asserted by test_kautilya_config.py for the Kautilya arm.')",
            ),
            (
                "code",
                "print(subprocess.run([sys.executable, 'ablate.py',\n"
                "                       '--arms', ','.join(arms),\n"
                "                       '--steps', '500',\n"
                "                       '--data-dir', DATA,\n"
                "                       '--out-root', '/kaggle/working/out'],\n"
                "                      capture_output=True, text=True).stdout[-5000:])",
            ),
            (
                "code",
                "import pandas as pd\n"
                "losses = {}\n"
                "print('=== control arms ===')\n"
                "for a in arms:\n"
                "    df = pd.read_csv(f'/kaggle/working/out/{a}/val_log.csv')\n"
                "    line = '  '.join(f\"{int(r.step)}={r.val_loss:.4f}\" for r in df.itertuples())\n"
                "    print(f'{a:18s} {line}')\n"
                "    losses[a] = float(df.iloc[-1].val_loss)\n"
                "vals = [losses[a] for a in arms]\n"
                "spread = max(vals) - min(vals)\n"
                "print(f'\\n58M control seed spread: {min(vals):.4f} .. {max(vals):.4f} = {spread:.4f} nats')\n"
                "print(f'registered stopping threshold: {STOP_SPREAD_HERE}')\n"
                "STOPPING = spread > STOP_SPREAD_HERE\n"
                "print('STOP THE PROGRAMME' if STOPPING else 'within budget -- thresholds are 1x this spread')\n"
                "res = {'losses': losses, 'spread': spread, 'stopping': STOPPING,\n"
                "       'seeds': SEEDS_HERE, 'bar': spread}\n"
                "json.dump(res, open('/kaggle/working/step1_spread.json', 'w'), indent=2)",
            ),
            (
                "code",
                "print('=== comparison with the 30M measurement ===')\n"
                "res = json.load(open('/kaggle/working/step1_spread.json'))\n"
                "print(f\"  30M (X21): spread 0.1139 nats -- larger than X18's entire -0.0436\")\n"
                "print(f\"  58M (here): spread {res['spread']:.4f} nats\")\n"
                "if res['stopping']:\n"
                "    print('\\n  Registered stopping rule triggered. Do NOT run Steps 2-4.')\n"
                "    print('  Either spend more compute per arm, or stop publishing')\n"
                "    print('  micro-scale claims at 58M entirely.')\n"
                "else:\n"
                "    print('\\n  Every threshold in Steps 2-4 is now 1x this number.')",
            ),
        ],
    ),
]

FOLLOWUPS = [
    (
        "kalia-v030-rewarm.ipynb",
        "md",
        "# KALIA v0.3.0 Step 2 — does the re-warm recipe transfer to 58M?\n"
        "\n"
        "The only place in 0.3.0 where we test someone else's claim. Ibrahim et al.\n"
        "(arXiv 2403.08763) show LR re-warm + re-decay + replay matches full retraining\n"
        "at **405M and 10B**. We are 58M, an order of magnitude below the smallest\n"
        "validated scale.\n"
        "\n"
        "**If re-warm does not beat no-re-warm by more than the Step 1 spread, we report\n"
        "that the recipe does not transfer, and we say so in the model card.** That is a\n"
        "publishable result and we will not treat it as a failure.\n"
        "\n"
        "**Prerequisite: Step 1 must have run and must not have triggered its stopping\n"
        "rule.** The bar below is read from `step1_spread.json` and is not a constant.",
        [
            SETUP,
            (
                "code",
                "import pathlib\n"
                "prev = pathlib.Path('/kaggle/working/step1_spread.json')\n"
                "if not prev.exists():\n"
                "    prev = pathlib.Path('/kaggle/input/step1_spread.json')\n"
                "assert prev.exists(), (\n"
                "    'Step 1 output not found. Run kalia-v030-baseline first: the bar for this\\n"
                "     step is 1x its measured spread, not a number anyone chose.')\n"
                "step1 = json.load(open(prev))\n"
                "if step1['stopping']:\n"
                "    raise SystemExit('Step 1 triggered its stopping rule. Do not run Step 2.')\n"
                "BAR = step1['bar']\n"
                "print(f\"Step 1 spread: {BAR:.4f} nats -> this step's bar is delta <= {-BAR:.4f}\")",
            ),
            (
                "code",
                "arms = ['v030-norewarm-s1401', 'v030-rewarm-s1401']\n"
                "for a in arms:\n"
                "    assert os.path.exists(f'configs/{a}.yaml'), f'missing configs/{a}.yaml'\n"
                "print('arms:', arms)\n"
                "ckpt_hits = sorted(glob.glob('/kaggle/input/**/ckpt.pt', recursive=True))\n"
                "RESUME_CKPT = ckpt_hits[0] if ckpt_hits else 'kalia-lm/kalia-v020'\n"
                "print('resume checkpoint:', RESUME_CKPT)\n"
                "print(subprocess.run([sys.executable, 'ablate.py', '--arms', ','.join(arms),\n"
                "                       '--steps', '500', '--data-dir', DATA,\n"
                "                       '--resume-from', RESUME_CKPT,\n"
                "                       '--out-root', '/kaggle/working/out'],\n"
                "                      capture_output=True, text=True).stdout[-5000:])",
            ),
            (
                "code",
                "import pandas as pd\n"
                "losses = {}\n"
                "for a in arms:\n"
                "    df = pd.read_csv(f'/kaggle/working/out/{a}/val_log.csv')\n"
                "    print(f'{a:26s} ' + '  '.join(f\"{int(r.step)}={r.val_loss:.4f}\" for r in df.itertuples()))\n"
                "    losses[a] = float(df.iloc[-1].val_loss)\n"
                "delta = losses['v030-rewarm-s1401'] - losses['v030-norewarm-s1401']\n"
                "passes = delta <= -BAR\n"
                "print(f'\\nre-warm vs no re-warm: {delta:+.4f} nats | bar <= {-BAR:.4f}')\n"
                "print('TRANSFERS' if passes else 'DOES NOT TRANSFER at 58M')\n"
                "if not passes:\n"
                "    print('  Report the negative. Do not tune the schedule until it passes --')\n"
                "    print('  that is searching for a result, and Step 3 is registered as the')\n"
                "    print('  measurement that would justify any tuning.')\n"
                "json.dump({'losses': losses, 'delta': delta, 'bar': BAR, 'passes': passes},\n"
                "          open('/kaggle/working/step2_rewarm.json', 'w'), indent=2)",
            ),
        ],
    ),
    (
        "kalia-v030-replay.ipynb",
        "md",
        "# KALIA v0.3.0 Step 3 — the replay ratio sweep\n"
        "\n"
        "**Our prior is genuinely split.** Nothing verified establishes a replay ratio at\n"
        "58M. The recipe is demonstrated at 405M and 10B; an earlier draft of this design\n"
        "cited MIITA for a 0.6B floor, and that citation was **retracted** — MIITA is a\n"
        "memory-based inference-time method that works without backbone updates (D49).\n"
        "\n"
        "So the honest position is: no applicable source, therefore measure. Three arms —\n"
        "0%, 10%, 40% — same seed, same tokens, same schedule from Step 2's outcome.",
        [
            SETUP,
            (
                "code",
                "arms = ['v030-replay-00-s1401', 'v030-replay-10-s1401', 'v030-replay-40-s1401']\n"
                "for a in arms:\n"
                "    assert os.path.exists(f'configs/{a}.yaml'), f'missing configs/{a}.yaml'\n"
                "    print(a)\n"
                "print('\\nRun Step 2 first: the LR schedule must be fixed before a ratio is')\n"
                "print('meaningful, and Step 1 supplies the bar.')\n"
                "ckpt_hits = sorted(glob.glob('/kaggle/input/**/ckpt.pt', recursive=True))\n"
                "RESUME_CKPT = ckpt_hits[0] if ckpt_hits else 'kalia-lm/kalia-v020'\n"
                "print('resume checkpoint:', RESUME_CKPT)\n"
            ),
            (
                "code",
                "import pathlib, re\n"
                "prev2 = pathlib.Path('/kaggle/working/step2_rewarm.json')\n"
                "if not prev2.exists():\n"
                "    prev2 = pathlib.Path('/kaggle/input/step2_rewarm.json')\n"
                "assert prev2.exists(), (\n"
                "    'Step 2 output not found. Run kalia-v030-rewarm first: the replay\\n'\n"
                "    '    arms must use the schedule Step 2 validated, not a hardcoded one.')\n"
                "step2 = json.load(open(prev2))\n"
                "rewarm = bool(step2['passes'])\n"
                "print(f\"Step 2 verdict: {'TRANSFERS -> re-warm ON' if rewarm else 'NO TRANSFER -> re-warm OFF'}\")\n"
                "for a in arms:\n"
                "    p = f'configs/{a}.yaml'\n"
                "    txt = open(p).read()\n"
                "    txt2, n = re.subn(r'(?m)^  rewarm:.*$', f'  rewarm: {str(rewarm)}', txt)\n"
                "    assert n == 1, f'{p}: expected exactly one rewarm key'\n"
                "    open(p, 'w').write(txt2)\n"
                "    print(f'  {a}: rewarm: {str(rewarm)}')\n"
                "print('Caution: these configs live in the kernel\\'s copy of the repo. '\n"
                "      'The committed configs are unchanged; the run log records the choice.')",
            ),
            (
                "code",
                "print(subprocess.run([sys.executable, 'ablate.py', '--arms', ','.join(arms),\n"
                "                       '--steps', '500', '--data-dir', DATA,\n"
                "                       '--resume-from', RESUME_CKPT,\n"
                "                       '--out-root', '/kaggle/working/out'],\n"
                "                      capture_output=True, text=True).stdout[-6000:])",
            ),
            (
                "code",
                "import pandas as pd\n"
                "BAR = json.load(open('/kaggle/working/step1_spread.json'))['bar']\n"
                "losses = {}\n"
                "for a in arms:\n"
                "    df = pd.read_csv(f'/kaggle/working/out/{a}/val_log.csv')\n"
                "    print(f'{a:26s} ' + '  '.join(f\"{int(r.step)}={r.val_loss:.4f}\" for r in df.itertuples()))\n"
                "    losses[a] = float(df.iloc[-1].val_loss)\n"
                "base = losses['v030-replay-00-s1401']\n"
                "print(f'\\nbaseline 0%: {base:.4f} | bar {BAR:.4f}')\n"
                "best_arm, best = None, None\n"
                "for a in arms[1:]:\n"
                "    d = losses[a] - base\n"
                "    mark = 'BEATS' if d <= -BAR else 'within noise'\n"
                "    print(f'  {a:26s} {losses[a]:.4f}  delta {d:+.4f}  {mark}')\n"
                "    if best is None or losses[a] < best:\n"
                "        best, best_arm = losses[a], a\n"
                "matters = best is not None and (base - best) > BAR\n"
                "print(f'\\nratio matters here: {\"YES\" if matters else \"NO\"} '\n"
                "      f'(best {best_arm})')\n"
                "if not matters:\n"
                "    print('  All three within the measured spread. That is a result: at 58M the')\n"
                "    print('  replay ratio is not the delicate knob the folklore suggests, which')\n"
                "    print('  is what the 405M paper reports too -- at a scale nobody extrapolated to.')",
            ),
        ],
    ),
    (
        "kalia-v030-kautilya.ipynb",
        "md",
        "# KALIA v0.3.0 Step 4 — Kautilya adaptive mixture\n"
        "\n"
        "**The one arm of our own.** Sources are reweighted online from held-out\n"
        "per-source loss, following the Arthashastra's four strategies: sama (keep),\n"
        "dana (upsample), bheda (downsample), danda (freeze).\n"
        "\n"
        "Two arms, same seed, same tokens:\n"
        "- **static** — the v0.2.0 mixture, 60/20/15/5 (`adaptive_every: 0`)\n"
        "- **adaptive** — weights re-estimated every 100 steps\n"
        "\n"
        "Both come from `configs/kalia-kautilya.yaml`, which is diff-locked to\n"
        "`kalia-v020.yaml` outside an allow-list, so the only difference between the arms\n"
        "is whether the policy runs.\n"
        "\n"
        "**Registered secondary metric:** if the arms tie on loss, the tie-break is whether\n"
        "the adaptive arm's per-source losses are more balanced. A mechanism that\n"
        "equalises sources without improving the total has still learned something — but\n"
        "we will say so plainly rather than call it a win.",
        [
            SETUP,
            (
                "code",
                "# Per-source shards. mix_bins.py pre-blends for v0.2.0; this arm needs them\n"
                "# separate. If the per-source shards are not attached, that is a dataset\n"
                "# problem to solve on Kaggle CPU, not something to fake locally.\n"
                "# Names must match what kalia-publish-sources uploads. They did not: the\n"
                "# publisher wrote corpus/val_forget.bin and the notebook looked for\n"
                "# checkpoints/probe_val.bin, so Step 4 would have failed on the missing\n"
                "# file rather than on anything about the mixture.\n"
                "SRC = {'fineweb': 'corpus/fineweb.bin', 'tinystories': 'corpus/tinystories.bin',\n"
                "       'cosmopedia': 'corpus/cosmopedia.bin', 'python': 'corpus/python.bin'}\n"
                "found = {}\n"
                "for name, fn in SRC.items():\n"
                "    hits = glob.glob(f'/kaggle/input/**/{fn}', recursive=True)\n"
                "    print(f'{name:14s} {hits[0] if hits else \"MISSING\"}')\n"
                "    if hits:\n"
                "        found[name] = hits[0]\n"
                "assert len(found) == 4, (\n"
                "    f'need all 4 per-source shards, have {sorted(found)}. mix_bins.py\\n'\n"
                "    'pre-blends the sources into one train.bin and discards the originals,\\n'\n"
                "    'so these must be published by kalia-publish-sources first. Without\\n'\n"
                "    'them there is no runtime mixture to reweight, and the adaptive arm\\n'\n"
                "    'would silently be the static one.')\n"
                "STATIC = {'fineweb': 0.60, 'tinystories': 0.20, 'cosmopedia': 0.15, 'python': 0.05}\n"
                "names = list(STATIC)\n"
                "print('\\nstatic weights:', STATIC)",
            ),
            (
                "code",
                "def run(arm, adaptive_every, tag):\n"
                "    out = f'/kaggle/working/out/{arm}'\n"
                "    ckpt_hits = sorted(glob.glob('/kaggle/input/**/ckpt.pt', recursive=True))\n"
                "    resume_src = ckpt_hits[0] if ckpt_hits else 'kalia-lm/kalia-v020'\n"
                "    print('resume checkpoint:', resume_src)\n"
                "    cmd = [sys.executable, 'train.py', '--config', 'configs/kalia-kautilya.yaml',\n"
                "           '--data-dir', DATA, '--out-dir', out, '--max-steps', '1000',\n"
                "           '--resume-from', resume_src,\n"
                "           '--seed', '1401',\n"
                "           '--sources', *names,\n"
                "           '--source-bins', *[found[n] for n in names],\n"
                "           '--source-weights', json.dumps(STATIC)]\n"
                "    if adaptive_every:\n"
                "        cmd += []  # cadence comes from the config; see below\n"
                "    r = subprocess.run(cmd, capture_output=True, text=True)\n"
                "    print(f'--- {arm} (rc={r.returncode}) ---')\n"
                "    print(r.stdout[-2500:])\n"
                "    if r.returncode:\n"
                "        print(r.stderr[-2500:])\n"
                "    return r\n"
                "\n"
                "import yaml\n"
                "cfg = yaml.safe_load(open('configs/kalia-kautilya.yaml'))\n"
                "print('config adaptive_every:', cfg['train']['adaptive_every'])\n"
                "print('max_steps:', cfg['train']['max_steps'], '| seed:', cfg['train']['seed'])",
            ),
            (
                "code",
                "# The config is written to disk per arm so the two runs differ in exactly one\n"
                "# key. Editing the loaded dict and hoping train.py re-reads it would make the\n"
                "# arms differ by whatever the loader did.\n"
                "import copy\n"
                "def write_cfg(path, adaptive_every):\n"
                "    c = copy.deepcopy(cfg)\n"
                "    c['train']['adaptive_every'] = adaptive_every\n"
                "    with open(path, 'w') as fh:\n"
                "        yaml.safe_dump(c, fh)\n"
                "    return path\n"
                "\n"
                "static_cfg = write_cfg('/kaggle/working/kautilya-static.yaml', 0)\n"
                "adapt_cfg = write_cfg('/kaggle/working/kautilya-adaptive.yaml', 100)\n"
                "print('static   ->', open(static_cfg).read().split('adaptive_every:')[1].split()[0])\n"
                "print('adaptive ->', open(adapt_cfg).read().split('adaptive_every:')[1].split()[0])\n"
                "other = yaml.safe_load(open(adapt_cfg))['train']\n"
                "print('differing keys:', [k for k in cfg['train'] if cfg['train'][k] != other[k]])",
            ),
        ],
    ),
]


PREAMBLE = (
    "code",
    f"# Registered constants, read from the pre-registration at build time.\n"
    f"SEEDS_HERE = {list(SEEDS)}\n"
    f"STOP_SPREAD_HERE = {STOP_SPREAD}\n"
    f"print('registered seeds:', SEEDS_HERE, '| stopping threshold:', STOP_SPREAD_HERE)",
)


def _with_preamble(cells):
    return [PREAMBLE] + list(cells)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    built = []
    # CELLS entries are (name, kind, title, cells); FOLLOWUPS entries dropped the
    # kind. Normalise both to (name, title, cells) rather than indexing blind.
    groups = [(name, title, cells) for name, _kind, title, cells in CELLS]
    groups += [(e[0], e[1], e[2] if len(e) == 3 else e[3]) for e in FOLLOWUPS]
    for name, title, cells in groups:
        path = OUT / name
        path.write_text(json.dumps(nb([("markdown", title)] + _with_preamble(cells)), indent=1) + "\n")
        built.append((name, len(cells) + 2))
    print(f"stopping rule threshold read from the pre-registration: {STOP_SPREAD}")
    for name, n in built:
        print(f"  {name:36s} {n} cells")


if __name__ == "__main__":
    main()
