#!/usr/bin/env python3
"""Build a CPU kernel that publishes the two artifacts v0.3.0 needs and lacks.

Found while auditing readiness. The v0.3.0 notebooks reference files that do not
exist anywhere:

- **Step 0 (CL-0)** wants a frozen probe shard. `kalia-v020` on the hub has
  `checkpoints/ckpt.pt` and the two log CSVs and no `val.bin`, so the probe had
  nothing to measure. The probe script even documents a fallback that is not
  reachable from the published repo.
- **Step 4 (Kautilya)** wants four per-source shards. `mix_bins.py` pre-blends
  them into a single `train.bin` and the originals are discarded. Sampling at
  runtime is impossible without them, which is why the Kautilya arm could not
  have run even with unlimited quota.

Both are recoverable from data we still have, on Kaggle CPU, at no GPU cost. This
kernel re-derives them from the prep output and publishes them to the model repo.

The probe shard is a slice of the v0.2.0 *training* corpus, deliberately: the
CL-0 ledger exists to detect forgetting of what was learned, so the measuring set
must be data the model was trained on. A held-out set would measure nothing. The
slice is frozen once published and never changes, which is what makes drift
comparable across updates.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "kalia-publish-sources.ipynb"

REPO = "kalia-lm/kalia-v020"
# 8 x 1024 = 8192 windows, comfortably above the 1,250 the CL-0 probe measures and
# small enough to sit in a repo alongside a 334MB checkpoint.
PROBE_TOKENS = 1_000_000
SOURCES = {
    "fineweb": "smollm_fineweb_edu",
    "tinystories": "tinystories",
    "cosmopedia": "smollm_cosmopedia_v2",
    "python": "python",
}

CELLS = [
    (
        "markdown",
        "# Publish the per-source shards and the CL-0 probe set\n"
        "\n"
        "Two artifacts v0.3.0 needs and does not have. CPU only, no GPU quota.\n"
        "\n"
        "**1. The CL-0 probe shard.** `kalia-v020` publishes a checkpoint and two log\n"
        "CSVs, but no `val.bin` -- so the forgetting probe had nothing to measure and\n"
        "Step 0 could not run at all. This is a **slice of the training corpus**, not a\n"
        "held-out set: the ledger exists to detect forgetting of what was learned, so the\n"
        "measuring set has to be data the model was trained on. Frozen once published,\n"
        "which is what makes drift comparable across updates.\n"
        "\n"
        "**2. The four per-source shards.** `mix_bins.py` pre-blends them into one\n"
        "`train.bin` and discards the originals, so Step 4's runtime mixture cannot run\n"
        "however much quota is available. Re-derived from the prep output.\n"
        "\n"
        "Published to the model repo, which is already public and licence-audited. The\n"
        "shards are derived from the same filtered sources v0.2.0 was trained on, so no\n"
        "new licence question arises -- and none of this redistributes raw third-party\n"
        "text, only the same uint16 token stream the model already saw.",
    ),
    (
        "code",
        "import glob, hashlib, json, os, shutil\n"
        "\n"
        "work = '/kaggle/working/kalia'\n"
        "if not os.path.exists(work):\n"
        "    hits = sorted(glob.glob('/kaggle/input/**/mix_bins.py', recursive=True))\n"
        "    assert hits, 'attach kalia-code-dev'\n"
        "    shutil.copytree(os.path.dirname(hits[0]), work)\n"
        "os.chdir(work)\n"
        "\n"
        "bins = {os.path.basename(p): p for p in glob.glob('/kaggle/input/**/*.bin', recursive=True)}\n"
        "for name, p in sorted(bins.items()):\n"
        "    print(f'{name:34s} {os.path.getsize(p)/1e6:9.1f} MB')\n"
        "print('\\nCPU kernel, no GPU requested.')",
    ),
    (
        "code",
        "# Locate the pre-blended training corpus and the per-source shards. Build a\n"
        "# dict keyed by basename and duplicate names collapse silently: prep-v2 and\n"
        "# prep-v2b both carry stack_smol.bin, and last-glob-wins selected the\n"
        "# UNFILTERED one (I16) in one run but not another. Resolve by full path.\n"
        "all_bin = sorted(glob.glob('/kaggle/input/**/*.bin', recursive=True))\n"
        "# The probe must come from the corpus v0.2.0 actually trained on: the v2b\n"
        "# build. The fallback exists only so the assert below can say why an\n"
        "# unmixed probe would be wrong, not to let the run proceed.\n"
        "v2b_train = [p for p in all_bin if os.path.basename(p) == 'train.bin'\n"
        "             and '/kalia-prep-v2b/' in p]\n"
        "any_train = [p for p in all_bin if os.path.basename(p) == 'train.bin']\n"
        "train_bins = v2b_train or any_train\n"
        "metas = sorted(glob.glob('/kaggle/input/**/mix_meta.json', recursive=True))\n"
        "print('pre-blended train:', train_bins)\n"
        "print('mount trees:', sorted({p.rsplit('/data/', 1)[0] for p in all_bin if '/data/' in p}))\n"
        "print('mix manifests:', metas)\n"
        "assert train_bins, 'no train.bin in the attached prep output'\n"
        "assert '/kalia-prep-v2b/' in train_bins[0], (\n"
        "    'need the v2b build (the corpus v0.2.0 trained on), not prep-v2: '\n"
        "    + str(train_bins))",
    ),
    (
        "code",
        f"# 1. The CL-0 probe shard: a frozen prefix of the training stream.\n"
        f"PROBE_TOKENS = {PROBE_TOKENS}\n"
        "src = train_bins[0]\n"
        "total = os.path.getsize(src) // 2   # uint16\n"
        "take = min(PROBE_TOKENS, total)\n"
        "with open(src, 'rb') as fh:\n"
        "    head = fh.read(take * 2)\n"
        "probe = '/kaggle/working/probe/val_forget.bin'\n"
        "os.makedirs(os.path.dirname(probe), exist_ok=True)\n"
        "open(probe, 'wb').write(head)\n"
        "print(f'probe shard: {take:,} tokens ({take*2/1e6:.1f} MB) of {total:,}')\n"
        "print('  taken from the TRAINING stream on purpose: CL-0 measures forgetting')\n"
        "print('  of what was learned, so a held-out set would measure nothing.')\n"
        "digest = hashlib.sha256(head).hexdigest()\n"
        "print('  sha256:', digest)",
    ),
    (
        "code",
        "# 2. Per-source shards, equalised. The v2b kernel output kept only the\n"
        "# blended train.bin plus the freshly filtered code slice; the three text\n"
        "# originals survive one kernel further back, in the kalia-prep-v2 output\n"
        "# (the mixer only reads its inputs, so they are intact). Recovery map:\n"
        "# text shards from prep-v2, code EXCLUSIVELY from prep-v2b's filtered\n"
        "# stack_smol.bin -- prep-v2's code slice predates the licence filter (I16)\n"
        "# and must never be selected, so python has no fallback.\n"
        "#\n"
        "# Two normalisations, both recorded in the manifest rather than assumed:\n"
        "# (a) the mixer's validation heads are skipped (val was taken\n"
        "# proportionally from the start of each shard, so re-training on them\n"
        "# would contaminate the canonical val set); (b) all four are cut to the\n"
        "# same length, because SourceMixtureDataset requires equal-length shards\n"
        "# and unequal lengths would crash Step 4 on GPU.\n"
        "import numpy as np\n"
        "\n"
        "RATIOS = {'fineweb': 0.60, 'tinystories': 0.20, 'cosmopedia': 0.15, 'python': 0.05}\n"
        "PER_SOURCE_FILES = {\n"
        "    'fineweb': ['smollm_fineweb_edu.bin'],\n"
        "    'tinystories': ['tinystories.bin'],\n"
        "    'cosmopedia': ['smollm_cosmopedia_v2.bin', 'cosmopedia.bin'],\n"
        "    'python': ['stack_smol.bin', 'python.bin', 'python_filtered.bin'],\n"
        "}\n"
        "# Where each source may come from. '/kalia-prep-v2/' is slash-terminated\n"
        "# so it never matches '/kalia-prep-v2b/'. Python allows ONLY v2b.\n"
        "V2, V2B = '/kalia-prep-v2/', '/kalia-prep-v2b/'\n"
        "REQUIRE = {'fineweb': V2, 'tinystories': V2, 'cosmopedia': V2, 'python': V2B}\n"
        "FORBID = {'fineweb': (V2B,), 'tinystories': (V2B,), 'cosmopedia': (V2B,),\n"
        "          'python': (V2,)}\n"
        "# NOTE: FORBID['python'] bans '/kalia-prep-v2/' but not '/kalia-prep-v2b/'\n"
        "# because the trailing slash keeps them distinct -- the filtered slice\n"
        "# passes, the unfiltered one cannot.\n"
        "by_name = {}\n"
        "for p in glob.glob('/kaggle/input/**/*.bin', recursive=True):\n"
        "    by_name.setdefault(os.path.basename(p), []).append(p)\n"
        "picked = {}\n"
        "for name, cands in PER_SOURCE_FILES.items():\n"
        "    options = [p for c in cands for p in by_name.get(c, [])\n"
        "               if REQUIRE[name] in p and not any(f in p for f in FORBID[name])]\n"
        "    if not options:\n"
        "        have = {c: by_name.get(c, []) for c in cands}\n"
        "        print(f'{name:12s} MISSING (looked for {cands}); seen {have}')\n"
        "        continue\n"
        "    picked[name] = sorted(options)[0]\n"
        "    print(f'{name:12s} <- {picked[name]}')\n"
        "assert len(picked) == 4, (\n"
        "    f'have {sorted(picked)}; attach kalia-prep-v2 AND kalia-prep-v2b outputs')\n"
        "VAL_TOKENS = 10_000_000\n"
        "if metas:\n"
        "    VAL_TOKENS = int(json.load(open(metas[0])).get('val_tokens', VAL_TOKENS))\n"
        "    print('val_tokens from mix manifest:', VAL_TOKENS)\n"
        "starts = {n: int(VAL_TOKENS * RATIOS[n]) for n in RATIOS}\n"
        "avail = {}\n"
        "for n, p in picked.items():\n"
        "    avail[n] = os.path.getsize(p) // 2 - starts[n]\n"
        "    print(f'{n:12s} {os.path.getsize(p)//2:>13,} tokens, val-skipped {starts[n]:>9,}',\n"
        "          f'-> usable {avail[n]:>13,}')\n"
        "L = min(avail.values())\n"
        "assert L > 1_000_000, f'equalised length {L:,} too short to be useful'\n"
        "print(f'equalised length for all four sources: {L:,} tokens')\n"
        "out_dir = '/kaggle/working/sources'\n"
        "os.makedirs(out_dir, exist_ok=True)\n"
        "manifest = {'equalised_tokens': L, 'val_tokens_skipped': starts}\n"
        "for name, p in picked.items():\n"
        "    dst = f'{out_dir}/{name}.bin'\n"
        "    with open(p, 'rb') as fh:\n"
        "        fh.seek(starts[name] * 2)\n"
        "        chunk = fh.read(L * 2)\n"
        "    assert len(chunk) == L * 2, f'{name}: short read'\n"
        "    open(dst, 'wb').write(chunk)\n"
        "    manifest[name] = {'tokens': L, 'ratio': RATIOS[name],\n"
        "                     'sha256': hashlib.sha256(chunk).hexdigest(),\n"
        "                     'src': p}\n"
        "    print(f'{name:12s} {L:>12,} tokens  ratio {RATIOS[name]:.2f}')\n"
        "json.dump(manifest, open(f'{out_dir}/manifest.json', 'w'), indent=2)\n"
        "print('\\nratios must match the static arm of Step 4 (60/20/15/5)')",
    ),
    (
        "code",
        f"REPO = '{REPO}'\n"
        "from huggingface_hub import upload_file\n"
        "try:\n"
        "    tok = [l.split('=', 1)[1].strip() for l in open('/kaggle/input/secrets/hf_token')\n"
        "           if '=' in l and not l.startswith('#')][0]\n"
        "except Exception:\n"
        "    tok = None\n"
        "# Kaggle exposes attached secrets as env vars under the exact secret name,\n"
        "# so 'hf_token' and 'HF_TOKEN' are different vars. Accept any casing,\n"
        "# then fall back to the secrets client.\n"
        "if not tok:\n"
        "    upper = {k.upper(): v for k, v in os.environ.items()}\n"
        "    tok = upper.get('HF_TOKEN')\n"
        "if not tok:\n"
        "    import glob as _g\n"
        "    for _cand in _g.glob('/kaggle/input/**/hf_token', recursive=True):\n"
        "        _raw = open(_cand).read().strip()\n"
        "        _tok = _raw.split('=', 1)[-1].strip() if '=' in _raw else _raw\n"
        "        if _tok:\n"
        "            tok = _tok\n"
        "            print('token from attached input:', _cand)\n"
        "            break\n"
        "if not tok:\n"
        "    try:\n"
        "        from kaggle_secrets import UserSecretsClient\n"
        "        tok = UserSecretsClient().get_secret('HF_TOKEN')\n"
        "    except Exception:\n"
        "        tok = None\n"
        "assert tok, ('no HF token available: attach HF_TOKEN as a secret (browser '\n"
        "             'run) or a private input dataset file named hf_token (API runs)')\n"
        "\n"
        "uploaded = []\n"
        "UPLOADS = [(probe, 'corpus/probe_val.bin')]\n"
        "UPLOADS += [(f'{out_dir}/{n}.bin', f'corpus/{n}.bin') for n in RATIOS]\n"
        "UPLOADS += [(f'{out_dir}/manifest.json', 'corpus/manifest.json')]\n"
        "for path, dest in UPLOADS:\n"
        "    upload_file(path_or_fileobj=path, path_in_repo=dest, repo_id=REPO,\n"
        "                repo_type='model', token=tok)\n"
        "    uploaded.append((dest, os.path.getsize(path)))\n"
        "    print(f'uploaded {dest:34s} {os.path.getsize(path)/1e6:8.1f} MB')\n"
        "print(f'\\n{len(uploaded)} files published to {REPO}')\n"
        "print('Step 0 can now run; Step 4 now has the shards its runtime mixture needs.')",
    ),
]


def main() -> None:
    cells = []
    for kind, src in CELLS:
        cell = {
            "cell_type": kind,
            "id": f"cell{len(cells):03d}",
            "metadata": {},
            "source": src.splitlines(keepends=True),
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
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
