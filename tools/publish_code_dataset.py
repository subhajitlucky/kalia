"""Stage and publish the `kalia-code-dev` Kaggle dataset.

This exists because publishing it by hand went wrong three separate ways, each of
which failed *silently* -- the upload reported success every time:

1. The dataset was staged flat, so `eval/probe_sentences.json` never existed in
   it. `eval_reversibility.py` and `gate_probe.py` both resolve their probe file
   from there, so in any kernel neither could find it. That is the deeper reason
   X18's Abhimanyu gap was never measurable despite `ablate.py` supporting
   `--reversibility`.
2. Adding `eval/` as a subdirectory still produced zero JSON, because
   `kaggle datasets version` defaults to `--dir-mode skip`, which ignores
   directories outright. It needed `-r zip`.
3. The staging directory lived in /tmp and was cleaned between runs, so the
   "republish" sometimes staged nothing and re-uploaded a truncated dataset.

So: one script, one layout, always `-r zip`, and it verifies by downloading the
published dataset back and asserting the files are actually present.

Usage:
    python tools/publish_code_dataset.py            # publish, then verify
    python tools/publish_code_dataset.py --verify   # verify only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = "subhajitlucky/kalia-code-dev"

# Files the kernels fetch by exact name. If any of these is missing the kernel
# dies on a bare FileNotFoundError with no clue as to why.
REQUIRED = [
    "ablate.py",
    "data.py",
    "eval_bench.py",
    "eval_reversibility.py",
    "eval_val.py",
    "gate_probe.py",
    "kalia_lm.py",
    "model.py",
    "optim.py",
    "train.py",
    "eval/probe_sentences.json",
    "eval/probe_prompts.json",
    "configs/micro-base.yaml",
    "configs/micro-nope.yaml",
    "configs/micro-gated.yaml",
    "configs/micro-docmask.yaml",
    "configs/kalia-v020.yaml",
]


def stage(dest: Path) -> list[str]:
    """Copy the dataset payload, preserving the directories the code expects."""
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    for src in sorted(ROOT.glob("*.py")):
        shutil.copy2(src, dest / src.name)
        copied.append(src.name)

    for src in sorted((ROOT / "tools").glob("*.py")):
        shutil.copy2(src, dest / src.name)
        copied.append(f"tools/{src.name}")

    (dest / "configs").mkdir(exist_ok=True)
    for src in sorted((ROOT / "configs").glob("*.yaml")):
        shutil.copy2(src, dest / "configs" / src.name)
        copied.append(f"configs/{src.name}")

    (dest / "eval").mkdir(exist_ok=True)
    for src in sorted((ROOT / "eval").glob("*.json")):
        shutil.copy2(src, dest / "eval" / src.name)
        copied.append(f"eval/{src.name}")

    for junk in ("__pycache__", ".pytest_cache"):
        shutil.rmtree(dest / junk, ignore_errors=True)

    (dest / "dataset-metadata.json").write_text(
        '{\n'
        '  "title": "KALIA code dev",\n'
        f'  "id": "{DATASET}",\n'
        '  "licenses": [{"name": "other"}],\n'
        '  "isPrivate": true\n'
        '}\n'
    )
    return copied


def verify() -> bool:
    """Download the published dataset and assert the payload is really there."""
    with tempfile.TemporaryDirectory() as tmp:
        rc = subprocess.run(
            ["kaggle", "datasets", "download", DATASET, "-p", tmp, "--force"],
            capture_output=True,
            text=True,
        )
        if rc.returncode != 0:
            print(f"  download failed: {rc.stderr[-300:]}")
            return False
        zips = list(Path(tmp).glob("*.zip"))
        if not zips:
            print("  no zip returned")
            return False
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(tmp)

        present = set()
        for p in Path(tmp).rglob("*"):
            if p.is_file() and p.suffix != ".zip":
                present.add(str(p.relative_to(tmp)))

        missing = [f for f in REQUIRED if f not in present]
        jsons = sorted(f for f in present if f.endswith(".json"))
        print(f"  published files: {len(present)} | json: {len(jsons)}")
        if missing:
            print(f"  MISSING {len(missing)}: {missing}")
            return False
        print(f"  all {len(REQUIRED)} required files present")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify only, do not publish")
    parser.add_argument("-m", "--message", default="sync kalia code dataset")
    args = parser.parse_args()

    if args.verify:
        print("verifying published dataset...")
        sys.exit(0 if verify() else 1)

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "kalia-code-dev"
        copied = stage(dest)
        print(f"staged {len(copied)} files -> {dest}")
        missing = [f for f in REQUIRED if f not in copied]
        if missing:
            print(f"  STAGE INCOMPLETE, refusing to publish: {missing}")
            sys.exit(1)

        # -r zip is load-bearing: the default is --dir-mode skip, which drops
        # configs/ and eval/ entirely and fails without any error.
        rc = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(dest), "-r", "zip", "-m", args.message],
            capture_output=True,
            text=True,
        )
        print(rc.stdout.strip()[-300:] or rc.stderr.strip()[-300:])
        if rc.returncode != 0:
            sys.exit(rc.returncode)

    print("verifying...")
    sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()
