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
import hashlib
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
    "tools/forgetting_probe.py",
    "eval/probe_sentences.json",
    "eval/probe_prompts.json",
    "LICENSE",
    "hf-model-card-v020.md",
    "configs/micro-base.yaml",
    "configs/micro-nope.yaml",
    "configs/micro-gated.yaml",
    "configs/micro-docmask.yaml",
    "configs/kalia-v020.yaml",
]


def kaggle_cli() -> str:
    """The CLI next to the interpreter running this, not whatever is on PATH.

    A bare "kaggle" worked in the shell this was written in and failed under the
    venv interpreter, which is how the dataset publish died on a FileNotFoundError
    with the payload already staged. tools/collect_kaggle_state.py already
    resolved it this way; this file did not.
    """
    candidate = Path(sys.executable).parent / "kaggle"
    return str(candidate) if candidate.exists() else "kaggle"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage(dest: Path) -> list[str]:
    """Copy the dataset payload, preserving the directories the code expects."""
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    for src in sorted(ROOT.glob("*.py")):
        shutil.copy2(src, dest / src.name)
        copied.append(src.name)

    # tools/ must keep its directory: the kernels invoke
    # `python tools/forgetting_probe.py` relative to the repo root. A flat
    # staging put the file at the root, where nothing looks for it -- found
    # when the first Step 0 run died on a missing tools/forgetting_probe.py
    # after the dataset "restore" had passed its own verification.
    (dest / "tools").mkdir(exist_ok=True)
    for src in sorted((ROOT / "tools").glob("*.py")):
        shutil.copy2(src, dest / "tools" / src.name)
        copied.append(f"tools/{src.name}")

    # LICENSE travels with the code because the release needs it: the export
    # kernel copies it into the model repo, and its absence failed the run.
    for extra in ("LICENSE", "README.md", "docs/public/hf-model-card-v020.md"):
        src = ROOT / extra
        if src.exists():
            name = extra if "/" not in extra else Path(extra).name
            shutil.copy2(src, dest / name)
            copied.append(name)

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


def verify(expect: dict[str, str] | None = None) -> bool:
    """Download the published dataset and assert the payload is really there.

    `expect` maps relative path -> sha256 of the staged local file. Presence alone
    is not verification: the first publish of the doc-mask eval fix passed the
    filename check while the content question -- is the *fix* actually on Kaggle? --
    needed a separate download and grep. A round trip that does not compare bytes
    cannot tell a republished dataset from a restaged one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        rc = subprocess.run(
            [kaggle_cli(), "datasets", "download", DATASET, "-p", tmp, "--force"],
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

        if expect:
            differing = []
            for rel, want in sorted(expect.items()):
                got = Path(tmp) / rel
                if not got.exists():
                    differing.append(f"{rel}: absent")
                elif hashlib.sha256(got.read_bytes()).hexdigest() != want:
                    differing.append(f"{rel}: content differs")
            if differing:
                print(f"  {len(differing)} file(s) do not match what was staged:")
                for line in differing[:10]:
                    print(f"    {line}")
                return False
            print(f"  all {len(expect)} staged files match byte-for-byte")
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
        # Digest what we are about to upload, so the round trip can compare bytes
        # rather than filenames. The first publish of the doc-mask eval fix passed
        # the filename check and still needed a manual download to answer "is the
        # fix actually on Kaggle?".
        digests = {
            str(p.relative_to(dest)): sha256(p)
            for p in sorted(dest.rglob("*"))
            if p.is_file()
        }
        missing = [f for f in REQUIRED if f not in copied]
        if missing:
            print(f"  STAGE INCOMPLETE, refusing to publish: {missing}")
            sys.exit(1)

        # -r zip is load-bearing: the default is --dir-mode skip, which drops
        # configs/ and eval/ entirely and fails without any error.
        rc = subprocess.run(
            [kaggle_cli(), "datasets", "version", "-p", str(dest), "-r", "zip", "-m", args.message],
            capture_output=True,
            text=True,
        )
        print(rc.stdout.strip()[-300:] or rc.stderr.strip()[-300:])
        if rc.returncode != 0:
            sys.exit(rc.returncode)

    print("verifying...")
    sys.exit(0 if verify(digests) else 1)


if __name__ == "__main__":
    main()
