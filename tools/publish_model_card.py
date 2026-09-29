"""Publish the public model card to Hugging Face, correctly and verifiably.

`hf upload` takes its arguments as `REPO_ID FILE_OR_FOLDER PATH_IN_REPO`. That
ordering is easy to get backwards, and getting it backwards fails *silently*:
the upload succeeds, returns a commit URL, and creates a stray directory in the
repo instead of updating the file you meant. That has happened twice on this
repo -- once creating `docs/public/hf-model-card.md` inside the HF repo, and once
writing the project's own README into that same stray path.

So the path is never passed positionally here. The local file and the remote
path are named explicitly, the upload is verified by downloading the result back
and comparing it byte-for-byte, and any stray directory the previous mistake left
behind is removed.

Usage:
    python tools/publish_model_card.py            # upload, verify, clean strays
    python tools/publish_model_card.py --verify   # verify only
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent

REPO = "kalia-lm/kalia-v012"
LOCAL = ROOT / "docs" / "public" / "hf-model-card.md"
REMOTE = "README.md"

# Directories that should never exist in the model repo. Both were created by
# the argument-order mistake, which uploads the local file to the *second*
# positional argument as a remote path.
STRAY_DIRS = ("docs",)


def content_matches() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        got = hf_hub_download(REPO, REMOTE, repo_type="model", local_dir=tmp, force_download=True)
        return Path(got).read_text(encoding="utf-8") == LOCAL.read_text(encoding="utf-8")


def clean_strays() -> None:
    api = HfApi()
    for d in STRAY_DIRS:
        try:
            api.delete_folder(path_in_repo=d, repo_id=REPO, repo_type="model")
            print(f"  removed stray directory: {d}/")
        except Exception as exc:  # noqa: BLE001 - absent is the normal case
            print(f"  no stray directory {d}/ ({type(exc).__name__})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    print(f"target: {REPO} :: {REMOTE}")
    print(f"source: {LOCAL.relative_to(ROOT)}")

    if not args.verify:
        # Named explicitly, never positionally, and cleaned up afterwards.
        api.upload_file(
            path_or_fileobj=str(LOCAL),
            path_in_repo=REMOTE,
            repo_id=REPO,
            repo_type="model",
            commit_message="model card: measured limitations, results and licence disclosure",
        )
        print("uploaded")
        clean_strays()

    print("verifying by downloading the live card back...")
    if content_matches():
        print("  live card is byte-identical to the repo source")
        return 0

    print("  MISMATCH: the live card does not match the source")
    return 1


if __name__ == "__main__":
    sys.exit(main())
