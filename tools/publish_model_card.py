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
    python tools/publish_model_card.py --release v020   # upload, verify, clean strays
    python tools/publish_model_card.py --release v020 --verify   # verify only
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent

# One entry per published checkpoint.
RELEASES = {
    "v012": {
        "repo": "kalia-lm/kalia-v012",
        "local": ROOT / "docs" / "public" / "hf-model-card.md",
    },
    "v020": {
        "repo": "kalia-lm/kalia-v020",
        "local": ROOT / "docs" / "public" / "hf-model-card-v020.md",
    },
}
REMOTE = "README.md"
CHARTS = ROOT / "docs" / "public" / "charts"
REMOTE_CHARTS = "charts"

# Directories that should never exist in the model repo. Both were created by
# the argument-order mistake, which uploads the local file to the *second*
# positional argument as a remote path.
STRAY_DIRS = ("docs",)


def content_matches(repo: str, local: Path) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        got = hf_hub_download(repo, REMOTE, repo_type="model", local_dir=tmp, force_download=True)
        return Path(got).read_text(encoding="utf-8") == local.read_text(encoding="utf-8")


def chart_paths(card: Path) -> list[Path]:
    """Only the charts this card actually references.

    The v0.1.2 card is a historical release card and must not grow the v0.2.0
    comparison charts just because they exist in the repository. The card is
    the source of truth for what belongs on that model page.
    """
    refs = re.findall(r"\((charts/[^)]+\.png)\)", card.read_text())
    paths: list[Path] = []
    for ref in refs:
        local = ROOT / "docs" / "public" / ref
        if not local.exists():
            raise SystemExit(f"{card.relative_to(ROOT)} references {ref}, which does not exist")
        if local not in paths:
            paths.append(local)
    if not paths:
        raise SystemExit(f"{card.relative_to(ROOT)} references no charts; nothing to upload")
    return paths


def charts_match(repo: str, card: Path) -> tuple[bool, str | None]:
    with tempfile.TemporaryDirectory() as tmp:
        for local in chart_paths(card):
            remote = f"{REMOTE_CHARTS}/{local.name}"
            try:
                got = hf_hub_download(repo, remote, repo_type="model", local_dir=tmp, force_download=True)
            except Exception:
                return False, local.name
            if Path(got).read_bytes() != local.read_bytes():
                return False, local.name
    return True, None


def clean_charts(api: HfApi, repo: str, card: Path) -> None:
    """Remove charts on the model repo that this card does not reference.

    The release pages are historical artifacts. The v0.1.2 page must not grow
    the v0.2.0 comparison charts just because they exist in the repository.
    """
    keep = {path.name for path in chart_paths(card)}
    for remote in api.list_repo_files(repo, repo_type="model"):
        if not remote.startswith(f"{REMOTE_CHARTS}/"):
            continue
        name = remote.rsplit("/", 1)[-1]
        if name in keep:
            continue
        api.delete_file(
            path_in_repo=remote,
            repo_id=repo,
            repo_type="model",
            commit_message=f"model card: remove unreferenced chart {name}",
        )
        print(f"  removed unreferenced chart: {remote}")


def clean_strays(api: HfApi, repo: str) -> None:
    for d in STRAY_DIRS:
        try:
            api.delete_folder(path_in_repo=d, repo_id=repo, repo_type="model")
            print(f"  removed stray directory: {d}/")
        except Exception as exc:  # noqa: BLE001 - absent is the normal case
            print(f"  no stray directory {d}/ ({type(exc).__name__})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=sorted(RELEASES), default="v012")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    repo = RELEASES[args.release]["repo"]
    local = RELEASES[args.release]["local"]

    api = HfApi()
    print(f"target: {repo} :: {REMOTE}")
    print(f"source: {local.relative_to(ROOT)}")

    if not args.verify:
        # Named explicitly, never positionally, and cleaned up afterwards.
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=REMOTE,
            repo_id=repo,
            repo_type="model",
            commit_message="model card: measured limitations, results and licence disclosure",
        )
        print("uploaded")
        for chart in chart_paths(local):
            api.upload_file(
                path_or_fileobj=str(chart),
                path_in_repo=f"{REMOTE_CHARTS}/{chart.name}",
                repo_id=repo,
                repo_type="model",
                commit_message="model card: add evaluation charts",
            )
            print(f"  uploaded chart: {REMOTE_CHARTS}/{chart.name}")
        clean_charts(api, repo, local)
        clean_strays(api, repo)

    print("verifying by downloading the live card back...")
    if not content_matches(repo, local):
        print("  MISMATCH: the live card does not match the source")
        return 1
    print("  live card is byte-identical to the repo source")

    print("verifying the live charts...")
    ok, bad = charts_match(repo, local)
    if not ok:
        print(f"  MISMATCH: charts/{bad} does not match the source")
        return 1
    print(f"  all {len(chart_paths(local))} referenced charts are byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
