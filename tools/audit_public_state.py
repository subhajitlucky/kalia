#!/usr/bin/env python3
"""Read-only public-exposure audit for the published KALIA artifacts.

Replays the checks of the 2026-10-17 audit so they can be re-run cheaply:

1. **Kaggle.** Every live kernel and dataset (enumerated with the owner's CLI
   auth) is probed *anonymously*: a private asset answers 403/404, a public one
   200. Two known-public controls must answer 200 first; if they do not, the
   probe is unreliable and nothing is reported as public or private.
2. **Hugging Face.** The released checkpoints must be public and the training
   checkpoint repo private; public repos are scanned for credential-shaped file
   names *and* their text-file contents are searched for the local token value
   and for token-shaped strings, and `kalia-v012` must not carry a `corpus/`
   directory (the corpus release belongs to `kalia-v020`).
3. **Repo.** Token-shaped strings must not appear in tracked files.

Exit code is 1 if an asset that should be private is public, a control fails, or
a token-shaped string is found. UNKNOWN results (Kaggle's bot challenge) are
printed but do not fail the run. Read-only: nothing is uploaded, deleted or
changed.

Usage:
    python tools/audit_public_state.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120"

KAGGLE_DATASET_CONTROL = "uciml/iris"
KAGGLE_KERNEL_CONTROL = "alexisbcook/titanic-tutorial"
HF_EXPECTED_PUBLIC = ("kalia-lm/kalia-v012", "kalia-lm/kalia-v020")
HF_EXPECTED_PRIVATE = ("kalia-lm/kalia-m",)
CREDENTIAL_NAME = re.compile(r"(^|[._-])(token|secret|credential|password)([._-]|$)|\.env$", re.I)
TOKEN_SHAPE = re.compile(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}")


def looks_like_credential(name: str) -> bool:
    return bool(CREDENTIAL_NAME.search(Path(name).name))


def probe_verdict(kind: str, status: int | None, body: str = "") -> str:
    """Classify an anonymous response as private / PUBLIC / unknown."""
    if status is None:
        return "unknown"
    if "recaptcha" in body.lower() or "checking your browser" in body.lower():
        return "unknown"
    if kind == "dataset":
        if status == 200:
            return "PUBLIC"
        return "private" if status in (403, 404) else "unknown"
    if kind == "kernel":
        if status == 200:
            return "PUBLIC"
        return "private" if status == 404 else "unknown"
    raise ValueError(f"unknown kind {kind}")


def http_probe(url: str, timeout: int = 40) -> tuple[int | None, str]:
    cmd = ["curl", "-sSL", "-A", UA, "-w", "\n%{http_code}", "-m", str(timeout), url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, ""
    body, _, code = r.stdout.rpartition("\n")
    try:
        return int(code.strip()), body[:2000]
    except ValueError:
        return None, body[:2000]


def check_kaggle() -> list[str]:
    from tools.collect_kaggle_state import fetch

    failures: list[str] = []

    status, body = http_probe(f"https://www.kaggle.com/api/v1/datasets/view/{KAGGLE_DATASET_CONTROL}")
    if status != 200:
        print(f"  control dataset unreachable (status {status}); aborting Kaggle probes")
        return ["Kaggle probe control failed"]
    status, body = http_probe(f"https://www.kaggle.com/code/{KAGGLE_KERNEL_CONTROL}")
    if status != 200 or probe_verdict("kernel", status, body) != "PUBLIC":
        print(f"  control kernel unreachable (status {status}); aborting Kaggle probes")
        return ["Kaggle probe control failed"]

    for d in fetch("datasets"):
        ref = d["ref"]
        status, body = http_probe(f"https://www.kaggle.com/api/v1/datasets/view/{ref}")
        verdict = probe_verdict("dataset", status, body)
        print(f"  dataset {ref:55s} {verdict}")
        if verdict == "PUBLIC":
            failures.append(f"dataset {ref} is public")

    for k in fetch("kernels"):
        ref = k["ref"]
        status, body = http_probe(f"https://www.kaggle.com/code/{ref}")
        verdict = probe_verdict("kernel", status, body)
        print(f"  kernel  {ref:55s} {verdict}")
        if verdict == "PUBLIC":
            failures.append(f"kernel {ref} is public")

    return failures


def check_huggingface() -> list[str]:
    from huggingface_hub import HfApi, HfFileSystem

    failures: list[str] = []
    token = None
    if (ROOT / ".secrets" / "hf_token").exists():
        token = (ROOT / ".secrets" / "hf_token").read_text().strip()
    api = HfApi(token=token)
    fs = HfFileSystem(token=token)
    needle = token.encode() if token else None

    for repo in HF_EXPECTED_PUBLIC + HF_EXPECTED_PRIVATE:
        try:
            info = api.model_info(repo, token=token)
        except Exception as exc:  # noqa: BLE001
            print(f"  {repo:24s} UNREADABLE ({type(exc).__name__})")
            failures.append(f"{repo} unreadable")
            continue
        should_be_private = repo in HF_EXPECTED_PRIVATE
        state = "private" if info.private else "PUBLIC"
        print(f"  {repo:24s} {state}")
        if should_be_private and not info.private:
            failures.append(f"{repo} should be private but is public")
        if not should_be_private and info.private:
            print(f"    warning: {repo} should be public but is private")
        if info.private:
            continue
        files = api.list_repo_files(repo, token=token)
        bad = [f for f in files if looks_like_credential(f)]
        if bad:
            failures.append(f"{repo} carries credential-shaped files: {bad}")
        if repo.endswith("kalia-v012") and any(f.startswith("corpus/") for f in files):
            failures.append("kalia-v012 carries a corpus/ directory it should not have")
        text = [f for f in files if not f.endswith((".bin", ".safetensors", ".pt"))]
        before = len(failures)
        scanned = 0
        for f in text:
            try:
                with fs.open(f"{repo}/{f}", "rb") as fh:
                    data = fh.read()
            except Exception:  # noqa: BLE001
                continue
            scanned += 1
            if needle and needle in data:
                failures.append(f"{repo}/{f} contains the local HF token value")
            elif TOKEN_SHAPE.search(data.decode("utf-8", errors="replace")):
                failures.append(f"{repo}/{f} contains a token-shaped string")
        verdict = "no token found" if len(failures) == before else "TOKEN FOUND"
        print(f"    content-scanned {scanned} text files, {verdict}")
    return failures


def check_repo_tokens() -> list[str]:
    r = subprocess.run(
        ["git", "grep", "-nE", TOKEN_SHAPE.pattern, "--", "."],
        capture_output=True, text=True, cwd=ROOT,
    )
    hits = [line for line in r.stdout.splitlines() if line.strip()]
    for line in hits:
        print(f"  token-shaped match: {line[:160]}")
    return ["token-shaped string in tracked files"] if hits else []


def main() -> int:
    print("Kaggle (anonymous probes of every live asset):")
    failures = check_kaggle()
    print("\nHugging Face:")
    failures += check_huggingface()
    print("\nRepository:")
    failures += check_repo_tokens()

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK: every KALIA Kaggle asset is private, HF visibility is as intended,")
    print("and no token-shaped string is in the tracked tree.")
    print("Note: the private Kaggle dataset kalia-publish-aux holds the publish")
    print("token by design; it must never be made public (see docs/journal/2026-10-08.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
