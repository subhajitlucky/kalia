"""Verify the pre-registration ledger.

The ledger claims each entry is "the SHA-256 of a pre-registration file at the
moment of its registration commit", and that anyone can verify a prediction
existed before its results by recomputing the hash at that commit.

Two ways that claim silently fails, both of which happened here:

1. **Hashing the wrong thing.** `git hash-object` returns a SHA-1 blob id (40
   hex), not the SHA-256 the ledger records (64 hex). A verifier built on it
   reports FAIL for every row and looks like tampering.
2. **Hashing the wrong revision.** Results sections are appended *after*
   registration -- the pre-registrations require it ("results are reported
   regardless of outcome"). So the working tree no longer matches the registered
   hash, by design. The hash must be recomputed at the commit where the file was
   registered.

So verification is: find the commit that introduced each file, hash the blob
*from that commit*, and compare. A mismatch there means the prediction was edited
after the fact, which is the thing the ledger exists to catch.

Usage:
    python tools/verify_prereg_ledger.py
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs" / "preregistrations" / "LEDGER.md"
ROW = re.compile(
    r"^\|\s*(?P<date>\d{4}-\d{2}-\d{2})\s*\|\s*(?P<id>[^|]+?)\s*\|\s*"
    r"`(?P<file>[^`]+)`\s*\|\s*`(?P<hash>[0-9a-f]+)`\s*\|"
)


def sha256_of_blob(rev: str, path: str) -> str | None:
    """SHA-256 of a file's contents at a given revision."""
    rc = subprocess.run(
        ["git", "show", f"{rev}:{path}"], capture_output=True, cwd=ROOT
    )
    if rc.returncode != 0:
        return None
    return hashlib.sha256(rc.stdout).hexdigest()


def registration_commit(path: str) -> str | None:
    """The commit that added the file, i.e. the newest commit naming it as added."""
    rc = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", path],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    commits = rc.stdout.split()
    return commits[0] if commits else None


def main() -> int:
    rows = []
    for line in LEDGER.read_text().splitlines():
        m = ROW.match(line)
        if m:
            rows.append(m.groupdict())

    if not rows:
        print("no ledger rows parsed")
        return 1

    print(f"pre-registration ledger: {len(rows)} entries\n")
    print(f"{'ID':22s} {'verified at registration commit':30s} {'tree':10s}")
    print("-" * 68)

    bad = []
    for row in rows:
        path, want = row["file"], row["hash"]

        if len(want) != 64:
            bad.append((row["id"], f"ledger hash is {len(want)} hex, not SHA-256 (64)"))
            print(f"{row['id']:22s} {'MALFORMED HASH':30s}")
            continue

        commit = registration_commit(path)
        if commit is None:
            bad.append((row["id"], "no commit adds this file"))
            print(f"{row['id']:22s} {'NOT IN GIT':30s}")
            continue

        at_reg = sha256_of_blob(commit, path)
        if at_reg is None:
            bad.append((row["id"], f"cannot read {path} at {commit[:8]}"))
            print(f"{row['id']:22s} {'UNREADABLE':30s}")
            continue

        ok_reg = at_reg == want
        tree = sha256_of_blob("HEAD", path)
        drift = "" if tree == want else "  (results appended after registration)"

        print(
            f"{row['id']:22s} {('OK ' + commit[:8]) if ok_reg else 'MISMATCH':30s} "
            f"{'same' if tree == want else 'changed':10s}{drift}"
        )
        if not ok_reg:
            bad.append((row["id"], f"registered {want[:12]} but {commit[:8]} has {at_reg[:12]}"))

    print()
    if bad:
        print(f"{len(bad)} entr{'y' if len(bad) == 1 else 'ies'} FAILED:")
        for rid, why in bad:
            print(f"  {rid}: {why}")
        return 1
    print(f"all {len(rows)} entries verify: every registered hash matches the file at the")
    print("commit that introduced it. Files marked 'changed' had results appended")
    print("afterwards, which the pre-registrations explicitly require.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
