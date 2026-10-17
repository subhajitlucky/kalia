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

What this tool can and cannot promise about the current tree:

- It **can** prove a blob with the registered hash existed at the newest commit
  that introduced the file.
- It **cannot** make the current file identical to that blob: the project's
  convention is that results and status updates are written into the same file
  afterwards. So every drift from the registered blob is classified by line
  diff, not assumed: "extended" means inserted-only, "EDITED" means text was
  replaced or deleted in place. In-place edits are reported because they can
  touch a prediction's wording, and the whole point of hash-anchoring is that a
  reader decides which git revision they trust.
- A file missing from HEAD is a hard failure: the registered artifact must still
  exist.

Run it in CI (`.github/workflows/ci.yml`) as well as by hand.

Usage:
    python tools/verify_prereg_ledger.py
"""

from __future__ import annotations

import difflib
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


def text_of_blob(rev: str, path: str) -> list[str] | None:
    rc = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, cwd=ROOT)
    if rc.returncode != 0:
        return None
    return rc.stdout.decode("utf-8", errors="replace").splitlines()


def classify_drift(registered: list[str], current: list[str]) -> tuple[str, int, int]:
    """Classify current vs registered as inserted-only or in-place edited.

    Returns (label, inserted_lines, replaced_or_deleted_lines). "extended" means
    every difference is an insertion; "EDITED" means some registered text was
    replaced or removed, which a reader should inspect at the registration commit.
    """
    inserted = deleted = 0
    edited = False
    matcher = difflib.SequenceMatcher(None, registered, current, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            inserted += j2 - j1
        else:  # replace or delete touches registered text
            edited = True
            deleted += i2 - i1
            inserted += j2 - j1
    return ("EDITED" if edited else "extended", inserted, deleted)


def is_shallow() -> bool:
    rc = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        capture_output=True, text=True, cwd=ROOT,
    )
    return rc.stdout.strip() == "true"


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
    if is_shallow():
        print("SHALLOW CLONE: registration history is truncated, so a hash cannot be")
        print("checked against the commit that introduced each file. In a depth-1")
        print("checkout every file looks added at HEAD: unedited pre-registrations")
        print("verify against the wrong commit and edited ones report as mismatches.")
        print("Fetch full history (actions/checkout fetch-depth: 0, or")
        print("`git fetch --unshallow`) and rerun.")
        return 1

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

    bad: list[tuple[str, str]] = []
    extended: list[str] = []
    edited: list[str] = []
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
        tree_hash = sha256_of_blob("HEAD", path)
        if tree_hash is None:
            bad.append((row["id"], "registered file is missing from HEAD"))
            print(
                f"{row['id']:22s} {('OK ' + commit[:8]) if ok_reg else 'MISMATCH':30s} "
                f"{'MISSING':10s}  registered file deleted at HEAD"
            )
            continue
        if tree_hash == want:
            tree, detail = "same", ""
        else:
            label, inserted, deleted = classify_drift(
                text_of_blob(commit, path) or [], text_of_blob("HEAD", path) or []
            )
            tree = "changed"
            detail = f"  ({label}: +{inserted}/-{deleted} lines vs registration)"
            (edited if label == "EDITED" else extended).append(row["id"])

        print(
            f"{row['id']:22s} {('OK ' + commit[:8]) if ok_reg else 'MISMATCH':30s} "
            f"{tree:10s}{detail}"
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
    print("commit that introduced it. Results may be written into the same file after")
    print("registration, so tree drift is classified, not assumed:")
    print(f"  {len(extended)} extended (insertions only), {len(edited)} with in-place edits.")
    if edited:
        print("IN-PLACE EDITS — the registered text was rewritten or removed in these")
        print("files; inspect the diff against the registration commit instead of")
        print("trusting the current text:")
        for rid in edited:
            print(f"  {rid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
