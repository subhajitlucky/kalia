"""Check that the public documents still say what the record says.

The 2026-10-18 whole-record audit found nine defects in the public documents and
no two in the same place: a model card against the shard notice, the README
against a decision entry, a licence matrix against its own table, an audit tool
against the dataset it had loaded. Every one was found by reading, and reading
does not scale or repeat.

This checks the four invariants that would have caught them, machine-checkably.
It is deliberately a small, closed rule set, not a general prose checker: each
rule below failed at least once in this repository's history, and a rule that
cannot name the defect it prevents does not belong here.

  1. datasets        every source the shard notice attributes must appear in
                     each card's HF `datasets:` list. Caught: both cards
                     omitting the code corpus entirely, and attributing it as
                     `bigcode/codeparrot-clean` where the loader uses
                     `codeparrot/codeparrot-clean`.
  2. probe role      no public document may call `corpus/probe_val.bin`
                     held-out. Caught: the same false statement in the shard
                     notice (2026-10-18) and then in the v0.2.0 card, which the
                     notice's correction missed.
  3. decision refs   every `D<n>` a public document cites must exist. Caught:
                     the README's index pointed at a decision table that had
                     stopped six entries earlier.
  4. release numbers the released val loss and bpB must appear, unchanged, in
                     the README and the card. Caught: nothing yet -- it is the
                     rule that keeps the one number the release rests on from
                     drifting away from the artifact that produced it.

Usage:
    python tools/audit_claims.py            # report
    python tools/audit_claims.py --check    # exit non-zero on any problem
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CARDS = ["docs/public/hf-model-card.md", "docs/public/hf-model-card-v020.md"]
NOTICE = "docs/legal/corpus-shard-notice.md"
DECISIONS = "docs/DECISIONS.md"
GATE = "docs/releases/v020-release-gate.md"
# Everything a reader can reach without cloning: the documents published or
# linked from the model repos, plus the two summary documents they point at.
PUBLIC = ["README.md", "docs/SYNTHESIS.md", *CARDS, NOTICE]

DATASET_ID = re.compile(r"`([\w.-]+/[\w.-]+)`")
DECISION_REF = re.compile(r"\bD(\d+)\b")


def read(root: Path, rel: str) -> str:
    return (root / rel).read_text()


# --- 1. the cards must attribute every source the notice attributes ---------

def notice_datasets(root: Path) -> set[str]:
    """Every `owner/name` the notice's source, licence and attribution cells
    name. Taken as a union rather than from one column because the notice
    records Cosmopedia in the licence cell and smollm-corpus in the source."""
    found: set[str] = set()
    for line in read(root, NOTICE).splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 6 or not cells[1].startswith("`"):
            continue  # not a shard row
        found |= set(DATASET_ID.findall(" | ".join(cells[3:6])))
    return found


def card_datasets(root: Path, rel: str) -> set[str]:
    text = read(root, rel)
    m = re.search(r"^---\n(.*?)\n---\n", text, re.S | re.M)
    if not m:
        raise ValueError(f"{rel}: no YAML front matter")
    block = re.search(r"^datasets:\n((?:- .*\n)+)", m.group(1), re.M)
    if not block:
        raise ValueError(f"{rel}: no `datasets:` list in its front matter")
    return {line[2:].strip() for line in block.group(1).splitlines()}


def check_datasets(root: Path) -> list[str]:
    required = notice_datasets(root)
    if not required:
        return [f"{NOTICE}: parsed no source datasets -- the table shape moved"]
    problems = []
    for rel in CARDS:
        missing = sorted(required - card_datasets(root, rel))
        if missing:
            problems.append(
                f"{rel}: HF `datasets:` omits {missing}, which {NOTICE} attributes"
            )
    return problems


# --- 2. the probe shard is not held-out data --------------------------------

NEGATIONS = ("not held-out", "not held out", "deliberately not")


def check_probe_role(root: Path) -> list[str]:
    problems = []
    for rel in PUBLIC:
        for n, line in enumerate(read(root, rel).splitlines(), 1):
            if "probe_val" not in line:
                continue
            if not re.search(r"held[- ]out", line):
                continue
            if any(neg in line for neg in NEGATIONS):
                continue
            problems.append(
                f"{rel}:{n}: calls probe_val held-out; it is a frozen prefix of the "
                "training stream, not a held-out set"
            )
    return problems


# --- 3. cited decisions must exist ------------------------------------------

def check_decision_refs(root: Path) -> list[str]:
    """A decision is declared either as a `## D<n>` heading or as a row of the
    summary table. Both count: D1-D47 are rows, D48 onward are body-only, and a
    checker that knew only about headings would report every citation of D42 as
    broken -- which is how this rule was written the first time."""
    text = read(root, DECISIONS)
    declared = set(re.findall(r"^## D(\d+)", text, re.M))
    declared |= set(re.findall(r"^\| D(\d+) \|", text, re.M))
    problems = []
    for rel in PUBLIC:
        cited = set(DECISION_REF.findall(read(root, rel)))
        unknown = sorted(cited - declared, key=int)
        if unknown:
            problems.append(f"{rel}: cites D{', D'.join(unknown)}, which {DECISIONS} does not define")
    return problems


# --- 4. the released numbers must not drift from the artifact --------------

def released_numbers(root: Path) -> dict:
    m = re.search(r"```json\n(.*?)\n```", read(root, GATE), re.S)
    if not m:
        raise ValueError(f"{GATE}: no JSON block")
    return json.loads(m.group(1))


def check_release_numbers(root: Path) -> list[str]:
    gate = released_numbers(root)
    wanted = [str(gate["val_loss"]), str(gate["bpb"])]
    problems = []
    for rel in ["README.md", CARDS[1]]:
        text = read(root, rel)
        missing = [w for w in wanted if w not in text]
        if missing:
            problems.append(
                f"{rel}: does not carry the released {'/'.join(missing)} that "
                f"{GATE} re-scored from the published artifact"
            )
    return problems


CHECKS = {
    "datasets": check_datasets,
    "probe-role": check_probe_role,
    "decision-refs": check_decision_refs,
    "release-numbers": check_release_numbers,
}


def audit(root: Path = ROOT) -> dict[str, list[str]]:
    return {name: fn(root) for name, fn in CHECKS.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit non-zero on any problem")
    args = parser.parse_args()

    results = audit()
    failed = 0
    for name, problems in results.items():
        if problems:
            failed += len(problems)
            print(f"FAIL {name}")
            for p in problems:
                print(f"  {p}")
        else:
            print(f"ok   {name}")
    if failed:
        print(f"\n{failed} problem(s). A published claim and the record disagree.")
        return 1 if args.check else 0
    print("\npublic documents agree with the record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
