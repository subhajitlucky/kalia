#!/usr/bin/env python3
"""Rebuild docs/results/experiments.json from the raw Kaggle logs.

Why this exists
---------------
Every experiment result in this project lived in two places only: a markdown
table inside a pre-registration, and a Kaggle log that is transient and
partially truncated. Nothing machine-readable was committed. The X21 result --
the one that closed the architecture line -- existed in the repository as
prose, and its four validation curves existed nowhere at all.

This script parses the raw logs (archived under ``docs/results/logs/``) into a
single JSON record, so that "what did we measure" is a question with a data
answer rather than a question about prose.

Two properties matter more than the parsing itself:

1. **Provenance.** Every number carries the kernel and the log line it came
   from, so a reader can check a value without re-running anything. A number
   with no source line is a number we typed, and those are marked ``source:
   "prereg"`` rather than pretending to be measured.

2. **Not inventing data.** Where a log is truncated, this records the
   truncation instead of filling the hole. ``kaggle_live_log.py`` returns a
   rolling window, so older arms fall out of it; that happened to X20 and X19,
   where only the treatment arm survived. Those arms are ``truncated: true``.

Usage:
    python tools/collect_results.py --check     # verify committed data is current
    python tools/collect_results.py             # rewrite it
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "results"
LOGDIR = RESULTS / "logs"
OUT = RESULTS / "experiments.json"

# arm name -> (params, source prereg that first ran it)
ARM_PARAMS = {
    "micro-base": 29_920_512,
    "micro-nope": 29_920_512,
    "micro-gated": 30_072_768,
    "micro-branchnorm": 29_922_816,
    "micro-staticgate": 29_999_040,
    "micro-base-s1338": 29_920_512,
    "micro-staticgate-s1338": 29_999_040,
    "micro-base-s1339": 29_920_512,
    "micro-staticgate-s1339": 29_999_040,
}

KERNELS = {
    "x18-close": "subhajitlucky/kalia-x18-close",
    "x19-branchnorm": "subhajitlucky/kalia-x19-branchnorm",
    "x20-staticgate": "subhajitlucky/kalia-x20-staticgate",
    "x21": "subhajitlucky/kalia-x21-replication",
}

CURVE = re.compile(r"^(micro-[a-z0-9-]+)\s+((?:\d+=[0-9.]+\s*)+)")
SUMMARY = re.compile(r"^(micro-[a-z0-9-]+)\s+\d+,([0-9.]+)")

# Arms whose numbers the archived logs do not contain. kaggle_live_log.py returns
# a rolling window, and x18-close's window closed before its ablation summary
# printed -- that kernel also ran benchmarks, so the log is 522 lines of
# lm-eval output. These are transcribed from the pre-registrations and are the
# weakest records in the file. X18's 4.7226 in particular is the number D48
# reclassified as a null, and it currently rests on a markdown table.
PREREG_ONLY = {
    "x18-close": {"micro-base": 4.7662, "micro-gated": 4.7226, "micro-nope": 4.7772},
    "x19-branchnorm": {"micro-base": 4.7662},
    "x20-staticgate": {"micro-base": 4.7662},
}


def unescape(raw: str) -> str:
    """kaggle_live_log.py returns one JSON-escaped blob, not a real log."""
    return raw.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\x00", "")


def parse_log(name: str) -> dict:
    path = LOGDIR / f"{name}.log"
    if not path.exists():
        return {"kernel": KERNELS.get(name, name), "present": False}
    text = unescape(path.read_text(errors="replace"))
    curves: dict[str, dict] = {}
    finals: dict[str, float] = {}
    for line in text.splitlines():
        line = line.rstrip("\r")
        m = CURVE.match(line)
        if m:
            arm, pairs = m.group(1), m.group(2)
            curves[arm] = {int(k): float(v) for k, v in (p.split("=") for p in pairs.split())}
        m = SUMMARY.match(line)
        if m:
            finals[m.group(1)] = float(m.group(2))
    return {
        "kernel": KERNELS.get(name, name),
        "present": True,
        "curves": curves,
        "summary_finals": finals,
        # ablate.py prints one summary row per arm; if a log has curves for
        # fewer arms than the experiment should have, it was truncated.
        "truncated": bool(curves) and len(finals) < len(curves),
    }


def build() -> dict:
    records: list[dict] = []
    for name in KERNELS:
        parsed = parse_log(name)
        if not parsed.get("present"):
            records.append({"experiment": name, "kernel": parsed["kernel"], "present": False})
            continue
        entry = {
            "experiment": name,
            "kernel": parsed["kernel"],
            "present": True,
            "truncated": parsed.get("truncated", False),
            "arms": [],
        }
        seen = set()
        for arm, curve in parsed["curves"].items():
            final = curve[max(curve)]
            seen.add(arm)
            records_arm = {
                "arm": arm,
                "params": ARM_PARAMS.get(arm),
                "val_loss_final": round(final, 4),
                "val_curve": {str(k): v for k, v in sorted(curve.items())},
                "source": f"docs/results/logs/{name}.log",
            }
            entry["arms"].append(records_arm)
        # Arms whose curve fell out of the rolling log window. Recorded, not guessed.
        for arm, val in parsed["summary_finals"].items():
            if arm not in seen:
                entry["arms"].append(
                    {
                        "arm": arm,
                        "params": ARM_PARAMS.get(arm),
                        "val_loss_final": val,
                        "val_curve": None,
                        "source": f"docs/results/logs/{name}.log",
                        "truncated": True,
                    }
                )
        # Arms the archived log does not cover at all. These numbers come from
        # the pre-registration prose, and are labelled as such: `source: prereg`
        # means "a human wrote this down", not "this was parsed from a run".
        # Marking them honestly matters more than filling every row.
        for arm, val in PREREG_ONLY.get(name, {}).items():
            if arm not in seen and arm not in entry["arms"]:
                entry["arms"].append(
                    {
                        "arm": arm,
                        "params": ARM_PARAMS.get(arm),
                        "val_loss_final": val,
                        "val_curve": None,
                        "source": f"prereg:{name}",
                        "truncated": True,
                    }
                )
        if not entry["arms"]:
            entry["truncated"] = True
        records.append(entry)

    # Deltas, computed here so no reader has to trust a subtraction in prose.
    by_exp = {r["experiment"]: r for r in records}
    for exp, control_arm in (
        ("x18-close", "micro-base"),
        ("x19-branchnorm", "micro-base"),
        ("x20-staticgate", "micro-base"),
    ):
        arms = {a["arm"]: a for a in by_exp.get(exp, {}).get("arms", [])}
        if control_arm not in arms:
            continue
        base = arms[control_arm]["val_loss_final"]
        for arm in arms.values():
            arm["delta_vs_control"] = round(arm["val_loss_final"] - base, 4)
    for exp in ("x21",):
        arms = {a["arm"]: a for a in by_exp.get(exp, {}).get("arms", [])}
        for seed in (1338, 1339):
            ctl, trt = arms.get(f"micro-base-s{seed}"), arms.get(f"micro-staticgate-s{seed}")
            if ctl and trt:
                d = round(trt["val_loss_final"] - ctl["val_loss_final"], 4)
                trt["delta_vs_control"] = d
                trt["seed"] = seed
                ctl["seed"] = seed
    return {
        "schema": "kalia-results/1",
        "note": (
            "Regenerate with tools/collect_results.py from the archived logs in "
            "docs/results/logs/. An arm with val_curve: null was truncated out of the "
            "rolling log window; its final value is recorded, the curve is not."
        ),
        "experiments": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    args = ap.parse_args()
    data = build()
    text = json.dumps(data, indent=2) + "\n"
    if args.check:
        if not OUT.exists():
            print(f"{OUT} does not exist")
            return 1
        current = json.loads(OUT.read_text())
        if current == data:
            print("experiments.json is current")
            return 0
        print("experiments.json is STALE relative to the archived logs")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    n_arms = sum(len(e.get("arms", [])) for e in data["experiments"])
    n_trunc = sum(1 for e in data["experiments"] for a in e.get("arms", []) if a.get("truncated"))
    print(f"wrote {OUT} ({len(data['experiments'])} experiments, {n_arms} arms, {n_trunc} truncated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
