"""Build docs/results/RUNS.md — one row per run, with its artifacts or its gap.

The whole-record audit found the archive uneven: recent work (X17, X28, X29) is
fully preserved while X16 — four arms, two seeds, the project's first rejected
experiment — exists only as four numbers in prose. Nothing was hiding it; the
absence simply had nowhere to show up.

This gives it somewhere. For every run it expands the globs where that run's
artifacts *would* live and reports what it finds. A run with nothing found is
reported as a gap, which is the point: the missing thing becomes a line in a
table rather than a silence.

The run list is declared because the repository's naming is genuinely irregular
— X17's curves are `val_log-control-s1338.csv` inside `x17/`, while X29's are per
arm inside `x29/<arm>/`. Encoding what "has artifacts" means for each run is
more honest than a heuristic that guesses and can be wrong in both directions.

Usage:
    python tools/build_runs_registry.py           # rewrite docs/results/RUNS.md
    python tools/build_runs_registry.py --check   # fail if the file is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "results" / "RUNS.md"

#: (id, what happened, registration, globs where its artifacts live).
#: An empty glob list is a run we know happened and never archived -- that is a
#: finding, not an omission, so it is written down rather than dropped.
RUNS: list[tuple[str, str, str | None, list[str]]] = [
    ("v0.1.0", "AdamW baseline; stopped at step 2,250", None, []),
    ("v0.1.1", "Muon for hidden matrices; micro-ablation 3.5937 vs 3.8041",
     None, []),
    ("v0.1.2", "QK-Norm + soft-cap; stopped at step 3,478 of 4,770 on quota",
     None, ["docs/curves/v012_*.csv", "docs/eval/*v012*"]),
    ("v0.2.0", "Compliance-rebuilt corpus, full 4,770 steps; S-A fails on two benchmarks",
     None, ["docs/releases/v020-release-gate.md", "docs/eval/*v2b*"]),
    ("X16", "Chunk-preserving reversal training; rejected on both bars, both seeds",
     "2026-09-23-X16-chakravyuha-reversal.md", []),
    ("X18", "NoPE + gated residual; gate measured inert",
     "2026-09-28-X18-frontier-attention.md", ["docs/results/logs/x18-close.log"]),
    ("X19", "Branch-norm isolation; the gain lives in the gate",
     "2026-09-28-X19-branch-norm-isolation.md", ["docs/results/logs/x19-branchnorm.log"]),
    ("X20", "Static per-channel gate; largest effect ever measured",
     "2026-09-29-X20-static-gate.md", ["docs/results/logs/x20-staticgate.log"]),
    ("X21", "Replication of X20 at fresh seeds; inverted",
     "2026-09-29-X21-replication.md", ["docs/results/logs/x21.log"]),
    ("v0.3.0 Step 1", "58M control seed spread, 0.2892; tripped the programme stop",
     "2026-09-29-v030-continual-update.md",
     ["docs/results/step1_spread*.json", "docs/results/logs/v030-baseline*"]),
    ("X25", "Test-time training, two stages; closed null across a 40x range",
     "2026-10-05-X25-ttt-inference.md", ["docs/results/x25-stage*.json"]),
    ("X26", "Kautilya adaptive mixture pilot; ate dessert",
     "2026-10-09-X26-kautilya-pilot.md", ["docs/results/x26-pilot/*"]),
    ("X28", "Kautilya trio (static/level/trend); dessert replicates",
     "2026-10-09-X28-kautilya-trend.md", ["docs/results/x28-trio/*"]),
    ("X29", "Patha replay ladder, four arms",
     "2026-10-11-X29-patha-replay.md", ["docs/results/x29/uniform/*", "docs/results/x29/krama/*",
                                        "docs/results/x29/jata/*", "docs/results/x29/ghana/*",
                                        "docs/results/x29/forgetting_x29.jsonl"]),
    ("X29 replication", "Second seed (1402) for uniform and static",
     None, ["docs/results/x29/replication/*"]),
    ("X29 noreplay", "The single-variable replay control at both seeds",
     None, ["docs/results/x29/noreplay/*"]),
    ("X17", "Intra-document masking, two seeds; registered 2026-09-28, run 2026-10-18",
     "2026-09-28-X17-doc-mask.md", ["docs/results/x17/*"]),
    ("X17 matched", "Rerun with the eval masking the run trained under; effect undetectable",
     None, ["docs/results/x17-matched/*"]),
    ("retrieval probe", "Copy utilisation, BM25 top-3, and the log-prob selector",
     None, ["docs/results/retrieval-probe.json"]),
    ("entity decode", "Entity-consistent decoding vs greedy",
     None, ["docs/results/entity-decode.json"]),
    ("entity holdout", "Does the entity rule survive a half it never selected on? No; coverage survives",
     None, ["docs/results/entity-holdout/*"]),
    ("corpus gates", "Contamination gate for the rebuilt corpus",
     None, ["docs/results/corpus-gates.json"]),
]

#: Runs whose raw artifacts are known to be absent or incomplete, with why. The
#: registry flags a gap by itself; this records what cannot be recovered.
KNOWN_GAPS = {
    "v0.1.0": "No logs, curves or eval files survived. The final loss is quoted in "
              "the README from the run's own console output.",
    "v0.1.1": "Same: the ablation number survives in prose, the run does not.",
    "X16": "**No raw artifacts at all.** Four arms, two seeds, the project's first "
           "rejected experiment; the numbers exist only in the journal entry. If "
           "anyone asked to see the run, there would be nothing to show.",
}


def expand(globs: list[str], root: Path = ROOT) -> list[str]:
    found = []
    for pattern in globs:
        found += [str(p.relative_to(root)) for p in sorted(root.glob(pattern)) if p.is_file()]
    return found


def build(root: Path = ROOT) -> str:
    rows = []
    gaps = []
    for run_id, what, registration, globs in RUNS:
        files = expand(globs, root)
        if not files:
            status = "**GAP** — nothing archived"
            gaps.append(run_id)
        else:
            status = f"{len(files)} file{'s' if len(files) != 1 else ''}"
        rows.append((run_id, what, registration, status, files))

    lines = [
        "# Run registry",
        "",
        "Every run, and whether its evidence is actually in the repository.",
        "",
        "Generated by `python tools/build_runs_registry.py`; `--check` fails when this",
        "file stops matching the archive. Do not hand-edit.",
        "",
        "A **GAP** means the run happened and the repository holds nothing from it.",
        "That is a finding, not an omission — the whole-record audit found the archive",
        "uneven, and this is where the unevenness is visible instead of silent.",
        "",
        "| run | what | registered | artifacts |",
        "|---|---|---|---|",
    ]
    for run_id, what, registration, status, _ in rows:
        reg = "yes" if registration else "—"
        lines.append(f"| `{run_id}` | {what} | {reg} | {status} |")

    lines += ["", "## Gaps", ""]
    if not gaps:
        lines.append("None.")
    else:
        for run_id in gaps:
            reason = KNOWN_GAPS.get(run_id, "_not recorded yet_")
            lines.append(f"- **{run_id}** — {reason}")

    lines += ["", "## Artifacts by run", ""]
    for run_id, _, _, status, files in rows:
        if not files:
            continue
        shown = files if len(files) <= 6 else files[:6] + [f"… {len(files) - 6} more"]
        lines.append(f"**{run_id}**")
        lines += [f"- `{f}`" for f in shown]
        lines.append("")

    total = sum(len(f) for _, _, _, _, f in rows)
    lines += [
        "---",
        "",
        f"{len(rows)} runs tracked, {total} files archived, {len(gaps)} with no evidence.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    text = build()
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is stale; run python tools/build_runs_registry.py")
            return 1
        print(f"{OUT.relative_to(ROOT)} is current")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
