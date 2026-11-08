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
import subprocess
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
    ("v0.2.0", "Compliance-rebuilt corpus, full 4,770 steps; at 2,000 samples only LAMBADA fails S-A",
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
    ("v0.3.0 epoch2", "Second epoch, 2,000-step re-warm; E-1/E-2/E-3 pass, release candidate (val 2.7743, all 2k tasks improved)",
     "2026-10-27-v030-second-epoch.md", ["docs/results/v030-epoch2/*"]),
    ("v0.3.1 merge", "v0.2.0/v0.3.0 weight soup; val 2.7579 (−0.0164), 2k all positive, CL-0 −0.0157; released public at kalia-v031",
     "2026-11-01-v031-merge-plan.md", ["docs/results/v031-merge/*"]),
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
    ("X31", "QK-Norm/soft-cap isolation at two seeds; QK-Norm carries the gain, soft-cap does not",
     "2026-10-21-X31-qk-softcap-isolation.md", ["docs/results/x31/*"]),
    ("bench pair 2k", "2,000-sample benchmark pair; the 500-sample ARC-Easy regression was noise",
     None, ["docs/results/bench-pair-2k/*"]),
    ("X32", "Entity-selection circuit probe; layer-9 heads carry entity mass but ablation hurts",
     "2026-10-22-X32-entity-selection-circuit.md", ["docs/results/x32/*"]),
    ("X33", "Linear midpoint readout; probe beats greedy +0.0155, ~35% of the coverage gap",
     "2026-10-22-X33-linear-readout.md", ["docs/results/x33/*"]),
    ("X34", "Seed-1007 replication of X33; the midpoint readout does not clear the registered bar",
     "2026-10-22-X34-seed-replication.md", ["docs/results/x34/*"]),
    ("X35", "Static code-mixture screen; code10 −0.0007, code25 +0.0058; S-1 fails, Stage 2 not run",
     "2026-10-26-X35-static-mixture-screen.md", ["docs/results/x35/**/*"]),
    ("X36", "Entity-memory micro; retention passes, val +0.076 vs matched control fails M-2; no scale",
     "2026-10-27-X36-entity-memory-micro.md", ["docs/results/x36/**/*"]),
    ("X37", "Pāṇinian rule grammar; S-1/S-2 fail, S-3 pass; no scale",
     "2026-10-27-X37-sutra-micro.md", ["docs/results/x37/**/*"]),
    ("X38", "Counterfactual residual bottleneck; C-1/C-1b pass but C-2/C-3/C-5 fail; stable stream collapses to rank 1.5; no scale",
     "2026-10-28-X38-crb-micro.md", ["docs/results/x38/**/*"]),
    ("X40", "Intervention pretraining; I-1 fails (+0.204/+0.169 val tax) and I-3 fails (+0.028/0.000 unseen over pairs); workspace alive but no composition gain; closed at 30M, no scale",
     "2026-11-01-X40-intervention-pretraining.md", ["docs/results/x40/**/*"]),
    ("X40 bench", "Clean 2x2 benchmark + gold-edit interface test; S-P at chance; number+place and rename+swap fail even with the edited prefix; the edit is not the missing piece; X40 stays closed",
     None, ["docs/results/x40-bench/**/*"]),
    ("V032 adapter", "Frozen-base output-adapter fine-tune of v0.3.1; val 2.7493/2.7495 vs 2.7580 (−0.0087/−0.0085), both seeds pass; frozen base; 2k guard passes but is neutral; val-only internal candidate, GitHub only, no HF",
     "2026-11-04-V032-output-adapter.md", ["docs/results/v032-adapter/**/*"]),
    ("X41 edit compiler", "LoRA edit compiler + frozen v0.3.1 reader; full 2x2 baseline normal S-P 0.170 vs gold 0.513; compiler number+place 0.000->0.694 both seeds, S-P 0.428/0.410, U-P 0.568/0.635; rename+swap 0.354/0.483, so C-3 fails on one seed; partial advancement",
     "2026-11-05-X41-edit-compiler.md", ["docs/results/x41-edit-compiler/**/*"]),
    ("X42 reader ensemble", "Reader-guided selection between the two X41 compiler edits; dev-selected MinEntropy; held-out test rename+swap 0.4286 vs 0.45 bar; oracle capped at best single seed 0.4898; null",
     "2026-11-05-X42-reader-ensemble.md", ["docs/results/x42-reader-ensemble/**/*"]),
    ("X43 THPT", "Template-held-out pair training; all four pair types taught on three extra templates only; S-P 0.480/0.493, number+place 0.694/0.694, rename+swap 0.592/0.646, U-P 0.860/0.873; D-1..D-6 pass; 0.97 GPU-h; capability release v0.4.0",
     "2026-11-05-X43-template-pairs.md", ["docs/results/x43-thpt/**/*"]),
    ("X44/X45 fresh", "Fresh names/templates; F-P 0.085 normal -> 0.523/0.493, number+place 0.455 both, rename+swap 0.406/0.416 (F-3 fail), 3-op 0.283/0.365 (F-6 fail); partial OOD generalization; X43 release remains scoped",
     "2026-11-05-X44-fresh-depth.md", ["docs/results/x44-fresh/**/*"]),
    ("X46 broad", "Broad-template compiler; 17 templates, 1/2/3-op, X44 fresh held out; F-P 0.683/0.685, rename+swap 0.723/0.663, 3-op 0.615/0.558, compiler EM 0.355/0.378; G-1..G-6 pass; 0.96 GPU-h; v0.4.1 capability release",
     "2026-11-05-X46-broad-compiler.md", ["docs/results/x46-broad/**/*"]),
    ("K-Tok v0", "Own byte-level BPE tokenizers 32k/50k/64k + digit-aware 32k on 50M corpus tokens; K-Tok 32k English 4.380 vs GPT-2 4.319 B/tok, code 3.162 vs 2.230, numbers 2.152 vs 2.478; frees 9.35M tied-embedding params; all lossless; 32k candidate for A/B",
     None, ["docs/results/ktok-v0/**/*"]),
    ("X48 tokenizer A/B", "GPT-2 BPE vs K-Tok 32k at 30M, same 312M-token text, 4770 steps, seed 1338; GPT-2 mean 0.3657 vs K-Tok 0.3629 (−0.28 pp); LAMBADA +3.85 but OpenBookQA −2.80/ARC-C −1.45; val bpB 4.5% worse; H-1/H-3 fail; GPT-2 BPE stays, K-Tok parked",
     "2026-11-05-X48-tokenizer-ab.md", ["docs/results/x48-tokenizer-ab/**/*"]),
    ("KALIA-Vision v0", "Frozen KALIA text encoder + 3.75M conditional DDPM from scratch on 50k synthetic 32x32 shapes; 20k steps/40 min; held-out shape 0.947, color 0.968, position 0.852; W-1..W-5 pass; v0.5.0-vision capability release",
     "2026-11-06-kalia-vision-v0.md", ["docs/results/kalia-vision-v0/**/*"]),
    ("KALIA-Vision v1", "Two-object 32x32 scenes; frozen KALIA text encoder + 3.75M conditional DDPM from scratch; 25k steps/45.5 min; left shape 0.991/color 0.980, right shape 0.993/color 0.976, exact scene 0.948; V1-1..V1-6 pass; v0.5.1-vision-two",
     "2026-11-06-kalia-vision-v1.md", ["docs/results/kalia-vision-v1/**/*"]),
    ("KALIA-Audio v0", "From-scratch 136k 1D CNN on six synthetic formant commands; 15 epochs/0.38 min; test 0.912, val 0.916; all six commands present; A-1..A-3 pass; v0.6.0-audio-commands",
     "2026-11-06-kalia-audio-v0.md", ["docs/results/kalia-audio-v0/**/*"]),
    ("KALIA-Audio v1.1", "Real Speech Commands 12 words; log-STFT 201x101 + 2D CNN from scratch; 10.8k train/4.9k official test; 391,884 params; 30 epochs/5.7 min; test 0.856, val 0.852; v1 raw-waveform 0.655; R-1..R-3 pass; v0.6.1-audio-speech",
     "2026-11-06-kalia-audio-v1.1-mel.md", ["docs/results/kalia-audio-v1.1/**/*"]),
    ("X49 data mix", "Reasoning-heavy 58M continued pretraining; control 60/20/15/5 vs treatment 45/5/40/10; 1,500 steps each; control 0.4198 vs treatment 0.4280 (+0.82 pp); BoolQ +5.45/WinoGrande +1.89 but weak-task avg -0.10; K-1/K-3 fail; mixture parked",
     "2026-11-06-X49-data-mix.md", ["docs/results/x49-data-mix/**/*"]),
    ("X50 soup", "50/50 v0.3.1 + X49-treatment soup; mean 0.4284 vs 0.4258 (+0.26 pp); weak-task avg -0.13 pp; S-1/S-3 fail; no v0.3.2",
     "2026-11-08-X50-soup.md", ["docs/results/x50-soup/**/*"]),
    ("KALIA-Vision v2", "From-scratch CNN encoder + 4 image tokens + frozen KALIA LoRA caption decoder on two-object scenes; 2.33M trainable; 8 epochs/24.4 min; held-out exact 1.000, blind 0.006, shuffled 0.002; V2-1..V2-5 pass; v0.5.2-vision-see",
     "2026-11-08-kalia-vision-v2.md", ["docs/results/kalia-vision-v2/**/*"]),
    ("KALIA-Vision v2.1", "Compositional split: 29/144 two-object combinations held out; 39,853 train; held-out exact 0.992, seen 0.997, gap 0.005; attributes 0.992-1.000; blind 0.000/shuffled 0.003; 2.33M trainable/20.3 min; C-1..C-5 pass; v0.5.3-vision-compose",
     "2026-11-08-kalia-vision-v2.1-compose.md", ["docs/results/kalia-vision-v2.1/**/*"]),
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


def _tracked(root: Path) -> set[str] | None:
    """Paths git tracks, or None outside a git checkout.

    The registry describes the *committed* archive. The working tree can hold
    ignored evaluation output that CI never sees; counting it made the local
    registry disagree with the checkout (271 files locally, 257 in CI). A clean
    `git archive` has no .git -- there every file is tracked, so None is right.
    """
    r = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
    )
    if r.returncode != 0:
        return None
    return {p for p in r.stdout.decode().split("\0") if p}


def expand(globs: list[str], root: Path = ROOT) -> list[str]:
    tracked = _tracked(root)
    found = []
    for pattern in globs:
        for p in sorted(root.glob(pattern)):
            if not p.is_file():
                continue
            rel = str(p.relative_to(root))
            if tracked is not None and rel not in tracked:
                continue
            found.append(rel)
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
