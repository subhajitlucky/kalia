"""Stage one CL-0 retention-probe cohort into /tmp for pushing.

The probe is re-pushed whenever the set of finished arms changes
(``kernel_sources`` decides which checkpoints exist), so staging is a script
rather than a remembered sequence of copies -- and it is the one step that can
silently score the wrong thing. It therefore checks three invariants before
exiting:

1. the cohort exists in the kernel's own ``COHORTS`` table;
2. the metadata it selects mounts exactly the kernels that cohort declares --
   the arm set lives in two files and they are not allowed to drift;
3. the kernel source being pushed is exactly what is committed at HEAD, so the
   revision stamped into the ledger row describes the code that ran.

Invariant 3 is checked because the ledger's ``commit`` field is a claim about
provenance: a probe staged from an uncommitted working tree would record a
revision that never ran.

The checks are plain functions so they can be tested without a clean tree and
without touching Kaggle; ``main`` is the only part that reads git and writes
files.

Usage:
    python tools/stage_x29_forget_probe.py                           # the 1401 ladder
    python tools/stage_x29_forget_probe.py --cohort x29-replication  # seed-1402 replication
    kaggle kernels push -p /tmp/kalia-x29-replication-probe
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL_SRC = ROOT / "tools" / "kalia_forget_probe_kernel.py"

# cohort -> (metadata file, staging directory). The two cohorts are separate
# kernels so that a re-push for one can never rebuild the other's ledger.
COHORT_META = {
    "x29-1401": ("kernel-metadata-x29-forget-probe.json", Path("/tmp/kalia-x29-forget-probe")),
    "x29-replication": (
        "kernel-metadata-x29-replication-probe.json",
        Path("/tmp/kalia-x29-replication-probe"),
    ),
}

BASELINE_DATASETS = {"subhajitlucky/kalia-code-dev", "subhajitlucky/kalia-v020-ckpt"}


def parse_cohorts(source: str) -> dict:
    """Read COHORTS and COHORT out of the kernel without importing it: it is a
    Kaggle script, and importing it would execute the run."""
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("COHORTS", "COHORT"):
                    found[target.id] = ast.literal_eval(node.value)
    missing = {"COHORTS", "COHORT"} - found.keys()
    if missing:
        raise ValueError(f"{KERNEL_SRC.name} no longer defines {sorted(missing)}")
    return found


def declared_slugs(cohorts: dict, cohort: str) -> set[str]:
    return {slug for _, slug, _ in cohorts[cohort]["arms"]}


def check_metadata(meta: dict, cohorts: dict, cohort: str) -> set[str]:
    """Return the kernels the metadata mounts, or raise if it disagrees with
    the cohort's own declaration."""
    declared = declared_slugs(cohorts, cohort)
    mounted = {ref.split("/")[-1] for ref in meta["kernel_sources"]}
    if mounted != declared:
        raise ValueError(
            f"metadata mounts {sorted(mounted)} but cohort {cohort!r} declares "
            f"{sorted(declared)} -- the arm set lives in two files; fix one"
        )
    datasets = set(meta["dataset_sources"])
    if datasets != BASELINE_DATASETS:
        raise ValueError(
            f"metadata attaches datasets {sorted(datasets)}, expected "
            f"{sorted(BASELINE_DATASETS)} -- the v0.2.0 checkpoint dataset is the "
            "ledger's baseline row"
        )
    return mounted


def patch_source(source: str, cohort: str, commit: str) -> str:
    """Pin COHORT and COMMIT in the kernel copy, refusing anything ambiguous."""
    patched, n_cohort = re.subn(r'^COHORT = ".*?"', f'COHORT = "{cohort}"', source,
                                count=1, flags=re.MULTILINE)
    patched, n_commit = re.subn(r'^COMMIT = ".*?"', f'COMMIT = "{commit}"', patched,
                                count=1, flags=re.MULTILINE)
    if (n_cohort, n_commit) != (1, 1):
        raise ValueError(f"expected one COHORT and one COMMIT line, patched {n_cohort}/{n_commit}")
    return patched


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="x29-1401", choices=sorted(COHORT_META))
    parser.add_argument("--dest", type=Path, default=None, help="override the staging directory")
    args = parser.parse_args()

    try:
        source = KERNEL_SRC.read_text()
        parsed = parse_cohorts(source)
        cohorts = parsed["COHORTS"]
        if args.cohort not in cohorts:
            raise ValueError(f"cohort {args.cohort!r} is not in COHORTS {sorted(cohorts)}")

        meta_name, default_dest = COHORT_META[args.cohort]
        meta = json.loads((ROOT / "kaggle" / meta_name).read_text())
        mounted = check_metadata(meta, cohorts, args.cohort)

        rel = KERNEL_SRC.relative_to(ROOT)
        dirty = git("status", "--porcelain", "--", str(rel))
        if dirty:
            raise ValueError(
                f"{rel} has uncommitted changes:\n{dirty}\n"
                "commit them first -- the ledger row's commit field must name the code that ran"
            )
        commit = git("rev-parse", "HEAD")
        patched = patch_source(source, args.cohort, commit)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    dest = args.dest or default_dest
    dest.mkdir(parents=True, exist_ok=True)
    (dest / meta["code_file"]).write_text(patched)
    shutil.copy2(ROOT / "kaggle" / meta_name, dest / "kernel-metadata.json")

    staged = parse_cohorts((dest / meta["code_file"]).read_text())
    assert staged["COHORT"] == args.cohort, staged["COHORT"]
    print(f"staged {args.cohort} @ {commit[:12]} -> {dest}")
    print(f"  id: {meta['id']}")
    print(f"  ledger: {cohorts[args.cohort]['ledger']}")
    print(f"  arms: {', '.join(label for label, _, _ in cohorts[args.cohort]['arms'])}")
    print(f"  kernels: {', '.join(sorted(mounted))}")
    print(f"\n  kaggle kernels push -p {dest}")


if __name__ == "__main__":
    main()
