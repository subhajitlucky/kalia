"""Static-check every notebook's code cells before any GPU or quota is spent on it.

There is a specific, recent failure this exists to prevent. A helper of mine ran

    r = subprocess.run([sys.executable, "-m", "pyflakes", path], ...)
    print(r.stdout.strip() or "pyflakes clean")

with pyflakes **not installed**. The module error goes to stderr; stdout is empty;
and `empty or "clean"` prints "pyflakes clean". So several notebooks were
announced as lint-clean by a check that had never run -- including ones that then
died on Kaggle with `NameError: name 'subprocess' is not defined`, which is exactly
the class of error pyflakes exists to catch. The gate was decorative and it
reported success.

So this script:

- **fails if the checker itself is missing**, rather than treating absence of
  output as absence of findings
- **keeps the `!pip` cell's imports** by stripping only the magic lines, because
  in these notebooks `!pip install` shares a cell with the imports and dropping
  the whole cell produces fifteen phantom "undefined name" findings
- distinguishes **undefined names** (fatal) from style notes (not)

The only reliable check of a notebook is still running it. This is the cheap
one that catches the mistakes a run costs an hour to find.

Usage:
    python tools/check_notebooks.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"


def cell_source(cell: dict) -> str:
    """Code cells, minus IPython magic but keeping the rest of the cell.

    Both `!` shell escapes and `%` line magics are stripped. `%` is Python's
    modulo operator, so only *leading* `%` is treated as magic -- otherwise this
    would corrupt real arithmetic. `kalia-reproduce.ipynb` needed the `%` case:
    it opens with `%cd kalia`, which pyflakes reported as an invalid syntax error
    and which is not a defect in the notebook.
    """
    if cell.get("cell_type") != "code":
        return ""
    kept = [
        line
        for line in "".join(cell.get("source", [])).splitlines()
        if not line.lstrip().startswith(("!", "%"))
    ]
    return "\n".join(kept)


def main() -> int:
    # The whole point: prove the checker exists before trusting its silence.
    probe = subprocess.run(
        [sys.executable, "-c", "import pyflakes"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        print("pyflakes is not installed -- refusing to report a clean result.")
        print("  pip install pyflakes")
        return 2

    failures: list[str] = []
    checked = 0
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        nb = json.loads(path.read_text())
        # Kaggle's nbformat validator now hard-errors on missing cell ids, and
        # the rejection surfaces as a blank pre-flight ERROR with zero logs --
        # five runs were lost to it on 2026-10-07 before the cause was found.
        # Every notebook must carry ids.
        missing_ids = [i for i, c in enumerate(nb["cells"]) if not c.get("id")]
        if missing_ids:
            failures.append(path.name)
            print(f"  {path.name:44s} MISSING CELL IDS at cells {missing_ids[:6]}")
            continue
        src = "\n\n".join(s for s in (cell_source(c) for c in nb["cells"]) if s.strip())
        if not src:
            continue
        checked += 1
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(src)
            tmp = fh.name
        res = subprocess.run(
            [sys.executable, "-m", "pyflakes", tmp], capture_output=True, text=True
        )
        if res.returncode != 0 and not res.stdout.strip():
            print(f"  {path.name:44s} CHECKER FAILED: {res.stderr.strip()[:80]}")
            failures.append(path.name)
            continue
        undefined = [l for l in res.stdout.splitlines() if "undefined name" in l]
        notes = [l for l in res.stdout.splitlines() if "undefined name" not in l]
        if undefined:
            failures.append(path.name)
            print(f"  {path.name:44s} {len(undefined)} UNDEFINED")
            for line in undefined[:5]:
                print(f"      {line.split(': ', 1)[-1]}")
        else:
            suffix = f" ({len(notes)} style note)" if notes else ""
            print(f"  {path.name:44s} clean{suffix}")

    print(f"\n{checked} notebooks checked")
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("no undefined names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
