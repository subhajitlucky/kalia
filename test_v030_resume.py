"""Update arms must resume from the v0.2.0 checkpoint.

Found by auditing readiness: the Step 2-4 notebooks invoked `ablate.py` and
`train.py` with no `--resume`, so every continual-update arm would have trained
from random initialisation instead of the v0.2.0 checkpoint — the same failure
class as the re-warm defect (a mechanism never switched on, reporting a
plausible null). The fresh-control notebook (baseline) must NOT resume.

These tests read the actual notebook JSON, so a dropped flag fails here
rather than on Kaggle after the quota is spent.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT / "notebooks"

UPDATE_NOTEBOOKS = (
    "kalia-v030-rewarm.ipynb",
    "kalia-v030-replay.ipynb",
    "kalia-v030-kautilya.ipynb",
)

FRESH_NOTEBOOKS = ("kalia-v030-baseline.ipynb",)

CHECKPOINT_REPO = "kalia-lm/kalia-v020"


def _code(notebook: str) -> str:
    cells = json.loads((NB / notebook).read_text())["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def test_update_notebooks_resume_from_the_release_checkpoint():
    for notebook in UPDATE_NOTEBOOKS:
        code = _code(notebook)
        assert "--resume" in code, f"{notebook} launches training without --resume"
        assert CHECKPOINT_REPO in code, f"{notebook} does not name {CHECKPOINT_REPO}"


def test_fresh_control_notebook_does_not_resume():
    for notebook in FRESH_NOTEBOOKS:
        code = _code(notebook)
        assert "--resume" not in code, f"{notebook} must train from scratch, not resume"


def test_ablate_forwards_resume_to_train():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "ablate.py", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--resume-from" in proc.stdout


def test_train_accepts_resume_from():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--resume-from" in proc.stdout
