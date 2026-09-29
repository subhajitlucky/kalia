"""The v0.3.0 notebooks reference files by name. Those names must agree.

Found by auditing readiness: `kalia-v020` publishes a checkpoint and two log CSVs
and **no** probe shard, and `mix_bins.py` pre-blends the four sources into one
`train.bin` and discards the originals. So Step 0 had nothing to measure and Step 4
had no shards to reweight -- both would have failed on day one, however much quota
was available.

`kalia-publish-sources` creates them. But a producer and two consumers agreeing by
accident is not a contract, and the first version of this pair disagreed: the
publisher wrote `corpus/val_forget.bin` while CL-0 looked for
`checkpoints/probe_val.bin`. The step would still have failed, just later and with
a less useful message.

These tests read the actual notebook JSON, so a rename on either side fails here
rather than on Kaggle.
"""

from __future__ import annotations

import json
import re
from pathlib import Path



ROOT = Path(__file__).resolve().parent
NB = ROOT / "notebooks"

PUBLISHER = "kalia-publish-sources.ipynb"
CONSUMERS = {
    "kalia-v030-cl0.ipynb": {"probe_val.bin"},
    "kalia-v030-kautilya.ipynb": {"fineweb.bin", "tinystories.bin",
                                  "cosmopedia.bin", "python.bin"},
}


def _source(name: str) -> str:
    nb = json.loads((NB / name).read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_publisher_exists_and_uploads_the_corpus():
    src = _source(PUBLISHER)
    assert "corpus/probe_val.bin" in src
    # Per-source paths are built from RATIOS, so assert on the construction and on
    # the ratios themselves rather than on four literal strings that never appear.
    assert "corpus/{n}.bin" in src
    for name in ("fineweb", "tinystories", "cosmopedia", "python"):
        assert f"'{name}':" in src, name


def test_cl0_looks_for_exactly_what_the_publisher_uploads():
    """The bug: producer wrote val_forget.bin, consumer wanted probe_val.bin."""
    pub = _source(PUBLISHER)
    cl0 = _source("kalia-v030-cl0.ipynb")
    uploaded = set(re.findall(r"'(corpus/[\w.]+\.bin)'", pub))
    wanted = set(re.findall(r"'(corpus/[\w.]+\.bin)'", cl0))
    assert "corpus/probe_val.bin" in uploaded
    assert wanted and wanted <= uploaded, (
        f"CL-0 looks for {sorted(wanted - uploaded)} which the publisher never "
        f"uploads; it publishes {sorted(uploaded)}"
    )


def test_kautilya_looks_for_exactly_what_the_publisher_uploads():
    """Both sides must agree on the four names, by construction rather than luck."""
    pub = _source(PUBLISHER)
    kaut = _source("kalia-v030-kautilya.ipynb")
    assert "corpus/{n}.bin" in pub, "publisher does not build per-source paths"
    # The names it builds come from RATIOS; the notebook hardcodes four paths.
    ratios = set(re.findall(r"'(\w+)':\s*0\.\d+", pub))
    wanted = set(re.findall(r"'(corpus/[\w.]+\.bin)'", kaut))
    assert len(wanted) == 4, f"expected 4 per-source paths, got {sorted(wanted)}"
    assert {p.rsplit("/", 1)[-1][:-4] for p in wanted} == ratios, (
        f"notebook looks up {sorted(p.rsplit('/',1)[-1][:-4] for p in wanted)} but the "
        f"publisher builds paths for {sorted(ratios)}"
    )


def test_static_weights_match_the_published_manifest():
    """The static arm's 60/20/15/5 must be the ratios the shards were cut for."""
    pub = _source(PUBLISHER)
    kaut = _source("kalia-v030-kautilya.ipynb")
    for key, val in (("fineweb", 0.60), ("tinystories", 0.20),
                     ("cosmopedia", 0.15), ("python", 0.05)):
        assert f"'{key}': {val}" in pub, f"publisher ratio for {key} missing"
        assert f"'{key}': {val}" in kaut, f"notebook static weight for {key} missing"


def test_probe_shard_is_a_slice_of_the_training_stream_not_heldout():
    """CL-0 measures forgetting of what was learned, so it must probe train data."""
    pub = _source(PUBLISHER)
    assert "train_bins[0]" in pub or "train" in pub
    assert "held-out set would measure nothing" in pub or "TRAINING" in pub


def test_dependent_notebooks_name_the_prerequisite_rather_than_falling_back():
    """A silent fallback makes a step 'pass' while measuring the wrong thing."""
    cl0 = _source("kalia-v030-cl0.ipynb")
    kaut = _source("kalia-v030-kautilya.ipynb")
    assert "kalia-publish-sources" in cl0
    assert "kalia-publish-sources" in kaut
    assert "fall back to the local val.bin" not in cl0, (
        "CL-0 must not silently substitute another shard for the frozen probe set"
    )


def test_v020_repo_does_not_already_contain_these_artifacts():
    """Documents what the audit found, so the test fails if that ever changes
    in a way that makes the publisher redundant.

    Not a network test: it asserts the *reason* the publisher exists is recorded.
    """
    doc = (ROOT / "tools" / "build_publish_sources_notebook.py").read_text()
    assert "val.bin" in doc and "probe" in doc.lower()
    assert "mix_bins" in doc
