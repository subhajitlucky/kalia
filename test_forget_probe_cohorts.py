"""The retention probe scores a declared cohort, and staging enforces it.

The probe used to infer its arm set from the mount-path text, which could not
see the no-replay arm (its checkpoint lives at ``out/kautilya-static/``), so
attaching it made the baseline assertion count two and the second-seed
replication could not be scored at all. These tests pin the two things that
replace the heuristic: the cohort tables in the kernel, and the agreement
between a cohort and the kernel_sources of the metadata that runs it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


def _load(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stager = _load("stage_x29_forget_probe")
KERNEL_SRC = ROOT / "tools" / "kalia_forget_probe_kernel.py"


@pytest.fixture(scope="module")
def cohorts() -> dict:
    return stager.parse_cohorts(KERNEL_SRC.read_text())["COHORTS"]


def _metadata_for(cohort: str) -> dict:
    name, _ = stager.COHORT_META[cohort]
    return json.loads((ROOT / "kaggle" / name).read_text())


def test_the_1401_cohort_is_still_the_registered_ladder(cohorts):
    """The refactor must not have changed what the archived ledger was read on:
    four patha arms, one checkpoint each, in the order they were scored."""
    assert cohorts["x29-1401"]["ledger"] == "forgetting_x29.jsonl"
    assert [(label, slug) for label, slug, _ in cohorts["x29-1401"]["arms"]] == [
        ("uniform", "kalia-x29-uniform"),
        ("krama", "kalia-x29-krama"),
        ("jata", "kalia-x29-jata"),
        ("ghana", "kalia-x29-ghana"),
    ]


def test_replication_cohort_is_a_2x2_of_replay_by_seed(cohorts):
    """replay (uniform) vs no-replay (static) at both seeds. The no-replay side
    is what the old path heuristic could not name."""
    arms = cohorts["x29-replication"]["arms"]
    assert [label for label, _, _ in arms] == [
        "uniform-s1401", "static-s1401", "uniform-s1402", "static-s1402",
    ]
    by_label = {label: (slug, path) for label, slug, path in arms}
    # the no-replay arms are the Kautilya trio runner with adaptive_every: 0,
    # and they write to a directory the "/x29-" test never matched
    for label in ("static-s1401", "static-s1402"):
        slug, path = by_label[label]
        assert slug.startswith("kalia-x28-static")
        assert path == "out/kautilya-static/ckpt.pt"
    # the replay arms are the X29 runner, and the seed is in the kernel slug
    for label, seed in (("uniform-s1401", 1401), ("uniform-s1402", 1402)):
        slug, path = by_label[label]
        assert slug == f"kalia-x29-uniform{'' if seed == 1401 else '-s1402'}"
        assert path == "out/x29-uniform/ckpt.pt"


@pytest.mark.parametrize("cohort", sorted(stager.COHORT_META))
def test_metadata_mounts_exactly_the_declared_kernels(cohorts, cohort):
    assert stager.check_metadata(_metadata_for(cohort), cohorts, cohort)


@pytest.mark.parametrize("cohort", sorted(stager.COHORT_META))
def test_every_declared_kernel_is_a_committed_kernel(cohorts, cohort):
    """A typo in a slug would otherwise only surface as a failed GPU-free run
    on Kaggle, after a round trip."""
    committed = set()
    for path in (ROOT / "kaggle").glob("kernel-metadata*.json"):
        committed.add(json.loads(path.read_text())["id"].split("/")[-1])
    unknown = stager.declared_slugs(cohorts, cohort) - committed
    assert not unknown, f"{cohort} names kernels with no metadata file: {sorted(unknown)}"


@pytest.mark.parametrize("cohort", sorted(stager.COHORT_META))
def test_labels_are_unique_and_paths_are_relative(cohorts, cohort):
    arms = cohorts[cohort]["arms"]
    labels = [label for label, _, _ in arms]
    assert len(labels) == len(set(labels))
    for label, slug, path in arms:
        assert not path.startswith("/"), f"{label}: checkpoint path must be relative to the kernel output"
        assert path.endswith("ckpt.pt"), label
        assert "/" not in slug, label


def test_metadata_disagreeing_with_its_cohort_is_refused(cohorts):
    meta = _metadata_for("x29-replication")
    meta["kernel_sources"] = meta["kernel_sources"][:-1]  # drop one mounted kernel
    with pytest.raises(ValueError, match="declares"):
        stager.check_metadata(meta, cohorts, "x29-replication")


def test_metadata_without_the_baseline_dataset_is_refused(cohorts):
    meta = _metadata_for("x29-replication")
    meta["dataset_sources"] = ["subhajitlucky/kalia-code-dev"]
    with pytest.raises(ValueError, match="baseline row"):
        stager.check_metadata(meta, cohorts, "x29-replication")


def test_patch_source_pins_cohort_and_commit():
    src = KERNEL_SRC.read_text()
    out = stager.patch_source(src, "x29-replication", "a" * 40)
    parsed = stager.parse_cohorts(out)
    assert parsed["COHORT"] == "x29-replication"
    assert stager.parse_cohorts(out)["COHORTS"] == stager.parse_cohorts(src)["COHORTS"]


def test_patch_source_refuses_a_source_it_cannot_pin():
    with pytest.raises(ValueError, match="expected one COHORT and one COMMIT"):
        stager.patch_source("COHORT = 'x29-1401'\n", "x29-replication", "b" * 40)


def test_unknown_cohort_is_rejected(tmp_path):
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "tools/stage_x29_forget_probe.py", "--cohort", "nope"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert r.returncode != 0
    assert "nope" in r.stderr


# --- the resolver, against a fake Kaggle mount -------------------------------
# The kernel is a script, so importing it would start a run. Only the resolver
# and the cohort table are compiled out of it, with the mount points pointed at
# a temporary tree -- the point being that "which checkpoints get scored" is
# decided by declared data and is therefore testable.

def _resolver():
    import ast
    import glob as glob_mod

    # resolve_cohort's defaults are bound at def time, so the mount constants
    # have to come along with it.
    needed = {"COHORTS", "KERNEL_MOUNT", "BASELINE_GLOB", "ATTACHED_GLOB"}
    tree = ast.parse(KERNEL_SRC.read_text())
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_cohort":
            wanted.append(node)
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) in needed for t in node.targets
        ):
            wanted.append(node)
    got = {getattr(t, "id", None) for n in wanted if isinstance(n, ast.Assign) for t in n.targets}
    assert len(wanted) == len(needed) + 1 and got == needed, (
        f"resolve_cohort or its constants moved; extracted {sorted(got)}"
    )
    ns = {"glob": glob_mod, "Path": Path}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "<kernel>", "exec"), ns)
    return ns


def _mount(root: Path, owner: str, slug: str, rel: str) -> Path:
    p = root / owner / slug / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"ckpt")
    return p


def _fake_tree(tmp_path: Path) -> dict:
    """A mount tree holding exactly the X29/x28 kernels a real run attaches."""
    _mount(tmp_path, "subhajitlucky", "kalia-x29-uniform", "out/x29-uniform/ckpt.pt")
    _mount(tmp_path, "subhajitlucky", "kalia-x28-static", "out/kautilya-static/ckpt.pt")
    _mount(tmp_path, "subhajitlucky", "kalia-x29-uniform-s1402", "out/x29-uniform/ckpt.pt")
    _mount(tmp_path, "subhajitlucky", "kalia-x28-static-s1402", "out/kautilya-static/ckpt.pt")
    return {
        "mount": str(tmp_path),
        "baseline_glob": str(tmp_path / "datasets" / "*" / "kalia-v020-ckpt" / "ckpt.pt"),
        "attached_glob": str(tmp_path / "**" / "ckpt.pt"),
    }


def test_resolver_names_the_no_replay_arms(tmp_path):
    """The exact failure that motivated the change: before, a static arm's
    checkpoint (out/kautilya-static/) was read as a second baseline and the run
    refused to start."""
    ns = _resolver()
    tree = _fake_tree(tmp_path)
    (tmp_path / "datasets" / "subhajitlucky" / "kalia-v020-ckpt").mkdir(parents=True)
    (tmp_path / "datasets" / "subhajitlucky" / "kalia-v020-ckpt" / "ckpt.pt").write_bytes(b"base")

    baseline, arms = ns["resolve_cohort"]("x29-replication", **tree)
    assert baseline.name == "ckpt.pt" and baseline.parent.name == "kalia-v020-ckpt"
    assert [label for label, _ in arms] == [
        "uniform-s1401", "static-s1401", "uniform-s1402", "static-s1402",
    ]
    assert "kautilya-static" in str(arms[1][1])
    assert "s1402" in str(arms[3][1])


def test_resolver_refuses_a_missing_arm(tmp_path):
    ns = _resolver()
    tree = _fake_tree(tmp_path)
    (tmp_path / "datasets" / "x" / "kalia-v020-ckpt").mkdir(parents=True)
    (tmp_path / "datasets" / "x" / "kalia-v020-ckpt" / "ckpt.pt").write_bytes(b"base")
    # the replication needs uniform-s1402; detach it
    for p in tmp_path.glob("*/kalia-x29-uniform-s1402/**/ckpt.pt"):
        p.unlink()
    with pytest.raises(AssertionError, match="uniform-s1402"):
        ns["resolve_cohort"]("x29-replication", **tree)


def test_resolver_refuses_an_undeclared_checkpoint(tmp_path):
    """A stale kernel_sources entry must not join the ledger silently."""
    ns = _resolver()
    tree = _fake_tree(tmp_path)
    (tmp_path / "datasets" / "x" / "kalia-v020-ckpt").mkdir(parents=True)
    (tmp_path / "datasets" / "x" / "kalia-v020-ckpt" / "ckpt.pt").write_bytes(b"base")
    _mount(tmp_path, "subhajitlucky", "kalia-x29-jata", "out/x29-jata/ckpt.pt")
    with pytest.raises(AssertionError, match="does not declare"):
        ns["resolve_cohort"]("x29-replication", **tree)


def test_resolver_refuses_two_baselines(tmp_path):
    ns = _resolver()
    tree = _fake_tree(tmp_path)
    for name in ("a", "b"):
        d = tmp_path / "datasets" / name / "kalia-v020-ckpt"
        d.mkdir(parents=True)
        (d / "ckpt.pt").write_bytes(b"base")
    with pytest.raises(AssertionError, match="exactly one v0.2.0 baseline"):
        ns["resolve_cohort"]("x29-replication", **tree)
