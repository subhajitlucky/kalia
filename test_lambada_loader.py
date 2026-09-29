"""Tests for the LAMBADA loader patch.

Two versions of this kernel failed before the tests existed, in opposite
directions, and both would have produced a number that looked fine:

- **v1** globbed all six parquet files and served 30,918 rows -- six concatenated
  copies of the task
- **v2** filtered on `"/en/" in path`, which the author believed selected one
  file and which actually matched two, so the assertion `len(urls) == 1` fired

The second is the instructive one. The filter was a guess about which file lm-eval
wants, written without reading lm-eval's task definition. A guess that happens to
select the wrong *number* of files fails loudly; a guess that happens to select
the right number of the wrong files fails silently and scores a different task.

So the config is pinned to what lm-eval 0.4.9 actually requests (`default`), and
these tests hold it there.
"""

from __future__ import annotations

import pytest

from lambada_loader import (
    CONFIG,
    EXPECTED_ROWS,
    REPO,
    config_parquets,
    make_loader,
)

# The real listing from EleutherAI/lambada_openai, six configs, one file each.
REPO_FILES = [
    ".gitattributes",
    "README.md",
    "de/test/de.parquet",
    "default/test/default.parquet",
    "en/test/en.parquet",
    "es/test/es.parquet",
    "fr/test/fr.parquet",
    "it/test/it.parquet",
]


class _Recorder:
    """Stand-in for datasets.load_dataset that records instead of loading."""

    def __init__(self, n_rows: int = EXPECTED_ROWS):
        self.calls: list[tuple] = []
        self.n_rows = n_rows

    def __call__(self, path, *args, **kwargs):
        self.calls.append((path, args, kwargs))
        return list(range(self.n_rows))


def test_config_parquets_keys_on_the_directory_not_the_filename():
    """`en/test/en.parquet` and `default/test/default.parquet` must not collide."""
    got = config_parquets(REPO_FILES)
    assert set(got) == {"de", "default", "en", "es", "fr", "it"}
    assert got["default"] == "default/test/default.parquet"
    assert got["en"] == "en/test/en.parquet"


def test_serves_exactly_one_file_not_six():
    """The v1 failure: every parquet concatenated into 30,918 rows."""
    rec = _Recorder()
    make_loader(config_parquets(REPO_FILES), rec)(REPO, name=CONFIG, split="test")
    path, _args, kwargs = rec.calls[0]
    assert path == "parquet", "must load through the parquet reader"
    files = kwargs["data_files"]
    assert isinstance(files, dict), "a single file keyed by split, not a list of six"
    assert list(files) == ["test"], f"only the test split, got {list(files)}"
    assert files["test"].endswith("default/test/default.parquet")
    assert kwargs["split"] == "test", "the split must exist in the loaded dataset"


def test_split_is_named_because_a_bare_data_files_load_calls_it_train():
    """Version 3's failure: `Unknown split "test". Should be one of ['train']`.

    Passing `data_files=<url>` gives one split named `train`, so asking for
    `test` fails at read time. The file must be keyed by its split name.
    """
    rec = _Recorder()
    make_loader(config_parquets(REPO_FILES), rec)(REPO, name=CONFIG)
    files = rec.calls[0][2]["data_files"]
    assert list(files) == ["test"], "keying the file by split name is what makes it loadable"
    assert rec.calls[0][2]["split"] == "test"


def test_serves_default_regardless_of_how_the_config_is_passed():
    """lm-eval passes the name positionally or by keyword; both must agree."""
    for args, kwargs in (
        ((), {"name": "default", "split": "test"}),
        (("default",), {"split": "test"}),
        ((), {"split": "test"}),
    ):
        rec = _Recorder()
        make_loader(config_parquets(REPO_FILES), rec)(REPO, *args, **dict(kwargs))
        assert rec.calls[0][2]["data_files"]["test"].endswith("default/test/default.parquet")


def test_refuses_a_different_config_rather_than_silently_serving_the_wrong_task():
    """A guess that picks the right *count* of wrong files must still fail."""
    rec = _Recorder()
    with pytest.raises(KeyError, match="not be comparable"):
        make_loader(config_parquets(REPO_FILES), rec)(REPO, name="de", split="test")
    assert not rec.calls, "nothing may be loaded when the config is wrong"


def test_other_datasets_pass_straight_through():
    """The patch must not touch any other task's measurement."""
    rec = _Recorder()
    make_loader(config_parquets(REPO_FILES), rec)("EleutherAI/hellaswag", name="default")
    assert rec.calls[0][0] == "EleutherAI/hellaswag"
    assert "data_files" not in rec.calls[0][2]


def test_drops_trust_remote_code_and_forces_the_test_split():
    rec = _Recorder()
    make_loader(config_parquets(REPO_FILES), rec)(
        REPO, name=CONFIG, split="train", trust_remote_code=True
    )
    kwargs = rec.calls[0][2]
    assert "trust_remote_code" not in kwargs
    assert kwargs["split"] == "test", "the parquet file holds only the test split"


def test_raises_rather_than_serving_when_the_config_is_absent():
    rec = _Recorder()
    with pytest.raises(KeyError, match="requests config"):
        make_loader({"en": "en/test/en.parquet"}, rec)(REPO, name=CONFIG)
    assert not rec.calls


def test_row_count_guard_is_enforced():
    """The v1 guard, kept as a guard: a wrong slice must not report a number."""
    from lambada_loader import load_lambada

    class FakeDatasets:
        def __init__(self):
            self.load_dataset = _Recorder(n_rows=30_918)

    import sys
    import types

    fake = types.ModuleType("datasets")
    fake.load_dataset = _Recorder(n_rows=30_918)
    fake.__dict__["load_dataset"] = fake.load_dataset
    saved = sys.modules.get("datasets")
    sys.modules["datasets"] = fake
    hub = types.ModuleType("huggingface_hub")

    class FakeApi:
        def list_repo_files(self, *a, **k):
            return REPO_FILES

    hub.HfApi = FakeApi
    saved_hub = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = hub
    try:
        with pytest.raises(ValueError, match="do not report this number"):
            load_lambada(REPO_FILES)
    finally:
        if saved is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = saved
        if saved_hub is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved_hub
