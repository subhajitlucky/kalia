"""Serve `EleutherAI/lambada_openai` to lm-eval without its loading script.

The repo still ships a `lambada_openai.py` loading script, and modern `datasets`
refuses to execute dataset scripts, so lm-eval's task definition fails to load
and LAMBADA is silently absent from the evaluation. This is the same failure that
made `deepmind/pg19` unusable in the long-form probe, and it has now reached the
registered evaluation itself.

The data is not missing. The repo carries one parquet file per config, named by
directory:

    de/test/de.parquet   default/test/default.parquet   en/test/en.parquet
    es/test/es.parquet   fr/test/fr.parquet              it/test/it.parquet

which is why naive fixes go wrong in two opposite directions:

- loading all six concatenates six copies of the task (5,153 x 6 = 30,918 rows)
- filtering on ``"/en/" in path`` matches **both** ``en/`` and ``default/``,
  because ``default/test/default.parquet`` contains neither but
  ``en/test/en.parquet`` does -- and a filter written for one silently drops or
  doubles the set

So the config is never guessed. lm-eval 0.4.9's own task definition
(``lm_eval/tasks/lambada/lambada_openai.yaml``) requests::

    dataset_path: EleutherAI/lambada_openai
    dataset_name: default
    test_split: test

and this module serves exactly that file, while passing every other dataset
request straight through. **lm-eval still does the scoring** -- reimplementing
LAMBADA would produce a number that is not comparable with v0.1.2's published
23.0, which would make S-A a comparison of two different measurements.

The standard task is 5,153 examples, which :func:`load_lambada` asserts, because a
wrong slice that still returns *some* rows is the failure mode that produces a
plausible wrong number.
"""

from __future__ import annotations

from typing import Any, Callable

REPO = "EleutherAI/lambada_openai"
CONFIG = "default"
SPLIT = "test"
EXPECTED_ROWS = 5153


def config_parquets(files: list[str]) -> dict[str, str]:
    """Map config name -> parquet path, from a repo file listing.

    Keys are the first path segment, which is how the repo lays the configs out
    (``en/test/en.parquet`` -> ``en``).
    """
    return {f.split("/")[0]: f for f in files if f.endswith(".parquet")}


def make_loader(
    parquet: dict[str, str],
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap ``datasets.load_dataset`` so only this repo is redirected.

    ``original`` is injected rather than imported so the dispatch can be tested
    without ``datasets`` installed, and so the wrapper cannot recurse into
    itself.
    """

    def patched(path: Any, *args: Any, **kwargs: Any) -> Any:
        if not (isinstance(path, str) and "lambada_openai" in path):
            return original(path, *args, **kwargs)

        if CONFIG not in parquet:
            raise KeyError(
                f"lm-eval requests config {CONFIG!r} but the repo exposes "
                f"{sorted(parquet)}"
            )
        rel = parquet[CONFIG]
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel}"

        # lm-eval passes the config name positionally or by keyword, depending on
        # the call site; both must resolve to the same file.
        requested = kwargs.get("name") or (args[0] if args else CONFIG)
        if requested != CONFIG:
            raise KeyError(
                f"lm-eval asked for config {requested!r}, expected {CONFIG!r}; "
                "serving a different task would not be comparable to v0.1.2"
            )

        kwargs.pop("name", None)
        kwargs.pop("trust_remote_code", None)
        # A bare `data_files=<url>` load exposes exactly one split, and the
        # parquet reader names it "train" -- so asking for "test" fails with
        # `Unknown split "test". Should be one of ['train']`. Passing the file
        # under a "test" key names the split we actually want.
        kwargs["data_files"] = {"test": url}
        kwargs["split"] = SPLIT
        return original("parquet", **kwargs)

    return patched


def load_lambada(repo_files: list[str]) -> Any:
    """Install the patch and return the LAMBADA dataset, asserting its size."""
    import datasets
    from huggingface_hub import HfApi

    parquet = config_parquets(repo_files)
    if CONFIG not in parquet:
        raise KeyError(f"no {CONFIG!r} config parquet in {sorted(parquet)}")

    datasets.load_dataset = make_loader(parquet, datasets.load_dataset)
    ds = datasets.load_dataset(REPO)
    if len(ds) != EXPECTED_ROWS:
        raise ValueError(
            f"LAMBADA has {len(ds)} rows, expected {EXPECTED_ROWS}. "
            "The wrong config or split is being served; do not report this number."
        )
    return ds
