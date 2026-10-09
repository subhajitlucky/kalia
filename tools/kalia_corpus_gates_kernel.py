"""Kaggle CPU kernel: benchmark-contamination gates over the training corpus.

Downloads the evaluation splits of the five scored benchmarks (PIQA, HellaSwag,
WinoGrande, ARC-Easy, LAMBADA), writes them as a JSONL, and scans ``train.bin``
and ``val.bin`` for exact 13-gram matches with ``corpus_gates.py``.

Internet is required (dataset downloads); no secrets, no GPU. Failures to fetch
an individual benchmark are recorded in the output rather than silently
reducing coverage.
"""

import glob
import json
import os
import shutil
import subprocess
import sys

work = "/kaggle/working/kalia"
if not os.path.exists(work):
    hits = sorted(glob.glob("/kaggle/input/**/corpus_gates.py", recursive=True))
    preferred = [p for p in hits if "/kalia-code-dev/" in p]
    assert preferred, (
        "kalia-code-dev not attached or lacks corpus_gates.py; other matches "
        "(do NOT trust them, they are stale bundles): " + str(hits)
    )
    shutil.copytree(os.path.dirname(preferred[0]), work)
os.chdir(work)
# Script-kernel imports resolve from this directory, not from the cwd, so the
# repo copy must be on sys.path before `from lambada_loader import ...`.
sys.path.insert(0, work)
for req in ("corpus_gates.py", "lambada_loader.py"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"


def ensure(pkg: str, spec: str | None = None) -> None:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", spec or pkg], check=True
        )


ensure("tiktoken")
ensure("datasets")

from datasets import load_dataset  # noqa: E402

docs: list[dict] = []
failures: dict[str, str] = {}


def add_rows(task: str, rows, fields: list[str]) -> None:
    for i, row in enumerate(rows):
        parts: list[str] = []
        for name in fields:
            value = row.get(name)
            if isinstance(value, list):
                parts.extend(str(v) for v in value)
            elif value is not None:
                parts.append(str(value))
        docs.append({"task": task, "index": i, "text": "\n".join(parts)})


try:
    # The piqa repo still ships a loading script and modern `datasets` refuses
    # scripts, so fetch the script's own data URLs directly. The scored split is
    # `validation` (dev); train and test are included too because a 13-gram is a
    # 13-gram regardless of which split it came from.
    import io
    import urllib.request
    import zipfile

    with urllib.request.urlopen(
        "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
        timeout=180,
    ) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in ("physicaliqa-train-dev/train.jsonl", "physicaliqa-train-dev/dev.jsonl"):
            with zf.open(name) as fh:
                for i, line in enumerate(fh):
                    row = json.loads(line)
                    docs.append(
                        {"task": "piqa", "index": i,
                         "text": "\n".join([row["goal"], row["sol1"], row["sol2"]])}
                    )
    with urllib.request.urlopen(
        "https://yonatanbisk.com/piqa/data/tests.jsonl", timeout=180
    ) as resp:
        for i, line in enumerate(resp.read().decode("utf-8").splitlines()):
            row = json.loads(line)
            docs.append(
                {"task": "piqa", "index": i,
                 "text": "\n".join([row["goal"], row["sol1"], row["sol2"]])}
            )
except Exception as exc:  # noqa: BLE001 - recorded, not raised
    failures["piqa"] = repr(exc)

try:
    add_rows("hellaswag",
             load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True),
             ["ctx", "endings"])
except Exception as exc:
    failures["hellaswag"] = repr(exc)

try:
    add_rows("winogrande",
             load_dataset("allenai/winogrande", "winogrande_xl", split="validation",
                          trust_remote_code=True),
             ["sentence", "option1", "option2"])
except Exception as exc:
    failures["winogrande"] = repr(exc)

try:
    # row["choices"] is a dict {"label": [...], "text": [...]}; the first version
    # handed the dict to add_rows, which str()'d it -- so the patterns were
    # Python reprs like "['A', 'B', 'C', 'D', '" and matched code/quizzes in the
    # corpus 601 times. Extract the choice texts explicitly.
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", trust_remote_code=True)
    for i, row in enumerate(arc):
        docs.append(
            {"task": "arc_easy", "index": i,
             "text": "\n".join([row["question"], *row["choices"]["text"]])}
        )
except Exception as exc:
    failures["arc_easy"] = repr(exc)

try:
    from huggingface_hub import HfApi

    from lambada_loader import REPO, load_lambada

    lambada = load_lambada(HfApi().list_repo_files(REPO, repo_type="dataset"))
    for i, row in enumerate(lambada):
        text = row.get("text") or next(v for v in row.values() if isinstance(v, str))
        docs.append({"task": "lambada", "index": i, "text": text})
except Exception as exc:
    failures["lambada"] = repr(exc)

print("benchmark documents per task:")
counts: dict[str, int] = {}
for doc in docs:
    counts[doc["task"]] = counts.get(doc["task"], 0) + 1
for task in sorted(counts):
    print(f"  {task}: {counts[task]}")
if failures:
    print("FAILED to fetch:", json.dumps(failures, indent=2)[:2000])
json.dump(failures, open("/kaggle/working/benchmark_fetch_failures.json", "w"), indent=2)
assert docs, "no benchmark texts fetched at all; refusing to run an empty gate"

bench_path = "/kaggle/working/benchmark_texts.jsonl"
with open(bench_path, "w") as fh:
    for doc in docs:
        fh.write(json.dumps(doc) + "\n")

shards: list[str] = []
for pattern in ("/kaggle/input/**/train.bin", "/kaggle/input/**/val.bin"):
    hits = sorted(glob.glob(pattern, recursive=True))
    if hits:
        shards.append(hits[0])
assert shards, "no shards found; attach kalia-prep-v2b as a kernel source"
print("shards:", shards)

cmd = [
    sys.executable,
    "corpus_gates.py",
    "--benchmark-jsonl", bench_path,
    "--out", "/kaggle/working/corpus_gates.json",
]
for shard in shards:
    cmd += ["--shard", shard]
proc = subprocess.run(cmd, text=True)
sys.exit(proc.returncode)
