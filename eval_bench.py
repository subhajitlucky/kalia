"""Zero-shot benchmark run for KALIA checkpoints via lm-evaluation-harness.

Runs a small, honest suite on CPU: ARC-Easy, PIQA, HellaSwag, LAMBADA,
WinoGrande (0-shot, 500 samples each). A 58M story model is expected to score
near chance on knowledge tasks; the numbers are published as-is.
"""

import argparse
import glob
import json
from pathlib import Path

import lm_eval

from kalia_lm import KaliaLM


def find_ckpt() -> str:
    hits = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
    preferred = [h for h in hits if "kalia-train-v012" in h]
    assert preferred or hits, "no ckpt.pt found - attach the training kernel output"
    return (preferred or hits)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--tasks", type=str, default="arc_easy,piqa,hellaswag,lambada_openai,winogrande")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/bench_results.json"))
    args = parser.parse_args()

    ckpt = args.ckpt or find_ckpt()
    print("checkpoint:", ckpt)
    model = KaliaLM(ckpt=ckpt, device="cpu")
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
        limit=args.limit,
        bootstrap_iters=100,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results["results"], indent=2))

    print("=" * 64)
    print("BENCHMARK SUMMARY (0-shot)")
    for task, metrics in results["results"].items():
        shown = {k: v for k, v in metrics.items() if k.endswith(",none") and isinstance(v, (int, float))}
        print(task, json.dumps(shown))


if __name__ == "__main__":
    main()
