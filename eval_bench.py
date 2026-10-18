"""Zero-shot benchmark run for KALIA checkpoints via lm-evaluation-harness.

Runs a small, honest suite on CPU: ARC-Easy, PIQA, HellaSwag, LAMBADA,
WinoGrande (0-shot). A 58M story model is expected to score near chance on
knowledge tasks; the numbers are published as-is.

**Sample size, and why the default is 2000.** D45 measured lm-eval's standard
error at `limit=500` at **1.99–2.24 pp** on these tasks — wider than every
movement the v0.2.0 evaluation reported, and wider than the 1.0 pp stability
allowance it was judged against. The 2026-10-17 review recommended raising the
limit and it was never applied; the benchmarks run on CPU, so this costs quota
nothing. Every number recorded before 2026-10-18 was measured at 500 and keeps
its ~2 pp error bar; the recorded notebooks are left as they ran. The default
here is 2000 for *new* runs, and a limit below `MIN_REPORTABLE_LIMIT` warns
loudly rather than passing silently, because a default nothing uses would be a
no-op.

`lm_eval` is imported inside `main` on purpose: it is a Kaggle-side dependency,
not in `requirements.txt`, and keeping it out of module scope lets the limit
policy be imported and tested without the harness installed.
"""

import argparse
import glob
import json
from pathlib import Path

# `from kalia_lm import KaliaLM` is deliberately *not* here: kalia_lm imports
# lm_eval, which is a Kaggle-side dependency absent from requirements.txt. The
# import lives inside main so the limit policy below can be tested anywhere.

# Below this, the standard error on these tasks exceeds most effects the project
# measures (D45: 1.99-2.24 pp at 500). Not a hard limit -- a historical run may
# legitimately be reproduced at 500 -- but it must be visible in the log.
MIN_REPORTABLE_LIMIT = 1000


def limit_warning(limit: int) -> str | None:
    """Return a warning if this sample size cannot resolve the project's effects.

    Pure and importable without lm_eval, so the policy is testable and the flag
    cannot go back to being silently under-powered.
    """
    if limit >= MIN_REPORTABLE_LIMIT:
        return None
    return (
        f"--limit {limit} is below {MIN_REPORTABLE_LIMIT}: at 500 samples D45 measured "
        "1.99-2.24 pp standard error, wider than most movements this project reports. "
        "Numbers from this run must carry that error bar and must not be called "
        "improvements. Use the default for anything that will be claimed."
    )


def find_ckpt() -> str:
    hits = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
    assert hits, "no ckpt.pt found - attach the training kernel output"
    # Prefer the checkpoint belonging to whichever training kernel is attached, so
    # a stale sibling output can never be scored in place of the intended model.
    for slug in ("kalia-train-v020", "kalia-train-v012", "kalia-train"):
        preferred = [h for h in hits if slug in h]
        if preferred:
            print(f"checkpoint source: {slug}")
            return preferred[0]
    print("checkpoint source: ambiguous, using first hit")
    return hits[0]


def build_parser() -> argparse.ArgumentParser:
    """Split out from main so the default sample size is testable: a default
    nothing exercises is how the limit stayed at 500 through two releases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--tasks", type=str, default="arc_easy,piqa,hellaswag,lambada_openai,winogrande")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/bench_results.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()

    warning = limit_warning(args.limit)
    if warning:
        print("WARNING:", warning)
    print(f"samples per task: {args.limit}")

    import lm_eval
    from kalia_lm import KaliaLM

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
