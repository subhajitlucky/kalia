"""Run KALIA micro-ablation arms sequentially and report validation loss.

Arms are cfg files sitting either in ``configs/`` (local repo) or next to this
script (flat Kaggle dataset). Each arm trains the same 30M-parameter model on
~100M tokens; the final validation loss decides the winner.
"""

import argparse
import glob
import subprocess
import sys
from pathlib import Path

ARMS = ["micro-base", "micro-muon", "micro-muon-qk"]


def config_path(arm: str) -> str:
    for candidate in (Path("configs") / f"{arm}.yaml", Path(f"{arm}.yaml")):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"no config for arm {arm}")


def find_data_dir() -> str:
    hits = sorted(glob.glob("/kaggle/input/**/train.bin", recursive=True))
    assert hits, "train.bin not found - attach the kalia-prep output"
    return str(Path(hits[0]).parent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run micro-ablation arms")
    parser.add_argument("--arms", type=str, default=",".join(ARMS))
    parser.add_argument("--steps", type=int, default=None, help="override max_steps")
    args = parser.parse_args()
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]

    data_dir = find_data_dir()
    print("data:", data_dir)
    results = []
    for arm in arms:
        out_dir = f"/kaggle/working/out/{arm}"
        print("=" * 64)
        print("ARM:", arm)
        cmd = [
            sys.executable,
            "train.py",
            "--config",
            config_path(arm),
            "--data-dir",
            data_dir,
            "--out-dir",
            out_dir,
        ]
        if args.steps is not None:
            cmd += ["--max-steps", str(args.steps)]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            results.append((arm, "FAILED"))
            continue
        val_path = Path(out_dir) / "val_log.csv"
        rows = val_path.read_text().strip().splitlines()[1:] if val_path.exists() else []
        results.append((arm, rows[-1] if rows else "no evals"))

    print("=" * 64)
    print("ABLATION SUMMARY (final step, val loss)")
    for arm, final in results:
        print(f"{arm:16s} {final}")


if __name__ == "__main__":
    main()
