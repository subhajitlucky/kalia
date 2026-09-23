"""Run KALIA micro-ablation arms sequentially and report validation loss.

Arms are cfg files sitting either in ``configs/`` (local repo) or next to this
script (flat Kaggle dataset). Each arm trains the same 30M-parameter model on
~100M tokens; the final validation loss decides the winner.

With ``--reversibility`` the Abhimanyu gap (reverse-token NLL minus forward
NLL, see ``eval_reversibility.py``) is measured on each arm's final checkpoint
and reported alongside validation loss (X16 pre-registration).
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

def find_sentences(explicit: str | None) -> Path:
    candidates = [Path(explicit)] if explicit else []
    candidates += [Path("eval/probe_sentences.json"), Path("probe_sentences.json")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("probe_sentences.json not found")


def abhimanyu_gap(ckpt_path: Path, sentences_path: Path) -> float:
    import json

    import torch
    import tiktoken

    from eval_reversibility import forward_and_reverse_nll, load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, device)
    encoder = tiktoken.get_encoding("gpt2")
    sentences = json.loads(sentences_path.read_text())["sentences"]
    result = forward_and_reverse_nll(model, encoder, sentences, device)
    print(
        f"reversibility: forward {result['forward_nll']} | reverse {result['reverse_nll']} | "
        f"gap {result['abhimanyu_gap']}"
    )
    return result["abhimanyu_gap"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run micro-ablation arms")
    parser.add_argument("--arms", type=str, default=",".join(ARMS))
    parser.add_argument("--steps", type=int, default=None, help="override max_steps")
    parser.add_argument("--reversibility", action="store_true", help="report Abhimanyu gap")
    parser.add_argument("--sentences", type=str, default=None, help="probe sentences json")
    parser.add_argument("--data-dir", type=str, default=None, help="override data directory")
    parser.add_argument("--out-root", type=str, default="/kaggle/working/out")
    args = parser.parse_args()
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    sentences_path = find_sentences(args.sentences) if args.reversibility else None

    data_dir = args.data_dir or find_data_dir()
    print("data:", data_dir)
    results = []
    for arm in arms:
        out_dir = str(Path(args.out_root) / arm)
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
            results.append((arm, "FAILED", None))
            continue
        val_path = Path(out_dir) / "val_log.csv"
        rows = val_path.read_text().strip().splitlines()[1:] if val_path.exists() else []
        gap = None
        if args.reversibility:
            gap = abhimanyu_gap(Path(out_dir) / "ckpt.pt", sentences_path)
        results.append((arm, rows[-1] if rows else "no evals", gap))

    print("=" * 64)
    print("ABLATION SUMMARY (final step, val loss, abhimanyu gap)")
    for arm, final, gap in results:
        line = f"{arm:24s} {final}"
        if gap is not None:
            line += f"  gap={gap:+.4f}"
        print(line)


if __name__ == "__main__":
    main()
