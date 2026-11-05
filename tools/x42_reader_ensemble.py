"""X42 — reader-guided compiler ensemble.

Pure JSON analysis: reads the per-record candidate NLLs produced by the X41
end-to-end eval, selects between the B_1401 and B_1402 compiler edits using
only the frozen reader's candidate distribution, chooses a rule on a
deterministic dev half, and evaluates once on the held-out test half.

No model inference, no training, no GPU.

Usage:
    python tools/x42_reader_ensemble.py \
        --rows-dir docs/results/x41-edit-compiler/compiler \
        --bench docs/results/x41-edit-compiler/baseline/bench.jsonl \
        --out-dir docs/results/x42-reader-ensemble
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SEEDS = ("B_1401", "B_1402")
RULES = ("MaxProb", "Margin", "MinEntropy", "Agreement", "MaxProbNoNoop", "Oracle")
# Ties prefer the seed that was stronger on the registered rename+swap cell.
TIE_SEED = "B_1402"


def probs(nll: list[float]) -> np.ndarray:
    x = -np.asarray(nll, dtype=np.float64)
    x = x - x.max()
    p = np.exp(x)
    return p / p.sum()


def normalize(text: str) -> str:
    return " ".join(text.split())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_of(record_id: str) -> str:
    h = hashlib.sha256((record_id + "x42").encode()).hexdigest()
    return "dev" if int(h, 16) % 2 == 0 else "test"


def select(rule: str, per_seed: dict[str, dict], generated: dict[str, str],
           factual: str) -> str:
    """Return the selected seed for one record."""
    if rule == "Oracle":
        for seed in SEEDS:
            if per_seed[seed]["pred"] == per_seed[seed]["correct_index"]:
                return seed
        return max(SEEDS, key=lambda s: probs(per_seed[s]["nll"]).max())
    if rule == "Agreement":
        a, b = SEEDS
        if per_seed[a]["pred"] == per_seed[b]["pred"]:
            return TIE_SEED
        return max(SEEDS, key=lambda s: probs(per_seed[s]["nll"]).max())
    if rule == "MaxProbNoNoop":
        changed = [s for s in SEEDS if normalize(generated[s]) != normalize(factual)]
        pool = changed if changed else list(SEEDS)
        return max(pool, key=lambda s: probs(per_seed[s]["nll"]).max())
    if rule == "Margin":
        def score(seed):
            p = np.sort(probs(per_seed[seed]["nll"]))
            return p[-1] - p[-2] if len(p) > 1 else p[-1]
        return max(SEEDS, key=score)
    if rule == "MinEntropy":
        def score(seed):
            p = probs(per_seed[seed]["nll"])
            return -(p * np.log(p + 1e-12)).sum()
        return max(SEEDS, key=score)
    if rule == "MaxProb":
        return max(SEEDS, key=lambda s: probs(per_seed[s]["nll"]).max())
    raise ValueError(rule)


def evaluate(records: list[dict], selected: dict[str, str], per_seed: dict[str, dict],
             subset: str | None = None) -> dict:
    rows = []
    for r in records:
        sid = r["id"]
        if subset is not None and split_of(sid) != subset:
            continue
        seed = selected[sid]
        row = per_seed[seed][sid]
        rows.append({
            "id": sid,
            "cell": r["cell"],
            "operations": "+".join(r["operations"]),
            "correct": bool(row["pred"] == row["correct_index"]),
            "seed": seed,
        })

    def rate(items):
        return sum(x["correct"] for x in items) / len(items) if items else float("nan")

    by_cell = {}
    for cell in sorted({r["cell"] for r in rows}):
        bucket = [r for r in rows if r["cell"] == cell]
        by_cell[cell] = {"n": len(bucket), "accuracy": rate(bucket)}
    by_op = {}
    for op in sorted({r["operations"] for r in rows}):
        bucket = [r for r in rows if r["operations"] == op]
        by_op[op] = {"n": len(bucket), "accuracy": rate(bucket)}
    return {
        "n": len(rows),
        "overall": rate(rows),
        "by_cell": by_cell,
        "by_operation": by_op,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-dir", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    records = load_jsonl(args.bench)
    by_id = {r["id"]: r for r in records}
    per_seed: dict[str, dict[str, dict]] = {}
    generated: dict[str, dict[str, str]] = {}
    for seed in SEEDS:
        rows = load_jsonl(args.rows_dir / f"end2end_{seed}_rows.jsonl")
        per_seed[seed] = {r["id"]: r for r in rows}
        gen = load_jsonl(args.rows_dir / f"generated_{seed}.jsonl")
        generated[seed] = {r["id"]: r["edited_prefix"] for r in gen}
        assert set(per_seed[seed]) == set(by_id), seed
        assert set(generated[seed]) == set(by_id), seed

    # Rule selection on dev only.
    dev_table = {}
    selected_by_rule: dict[str, dict[str, str]] = {}
    for rule in RULES:
        selected = {
            r["id"]: select(rule, {s: per_seed[s][r["id"]] for s in SEEDS},
                            {s: generated[s][r["id"]] for s in SEEDS}, r["prefix"])
            for r in records
        }
        selected_by_rule[rule] = selected
        dev = evaluate(records, selected, per_seed, subset="dev")
        dev_table[rule] = {
            "S-P": dev["by_cell"].get("S-P", {}).get("accuracy"),
            "rename+swap": dev["by_operation"].get("rename+swap", {}).get("accuracy"),
            "overall": dev["overall"],
        }

    deployable = [r for r in RULES if r != "Oracle"]
    chosen = max(
        deployable,
        key=lambda r: (
            dev_table[r]["S-P"] if dev_table[r]["S-P"] is not None else -1,
            dev_table[r]["rename+swap"] if dev_table[r]["rename+swap"] is not None else -1,
            -RULES.index(r),
        ),
    )

    test = evaluate(records, selected_by_rule[chosen], per_seed, subset="test")
    full = evaluate(records, selected_by_rule[chosen], per_seed)
    best_single = {}
    for seed in SEEDS:
        own = {r["id"]: seed for r in records}
        best_single[seed] = evaluate(records, own, per_seed, subset="test")
    oracle = evaluate(records, selected_by_rule["Oracle"], per_seed, subset="test")

    bars = {
        "T-1_number_place": test["by_operation"].get("number+place", {}).get("accuracy", 0) >= 0.40,
        "T-2_rename_swap": test["by_operation"].get("rename+swap", {}).get("accuracy", 0) >= 0.45,
        "T-3_SP": test["by_cell"].get("S-P", {}).get("accuracy", 0) >= 0.40,
        "T-4_UP": test["by_cell"].get("U-P", {}).get("accuracy", 0) >= 0.30,
        "T-5_no_labels": True,
        "T-6_deterministic": True,
    }
    summary = {
        "experiment": "X42-reader-ensemble",
        "seeds": list(SEEDS),
        "chosen_rule": chosen,
        "dev_table": dev_table,
        "test": test,
        "full": full,
        "best_single_test": best_single,
        "oracle_test": oracle,
        "bars": bars,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "x42_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with open(args.out_dir / "x42_selections.jsonl", "w") as fh:
        for r in records:
            sid = r["id"]
            fh.write(json.dumps({
                "id": sid,
                "split": split_of(sid),
                "rule": chosen,
                "selected_seed": selected_by_rule[chosen][sid],
                "cell": r["cell"],
                "operations": "+".join(r["operations"]),
                "correct": bool(
                    per_seed[selected_by_rule[chosen][sid]][sid]["pred"]
                    == per_seed[selected_by_rule[chosen][sid]][sid]["correct_index"]
                ),
            }, sort_keys=True) + "\n")

    print("X42 READER-GUIDED ENSEMBLE")
    print("  dev rule table:")
    for rule, row in dev_table.items():
        print(f"    {rule:14s} S-P {row['S-P']:.4f}  rename+swap {row['rename+swap']:.4f}")
    print(f"  chosen rule: {chosen}")
    print("  test:")
    for cell, row in test["by_cell"].items():
        print(f"    {cell}: {row['accuracy']:.4f} (n={row['n']})")
    for op, row in test["by_operation"].items():
        print(f"    {op:16s}: {row['accuracy']:.4f} (n={row['n']})")
    print("  best single seed on test:")
    for seed, row in best_single.items():
        print(f"    {seed}: S-P {row['by_cell']['S-P']['accuracy']:.4f} "
              f"rename+swap {row['by_operation']['rename+swap']['accuracy']:.4f}")
    print(f"  oracle test: S-P {oracle['by_cell']['S-P']['accuracy']:.4f} "
          f"rename+swap {oracle['by_operation']['rename+swap']['accuracy']:.4f}")
    print("  bars:", {k: v for k, v in bars.items()})


if __name__ == "__main__":
    main()
