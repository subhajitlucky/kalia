"""X40: seen/unseen candidate scoring and workspace diagnostics.

The key bar is I-3: the IP arm must beat the `pairs` arm on **unseen
compositions**, not just on the interventions it trained on. This module keeps
the scoring rule fixed and auditable:

- score each candidate by its mean NLL over the suffix tokens;
- choose the lowest NLL;
- seen = 2-way singles, unseen = 4-way compositions;
- workspace stats use the prompt-boundary workspace only.

No evaluation tuning: the candidate set and the scoring rule are fixed by the
pair generator and this file.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from intervention_pairs import InterventionDataset, causal_pad_mask


@torch.no_grad()
def score_candidates(
    model,
    ds: InterventionDataset,
    device: torch.device,
    batch_size: int = 16,
) -> dict[tuple[int, int], float]:
    """Mean suffix NLL for every (record, candidate) pair."""
    items = [
        (i, j)
        for i, record in enumerate(ds.records)
        for j in range(len(record["candidates"]))
    ]
    scores: dict[tuple[int, int], float] = {}
    model.eval()
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        xs, ys, masks, tms = [], [], [], []
        for i, j in chunk:
            x, y, mask, token_mask = ds.candidate(
                i, ds.records[i]["candidates"][j]
            )
            xs.append(x)
            ys.append(y)
            masks.append(mask)
            tms.append(token_mask)
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        mask = torch.stack(masks).to(device)
        token_mask = torch.stack(tms).to(device)
        logits, _ = model(
            x, attn_mask=causal_pad_mask(token_mask), token_mask=token_mask
        )
        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
        ).view(y.size(0), -1)
        nll = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        for (i, j), value in zip(chunk, nll.tolist()):
            scores[(i, j)] = float(value)
    return scores


def accuracy_from_scores(
    ds: InterventionDataset,
    scores: dict[tuple[int, int], float],
) -> dict:
    """Turn candidate NLLs into seen/unseen accuracy and per-op diagnostics."""
    rows = []
    for i, record in enumerate(ds.records):
        candidate_scores = [
            scores[(i, j)] for j in range(len(record["candidates"]))
        ]
        pred = min(range(len(candidate_scores)), key=lambda j: candidate_scores[j])
        rows.append({
            "id": record["id"],
            "operations": "+".join(record["operations"]),
            "composition": bool(record["composition"]),
            "n_way": int(record["n_way"]),
            "cell": record.get("cell"),
            "correct": bool(pred == record["correct_index"]),
            "pred": int(pred),
            "correct_index": int(record["correct_index"]),
            "nll": candidate_scores,
        })

    def rate(items):
        return sum(r["correct"] for r in items) / len(items) if items else float("nan")

    seen = [r for r in rows if not r["composition"]]
    unseen = [r for r in rows if r["composition"]]
    by_op: dict[str, dict] = {}
    for r in rows:
        bucket = by_op.setdefault(r["operations"], {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(r["correct"])
    for key, bucket in by_op.items():
        bucket["accuracy"] = bucket["correct"] / bucket["n"]

    # The 2x2 benchmark uses `cell`; the X40 records do not, so this is empty
    # for the original screen and does not change its verdict.
    by_cell: dict[str, dict] = {}
    for r in rows:
        cell = r.get("cell")
        if not cell:
            continue
        bucket = by_cell.setdefault(
            cell, {"n": 0, "correct": 0, "n_way": r["n_way"]}
        )
        bucket["n"] += 1
        bucket["correct"] += int(r["correct"])
    for key, bucket in by_cell.items():
        bucket["accuracy"] = bucket["correct"] / bucket["n"]
        bucket["chance"] = 1.0 / bucket["n_way"]

    return {
        "seen": rate(seen),
        "unseen": rate(unseen),
        "overall": rate(rows),
        "seen_n": len(seen),
        "unseen_n": len(unseen),
        "seen_chance": 1.0 / 2.0,
        "unseen_chance": 1.0 / 4.0,
        "by_operation": by_op,
        "by_cell": by_cell,
        "rows": rows,
    }


def _effective_rank(matrix: torch.Tensor) -> float:
    if matrix.numel() == 0:
        return 0.0
    sv = torch.linalg.svdvals(matrix.float())
    p = sv / sv.sum().clamp(min=1e-12)
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))


def _probe_accuracy(
    features: torch.Tensor,
    labels: list[str],
    seed: int = 0,
    ridge: float = 1.0,
) -> tuple[float, float, int]:
    """Ridge linear probe, deterministic 50/50 split. Returns (acc, chance, n)."""
    classes = sorted(set(labels))
    if len(classes) < 2 or features.size(0) < 4:
        return float("nan"), float("nan"), 0
    y = torch.tensor([classes.index(label) for label in labels], dtype=torch.long)
    n = features.size(0)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train = n // 2
    train, test = order[:n_train], order[n_train:]
    x = features.float()
    mean = x[train].mean(dim=0)
    std = x[train].std(dim=0).clamp(min=1e-6)
    z = (x - mean) / std
    z = torch.cat([z, torch.ones(n, 1)], dim=1)
    train_x = z[train]
    test_x = z[test]
    one_hot = F.one_hot(y[train], num_classes=len(classes)).float()
    gram = train_x.t() @ train_x
    reg = ridge * torch.eye(gram.size(0), dtype=gram.dtype)
    weights = torch.linalg.solve(gram + reg, train_x.t() @ one_hot)
    pred = (test_x @ weights).argmax(dim=1)
    acc = float((pred == y[test]).float().mean())
    return acc, 1.0 / len(classes), int(n)


@torch.no_grad()
def workspace_stats(
    model,
    ds: InterventionDataset,
    device: torch.device,
    batch_size: int = 64,
    probe_seed: int = 0,
) -> dict:
    """Slot entropy, effective rank and an intervention-type linear probe."""
    if getattr(model, "event", None) is None:
        return {}
    workspaces = []
    assignments = []
    labels: list[str] = []
    model.eval()
    for start in range(0, len(ds), batch_size):
        indices = list(range(start, min(start + batch_size, len(ds))))
        rows = [ds.example(i) for i in indices]
        x = torch.stack([row[0] for row in rows]).to(device)
        token_mask = torch.stack([row[3] for row in rows]).to(device)
        _, _, events = model(
            x, attn_mask=causal_pad_mask(token_mask),
            token_mask=token_mask, return_events=True,
        )
        workspaces.append(events["workspace"].float().cpu())
        assignments.append(events["assign"].float().cpu())
        labels.extend(
            "+".join(ds.records[i]["operations"]) for i in indices
        )
    values = torch.cat(workspaces, dim=0)
    assign = torch.cat(assignments, dim=0)
    p = assign.mean(dim=0)
    p = p / p.sum().clamp(min=1e-12)
    slot_entropy = float(-(p * (p + 1e-12).log2()).sum())
    slot_matrix = values.mean(dim=0)  # (K, slot_dim)
    feature = values.flatten(1)
    probe_acc, probe_chance, probe_n = _probe_accuracy(feature, labels, probe_seed)
    return {
        "slot_entropy_bits": slot_entropy,
        "effective_rank": _effective_rank(slot_matrix),
        "slot_norm": float(values.norm(dim=-1).mean()),
        "probe_accuracy": probe_acc,
        "probe_chance": probe_chance,
        "probe_gain": probe_acc - probe_chance,
        "probe_n": probe_n,
        "probe_classes": sorted(set(labels)),
        "n": int(values.size(0)),
        "slots": int(values.size(1)),
        "slot_dim": int(values.size(2)),
    }


def summarise_arm(
    model,
    ds: InterventionDataset,
    device: torch.device,
    score_batch_size: int = 16,
    stat_batch_size: int = 64,
    probe_seed: int = 0,
) -> dict:
    scores = score_candidates(model, ds, device, score_batch_size)
    accuracy = accuracy_from_scores(ds, scores)
    stats = workspace_stats(model, ds, device, stat_batch_size, probe_seed)
    return {"accuracy": accuracy, "workspace": stats}
