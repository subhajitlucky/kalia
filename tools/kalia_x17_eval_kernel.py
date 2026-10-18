"""Kaggle CPU kernel: X17's 2x2 — train-time mask x eval-time mask.

X17 measured the treatment at **+0.0444 nats** (worse) against its paired
control, at both seeds, 6x the paired spread. But both arms were evaluated with
full causal attention: the control was trained and tested matched, and the
treatment was tested in a regime it had never trained in. The registration
predicted this — *"the eval must use the same masking as training to be
meaningful"* — in its pass branch, and the experiment never applied it.

So this scores all four archived checkpoints under both evaluation settings and
reports the cell the registration should have had:

|            | eval unmasked | eval masked |
|---|---|---|
| train unmasked (control) | as recorded | control under mask |
| train masked (treatment) | as recorded | **matched — the missing cell** |

**What is held constant, and why it decides the answer.** Everything except the
attention mask:

- **The same windows.** A fresh `torch.Generator` seeded 1234 for every cell, so
  all eight runs see identical batches. `evaluate()` in train.py draws from the
  global RNG, so consecutive evaluations see *different* windows — the same
  defect the whole-record audit found in the in-loop val curve. Repeating it here
  would put a sampling difference underneath a 0.04-nat question.
- **The same loss tokens.** `loss_mask=None` in every cell. A document mask also
  drops boundary positions from the loss, so applying it would average over a
  different token set than the unmasked cell and make the columns
  incomparable — the exact problem this kernel exists to remove. Only
  `attn_mask` varies.

The loss therefore asks one question: *does restricting attention to within-document
at evaluation time help this model?*

CPU on purpose: this is a measurement, not a training run, and it costs no quota.
"""

import glob
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

KERNEL_ID = "unknown"  # staging pins, from the committed metadata
KERNEL_REVISION = "unknown"  # staging pins, from git HEAD

ARM = "mask2x2"  # staging pins this; it names the experiment and its outputs
ARMS = ("control-s1338", "treatment-s1338", "control-s1339", "treatment-s1339")
BATCHES = 100
BATCH_SIZE = 8
EVAL_SEED = 1234

import torch  # noqa: E402


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_run_manifest(out_path, **fields) -> dict:
    fields.update(
        kernel_id=KERNEL_ID,
        kernel_revision=KERNEL_REVISION,
        recorded=datetime.now(timezone.utc).isoformat(),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(fields, fh, indent=2, sort_keys=True)
    print("run manifest: " + json.dumps(fields, sort_keys=True))
    return fields


# --- the code, resolved the same way every kernel resolves it ----------------
hits = sorted(glob.glob("/kaggle/input/**/model.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, (
    "kalia-code-dev not attached or lacks model.py; other matches "
    "(do NOT trust them, they are stale bundles): " + str(hits)
)
code_root = os.path.dirname(preferred[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("model.py", "data.py", "eval_reversibility.py"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"
print("code:", code_root)

from data import TokenDataset, document_mask  # noqa: E402
from eval_reversibility import load_model  # noqa: E402

val_hits = sorted(glob.glob("/kaggle/input/**/val.bin", recursive=True))
assert val_hits, "attach subhajitlucky/kalia-prep-v2b as a kernel source for val.bin"
val_bin = val_hits[0]
print("val shard:", val_bin)

ckpt_root = None
for hit in sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)):
    if "/x17" in hit:
        ckpt_root = Path(hit).parent.parent
        break
assert ckpt_root, "attach subhajitlucky/kalia-x17-docmask as a kernel source"
print("checkpoints:", ckpt_root)


@torch.no_grad()
def eval_loss(ckpt: Path, use_attn_mask: bool, device) -> float:
    """Mean per-token loss over BATCHES fixed windows.

    A fresh generator per call, so every cell sees the same windows; loss_mask
    is never passed, so every cell averages over the same tokens.
    """
    model = load_model(ckpt, device)
    ds = TokenDataset(val_bin, model.cfg.context_len)
    gen = torch.Generator().manual_seed(EVAL_SEED)  # same windows for every cell
    total, tokens = 0.0, 0
    for _ in range(BATCHES):
        if use_attn_mask:
            x, y, attn, _ = ds.get_batch(BATCH_SIZE, device, gen, return_masks=True)
            _, loss = model(x, y, attn_mask=attn)
        else:
            x, y = ds.get_batch(BATCH_SIZE, device, gen)
            _, loss = model(x, y)
        total += loss.item() * y.numel()
        tokens += int(y.numel())
    del model
    return total / tokens


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | {BATCHES} batches x {BATCH_SIZE} windows, seed {EVAL_SEED}")

    cells = {}
    for arm in ARMS:
        ckpt = ckpt_root / arm / "ckpt.pt"
        assert ckpt.exists(), f"missing checkpoint {ckpt}"
        for use_mask in (False, True):
            key = f"{arm}|eval_{'masked' if use_mask else 'unmasked'}"
            cells[key] = eval_loss(ckpt, use_mask, device)
            print(f"  {key:34s} {cells[key]:.6f}")

    write_run_manifest(
        "/kaggle/working/mask2x2.json",
        experiment=ARM,
        batches=BATCHES,
        batch_size=BATCH_SIZE,
        eval_seed=EVAL_SEED,
        loss_mask="never (only attn_mask varies)",
        cells=cells,
        code_snapshot=code_root,
        code_fingerprint=hashlib.sha256(
            b"".join(sorted(p.read_bytes() for p in Path(code_root).glob("*.py")))
        ).hexdigest(),
        val_bin={"path": val_bin, "sha256": sha256_file(val_bin)},
        checkpoints={arm: sha256_file(ckpt_root / arm / "ckpt.pt") for arm in ARMS},
    )

    # --- the 2x2, and the cell the registration was missing -----------------
    print("\n" + "=" * 68)
    print("X17 2x2 (mean per-token val loss, identical windows and loss tokens)")
    print(f"{'':26}{'eval unmasked':>16}{'eval masked':>14}")
    for label, arm in (("train unmasked (control)", "control"),
                       ("train masked (treatment)", "treatment")):
        means = []
        for masked in (False, True):
            tag = "masked" if masked else "unmasked"
            per_seed = [cells[f"{arm}-{s}|eval_{tag}"] for s in ("1338", "1339")]
            means.append(sum(per_seed) / len(per_seed))
        print(f"  {label:24s}{means[0]:16.4f}{means[1]:14.4f}")

    print("\npaired deltas (treatment - control), per seed:")
    for seed in ("1338", "1339"):
        for masked, name in ((False, "eval unmasked"), (True, "eval masked")):
            tag = "masked" if masked else "unmasked"
            c = cells[f"control-{seed}|eval_{tag}"]
            t = cells[f"treatment-{seed}|eval_{tag}"]
            print(f"  seed {seed} {name:14s} control {c:.4f}  treatment {t:.4f}  delta {t - c:+.4f}")

    print("\nThe matched comparison is train-masked/eval-masked against")
    print("train-unmasked/eval-unmasked. The mismatched one is what X17's")
    print("registration actually measured.")
    print("archived: /kaggle/working/mask2x2.json")


if __name__ == "__main__":
    main()
