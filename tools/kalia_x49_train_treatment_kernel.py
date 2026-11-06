"""Kaggle GPU kernel: one X49 continued-pretraining arm.

Set MIXTURE, SEED and MAX_MINUTES at the top. Resume v0.3.1, train 2,000
re-warm steps on the selected mixture. The full arm should fit a 9-hour
session; MAX_MINUTES is a safety stop for a smoke run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MIXTURE = "treatment"
SEED = 1501
MAX_MINUTES = 0  # 0 = full run
MAX_STEPS = None  # None = config max_steps
CONFIG = "configs/x49-cont.yaml"

BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"

hits = sorted(Path("/kaggle/input").rglob("train.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))
print("code:", code_root, flush=True)

data_hits = sorted(Path("/kaggle/input").rglob("train.bin"))
preferred_data = [p for p in data_hits if f"x49-data/{MIXTURE}" in str(p)]
assert preferred_data, f"x49-data/{MIXTURE} not attached: {data_hits}"
train_bin = preferred_data[0]
val_bin = train_bin.parent / "val.bin"
assert val_bin.exists(), val_bin
data_dir = Path(f"/kaggle/working/data_{MIXTURE}")
data_dir.mkdir(parents=True, exist_ok=True)
for name, src in (("train.bin", train_bin), ("val.bin", val_bin)):
    dst = data_dir / name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
print("data:", train_bin, val_bin, flush=True)

try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "safetensors", "huggingface_hub"], check=True)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

import torch  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

base_dir = Path("/kaggle/working/base")
base_dir.mkdir(parents=True, exist_ok=True)
sf = hf_hub_download(BASE_REPO, "model.safetensors", local_dir=str(base_dir))
cf = hf_hub_download(BASE_REPO, "config.json", local_dir=str(base_dir))
base_sha = hashlib.sha256(Path(sf).read_bytes()).hexdigest()
assert base_sha == BASE_SHA, (base_sha, BASE_SHA)
raw_cfg = json.loads(Path(cf).read_text())
cfg = {k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__}
model = GPT(GPTConfig(**cfg))
load = model.load_state_dict(load_file(sf), strict=False)
assert not load.missing_keys and not load.unexpected_keys
base_ckpt = Path("/kaggle/working/base_v031.pt")
torch.save({"model": model.state_dict(), "config": {"model": cfg},
            "step": 0, "tokens": 0}, base_ckpt)
print("base sha:", base_sha, flush=True)

out_dir = Path(f"/kaggle/working/x49/{MIXTURE}_{SEED}")
out_dir.mkdir(parents=True, exist_ok=True)
config_path = Path(CONFIG)
if MAX_STEPS is not None:
    import yaml
    cfg = yaml.safe_load(config_path.read_text())
    cfg["train"]["max_steps"] = int(MAX_STEPS)
    cfg["train"]["update_budget"] = int(MAX_STEPS)
    cfg["train"]["warmup_steps"] = min(5, max(1, int(MAX_STEPS) // 5))
    cfg["train"]["eval_interval"] = int(MAX_STEPS)
    cfg["train"]["eval_steps"] = 20
    config_path = Path("/kaggle/working/x49_smoke.yaml")
    config_path.write_text(yaml.safe_dump(cfg))
cmd = [sys.executable, "train.py", "--config", str(config_path),
       "--data-dir", str(data_dir), "--out-dir", str(out_dir),
       "--resume-from", str(base_ckpt), "--seed", str(SEED)]
if MAX_STEPS is not None:
    cmd += ["--max-steps", str(MAX_STEPS)]
if MAX_MINUTES:
    cmd += ["--max-minutes", str(MAX_MINUTES)]
print("cmd:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

ckpt = out_dir / "ckpt.pt"
assert ckpt.exists(), ckpt
meta = {
    "experiment": "X49-arm",
    "mixture": MIXTURE,
    "seed": SEED,
    "ckpt": str(ckpt),
    "ckpt_bytes": ckpt.stat().st_size,
    "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    "base_sha256": base_sha,
    "train_bin": str(train_bin),
    "val_bin": str(val_bin),
}
(out_dir / "arm_manifest.json").write_text(
    json.dumps(meta, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
