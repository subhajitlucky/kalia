"""Kaggle CPU kernel: 50/50 soup of v0.3.1 and X49 treatment."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys
from pathlib import Path
import torch

hits = sorted(Path("/kaggle/input").rglob("ckpt.pt"))
hits = [p for p in hits if "x49-treatment" in str(p)]
assert hits, "X49 treatment output not attached"
treat_path = hits[0]
try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "safetensors", "huggingface_hub"], check=True)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
from model import GPT, GPTConfig

base_dir = Path("/kaggle/working/base"); base_dir.mkdir(parents=True, exist_ok=True)
sf = hf_hub_download("kalia-lm/kalia-v031", "model.safetensors", local_dir=str(base_dir))
cf = hf_hub_download("kalia-lm/kalia-v031", "config.json", local_dir=str(base_dir))
raw = json.loads(Path(cf).read_text())
cfg = {k: v for k, v in raw.items() if k in GPTConfig.__dataclass_fields__}
base = GPT(GPTConfig(**cfg))
base.load_state_dict(load_file(sf), strict=False)
treat = torch.load(treat_path, map_location="cpu", weights_only=False)
assert treat["config"]["model"] == cfg
bs = base.state_dict()
ts = treat["model"]
assert set(bs) == set(ts), (set(bs) ^ set(ts))
soup = {k: (bs[k].float() + ts[k].float()) * 0.5 for k in bs}
out = Path("/kaggle/working/x50/soup"); out.mkdir(parents=True, exist_ok=True)
ckpt = out / "ckpt.pt"
torch.save({"model": soup, "config": {"model": cfg}, "step": 0, "tokens": 0}, ckpt)
meta = {"experiment": "X50-soup", "base": "v0.3.1", "treatment": str(treat_path),
        "ratio": 0.5, "ckpt_bytes": ckpt.stat().st_size,
        "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()}
(out / "soup_manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
print(json.dumps(meta, indent=2, sort_keys=True))
