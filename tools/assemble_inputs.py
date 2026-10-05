#!/usr/bin/env python3
"""Assemble kalia-v030-inputs from prep outputs + local checkpoint.

Usage: assemble_inputs.py <prep_output_dir> <ckpt.pt> <out_dir>
  - copies train.bin, val.bin (+ per-source bins if present)
  - slices the first 1M training tokens as probe_val.bin (CL-0 measuring set)
  - copies ckpt.pt
  - prints sha256 of every artifact (G2 record)
"""
import hashlib
import shutil
import sys
from pathlib import Path

PROBE_TOKENS = 1_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    prep, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ("train.bin", "val.bin", "tinystories.bin",
                 "smollm_fineweb_edu.bin", "cosmopedia.bin", "stack_smol.bin"):
        src = prep / name
        if not src.exists():
            print(f"skip {name}: not in prep output")
            continue
        shutil.copyfile(src, out / name)
        manifest[name] = sha256(out / name)
        print(f"{name:28s} {src.stat().st_size / 1e6:9.1f} MB  {manifest[name][:16]}...")
    train = out / "train.bin"
    assert train.exists(), "train.bin missing — cannot derive probe shard"
    with open(train, "rb") as fh:
        head = fh.read(PROBE_TOKENS * 2)
    assert len(head) == PROBE_TOKENS * 2, "train.bin shorter than probe slice"
    (out / "probe_val.bin").write_bytes(head)
    manifest["probe_val.bin"] = sha256(out / "probe_val.bin")
    shutil.copyfile(ckpt, out / "ckpt.pt")
    manifest["ckpt.pt"] = sha256(out / "ckpt.pt")
    print(f"probe_val.bin + ckpt.pt staged; {len(manifest)} artifacts hashed")


if __name__ == "__main__":
    main()
