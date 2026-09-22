"""Train KALIA: single-process or DDP (torchrun), resumable, time-budgeted."""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW

from data import TokenDataset
from model import GPT, GPTConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    p.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt or --hub-repo")
    p.add_argument("--hub-repo", type=str, default=None, help="HF repo id for checkpoint sync")
    p.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    p.add_argument("--max-minutes", type=float, default=None, help="stop after this many minutes")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def setup_distributed():
    if "RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world
    return 0, 1


def get_lr(step: int, train_cfg: dict) -> float:
    peak = train_cfg["learning_rate"]
    if step < train_cfg["warmup_steps"]:
        return peak * (step + 1) / train_cfg["warmup_steps"]
    if step >= train_cfg["max_steps"]:
        return peak * train_cfg["min_lr_ratio"]
    progress = (step - train_cfg["warmup_steps"]) / max(
        1, train_cfg["max_steps"] - train_cfg["warmup_steps"]
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (train_cfg["min_lr_ratio"] + coeff * (1 - train_cfg["min_lr_ratio"]))


def save_checkpoint(path: Path, model, optimizer, step: int, tokens: int, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens": tokens,
            "config": config,
        },
        path,
    )


def push_to_hub(local_path: Path, repo_id: str, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo, repo_id=repo_id)


def pull_from_hub(repo_id: str, path_in_repo: str, local_path: Path) -> bool:
    from huggingface_hub import hf_hub_download

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id, filename=path_in_repo, token=os.environ.get("HF_TOKEN")
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "start fresh"
        print(f"resume: no checkpoint pulled ({exc})")
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(Path(downloaded).read_bytes())
    return True


@torch.no_grad()
def evaluate(model, dataset, eval_steps: int) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(eval_steps):
        x, y = dataset.get_batch(8, next(model.parameters()).device)
        _, loss = model(x, y)
        total += loss.item()
    if was_training:
        model.train()
    return total / eval_steps


@torch.no_grad()
def print_sample(model, sample_tokens: int) -> None:
    raw = model.module if hasattr(model, "module") else model
    device = next(raw.parameters()).device
    prompt = torch.tensor([[464]], device=device)  # "Once" via GPT-2 BPE
    out = raw.generate(prompt, max_new_tokens=sample_tokens, temperature=0.8, top_k=200)
    print("sample ids:", out[0].tolist()[:20], "...")


def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    train_cfg = dict(config["train"])
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
    if args.seed is not None:
        train_cfg["seed"] = args.seed

    rank, world = setup_distributed()
    is_master = rank == 0
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    is_cuda = device.type == "cuda"

    torch.manual_seed(train_cfg["seed"] + rank)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = GPTConfig(**config["model"])
    model = GPT(model_cfg).to(device)
    if is_master:
        print(f"KALIA params: {model.num_params():,} | device: {device} | world: {world}")

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=train_cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    start_step, tokens_seen = 0, 0
    ckpt_path = args.out_dir / "ckpt.pt"
    if args.resume:
        if args.hub_repo:
            pull_from_hub(args.hub_repo, "checkpoints/ckpt.pt", ckpt_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step, tokens_seen = ckpt["step"], ckpt["tokens"]
            if is_master:
                print(f"resumed from step {start_step} ({tokens_seen:,} tokens)")

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()]
        )

    train_ds = TokenDataset(args.data_dir / "train.bin", model_cfg.context_len)
    val_path = args.data_dir / "val.bin"
    val_ds = TokenDataset(val_path, model_cfg.context_len) if val_path.exists() else None
    generator = torch.Generator().manual_seed(train_cfg["seed"] + rank)

    log_path = args.out_dir / "train_log.csv"
    if is_master:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if not log_path.exists() or start_step == 0:
            with open(log_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["step", "loss", "lr", "tokens", "elapsed_s"])

    tokens_per_step = (
        train_cfg["micro_batch_size"] * train_cfg["grad_accum_steps"] * model_cfg.context_len * world
    )
    start_time = time.time()
    last_ckpt_time = start_time
    model.train()

    step = start_step
    try:
        while step < train_cfg["max_steps"]:
            if args.max_minutes is not None and (time.time() - start_time) / 60 >= args.max_minutes:
                if is_master:
                    print("time budget reached; stopping cleanly")
                break

            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            t0 = time.time()
            for _ in range(train_cfg["grad_accum_steps"]):
                x, y = train_ds.get_batch(train_cfg["micro_batch_size"], device, generator)
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=is_cuda):
                    _, loss = model(x, y)
                    loss = loss / train_cfg["grad_accum_steps"]
                scaler.scale(loss).backward()
                loss_total += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            lr = get_lr(step, train_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()

            tokens_seen += tokens_per_step
            step += 1

            if is_master and step % train_cfg["log_interval"] == 0:
                elapsed = time.time() - t0
                tok_per_s = tokens_per_step / elapsed
                print(
                    f"step {step}/{train_cfg['max_steps']} | loss {loss_total:.4f} | "
                    f"lr {lr:.2e} | {tok_per_s:,.0f} tok/s"
                )
                with open(log_path, "a", newline="") as fh:
                    csv.writer(fh).writerow(
                        [step, f"{loss_total:.4f}", f"{lr:.6e}", tokens_seen, f"{time.time() - start_time:.1f}"]
                    )

            if (
                val_ds is not None
                and train_cfg["eval_interval"] > 0
                and step % train_cfg["eval_interval"] == 0
            ):
                if is_master:
                    val_loss = evaluate(model, val_ds, train_cfg["eval_steps"])
                    print(f"step {step} | val loss {val_loss:.4f}")

            if train_cfg["sample_interval"] > 0 and step % train_cfg["sample_interval"] == 0 and is_master:
                print_sample(model, train_cfg["sample_tokens"])

            due = train_cfg["checkpoint_interval_minutes"] > 0 and (
                time.time() - last_ckpt_time
            ) / 60 >= train_cfg["checkpoint_interval_minutes"]
            if due and is_master:
                save_checkpoint(ckpt_path, model, optimizer, step, tokens_seen, config)
                print(f"checkpoint saved at step {step}")
                if args.hub_repo:
                    push_to_hub(ckpt_path, args.hub_repo, "checkpoints/ckpt.pt")
                    push_to_hub(log_path, args.hub_repo, "logs/train_log.csv")
                last_ckpt_time = time.time()

        if is_master:
            save_checkpoint(ckpt_path, model, optimizer, step, tokens_seen, config)
            print(f"final checkpoint saved at step {step}")
            if args.hub_repo:
                push_to_hub(ckpt_path, args.hub_repo, "checkpoints/ckpt.pt")
                push_to_hub(log_path, args.hub_repo, "logs/train_log.csv")
    finally:
        if world > 1:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
