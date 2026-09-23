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
from optim import MuonWithAuxAdam, split_muon_params


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    p.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt or --hub-repo")
    p.add_argument("--hub-repo", type=str, default=None, help="HF repo id for checkpoint sync")
    p.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    p.add_argument("--max-minutes", type=float, default=None, help="stop after this many minutes")
    p.add_argument("--target-val-loss", type=float, default=None, help="decay and stop once reached")
    p.add_argument("--decay-steps-after-target", type=int, default=None)
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


def base_lr_scale(step: int, train_cfg: dict) -> float:
    """Cosine schedule multiplier in [min_lr_ratio, 1] (warmup + cosine)."""
    if step < train_cfg["warmup_steps"]:
        return (step + 1) / train_cfg["warmup_steps"]
    if step >= train_cfg["max_steps"]:
        return train_cfg["min_lr_ratio"]
    progress = (step - train_cfg["warmup_steps"]) / max(
        1, train_cfg["max_steps"] - train_cfg["warmup_steps"]
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return train_cfg["min_lr_ratio"] + coeff * (1 - train_cfg["min_lr_ratio"])


def lr_scale(step: int, train_cfg: dict, decay_start: int | None = None) -> float:
    """Schedule multiplier; after a target loss is reached, decay linearly and stop."""
    if decay_start is None or step < decay_start:
        return base_lr_scale(step, train_cfg)
    decay_steps = max(1, int(train_cfg.get("decay_steps_after_target", 1)))
    progress = min(1.0, (step - decay_start) / decay_steps)
    start = base_lr_scale(decay_start, train_cfg)
    floor = train_cfg["min_lr_ratio"]
    return start * (1 - progress) + floor * progress


def build_optimizer(model, train_cfg: dict):
    """AdamW for everything, or Muon on hidden 2D weights + AdamW for the rest.

    Every param group carries a ``base_lr`` so the training loop can apply one
    shared schedule multiplier.
    """
    if train_cfg.get("optimizer", "adamw") == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            betas=(train_cfg["beta1"], train_cfg["beta2"]),
            weight_decay=train_cfg["weight_decay"],
        )
        for group in optimizer.param_groups:
            group["base_lr"] = train_cfg["learning_rate"]
        return optimizer

    hidden, other = split_muon_params(model)
    return MuonWithAuxAdam(
        hidden,
        other,
        muon_lr=train_cfg["muon_learning_rate"],
        muon_momentum=train_cfg.get("muon_momentum", 0.95),
        muon_weight_decay=train_cfg.get("muon_weight_decay", 0.0),
        ns_steps=train_cfg.get("ns_steps", 5),
        muon_plus=train_cfg.get("muon_plus", "none"),
        adam_lr=train_cfg["learning_rate"],
        adam_betas=(train_cfg["beta1"], train_cfg["beta2"]),
        adam_weight_decay=train_cfg["weight_decay"],
    )


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
    except Exception as exc:  # noqa: BLE001 - any failure means "not available"
        print(f"resume: could not pull {path_in_repo} ({exc})")
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and atomically swap it in, so concurrent readers
    # can never observe a partially written checkpoint.
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    tmp_path.write_bytes(Path(downloaded).read_bytes())
    os.replace(tmp_path, local_path)
    return True


@torch.no_grad()
def evaluate(model, dataset, eval_steps: int, bytes_per_token: float | None = None):
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(eval_steps):
        x, y = dataset.get_batch(8, next(model.parameters()).device)
        _, loss = model(x, y)
        total += loss.item()
    if was_training:
        model.train()
    mean_loss = total / eval_steps
    bpb = mean_loss / math.log(2) / bytes_per_token if bytes_per_token else None
    return mean_loss, bpb


def compute_bytes_per_token(dataset, encoder, n_batches: int = 20) -> float:
    """Average UTF-8 bytes per token on this dataset (for bits-per-byte)."""
    total_bytes = 0
    total_tokens = 0
    for _ in range(n_batches):
        x, _ = dataset.get_batch(8, torch.device("cpu"))
        for row in x:
            total_bytes += len(encoder.decode(row.tolist()).encode("utf-8"))
            total_tokens += int(row.numel())
    return total_bytes / total_tokens


def update_ema(ema_state: dict, model_state: dict, beta: float) -> None:
    """EMA of model weights (for a smoother final checkpoint)."""
    for key, value in model_state.items():
        ema_state[key].mul_(beta).add_(value.detach().float(), alpha=1 - beta)


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
    if args.target_val_loss is not None:
        train_cfg["target_val_loss"] = args.target_val_loss
    if args.decay_steps_after_target is not None:
        train_cfg["decay_steps_after_target"] = args.decay_steps_after_target

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

    optimizer = build_optimizer(model, train_cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    start_step, tokens_seen = 0, 0
    ckpt_path = args.out_dir / "ckpt.pt"
    if args.resume:
        if args.hub_repo:
            if is_master:
                pull_from_hub(args.hub_repo, "checkpoints/ckpt.pt", ckpt_path)
                # Restore logs so a resumed session appends instead of replacing them.
                pull_from_hub(args.hub_repo, "logs/train_log.csv", args.out_dir / "train_log.csv")
                pull_from_hub(args.hub_repo, "logs/val_log.csv", args.out_dir / "val_log.csv")
            if world > 1:
                # Wait for rank0 to finish writing before any rank reads.
                if torch.cuda.is_available():
                    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
                else:
                    torch.distributed.barrier()
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

    bytes_per_token = None
    if train_cfg.get("eval_bpb") and val_ds is not None and is_master:
        import tiktoken

        encoder = tiktoken.get_encoding("gpt2")
        bytes_per_token = compute_bytes_per_token(val_ds, encoder)
        print(f"eval: {bytes_per_token:.3f} bytes/token (bpB enabled)")

    ema_state = None
    if train_cfg.get("ema_decay", 0.0) > 0:
        raw = model.module if hasattr(model, "module") else model
        ema_state = {k: v.detach().clone().float() for k, v in raw.state_dict().items()}
        if is_master:
            print(f"EMA enabled (decay {train_cfg['ema_decay']})")

    log_path = args.out_dir / "train_log.csv"
    val_log_path = args.out_dir / "val_log.csv"
    if is_master:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if not log_path.exists() or start_step == 0:
            with open(log_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["step", "loss", "lr", "tokens", "elapsed_s"])
        if not val_log_path.exists() or start_step == 0:
            with open(val_log_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["step", "val_loss", "bpb"])

    tokens_per_step = (
        train_cfg["micro_batch_size"] * train_cfg["grad_accum_steps"] * model_cfg.context_len * world
    )
    start_time = time.time()
    last_ckpt_time = start_time
    model.train()

    step = start_step
    decay_start = None
    decay_stop_scheduled = False
    hard_max_steps = train_cfg["max_steps"]
    try:
        while step < hard_max_steps:
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
            scale = lr_scale(step, train_cfg, decay_start)
            lr = train_cfg["learning_rate"] * scale
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
            scaler.step(optimizer)
            scaler.update()

            if ema_state is not None:
                raw = model.module if hasattr(model, "module") else model
                update_ema(ema_state, raw.state_dict(), train_cfg["ema_decay"])

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
                target = train_cfg.get("target_val_loss")
                if is_master:
                    val_loss, val_bpb = evaluate(
                        model, val_ds, train_cfg["eval_steps"], bytes_per_token
                    )
                    msg = f"step {step} | val loss {val_loss:.4f}"
                    if val_bpb is not None:
                        msg += f" | bpB {val_bpb:.4f}"
                    print(msg)
                    row = [step, f"{val_loss:.4f}"]
                    if val_bpb is not None:
                        row.append(f"{val_bpb:.4f}")
                    with open(val_log_path, "a", newline="") as fh:
                        csv.writer(fh).writerow(row)
                    if target is not None and decay_start is None and val_loss <= target:
                        decay_start = step
                        print(f"target val loss {target} reached at step {step}")
                if target is not None and world > 1:
                    flag = torch.tensor(
                        [decay_start if decay_start is not None else -1],
                        device=device,
                        dtype=torch.long,
                    )
                    torch.distributed.broadcast(flag, src=0)
                    if decay_start is None and int(flag.item()) >= 0:
                        decay_start = int(flag.item())
                if target is not None and decay_start is not None and not decay_stop_scheduled:
                    decay_stop_scheduled = True
                    extra = int(train_cfg.get("decay_steps_after_target", 0))
                    hard_max_steps = min(train_cfg["max_steps"], decay_start + extra)
                    if is_master:
                        print(f"target reached: decaying {extra} steps, then stopping")

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
                    push_to_hub(val_log_path, args.hub_repo, "logs/val_log.csv")
                last_ckpt_time = time.time()

        if is_master:
            save_checkpoint(ckpt_path, model, optimizer, step, tokens_seen, config)
            print(f"final checkpoint saved at step {step}")
            if args.hub_repo:
                push_to_hub(ckpt_path, args.hub_repo, "checkpoints/ckpt.pt")
                push_to_hub(log_path, args.hub_repo, "logs/train_log.csv")
                push_to_hub(val_log_path, args.hub_repo, "logs/val_log.csv")

            if ema_state is not None and val_ds is not None:
                raw = model.module if hasattr(model, "module") else model
                dtype = next(raw.parameters()).dtype
                backup = {k: v.detach().clone() for k, v in raw.state_dict().items()}
                raw.load_state_dict({k: v.to(dtype) for k, v in ema_state.items()})
                ema_loss, ema_bpb = evaluate(model, val_ds, train_cfg["eval_steps"], bytes_per_token)
                msg = f"EMA val loss {ema_loss:.4f}"
                if ema_bpb is not None:
                    msg += f" | bpB {ema_bpb:.4f}"
                print(msg)
                ema_path = args.out_dir / "ckpt_ema.pt"
                torch.save(
                    {"model": raw.state_dict(), "step": step, "tokens": tokens_seen, "config": config},
                    ema_path,
                )
                raw.load_state_dict(backup)
                print(f"EMA checkpoint saved to {ema_path}")
                if args.hub_repo:
                    push_to_hub(ema_path, args.hub_repo, "checkpoints/ckpt_ema.pt")
    finally:
        if world > 1:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
