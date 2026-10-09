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

import json

from data import MixtureDataset, TokenDataset
from mixture import SourceMixtureDataset, per_source_loss, strategy_counts
from model import GPT, GPTConfig
from optim import MuonWithAuxAdam, split_muon_params


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    p.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt or --hub-repo")
    p.add_argument("--resume-from", type=str, default=None,
                   help="resume from a local ckpt.pt file, a directory containing one, "
                        "or an HF repo id (implies --resume; repo pull needs HF_TOKEN)")
    p.add_argument("--hub-repo", type=str, default=None, help="HF repo id for checkpoint sync")
    p.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    p.add_argument("--max-minutes", type=float, default=None, help="stop after this many minutes")
    p.add_argument("--target-val-loss", type=float, default=None, help="decay and stop once reached")
    p.add_argument("--decay-steps-after-target", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--doc-mask",
        action="store_true",
        help="block attention and loss across document boundaries (EOS-delimited)",
    )
    p.add_argument(
        "--replay-bin",
        type=Path,
        default=None,
        help="frozen replay shard for continual learning (CL-1); requires replay_prob > 0",
    )
    # Kautilya arm (v0.3.0 Step 4): per-source bins sampled at runtime weights,
    # instead of the offline pre-blend produced by mix_bins.py.
    p.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="source names, e.g. fineweb tinystories cosmopedia python",
    )
    p.add_argument(
        "--source-bins",
        nargs="*",
        type=Path,
        default=None,
        help="one .bin per source, in the same order as --sources",
    )
    p.add_argument(
        "--source-weights",
        default=None,
        help='JSON object of source weights, e.g. {"fineweb":0.6,"python":0.05}',
    )
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
    # Config-driven so a micro-ablation arm can enable it without ablate.py
    # needing to know about the flag; --doc-mask forces it on for one-off runs.
    doc_mask = bool(args.doc_mask or train_cfg.get("doc_mask", False))
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
        print(f"doc_mask: {'on' if doc_mask else 'off'}")

    optimizer = build_optimizer(model, train_cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    start_step, tokens_seen = 0, 0
    # Where the LR schedule's step 0 sits. A fresh run starts at 0; a resumed
    # one rebases so the update gets its own warmup+decay. Bound here rather
    # than inside the resume branch: the first version put the else next to
    # `if ckpt_path.exists()` but it bound to the inner `if rewarm`, so a fresh
    # run reached the training loop with lr_origin unbound and 10 smoke tests
    # failed. The full suite is what caught it.
    lr_origin = 0
    ckpt_path = args.out_dir / "ckpt.pt"
    if args.resume_from and not args.resume:
        src = Path(args.resume_from)
        if src.exists():
            import shutil
            if src.is_dir():
                src = src / "ckpt.pt"
            assert src.exists(), f"resume checkpoint not found: {src}"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, ckpt_path)
            if is_master:
                print(f"resume: staged local checkpoint {src} -> {ckpt_path}")
            args.resume = True
        else:
            args.hub_repo = args.hub_repo or args.resume_from
            args.resume = True
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
            # Schedule origin, for continual updates.
            #
            # A resumed run must NOT continue the source run's schedule. Resuming
            # v0.2.0 at step 4770 of 4770 makes base_lr_scale return min_lr_ratio
            # for every remaining step, so the LR sits at its decayed floor
            # forever -- which is exactly the condition Ibrahim et al. (2403.08763)
            # identify as the worst case for continual adaptation, and exactly what
            # the re-warm arm in Step 2 exists to test. Without this, the re-warm
            # arm would silently be a no-re-warm arm and the experiment would report
            # a null for a mechanism it never ran.
            #
            # `lr_step_origin` is where the new schedule's step 0 sits. The weights
            # resume; the clock does not. tokens_seen still accumulates across the
            # boundary so data accounting stays continuous.
            # `rewarm` decides the schedule origin, and it defaults to False.
            #
            # Two bugs met here, both of which would have made Step 2 compare two
            # identical arms and report a meaningless null:
            #
            # 1. The origin was rebased unconditionally, so the "no re-warm"
            #    control also restarted its warmup and climbed back to the same
            #    peak (measured: both arms peak at 0.00060).
            # 2. `rewarm` defaulted to True, so a plain `--resume` with no such key
            #    silently changed the schedule of every resumed run in the project.
            #    Resuming v0.2.0 mid-run used to continue its cosine; now it would
            #    have restarted it. That is a behaviour change to historical runs
            #    and must be opt-in.
            if train_cfg.get("rewarm", False):
                lr_origin = start_step
                if is_master:
                    print("  LR schedule: re-warm from step 0 of the update budget")
            else:
                lr_origin = 0
                if is_master:
                    print(
                        f"  LR schedule: continue the source run's cosine from "
                        f"step {start_step:,} (rewarm disabled)"
                    )

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()]
        )

    # CL-1: replay keeps a fraction of each batch on the *original* corpus, which is
    # what prevents a continual run from eroding what it learned before. The shard
    # must be frozen data, not a slice of whatever is new today.
    # Kautilya: how often to re-estimate source weights. 0 disables reweighting,
    # leaving a *static* per-source mixture -- which is the registered control
    # arm, so the same code path serves both and only the cadence differs.
    adaptive_every = int(train_cfg.get("adaptive_every", 0))
    replay_prob = float(train_cfg.get("replay_prob", 0.0))
    if replay_prob > 0 and args.replay_bin is None:
        raise SystemExit("replay_prob > 0 requires --replay-bin pointing at a frozen shard")

    rev_cfg = config.get("reversal") or {}
    # The base dataset is only needed for the plain and replay paths. Constructing
    # it unconditionally meant a per-source run still required a train.bin that it
    # never reads -- so the arm could not be run against per-source shards alone,
    # which is the whole point of the arm. Decide first, then build.
    per_source_mode = bool(args.sources or args.source_bins or args.source_weights)

    def _build_base() -> TokenDataset:
        return TokenDataset(
            args.data_dir / "train.bin",
            model_cfg.context_len,
            reversal_prob=float(rev_cfg.get("prob", 0.0)),
            reversal_min_chunk=int(rev_cfg.get("min_chunk", 4)),
            reversal_max_chunk=int(rev_cfg.get("max_chunk", 16)),
        )

    base_train_ds = None if (per_source_mode and replay_prob <= 0) else _build_base()
    if replay_prob > 0:
        replay_ds = TokenDataset(args.replay_bin, model_cfg.context_len)
        train_ds = MixtureDataset(base_train_ds, replay_ds, replay_prob=replay_prob)
        if is_master:
            print(f"replay: {replay_prob:.0%} of each batch from {args.replay_bin}")
    elif per_source_mode:
        # Guard the parallel lists before zip(), which would silently truncate to
        # the shorter one and drop a source without saying so. A dropped source
        # changes the mixture, which is the one thing this arm is measuring.
        if not (args.sources and args.source_bins and args.source_weights):
            missing = [
                name
                for name, val in (
                    ("--sources", args.sources),
                    ("--source-bins", args.source_bins),
                    ("--source-weights", args.source_weights),
                )
                if not val
            ]
            raise SystemExit(
                f"Kautilya per-source training needs all three of --sources, "
                f"--source-bins, --source-weights; missing {missing}"
            )
        if len(args.sources) != len(args.source_bins):
            raise SystemExit(
                f"--sources has {len(args.sources)} entries but --source-bins has "
                f"{len(args.source_bins)}; they must correspond one-to-one"
            )
        # Kautilya arm (v0.3.0 Step 4): sample per source at runtime weights instead
        # of consuming the offline pre-blend from mix_bins.py. Mixing these two is
        # the mistake the pre-registration warns about -- an offline re-blend and a
        # runtime reweight are different experiments, and a run that silently did
        # both would be uninterpretable.
        source_weights = json.loads(args.source_weights or "{}")
        unknown = set(source_weights) - set(args.sources)
        if unknown:
            raise SystemExit(f"--source-weights names sources not in --sources: {sorted(unknown)}")
        missing = [n for n in args.sources if n not in source_weights]
        if missing:
            raise SystemExit(f"--source-weights is missing weights for: {sorted(missing)}")
        src_datasets = {
            name: TokenDataset(path, model_cfg.context_len)
            for name, path in zip(args.sources, args.source_bins)
        }
        train_ds = SourceMixtureDataset(
            sources=src_datasets,
            weights={n: float(source_weights[n]) for n in args.sources},
            seed=int(train_cfg["seed"]) + rank,
        )
        if is_master:
            probs = train_ds.probabilities()
            pretty = ", ".join(f"{n} {p:.1%}" for n, p in sorted(probs.items()))
            print(f"per-source mixture (Kautilya arm): {pretty}")
            if adaptive_every > 0:
                print(f"  reweighting every {adaptive_every} steps from held-out per-source loss")
    else:
        train_ds = base_train_ds
    if is_master and base_train_ds is not None and base_train_ds.reversal_prob > 0:
        print(
            f"reversal: prob {train_ds.reversal_prob} | chunks "
            f"{train_ds.reversal_min_chunk}-{train_ds.reversal_max_chunk}"
        )
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
    per_source_path = args.out_dir / "per_source_log.csv"
    if is_master:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if not log_path.exists() or start_step == 0:
            with open(log_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["step", "loss", "lr", "tokens", "elapsed_s"])
        if not val_log_path.exists() or start_step == 0:
            with open(val_log_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["step", "val_loss", "bpb"])
        if adaptive_every > 0 and isinstance(train_ds, SourceMixtureDataset):
            if not per_source_path.exists() or start_step == 0:
                names = train_ds.source_names
                with open(per_source_path, "w", newline="") as fh:
                    csv.writer(fh).writerow(
                        ["step"]
                        + [f"loss_{n}" for n in names]
                        + [f"weight_{n}" for n in names]
                        + [f"strategy_{n}" for n in names]
                    )

    tokens_per_step = (
        train_cfg["micro_batch_size"] * train_cfg["grad_accum_steps"] * model_cfg.context_len * world
    )
    start_time = time.time()
    last_ckpt_time = start_time
    model.train()

    step = start_step
    decay_start = None
    decay_stop_scheduled = False
    # The step budget is measured from the *update's* origin, not from zero.
    #
    # A continual update resumes a finished checkpoint: v0.2.0 ends at step 4770
    # and the update config asks for max_steps 500. Interpreting that as "stop at
    # absolute step 500" means the loop condition `step < hard_max_steps` is false
    # on entry and the update silently trains for zero steps while still printing
    # "re-warm from step 0 of the update budget". Found by running a real resume,
    # not by reading the code -- the schedule rebasing looked correct in isolation.
    #
    # update_budget semantics, explicit and opt-in via `rewarm`: the run performs
    # max_steps steps *of update*, counted from lr_origin. Without `rewarm` the old
    # absolute behaviour is kept, because every historical run in this project
    # used max_steps as an absolute ceiling and silently changing that would alter
    # how v0.2.0 and its ablations were run.
    # Step budget, absolute or relative to the resume point.
    #
    # `--max-steps N` keeps its historical meaning: stop at absolute step N. The
    # smoke test asserts it (resume from 10 with --max-steps 20 ends at 20), and
    # changing it would silently alter how every resumed run in this project
    # behaves -- including runs already recorded.
    #
    # A continual update needs the opposite: "train N more steps". That is
    # `update_budget`, a config key, set on the v0.3.0 arms. Without it a resumed
    # v0.2.0 run with max_steps 500 trains zero steps, because step starts at 4770
    # and `step < hard_max_steps` is false on entry -- while still printing
    # "re-warm from step 0 of the update budget". Found by running a real resume.
    #
    # Two bugs met at the previous attempt and are both avoided here: the budget
    # was not gated on `rewarm` (so the no-re-warm control trained nothing, and
    # Step 2 would have compared one arm that moved against one that did not), and
    # the schedule origin was rebased unconditionally (so the control re-warmed
    # too, both arms peaking at 0.00060).
    update_budget = train_cfg.get("update_budget")
    if update_budget and start_step:
        hard_max_steps = start_step + int(update_budget)
        if is_master:
            print(
                f"update budget: {int(update_budget):,} steps from step {start_step:,}"
                f" -> hard stop {hard_max_steps:,}"
            )
    else:
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
                batch = train_ds.get_batch(
                    train_cfg["micro_batch_size"],
                    device,
                    generator,
                    return_masks=doc_mask,
                )
                if doc_mask:
                    x, y, attn_mask, loss_mask = batch
                else:
                    x, y = batch
                    attn_mask = loss_mask = None
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=is_cuda):
                    _, loss = model(x, y, attn_mask=attn_mask, loss_mask=loss_mask)
                    loss = loss / train_cfg["grad_accum_steps"]
                scaler.scale(loss).backward()
                loss_total += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            scale = lr_scale(step - lr_origin, train_cfg, decay_start)
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

            # Kautilya reweighting. Runs on the eval boundary so the per-source
            # loss and the policy decision land in the same place as the val
            # curve, and so the cost is amortised rather than paid every step.
            #
            # Only master reweights, and the resulting weights are broadcast, so
            # every rank samples the same mixture. Reweighting per-rank would make
            # ranks disagree about the data they are training on.
            if (
                adaptive_every > 0
                and isinstance(train_ds, SourceMixtureDataset)
                and step > 0
                and step % adaptive_every == 0
            ):
                if is_master:
                    losses = per_source_loss(
                        model, train_ds, device, generator=generator,
                        batches=int(train_cfg.get("adaptive_batches", 4)),
                        batch_size=int(train_cfg.get("adaptive_batch_size", 16)),
                    )
                    applied = train_ds.update_weights_from_signal(losses)
                    counts = strategy_counts(applied)
                    if per_source_path is not None:
                        with open(per_source_path, "a", newline="") as fh:
                            cw = csv.writer(fh)
                            cw.writerow(
                                [step]
                                + [f"{losses[n]:.4f}" for n in train_ds.source_names]
                                + [f"{train_ds.probabilities()[n]:.4f}" for n in train_ds.source_names]
                                + [applied[n] for n in train_ds.source_names]
                            )
                    probs = train_ds.probabilities()
                    detail = " ".join(f"{n}={probs[n]:.3f}" for n in train_ds.source_names)
                    strat = " ".join(f"{k}:{v}" for k, v in counts.items() if v)
                    print(f"step {step} | per-source loss " +
                          " ".join(f"{n}={losses[n]:.3f}" for n in train_ds.source_names))
                    print(f"step {step} | weights {detail} | {strat}")
                # Broadcast the *post-update* weights. The first real run (X26)
                # sent the pre-update tensor, so every non-master rank sampled a
                # stale mixture until the next reweight -- the one thing the
                # broadcast exists to prevent. set_probabilities stores it on
                # CPU because the sampler draws with a CPU generator, and the
                # barrier keeps ranks from entering the next step's collectives
                # while the master is still finishing the reweight.
                if world > 1:
                    probs_t = torch.tensor(
                        [train_ds.probabilities()[n] for n in train_ds.source_names],
                        device=device,
                    )
                    torch.distributed.broadcast(probs_t, src=0)
                    if not is_master:
                        train_ds.set_probabilities(probs_t)
                    torch.distributed.barrier()

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
