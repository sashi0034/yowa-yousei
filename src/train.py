#!/usr/bin/env python3
"""Train the small GPT model on prepared token-id files."""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from dataset import BatchConfig, TokenBatchLoader, resolve_device
from model import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_N_EMBD,
    DEFAULT_N_HEAD,
    DEFAULT_N_LAYER,
    DEFAULT_VOCAB_SIZE,
    GPT,
    GPTConfig,
)


DEFAULT_TRAIN_BIN_PATH = Path("data/processed/train.bin")
DEFAULT_VAL_BIN_PATH = Path("data/processed/val.bin")
DEFAULT_OUT_DIR = Path("checkpoints")
DEFAULT_MAX_STEPS = 50000
DEFAULT_EVAL_INTERVAL = 500
DEFAULT_EVAL_ITERS = 100
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_MIN_LR = 3e-5
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_BATCH_SIZE = 8
DEFAULT_LOG_INTERVAL = 10


def configure_stdio() -> None:
   # Ensure logs appear immediately, even when stdout/stderr are redirected.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True, write_through=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT causal language model.")
    parser.add_argument("--train-bin", type=Path, default=DEFAULT_TRAIN_BIN_PATH)
    parser.add_argument("--val-bin", type=Path, default=DEFAULT_VAL_BIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--n-layer", type=int, default=DEFAULT_N_LAYER)
    parser.add_argument("--n-head", type=int, default=DEFAULT_N_HEAD)
    parser.add_argument("--n-embd", type=int, default=DEFAULT_N_EMBD)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)

    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--eval-iters", type=int, default=DEFAULT_EVAL_ITERS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_MIN_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, cuda, or a device like cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
        help="AMP dtype. auto uses bf16 on supported CUDA GPUs, else fp16 on CUDA.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = [
        "vocab_size",
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
        "max_steps",
        "eval_interval",
        "eval_iters",
        "gradient_accumulation_steps",
        "batch_size",
        "log_interval",
    ]
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")

    if args.n_embd % args.n_head != 0:
        raise SystemExit("--n-embd must be divisible by --n-head")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    if args.min_lr < 0:
        raise SystemExit("--min-lr must be zero or positive")
    if args.dropout < 0 or args.dropout >= 1:
        raise SystemExit("--dropout must be in [0, 1)")
    if args.grad_clip < 0:
        raise SystemExit("--grad-clip must be zero or positive")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be zero or positive")


def resolve_amp_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def autocast_context(device: torch.device, dtype: torch.dtype) -> Any:
    if device.type == "cuda" and dtype != torch.float32:
        return torch.amp.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def get_lr(step: int, args: argparse.Namespace) -> float:
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return args.learning_rate * (step + 1) / args.warmup_steps
    if step >= args.max_steps:
        return args.min_lr

    decay_steps = max(1, args.max_steps - args.warmup_steps)
    decay_step = max(0, step - args.warmup_steps)
    decay_ratio = min(1.0, decay_step / decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@torch.no_grad()
def estimate_loss(
    model: GPT,
    loader: TokenBatchLoader,
    eval_iters: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split in ["train", "val"]:
        losses = torch.empty(eval_iters)
        for idx in range(eval_iters):
            x, y = loader.get_batch(split)
            with autocast_context(device, amp_dtype):
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("loss was not computed")
            losses[idx] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model_config: GPTConfig,
    args: argparse.Namespace,
    step: int,
    best_val_loss: float,
    losses: dict[str, float],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": asdict(model_config),
        "args": vars(args),
        "step": step,
        "best_val_loss": best_val_loss,
        "train_loss": losses["train"],
        "val_loss": losses["val"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main() -> None:
    configure_stdio()

    args = parse_args()
    validate_args(args)

    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.dtype, device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    loader = TokenBatchLoader(
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        config=BatchConfig(
            block_size=args.block_size,
            batch_size=args.batch_size,
            device=str(device),
        ),
        seed=args.seed,
    )
    model_config = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=(device.type == "cuda" and amp_dtype == torch.float16),
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"amp dtype: {amp_dtype}")
    print(f"parameters: {param_count:,}")
    print(f"train tokens: {len(loader.train_tokens):,}")
    print(f"val tokens: {len(loader.val_tokens):,}")

    best_val_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)
    step_start_time = time.time()

    for step in range(1, args.max_steps + 1):
        lr = get_lr(step - 1, args)
        set_optimizer_lr(optimizer, lr)

        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            x, y = loader.get_batch("train")
            with autocast_context(device, amp_dtype):
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("loss was not computed")

            accumulated_loss += loss.item()
            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == 1:
            elapsed = time.time() - step_start_time
            step_start_time = time.time()
            avg_loss = accumulated_loss / args.gradient_accumulation_steps
            print(
                f"step {step:6d} | train loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | {elapsed:.2f}s"
            )

        should_eval = step % args.eval_interval == 0 or step == args.max_steps
        if should_eval:
            losses = estimate_loss(model, loader, args.eval_iters, device, amp_dtype)
            print(
                f"eval {step:6d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}"
            )

            is_best = losses["val"] < best_val_loss
            if is_best:
                best_val_loss = losses["val"]
                save_checkpoint(
                    args.out_dir / "best.pt",
                    model,
                    optimizer,
                    scaler,
                    model_config,
                    args,
                    step,
                    best_val_loss,
                    losses,
                )
                print(f"saved checkpoint: {args.out_dir / 'best.pt'}")

            save_checkpoint(
                args.out_dir / "latest.pt",
                model,
                optimizer,
                scaler,
                model_config,
                args,
                step,
                best_val_loss,
                losses,
            )
            print(f"saved checkpoint: {args.out_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
