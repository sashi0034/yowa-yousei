#!/usr/bin/env python3
"""Print metadata stored inside training checkpoints without generating text."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


DEFAULT_CHECKPOINTS = [
    Path("checkpoints/best.pt"),
    Path("checkpoints/latest.pt"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print step, losses, model config, and training args saved inside "
            "checkpoints produced by train.py."
        )
    )
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        help=(
            "Checkpoint files to inspect. "
            "Defaults to checkpoints/best.pt and checkpoints/latest.pt."
        ),
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Also print the saved model_config dictionary.",
    )
    parser.add_argument(
        "--show-args",
        action="store_true",
        help="Also print the saved training args dictionary.",
    )
    return parser.parse_args()


def resolve_checkpoint_paths(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    return [path for path in DEFAULT_CHECKPOINTS if path.exists()] or list(
        DEFAULT_CHECKPOINTS
    )


def format_mtime(timestamp: float) -> str:
    if timestamp <= 0:
        return "-"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def format_size(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def format_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def print_summary_row(path: Path, checkpoint: dict[str, Any] | None) -> None:
    if checkpoint is None:
        print(f"{path}  (missing or unreadable)")
        return
    stat = path.stat()
    step = checkpoint.get("step", "-")
    train_loss = format_float(checkpoint.get("train_loss"))
    val_loss = format_float(checkpoint.get("val_loss"))
    best_val_loss = format_float(checkpoint.get("best_val_loss"))
    print(
        f"{str(path):<30} "
        f"step {str(step):>7} | "
        f"train {train_loss:>8} | "
        f"val {val_loss:>8} | "
        f"best_val {best_val_loss:>8} | "
        f"{format_size(stat.st_size):>10} | "
        f"{format_mtime(stat.st_mtime)}"
    )


def print_details(
    checkpoint: dict[str, Any], show_config: bool, show_args: bool
) -> None:
    if show_config and "model_config" in checkpoint:
        print("  model_config:")
        for key, value in checkpoint["model_config"].items():
            print(f"    {key}: {value}")
    if show_args and "args" in checkpoint:
        print("  args:")
        for key, value in checkpoint["args"].items():
            print(f"    {key}: {value}")


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    args = parse_args()
    paths = resolve_checkpoint_paths(args.checkpoints)

    print(
        f"{'path':<30} "
        f"{'step':>13} | "
        f"{'train':>14} | "
        f"{'val':>12} | "
        f"{'best_val':>17} | "
        f"{'size':>10} | "
        f"mtime"
    )
    print("-" * 130)

    missing = False
    for path in paths:
        checkpoint = load_checkpoint(path)
        print_summary_row(path, checkpoint)
        if checkpoint is None:
            missing = True
            continue
        if args.show_config or args.show_args:
            print_details(checkpoint, args.show_config, args.show_args)

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
