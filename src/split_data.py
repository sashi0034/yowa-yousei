#!/usr/bin/env python3
"""Split cleaned chapter text into train/val files.

The cleaned corpus can be large, so this script streams chapter blocks instead
of loading data/processed/clean.txt into memory. Blocks are expected to end with
the literal line "<eos>".
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


DEFAULT_INPUT_PATH = Path("data/processed/clean.txt")
DEFAULT_TRAIN_PATH = Path("data/processed/train.txt")
DEFAULT_VAL_PATH = Path("data/processed/val.txt")
DEFAULT_TRAIN_SMALL_PATH = Path("data/processed/train_small.txt")
DEFAULT_VAL_SMALL_PATH = Path("data/processed/val_small.txt")


@dataclass
class SplitStats:
    train_blocks: int = 0
    val_blocks: int = 0
    train_bytes: int = 0
    val_bytes: int = 0
    train_small_blocks: int = 0
    val_small_blocks: int = 0
    train_small_bytes: int = 0
    val_small_bytes: int = 0


def iter_blocks(path: Path) -> Iterator[str]:
    block_lines: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for line in file:
            if not block_lines and not line.strip():
                continue
            block_lines.append(line)
            if line.strip() == "<eos>":
                yield finish_block(block_lines)
                block_lines = []

    if block_lines:
        yield finish_block(block_lines)


def finish_block(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    return "".join(lines).rstrip("\n") + "\n\n"


def should_use_val(block: str, val_ratio: float, seed: int) -> bool:
    digest = hashlib.blake2b(
        block.encode("utf-8"),
        digest_size=8,
        person=b"split-v1",
        key=seed.to_bytes(8, "little", signed=False),
    ).digest()
    bucket = int.from_bytes(digest, "big") / float(1 << 64)
    return bucket < val_ratio


def write_block(file: TextIO, block: str) -> int:
    file.write(block)
    return len(block.encode("utf-8"))


def maybe_write_small(file: TextIO, block: str, current_bytes: int, limit: int) -> int:
    if limit <= 0 or current_bytes >= limit:
        return 0
    return write_block(file, block)


def split_data(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    train_small_path: Path,
    val_small_path: Path,
    val_ratio: float,
    seed: int,
    train_small_bytes: int,
    val_small_bytes: int,
) -> SplitStats:
    if not input_path.exists():
        raise SystemExit(f"input does not exist: {input_path}")
    if not 0.0 < val_ratio < 1.0:
        raise SystemExit("--val-ratio must be between 0 and 1")

    for path in [train_path, val_path, train_small_path, val_small_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    stats = SplitStats()
    with (
        train_path.open("w", encoding="utf-8", newline="\n") as train_file,
        val_path.open("w", encoding="utf-8", newline="\n") as val_file,
        train_small_path.open("w", encoding="utf-8", newline="\n") as train_small_file,
        val_small_path.open("w", encoding="utf-8", newline="\n") as val_small_file,
    ):
        for index, block in enumerate(iter_blocks(input_path), start=1):
            if not block:
                continue

            if should_use_val(block, val_ratio, seed):
                block_bytes = write_block(val_file, block)
                stats.val_blocks += 1
                stats.val_bytes += block_bytes
                small_bytes = maybe_write_small(
                    val_small_file, block, stats.val_small_bytes, val_small_bytes
                )
                if small_bytes:
                    stats.val_small_blocks += 1
                    stats.val_small_bytes += small_bytes
            else:
                block_bytes = write_block(train_file, block)
                stats.train_blocks += 1
                stats.train_bytes += block_bytes
                small_bytes = maybe_write_small(
                    train_small_file, block, stats.train_small_bytes, train_small_bytes
                )
                if small_bytes:
                    stats.train_small_blocks += 1
                    stats.train_small_bytes += small_bytes

            if index % 10000 == 0:
                print(f"processed {index} blocks", flush=True)

    if stats.train_blocks == 0 or stats.val_blocks == 0:
        raise SystemExit(
            "split produced an empty train or val file; try a larger corpus or val ratio"
        )
    return stats


def format_size(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split data/processed/clean.txt into train/val text files."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--train-small", type=Path, default=DEFAULT_TRAIN_SMALL_PATH)
    parser.add_argument("--val-small", type=Path, default=DEFAULT_VAL_SMALL_PATH)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.01,
        help="Approximate validation split ratio by chapter block.",
    )
    parser.add_argument("--seed", type=int, default=20240527)
    parser.add_argument(
        "--train-small-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="Stop writing train_small.txt after roughly this many bytes.",
    )
    parser.add_argument(
        "--val-small-bytes",
        type=int,
        default=1 * 1024 * 1024,
        help="Stop writing val_small.txt after roughly this many bytes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = split_data(
        input_path=args.input,
        train_path=args.train,
        val_path=args.val,
        train_small_path=args.train_small,
        val_small_path=args.val_small,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_small_bytes=args.train_small_bytes,
        val_small_bytes=args.val_small_bytes,
    )

    total_bytes = stats.train_bytes + stats.val_bytes
    actual_val_ratio = stats.val_bytes / total_bytes if total_bytes else 0.0
    print()
    print(f"train: {stats.train_blocks} blocks, {format_size(stats.train_bytes)}")
    print(f"val: {stats.val_blocks} blocks, {format_size(stats.val_bytes)}")
    print(f"actual val ratio by bytes: {actual_val_ratio:.4%}")
    print(
        "train_small: "
        f"{stats.train_small_blocks} blocks, {format_size(stats.train_small_bytes)}"
    )
    print(
        "val_small: "
        f"{stats.val_small_blocks} blocks, {format_size(stats.val_small_bytes)}"
    )


if __name__ == "__main__":
    main()
