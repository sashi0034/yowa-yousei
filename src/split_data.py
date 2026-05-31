#!/usr/bin/env python3
"""Split cleaned work text into train/val files.

The cleaned corpus can be large, so this script streams work blocks instead
of loading data/processed/clean.txt into memory. Blocks are expected to end with
the literal line "<eos>".
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

from corpus_markers import CHAPTER_SEPARATOR, EOS_MARKER


DEFAULT_INPUT_PATH = Path("data/processed/clean.txt")
DEFAULT_TRAIN_PATH = Path("data/processed/train.txt")
DEFAULT_VAL_PATH = Path("data/processed/val.txt")
DEFAULT_TRAIN_SMALL_PATH = Path("data/processed/train_small.txt")
DEFAULT_VAL_SMALL_PATH = Path("data/processed/val_small.txt")

MIN_VAL_PARTS_PER_WORK = 2
FALLBACK_LINES_PER_PART = 80


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
            if line.strip() == EOS_MARKER:
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


def finish_part(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip("\r\n") for line in lines)


def block_body(block: str) -> str:
    lines: list[str] = []
    for line in block.splitlines():
        if line.strip() == EOS_MARKER:
            break
        lines.append(line)
    return finish_part(lines)


def split_work_parts(block: str) -> list[str]:
    parts: list[str] = []
    current_lines: list[str] = []
    for line in block_body(block).splitlines():
        if line.strip() == CHAPTER_SEPARATOR:
            part = finish_part(current_lines)
            if part:
                parts.append(part)
            current_lines = []
            continue
        current_lines.append(line)

    part = finish_part(current_lines)
    if part:
        parts.append(part)

    if len(parts) > 1:
        return parts

    lines = [line for line in block_body(block).splitlines() if line.strip()]
    if len(lines) <= FALLBACK_LINES_PER_PART:
        return [finish_part(lines)] if lines else []
    return [
        finish_part(lines[index : index + FALLBACK_LINES_PER_PART])
        for index in range(0, len(lines), FALLBACK_LINES_PER_PART)
    ]


def should_use_val(block: str, val_ratio: float, seed: int) -> bool:
    digest = hashlib.blake2b(
        block.encode("utf-8"),
        digest_size=8,
        person=b"split-v1",
        key=seed.to_bytes(8, "little", signed=False),
    ).digest()
    bucket = int.from_bytes(digest, "big") / float(1 << 64)
    return bucket < val_ratio


def stable_hash_int(text: str, seed: int) -> int:
    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
        person=b"split-v2",
        key=seed.to_bytes(8, "little", signed=False),
    ).digest()
    return int.from_bytes(digest, "big")


def val_part_count(part_count: int, val_ratio: float) -> int:
    if part_count <= 1:
        return part_count

    min_parts = 1
    if part_count >= MIN_VAL_PARTS_PER_WORK * 4:
        min_parts = MIN_VAL_PARTS_PER_WORK
    requested = max(
        1,
        math.ceil(part_count * val_ratio),
        min_parts,
    )
    return min(requested, max(1, part_count // 2))


def select_val_indices(parts: list[str], val_ratio: float, seed: int) -> set[int]:
    count = val_part_count(len(parts), val_ratio)
    if count <= 0:
        return set()
    if count >= len(parts):
        return set(range(len(parts)))

    selected: set[int] = set()
    for slot in range(count):
        start = slot * len(parts) // count
        end = (slot + 1) * len(parts) // count
        candidates = range(start, max(start + 1, end))
        selected.add(
            min(
                candidates,
                key=lambda index: stable_hash_int(
                    f"{slot}:{index}:{parts[index]}", seed
                ),
            )
        )
    return selected


def render_document(parts: list[str]) -> str:
    body = f"\n\n{CHAPTER_SEPARATOR}\n\n".join(
        part.strip() for part in parts if part.strip()
    )
    if not body:
        return ""
    return f"{body}\n{EOS_MARKER}\n\n"


def iter_runs(parts: list[str], selected_indices: set[int]) -> Iterator[list[str]]:
    run: list[str] = []
    for index, part in enumerate(parts):
        if index in selected_indices:
            run.append(part)
            continue
        if run:
            yield run
            run = []
    if run:
        yield run


def write_block(file: TextIO, block: str) -> int:
    file.write(block)
    return len(block.encode("utf-8"))


def maybe_write_small(file: TextIO, block: str, current_bytes: int, limit: int) -> int:
    if limit <= 0 or current_bytes >= limit:
        return 0
    return write_block(file, block)


def write_train_block(
    train_file: TextIO,
    train_small_file: TextIO,
    stats: SplitStats,
    block: str,
    train_small_bytes: int,
) -> None:
    block_bytes = write_block(train_file, block)
    stats.train_blocks += 1
    stats.train_bytes += block_bytes
    small_bytes = maybe_write_small(
        train_small_file, block, stats.train_small_bytes, train_small_bytes
    )
    if small_bytes:
        stats.train_small_blocks += 1
        stats.train_small_bytes += small_bytes


def write_val_block(
    val_file: TextIO,
    val_small_file: TextIO,
    stats: SplitStats,
    block: str,
    val_small_bytes: int,
) -> None:
    block_bytes = write_block(val_file, block)
    stats.val_blocks += 1
    stats.val_bytes += block_bytes
    small_bytes = maybe_write_small(
        val_small_file, block, stats.val_small_bytes, val_small_bytes
    )
    if small_bytes:
        stats.val_small_blocks += 1
        stats.val_small_bytes += small_bytes


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

            # Previous behavior: put the whole work in either train or val.
            # if should_use_val(block, val_ratio, seed):
            #     write_val_block(val_file, val_small_file, stats, block, val_small_bytes)
            # else:
            #     write_train_block(
            #         train_file, train_small_file, stats, block, train_small_bytes
            #     )

            # Current behavior: split each work and sample val parts from within it.
            parts = split_work_parts(block)
            if len(parts) <= 1:
                rendered_block = render_document(parts)
                if not rendered_block:
                    continue
                if should_use_val(block, val_ratio, seed):
                    write_val_block(
                        val_file, val_small_file, stats, rendered_block, val_small_bytes
                    )
                else:
                    write_train_block(
                        train_file,
                        train_small_file,
                        stats,
                        rendered_block,
                        train_small_bytes,
                    )
                continue

            val_indices = select_val_indices(parts, val_ratio, seed)
            train_indices = set(range(len(parts))) - val_indices

            for run in iter_runs(parts, train_indices):
                train_block = render_document(run)
                if train_block:
                    write_train_block(
                        train_file,
                        train_small_file,
                        stats,
                        train_block,
                        train_small_bytes,
                    )

            for run in iter_runs(parts, val_indices):
                val_block = render_document(run)
                if val_block:
                    write_val_block(
                        val_file, val_small_file, stats, val_block, val_small_bytes
                    )

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
        help="Approximate validation split ratio within each work.",
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
