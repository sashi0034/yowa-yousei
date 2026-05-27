#!/usr/bin/env python3
"""Load token-id .bin files and create GPT training batches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


DEFAULT_TRAIN_BIN_PATH = Path("data/processed/train.bin")
DEFAULT_VAL_BIN_PATH = Path("data/processed/val.bin")
DEFAULT_BLOCK_SIZE = 512
DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class BatchConfig:
    block_size: int = DEFAULT_BLOCK_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "auto"


class TokenBatchLoader:
    """Sample random causal-LM batches from contiguous uint16 token files."""

    def __init__(
        self,
        train_bin: Path = DEFAULT_TRAIN_BIN_PATH,
        val_bin: Path = DEFAULT_VAL_BIN_PATH,
        config: BatchConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or BatchConfig()
        self.device = resolve_device(self.config.device)
        self.rng = np.random.default_rng(seed)

        self.train_tokens = load_memmap(train_bin)
        self.val_tokens = load_memmap(val_bin)
        validate_token_count("train", self.train_tokens, self.config.block_size)
        validate_token_count("val", self.val_tokens, self.config.block_size)

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._tokens_for_split(split)
        max_start = len(tokens) - self.config.block_size
        starts = self.rng.integers(
            low=0,
            high=max_start,
            size=self.config.batch_size,
            endpoint=False,
        )

        x_rows: list[torch.Tensor] = []
        y_rows: list[torch.Tensor] = []
        for start in starts:
            chunk = np.asarray(
                tokens[start : start + self.config.block_size + 1],
                dtype=np.int64,
            )
            x_rows.append(torch.from_numpy(chunk[:-1]))
            y_rows.append(torch.from_numpy(chunk[1:]))

        x = torch.stack(x_rows)
        y = torch.stack(y_rows)
        return move_batch_to_device(x, y, self.device)

    def _tokens_for_split(self, split: str) -> np.memmap:
        if split == "train":
            return self.train_tokens
        if split == "val":
            return self.val_tokens
        raise ValueError("split must be 'train' or 'val'")


def load_memmap(path: Path) -> np.memmap:
    if not path.exists():
        raise SystemExit(f"token file does not exist: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"token file is empty: {path}")
    if path.stat().st_size % np.dtype(np.uint16).itemsize != 0:
        raise SystemExit(f"token file size is not aligned to uint16: {path}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def validate_token_count(label: str, tokens: np.memmap, block_size: int) -> None:
    if block_size <= 0:
        raise SystemExit("--block-size must be positive")
    if len(tokens) <= block_size:
        raise SystemExit(
            f"{label} data has {len(tokens):,} tokens, but needs more than "
            f"--block-size={block_size:,}"
        )


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false")
    return resolved


def move_batch_to_device(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample x/y batches from train.bin and val.bin for GPT training."
    )
    parser.add_argument("--train-bin", type=Path, default=DEFAULT_TRAIN_BIN_PATH)
    parser.add_argument("--val-bin", type=Path, default=DEFAULT_VAL_BIN_PATH)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, cuda, or a device like cuda:0.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Which split to sample for the quick verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = TokenBatchLoader(
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        config=BatchConfig(
            block_size=args.block_size,
            batch_size=args.batch_size,
            device=args.device,
        ),
        seed=args.seed,
    )

    x, y = loader.get_batch(args.split)
    shifted = bool(torch.equal(x[:, 1:], y[:, :-1]))

    print(f"train tokens: {len(loader.train_tokens):,}")
    print(f"val tokens: {len(loader.val_tokens):,}")
    print(f"device: {x.device}")
    print(f"x.shape: {tuple(x.shape)}")
    print(f"y.shape: {tuple(y.shape)}")
    print(f"y is one token ahead of x: {shifted}")
    if not shifted:
        raise SystemExit("batch verification failed")


if __name__ == "__main__":
    main()
