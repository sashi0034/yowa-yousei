#!/usr/bin/env python3
"""token-id の .bin ファイルを読み込み、GPT 学習用のミニバッチを作る。

言語モデル学習では、入力 x と正解 y を「1 トークンずらした列」として作る。
このファイルはその前処理を担い、train.py からは `loader.get_batch(...)` だけで
学習に使える Tensor を受け取れるようにする。
"""

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
    """ミニバッチ作成に必要な設定をまとめるデータクラス。

    block_size は 1 サンプルの文脈長、batch_size は 1 回に並べるサンプル数、
    device は Tensor を CPU/GPU のどこへ置くかを表す。
    """

    block_size: int = DEFAULT_BLOCK_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "auto"


class TokenBatchLoader:
    """連続した token-id ファイルから、causal LM 用バッチをランダムに切り出す。

    causal LM は「過去のトークンだけを見て、次のトークンを予測する」学習。
    そのため、この loader は x と y が 1 つ右にずれた形になるようにバッチを作る。
    """

    def __init__(
        self,
        train_bin: Path = DEFAULT_TRAIN_BIN_PATH,
        val_bin: Path = DEFAULT_VAL_BIN_PATH,
        config: BatchConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """train/val の token-id ファイルを開き、バッチ生成の準備をする。

        np.memmap でファイルを開くため、大きなデータセットでも全体をメモリへ
        読み込まずにランダムアクセスできる。
        """

        self.config = config or BatchConfig()
        self.device = resolve_device(self.config.device)
        self.rng = np.random.default_rng(seed)

        self.train_tokens = load_memmap(train_bin)
        self.val_tokens = load_memmap(val_bin)
        validate_token_count("train", self.train_tokens, self.config.block_size)
        validate_token_count("val", self.val_tokens, self.config.block_size)

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """指定 split からランダムに文章片を取り、入力 x と正解 y を返す。

        返り値の x/y はどちらも shape が (batch_size, block_size)。
        y は x より 1 トークン先なので、モデルは各位置で「次の token-id」を当てる。
        """

        tokens = self._tokens_for_split(split)
        # block_size 個の入力 x に対して、1 つ先の正解 y も必要なので、
        # 実際には block_size + 1 個の連続トークンを切り出す。
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
            # 例: chunk = [10, 20, 30, 40] なら
            # x = [10, 20, 30], y = [20, 30, 40]。
            # モデルは各位置の x から「次のトークン」である y を当てるように学習する。
            chunk = np.asarray(
                tokens[start : start + self.config.block_size + 1],
                dtype=np.int64,
            )
            x_rows.append(torch.from_numpy(chunk[:-1]))
            y_rows.append(torch.from_numpy(chunk[1:]))

        x = torch.stack(x_rows)
        y = torch.stack(y_rows)
        # 返す shape はどちらも (batch_size, block_size)。
        # train.py ではこの x/ y をそのまま model(x, y) へ渡す。
        return move_batch_to_device(x, y, self.device)

    def _tokens_for_split(self, split: str) -> np.memmap:
        """split 名に対応する token 配列を返す内部ヘルパー。"""

        if split == "train":
            return self.train_tokens
        if split == "val":
            return self.val_tokens
        raise ValueError("split must be 'train' or 'val'")


def load_memmap(path: Path) -> np.memmap:
    """uint16 の token-id ファイルを np.memmap として開く。

    ファイル存在、空ファイル、uint16 境界に合わないサイズをここで検査して、
    学習中に分かりにくいエラーになるのを防ぐ。
    """

    # .bin 全体を一度に RAM へ読まず、必要な範囲だけ OS に読み込ませる。
    # memmap によってコーパスが大きくなっても扱いやすい。
    if not path.exists():
        raise SystemExit(f"token file does not exist: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"token file is empty: {path}")
    if path.stat().st_size % np.dtype(np.uint16).itemsize != 0:
        raise SystemExit(f"token file size is not aligned to uint16: {path}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def validate_token_count(label: str, tokens: np.memmap, block_size: int) -> None:
    """1 サンプルを作るのに十分な token 数があるか確認する。"""

    if block_size <= 0:
        raise SystemExit("--block-size must be positive")
    # 1 サンプル作るには、入力 block_size 個に加えて「次トークン」1 個が必要。
    if len(tokens) <= block_size:
        raise SystemExit(
            f"{label} data has {len(tokens):,} tokens, but needs more than "
            f"--block-size={block_size:,}"
        )


def resolve_device(device: str) -> torch.device:
    """`auto`/`cpu`/`cuda` などの文字列を PyTorch の device に変換する。"""

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
    """作成した x/y Tensor を、学習に使う CPU/GPU へ移動する。"""

    if device.type == "cuda":
        # pin_memory + non_blocking=True で、CPU から GPU への転送を効率化する。
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def parse_args() -> argparse.Namespace:
    """バッチ作成の単体確認用 CLI 引数を読み取る。"""

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
    """dataset.py を単体実行したときに、バッチ形状と x/y のずれを確認する。"""

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
