#!/usr/bin/env python3
"""Convert train/val text files into contiguous token-id binary files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import sentencepiece as spm


DEFAULT_TOKENIZER_PATH = Path("tokenizer/yowa_yousei_sp.model")
DEFAULT_TRAIN_PATH = Path("data/processed/train.txt")
DEFAULT_VAL_PATH = Path("data/processed/val.txt")
DEFAULT_TRAIN_OUTPUT_PATH = Path("data/processed/train.bin")
DEFAULT_VAL_OUTPUT_PATH = Path("data/processed/val.bin")
DEFAULT_EOS_MARKER = "<eos>"


@dataclass
class PrepareStats:
    documents: int = 0
    tokens: int = 0
    bytes_written: int = 0
    documents_without_eos_marker: int = 0


def iter_documents(path: Path, eos_marker: str) -> Iterator[tuple[str, bool]]:
    """Yield document text, using a literal eos marker line as the boundary."""
    lines: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for line in file:
            if not lines and not line.strip():
                continue
            if line.strip() == eos_marker:
                yield "".join(lines).rstrip("\n"), True
                lines = []
                continue
            lines.append(line)

    if lines:
        yield "".join(lines).rstrip("\n"), False


def load_tokenizer(path: Path) -> spm.SentencePieceProcessor:
    if not path.exists():
        raise SystemExit(f"tokenizer model does not exist: {path}")

    processor = spm.SentencePieceProcessor()
    processor.load(str(path))

    vocab_size = processor.get_piece_size()
    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise SystemExit(
            f"vocab size {vocab_size} is too large for np.uint16; use uint32 instead"
        )
    if processor.eos_id() < 0:
        raise SystemExit("tokenizer does not define eos_id")
    return processor


def encode_file(
    input_path: Path,
    output_path: Path,
    processor: spm.SentencePieceProcessor,
    eos_marker: str,
    progress_every: int,
) -> PrepareStats:
    if not input_path.exists():
        raise SystemExit(f"input does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    eos_id = processor.eos_id()
    stats = PrepareStats()

    with output_path.open("wb") as output_file:
        for text, had_eos_marker in iter_documents(input_path, eos_marker):
            if not text:
                continue

            ids = processor.encode(text, out_type=int)
            ids.append(eos_id)
            token_array = np.asarray(ids, dtype=np.uint16)
            token_array.tofile(output_file)

            stats.documents += 1
            stats.tokens += int(token_array.size)
            stats.bytes_written += int(token_array.nbytes)
            if not had_eos_marker:
                stats.documents_without_eos_marker += 1

            if progress_every > 0 and stats.documents % progress_every == 0:
                print(
                    f"{input_path}: {stats.documents} documents, "
                    f"{stats.tokens:,} tokens",
                    flush=True,
                )

    return stats


def decode_sample(
    path: Path,
    processor: spm.SentencePieceProcessor,
    sample_tokens: int,
) -> str:
    if sample_tokens <= 0 or not path.exists():
        return ""

    tokens = np.fromfile(path, dtype=np.uint16, count=sample_tokens)
    eos_id = processor.eos_id()
    pieces: list[str] = []
    current: list[int] = []

    for token in tokens.tolist():
        token_id = int(token)
        if token_id == eos_id:
            if current:
                pieces.append(processor.decode(current))
                current = []
            pieces.append("\n<eos>\n")
            continue
        current.append(token_id)

    if current:
        pieces.append(processor.decode(current))

    return "".join(pieces)


def format_size(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def print_stats(label: str, path: Path, stats: PrepareStats) -> None:
    print(
        f"{label}: {stats.documents} documents, {stats.tokens:,} tokens, "
        f"{format_size(stats.bytes_written)} -> {path}"
    )
    if stats.documents_without_eos_marker:
        print(
            f"  note: added eos_id to {stats.documents_without_eos_marker} "
            "document(s) that reached EOF without an <eos> marker"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode train/val text into np.uint16 token-id .bin files."
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT_PATH)
    parser.add_argument("--val-output", type=Path, default=DEFAULT_VAL_OUTPUT_PATH)
    parser.add_argument(
        "--eos-marker",
        default=DEFAULT_EOS_MARKER,
        help="Line that marks the end of one document in the text corpus.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress after this many documents. 0 disables progress logs.",
    )
    parser.add_argument(
        "--sample-tokens",
        type=int,
        default=300,
        help="Decode this many tokens from the train output as a quick sanity check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = load_tokenizer(args.tokenizer)

    print(f"tokenizer: {args.tokenizer}")
    print(f"vocab size: {processor.get_piece_size()}")
    print(f"eos id: {processor.eos_id()}")
    print()

    train_stats = encode_file(
        input_path=args.train,
        output_path=args.train_output,
        processor=processor,
        eos_marker=args.eos_marker,
        progress_every=args.progress_every,
    )
    val_stats = encode_file(
        input_path=args.val,
        output_path=args.val_output,
        processor=processor,
        eos_marker=args.eos_marker,
        progress_every=args.progress_every,
    )

    print()
    print_stats("train", args.train_output, train_stats)
    print_stats("val", args.val_output, val_stats)

    sample = decode_sample(args.train_output, processor, args.sample_tokens)
    if sample:
        print()
        print(f"decoded first {args.sample_tokens} train tokens:")
        print(sample)


if __name__ == "__main__":
    main()
