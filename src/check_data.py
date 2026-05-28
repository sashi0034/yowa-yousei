#!/usr/bin/env python3
"""Check generated data files for size and timestamp consistency."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_CLEAN_PATH = Path("data/processed/clean.txt")
DEFAULT_TRAIN_PATH = Path("data/processed/train.txt")
DEFAULT_VAL_PATH = Path("data/processed/val.txt")
DEFAULT_TRAIN_SMALL_PATH = Path("data/processed/train_small.txt")
DEFAULT_VAL_SMALL_PATH = Path("data/processed/val_small.txt")
DEFAULT_TOKENIZER_PATH = Path("tokenizer/yowa_yousei_sp.model")
DEFAULT_VOCAB_PATH = Path("tokenizer/yowa_yousei_sp.vocab")
DEFAULT_TRAIN_BIN_PATH = Path("data/processed/train.bin")
DEFAULT_VAL_BIN_PATH = Path("data/processed/val.bin")
UINT16_BYTES = 2
TIMESTAMP_TOLERANCE_SECONDS = 1.0


@dataclass(frozen=True)
class FileInfo:
    label: str
    path: Path
    required: bool = True
    token_file: bool = False

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def size(self) -> int:
        return self.path.stat().st_size if self.exists else 0

    @property
    def mtime(self) -> float:
        return self.path.stat().st_mtime if self.exists else 0.0


@dataclass(frozen=True)
class CheckResult:
    status: str
    label: str
    message: str


def format_size(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def format_mtime(timestamp: float) -> str:
    if timestamp <= 0:
        return "-"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def file_status(info: FileInfo) -> str:
    if not info.exists:
        return "FAIL" if info.required else "WARN"
    if info.size == 0:
        return "FAIL"
    if info.token_file and info.size % UINT16_BYTES != 0:
        return "FAIL"
    return "OK"


def file_detail(info: FileInfo) -> str:
    if not info.exists:
        return "missing"
    if info.size == 0:
        return "empty"
    if info.token_file:
        if info.size % UINT16_BYTES != 0:
            return "not uint16 aligned"
        return f"{info.size // UINT16_BYTES:,} tokens"
    return ""


def freshness_check(output: FileInfo, inputs: list[FileInfo]) -> CheckResult:
    existing_inputs = [info for info in inputs if info.exists]
    if not output.exists:
        return CheckResult("FAIL", output.label, f"missing output: {output.path}")
    if not existing_inputs:
        return CheckResult("WARN", output.label, "no existing inputs to compare")

    newest_input = max(existing_inputs, key=lambda info: info.mtime)
    if output.mtime + TIMESTAMP_TOLERANCE_SECONDS < newest_input.mtime:
        return CheckResult(
            "FAIL",
            output.label,
            (
                f"{output.path} is older than {newest_input.path} "
                f"({format_mtime(output.mtime)} < {format_mtime(newest_input.mtime)})"
            ),
        )
    return CheckResult(
        "OK",
        output.label,
        f"newer than inputs (latest input: {newest_input.path})",
    )


def presence_checks(files: list[FileInfo]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for info in files:
        status = file_status(info)
        if status == "OK":
            message = f"{info.path} exists ({format_size(info.size)})"
        elif not info.exists:
            message = f"{info.path} does not exist"
        elif info.size == 0:
            message = f"{info.path} is empty"
        else:
            message = f"{info.path} size is not aligned to uint16"
        results.append(CheckResult(status, info.label, message))
    return results


def size_relationship_checks(files_by_label: dict[str, FileInfo]) -> list[CheckResult]:
    results: list[CheckResult] = []

    def compare_larger(left_label: str, right_label: str) -> None:
        left = files_by_label[left_label]
        right = files_by_label[right_label]
        if not left.exists or not right.exists or left.size == 0 or right.size == 0:
            return
        if left.size <= right.size:
            results.append(
                CheckResult(
                    "WARN",
                    f"{left.label}/{right.label}",
                    f"{left.path} is not larger than {right.path}",
                )
            )
        else:
            results.append(
                CheckResult(
                    "OK",
                    f"{left.label}/{right.label}",
                    f"{left.label} is larger than {right.label}",
                )
            )

    def compare_smaller_or_equal(small_label: str, full_label: str) -> None:
        small = files_by_label[small_label]
        full = files_by_label[full_label]
        if not small.exists or not full.exists or small.size == 0 or full.size == 0:
            return
        if small.size > full.size:
            results.append(
                CheckResult(
                    "WARN",
                    f"{small.label}/{full.label}",
                    f"{small.path} is larger than {full.path}",
                )
            )
        else:
            results.append(
                CheckResult(
                    "OK",
                    f"{small.label}/{full.label}",
                    f"{small.label} is not larger than {full.label}",
                )
            )

    compare_larger("train", "val")
    compare_larger("train_bin", "val_bin")
    if "train_small" in files_by_label:
        compare_smaller_or_equal("train_small", "train")
    if "val_small" in files_by_label:
        compare_smaller_or_equal("val_small", "val")
    return results


def print_file_table(files: list[FileInfo]) -> None:
    print("Files")
    print(f"{'status':<6} {'label':<13} {'size':>12} {'mtime':<19} {'detail':<20} path")
    print("-" * 96)
    for info in files:
        size = format_size(info.size) if info.exists else "-"
        print(
            f"{file_status(info):<6} "
            f"{info.label:<13} "
            f"{size:>12} "
            f"{format_mtime(info.mtime):<19} "
            f"{file_detail(info):<20} "
            f"{info.path}"
        )


def print_checks(results: list[CheckResult]) -> None:
    print()
    print("Checks")
    print("-" * 96)
    for result in results:
        print(f"{result.status:<6} {result.label:<20} {result.message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify data files produced by clean_text.py, split_data.py, "
            "train_tokenizer.py, and prepare_data.py."
        )
    )
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN_PATH)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--train-small", type=Path, default=DEFAULT_TRAIN_SMALL_PATH)
    parser.add_argument("--val-small", type=Path, default=DEFAULT_VAL_SMALL_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--train-bin", type=Path, default=DEFAULT_TRAIN_BIN_PATH)
    parser.add_argument("--val-bin", type=Path, default=DEFAULT_VAL_BIN_PATH)
    parser.add_argument(
        "--no-small",
        action="store_true",
        help="Do not require train_small.txt and val_small.txt.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Return a non-zero exit code when WARN checks are present.",
    )
    return parser.parse_args()


def build_files(args: argparse.Namespace) -> list[FileInfo]:
    files = [
        FileInfo("clean", args.clean),
        FileInfo("train", args.train),
        FileInfo("val", args.val),
    ]
    if not args.no_small:
        files.extend(
            [
                FileInfo("train_small", args.train_small),
                FileInfo("val_small", args.val_small),
            ]
        )
    files.extend(
        [
            FileInfo("tokenizer", args.tokenizer),
            FileInfo("vocab", args.vocab),
            FileInfo("train_bin", args.train_bin, token_file=True),
            FileInfo("val_bin", args.val_bin, token_file=True),
        ]
    )
    return files


def run_checks(files: list[FileInfo]) -> list[CheckResult]:
    files_by_label = {info.label: info for info in files}
    results = presence_checks(files)
    clean = files_by_label["clean"]
    train = files_by_label["train"]
    val = files_by_label["val"]
    tokenizer = files_by_label["tokenizer"]
    vocab = files_by_label["vocab"]
    train_bin = files_by_label["train_bin"]
    val_bin = files_by_label["val_bin"]

    results.extend(
        [
            freshness_check(train, [clean]),
            freshness_check(val, [clean]),
        ]
    )
    if "train_small" in files_by_label:
        results.append(freshness_check(files_by_label["train_small"], [clean]))
    if "val_small" in files_by_label:
        results.append(freshness_check(files_by_label["val_small"], [clean]))

    results.extend(
        [
            freshness_check(tokenizer, [train]),
            freshness_check(vocab, [train]),
            freshness_check(train_bin, [train, tokenizer]),
            freshness_check(val_bin, [val, tokenizer]),
        ]
    )
    results.extend(size_relationship_checks(files_by_label))
    return results


def main() -> None:
    args = parse_args()
    files = build_files(args)
    results = run_checks(files)

    print_file_table(files)
    print_checks(results)

    fail_count = sum(1 for result in results if result.status == "FAIL")
    warn_count = sum(1 for result in results if result.status == "WARN")
    print()
    print(f"Summary: {fail_count} FAIL, {warn_count} WARN")
    if fail_count == 0 and (warn_count == 0 or not args.strict_warnings):
        print("OK: data timestamps and sizes look consistent.")
        return
    raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
