#!/usr/bin/env python3
"""Clean raw Japanese web novel text for language-model training.

The raw files in this project are mostly concatenated web-novel chapters.
Chapter boundaries look like "[n1234ab/1]" or
"[episodes/1234567890 (1 / 200)]"; the first non-empty line after
the boundary is treated as the chapter title and excluded from training text.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from corpus_markers import CHAPTER_SEPARATOR, EOS_MARKER


DEFAULT_RAW_DIR = Path("data/raw/novels")
DEFAULT_CLEAN_DIR = Path("data/processed/clean")
DEFAULT_COMBINED_PATH = Path("data/processed/clean.txt")


CHAPTER_MARKER_RE = re.compile(
    r"""^\[
    (?:
        [a-z][a-z0-9]+/\d+
        |
        (?:episodes|chapter)/\d+\s*\(\s*\d+\s*/\s*\d+\s*\)
    )
    \]$""",
    re.IGNORECASE | re.VERBOSE,
)

HTML_TAG_RE = re.compile(r"<[^>\n]{1,200}>")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200b\u202f\u205f\u3000]+")

AOZORA_RUBY_RE = re.compile(r"｜([^《》\n]{1,80})《[^《》\n]{1,80}》")
RUBY_AFTER_WORD_RE = re.compile(
    r"([一-龯々〆ヵヶぁ-んァ-ヴーA-Za-z0-9・]{1,40})《[^《》\n]{1,80}》"
)
PAREN_KANA_RUBY_RE = re.compile(
    r"([一-龯々〆ヵヶ]{1,20})[（(]([ぁ-んァ-ヴー・ー]{1,30})[）)]"
)


@dataclass
class Chapter:
    title: str
    lines: list[str]


@dataclass
class FileStats:
    raw_path: Path
    clean_path: Path
    chapters: int
    body_lines: int
    skipped_meta_lines: int
    bytes_written: int


def normalize_text(text: str) -> str:
    """Apply conservative Unicode and markup cleanup to a line."""
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = text.replace("\ufeff", "")
    text = text.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")
    text = HTML_TAG_RE.sub("", text)
    text = CONTROL_CHAR_RE.sub("", text)
    return text


def simplify_ruby(text: str) -> str:
    """Keep the base text and drop common ruby readings."""
    previous = None
    while previous != text:
        previous = text
        text = AOZORA_RUBY_RE.sub(r"\1", text)
        text = RUBY_AFTER_WORD_RE.sub(r"\1", text)
        text = PAREN_KANA_RUBY_RE.sub(r"\1", text)
    return text


def clean_line(line: str) -> str:
    line = normalize_text(line.rstrip("\r\n"))
    line = simplify_ruby(line)
    line = SPACE_RE.sub(" ", line)
    return line.strip()


def is_meta_line(line: str) -> bool:
    # TODO: Revisit metadata removal once the raw-data patterns are clearer.
    return False


def parse_chapters(raw_text: str) -> tuple[list[Chapter], int]:
    chapters: list[Chapter] = []
    current_title: str | None = None
    current_lines: list[str] = []
    awaiting_title = False
    skipped_meta_lines = 0

    def finish_current() -> None:
        nonlocal current_title, current_lines
        if current_title is None:
            return
        stripped = trim_blank_edges(current_lines)
        if stripped:
            chapters.append(Chapter(current_title, stripped))
        current_title = None
        current_lines = []

    for raw_line in raw_text.splitlines():
        normalized_for_marker = normalize_text(raw_line).strip()
        if CHAPTER_MARKER_RE.match(normalized_for_marker):
            finish_current()
            awaiting_title = True
            continue

        line = clean_line(raw_line)
        if is_meta_line(line):
            skipped_meta_lines += 1
            continue

        if awaiting_title:
            if not line:
                continue
            current_title = line
            current_lines = []
            awaiting_title = False
            continue

        if current_title is None:
            if line:
                current_title = "本文"
                current_lines = [line]
            continue

        current_lines.append(line)

    finish_current()
    return chapters, skipped_meta_lines


def trim_blank_edges(lines: Iterable[str]) -> list[str]:
    result = list(lines)
    while result and not result[0]:
        result.pop(0)
    while result and not result[-1]:
        result.pop()
    return result


def collapse_blank_lines(lines: Iterable[str], max_blank_lines: int = 1) -> list[str]:
    result: list[str] = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= max_blank_lines:
                result.append("")
            continue
        blank_count = 0
        result.append(line)
    return trim_blank_edges(result)


def render_clean_text(chapters: Iterable[Chapter]) -> str:
    blocks: list[str] = []
    for chapter in chapters:
        body = collapse_blank_lines(chapter.lines)
        if not body:
            continue
        blocks.append("\n".join(body))
    return f"\n\n{CHAPTER_SEPARATOR}\n\n".join(blocks) + (
        f"\n{EOS_MARKER}\n" if blocks else ""
    )


def clean_file(raw_path: Path, clean_dir: Path) -> FileStats:
    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    chapters, skipped_meta_lines = parse_chapters(raw_text)
    clean_text = render_clean_text(chapters)

    clean_path = clean_dir / raw_path.name
    clean_path.write_text(clean_text, encoding="utf-8", newline="\n")

    return FileStats(
        raw_path=raw_path,
        clean_path=clean_path,
        chapters=len(chapters),
        body_lines=sum(1 for chapter in chapters for line in chapter.lines if line),
        skipped_meta_lines=skipped_meta_lines,
        bytes_written=clean_path.stat().st_size,
    )


def iter_raw_files(raw_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(raw_dir.glob("*.txt"), key=lambda path: path.name)
    if limit is not None:
        files = files[:limit]
    return files


def combine_clean_files(clean_paths: Iterable[Path], combined_path: Path) -> int:
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with combined_path.open("w", encoding="utf-8", newline="\n") as out:
        first = True
        for path in clean_paths:
            text = path.read_text(encoding="utf-8")
            if not text:
                continue
            if not first:
                out.write("\n")
                written += 1
            out.write(text)
            written += len(text.encode("utf-8"))
            first = False
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean data/raw/novels/*.txt into chapter-formatted training text."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--combined", type=Path, default=DEFAULT_COMBINED_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files. Useful for checking output quickly.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove the clean output directory before writing new files.",
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="Write per-file outputs only; skip data/processed/clean.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir
    clean_dir = args.clean_dir

    if not raw_dir.exists():
        raise SystemExit(f"raw directory does not exist: {raw_dir}")

    if args.reset and clean_dir.exists():
        shutil.rmtree(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    raw_files = iter_raw_files(raw_dir, args.limit)
    if not raw_files:
        raise SystemExit(f"no .txt files found in: {raw_dir}")

    stats: list[FileStats] = []
    for index, raw_path in enumerate(raw_files, start=1):
        file_stats = clean_file(raw_path, clean_dir)
        stats.append(file_stats)
        if index % 50 == 0 or index == len(raw_files):
            print(f"processed {index}/{len(raw_files)} files", flush=True)

    combined_bytes = 0
    if not args.no_combine:
        combined_bytes = combine_clean_files((stat.clean_path for stat in stats), args.combined)

    empty_outputs = [stat.raw_path.name for stat in stats if stat.chapters == 0]
    print()
    print(f"raw files: {len(stats)}")
    print(f"chapters: {sum(stat.chapters for stat in stats)}")
    print(f"body lines: {sum(stat.body_lines for stat in stats)}")
    print(f"skipped meta lines: {sum(stat.skipped_meta_lines for stat in stats)}")
    print(f"clean dir: {clean_dir}")
    if not args.no_combine:
        print(f"combined: {args.combined} ({combined_bytes} bytes)")
    if empty_outputs:
        print(f"empty outputs: {len(empty_outputs)}")
        for name in empty_outputs[:20]:
            print(f"  {name}")


if __name__ == "__main__":
    main()
