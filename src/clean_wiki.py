#!/usr/bin/env python3
"""Convert WikiExtractor JSONL output into the project text-corpus format."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from clean_text import clean_line, collapse_blank_lines
from corpus_markers import EOS_MARKER


DEFAULT_INPUT_DIR = Path("data/intermediate/wiki_extracted")
DEFAULT_OUTPUT_PATH = Path("data/processed/wiki_clean.txt")

SECTION_HEADING_RE = re.compile(r"^=+\s*(?P<title>[^=]+?)\s*=+$")
LIST_PREFIX_RE = re.compile(r"^[*#:;]+\s*")
PAREN_RE = re.compile(r"([（(])([^（）()]{0,120})([）)])")
INLINE_SECTION_HEADING_RE = re.compile(r"^[一-龯々〆ヵヶぁ-んァ-ヴーA-Za-z0-9・ー]{1,40}[.]$")
KANA_READING_RE = re.compile(r"^[ぁ-んァ-ヴー・ー\s、,，/／・]+$")

STOP_SECTION_TITLES = {
    "脚注",
    "注釈",
    "出典",
    "参考文献",
    "関連項目",
    "外部リンク",
    "参考資料",
    "関連文献",
    "関連書籍",
    "ギャラリー",
}


@dataclass
class WikiStats:
    json_lines: int = 0
    articles_seen: int = 0
    articles_written: int = 0
    articles_too_short: int = 0
    json_errors: int = 0
    bytes_written: int = 0


def iter_json_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise SystemExit(f"input does not exist: {input_path}")
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )


def iter_json_lines(input_path: Path) -> Iterator[str]:
    if str(input_path) == "-":
        yield from sys.stdin
        return

    for path in iter_json_paths(input_path):
        with path.open("r", encoding="utf-8", errors="replace") as file:
            yield from file


def iter_wiki_records(lines: Iterable[str], stats: WikiStats) -> Iterator[dict]:
    for line in lines:
        stats.json_lines += 1
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats.json_errors += 1
            continue
        if isinstance(record, dict):
            yield record


def iter_wiki_records_from_paths(paths: Iterable[Path], stats: WikiStats) -> Iterator[dict]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as file:
            yield from iter_wiki_records(file, stats)


def normalize_section_title(line: str) -> str | None:
    match = SECTION_HEADING_RE.match(line)
    if match:
        title = match.group("title").strip()
        return title.rstrip(".。").strip()
    if INLINE_SECTION_HEADING_RE.match(line):
        return line.rstrip(".").strip()
    title = line.rstrip(".。").strip()
    if title in STOP_SECTION_TITLES:
        return title
    return None


def should_skip_line(line: str) -> bool:
    if not line:
        return False
    if line.startswith(("http://", "https://")):
        return True
    if line.startswith(("ISBN ", "doi:", "DOI:")):
        return True
    return False


def clean_wiki_line(line: str, title: str) -> str:
    line = clean_line(line)
    line = clean_parenthetical_punctuation(line)
    return line


def clean_parenthetical_punctuation(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        open_paren, content, close_paren = match.groups()
        content = content.strip()
        content = content.strip("、,，;；:：")
        content = re.sub(r"\s+", " ", content).strip()
        if not content or KANA_READING_RE.match(content):
            return ""
        return f"{open_paren}{content}{close_paren}"

    previous = None
    while previous != text:
        previous = text
        text = PAREN_RE.sub(replace, text)
    return text


def clean_article(title: str, text: str, min_chars: int) -> str | None:
    lines: list[str] = []
    skipped_title = False

    for raw_line in text.splitlines():
        line = clean_wiki_line(raw_line, title)
        if not line:
            lines.append("")
            continue

        line = LIST_PREFIX_RE.sub("", line).strip()
        if not line:
            lines.append("")
            continue

        if not skipped_title and line == title:
            skipped_title = True
            continue

        section_title = normalize_section_title(line)
        if section_title:
            if section_title in STOP_SECTION_TITLES:
                break
            continue

        if should_skip_line(line):
            continue

        lines.append(line)

    body_lines = collapse_blank_lines(lines)
    body = "\n".join(body_lines).strip()
    if len(body) < min_chars:
        return None
    return f"{body}\n{EOS_MARKER}\n\n"


def clean_wiki(
    input_path: Path,
    output_path: Path,
    min_chars: int,
    max_articles: int | None,
    progress_every: int,
    append: bool,
) -> WikiStats:
    if str(input_path) != "-":
        paths = iter_json_paths(input_path)
        if not paths:
            raise SystemExit(f"no WikiExtractor output files found in: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = WikiStats()
    mode = "a" if append else "w"

    with output_path.open(mode, encoding="utf-8", newline="\n") as output_file:
        if append and output_path.exists() and output_path.stat().st_size > 0:
            output_file.write("\n")

        records = iter_wiki_records(iter_json_lines(input_path), stats)
        for record in records:
            title = str(record.get("title") or "").strip()
            text = str(record.get("text") or "")
            if not title or not text:
                continue

            stats.articles_seen += 1
            article = clean_article(title, text, min_chars)
            if article is None:
                stats.articles_too_short += 1
                continue

            output_file.write(article)
            stats.articles_written += 1
            stats.bytes_written += len(article.encode("utf-8"))

            if progress_every > 0 and stats.articles_written % progress_every == 0:
                print(
                    f"written {stats.articles_written:,} articles "
                    f"({format_size(stats.bytes_written)})",
                    flush=True,
                )

            if max_articles is not None and stats.articles_written >= max_articles:
                break

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
        description="Clean WikiExtractor JSONL into an <eos>-separated corpus."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Skip articles with fewer cleaned characters than this.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Stop after writing this many articles. Useful for quick checks.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress after this many written articles. 0 disables logs.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append cleaned wiki articles to the output instead of replacing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = clean_wiki(
        input_path=args.input,
        output_path=args.output,
        min_chars=args.min_chars,
        max_articles=args.max_articles,
        progress_every=args.progress_every,
        append=args.append,
    )

    print()
    print(f"json lines: {stats.json_lines:,}")
    print(f"articles seen: {stats.articles_seen:,}")
    print(f"articles written: {stats.articles_written:,}")
    print(f"articles too short: {stats.articles_too_short:,}")
    print(f"json errors: {stats.json_errors:,}")
    print(f"output: {args.output} ({format_size(stats.bytes_written)})")


if __name__ == "__main__":
    main()
