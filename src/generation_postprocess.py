"""Postprocess generated text for display."""

from __future__ import annotations

import re


CHAPTER_SEPARATOR_TOKEN = "<chapter_sep>"
DISPLAY_CHAPTER_SEPARATOR = "-----------------------------------------------"

_DIALOGUE_RE = re.compile(r"「[^「」\n]*」")
_UNICODE_EXCLAMATION_RE = re.compile(r"([^\x00-\x7F])!")
_UNICODE_QUESTION_RE = re.compile(r"([^\x00-\x7F])\?")
_NEWLINE_MARKER = "\uE000"


def _mark_dialogue_line_breaks(text: str) -> str:
    return _DIALOGUE_RE.sub(
        lambda match: f"{_NEWLINE_MARKER}{match.group(0)}{_NEWLINE_MARKER}",
        text,
    )


def _normalize_marked_newlines(text: str) -> str:
    marker = re.escape(_NEWLINE_MARKER)
    text = re.sub(rf"[ \t]*{marker}[ \t]*", _NEWLINE_MARKER, text)
    text = re.sub(rf"{marker}+", _NEWLINE_MARKER, text)
    text = re.sub(rf"{marker}\n", "\n", text)
    text = re.sub(rf"\n{marker}", "\n", text)
    return text.replace(_NEWLINE_MARKER, "\n")


def postprocess_generated_text(text: str) -> str:
    """Format generated text for display without changing model behavior."""

    text = _UNICODE_EXCLAMATION_RE.sub(r"\1！", text)
    text = _UNICODE_QUESTION_RE.sub(r"\1？", text)
    text = text.replace(
        CHAPTER_SEPARATOR_TOKEN,
        f"{_NEWLINE_MARKER}{DISPLAY_CHAPTER_SEPARATOR}{_NEWLINE_MARKER}",
    )
    text = _mark_dialogue_line_breaks(text)
    text = _normalize_marked_newlines(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
