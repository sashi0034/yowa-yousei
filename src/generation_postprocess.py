"""生成された文章を、画面表示しやすい形へ整える。

ここで行うのは記号や改行の見た目の調整だけで、モデルの生成確率や
token の選び方には影響しない。generate.py の最後で decode 済み文字列に適用する。
"""

from __future__ import annotations

import re

from corpus_markers import CHAPTER_SEPARATOR

DISPLAY_CHAPTER_SEPARATOR = "-----------------------------------------------"

# 正規表現はモジュール読み込み時に一度だけ compile し、生成のたびに再利用する。
_DIALOGUE_RE = re.compile(r"「[^「」\n]*」")
_UNICODE_EXCLAMATION_RE = re.compile(r"([^\x00-\x7F])!")
_UNICODE_QUESTION_RE = re.compile(r"([^\x00-\x7F])\?")
# 本文中にほぼ出ない私用領域文字を、一時的な改行マーカーとして使う。
# 先にマーカーで位置を覚えてから最後に\n へ戻すと、重複改行の整理がしやすい。
_NEWLINE_MARKER = "\uE000"


def _mark_dialogue_line_breaks(text: str) -> str:
    """会話文「...」の前後に一時マーカーを置き、後で改行できるようにする。

    直接 `\n` を入れる前に専用マーカーで印をつけることで、他の整形処理と合わせて
    連続改行や余計な空白を最後にまとめて整理できる。
    """

    # 会話文「...」の前後へ改行マーカーを置き、台詞が地の文に埋もれないようにする。
    return _DIALOGUE_RE.sub(
        lambda match: f"{_NEWLINE_MARKER}{match.group(0)}{_NEWLINE_MARKER}",
        text,
    )


def _normalize_marked_newlines(text: str) -> str:
    """一時改行マーカーを整理し、最終的な `\n` に変換する。

    マーカーの周辺にある空白、連続したマーカー、既存改行との重なりを整えてから
    実際の改行へ戻すため、表示時に空行が増えすぎにくくなる。
    """

    marker = re.escape(_NEWLINE_MARKER)
    # マーカー周辺の空白や連続マーカーを畳んでから、最終的な改行へ変換する。
    text = re.sub(rf"[ \t]*{marker}[ \t]*", _NEWLINE_MARKER, text)
    text = re.sub(rf"{marker}+", _NEWLINE_MARKER, text)
    # すでに改行がある場所では、マーカー由来の余計な空行を増やさない。
    text = re.sub(rf"{marker}\n", "\n", text)
    text = re.sub(rf"\n{marker}", "\n", text)
    return text.replace(_NEWLINE_MARKER, "\n")


def postprocess_generated_text(text: str) -> str:
    """生成済みテキストを表示向けに整形する。

    ASCII の !/? を日本語文脈では全角へ寄せ、章区切りや会話文の改行を整える。
    あくまで decode 後の文字列処理なので、学習済みモデルそのものは変更しない。
    """

    # 日本語文字の直後に ASCII の!/?が出た場合、全角記号に寄せて読みやすくする。
    text = _UNICODE_EXCLAMATION_RE.sub(r"\1！", text)
    text = _UNICODE_QUESTION_RE.sub(r"\1？", text)
    # 学習データ内の章区切りマーカーを、そのまま出す代わりに表示向け罫線へ置き換える。
    text = text.replace(
        CHAPTER_SEPARATOR,
        f"{_NEWLINE_MARKER}{DISPLAY_CHAPTER_SEPARATOR}{_NEWLINE_MARKER}",
    )
    text = _mark_dialogue_line_breaks(text)
    text = _normalize_marked_newlines(text)
    # 生成文では改行が連続しすぎることがあるため、最大 2 連続に整える。
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
