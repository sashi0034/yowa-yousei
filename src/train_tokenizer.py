#!/usr/bin/env python3
"""Train a SentencePiece tokenizer for the Japanese LM corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_INPUT_PATH = Path("data/processed/train.txt")
DEFAULT_MODEL_PREFIX = Path("tokenizer/yowa_yousei_sp")
DEFAULT_SELF_TEST_TEXT = "彼女は静かに笑った。"


def import_sentencepiece():
    try:
        import sentencepiece as spm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "sentencepiece is not installed. Run: "
            ".venv/bin/python -m pip install sentencepiece"
        ) from exc
    return spm


def train_tokenizer(
    input_path: Path,
    model_prefix: Path,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    byte_fallback: bool,
    input_sentence_size: int,
    shuffle_input_sentence: bool,
    hard_vocab_limit: bool,
) -> None:
    if not input_path.exists():
        raise SystemExit(f"input does not exist: {input_path}")
    if vocab_size <= 0:
        raise SystemExit("--vocab-size must be positive")

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm = import_sentencepiece()

    args = {
        "input": str(input_path),
        "model_prefix": str(model_prefix),
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "byte_fallback": byte_fallback,
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        "hard_vocab_limit": hard_vocab_limit,
    }
    if input_sentence_size > 0:
        args["input_sentence_size"] = input_sentence_size
        args["shuffle_input_sentence"] = shuffle_input_sentence

    print(f"input: {input_path}")
    print(f"model prefix: {model_prefix}")
    print(f"vocab size: {vocab_size}")
    print(f"model type: {model_type}")
    print(f"character coverage: {character_coverage}")
    print(f"byte fallback: {byte_fallback}")
    print()

    try:
        spm.SentencePieceTrainer.train(**args)
    except RuntimeError as exc:
        message = str(exc)
        if "Vocabulary size too high" in message or "vocab_size" in message:
            raise SystemExit(
                "SentencePiece failed because the requested vocabulary is too large "
                "for this input. Try the full train.txt, lower --vocab-size, or add "
                "--soft-vocab-limit while experimenting with train_small.txt."
            ) from exc
        raise


def run_self_test(model_path: Path, text: str) -> None:
    if not model_path.exists():
        raise SystemExit(f"model was not generated: {model_path}")

    spm = import_sentencepiece()
    processor = spm.SentencePieceProcessor()
    processor.load(str(model_path))

    ids = processor.encode(text, out_type=int)
    decoded = processor.decode(ids)

    print()
    print(f"model: {model_path}")
    print(f"vocab size: {processor.get_piece_size()}")
    print(f"self test text: {text}")
    print(f"ids: {ids}")
    print(f"decoded: {decoded}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train tokenizer/yowa_yousei_sp.model and "
            "tokenizer/yowa_yousei_sp.vocab."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-prefix", type=Path, default=DEFAULT_MODEL_PREFIX)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--model-type", choices=["unigram", "bpe"], default="unigram")
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument(
        "--no-byte-fallback",
        action="store_true",
        help="Disable byte fallback. Enabled by default for robust encoding.",
    )
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=0,
        help="Optional SentencePiece sampling limit. 0 means use the whole input.",
    )
    parser.add_argument(
        "--no-shuffle-input-sentence",
        action="store_true",
        help="Disable shuffling when --input-sentence-size is set.",
    )
    parser.add_argument(
        "--soft-vocab-limit",
        action="store_true",
        help="Allow a smaller vocab if the corpus cannot support --vocab-size.",
    )
    parser.add_argument("--self-test-text", default=DEFAULT_SELF_TEST_TEXT)
    parser.add_argument(
        "--no-self-test",
        action="store_true",
        help="Skip encode/decode verification after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_tokenizer(
        input_path=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        byte_fallback=not args.no_byte_fallback,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=not args.no_shuffle_input_sentence,
        hard_vocab_limit=not args.soft_vocab_limit,
    )

    if not args.no_self_test:
        run_self_test(args.model_prefix.with_suffix(".model"), args.self_test_text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
