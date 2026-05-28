#!/usr/bin/env python3
"""Long-running generation server.

Loads the tokenizer and checkpoint once at startup, then reads one JSON
request per line from stdin and writes one JSON response per line to stdout.
All log lines go to stderr so stdout is reserved for the line protocol.

Protocol
--------

Startup (first line on stdout):

    {"event": "ready", "checkpoint": "...", "device": "cuda", "step": 1234, "val_loss": 1.23}

Request (one JSON object per line on stdin):

    {"id": "req-1", "prompt": "...", "max_new_tokens": 200, "temperature": 0.8, ...}

Response (one JSON object per line on stdout):

    {"id": "req-1", "ok": true, "text": "..."}
    {"id": "req-1", "ok": false, "error": "..."}

The server exits when stdin is closed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from dataset import resolve_device
from generate import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_REPETITION_WINDOW,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    GenerationOptions,
    ModelBundle,
    generate_text,
    prepare_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a long-running generation process that reads JSON requests "
            "from stdin and writes JSON responses to stdout."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, cuda, or a device like cuda:0.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Default value used when a request omits max_new_tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Default value used when a request omits temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Default value used when a request omits top_p.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Default value used when a request omits top_k.",
    )
    parser.add_argument(
        "--stop-at-eos",
        action="store_true",
        help="Default stop-at-eos behavior used when a request omits stop_at_eos.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
    )
    parser.add_argument(
        "--repetition-window",
        type=int,
        default=DEFAULT_REPETITION_WINDOW,
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=DEFAULT_NO_REPEAT_NGRAM_SIZE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def log(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def write_event(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


_OPTION_FIELDS = {field.name for field in dataclasses.fields(GenerationOptions)}


def build_options(defaults: GenerationOptions, request: dict[str, Any]) -> GenerationOptions:
    overrides: dict[str, Any] = {}
    for key in _OPTION_FIELDS:
        if key not in request:
            continue
        value = request[key]
        if key == "stop_at_eos":
            overrides[key] = bool(value)
        elif key in {"max_new_tokens", "top_k", "repetition_window", "no_repeat_ngram_size"}:
            overrides[key] = int(value)
        elif key == "seed":
            overrides[key] = None if value is None else int(value)
        else:
            overrides[key] = float(value)
    return dataclasses.replace(defaults, **overrides)


def handle_request(line: str, bundle: ModelBundle, defaults: GenerationOptions) -> None:
    request_id: Any = None
    try:
        request = json.loads(line)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        request_id = request.get("id")
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError("prompt is required and must be a non-empty string")
        options = build_options(defaults, request)
        text = generate_text(bundle, prompt, options)
        write_event({"id": request_id, "ok": True, "text": text})
    except Exception as error:
        log(f"request failed: {error}\n{traceback.format_exc().rstrip()}")
        write_event({"id": request_id, "ok": False, "error": str(error)})


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    log(f"loading checkpoint={args.checkpoint} tokenizer={args.tokenizer} device={device}")
    bundle = prepare_bundle(args.checkpoint, args.tokenizer, device)
    defaults = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop_at_eos=args.stop_at_eos,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        seed=args.seed,
    )

    ready_payload: dict[str, Any] = {
        "event": "ready",
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "device": str(device),
    }
    if "step" in bundle.metadata:
        ready_payload["step"] = bundle.metadata["step"]
    if "val_loss" in bundle.metadata:
        ready_payload["val_loss"] = float(bundle.metadata["val_loss"])
    write_event(ready_payload)
    log("generation server is ready")

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        handle_request(line, bundle, defaults)

    log("stdin closed, shutting down generation server")


if __name__ == "__main__":
    main()
