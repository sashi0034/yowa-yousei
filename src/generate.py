#!/usr/bin/env python3
"""Generate text from a trained GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import torch.nn.functional as F

from dataset import resolve_device
from model import GPT, GPTConfig


DEFAULT_CHECKPOINT_PATH = Path("checkpoints/best.pt")
DEFAULT_TOKENIZER_PATH = Path("tokenizer/yowa_yousei_sp.model")
DEFAULT_PROMPT = "彼女は静かに目を覚ますと、そこは"
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_REPETITION_WINDOW = 128
DEFAULT_NO_REPEAT_NGRAM_SIZE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Japanese text from a GPT checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Read the prompt from a UTF-8 text file instead of --prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus sampling threshold. Use 1.0 to disable.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Keep only the k most likely tokens before sampling. 0 disables it.",
    )
    parser.add_argument(
        "--stop-at-eos",
        action="store_true",
        help="Stop generation when the tokenizer's eos_id is sampled.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help="Penalize recently used tokens. 1.0 disables it.",
    )
    parser.add_argument(
        "--repetition-window",
        type=int,
        default=DEFAULT_REPETITION_WINDOW,
        help="How many recent tokens are considered by --repetition-penalty.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=DEFAULT_NO_REPEAT_NGRAM_SIZE,
        help="Ban tokens that would repeat an n-gram. 0 disables it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, cuda, or a device like cuda:0.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 0:
        raise SystemExit("--max-new-tokens must be zero or positive")
    if args.temperature < 0:
        raise SystemExit("--temperature must be zero or positive")
    if args.top_p <= 0 or args.top_p > 1:
        raise SystemExit("--top-p must be in (0, 1]")
    if args.top_k < 0:
        raise SystemExit("--top-k must be zero or positive")
    if args.repetition_penalty < 1:
        raise SystemExit("--repetition-penalty must be greater than or equal to 1")
    if args.repetition_window < 0:
        raise SystemExit("--repetition-window must be zero or positive")
    if args.no_repeat_ngram_size < 0:
        raise SystemExit("--no-repeat-ngram-size must be zero or positive")


def load_tokenizer(path: Path) -> spm.SentencePieceProcessor:
    if not path.exists():
        raise SystemExit(f"tokenizer model does not exist: {path}")
    processor = spm.SentencePieceProcessor()
    processor.load(str(path))
    return processor


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GPT, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" not in checkpoint:
        raise SystemExit("checkpoint does not contain model_config")
    if "model" not in checkpoint:
        raise SystemExit("checkpoint does not contain model weights")

    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is None:
        return args.prompt
    if not args.prompt_file.exists():
        raise SystemExit(f"prompt file does not exist: {args.prompt_file}")
    return args.prompt_file.read_text(encoding="utf-8")


def encode_prompt(processor: spm.SentencePieceProcessor, prompt: str) -> list[int]:
    ids = processor.encode(prompt, out_type=int)
    if ids:
        return ids
    bos_id = processor.bos_id()
    if bos_id < 0:
        raise SystemExit("prompt is empty and tokenizer does not define bos_id")
    return [bos_id]


def filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    return logits.masked_fill(logits < values[:, [-1]], -torch.inf)


def filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )
    return logits.masked_fill(indices_to_remove, -torch.inf)


def apply_repetition_penalty(
    logits: torch.Tensor,
    ids: torch.Tensor,
    penalty: float,
    window: int,
) -> torch.Tensor:
    if penalty == 1.0 or window == 0:
        return logits

    recent_ids = ids[:, -window:] if window > 0 else ids
    for batch_index in range(logits.size(0)):
        token_ids = set(int(token_id) for token_id in recent_ids[batch_index].tolist())
        if not token_ids:
            continue
        token_indices = torch.tensor(
            sorted(token_ids),
            dtype=torch.long,
            device=logits.device,
        )
        selected = logits[batch_index, token_indices]
        logits[batch_index, token_indices] = torch.where(
            selected < 0,
            selected * penalty,
            selected / penalty,
        )
    return logits


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    ids: torch.Tensor,
    ngram_size: int,
) -> torch.Tensor:
    if ngram_size <= 1 or ids.size(1) < ngram_size - 1:
        return logits

    prefix = tuple(int(token_id) for token_id in ids[0, -(ngram_size - 1) :].tolist())
    banned: set[int] = set()
    token_ids = ids[0].tolist()
    for index in range(len(token_ids) - ngram_size + 1):
        ngram = tuple(int(token_id) for token_id in token_ids[index : index + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])

    if banned:
        logits[:, sorted(banned)] = -torch.inf
    return logits


@torch.no_grad()
def generate_ids(
    model: GPT,
    input_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    repetition_window: int,
    no_repeat_ngram_size: int,
    eos_id: int,
    stop_at_eos: bool,
    device: torch.device,
) -> list[int]:
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        logits = apply_repetition_penalty(
            logits=logits,
            ids=idx,
            penalty=repetition_penalty,
            window=repetition_window,
        )
        logits = apply_no_repeat_ngram(
            logits=logits,
            ids=idx,
            ngram_size=no_repeat_ngram_size,
        )

        if temperature == 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = filter_top_k(logits, top_k)
            logits = filter_top_p(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)
        if stop_at_eos and eos_id >= 0 and int(next_id.item()) == eos_id:
            break

    return idx[0].tolist()


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = load_tokenizer(args.tokenizer)
    model, checkpoint = load_model(args.checkpoint, device)

    if tokenizer.get_piece_size() != model.config.vocab_size:
        raise SystemExit(
            f"tokenizer vocab size ({tokenizer.get_piece_size()}) does not match "
            f"model vocab size ({model.config.vocab_size})"
        )

    prompt = read_prompt(args)
    input_ids = encode_prompt(tokenizer, prompt)
    output_ids = generate_ids(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        eos_id=tokenizer.eos_id(),
        stop_at_eos=args.stop_at_eos,
        device=device,
    )

    print(f"checkpoint: {args.checkpoint}")
    if "step" in checkpoint:
        print(f"step: {checkpoint['step']}")
    if "val_loss" in checkpoint:
        print(f"val loss: {checkpoint['val_loss']:.4f}")
    print(f"device: {device}")
    print()
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
