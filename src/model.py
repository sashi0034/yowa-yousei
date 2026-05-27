#!/usr/bin/env python3
"""A small GPT-style causal language model."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_VOCAB_SIZE = 32000
DEFAULT_BLOCK_SIZE = 512
DEFAULT_N_LAYER = 8
DEFAULT_N_HEAD = 8
DEFAULT_N_EMBD = 512
DEFAULT_DROPOUT = 0.1


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = DEFAULT_VOCAB_SIZE
    block_size: int = DEFAULT_BLOCK_SIZE
    n_layer: int = DEFAULT_N_LAYER
    n_head: int = DEFAULT_N_HEAD
    n_embd: int = DEFAULT_N_EMBD
    dropout: float = DEFAULT_DROPOUT
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a lower-triangular causal mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embd_size = x.size()
        q, k, v = self.c_attn(x).split(embd_size, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        att = att.masked_fill(mask == 0, -torch.inf)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embd_size)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """The MLP part of a Transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """A pre-LayerNorm Transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Decoder-only Transformer for causal language modeling."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} is longer than block_size "
                f"{self.config.block_size}"
            )

        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        token_emb = self.transformer["wte"](idx)
        position_emb = self.transformer["wpe"](positions)
        x = self.transformer["drop"](token_emb + position_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -torch.inf)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a GPT model and run a quick forward-pass check."
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--n-layer", type=int, default=DEFAULT_N_LAYER)
    parser.add_argument("--n-head", type=int, default=DEFAULT_N_HEAD)
    parser.add_argument("--n-embd", type=int, default=DEFAULT_N_EMBD)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, cuda, or a device like cuda:0.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false")
    return resolved


def main() -> None:
    args = parse_args()
    if args.block_size <= 0:
        raise SystemExit("--block-size must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    config = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    model.train()

    x = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, config.block_size),
        device=device,
    )
    y = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, config.block_size),
        device=device,
    )
    logits, loss = model(x, y)

    param_count = sum(p.numel() for p in model.parameters())
    expected_loss = math.log(config.vocab_size)
    print(f"device: {device}")
    print(f"parameters: {param_count:,}")
    print(f"logits.shape: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.4f}")
    print(f"expected initial loss: log({config.vocab_size}) = {expected_loss:.4f}")


if __name__ == "__main__":
    main()
