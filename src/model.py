#!/usr/bin/env python3
"""小さな GPT 形式の causal language model。

train.py からは `GPT(config)` として作られ、`model(x, y)` によって
次トークン予測の logits と loss を返す。Transformer の主要部品
である attention、MLP、残差接続、位置埋め込みもこのファイルにまとまっている。
"""

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
    """GPT モデルの構造を決める設定値をまとめる。

    vocab_size は扱える token-id の種類数、block_size は最大文脈長、
    n_layer/n_head/n_embd は Transformer の大きさを決める。
    """

    vocab_size: int = DEFAULT_VOCAB_SIZE
    block_size: int = DEFAULT_BLOCK_SIZE
    n_layer: int = DEFAULT_N_LAYER
    n_head: int = DEFAULT_N_HEAD
    n_embd: int = DEFAULT_N_EMBD
    dropout: float = DEFAULT_DROPOUT
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """未来を見ない制約つきの multi-head self-attention。

    各トークン位置が、自分より前のトークンから必要な情報を集める層。
    causal mask により、正解である未来トークンを覗き見しないようにする。
    """

    def __init__(self, config: GPTConfig) -> None:
        """attention に必要な線形層、dropout、causal mask を初期化する。"""

        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask は「未来のトークンを見ない」ための下三角行列。
        # これにより、文章生成時と同じ条件で次トークン予測を学習できる。
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """入力表現 x から、過去文脈を混ぜ込んだ新しい表現を計算する。

        shape は (batch, seq, embd) のまま返る。内部では head ごとに分けて
        Query/Key/Value attention を行い、最後に再び結合する。
        """

        batch_size, seq_len, embd_size = x.size()

        # かの有名な 'Attention'
        # 1 回の線形層で Query/Key/Value をまとめて作り、後で 3 つに分ける。
        # Query は「何を探すか」、Key は「各位置が持つ手がかり」、Value は「集める情報」。
        q, k, v = self.c_attn(x).split(embd_size, dim=2)

        # (batch, seq, embd) を (batch, head, seq, head_size) へ並べ替え、
        # 複数の attention head が別々の観点で文脈を見られるようにする。
        q = q.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)

        # Query と Key の内積が「どの過去位置をどれだけ見るか」のスコアになる。
        # sqrt(head_size) で割るのは、'スケール化'。内積が大きくなりすぎて softmax が極端になるのを防ぐ。
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        att = att.masked_fill(mask == 0, -torch.inf)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # attention 重みで Value を混ぜ、各位置が必要な過去文脈を集めた表現にする。
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embd_size)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """Transformer block 内の MLP 部分。

    attention が「どの過去位置を見るか」を混ぜるのに対し、FeedForward は
    各位置ごとの表現を非線形に変換して、より豊かな特徴へ作り替える。
    """

    def __init__(self, config: GPTConfig) -> None:
        """2 層の線形変換と GELU/dropout からなる MLP を作る。"""

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """各 token 位置の埋め込み表現を MLP で変換する。"""

        return self.net(x)


class Block(nn.Module):
    """pre-LayerNorm 形式の Transformer block。

    1 block は attention と MLP を 1 回ずつ通す単位。
    これを複数層重ねることで、より長く複雑な文脈を扱えるようにする。
    """

    def __init__(self, config: GPTConfig) -> None:
        """LayerNorm、causal attention、FeedForward を組み合わせて block を作る。"""

        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """attention と MLP を残差接続つきで適用し、表現を更新する。"""

        # 残差接続により、attention/MLP が学習初期に多少不安定でも情報が流れやすくなる。
        # LayerNorm を先に置く pre-LN 構成は、深めの Transformer でも学習を安定させやすい。
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x


class GPT(nn.Module):
    """causal language modeling 用の decoder-only Transformer。

    入力 token-id 列から各位置の次トークン分布を予測する。
    targets が渡された場合は、学習で最小化する cross entropy loss も返す。
    """

    def __init__(self, config: GPTConfig) -> None:
        """埋め込み層、Transformer block 群、出力層を作り、重みを初期化する。"""

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
        """token-id 列を受け取り、次トークン予測の logits と必要なら loss を返す。

        idx は入力 token-id、targets は正解 token-id。train.py では
        `logits, loss = model(x, y)` の形で呼ばれ、loss に対して backward する。
        """

        _, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} is longer than block_size "
                f"{self.config.block_size}"
            )

        # token_emb は「単語/サブワードそのもの」の意味、position_emb は「何番目か」の情報。
        # Transformer 本体は順序を直接知らないので、位置埋め込みを足して順序を伝える。
        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        token_emb = self.transformer["wte"](idx)
        position_emb = self.transformer["wpe"](positions)
        x = self.transformer["drop"](token_emb + position_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)

        # logits の shape は (batch, seq, vocab_size)。
        # 各位置ごとに「語彙中のどのトークンが次に来そうか」の未正規化スコアを持つ。
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # cross_entropy は logits と正解トークン id から、次トークン予測の外れ具合を測る。
            # PyTorch は (class 次元) を 2 次元目に期待するため、全位置をまとめて 2 次元に潰す。
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def _init_weights(self, module: nn.Module) -> None:
        """Linear/Embedding 層の重みを、Transformer でよく使う小さな正規分布で初期化する。"""

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
        """学習済みモデルを使って、idx の続きを token-id として生成する。

        生成時は最後の位置の logits から次トークンをサンプリングし、
        それを入力の末尾へ足す処理を max_new_tokens 回繰り返す。
        """

        self.eval()
        for _ in range(max_new_tokens):
            # 学習時の最大長 (block_size) を超えた文脈は扱えないため、末尾だけを条件にする。
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            # 生成では最後の位置だけを使う。そこが「次に出す 1 トークン」の分布を表す。
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # top_k は候補を上位 k 個に絞り、低確率トークンの暴発を抑えるための生成時テクニック。
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -torch.inf)
            probs = F.softmax(logits, dim=-1)
            # 確率分布からサンプリングすることで、毎回まったく同じ文になりにくくする。
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def parse_args() -> argparse.Namespace:
    """model.py を単体確認するための CLI 引数を読み取る。"""

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
    """`auto`/`cpu`/`cuda` などの文字列を PyTorch の device に変換する。"""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false")
    return resolved


def main() -> None:
    """モデルを単体で作り、ランダム入力で forward/loss 計算が動くか確認する。"""

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
