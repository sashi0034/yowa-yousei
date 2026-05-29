#!/usr/bin/env python3
"""準備済みの token-id ファイルから、小さな GPT 言語モデルを学習するスクリプト。

このファイルは「データを読む → モデルを作る → loss を計算する →
backward で勾配を出す → optimizer で重みを更新する」という、機械学習の
学習ループ全体を担当する。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from dataset import BatchConfig, TokenBatchLoader, resolve_device
from model import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_N_EMBD,
    DEFAULT_N_HEAD,
    DEFAULT_N_LAYER,
    DEFAULT_VOCAB_SIZE,
    GPT,
    GPTConfig,
)


DEFAULT_TRAIN_BIN_PATH = Path("data/processed/train.bin")
DEFAULT_VAL_BIN_PATH = Path("data/processed/val.bin")
DEFAULT_OUT_DIR = Path("checkpoints")
DEFAULT_MAX_STEPS = 50000
DEFAULT_EVAL_INTERVAL = 500
DEFAULT_EVAL_ITERS = 100
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_MIN_LR = 3e-5
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_BATCH_SIZE = 8
DEFAULT_LOG_INTERVAL = 10


def configure_stdio() -> None:
    """標準出力/標準エラーを行バッファリングにして、学習ログをすぐ表示する。"""

    # 学習ログは長時間眺めることが多いので、リダイレクト時もすぐ表示されるようにする。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True, write_through=True)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読み取り、学習設定として使える形にまとめる。

    batch size、学習率、モデルサイズ、評価間隔など、実験で変えたい値を
    Python コードを書き換えずに指定できるようにする。
    """

    parser = argparse.ArgumentParser(description="Train a GPT causal language model.")
    parser.add_argument("--train-bin", type=Path, default=DEFAULT_TRAIN_BIN_PATH)
    parser.add_argument("--val-bin", type=Path, default=DEFAULT_VAL_BIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--n-layer", type=int, default=DEFAULT_N_LAYER)
    parser.add_argument("--n-head", type=int, default=DEFAULT_N_HEAD)
    parser.add_argument("--n-embd", type=int, default=DEFAULT_N_EMBD)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)

    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--eval-iters", type=int, default=DEFAULT_EVAL_ITERS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_MIN_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, cuda, or a device like cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
        help="AMP dtype. auto uses bf16 on supported CUDA GPUs, else fp16 on CUDA.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """学習設定の値が破綻していないかを、学習開始前に検査する。

    例えば block_size や batch_size が 0 以下だとバッチを作れず、
    n_embd が n_head で割り切れないと multi-head attention の各 head に
    同じ幅を割り当てられない。
    """

    positive_ints = [
        "vocab_size",
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
        "max_steps",
        "eval_interval",
        "eval_iters",
        "gradient_accumulation_steps",
        "batch_size",
        "log_interval",
    ]
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")

    if args.n_embd % args.n_head != 0:
        raise SystemExit("--n-embd must be divisible by --n-head")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    if args.min_lr < 0:
        raise SystemExit("--min-lr must be zero or positive")
    if args.dropout < 0 or args.dropout >= 1:
        raise SystemExit("--dropout must be in [0, 1)")
    if args.grad_clip < 0:
        raise SystemExit("--grad-clip must be zero or positive")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be zero or positive")


def resolve_amp_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    """AMP で使う浮動小数点 dtype を、指定値と実行デバイスから決める。

    AMP は GPU 学習を速くするための混合精度計算。CUDA では float16/bfloat16 を
    使えるが、CPU では通常の float32 にして挙動を分かりやすく保つ。
    """

    # AMP(Automatic Mixed Precision) は、一部の計算を低精度で行って GPU を速く使う仕組み。
    # CPU では混合精度の恩恵が小さいため、通常の float32 に固定する。
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def autocast_context(device: torch.device, dtype: torch.dtype) -> Any:
    """必要なときだけ PyTorch の autocast コンテキストを返す。

    学習ループ側は常に `with autocast_context(...)` と書けるため、
    CPU/float32 と CUDA/混合精度の分岐をモデル計算部分から追い出せる。
    """

    # autocast の中では、PyTorch が安全な演算だけを指定 dtype へ自動変換する。
    # CUDA 以外や float32 指定では何もしないコンテキストを返し、同じコードで動かせるようにする。
    if device.type == "cuda" and dtype != torch.float32:
        return torch.amp.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def get_lr(step: int, args: argparse.Namespace) -> float:
    """現在 step で使う学習率を warmup + cosine decay で計算する。

    学習率は「1 回の更新で重みをどれくらい動かすか」を決める重要な値。
    序盤は小さく始め、後半は徐々に下げることで学習を安定させる。
    """

    # 学習率スケジュール:
    # 1. warmup 中は 0 付近から少しずつ上げて、学習初期の不安定さを抑える。
    # 2. その後は cosine decay で滑らかに min_lr へ近づける。
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return args.learning_rate * (step + 1) / args.warmup_steps
    if step >= args.max_steps:
        return args.min_lr

    decay_steps = max(1, args.max_steps - args.warmup_steps)
    decay_step = max(0, step - args.warmup_steps)
    decay_ratio = min(1.0, decay_step / decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """optimizer 内のすべての parameter group に同じ学習率を設定する。"""

    # PyTorch の optimizer は param_group ごとに学習率を持つので、全グループを更新する。
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@torch.no_grad()
def estimate_loss(
    model: GPT,
    loader: TokenBatchLoader,
    eval_iters: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    """train/val それぞれの平均 loss を、複数バッチで推定する。

    学習中の 1 バッチの loss はかなり揺れるため、複数回サンプルして平均する。
    train loss は学習データへの当てはまり、val loss は未知データへの汎化の目安になる。
    """

    # 評価中は Dropout などを止めたいので eval() へ切り替える。
    # @torch.no_grad() により勾配を保存せず、メモリ使用量と計算量を抑える。
    model.eval()
    out: dict[str, float] = {}
    for split in ["train", "val"]:
        losses = torch.empty(eval_iters)
        for idx in range(eval_iters):
            # ランダムに複数バッチを取り、1 バッチだけに偏らない平均 loss を出す。
            x, y = loader.get_batch(split)
            with autocast_context(device, amp_dtype):
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("loss was not computed")
            losses[idx] = loss.item()
        out[split] = losses.mean().item()
    # 評価後は学習ループへ戻るため、Dropout などを再び有効にする。
    model.train()
    return out


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model_config: GPTConfig,
    args: argparse.Namespace,
    step: int,
    best_val_loss: float,
    losses: dict[str, float],
) -> None:
    """モデル重みと学習再開に必要な状態を checkpoint ファイルへ保存する。

    model だけでなく optimizer/scaler/設定値/現在 step も保存することで、
    後から「どの条件で、どこまで学習したモデルか」を確認しやすくなる。
    """

    # checkpoint には「重み」だけでなく optimizer/scaler も保存する。
    # これにより中断後も、AdamW の内部状態や混合精度学習の状態を引き継いで再開できる。
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": asdict(model_config),
        "args": vars(args),
        "step": step,
        "best_val_loss": best_val_loss,
        "train_loss": losses["train"],
        "val_loss": losses["val"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main() -> None:
    """学習全体を実行するエントリーポイント。

    引数検査、データローダ作成、モデル/optimizer 準備、勾配蓄積つき学習、
    定期評価、checkpoint 保存までを順に行う。
    """

    configure_stdio()

    args = parse_args()
    validate_args(args)

    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.dtype, device)
    # 乱数 seed を固定すると、バッチ抽出や重み初期化が再現しやすくなる。
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        # TF32 は Ampere 以降の GPU で使える高速な行列計算形式。精度を少し緩めて速度を得る。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # TokenBatchLoader は train.bin/val.bin から、言語モデル用の (x, y) バッチを作る。
    # x は入力トークン列、y は「次に来る正解トークン列」になる。
    loader = TokenBatchLoader(
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        config=BatchConfig(
            block_size=args.block_size,
            batch_size=args.batch_size,
            device=str(device),
        ),
        seed=args.seed,
    )
    # GPTConfig にモデルの大きさを集約し、同じ設定を checkpoint にも保存する。
    model_config = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(model_config).to(device)
    # AdamW は Adam に weight decay を組み合わせた optimizer。
    # Transformer 系の学習でよく使われ、重みが大きくなりすぎるのを抑える。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # float16 学習では小さな勾配が 0 に丸められることがある。
    # GradScaler は loss を一時的に大きくして backward し、数値の下振れを防ぐ。
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=(device.type == "cuda" and amp_dtype == torch.float16),
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"amp dtype: {amp_dtype}")
    print(f"parameters: {param_count:,}")
    print(f"train tokens: {len(loader.train_tokens):,}")
    print(f"val tokens: {len(loader.val_tokens):,}")

    best_val_loss = float("inf")
    # set_to_none=True は勾配テンソルを 0 で埋めず None に戻すため、少しメモリ効率がよい。
    optimizer.zero_grad(set_to_none=True)
    step_start_time = time.time()

    for step in range(1, args.max_steps + 1):
        # ここからが典型的な deep learning の「学習ループ」。
        # 1 step は、いくつかのミニバッチで勾配を計算し、最後に重みを 1 回更新する単位。
        # step ごとに学習率を変え、optimizer へ反映する。
        lr = get_lr(step - 1, args)
        set_optimizer_lr(optimizer, lr)

        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            # train.bin からランダムな文章片を取り出す。
            # 同じデータを順番に読むのではなくランダムにすることで、更新の偏りを減らす。
            x, y = loader.get_batch("train")
            with autocast_context(device, amp_dtype):
                # model は各位置で「次のトークン候補」の logits を出し、y との cross entropy を返す。
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("loss was not computed")

            accumulated_loss += loss.item()
            # 勾配蓄積では複数ミニバッチの勾配を足してから 1 回だけ更新する。
            # loss を割っておくと、実効的な平均 loss で backward したのと同じスケールになる。
            loss = loss / args.gradient_accumulation_steps
            # backward は loss を小さくするために、各パラメータをどちらへ動かすべきか
            # という勾配を計算して .grad に溜める。
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            # 勾配爆発を防ぐため、全パラメータの勾配ノルムを上限内に収める。
            # clip 前に unscale_ して、GradScaler で拡大された値ではなく本来の勾配を扱う。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # backward で溜めた勾配を使って重みを 1 回更新する。
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == 1:
            elapsed = time.time() - step_start_time
            step_start_time = time.time()
            avg_loss = accumulated_loss / args.gradient_accumulation_steps
            print(
                f"step {step:6d} | train loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | {elapsed:.2f}s"
            )

        should_eval = step % args.eval_interval == 0 or step == args.max_steps
        if should_eval:
            # 評価では optimizer.step() をしない。つまり重みは変えず、
            # 現在のモデルがどれくらい予測できるかだけを見る。
            losses = estimate_loss(model, loader, args.eval_iters, device, amp_dtype)
            print(
                f"eval {step:6d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}"
            )

            # val loss は「学習に直接使っていないデータ」での性能。
            # train loss だけ下がって val loss が悪化する場合は、過学習の兆候になる。
            is_best = losses["val"] < best_val_loss
            if is_best:
                best_val_loss = losses["val"]
                save_checkpoint(
                    args.out_dir / "best.pt",
                    model,
                    optimizer,
                    scaler,
                    model_config,
                    args,
                    step,
                    best_val_loss,
                    losses,
                )
                print(f"saved checkpoint: {args.out_dir / 'best.pt'}")

            save_checkpoint(
                args.out_dir / "latest.pt",
                model,
                optimizer,
                scaler,
                model_config,
                args,
                step,
                best_val_loss,
                losses,
            )
            print(f"saved checkpoint: {args.out_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
