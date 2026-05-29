#!/usr/bin/env python3
"""学習済み GPT checkpoint から文章を生成するスクリプト。

train.py で保存した model/checkpoint と SentencePiece tokenizer を読み込み、
プロンプトを token-id に変換してから、1 token ずつ続きをサンプリングする。
生成品質を調整する temperature/top-p/top-k/repetition penalty などもここで扱う。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import torch.nn.functional as F

from dataset import resolve_device
from generation_postprocess import postprocess_generated_text
from model import GPT, GPTConfig


DEFAULT_CHECKPOINT_PATH = Path("checkpoints/best.pt")
DEFAULT_TOKENIZER_PATH = Path("tokenizer/yowa_yousei_sp.model")
DEFAULT_PROMPT = "彼女は静かに目を覚ますと、そこは"
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 0
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_REPETITION_WINDOW = 128
DEFAULT_NO_REPEAT_NGRAM_SIZE = 4
DEFAULT_SEED = 0


@dataclass(frozen=True)
class GenerationOptions:
    """1 回の生成リクエストで使うパラメータをまとめる。

    max_new_tokens は生成する長さ、temperature/top_p/top_k は候補 token の選び方、
    repetition_penalty/no_repeat_ngram_size は同じ表現の繰り返しを抑える設定。
    """

    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    stop_at_eos: bool = False
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    repetition_window: int = DEFAULT_REPETITION_WINDOW
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE
    seed: int | None = DEFAULT_SEED


@dataclass(frozen=True)
class ModelBundle:
    """読み込み済みのモデル・tokenizer・実行 device をひとまとめにする。

    生成 API やサーバーでは同じ model/tokenizer を何度も使うため、
    毎回 checkpoint を読み直さず、この bundle を使い回せるようにする。
    """

    model: GPT
    tokenizer: spm.SentencePieceProcessor
    device: torch.device
    eos_id: int
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """コマンドラインから生成条件を受け取り、argparse の Namespace にまとめる。

    checkpoint/tokenizer/prompt に加え、サンプリング方法や繰り返し抑制の強さを
    CLI から実験できるようにする。
    """

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
        default=DEFAULT_TOP_K,
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, cuda, or a device like cuda:0.",
    )
    return parser.parse_args()


def validate_options(options: GenerationOptions) -> None:
    """生成オプションの値が、サンプリング処理で破綻しない範囲か検査する。

    例えば top_p は確率の累積閾値なので (0, 1]、repetition_penalty は
    1.0 を「無効」とするため 1 以上である必要がある。
    """

    if options.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be zero or positive")
    if options.temperature < 0:
        raise ValueError("temperature must be zero or positive")
    if options.top_p <= 0 or options.top_p > 1:
        raise ValueError("top_p must be in (0, 1]")
    if options.top_k < 0:
        raise ValueError("top_k must be zero or positive")
    if options.repetition_penalty < 1:
        raise ValueError("repetition_penalty must be greater than or equal to 1")
    if options.repetition_window < 0:
        raise ValueError("repetition_window must be zero or positive")
    if options.no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be zero or positive")


def options_from_args(args: argparse.Namespace) -> GenerationOptions:
    """argparse の結果から、生成処理で使う GenerationOptions を作る。"""

    return GenerationOptions(
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


def load_tokenizer(path: Path) -> spm.SentencePieceProcessor:
    """SentencePiece tokenizer を読み込み、文字列と token-id の変換を可能にする。

    生成では prompt を token-id に変換し、最後に生成された token-id 列を
    文字列へ戻すため、学習時と同じ tokenizer が必要になる。
    """

    if not path.exists():
        raise SystemExit(f"tokenizer model does not exist: {path}")
    # 学習時と同じ SentencePiece モデルを使い、文字列と token id を相互変換する。
    processor = spm.SentencePieceProcessor()
    processor.load(str(path))
    return processor


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GPT, dict[str, Any]]:
    """checkpoint から GPT の構造と学習済み重みを復元する。

    checkpoint 内の model_config で同じ形の GPT を作り、保存済みの state_dict を
    読み込むことで、train.py の学習結果を推論用モデルとして使える。
    """

    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")

    # checkpoint には学習済み重みだけでなく、モデル構造を復元する config も入っている。
    # map_location=device により、CPU/GPU のどちらで保存された checkpoint でも指定 device へ読める。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" not in checkpoint:
        raise SystemExit("checkpoint does not contain model_config")
    if "model" not in checkpoint:
        raise SystemExit("checkpoint does not contain model weights")

    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    # 生成時は Dropout を止め、同じ入力ならモデル本体の出力が安定するようにする。
    model.eval()
    return model, checkpoint


def prepare_bundle(
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: torch.device,
) -> ModelBundle:
    """tokenizer とモデルを読み込み、繰り返し生成に使える bundle を作る。

    tokenizer の語彙数とモデルの vocab_size が一致することもここで確認する。
    ここがずれると、同じ token-id が別の文字片を意味してしまい生成が壊れる。
    """

    tokenizer = load_tokenizer(tokenizer_path)
    model, checkpoint = load_model(checkpoint_path, device)

    # tokenizer とモデルの語彙数がずれていると、token id と logits の対応が壊れる。
    # 例えば id=1234 が学習時と別の文字片を指すと、生成結果は意味を失う。
    if tokenizer.get_piece_size() != model.config.vocab_size:
        raise SystemExit(
            f"tokenizer vocab size ({tokenizer.get_piece_size()}) does not match "
            f"model vocab size ({model.config.vocab_size})"
        )

    metadata: dict[str, Any] = {}
    if "step" in checkpoint:
        metadata["step"] = checkpoint["step"]
    if "val_loss" in checkpoint:
        metadata["val_loss"] = checkpoint["val_loss"]

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eos_id=tokenizer.eos_id(),
        metadata=metadata,
    )


def read_prompt(args: argparse.Namespace) -> str:
    """CLI の --prompt または --prompt-file から生成開始文を読み取る。"""

    if args.prompt_file is None:
        return args.prompt
    if not args.prompt_file.exists():
        raise SystemExit(f"prompt file does not exist: {args.prompt_file}")
    return args.prompt_file.read_text(encoding="utf-8")


def encode_prompt(processor: spm.SentencePieceProcessor, prompt: str) -> list[int]:
    """プロンプト文字列を、モデルへ入力できる token-id 列に変換する。

    モデルは文字列を直接読めないため、SentencePiece で数値列へ変換する。
    空プロンプトでは bos_id があれば「文章の始まり」から生成を始める。
    """

    # 生成は数値の token id 列から始まるため、まずプロンプト文字列を token id へ変換する。
    ids = processor.encode(prompt, out_type=int)
    if ids:
        return ids
    # 空プロンプトの場合でも、bos_id があれば「文章の始まり」トークンから生成できる。
    bos_id = processor.bos_id()
    if bos_id < 0:
        raise SystemExit("prompt is empty and tokenizer does not define bos_id")
    return [bos_id]


def filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """logits のうち上位 top_k 個以外を -inf にして、候補から除外する。

    softmax 前に -inf にすると確率が 0 になるため、サンプリングで選ばれなくなる。
    top_k=0 のときは無効として何も変更しない。
    """

    if top_k <= 0:
        return logits
    # top-k sampling: 確率が高い上位 k 個だけを候補に残す。
    # 低確率トークンを完全に除外することで、突飛な脱線を抑える。
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    return logits.masked_fill(logits < values[:, [-1]], -torch.inf)


def filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """累積確率が top_p に収まる高確率 token 群だけを残す。

    nucleus sampling と呼ばれる方法で、確信が強い場面では候補を少なくし、
    曖昧な場面では候補を広げるように、文脈に応じて候補数が変わる。
    """

    if top_p >= 1.0:
        return logits

    # top-p(nucleus) sampling:
    # 確率の高い順に足していき、累積確率が top_p を超えるまでの候補だけを残す。
    # top-k と違い、状況に応じて候補数が増減する。
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # top_p を初めて超えたトークンは残し、それ以降を除外する。
    # すべて消えてしまうとサンプリングできないため、先頭は必ず残す。
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    # sort 後の位置で作った除外 mask を、元の語彙 id の並びへ戻す。
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
    """直近に出た token の logits を下げ、同じ token の連発を抑える。

    生成モデルは一度同じ表現に入り込むと繰り返しやすいことがある。
    repetition_window 内に出た token を少し選ばれにくくして、その偏りを弱める。
    """

    if penalty == 1.0 or window == 0:
        return logits

    # repetition penalty は、直近に出た token の logits を下げて同じ語句の連発を抑える。
    # window を指定すると、古すぎる token までは罰しない。
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
        # logits が正なら割る、負なら掛けることで、どちらの場合も選ばれにくい方向へ動かす。
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
    """同じ n-gram をもう一度作る token を禁止する。

    n-gram は連続する n 個の token の並び。直前の n-1 token に続けると
    既出の n-gram が再現される token を -inf にして、反復文を減らす。
    """

    if ngram_size <= 1 or ids.size(1) < ngram_size - 1:
        return logits

    # no-repeat ngram は「同じ n-gram をもう一度作る token」を禁止する。
    # 例: n=4 で直前 3token が既出 4-gram の先頭 3token と一致したら、その 4token 目を ban する。
    prefix = tuple(int(token_id) for token_id in ids[0, -(ngram_size - 1) :].tolist())
    banned: set[int] = set()
    token_ids = ids[0].tolist()
    for index in range(len(token_ids) - ngram_size + 1):
        ngram = tuple(int(token_id) for token_id in token_ids[index : index + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])

    if banned:
        # -inf にすると softmax 後の確率が 0 になり、その token はサンプリングされない。
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
    """プロンプト token-id から始めて、続きを 1 token ずつ生成する。

    各ループでモデルが次 token の logits を出し、temperature/top-k/top-p や
    繰り返し抑制を適用したあと、次 token を 1 つ選んで文脈へ追加する。
    """

    # idx は現在の全文脈。shape は (batch=1, current_seq_len)。
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # GPT は block_size より長い文脈を一度に見られないため、末尾だけを条件にする。
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        # 次に生成する 1 トークンには、最後の位置の logits だけを使う。
        logits = logits[:, -1, :]

        # サンプリング前に logits へ制約をかけ、繰り返しすぎる出力を抑える。
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
            # temperature=0 は最も高い logit を選ぶ greedy decoding。毎回決定的な出力になる。
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # temperature が高いほど分布が平らになり、低確率 token も選ばれやすくなる。
            # 低いほど高確率 token に寄り、保守的な出力になる。
            logits = logits / temperature
            logits = filter_top_k(logits, top_k)
            logits = filter_top_p(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            # softmax で確率に変換したあと、確率に従って次 token を 1 つ選ぶ。
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)
        # eos_id は「文の終わり」を表す特別 token。指定時はそこで生成を打ち切る。
        if stop_at_eos and eos_id >= 0 and int(next_id.item()) == eos_id:
            break

    return idx[0].tolist()


def generate_text(
    bundle: ModelBundle,
    prompt: str,
    options: GenerationOptions,
) -> str:
    """読み込み済み bundle を使って、1 回分の文章生成を実行する。

    prompt を token-id 化し、generate_ids で続きを生成してから、tokenizer で
    文字列へ戻し、最後に表示しやすい形へ後処理する。
    """

    validate_options(options)

    if options.seed is not None:
        # seed を固定すると、multinomial sampling を使う生成でも結果を再現しやすい。
        torch.manual_seed(options.seed)
        if bundle.device.type == "cuda":
            torch.cuda.manual_seed_all(options.seed)

    input_ids = encode_prompt(bundle.tokenizer, prompt)
    output_ids = generate_ids(
        model=bundle.model,
        input_ids=input_ids,
        max_new_tokens=options.max_new_tokens,
        temperature=options.temperature,
        top_p=options.top_p,
        top_k=options.top_k,
        repetition_penalty=options.repetition_penalty,
        repetition_window=options.repetition_window,
        no_repeat_ngram_size=options.no_repeat_ngram_size,
        eos_id=bundle.eos_id,
        stop_at_eos=options.stop_at_eos,
        device=bundle.device,
    )
    # 最後に token id 列を文字列へ戻し、表示用の改行や記号だけを整える。
    return postprocess_generated_text(bundle.tokenizer.decode(output_ids))


def main() -> None:
    """generate.py を CLI として実行したときのエントリーポイント。

    引数読み取り、モデル/tokenizer 読み込み、プロンプト読み取り、生成、結果表示を行う。
    """

    args = parse_args()
    options = options_from_args(args)
    try:
        validate_options(options)
    except ValueError as error:
        raise SystemExit(str(error))

    device = resolve_device(args.device)
    bundle = prepare_bundle(args.checkpoint, args.tokenizer, device)

    prompt = read_prompt(args)
    text = generate_text(bundle, prompt, options)

    print(f"checkpoint: {args.checkpoint}")
    if "step" in bundle.metadata:
        print(f"step: {bundle.metadata['step']}")
    if "val_loss" in bundle.metadata:
        print(f"val loss: {bundle.metadata['val_loss']:.4f}")
    print(f"device: {device}")
    print()
    print(text)


if __name__ == "__main__":
    main()
