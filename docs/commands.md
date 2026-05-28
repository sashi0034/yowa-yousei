# Commands

必要なライブラリをインストールする。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch sentencepiece numpy tqdm
```

# clean_text.py: data/raw から data/processed/clean.txt を作成

先頭 3ファイルだけでクリーニングを試す。

```bash
python src/clean_text.py --limit 3 --reset
```

全 raw データをクリーニングして、個別ファイルと結合ファイルを作る。

```bash
python src/clean_text.py --reset
```

`clean_text.py` のオプションを見る。

```bash
python src/clean_text.py --help
```

# split_data.py: data/processed/clean.txt から train.txt, val.txt 作成

クリーニング済みコーパスを train / val に分割する。

```bash
python src/split_data.py
```

`split_data.py` のオプションを見る。

```bash
python src/split_data.py --help
```

# train_tokenizer.py: train.txt から SentencePiece モデル作成

開発用の小さいデータでトークナイザ学習を試す。

```bash
python src/train_tokenizer.py \
  --input data/processed/train_small.txt \
  --model-prefix tokenizer/yowa_yousei_sp_small
```

全 train データでトークナイザを学習する。

```bash
python src/train_tokenizer.py
```

ランダムサンプリングで学習を行う。

```bash
python src/train_tokenizer.py \
  --input-sentence-size 2000000
```

`train_tokenizer.py` のオプションを見る。

```bash
python src/train_tokenizer.py --help
```

# prepare_data.py: train.txt / val.txt を token id の .bin に変換

開発用の小さいデータで token id 変換を試す。
これは動作確認用で、本番学習には使わない。

```bash
python src/prepare_data.py \
  --tokenizer tokenizer/yowa_yousei_sp_small.model \
  --train data/processed/train_small.txt \
  --val data/processed/val_small.txt \
  --train-output data/processed/train_small.bin \
  --val-output data/processed/val_small.bin
```

全 train / val データを変換する。
本番学習に使う `.bin` は通常版 `tokenizer/yowa_yousei_sp.model` で作る。

```bash
python src/prepare_data.py
```

`prepare_data.py` のオプションを見る。

```bash
python src/prepare_data.py --help
```

# dataset.py: train.bin / val.bin から学習用バッチを作る

開発用の小さい `.bin` で DataLoader の動作を確認する。

```bash
python src/dataset.py \
  --train-bin data/processed/train_small.bin \
  --val-bin data/processed/val_small.bin \
  --device cuda
```

全データの `.bin` で DataLoader の動作を確認する。

```bash
python src/dataset.py
```

`dataset.py` のオプションを見る。

```bash
python src/dataset.py --help
```

# model.py: GPTモデル本体の動作確認

軽い設定で forward / loss 計算を確認する。

```bash
python src/model.py \
  --block-size 16 \
  --batch-size 2 \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 64 \
  --device cuda
```

todo.md 記載に近い構成で確認する (メモリを抑えるため、確認時だけ `batch-size` を小さくしている)。

```bash
python src/model.py --batch-size 1 --device cpu
```

`model.py` のオプションを見る。

```bash
python src/model.py --help
```

# train.py: GPTモデルを学習する

まずは CPU でもすぐ終わる小型設定で、学習ループと checkpoint 保存を確認する。

```bash
python src/train.py \
  --train-bin data/processed/train_small.bin \
  --val-bin data/processed/val_small.bin \
  --out-dir checkpoints/debug \
  --block-size 16 \
  --batch-size 2 \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 64 \
  --max-steps 2 \
  --eval-interval 1 \
  --eval-iters 1 \
  --gradient-accumulation-steps 2 \
  --log-interval 1 \
  --device cpu
```

GPU で初期設定の学習を始める。

```bash
python src/train.py --device cuda
```

GPU メモリが足りない場合は、まず `batch-size` を下げる。

```bash
python src/train.py \
  --device cuda \
  --batch-size 4 \
  --gradient-accumulation-steps 16
```

`train.py` のオプションを見る。

```bash
python src/train.py --help
```

# generate.py: checkpointから文章を生成する

学習済み checkpoint から続きを生成する。

```bash
python src/generate.py \
  --checkpoint checkpoints/best.pt \
  --tokenizer tokenizer/yowa_yousei_sp.model \
  --prompt "彼女は静かに目を覚ますと、そこは"
```

生成量や sampling を調整する。

```bash
python src/generate.py \
  --checkpoint checkpoints/best.pt \
  --tokenizer tokenizer/yowa_yousei_sp.model \
  --prompt "「どうしてここにいるの？」" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

開発用の小型checkpointを試す場合は、対応する小型tokenizerを使う。

```bash
python src/generate.py \
  --checkpoint checkpoints/debug/latest.pt \
  --tokenizer tokenizer/yowa_yousei_sp_small.model \
  --prompt "彼女は静かに目を覚ますと、そこは" \
  --max-new-tokens 50 \
  --device cpu
```

`generate.py` のオプションを見る。

```bash
python src/generate.py --help
```
