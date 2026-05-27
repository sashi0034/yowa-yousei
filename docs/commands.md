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

# split_data.py: data/raw から data/processed/train.txt, val.txt 作成

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
