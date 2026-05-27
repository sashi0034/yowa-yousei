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

`train_tokenizer.py` のオプションを見る。

```bash
python src/train_tokenizer.py --help
```
