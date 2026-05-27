# Commands

必要なライブラリをインストールする。

```bash
pip install torch sentencepiece numpy tqdm
```

# clean_text.py: data/raw から data/processed/clean.txt を作成

先頭 3ファイルだけでクリーニングを試す。

```bash
python3 src/clean_text.py --limit 3 --reset
```

全 raw データをクリーニングして、個別ファイルと結合ファイルを作る。

```bash
python3 src/clean_text.py --reset
```

`clean_text.py` のオプションを見る。

```bash
python3 src/clean_text.py --help
```

# split_data.py: data/raw から data/processed/train.txt, val.txt 作成

クリーニング済みコーパスを train / val に分割する。

```bash
python3 src/split_data.py
```

`split_data.py` のオプションを見る。

```bash
python3 src/split_data.py --help
```
