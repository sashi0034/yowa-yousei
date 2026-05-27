# Commands

必要なライブラリをインストールする。

```bash
pip install torch sentencepiece numpy tqdm
```

先頭3ファイルだけでクリーニングを試す。

```bash
python3 src/clean_text.py --limit 3 --reset
```

全rawデータをクリーニングして、個別ファイルと結合ファイルを作る。

```bash
python3 src/clean_text.py --reset
```

`clean_text.py` のオプションを見る。

```bash
python3 src/clean_text.py --help
```

クリーニング済みコーパスを train / val に分割する。

```bash
python3 src/split_data.py
```

`split_data.py` のオプションを見る。

```bash
python3 src/split_data.py --help
```
