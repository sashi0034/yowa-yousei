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
