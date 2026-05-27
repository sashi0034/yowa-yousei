# 自作日本語 LLM 初号機 TODO

ゴール: SentencePiece + 小型GPT + PyTorch で Causal LM を最後まで学習・生成できる状態にする

---

## 0. 方針を固定する

- モデル方式を decoder-only Transformer / GPT型 にする
- 学習方式を Causal Language Modeling にする
- 最初の目的を日本語文章の続きを生成するモデルに絞る
- instruction tuning / LoRA / RLHF / DPO は後回しにする
- まずは「小さくても最後まで動く」ことを優先する

---

## 1. プロジェクト作成

```text
yowa-yousei/
  data/
    raw/
    processed/
  tokenizer/
  src/
  checkpoints/
```

- [ ] Python 仮想環境を作る
- [ ] 必要ライブラリを入れる

```bash
pip install torch sentencepiece numpy tqdm
```

余裕があれば追加。

```bash
pip install wandb safetensors
```

---

## 2. 生データを置く

- [x] 日本語文章データを `data/raw/*.txt` にまとめる
- [ ] 冗長なデータを削除
- [x] 区切りには `<eos>` 相当を使う想定にする?

例:

```text
妖精さんは言いました。
「こんにちは」
あれは嘘です。
<eos>
にゃんぱすー。
```

---

## 3. データクリーニング

`src/clean_text.py` を作る。

- [x] Unicode正規化 `NFKC` により、全角英数字・半角カナ・互換文字を正規化する
- [x] HTMLタグや制御文字除去
- [x] ルビ表記の除去または簡易変換
- [x] メタ文章を可能な範囲で除去
- [x] 余計な空白の整理
- [x] 出力を `data/processed/clean/*.txt` に保存する
- [x] 結合したものを `data/processed/clean.txt` として作成
- [x] 区切りは以下のようにしたい

```
<title>作品タイトル</title>
<chapter>話タイトル</chapter>
本文
<eos>
```

成功条件:

- [x] `clean/*.txt` を目視して、本文として読める
- [x] 変なメタ文が大量に残っていない
- [x] 文字化けがない

NOTE:

- 場面転換を複数空行で行っている小説があった気がする。対応を検討したい

---

## 4. train / val に分割する

`src/split_data.py` を作る。

- [x] `clean.txt` を `train.txt` と `val.txt` に分ける
- [x] しかし、`clean.txt` はサイズが非常に大きいため、開発効率化のために `train_small.txt`, `val_small.txt` を作る。
- [x] だいたい `train:val = 99:1` か `98:2`
- [x] 作品単位・話単位で分けられるなら、ランダムな行単位ではなく話単位で分ける

出力:

```text
data/processed/train.txt
data/processed/val.txt
```

成功条件:

- [x] `train.txt` が大半のデータを含む
- [x] `val.txt` が学習に混ざっていない
- [x] `val.txt` でも自然な日本語が読める

---

## 5. SentencePieceトークナイザを学習する

`src/train_tokenizer.py` を作る。

- [x] `train.txt` や `train_small.txt` から SentencePiece モデルを学習する
- [x] `vocab_size = 32000`
- [x] `model_type = unigram`
- [x] `character_coverage = 0.9995`
- [x] `byte_fallback = true`
- [x] 特殊トークンを固定する

```text
unk_id = 0
bos_id = 1
eos_id = 2
pad_id = 3
```

出力:

```text
tokenizer/yowa_yousei_sp.model       # train.txt で作る通常版
tokenizer/yowa_yousei_sp.vocab
tokenizer/yowa_yousei_sp_small.model # train_small.txt で作る軽量確認版
tokenizer/yowa_yousei_sp_small.vocab
```

コマンド例:

```bash
spm_train \
  --input=data/processed/train.txt \
  --model_prefix=tokenizer/yowa_yousei_sp \
  --vocab_size=32000 \
  --model_type=unigram \
  --character_coverage=0.9995 \
  --byte_fallback=true \
  --unk_id=0 \
  --bos_id=1 \
  --eos_id=2 \
  --pad_id=3
```

成功条件:

- [x] `yowa_yousei_sp_small.model` が生成される
- [x] 日本語文を encode / decode できる
- [x] decode 結果が大きく崩れない

確認用:

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/yowa_yousei_sp.model")

text = "彼女は静かに笑った。"
ids = sp.encode(text, out_type=int)

print(ids)
print(sp.decode(ids))
```

---

## 6. テキストを token id 列に変換する

`src/prepare_data.py` を作る。

- [x] `train.txt` を token id 列に変換する
- [x] `val.txt` を token id 列に変換する
- [x] 各話・各文書の末尾に `eos_id=2` を入れる
- [x] `np.uint16` で保存する
  - `vocab_size=32000` なら `uint16` で足りる

出力:

```text
data/processed/train.bin
data/processed/val.bin
```

成功条件:

- [ ] `train.bin` と `val.bin` が生成される
  - 通常版 `tokenizer/yowa_yousei_sp.model` 作成後に実行する
- [x] token 数を表示できる
- [x] 数百トークンを decode して自然な文章に戻る

確認済み:

- [x] `yowa_yousei_sp_small.model` で `train_small.bin` / `val_small.bin` を生成できる

---

## 7. DataLoader を作る

`src/dataset.py` または `train.py` 内に作る。

- [x] `train.bin` / `val.bin` を `np.memmap` で読む
- [x] ランダム位置から `block_size + 1` 個の token を切り出す
- [x] `x = tokens[i : i + block_size]`
- [x] `y = tokens[i + 1 : i + 1 + block_size]`
- [x] GPU に転送する

初期設定:

```text
block_size = 512
batch_size = 8
gradient_accumulation_steps = 8
```

成功条件:

- [x] `x.shape == (batch_size, block_size)`
- [x] `y.shape == (batch_size, block_size)`
- [x] `y` が `x` の1トークン先になっている

確認済み:

- [x] `train_small.bin` / `val_small.bin` で DataLoader の動作確認ができる
- [ ] `train.bin` / `val.bin` で DataLoader の動作確認ができる

---

## 8. GPT モデルを実装する

`src/model.py` を作る。

最初の構成:

```text
vocab_size = 32000
block_size = 512
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
```

実装TODO:

- [x] Token Embedding
- [x] Position Embedding
- [x] Transformer Block
- [x] Causal Self-Attention
- [x] Feed Forward
- [x] LayerNorm
- [x] LM Head
- [x] causal mask
- [x] forwardで `logits` と `loss` を返す

損失:

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1)
)
```

成功条件:

- [x] `model(x, y)` が通る
- [x] `logits.shape == (batch_size, block_size, vocab_size)`
- [x] 初期lossがだいたい `log(32000) ≒ 10.37` 付近になる

確認済み:

- [x] 軽量設定で `logits.shape == (2, 16, 32000)` / `loss == 10.3731`
- [x] 初期設定のモデル幅で `logits.shape == (1, 512, 32000)` / `loss == 10.5075`
- [x] `train.bin` 由来の小さいバッチで `model(x, y)` が通る

---

## 9. 学習ループを書く

`src/train.py` を作る。

- [x] AdamW を使う
- [x] learning rate を `3e-4` から始める
- [x] warmupを 入れる
- [x] cosine decay を入れる
- [x] gradient accumulation を入れる
- [x] gradient clipping を入れる
- [x] AMPを入れる
  - NVIDIA GPU なら `bf16` または `fp16`
- [x] 定期的に validation loss を計算する
- [x] checkpoint 保存する

初期設定:

```text
max_steps = 50000
eval_interval = 500
eval_iters = 100
learning_rate = 3e-4
weight_decay = 0.1
grad_clip = 1.0
warmup_steps = 1000
```

成功条件:

- [ ] loss が `10.3 → 8 → 6 → 5...` のように下がる
- [ ] train loss だけでなく val loss も下がる
- [x] checkpoint が保存される
- [ ] GPU メモリが破裂しない

確認済み:

- [x] 小型設定で2step学習が通る
- [x] train / val loss を表示できる
- [x] `best.pt` / `latest.pt` を保存できる

---

## 10. 生成スクリプトを書く

`src/generate.py` を作る。

- [ ] checkpoint を読み込む
- [ ] SentencePiece tokenizer を読み込む
- [ ] prompt を encode する
- [ ] autoregressive に 1トークンずつ生成する
- [ ] temperature sampling を入れる
- [ ] top-k または top-p sampling を入れる

初期設定:

```text
temperature = 0.8
top_p = 0.9
max_new_tokens = 200
```

成功条件:

- [ ] prompt から日本語の続きを生成できる
- [ ] 学習が進むほど出力が自然になる
- [ ] 同じ prompt で checkpoint ごとの差を確認できる

---

## 11. 固定プロンプト評価を作る

毎回同じプロンプトで生成する。

例:

```text
彼女は静かに目を覚ますと、そこは
「どうしてここにいるの？」
```

TODO:

- [ ] `prompts.txt` を作る
- [ ] checkpointごとに生成結果を保存する
- [ ] 破綻・繰り返し・文体・句読点を目視確認する

成功条件:

- [ ] lossだけでなく実際の生成品質を比較できる
- [ ] 過学習や繰り返し癖に気づける

---

## 12. 最初の改善ポイント

初号機が動いた後にやる。

### データ改善

- [ ] 変なメタ文をさらに除去する
- [ ] 重複作品を削る
- [ ] 作品ごとの偏りを確認する
- [ ] 短すぎる文書を除外する
- [ ] `<eos>` の入れ方を改善する

### モデル改善

- [ ] `block_size = 1024` に上げる
- [ ] `n_layer = 12` に上げる
- [ ] `n_embd = 768` に上げる
- [ ] dropout を調整する
- [ ] tokenizer vocab を `16000` と `32000` で比較する

### 学習改善

- [ ] learning rate を調整する
- [ ] batch size / accumulation を調整する
- [ ] warmup_steps を調整する
- [ ] val loss が悪化し始める地点を確認する
- [ ] checkpoint averaging は後回しでよい

---

## 13. 後回しでよいもの

最初はやらない。

- [ ] MoE
- [ ] RoPE
- [ ] FlashAttention
- [ ] 分散学習
- [ ] RLHF
- [ ] DPO
- [ ] instruction tuning
- [ ] chat template
- [ ] RAG
- [ ] tokenizer 自作
- [ ] CUDA kernel 自作

特に **FlashAttention / RoPE / DPO / LoRA** あたりは魅力的だが、初号機では不要。  
まずは普通の GPT で、loss が下がって文章が出るところまで行く。

---

## 最短チェックリスト

まずここまで行けば勝ち。

```text
[ ] clean.txt ができた
[ ] train.txt / val.txt ができた
[ ] yowa_yousei_sp.model ができた
[ ] train.bin / val.bin ができた
[ ] get_batch() が動いた
[ ] model(x, y) が動いた
[ ] 初期 loss が約 10.37 になった
[ ] train loss が下がった
[ ] checkpoint 保存できた
[ ] generate.py で日本語が出た
```
