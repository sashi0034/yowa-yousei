# slack-chat

Slack のチャンネル投稿を Socket Mode で受け取り、`Q.` から始まるメッセージを生成サーバの prompt として実行します。解釈したメッセージには `:heartbeat:` リアクションを付け、生成結果は同じチャンネルへ投稿します。

起動時に `src/generate_server.py` を 1 回だけ `spawn` し、checkpoint とトークナイザをメモリに常駐させ続けます。プロンプトのたびに Python を立ち上げ直したり、`torch.load` や `state_dict` のロードを繰り返したりしません。

メッセージは 1 件ずつ順番に処理します。複数の `Q.` 投稿が一気に来ても、生成は直列に行われます。生成サーバが応答せずタイムアウトした場合はプロセスを再起動し、次のリクエストはそちらで処理します。

## セットアップ

```bash
cd slack-chat
npm install
cp .env.example .env
```

`.env` に Slack token と生成設定を書きます。

```bash
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...

PYTHON_BIN=python
GENERATE_SERVER_SCRIPT=src/generate_server.py
CHECKPOINT_PATH=checkpoints/best.pt
TOKENIZER_PATH=tokenizer/yowa_yousei_sp.model
DEVICE=auto
```

開発実行:

```bash
npm run dev
```

ビルドして実行:

```bash
npm run build
npm start
```

## Slack App 設定

Slack App の **Socket Mode** を有効にして、App-Level Token を作成します。

App-Level Token:

- `connections:write`

Bot Token Scopes:

- `chat:write`
- `reactions:write`
- `channels:history` public channel の投稿を読む場合
- `groups:history` private channel の投稿を読む場合
- `im:history` DM の投稿を読む場合
- `mpim:history` group DM の投稿を読む場合

Event Subscriptions の Bot Events:

- `message.channels` public channel の投稿を扱う場合
- `message.groups` private channel の投稿を扱う場合
- `message.im` DM を扱う場合
- `message.mpim` group DM を扱う場合

private channel で使う場合は、Bot をそのチャンネルに招待してください。

### App Manifest YAML 例

Slack App 作成時に manifest を使う場合は、次の YAML をベースにできます。

```yaml
_metadata:
  major_version: 1
display_information:
  name: yowa-yousei
  description: Q. から始まる投稿に生成文で返信する bot
features:
  bot_user:
    display_name: yowa-yousei
    always_online: false
oauth_config:
  scopes:
    bot:
      - chat:write
      - reactions:write
      - channels:history
      - groups:history
      - im:history
      - mpim:history
settings:
  event_subscriptions:
    bot_events:
      - message.channels
      - message.groups
      - message.im
      - message.mpim
  org_deploy_enabled: false
  socket_mode_enabled: true
  is_hosted: false
  token_rotation_enabled: false
```

manifest で `socket_mode_enabled: true` にしても、`SLACK_APP_TOKEN` に入れる App-Level Token は別途 Slack App の **Basic Information > App-Level Tokens** から作成します。scope は `connections:write` を付けてください。

## 使い方

Slack に次のように投稿します。

```text
Q. 彼女は静かに目を覚ますと、そこは
```

全角も正規化します。

```text
Ｑ．彼女は静かに目を覚ますと、そこは
```

内部では起動時に次の形でサーバプロセスが立ち上がり、メッセージごとに stdin/stdout の JSON で prompt をやり取りします。

```bash
python src/generate_server.py \
  --checkpoint checkpoints/best.pt \
  --tokenizer tokenizer/yowa_yousei_sp.model
```

`src/generate.py` の単発 CLI は今までどおりそのまま使えます。

## 環境変数

- `SLACK_BOT_TOKEN`: Bot User OAuth Token。必須。
- `SLACK_APP_TOKEN`: Socket Mode 用 App-Level Token。必須。
- `REPO_ROOT`: リポジトリルート。未指定なら `slack-chat` の親ディレクトリ。
- `PYTHON_BIN`: Python コマンド。既定値は `python`。
- `GENERATE_SERVER_SCRIPT`: 生成サーバスクリプト。既定値は `src/generate_server.py`。
- `CHECKPOINT_PATH`: checkpoint path。既定値は `checkpoints/best.pt`。
- `TOKENIZER_PATH`: tokenizer path。既定値は `tokenizer/yowa_yousei_sp.model`。
- `MAX_NEW_TOKENS`: 起動時に `--max-new-tokens` として渡すデフォルト値。
- `TEMPERATURE`: 起動時に `--temperature` として渡すデフォルト値。
- `TOP_P`: 起動時に `--top-p` として渡すデフォルト値。
- `TOP_K`: 起動時に `--top-k` として渡すデフォルト値。
- `DEVICE`: 起動時に `--device` として渡す値。
- `STOP_AT_EOS`: `true` のとき起動引数に `--stop-at-eos` を付ける。
- `READY_TIMEOUT_MS`: 生成サーバが ready を返すまでの timeout。既定値は `180000`。
- `GENERATION_TIMEOUT_MS`: 1 件あたりの timeout。既定値は `300000`。timeout 時はサーバを再起動して次のリクエストに備えます。
