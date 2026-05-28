import bolt from "@slack/bolt";
import "dotenv/config";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  GenerationServer,
  type GenerationLogger,
  type GenerationServerConfig,
} from "./generation-server.js";

const { App, LogLevel } = bolt;

type SlackClient = {
  reactions: {
    add(args: { channel: string; name: string; timestamp: string }): Promise<unknown>;
    remove(args: { channel: string; name: string; timestamp: string }): Promise<unknown>;
  };
  chat: {
    postMessage(args: { channel: string; text: string }): Promise<unknown>;
  };
};

type Logger = {
  warn(message: string): void;
  error(message: string): void;
};

type QueueItem = {
  channel: string;
  messageTs: string;
  prompt: string;
  hasQueueReaction: boolean;
};

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = process.env.REPO_ROOT ?? path.resolve(currentDir, "..", "..");

const serverConfig: GenerationServerConfig = {
  repoRoot,
  pythonBin: process.env.PYTHON_BIN ?? "python",
  serverScriptPath: process.env.GENERATE_SERVER_SCRIPT ?? "src/generate_server.py",
  checkpointPath: process.env.CHECKPOINT_PATH ?? "checkpoints/best.pt",
  tokenizerPath: process.env.TOKENIZER_PATH ?? "tokenizer/yowa_yousei_sp.model",
  device: process.env.DEVICE,
  maxNewTokens: process.env.MAX_NEW_TOKENS,
  temperature: process.env.TEMPERATURE,
  topP: process.env.TOP_P,
  topK: process.env.TOP_K,
  stopAtEos: parseBoolean(process.env.STOP_AT_EOS, false),
  readyTimeoutMs: parsePositiveInt(process.env.READY_TIMEOUT_MS, 180_000),
  requestTimeoutMs: parsePositiveInt(process.env.GENERATION_TIMEOUT_MS, 300_000),
};

const app = new App({
  token: requiredEnv("SLACK_BOT_TOKEN"),
  appToken: requiredEnv("SLACK_APP_TOKEN"),
  socketMode: true,
  logLevel: LogLevel.INFO,
});

const serverLogger: GenerationLogger = {
  info(message) {
    console.log(`[generation-server] ${message}`);
  },
  warn(message) {
    console.warn(`[generation-server] ${message}`);
  },
  error(message) {
    console.error(`[generation-server] ${message}`);
  },
};

const generationServer = new GenerationServer(serverConfig, serverLogger);

const queue: QueueItem[] = [];
const seenMessages = new Set<string>();
const maxSeenMessages = 1_000;
const queueReactionName = "heartbeat";
let processing = false;

app.message(async ({ message, client, logger }) => {
  if (!isUserMessage(message)) {
    return;
  }

  const prompt = extractPrompt(message.text);
  if (prompt === null) {
    return;
  }

  const channel = message.channel;
  const messageTs = message.ts;
  const key = `${channel}:${messageTs}`;
  if (seenMessages.has(key)) {
    return;
  }
  rememberMessage(key);

  let hasQueueReaction = false;
  try {
    await client.reactions.add({
      channel,
      name: queueReactionName,
      timestamp: messageTs,
    });
    hasQueueReaction = true;
  } catch (error) {
    logger.warn(`failed to add ${queueReactionName} reaction: ${formatError(error)}`);
  }

  queue.push({
    channel,
    messageTs,
    prompt,
    hasQueueReaction,
  });

  void drainQueue(client, logger);
});

async function drainQueue(client: SlackClient, logger: Logger): Promise<void> {
  if (processing) {
    return;
  }

  processing = true;
  try {
    while (queue.length > 0) {
      const item = queue.shift();
      if (item === undefined) {
        continue;
      }
      await handleQueueItem(item, client, logger);
    }
  } finally {
    processing = false;
    if (queue.length > 0) {
      void drainQueue(client, logger);
    }
  }
}

async function handleQueueItem(
  item: QueueItem,
  client: SlackClient,
  logger: Logger,
): Promise<void> {
  try {
    const result = await generationServer.generate(item.prompt);
    await client.chat.postMessage({
      channel: item.channel,
      text: truncateForSlack(result || "(生成結果が空でした)"),
    });
  } catch (error) {
    logger.error(
      `generation failed for ${item.channel}:${item.messageTs}: ${formatError(error)}`,
    );
    await client.chat.postMessage({
      channel: item.channel,
      text: `生成に失敗しました: ${formatError(error)}`,
    });
  } finally {
    if (item.hasQueueReaction) {
      await removeQueueReaction(item, client, logger);
    }
  }
}

async function removeQueueReaction(
  item: QueueItem,
  client: SlackClient,
  logger: Logger,
): Promise<void> {
  try {
    await client.reactions.remove({
      channel: item.channel,
      name: queueReactionName,
      timestamp: item.messageTs,
    });
  } catch (error) {
    logger.warn(`failed to remove ${queueReactionName} reaction: ${formatError(error)}`);
  }
}

function extractPrompt(text: string | undefined): string | null {
  if (text === undefined) {
    return null;
  }

  const normalized = text.normalize("NFKC").trimStart();
  const match = normalized.match(/^Q[.。]\s*(?<prompt>[\s\S]*)$/);
  if (match === null) {
    return null;
  }

  const prompt = match.groups?.prompt.trim();
  return prompt ? prompt : null;
}

function isUserMessage(message: unknown): message is {
  channel: string;
  ts: string;
  text?: string;
  subtype?: string;
  bot_id?: string;
} {
  if (typeof message !== "object" || message === null) {
    return false;
  }
  const candidate = message as Record<string, unknown>;
  return (
    typeof candidate.channel === "string" &&
    typeof candidate.ts === "string" &&
    candidate.subtype === undefined &&
    candidate.bot_id === undefined
  );
}

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined || value === "") {
    return fallback;
  }
  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (value === undefined || value === "") {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value === undefined || value === "") {
    throw new Error(`${name} is required`);
  }
  return value;
}

function formatError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function rememberMessage(key: string): void {
  seenMessages.add(key);
  if (seenMessages.size <= maxSeenMessages) {
    return;
  }

  const oldestKey = seenMessages.values().next().value;
  if (oldestKey !== undefined) {
    seenMessages.delete(oldestKey);
  }
}

function truncateForSlack(text: string): string {
  const maxLength = 39_000;
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength)}\n\n...(長すぎるため省略しました)`;
}

function shutdown(signal: NodeJS.Signals): void {
  console.log(`received ${signal}, shutting down`);
  generationServer.shutdown();
  void app
    .stop()
    .catch((error) => console.error(`failed to stop app cleanly: ${formatError(error)}`))
    .finally(() => process.exit(0));
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

await app.start();
console.log("slack-chat is running in Socket Mode");
