import bolt from "@slack/bolt";
import "dotenv/config";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const { App, LogLevel } = bolt;

type SlackClient = {
  reactions: {
    add(args: { channel: string; name: string; timestamp: string }): Promise<unknown>;
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
};

type GenerateConfig = {
  repoRoot: string;
  pythonBin: string;
  scriptPath: string;
  checkpointPath: string;
  tokenizerPath: string;
  maxNewTokens?: string;
  temperature?: string;
  topP?: string;
  topK?: string;
  device?: string;
  stopAtEos: boolean;
  timeoutMs: number;
};

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = process.env.REPO_ROOT ?? path.resolve(currentDir, "..", "..");

const generateConfig: GenerateConfig = {
  repoRoot,
  pythonBin: process.env.PYTHON_BIN ?? "python",
  scriptPath: process.env.GENERATE_SCRIPT ?? "src/generate.py",
  checkpointPath: process.env.CHECKPOINT_PATH ?? "checkpoints/best.pt",
  tokenizerPath: process.env.TOKENIZER_PATH ?? "tokenizer/yowa_yousei_sp.model",
  maxNewTokens: process.env.MAX_NEW_TOKENS,
  temperature: process.env.TEMPERATURE,
  topP: process.env.TOP_P,
  topK: process.env.TOP_K,
  device: process.env.DEVICE,
  stopAtEos: parseBoolean(process.env.STOP_AT_EOS, false),
  timeoutMs: parsePositiveInt(process.env.GENERATION_TIMEOUT_MS, 300_000),
};

const app = new App({
  token: requiredEnv("SLACK_BOT_TOKEN"),
  appToken: requiredEnv("SLACK_APP_TOKEN"),
  socketMode: true,
  logLevel: LogLevel.INFO,
});

const queue: QueueItem[] = [];
const seenMessages = new Set<string>();
const maxSeenMessages = 1_000;
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

  try {
    await client.reactions.add({
      channel,
      name: "heartbeat",
      timestamp: messageTs,
    });
  } catch (error) {
    logger.warn(`failed to add heartbeat reaction: ${formatError(error)}`);
  }

  queue.push({
    channel,
    messageTs,
    prompt,
  });

  void drainQueue(client, logger);
});

async function drainQueue(
  client: SlackClient,
  logger: Logger,
): Promise<void> {
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
    const result = await runGenerate(item.prompt, generateConfig);
    await client.chat.postMessage({
      channel: item.channel,
      text: truncateForSlack(result || "(生成結果が空でした)"),
    });
  } catch (error) {
    logger.error(`generation failed for ${item.channel}:${item.messageTs}: ${formatError(error)}`);
    await client.chat.postMessage({
      channel: item.channel,
      text: `生成に失敗しました: ${formatError(error)}`,
    });
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

function runGenerate(prompt: string, config: GenerateConfig): Promise<string> {
  const args = [
    config.scriptPath,
    "--checkpoint",
    config.checkpointPath,
    "--tokenizer",
    config.tokenizerPath,
    "--prompt",
    prompt,
  ];

  addOptionalArg(args, "--max-new-tokens", config.maxNewTokens);
  addOptionalArg(args, "--temperature", config.temperature);
  addOptionalArg(args, "--top-p", config.topP);
  addOptionalArg(args, "--top-k", config.topK);
  addOptionalArg(args, "--device", config.device);
  if (config.stopAtEos) {
    args.push("--stop-at-eos");
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    let closed = false;
    const child = spawn(config.pythonBin, args, {
      cwd: config.repoRoot,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      setTimeout(() => {
        if (!closed) {
          child.kill("SIGKILL");
        }
      }, 5_000);
      settled = true;
      reject(new Error(`generation timed out after ${config.timeoutMs}ms`));
    }, config.timeoutMs);

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });

    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    child.on("error", (error) => {
      clearTimeout(timer);
      settled = true;
      reject(error);
    });

    child.on("close", (code, signal) => {
      closed = true;
      clearTimeout(timer);
      if (settled) {
        return;
      }
      settled = true;
      if (code !== 0) {
        reject(new Error(stderr.trim() || `generation exited with code ${code ?? signal}`));
        return;
      }
      resolve(extractGeneratedText(stdout));
    });
  });
}

function extractGeneratedText(stdout: string): string {
  const normalized = stdout.replace(/\r\n/g, "\n").trimEnd();
  const separatorIndex = normalized.indexOf("\n\n");
  if (separatorIndex === -1) {
    return normalized.trim();
  }
  return normalized.slice(separatorIndex + 2).trim();
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

function addOptionalArg(args: string[], flag: string, value: string | undefined): void {
  if (value !== undefined && value !== "") {
    args.push(flag, value);
  }
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

await app.start();
console.log("slack-chat is running in Socket Mode");
