import { spawn, type ChildProcess } from "node:child_process";

export type GenerationServerConfig = {
  repoRoot: string;
  pythonBin: string;
  serverScriptPath: string;
  checkpointPath: string;
  tokenizerPath: string;
  device?: string;
  maxNewTokens?: string;
  temperature?: string;
  topP?: string;
  topK?: string;
  stopAtEos: boolean;
  readyTimeoutMs: number;
  requestTimeoutMs: number;
};

export type GenerationLogger = {
  info(message: string): void;
  warn(message: string): void;
  error(message: string): void;
};

type PendingRequest = {
  resolve: (text: string) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
};

export class GenerationServer {
  private child: ChildProcess | null = null;
  private stdoutBuffer = "";
  private readonly pending = new Map<string, PendingRequest>();
  private nextId = 1;
  private readyPromise: Promise<void> | null = null;
  private resolveReady: (() => void) | null = null;
  private rejectReady: ((error: Error) => void) | null = null;
  private readyTimer: NodeJS.Timeout | null = null;
  private shuttingDown = false;

  constructor(
    private readonly config: GenerationServerConfig,
    private readonly logger: GenerationLogger,
  ) {}

  async generate(prompt: string): Promise<string> {
    await this.ensureReady();
    const child = this.child;
    if (child === null || child.stdin === null || child.stdin.destroyed) {
      throw new Error("generation server is not running");
    }

    const id = `req-${this.nextId++}`;
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        if (!this.pending.has(id)) {
          return;
        }
        this.pending.delete(id);
        reject(new Error(`generation timed out after ${this.config.requestTimeoutMs}ms`));
        this.reset(`timeout for request ${id}`);
      }, this.config.requestTimeoutMs);

      this.pending.set(id, { resolve, reject, timer });

      const payload = JSON.stringify({ id, prompt }) + "\n";
      child.stdin!.write(payload, "utf8", (writeError) => {
        if (writeError === null || writeError === undefined) {
          return;
        }
        const pending = this.pending.get(id);
        if (pending === undefined) {
          return;
        }
        clearTimeout(pending.timer);
        this.pending.delete(id);
        reject(writeError);
      });
    });
  }

  shutdown(): void {
    this.shuttingDown = true;
    this.reset("shutdown requested");
  }

  private async ensureReady(): Promise<void> {
    if (this.readyPromise === null) {
      this.start();
    }
    await this.readyPromise!;
  }

  private start(): void {
    const args = this.buildServerArgs();
    this.logger.info(
      `starting generation server: ${this.config.pythonBin} ${args.join(" ")}`,
    );

    const child = spawn(this.config.pythonBin, args, {
      cwd: this.config.repoRoot,
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child = child;
    this.stdoutBuffer = "";

    this.readyPromise = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    this.readyPromise.catch(() => {
      // Avoid unhandled rejections; callers handle the rejection through ensureReady.
    });
    this.readyTimer = setTimeout(() => {
      this.fail(
        new Error(
          `generation server did not become ready within ${this.config.readyTimeoutMs}ms`,
        ),
      );
    }, this.config.readyTimeoutMs);

    child.stdout!.setEncoding("utf8");
    child.stderr!.setEncoding("utf8");
    child.stdin!.setDefaultEncoding("utf8");

    child.stdout!.on("data", (chunk: string) => this.handleStdoutChunk(chunk));
    child.stderr!.on("data", (chunk: string) => this.handleStderrChunk(chunk));
    child.on("error", (error) => {
      this.logger.error(`generation server process error: ${error.message}`);
      this.fail(error);
    });
    child.on("exit", (code, signal) => this.handleExit(code, signal));
  }

  private buildServerArgs(): string[] {
    const args = [
      this.config.serverScriptPath,
      "--checkpoint",
      this.config.checkpointPath,
      "--tokenizer",
      this.config.tokenizerPath,
    ];
    addOptionalArg(args, "--device", this.config.device);
    addOptionalArg(args, "--max-new-tokens", this.config.maxNewTokens);
    addOptionalArg(args, "--temperature", this.config.temperature);
    addOptionalArg(args, "--top-p", this.config.topP);
    addOptionalArg(args, "--top-k", this.config.topK);
    if (this.config.stopAtEos) {
      args.push("--stop-at-eos");
    }
    return args;
  }

  private handleStdoutChunk(chunk: string): void {
    this.stdoutBuffer += chunk;
    let newlineIndex = this.stdoutBuffer.indexOf("\n");
    while (newlineIndex !== -1) {
      const rawLine = this.stdoutBuffer.slice(0, newlineIndex);
      this.stdoutBuffer = this.stdoutBuffer.slice(newlineIndex + 1);
      const line = rawLine.replace(/\r$/, "").trim();
      if (line !== "") {
        this.handleLine(line);
      }
      newlineIndex = this.stdoutBuffer.indexOf("\n");
    }
  }

  private handleStderrChunk(chunk: string): void {
    const lines = chunk.replace(/\r/g, "").split("\n");
    for (const line of lines) {
      if (line === "") {
        continue;
      }
      this.logger.info(`[generate_server] ${line}`);
    }
  }

  private handleLine(line: string): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      this.logger.warn(`generation server emitted non-json stdout line: ${line}`);
      return;
    }
    if (typeof parsed !== "object" || parsed === null) {
      this.logger.warn(`generation server emitted non-object stdout line: ${line}`);
      return;
    }
    const message = parsed as Record<string, unknown>;
    if (message.event === "ready") {
      this.handleReady(message);
      return;
    }
    this.handleResponse(message);
  }

  private handleReady(message: Record<string, unknown>): void {
    if (this.readyTimer !== null) {
      clearTimeout(this.readyTimer);
      this.readyTimer = null;
    }
    const resolveReady = this.resolveReady;
    this.resolveReady = null;
    this.rejectReady = null;
    this.logger.info(`generation server ready: ${JSON.stringify(message)}`);
    if (resolveReady !== null) {
      resolveReady();
    }
  }

  private handleResponse(message: Record<string, unknown>): void {
    const id = message.id;
    if (typeof id !== "string") {
      this.logger.warn(
        `generation server response missing string id: ${JSON.stringify(message)}`,
      );
      return;
    }
    const pending = this.pending.get(id);
    if (pending === undefined) {
      this.logger.warn(`generation server response for unknown id ${id}`);
      return;
    }
    clearTimeout(pending.timer);
    this.pending.delete(id);
    if (message.ok === true) {
      const text = typeof message.text === "string" ? message.text : "";
      pending.resolve(text);
      return;
    }
    const errorMessage =
      typeof message.error === "string" ? message.error : "unknown server error";
    pending.reject(new Error(errorMessage));
  }

  private handleExit(code: number | null, signal: NodeJS.Signals | null): void {
    const description = `code=${code ?? "?"} signal=${signal ?? "?"}`;
    if (this.shuttingDown) {
      this.logger.info(`generation server stopped (${description})`);
    } else {
      this.logger.error(`generation server exited unexpectedly (${description})`);
    }
    if (
      this.child === null &&
      this.readyPromise === null &&
      this.pending.size === 0
    ) {
      return;
    }
    this.fail(new Error(`generation server exited (${description})`));
  }

  private fail(error: Error): void {
    if (this.readyTimer !== null) {
      clearTimeout(this.readyTimer);
      this.readyTimer = null;
    }
    const rejectReady = this.rejectReady;
    this.resolveReady = null;
    this.rejectReady = null;
    this.readyPromise = null;
    this.child = null;
    this.stdoutBuffer = "";
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
    if (rejectReady !== null) {
      rejectReady(error);
    }
  }

  private reset(reason: string): void {
    if (this.child === null) {
      return;
    }
    this.logger.warn(`resetting generation server: ${reason}`);
    const child = this.child;
    this.fail(new Error(`generation server reset: ${reason}`));
    try {
      child.stdin?.end();
    } catch {
      // ignore
    }
    try {
      child.kill("SIGTERM");
    } catch {
      // ignore
    }
    setTimeout(() => {
      try {
        if (child.exitCode === null && child.signalCode === null) {
          child.kill("SIGKILL");
        }
      } catch {
        // ignore
      }
    }, 5_000);
  }
}

function addOptionalArg(args: string[], flag: string, value: string | undefined): void {
  if (value !== undefined && value !== "") {
    args.push(flag, value);
  }
}
