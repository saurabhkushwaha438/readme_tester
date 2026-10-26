import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { Writable } from "node:stream";
import type { Duplex } from "node:stream";
import Docker from "dockerode";

/**
 * Phase 2: the Docker layer.
 *
 * Every function here takes a container id, never a Docker client, so the
 * graph state stays plain JSON (Phase 9's checkpointer and any resume depend on
 * that). The client is created lazily once per process.
 *
 * Tools: startContainer, exec (with a timeout), readFile, listFiles, destroy.
 */

/** Where the host clone is mounted inside the container. */
export const CONTAINER_WORKDIR = "/workspace";

/**
 * Image used for a run. Repo setup commands decide what this needs, so it is
 * overridable: `README_TEST_IMAGE=python:3.12-bookworm npm start -- <url>`.
 * The default ships git, node, npm, bash, coreutils and GNU find — everything
 * listFiles and the timeout wrapper rely on.
 */
export const DEFAULT_IMAGE = process.env.README_TEST_IMAGE ?? "node:22-bookworm";

/** Wall clock allowed for a single command, overridable per call. */
export const DEFAULT_EXEC_TIMEOUT_MS = Number(
  process.env.README_TEST_EXEC_TIMEOUT_MS ?? 120_000,
);

/**
 * Bytes of stdout/stderr kept for the LLM. The plan calls this non-optional:
 * a single `npm install --verbose` can emit megabytes, and every one of those
 * bytes costs Groq tokens.
 */
export const LLM_OUTPUT_TAIL_BYTES = 2048;

/** readFile keeps the head of the file instead, up to this many bytes. */
const MAX_READ_BYTES = 1024 * 1024;

let client: Docker | undefined;

/** The shared Docker client. Connects lazily; honours DOCKER_HOST. */
export function docker(): Docker {
  client ??= new Docker();
  return client;
}

/**
 * Containers this process started and has not removed yet.
 *
 * Phase 9 wants a `finally` destroy, but state is not available after a node
 * throws. Tracking ids here lets the CLI clean up whatever a failed run left
 * behind, instead of accumulating stopped containers across runs.
 */
const active = new Set<string>();

/** Fail fast with an actionable message when no daemon is listening. */
export async function pingDocker(): Promise<void> {
  try {
    await docker().ping();
  } catch (err) {
    throw new Error(
      `Docker daemon unreachable (${message(err)}). Start Docker Desktop or dockerd and retry.`,
    );
  }
}

const message = (err: unknown): string => (err instanceof Error ? err.message : String(err));

const isNotFound = (err: unknown): boolean =>
  typeof err === "object" && err !== null && (err as { statusCode?: number }).statusCode === 404;

/** POSIX single-quote a string so it survives `bash -lc`. */
const shellQuote = (value: string): string => `'${value.replace(/'/g, `'\\''`)}'`;

// ---------------------------------------------------------------------------
// Output capture
// ---------------------------------------------------------------------------

type Keep = "head" | "tail";

/**
 * A Writable that retains a bounded window of a stream instead of all of it.
 *
 * `tail` is what exec uses: the last N bytes, which is where build errors
 * usually are. `head` is what readFile uses: a 4 MB README should still start
 * at the top.
 */
class Capture {
  readonly stream: Writable;
  private readonly chunks: Buffer[] = [];
  private kept = 0;
  private total = 0;

  constructor(
    private readonly cap: number,
    private readonly keep: Keep,
  ) {
    this.stream = new Writable({
      write: (chunk: Buffer | string, _enc, cb) => {
        this.add(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
        cb();
      },
    });
  }

  private add(chunk: Buffer): void {
    this.total += chunk.length;

    if (this.keep === "head") {
      const room = this.cap - this.kept;
      if (room <= 0) return;
      const slice = chunk.length > room ? chunk.subarray(0, room) : chunk;
      this.chunks.push(slice);
      this.kept += slice.length;
      return;
    }

    this.chunks.push(chunk);
    this.kept += chunk.length;
    // Drop whole leading chunks while the rest alone still covers the window.
    while (this.chunks.length > 1 && this.kept - this.chunks[0]!.length >= this.cap) {
      this.kept -= this.chunks.shift()!.length;
    }
    const excess = this.kept - this.cap;
    if (excess > 0) {
      this.chunks[0] = this.chunks[0]!.subarray(excess);
      this.kept -= excess;
    }
  }

  /** The retained window, with a marker when bytes were dropped. */
  text(): string {
    const dropped = this.total - this.kept;
    let body = Buffer.concat(this.chunks).toString("utf8");
    // A window can start mid-codepoint; drop the replacement chars that leaves.
    if (this.keep === "tail") body = body.replace(/^\uFFFD+/, "");

    if (dropped === 0) return body;
    return this.keep === "head"
      ? `${body}\n…[truncated: kept first ${this.kept} of ${this.total} bytes]`
      : `…[truncated: kept last ${this.kept} of ${this.total} bytes]…\n${body}`;
  }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

export interface StartContainerOptions {
  /** Host directory to bind into the container as /workspace. */
  hostWorkdir: string;
  image?: string;
  env?: Record<string, string>;
}

/** Pull the image if it is not local yet. */
async function ensureImage(image: string): Promise<void> {
  const d = docker();
  try {
    await d.getImage(image).inspect();
    return;
  } catch (err) {
    if (!isNotFound(err)) throw new Error(`could not inspect image ${image}: ${message(err)}`);
  }

  console.log(`[docker] pulling ${image} (first run only)…`);
  const stream = await d.pull(image);
  await new Promise<void>((resolvePromise, reject) => {
    d.modem.followProgress(stream, (err) => (err ? reject(err) : resolvePromise()));
  });
}

/**
 * Start a long-lived container with the repo bind-mounted at /workspace and
 * return its id. The container is idle (`sleep infinity`); all work happens
 * through `exec`.
 *
 * `Init: true` gives the container a real init as PID 1 so exited child
 * processes are reaped instead of lingering as zombies in a long run.
 */
export async function startContainer(opts: StartContainerOptions): Promise<string> {
  const hostWorkdir = resolve(opts.hostWorkdir);
  if (!existsSync(hostWorkdir)) {
    throw new Error(`host workdir does not exist: ${hostWorkdir}`);
  }

  const image = opts.image ?? DEFAULT_IMAGE;
  const d = docker();
  await ensureImage(image);

  // The daemon's mount parser splits on ':', which is also the Windows drive
  // separator, so hand it forward slashes there.
  const hostPath = process.platform === "win32" ? hostWorkdir.replace(/\\/g, "/") : hostWorkdir;

  const container = await d.createContainer({
    Image: image,
    Cmd: ["sleep", "infinity"],
    WorkingDir: CONTAINER_WORKDIR,
    Tty: false,
    Labels: { "readme-tester": "sandbox" },
    Env: Object.entries(opts.env ?? {}).map(([key, value]) => `${key}=${value}`),
    HostConfig: {
      Binds: [`${hostPath}:${CONTAINER_WORKDIR}`],
      AutoRemove: false,
      Init: true,
    },
  });

  try {
    await container.start();
  } catch (err) {
    await container.remove({ force: true }).catch(() => undefined);
    throw new Error(`could not start container from ${image}: ${message(err)}`);
  }

  active.add(container.id);
  console.log(`[docker] started ${container.id.slice(0, 12)} from ${image}`);
  return container.id;
}

/** True while the container still exists and is running (used to resume a run). */
export async function isAlive(containerId: string): Promise<boolean> {
  try {
    const info = await docker().getContainer(containerId).inspect();
    return Boolean(info.State?.Running);
  } catch (err) {
    if (isNotFound(err)) return false;
    throw new Error(`could not inspect container ${containerId.slice(0, 12)}: ${message(err)}`);
  }
}

/** Force-remove the container. Safe to call twice; 404 is treated as success. */
export async function destroy(containerId: string): Promise<void> {
  active.delete(containerId);
  try {
    await docker().getContainer(containerId).remove({ force: true, v: true });
    console.log(`[docker] destroyed ${containerId.slice(0, 12)}`);
  } catch (err) {
    if (isNotFound(err)) return;
    throw new Error(`could not destroy container ${containerId.slice(0, 12)}: ${message(err)}`);
  }
}

/**
 * Cleanup hook for the CLI's `finally`: remove every container this process
 * started and has not already removed. Failures are logged, never thrown — a
 * cleanup problem must not mask the real error the run is unwinding from.
 */
export async function destroyAll(): Promise<void> {
  for (const id of [...active]) {
    try {
      await destroy(id);
    } catch (err) {
      console.warn(`[docker] cleanup failed for ${id.slice(0, 12)}: ${message(err)}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

export interface ExecOptions {
  /** Wall clock for this command. Defaults to DEFAULT_EXEC_TIMEOUT_MS. */
  timeoutMs?: number;
  /** Container directory to run in. Defaults to /workspace. */
  workdir?: string;
}

export interface ExecResult {
  /** Process exit code, or null when the daemon never reported one. */
  exitCode: number | null;
  /** Last ~2 KB of stdout, with a truncation marker when bytes were dropped. */
  stdout: string;
  /** Last ~2 KB of stderr, same truncation. */
  stderr: string;
  /** True when the command hit the timeout and was killed. */
  timedOut: boolean;
  durationMs: number;
}

interface RunOptions {
  timeoutMs: number;
  workdir: string;
  cap: number;
  keep: Keep;
}

type RunResult = Omit<ExecResult, "timedOut"> & { timedOut: boolean };

/**
 * Run a command inside the container and capture bounded output.
 *
 * The timeout is enforced inside the container by coreutils `timeout`, so the
 * process tree is actually killed and not just abandoned. A JS-side backstop
 * covers the case where the exec stream itself wedges (daemon hiccup, hung
 * pipe) — it destroys the stream so a run can never hang forever.
 */
async function run(containerId: string, command: string, opts: RunOptions): Promise<RunResult> {
  const d = docker();
  const container = d.getContainer(containerId);
  const seconds = Math.max(1, Math.ceil(opts.timeoutMs / 1000));

  let execInstance;
  try {
    execInstance = await container.exec({
      Cmd: ["timeout", "--kill-after=5s", `${seconds}s`, "bash", "-lc", command],
      AttachStdout: true,
      AttachStderr: true,
      WorkingDir: opts.workdir,
    });
  } catch (err) {
    throw new Error(`could not create exec for \`${command}\`: ${message(err)}`);
  }

  const started = Date.now();
  const stdout = new Capture(opts.cap, opts.keep);
  const stderr = new Capture(opts.cap, opts.keep);

  let stream: Duplex;
  try {
    stream = await execInstance.start({});
  } catch (err) {
    throw new Error(`could not start \`${command}\` in ${containerId.slice(0, 12)}: ${message(err)}`);
  }

  // Docker multiplexes stdout/stderr on one stream with 8-byte frame headers.
  d.modem.demuxStream(stream, stdout.stream, stderr.stream);

  let backstopFired = false;
  await new Promise<void>((resolvePromise) => {
    let timer: NodeJS.Timeout | undefined;

    const finish = () => {
      if (timer) clearTimeout(timer);
      resolvePromise();
    };

    timer = setTimeout(() => {
      backstopFired = true;
      stream.destroy();
      finish();
    }, opts.timeoutMs + 5_000);
    timer.unref();

    stream.once("end", finish);
    stream.once("close", finish);
    stream.once("error", finish);
  });

  let exitCode: number | null = null;
  try {
    exitCode = (await execInstance.inspect()).ExitCode ?? null;
  } catch (err) {
    // A destroyed stream can cost us the inspect call; the output still stands.
    console.log(`[docker] could not read exit code for \`${command}\`: ${message(err)}`);
  }

  // 124 is `timeout`'s own exit code; 137 is the SIGKILL from --kill-after.
  const timedOut = backstopFired || exitCode === 124 || exitCode === 137;

  return {
    exitCode,
    stdout: stdout.text(),
    stderr: stderr.text(),
    timedOut,
    durationMs: Date.now() - started,
  };
}

/** Run one documented command. Output is truncated to the LLM tail budget. */
export async function exec(
  containerId: string,
  command: string,
  opts: ExecOptions = {},
): Promise<ExecResult> {
  return run(containerId, command, {
    timeoutMs: opts.timeoutMs ?? DEFAULT_EXEC_TIMEOUT_MS,
    workdir: opts.workdir ?? CONTAINER_WORKDIR,
    cap: LLM_OUTPUT_TAIL_BYTES,
    keep: "tail",
  });
}

// ---------------------------------------------------------------------------
// Filesystem
// ---------------------------------------------------------------------------

const absolute = (path: string): string =>
  path.startsWith("/") ? path : `${CONTAINER_WORKDIR}/${path.replace(/^\.\//, "")}`;

/** Read a text file from the container. Docs are not truncated to 2 KB. */
export async function readFile(containerId: string, path: string): Promise<string> {
  const target = absolute(path);
  const result = await run(containerId, `cat -- ${shellQuote(target)}`, {
    timeoutMs: 30_000,
    workdir: CONTAINER_WORKDIR,
    cap: MAX_READ_BYTES,
    keep: "head",
  });

  if (result.timedOut) throw new Error(`timed out reading ${path}`);
  if (result.exitCode !== 0) {
    throw new Error(`could not read ${path}: ${result.stderr.trim() || `exit ${result.exitCode}`}`);
  }
  return result.stdout;
}

export interface FileEntry {
  /** Path relative to the listed directory. */
  path: string;
  type: "file" | "dir" | "other";
}

/**
 * List entries under `dir` (repo-relative or absolute), default one level deep.
 *
 * Relies on GNU find, which the default image has; an Alpine/busybox image
 * would need the `-printf` swapped out.
 */
export async function listFiles(
  containerId: string,
  dir = ".",
  opts: { maxDepth?: number } = {},
): Promise<FileEntry[]> {
  const depth = Math.max(1, Math.trunc(opts.maxDepth ?? 1));
  const result = await run(
    containerId,
    `find . -mindepth 1 -maxdepth ${depth} -printf '%y\\t%P\\n'`,
    {
      timeoutMs: 30_000,
      workdir: absolute(dir),
      cap: MAX_READ_BYTES,
      keep: "head",
    },
  );

  if (result.timedOut) throw new Error(`timed out listing ${dir}`);
  if (result.exitCode !== 0) {
    throw new Error(`could not list ${dir}: ${result.stderr.trim() || `exit ${result.exitCode}`}`);
  }

  return result.stdout
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => {
      const [flag, ...rest] = line.split("\t");
      return {
        path: rest.join("\t"),
        type: flag === "d" ? ("dir" as const) : flag === "f" ? ("file" as const) : ("other" as const),
      };
    })
    .sort((a, b) => a.path.localeCompare(b.path));
}
