/**
 * Phase 3, host side: get the repo onto disk.
 *
 * The clone happens on the host (git is installed there; the container only
 * needs a bind mount) into a temp directory that is removed when the run ends.
 *
 * Workdirs created here are tracked so `removeAllWorkdirs()` can clean up after
 * a crash — including when the graph throws instead of reaching `report`.
 */
import { execFile } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const run = promisify(execFile);

/** Directory name of the clone inside the temp workdir. */
export const CLONE_DIR_NAME = "repo";

/** A slow repo (monorepo, shallow clone) can still take a while. */
export const DEFAULT_CLONE_TIMEOUT_MS = Number(
  process.env.README_TEST_CLONE_TIMEOUT_MS ?? 180_000,
);

/** Workdirs we created; `removeAllWorkdirs()` empties this. */
const created: string[] = [];

export interface CloneResult {
  /** Temp directory holding the clone; removed at the end of the run. */
  workdir: string;
  /** The checkout itself — this is what gets bind-mounted as /workspace. */
  cloneDir: string;
}

const message = (err: unknown): string => (err instanceof Error ? err.message : String(err));

/** Last few lines of git's stderr, where the actual reason lives. */
function gitReason(err: unknown): string {
  const stderr = (err as { stderr?: unknown }).stderr;
  const text = typeof stderr === "string" ? stderr.trim() : "";
  if (!text) return message(err);
  const tail = text.split("\n").slice(-4).join("\n");
  return tail.length > 600 ? `${tail.slice(-600)}…` : tail;
}

/**
 * Shallow-clone `repoUrl` into a fresh temp workdir.
 *
 * `--depth 1 --single-branch` keeps a test run cheap; nothing here needs
 * history. Local paths work too, which is what the tests use.
 */
export async function cloneRepo(
  repoUrl: string,
  opts: { timeoutMs?: number } = {},
): Promise<CloneResult> {
  const timeout = opts.timeoutMs ?? DEFAULT_CLONE_TIMEOUT_MS;
  const workdir = await mkdtemp(join(tmpdir(), "readme-test-"));
  created.push(workdir);
  const cloneDir = join(workdir, CLONE_DIR_NAME);

  console.log(`[repo] cloning ${repoUrl} -> ${workdir}`);
  try {
    await run(
      "git",
      ["clone", "--depth", "1", "--single-branch", "--no-tags", "--quiet", repoUrl, cloneDir],
      { timeout, windowsHide: true, maxBuffer: 8 * 1024 * 1024 },
    );
  } catch (err) {
    await removeWorkdir(workdir);
    const killed = Boolean((err as { killed?: boolean }).killed);
    if (killed) throw new Error(`git clone timed out after ${Math.round(timeout / 1000)}s: ${repoUrl}`);
    const code = (err as { code?: number | string }).code;
    if (code === "ENOENT") throw new Error("git is not installed or not on PATH.");
    throw new Error(`git clone failed for ${repoUrl}:\n${gitReason(err)}`);
  }

  return { workdir, cloneDir };
}

/** Remove one workdir. Missing is fine; failures are only logged. */
export async function removeWorkdir(dir: string): Promise<void> {
  const index = created.indexOf(dir);
  if (index >= 0) created.splice(index, 1);
  try {
    await rm(dir, { recursive: true, force: true, maxRetries: 3 });
  } catch (err) {
    console.warn(`[repo] could not remove ${dir}: ${message(err)}`);
  }
}

/** Cleanup hook for the CLI's `finally`: never leave temp clones behind. */
export async function removeAllWorkdirs(): Promise<void> {
  for (const dir of [...created]) {
    await removeWorkdir(dir);
  }
}
