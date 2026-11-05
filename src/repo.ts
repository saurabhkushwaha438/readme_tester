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
import { stat } from "node:fs/promises";
import { mkdtemp, rm } from "node:fs/promises";
import { cp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { infoLine, warnLine } from "./log.js";

async function exists(p: string): Promise<boolean> {
  try {
    const s = await stat(p);
    return s.isDirectory();
  } catch {
    return false;
  }
}

const run = promisify(execFile);

/** Directory name of the clone inside the temp workdir. */
export const CLONE_DIR_NAME = "repo";

/** A slow repo (monorepo, full history) can still take a while. */
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
 * Clone `repoUrl` into a fresh temp workdir.
 *
 * Phase 10 changed the default from a tagless shallow clone to a full one.
 * `--no-tags` looked like a harmless speedup, but anything that derives its
 * version from `git describe` (fzf's Makefile, among others) then fails with
 * "Not on git repository" — a harness artifact reported as a docs failure.
 * `README_TEST_SHALLOW=1` restores the cheap `--depth 1 --no-tags` clone for
 * very large repos where history is not worth the wait.
 *
 * Local paths work too, which is what the tests use.
 */
export async function cloneRepo(
  repoUrl: string,
  opts: { timeoutMs?: number } = {},
): Promise<CloneResult> {
  const timeout = opts.timeoutMs ?? DEFAULT_CLONE_TIMEOUT_MS;
  const workdir = await mkdtemp(join(tmpdir(), "readme-test-"));
  created.push(workdir);
  const cloneDir = join(workdir, CLONE_DIR_NAME);

  infoLine(`[repo] cloning ${repoUrl} -> ${workdir}`);

  // file:// is transport only. Git for Windows resolves it differently from an
  // interactive shell, so translate it to a native path and try a filesystem
  // copy first: that is what the tests use, and it is what the container mount
  // needs.
  let localPath = repoUrl;
  if (repoUrl.startsWith("file://")) {
    try {
      localPath = fileURLToPath(repoUrl);
    } catch {
      localPath = repoUrl.slice("file://".length);
    }
  }

  try {
    if ((await exists(localPath)) && (await isGitRepo(localPath))) {
      // Direct copy is the reliable path for local repos on Windows; it avoids
      // git's path-translation differences between a shell and execFile.
      await cp(localPath, cloneDir, { recursive: true, force: true });
      infoLine(`[repo] copied local repo ${localPath} -> ${cloneDir}`);
      return { workdir, cloneDir };
    }

    // Not a local git repo (or not local at all): fall back to git clone.
    const shallow = process.env.README_TEST_SHALLOW === "1";
    const args = [
      // Git for Windows defaults to core.autocrlf=true, so a clone on the host
      // checks shell scripts out with CRLF. The Linux container then runs
      // `bash scripts/prepare.sh` and dies on `$'\r': command not found` — a
      // checkout artifact, not a docs bug (Nest's CONTRIBUTING.md hit this in
      // Phase 10). Pin the checkout to LF for both the clone and the new repo.
      "-c",
      "core.autocrlf=false",
      "clone",
      ...(shallow ? ["--depth", "1", "--no-tags"] : []),
      "--config",
      "core.autocrlf=false",
      "--config",
      "core.eol=lf",
      "--single-branch",
      "--quiet",
      repoUrl,
      cloneDir,
    ];
    await run("git", args, { timeout, windowsHide: true, maxBuffer: 8 * 1024 * 1024 });
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

/** Is `p` a git repository (has a .git directory or is the .git itself)? */
async function isGitRepo(p: string): Promise<boolean> {
  // .git can be a file (submodule) or a directory; both count.
  try {
    const s = await stat(join(p, ".git"));
    s; // suppress unused — we only care that stat succeeded
  } catch {
    return false;
  }
  return true;
}

/** Remove one workdir. Missing is fine; failures are only logged. */
export async function removeWorkdir(dir: string): Promise<void> {
  const index = created.indexOf(dir);
  if (index >= 0) created.splice(index, 1);
  try {
    await rm(dir, { recursive: true, force: true, maxRetries: 3 });
  } catch (err) {
    warnLine(`[repo] could not remove ${dir}: ${message(err)}`);
  }
}

/** Cleanup hook for the CLI's `finally`: never leave temp clones behind. */
export async function removeAllWorkdirs(): Promise<void> {
  for (const dir of [...created]) {
    await removeWorkdir(dir);
  }
}
