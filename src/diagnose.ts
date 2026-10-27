/**
 * Phase 6: diagnose a failed step.
 *
 * Input is the failing command plus the truncated stderr/stdout the Docker
 * layer already captured. The strong model proposes one fix as structured
 * output; we run it once. A fix that works turns the step into `workaround`,
 * anything else leaves it `fail`.
 *
 * The model never gets to run a command unchecked: `checkCommand` rejects
 * destructive, privileged, publishing and non-terminating commands before
 * anything reaches the container.
 */
import { z } from "zod";
import { exec, type ExecResult } from "./docker.js";
import { heuristicLabels } from "./docs.js";
import { getStrong } from "./llm.js";
import type { Step, StepResult, StepStatus } from "./state.js";

/** A proposed fix must be a command, not an essay. */
const MAX_FIX_CHARS = 2000;

/**
 * What the model may answer.
 *
 * `command: null` is a real answer, not a failure: plenty of documented steps
 * fail for reasons no in-container command can fix (no Docker daemon, no
 * credentials, the docs are simply wrong), and saying so is more useful than
 * inventing a plausible-looking command that fails too.
 */
export const FixSchema = z.object({
  command: z
    .string()
    .nullable()
    .describe(
      "One shell command that should make the failed step succeed, run from the repo root. " +
        "null when no command-level fix is honest.",
    ),
  reason: z
    .string()
    .describe("One or two sentences: why the documented command failed, and what the fix changes."),
});
export type Fix = z.infer<typeof FixSchema>;

/** Test seam: same shape as the extract classifier's injected model call. */
export type Proposer = (input: DiagnoseInput, attempt: number) => Promise<Fix>;

const message = (err: unknown): string => (err instanceof Error ? err.message : String(err));

const flat = (command: string): string => command.replace(/\s+/g, " ").trim();

/**
 * Reasons end up in the report, so keep them to one readable line: a provider
 * error body is a wall of JSON and a stack trace is worse.
 */
function brief(text: string, max = 220): string {
  const oneLine = text.replace(/\s+/g, " ").trim();
  return oneLine.length > max ? `${oneLine.slice(0, max)}…` : oneLine;
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

export interface DiagnoseInput {
  containerId: string;
  step: Step;
  /** The failing row `run_step` appended. Its stderr is the whole diagnosis. */
  failure: StepResult;
  image?: string;
  timeoutMs?: number;
  propose?: Proposer;
}

/**
 * Build the diagnosis prompt.
 *
 * stdout and stderr are already truncated to the last ~2 KB by `exec`, which is
 * exactly the budget the plan specifies. Both are included: plenty of tools
 * print the cause on stdout and only stack frames on stderr.
 */
export function buildFixPrompt(input: DiagnoseInput): string {
  const { step, failure } = input;
  const stdout = failure.stdout.trim() || "(empty)";
  const stderr = failure.stderr.trim() || "(empty)";

  return `A documented setup command failed in a clean container while auditing a repository's instructions.

Environment:
- The repository is cloned at /workspace; this command runs there.
- Image: ${input.image ?? "node:22-bookworm"}, running as root, non-interactive, no TTY.
- Bash is available. No Docker daemon, no sudo, no browser, no credentials.

Rules:
1. Propose ONE command that makes the documented step succeed. It must exit on its own — never start a server, watcher or interactive session.
2. Keep the documented intent: fix the install, do not skip it or stub it out.
3. Prefer the smallest honest fix: a missing system package, the repository's real install command (npm ci, pnpm install --frozen-lockfile), a flag the docs omit.
4. Do not return the failing command unchanged.
5. If the failure cannot be fixed from inside the container, return command=null and explain why.

Failed step:
  command: ${step.command.replace(/\n/g, "\n           ")}
  source: ${step.source}${step.heading ? ` (section: ${step.heading})` : ""}
  exit code: ${failure.exitCode ?? "unknown"}${
    failure.reason
      ? `\n  runner note: ${failure.reason} (the sandbox runner, not the command itself)`
      : ""
  }

stdout (last ~2 KB):
${stdout}

stderr (last ~2 KB):
${stderr}`;
}

/** The production proposer: one structured call to the strong model. */
function llmProposer(): Proposer {
  const structured = getStrong().withStructuredOutput(FixSchema, { name: "propose_setup_fix" });
  return async (input) => FixSchema.parse(await structured.invoke(buildFixPrompt(input)));
}

// ---------------------------------------------------------------------------
// Denylist
// ---------------------------------------------------------------------------

/**
 * Commands no documented setup step needs and a hallucinating model might
 * still produce. Deliberately small: this is a guard against obvious damage,
 * not a sandbox (the container is already the sandbox).
 */
const DENY: ReadonlyArray<{ pattern: RegExp; why: string }> = [
  { pattern: /\b(mkfs(\.\w+)?|fdisk|parted|wipefs)\b/, why: "formats a block device" },
  { pattern: /\bdd\b[^|;&\n]*\bof=\/dev\//, why: "writes raw bytes to a device" },
  { pattern: /:\s*\(\s*\)\s*\{[^}]*\}\s*;\s*:/, why: "fork bomb" },
  { pattern: /\b(shutdown|reboot|halt|poweroff)\b/, why: "takes the machine down" },
  {
    pattern: /\b(curl|wget)\b[^|;&\n]*\|\s*(?:sudo\s+)?(?:ba|z|da|k)?sh\b/,
    why: "pipes a download straight into a shell",
  },
  {
    pattern: /\beval\b[^|;&\n]*\$\(\s*(curl|wget)\b/,
    why: "evaluates a downloaded script",
  },
  { pattern: /(^|[\s;&|])(sudo|doas|pkexec)\b/, why: "needs root escalation" },
  { pattern: /\bsu\s+-/, why: "needs root escalation" },
  { pattern: /\bgit\s+push\b/, why: "writes to a remote repository" },
  { pattern: /\b(npm|yarn|pnpm|bun)\s+publish\b/, why: "publishes a package" },
];

/** Directories whose recursive deletion is never a setup fix. */
const SYSTEM_PATHS = /^(~|\$HOME|\/(etc|usr|bin|sbin|var|root|boot|opt|lib|lib64|proc|sys|dev))(\/|$)/;

/**
 * Catch `rm -rf /`, `rm -rf /*`, `rm -rf ~`, `rm -rf /etc` — while letting
 * `rm -rf node_modules` and `rm -rf /workspace/dist` through, which are normal
 * setup commands.
 */
function isSystemDelete(command: string): boolean {
  for (const call of command.matchAll(/\brm\s+([^;&|\n]+)/g)) {
    const args = call[1]!.trim().split(/\s+/);
    const flags = args.filter((arg) => arg.startsWith("-")).join(" ");
    const recursive = /[rR]|--recursive/.test(flags);
    const forced = /[fF]|--force/.test(flags);
    if (!recursive || !forced) continue;

    for (const raw of args.filter((arg) => !arg.startsWith("-"))) {
      const target = raw.replace(/^["']|["']$/g, "");
      if (target === "/" || target === "/*" || target === "/.") return true;
      if (SYSTEM_PATHS.test(target)) return true;
    }
  }
  return false;
}

export interface CommandVerdict {
  allowed: boolean;
  reason?: string;
}

/**
 * Gate every model-proposed command. Checked before the container sees it, so a
 * rejection costs nothing but a log line.
 */
export function checkCommand(command: string): CommandVerdict {
  const cmd = command.trim();
  if (!cmd) return { allowed: false, reason: "empty command" };
  if (cmd.length > MAX_FIX_CHARS) {
    return { allowed: false, reason: `longer than ${MAX_FIX_CHARS} chars` };
  }

  for (const rule of DENY) {
    if (rule.pattern.test(cmd)) return { allowed: false, reason: rule.why };
  }
  if (isSystemDelete(cmd)) {
    return { allowed: false, reason: "recursive delete of the filesystem or a system directory" };
  }

  // Same heuristics the Phase 4 fallback uses: a fix that waits for stdin or
  // never exits would burn the full exec timeout and say nothing.
  if (!heuristicLabels(cmd, "", undefined).runnable) {
    return {
      allowed: false,
      reason: "would not finish unattended (interactive, long-running, or a placeholder)",
    };
  }

  return { allowed: true };
}

// ---------------------------------------------------------------------------
// Diagnosis
// ---------------------------------------------------------------------------

export interface Diagnosis {
  /** `workaround` only when the fix ran and the documented command then passed. */
  status: Extract<StepStatus, "workaround" | "fail">;
  /** The failing documented command on `fail`, the working fix on `workaround`. */
  command: string;
  exitCode: number | null;
  stdout: string;
  stderr: string;
  durationMs: number;
  reason: string;
  /** The model's proposed command, when it produced one. */
  attemptedFix?: string;
  /** Commands that actually reached the container: 0, 1, or 2 (with re-check). */
  commandsRun: number;
  /** Transport/parse retries spent, added to the run's `llmRetries`. */
  llmRetries: number;
}

/** A failure that never reached the container. */
function declined(input: DiagnoseInput, reason: string, command?: string): Diagnosis {
  return {
    status: "fail",
    command: command ?? input.step.command,
    exitCode: input.failure.exitCode,
    stdout: "",
    stderr: "",
    durationMs: 0,
    reason,
    ...(command ? { attemptedFix: command } : {}),
    commandsRun: 0,
    llmRetries: 0,
  };
}

/**
 * Ask for one fix and try it.
 *
 * Every failure mode here is fail-soft: a rate limit, an unparseable answer, a
 * rejected command or a fix that does not work all leave the step `fail` and
 * let the run continue to the next step. Only a dead daemon is loud, and
 * `run_step` is where that is handled.
 */
export async function diagnoseStep(input: DiagnoseInput): Promise<Diagnosis> {
  const propose = input.propose ?? llmProposer();

  let fix: Fix | undefined;
  let llmRetries = 0;
  let lastError: string | undefined;

  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      fix = await propose(input, attempt);
      break;
    } catch (err) {
      lastError = message(err);
      if (attempt === 1) {
        llmRetries++;
        console.warn(`[diagnose] proposal failed (${lastError}); retrying once`);
      }
    }
  }

  if (!fix) {
    return {
      ...declined(input, `diagnosis unavailable: ${brief(lastError ?? "no answer")}`),
      llmRetries,
    };
  }

  if (fix.command === null || fix.command.trim() === "") {
    return {
      ...declined(input, `model proposed no fix: ${brief(fix.reason)}`),
      llmRetries,
    };
  }

  const proposal = fix.command.trim();
  const verdict = checkCommand(proposal);
  if (!verdict.allowed) {
    return {
      ...declined(input, `rejected unsafe fix (${verdict.reason}): ${proposal}`, proposal),
      llmRetries,
    };
  }

  if (flat(proposal) === flat(input.step.command)) {
    return {
      ...declined(input, "model returned the failing command unchanged", proposal),
      llmRetries,
    };
  }

  let out: ExecResult;
  try {
    out = await exec(input.containerId, proposal, { timeoutMs: input.timeoutMs });
  } catch (err) {
    return {
      ...declined(input, `could not run the fix: ${message(err)}`, proposal),
      llmRetries,
    };
  }

  const worked = out.exitCode === 0 && !out.timedOut;
  const note = brief(fix.reason);

  if (!worked) {
    return {
      status: "fail",
      command: proposal,
      exitCode: out.exitCode,
      stdout: out.stdout,
      stderr: out.stderr,
      durationMs: out.durationMs,
      reason: out.timedOut
        ? `${note} — the fix timed out after ${Math.round(out.durationMs / 1000)}s`
        : `${note} — the fix failed too (exit ${out.exitCode ?? "?"})`,
      attemptedFix: proposal,
      commandsRun: 1,
      llmRetries,
    };
  }

  // A fix exiting 0 proves nothing on its own: `apt-get install git` succeeds
  // whether or not git was the problem. Re-running the documented command is
  // the only honest test of "workaround", and it costs one extra command.
  let verified: ExecResult | undefined;
  try {
    verified = await exec(input.containerId, input.step.command, { timeoutMs: input.timeoutMs });
  } catch (err) {
    return {
      status: "fail",
      command: proposal,
      exitCode: out.exitCode,
      stdout: out.stdout,
      stderr: out.stderr,
      durationMs: out.durationMs,
      reason: `${note} — the fix exited 0, but the documented command could not be re-run: ${message(err)}`,
      attemptedFix: proposal,
      commandsRun: 1,
      llmRetries,
    };
  }

  const fixed = verified.exitCode === 0 && !verified.timedOut;

  return {
    status: fixed ? "workaround" : "fail",
    // On success the interesting command is the fix; on failure the row belongs
    // to the documented command, which is what still does not work.
    command: fixed ? proposal : input.step.command,
    exitCode: verified.exitCode,
    stdout: verified.stdout,
    stderr: verified.stderr,
    durationMs: out.durationMs + verified.durationMs,
    reason: fixed
      ? `${note} — the documented command passes after the fix`
      : `${note} — the fix exited 0, but the documented command still fails (exit ${verified.exitCode ?? "?"})`,
    attemptedFix: proposal,
    commandsRun: 2,
    llmRetries,
  };
}
