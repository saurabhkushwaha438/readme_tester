/**
 * Phase 8: the markdown report.
 *
 * Two halves, split the same way as the CI cross-check:
 *
 * - Everything factual is templated in code. Which steps passed, what exited
 *   non-zero, the error excerpt, which doc file and heading a step came from,
 *   and the exact `- old / + new` command diff in the suggested patch.
 * - The strong model writes *wording only*: one or two sentences per patch
 *   explaining why the change is right. It cannot add a patch, choose a
 *   different command, or change a status — a rate limit costs the prose and
 *   leaves the deterministic fallback in its place.
 *
 * Error excerpts come from the ~2 KB the Docker layer kept, which is the whole
 * reason Phase 2 truncates: the tail is what names the failure, and the report
 * is where a human finally reads it.
 */
import { z } from "zod";
import { actionOf, asProse } from "./ci.js";
import { callModel, getStrong, LlmBudgetExhausted } from "./llm.js";
import { warnLine } from "./log.js";
import {
  BUDGET_SKIP_REASON,
  SELF_CLONE_SKIP_REASON,
  type CiFinding,
  type Mode,
  type Step,
  type StepResult,
  type StepStatus,
} from "./state.js";

/** Enough context to recognise a failure without pasting a build log. */
const EXCERPT_LINES = 12;
const EXCERPT_CHARS = 1400;
const EXCERPT_LINE_CHARS = 400;

/** Table cells are one line by construction. */
const CELL_CHARS = 110;

const message = (err: unknown): string => (err instanceof Error ? err.message : String(err));

const ANSI = /\u001b\[[0-9;]*m/g;

const plain = (text: string): string => text.replace(ANSI, "").replace(/\r\n?/g, "\n");

function brief(text: string, max = CELL_CHARS): string {
  const flat = plain(text).replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max)}…` : flat;
}

/** Command text for a table cell: one line, pipes escaped. */
function inline(text: string, max = CELL_CHARS): string {
  return brief(text, max).replace(/\|/g, "\\|");
}

/**
 * A command as an inline code span, for a table cell or a failure heading.
 *
 * Not `inline` in backticks: a documented command may contain backticks of its
 * own ("Run `bun run lint:fix` first"), and single backticks would close at the
 * first inner one, spilling the rest of the command into the cell. `codeSpan`
 * picks a delimiter long enough to contain them.
 */
function code(text: string, max = CELL_CHARS): string {
  return codeSpan(inline(text, max));
}

const plural = (count: number, word: string, words = `${word}s`): string =>
  `${count} ${count === 1 ? word : words}`;

/**
 * A fenced block that survives backticks inside the content: the fence is
 * always longer than the longest run of backticks it has to contain.
 */
function fence(text: string, lang = "text"): string {
  const longest = Math.max(0, ...(text.match(/`+/g) ?? []).map((run) => run.length));
  const ticks = "`".repeat(Math.max(3, longest + 1));
  return `${ticks}${lang}\n${text}\n${ticks}`;
}

/**
 * An inline code span that survives backticks inside it — the same rule as
 * `fence`, one line instead of a block.
 *
 * A documented command can legitimately contain backticks ("Run `bun run
 * lint:fix` first"), and single backticks would close at the first inner one,
 * spilling the rest of the command into the cell as prose. Delimit with a run
 * longer than any inside; when the content itself starts or ends with a
 * backtick, pad with spaces so that backtick cannot merge with the delimiter.
 * CommonMark strips exactly one space from each end, so nothing renders.
 */
function codeSpan(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  if (flat === "") return "";

  const longest = Math.max(0, ...(flat.match(/`+/g) ?? []).map((run) => run.length));
  const ticks = "`".repeat(longest + 1);
  const pad = flat.startsWith("`") || flat.endsWith("`") ? " " : "";

  return `${ticks}${pad}${flat}${pad}${ticks}`;
}

// ---------------------------------------------------------------------------
// Error excerpts
// ---------------------------------------------------------------------------

// No trailing \b: several alternatives end in punctuation (`npm ERR!`), and a
// word boundary after `!` never matches.
const ERROR_HINT = /\b(error|err!|failed|failure|fatal|cannot|can't|not found|denied|missing|unable|exception|EACCES|ENOENT)/i;

export interface ExcerptOptions {
  maxLines?: number;
  maxChars?: number;
}

/**
 * The last few meaningful lines of whatever the failing step printed.
 *
 * stderr wins when it has anything, because that is where compilers and package
 * managers put the cause; stdout is the fallback for tools that are less tidy.
 * The *tail* is kept: `npm ERR!` headers are long, and the decisive line is at
 * the bottom.
 */
export function errorExcerpt(result: StepResult, opts: ExcerptOptions = {}): string {
  const maxLines = opts.maxLines ?? EXCERPT_LINES;
  const maxChars = opts.maxChars ?? EXCERPT_CHARS;

  const source = result.stderr.trim() ? result.stderr : result.stdout;
  const lines = plain(source)
    .split("\n")
    .map((line) => line.replace(/\s+$/, ""))
    .filter((line) => line.trim() !== "");

  if (lines.length === 0) return "(no output captured)";

  const dropped = lines.length > maxLines;
  const picked = lines
    .slice(-maxLines)
    .map((line) => (line.length > EXCERPT_LINE_CHARS ? `${line.slice(0, EXCERPT_LINE_CHARS)}…` : line));

  // Joining after the per-line cap can still overshoot; keep the tail, which is
  // where the error is, and mark the cut. A leading `…` also marks a cut made by
  // dropping earlier lines, so the block never pretends to start at the top.
  let text = picked.join("\n");
  if (text.length > maxChars) text = `…\n${text.slice(text.length - (maxChars - 2))}`;
  else if (dropped) text = `…\n${text}`;
  return text;
}

/**
 * One line worth putting next to the failure in the table of failures: the
 * first line that looks like an error, else the last line printed at all.
 */
export function keyErrorLine(result: StepResult): string | undefined {
  const source = result.stderr.trim() ? result.stderr : result.stdout;
  const lines = plain(source)
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const hinted = lines.find((line) => ERROR_HINT.test(line));
  const chosen = hinted ?? lines.at(-1);
  return chosen === undefined ? undefined : brief(chosen, 180);
}

// ---------------------------------------------------------------------------
// Suggested patches
// ---------------------------------------------------------------------------

/**
 * One doc edit the run can justify.
 *
 * A patch only exists for a `workaround` step: the documented command failed,
 * a different command was run, and the documented command then passed. That is
 * the only case where the command-level claim is verified. A `fail` step has no
 * command to substitute, so it gets an explanation in the report and no diff —
 * inventing one would be exactly the speculation this tool is meant to catch.
 */
/**
 * How the working command relates to the documented one.
 *
 * - replace: same job, done differently (`npm install` -> `npm ci`,
 *   `npm test` -> `npm test -- --runInBand`). The doc line itself is wrong.
 * - prerequisite: a different job that has to happen first (installing a missing
 *   toolchain). The documented command is right, the docs just never mention the
 *   setup it needs.
 */
export type PatchKind = "replace" | "prerequisite";

export interface Patch {
  stepId: string;
  kind: PatchKind;
  source: string;
  heading?: string;
  /** The command the docs tell a reader to run. */
  from: string;
  /** The command that made the documented step succeed. */
  to: string;
  /** Deterministic wording, used when the model is unavailable. */
  fallbackNote: string;
}

/**
 * Drop a redundant `sh -c "..."` wrapper before putting a command in a doc.
 *
 * The model often proposes `sh -c "npm install && npm ci"`; a fenced block in a
 * README is already run by a shell, so the wrapper is noise in the suggested
 * patch. Only exact, single-command wrappers are unwrapped — anything else is
 * left alone rather than guessed at.
 */
export function unwrapShell(command: string): string {
  const match = /^\s*(?:ba|z|da)?sh\s+-c\s+(["'])([\s\S]*)\1\s*$/.exec(command);
  return match?.[2]?.trim() || command;
}

const firstWord = (command: string): string =>
  command.trim().split(/\s+/)[0]?.toLowerCase() ?? "";

/**
 * Classify a workaround before writing a patch for it.
 *
 * The fix is a replacement only when it does the same kind of work as the
 * documented command; `npm install --save-dev typescript` is not a substitute
 * for `npm run build`, it is a step the docs forgot. Phase 7's `actionOf` is the
 * same classifier the CI diff uses, so the two sections cannot disagree about
 * what a command is for.
 */
export function patchKind(from: string, to: string): PatchKind {
  const documented = actionOf(from);
  const fix = actionOf(to);
  if (documented && fix) return documented === fix ? "replace" : "prerequisite";
  if (!documented && !fix) return firstWord(from) === firstWord(to) ? "replace" : "prerequisite";
  return "prerequisite";
}

/**
 * One row per step, using its final result.
 *
 * A step that fails is recorded twice: once by `run_step` and once by
 * `diagnose`, which supersedes it (pass -> workaround/fail). Rendering both is
 * how the report ends up with two identical failure sections for one command and
 * a "5 steps failing" footer over a 4-step run.
 *
 * The visible row is the last result. When that one carries no output — a
 * declined diagnosis never touched the container — the documented command's
 * earlier stderr is kept, because that excerpt is the evidence. Durations are
 * summed: the step cost both runs.
 */
export function finalResults(results: StepResult[]): StepResult[] {
  const order: string[] = [];
  const byStep = new Map<string, StepResult>();

  const hasOutput = (result: StepResult): boolean =>
    result.stderr.trim() !== "" || result.stdout.trim() !== "";

  for (const result of results) {
    const previous = byStep.get(result.stepId);
    if (!previous) {
      order.push(result.stepId);
      byStep.set(result.stepId, result);
      continue;
    }

    const merged: StepResult = {
      ...result,
      stdout: hasOutput(result) ? result.stdout : previous.stdout,
      stderr: hasOutput(result) ? result.stderr : previous.stderr,
      durationMs: previous.durationMs + result.durationMs,
    };
    byStep.set(result.stepId, merged);
  }

  return order.map((stepId) => byStep.get(stepId)!);
}

/** Deterministic patch plan. Deduped: the same command is often documented twice. */
export function planPatches(steps: Step[], results: StepResult[]): Patch[] {
  const byId = new Map(steps.map((step) => [step.id, step]));
  const patches: Patch[] = [];
  const seen = new Set<string>();

  for (const result of finalResults(results)) {
    if (result.status !== "workaround") continue;

    const step = byId.get(result.stepId);
    if (!step) continue;

    const from = step.command.trim();
    const to = result.command.trim();
    if (!from || !to || from === to) continue;

    const key = `${step.source}\u0000${from}\u0000${to}`;
    if (seen.has(key)) continue;
    seen.add(key);

    patches.push({
      stepId: step.id,
      kind: patchKind(from, to),
      source: step.source,
      ...(step.heading ? { heading: step.heading } : {}),
      from: step.command,
      to: result.command,
      fallbackNote: brief(
        result.reason ?? "the documented command only passes after this change",
        320,
      ),
    });
  }

  return patches;
}

// ---------------------------------------------------------------------------
// Patch wording (the only model call)
// ---------------------------------------------------------------------------

export const PatchWordingSchema = z.object({
  patches: z.array(
    z.object({
      stepId: z.string().describe("The step id, exactly as listed in the input."),
      wording: z
        .string()
        .describe(
          "One or two sentences for a pull request description: what goes wrong with the documented command and why the replacement is correct. No headings, no lists.",
        ),
    }),
  ),
});
export type PatchWording = z.infer<typeof PatchWordingSchema>;

/** Test seam, same shape as the extract classifier and the CI summarizer. */
export type PatchWriter = (patches: Patch[], attempt: number) => Promise<PatchWording>;

export function buildPatchPrompt(patches: Patch[], attempt = 1): string {
  const entries = patches
    .map(
      (patch) =>
        `- stepId: ${patch.stepId} (${patch.kind})\n` +
        `  doc: ${patch.source}${patch.heading ? ` (section: "${patch.heading}")` : ""}\n` +
        `  documented: ${brief(patch.from, 160)}\n` +
        `  ${patch.kind === "replace" ? "works instead" : "must also run"}: ${brief(patch.to, 160)}\n` +
        `  diagnosis: ${patch.fallbackNote}`,
    )
    .join("\n");

  const retryNote =
    attempt > 1
      ? "\nYour previous answer was rejected. Answer with valid JSON only, using the step ids above.\n"
      : "";

  return `A repository's setup documentation was tested by running every documented command in a clean container. The commands below failed as written, and a further command made the documented step succeed.

CHANGES ALREADY DECIDED (do not alter, add to, or reorder them):
${entries}

For each stepId, write one or two sentences of maintainer-facing prose for the pull request that fixes the documentation. A "replace" patch swaps a wrong command for the working one; a "prerequisite" patch documents an extra command that has to run before the documented one, which is itself correct and stays unchanged.

Rules:
1. Use only the commands and diagnosis above. Never invent a file, a version, a cause or a command.
2. Do not repeat the commands as code blocks or restate them line by line — the diff is already in the report.
3. Say what a reader following the current instructions hits, not which tool is generally unreliable.
4. Plain prose, one or two sentences, no markdown formatting.
${retryNote}`;
}

/** The production writer: strong model, structured output. */
function llmWriter(): PatchWriter {
  const structured = getStrong().withStructuredOutput(PatchWordingSchema, {
    name: "write_readme_patch_wording",
  });

  return async (patches, attempt) =>
    PatchWordingSchema.parse(
      await callModel("report", () => structured.invoke(buildPatchPrompt(patches, attempt))),
    );
}

/**
 * Ask for the patch wording.
 *
 * Fail-soft like the CI summary: every patch already has a fallback sentence
 * derived from the diagnosis, so an outage changes the prose, not the patch.
 */
export async function suggestPatchWording(
  patches: Patch[],
  opts: { write?: PatchWriter } = {},
): Promise<{ wording: Map<string, string>; llmRetries: number }> {
  const wording = new Map<string, string>();
  if (patches.length === 0) return { wording, llmRetries: 0 };

  const write = opts.write ?? llmWriter();
  const known = new Set(patches.map((patch) => patch.stepId));
  let lastError: string | undefined;

  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const answer = await write(patches, attempt);
      for (const entry of answer.patches) {
        const text = plain(entry.wording).trim();
        // The model may only fill in wording for the patches it was given.
        if (text === "" || !known.has(entry.stepId) || !entry.stepId) continue;
        wording.set(entry.stepId, brief(text, 400));
      }
      return { wording, llmRetries: attempt - 1 };
    } catch (err) {
      lastError = message(err);
      if (err instanceof LlmBudgetExhausted) break;
      if (attempt === 1) warnLine(`[report] patch wording failed (${lastError}); retrying once`);
    }
  }

  warnLine(
    `[report] patch wording unavailable (${brief(lastError ?? "no answer", 160)}); using the diagnosis`,
  );
  return { wording, llmRetries: 1 };
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

export interface ReportInput {
  repoUrl: string;
  mode: Mode;
  steps: Step[];
  results: StepResult[];
  ciFindings: CiFinding[];
  ciSummary: string;
  cmdCount: number;
  llmRetries: number;
  /** Phase 9: shown next to the command count so a capped run is visible. */
  commandBudget?: number;
  /** Model calls spent, retries included. */
  llmCalls?: number;
  llmCallBudget?: number;
  image?: string;
  /** Injected so a test can pin the date line. */
  createdAt?: Date;
  /** Patch wording by stepId; missing entries fall back to the diagnosis. */
  wording?: Map<string, string>;
}

/** "4 passed, 1 needed a workaround, 1 failed." */
function summarise(results: StepResult[]): string {
  if (results.length === 0) {
    return "No documented commands were found, so there was nothing to run.";
  }

  const count = (status: StepStatus) => results.filter((r) => r.status === status).length;
  const pass = count("pass");
  const workaround = count("workaround");
  const fail = count("fail");
  const ran = pass + workaround + fail;

  // A skipped step is either unrunnable or unfunded; the report should not call
  // a budget stop "not runnable", because reacting to it is a different job.
  const skips = results.filter((result) => result.status === "skip");
  const budget = skips.filter((result) =>
    (result.reason ?? "").startsWith(BUDGET_SKIP_REASON),
  ).length;
  const cloned = skips.filter((result) =>
    (result.reason ?? "").startsWith(SELF_CLONE_SKIP_REASON),
  ).length;
  const blocked = skips.length - budget - cloned;

  const parts: string[] = [];
  if (pass > 0) parts.push(`${pass} passed`);
  if (workaround > 0) parts.push(`${workaround} needed a workaround`);
  if (fail > 0) parts.push(`${fail} failed`);

  const head =
    ran === results.length
      ? `${plural(ran, "documented step")} ran`
      : `Ran ${ran} of ${plural(results.length, "documented step")}`;

  const list = parts.length > 0 ? `: ${parts.join(", ")}.` : ".";

  const skipped: string[] = [];
  if (blocked > 0) skipped.push(`${plural(blocked, "step")} skipped as not runnable unattended`);
  if (cloned > 0)
    skipped.push(`${plural(cloned, "step")} skipped because the run already cloned the repo`);
  if (budget > 0) skipped.push(`${plural(budget, "step")} skipped after the command budget`);

  return `${head}${list}${skipped.length > 0 ? ` ${skipped.join(", ")}.` : ""}`;
}

function stepsTable(steps: Step[], results: StepResult[]): string[] {
  if (results.length === 0) return [];

  const byId = new Map(steps.map((step) => [step.id, step]));
  const rows = results.map((result, index) => {
    const step = byId.get(result.stepId);
    const where = step ? `${step.source}${step.heading ? ` › ${step.heading}` : ""}` : "-";
    const seconds = result.durationMs > 0 ? `${(result.durationMs / 1000).toFixed(1)}s` : "-";
    return `| ${index + 1} | ${result.stepId} | ${code(result.command)} | ${result.status} | ${
      result.exitCode ?? "-"
    } | ${seconds} | ${inline(where, 60)} |`;
  });

  return [
    "## Steps",
    "",
    "| # | Step | Command | Status | Exit | Time | Documented in |",
    "| --- | --- | --- | --- | --- | --- | --- |",
    ...rows,
    "",
  ];
}

function failureSections(steps: Step[], results: StepResult[]): string[] {
  const byId = new Map(steps.map((step) => [step.id, step]));
  const failures = results.filter((result) => result.status === "fail");
  if (failures.length === 0) return [];

  const out = ["## Failures", ""];

  for (const result of failures) {
    const step = byId.get(result.stepId);
    const where = step
      ? `\`${step.source}\`${step.heading ? ` › "${brief(step.heading, 80)}"` : ""}`
      : "unknown source";
    const key = keyErrorLine(result);

    out.push(`### ${result.stepId} — ${code(result.command, 100)}`, "");
    out.push(`- **Documented in:** ${where}`);
    out.push(`- **Command:** ${code(result.command, 220)}`);
    out.push(`- **Exit code:** ${result.exitCode ?? "unknown"}`);
    if (result.durationMs > 0) out.push(`- **Duration:** ${(result.durationMs / 1000).toFixed(1)}s`);
    // Called a note, not a problem: this text is the model's reading of the
    // failure (or a runner note about a timeout). The exit code above and the
    // quoted output below are the evidence; the prose is a claim about them.
    if (result.attemptedFix) {
      out.push(`- **Attempted fix:** ${code(result.attemptedFix, 220)}`);
    }
    if (result.reason) out.push(`- **Note:** ${asProse(result.reason, 700)}`);
    else if (key) out.push(`- **Note:** ${key}`);

    out.push("", fence(errorExcerpt(result)), "");
  }

  return out;
}

/**
 * Steps that were never executed.
 *
 * Mostly interactive blocks, but Phase 9's command budget lands here too, and
 * a report that quietly omits the tail of the docs is worse than one that says
 * why it stopped.
 */
function skippedSection(steps: Step[], results: StepResult[]): string[] {
  const skipped = results.filter((result) => result.status === "skip");
  if (skipped.length === 0) return [];

  const byId = new Map(steps.map((step) => [step.id, step]));

  return [
    "## Not run",
    "",
    "| Step | Command | Documented in | Why |",
    "| --- | --- | --- | --- |",
    ...skipped.map((result) => {
      const step = byId.get(result.stepId);
      return `| ${result.stepId} | ${code(result.command, 80)} | ${inline(
        step?.source ?? "-",
        40,
      )} | ${inline(result.reason ?? "not run", 140)} |`;
    }),
    "",
  ];
}

function patchSections(
  patches: Patch[],
  wording: Map<string, string> | undefined,
  results: StepResult[],
): string[] {
  const out = ["## Suggested README patch", ""];

  if (patches.length === 0) {
    const failures = results.filter((result) => result.status === "fail");
    const executed = results.filter((result) => result.status !== "skip").length;
    out.push(
      results.length === 0
        ? "No documented commands were found, so there is nothing to patch."
        : executed === 0
          ? "No documented step could be run unattended, so there is no verified command change to suggest."
          : failures.length > 0
            ? "No command-level fix worked for the failing step(s) above, so there is no verified replacement command to suggest. " +
              "These need a hand-written fix: a missing prerequisite, a missing service, or instructions that no longer match the code."
            : executed === results.length
              ? "All documented steps ran as written - nothing to patch."
              : "Every step that could run passed as written - nothing to patch.",
      "",
    );
    return out;
  }

  patches.forEach((patch, index) => {
    const where = `\`${patch.source}\`${patch.heading ? ` › "${brief(patch.heading, 80)}"` : ""}`;
    out.push(`### ${index + 1}. ${where} (step ${patch.stepId})`, "");

    // A prerequisite is added above the documented line, which stays as it is;
    // a replacement is a one-for-one swap. Rendering both as a swap would tell a
    // maintainer to delete the command that actually works.
    const replacement = unwrapShell(patch.to.trim()).replace(/\n/g, "\n  ");
    const body =
      patch.kind === "replace"
        ? `- ${patch.from.trim().replace(/\n/g, "\n  ")}\n+ ${replacement}`
        : `+ ${replacement}`;
    out.push(fence(body, "diff"), "");
    out.push(wording?.get(patch.stepId) ?? patch.fallbackNote, "");
  });

  return out;
}

function ciSection(findings: CiFinding[], summary: string): string[] {
  if (findings.length === 0) return [];

  return [
    "## CI cross-check",
    "",
    ...(summary ? [summary, ""] : []),
    "| Kind | Command | CI source | Note |",
    "| --- | --- | --- | --- |",
    ...findings.map(
      (finding) =>
        `| ${finding.kind} | ${code(finding.command, 90)} | ${inline(
          finding.ciSource ?? "-",
          50,
        )} | ${inline(finding.note, 200)} |`,
    ),
    "",
  ];
}

/**
 * Build the report. Pure: same input, same markdown (hence `createdAt`).
 */
export function buildReport(input: ReportInput): string {
  const createdAt = input.createdAt ?? new Date();
  const date = createdAt.toISOString().slice(0, 10);
  // `results` holds two entries for a diagnosed step; every section below wants
  // one row per step, so collapse once and pass that everywhere.
  const results = finalResults(input.results);
  const patches = planPatches(input.steps, results);
  const failed = results.filter((result) => result.status === "fail").length;

  const lines = [
    `# README test: ${input.repoUrl}`,
    "",
    `**Mode:** ${input.mode} · **Run:** ${date}${input.image ? ` · **Image:** \`${input.image}\`` : ""}`,
    "",
    "## Summary",
    "",
    summarise(results),
    "",
    `- ${plural(input.cmdCount, "command")} executed in the container` +
      (input.commandBudget !== undefined ? ` (budget ${input.commandBudget})` : ""),
    ...(input.llmCalls !== undefined
      ? [
          `- ${plural(input.llmCalls, "LLM call")} made` +
            (input.llmCallBudget !== undefined ? ` (budget ${input.llmCallBudget})` : ""),
        ]
      : []),
    `- ${plural(input.llmRetries, "LLM retry", "LLM retries")} spent on transport/parse errors`,
    `- ${plural(patches.length, "suggested patch", "suggested patches")}`,
    "",
    ...stepsTable(input.steps, results),
    ...failureSections(input.steps, results),
    ...skippedSection(input.steps, results),
    ...patchSections(patches, input.wording, results),
    ...ciSection(input.ciFindings, input.ciSummary),
    "---",
    "",
    `Generated by readme-test. ${failed > 0 ? `${plural(failed, "step")} still failing.` : ""}`.trim(),
    "",
  ];

  // No global blank-line squeeze: an error excerpt may legitimately contain
  // consecutive blank lines, and collapsing them would misquote the output.
  return lines.join("\n").trimEnd() + "\n";
}
