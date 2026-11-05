/**
 * Phase 7: cross-check the documented commands against CI.
 *
 * The split is strict. Everything that decides a finding happens in code:
 * `readCiConfig` lifts commands out of `.github/workflows/*.yml`, `Makefile`
 * and `package.json`, `actionOf` reduces each one to an action ("install",
 * "test", ...), and `diffDocsAgainstCi` compares what the docs ask a reader to
 * do with what CI actually does. The fast model only writes the paragraph above
 * those findings — it cannot add, drop or reword one, so a rate limit costs a
 * sentence rather than the whole section.
 *
 * No YAML dependency: the workflow reader is a line parser for `run:` keys,
 * which is what the overwhelming majority of Actions files use. The shapes it
 * does not resolve (anchors, `${{ }}` interpolation, `uses:` steps) are skipped
 * rather than guessed — a wrong command in the diff is worse than a missing one.
 */
import { z } from "zod";
import type { FileEntry } from "./docker.js";
import { callModel, getFast, LlmBudgetExhausted } from "./llm.js";
import { warnLine } from "./log.js";
import type { CiFinding, DocumentedCommand } from "./state.js";

/** Reading every workflow in a monorepo would cost more than it finds. */
const MAX_WORKFLOWS = 10;

/** Commands per section of the summary prompt; keeps the call small. */
const MAX_PROMPT_COMMANDS = 60;

const MAX_NOTE_CHARS = 240;

const message = (err: unknown): string => (err instanceof Error ? err.message : String(err));

/** One line, capped — notes end up in a markdown table cell. */
function brief(text: string, max = MAX_NOTE_CHARS): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max)}…` : flat;
}

/**
 * A paragraph, capped at a sentence boundary.
 *
 * Model prose in the report is read as prose, so a hard character cut leaves a
 * sentence ending in "does no…". Cutting after the last full sentence keeps the
 * paragraph a paragraph; the fallback for one enormous sentence is still a
 * character cut, deliberately marked.
 */
export function asProse(text: string, max: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  if (flat.length <= max) return flat;
  const cut = flat.slice(0, max);
  const end = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf("! "), cut.lastIndexOf("? "));
  return end > max / 2 ? cut.slice(0, end + 1) : `${cut.trimEnd()}…`;
}

// ---------------------------------------------------------------------------
// What was found
// ---------------------------------------------------------------------------

/** A command CI runs, and where it came from. */
export interface CiCommand {
  command: string;
  /** Repo-relative source: `.github/workflows/ci.yml`, `Makefile`, `package.json`. */
  source: string;
  /** Job step name, Make target, or `scripts.<name>`. */
  context?: string;
}

export interface CiConfig {
  commands: CiCommand[];
  /** Files that were read, in discovery order. */
  sources: string[];
  /**
   * How often each command appears across all sources, keyed by
   * `normalizeCommand`. Counted before de-duplication, because five identical
   * `pnpm install` steps share a name and collapse to one entry.
   */
  counts: Record<string, number>;
}

/** The IO the reader needs, injected so parsing is testable without Docker. */
export interface CiReader {
  /** Reads a repo-relative file. */
  read: (path: string) => Promise<string>;
  /** Lists entries under a directory. May throw when it does not exist. */
  list: (dir: string) => Promise<FileEntry[]>;
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/** `"npm test"` -> `npm test`; also drops a trailing YAML comment. */
function scalar(value: string): string {
  const text = value.trim();
  if (text.startsWith('"') || text.startsWith("'")) {
    const end = text.indexOf(text[0]!, 1);
    if (end > 0) return text.slice(1, end).trim();
  }
  const hash = text.search(/\s#/);
  return (hash > 0 ? text.slice(0, hash) : text).trim();
}

/**
 * Turn the body of a `run: |` block into individual commands.
 *
 * Line-oriented on purpose: a CI script is usually a sequence of independent
 * commands, and each one diffs better on its own. `\`-continued lines are joined
 * first so a wrapped command stays one command. Heredocs are not understood —
 * their body lines simply fail to match any action and drop out of the diff.
 */
function splitShellScript(lines: string[]): string[] {
  const present = lines.filter((line) => line.trim() !== "");
  if (present.length === 0) return [];
  const indent = Math.min(...present.map((line) => line.length - line.trimStart().length));

  const commands: string[] = [];
  let buffer = "";

  for (const line of lines) {
    const body = line.slice(indent).trim();
    if (body === "" || body.startsWith("#")) continue;

    buffer = buffer === "" ? body : `${buffer} ${body}`;
    if (buffer.endsWith("\\")) {
      buffer = buffer.slice(0, -1).trim();
      continue;
    }
    commands.push(buffer);
    buffer = "";
  }

  if (buffer !== "") commands.push(buffer);
  return commands;
}

/** Every `run:` in one GitHub Actions workflow. */
export function parseWorkflow(text: string, source: string): CiCommand[] {
  const lines = text.split(/\r?\n/);
  const commands: CiCommand[] = [];
  let step: string | undefined;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const trimmed = line.trim();
    if (trimmed === "" || trimmed.startsWith("#")) continue;

    // Indented `name:` is a step (or job) name: useful context, never a command.
    const name = /^\s+(?:-\s+)?name\s*:\s*(.+)$/.exec(line);
    if (name) {
      step = scalar(name[1]!);
      continue;
    }

    const run = /^(\s*)(?:-\s+)?run\s*:[ \t]*(.*)$/.exec(line);
    if (!run) continue;

    const keyIndent = run[1]!.length;
    const rest = run[2]!.trim();
    const context = step;

    // Block scalar: everything more indented than the key belongs to it.
    if (/^[|>][+-]?\d*$/.test(rest)) {
      const body: string[] = [];
      let j = i + 1;
      for (; j < lines.length; j++) {
        const next = lines[j]!;
        if (next.trim() === "") {
          body.push("");
          continue;
        }
        const indent = next.length - next.trimStart().length;
        if (indent <= keyIndent) break;
        body.push(next);
      }
      i = j - 1;

      for (const command of splitShellScript(body)) {
        commands.push({ command, source, ...(context ? { context } : {}) });
      }
      continue;
    }

    if (rest === "") continue;
    const command = scalar(rest);
    if (command !== "") commands.push({ command, source, ...(context ? { context } : {}) });
  }

  return commands;
}

/** Recipes from a Makefile, attributed to their target. */
export function parseMakefile(text: string, source: string): CiCommand[] {
  const commands: CiCommand[] = [];
  let target: string | undefined;

  for (const raw of text.split(/\r?\n/)) {
    if (raw.startsWith("\t")) {
      // `@`/`-`/`+` are make's silent/ignore-error/always-run prefixes, not shell.
      const command = raw.replace(/^[\t ]*[@+-]?\s*/, "").trim();
      if (!target || command === "" || command.startsWith("#")) continue;
      commands.push({ command, source, context: target });
      continue;
    }

    const trimmed = raw.trim();
    if (trimmed === "" || trimmed.startsWith("#")) {
      target = undefined;
      continue;
    }

    // A target definition, not `VAR := value` (the `(?!=)` is load-bearing).
    const match = /^([A-Za-z0-9_][A-Za-z0-9_.\-/]*)\s*:(?!=)/.exec(raw);
    target = match?.[1];
  }

  return commands;
}

export type PackageManager = "npm" | "pnpm" | "yarn" | "bun";

/**
 * Which package manager the repo actually uses.
 *
 * A finding that says `npm run format` about a Bun repo sends the reader to a
 * command the repo never runs. The `packageManager` field wins when present;
 * otherwise the lockfile is the evidence, in the order npm itself documents.
 */
export function packageManagerOf(json: string, entries: FileEntry[]): PackageManager {
  try {
    const field = (JSON.parse(json) as { packageManager?: unknown } | null)?.packageManager;
    if (typeof field === "string") {
      const name = field.split("@")[0]?.toLowerCase();
      if (name === "npm" || name === "pnpm" || name === "yarn" || name === "bun") return name;
    }
  } catch {
    // Not JSON: the lockfile still tells us what CI would run.
  }

  const files = new Set(
    entries.filter((entry) => entry.type === "file").map((entry) => entry.path.toLowerCase()),
  );
  if (files.has("bun.lockb") || files.has("bun.lock")) return "bun";
  if (files.has("pnpm-lock.yaml")) return "pnpm";
  if (files.has("yarn.lock")) return "yarn";
  return "npm";
}

/**
 * Every `scripts` entry of a package.json.
 *
 * The command recorded is the *invocation* (`npm run lint`), not the script
 * body: that is what a reader and CI both type, and comparing `jest --ci`
 * against a README's `npm test` would report a mismatch on every repo that
 * happens to define its scripts in JavaScript. The name still names the action
 * through `context`.
 */
export function parsePackageScripts(
  json: string,
  source: string,
  manager: PackageManager = "npm",
): CiCommand[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return [];
  }

  const scripts = (parsed as { scripts?: unknown } | null)?.scripts;
  if (typeof scripts !== "object" || scripts === null) return [];

  return Object.entries(scripts as Record<string, unknown>)
    .filter((entry): entry is [string, string] => typeof entry[1] === "string")
    .map(([name]) => ({ command: `${manager} run ${name}`, source, context: `scripts.${name}` }));
}

/**
 * Find and parse every CI command source in the checkout.
 *
 * Missing files are the normal case, not an error: plenty of repos ship a README
 * and no CI at all. A file that cannot be read is warned about and skipped, so
 * one unreadable workflow never fails the run.
 */
export async function readCiConfig(reader: CiReader): Promise<CiConfig> {
  const commands: CiCommand[] = [];
  const sources: string[] = [];

  const read = async (path: string): Promise<string | undefined> => {
    try {
      const text = await reader.read(path);
      sources.push(path);
      return text;
    } catch (err) {
      warnLine(`[ci_crosscheck] could not read ${path}: ${message(err)}`);
      return undefined;
    }
  };

  // .github/workflows/*.yml
  try {
    const entries = await reader.list(".github/workflows");
    const files = entries
      .filter((entry) => entry.type === "file" && /\.ya?ml$/i.test(entry.path))
      .slice(0, MAX_WORKFLOWS)
      .map((entry) => `.github/workflows/${entry.path}`);

    for (const file of files) {
      const text = await read(file);
      if (text !== undefined) commands.push(...parseWorkflow(text, file));
    }
  } catch {
    // No workflows directory: a repo without GitHub Actions.
  }

  // Makefile / makefile / GNUmakefile, plus package.json, at the repo root.
  let root: FileEntry[] = [];
  try {
    root = await reader.list(".");
  } catch (err) {
    warnLine(`[ci_crosscheck] could not list the repo root: ${message(err)}`);
  }

  const makefile = root.find(
    (entry) => entry.type === "file" && /^(gnu)?makefile$/i.test(entry.path),
  );
  if (makefile) {
    const text = await read(makefile.path);
    if (text !== undefined) commands.push(...parseMakefile(text, makefile.path));
  }

  if (root.some((entry) => entry.type === "file" && entry.path === "package.json")) {
    const text = await read("package.json");
    if (text !== undefined) {
      commands.push(...parsePackageScripts(text, "package.json", packageManagerOf(text, root)));
    }
  }

  // One workflow may repeat a command across a matrix; the diff only cares that
  // it exists, and the prompt pays per line.
  const counts: Record<string, number> = {};
  for (const command of commands) {
    const key = normalizeCommand(command.command);
    counts[key] = (counts[key] ?? 0) + 1;
  }

  const seen = new Set<string>();
  const unique = commands.filter((command) => {
    const key = `${command.source}\u0000${command.context ?? ""}\u0000${command.command}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  return { commands: unique, sources, counts };
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/**
 * The kind of work a command does. Comparing by action rather than by literal
 * string is what makes the diff useful: `yarn install` in the docs and
 * `npm ci` in CI are the same job done two ways, and that disagreement is the
 * finding.
 *
 * `dev` exists only to be excluded: CI never runs a dev server, so reporting it
 * as missing would be noise.
 */
export type CiAction = "install" | "test" | "build" | "typecheck" | "lint" | "format" | "dev";

const ACTION_LABEL: Record<CiAction, string> = {
  install: "dependency install",
  test: "test run",
  build: "build",
  typecheck: "type check",
  lint: "lint",
  format: "format check",
  dev: "dev server",
};

/**
 * A script or target name -> the action it performs.
 *
 * Names are matched whole first, then by their `:`/`_`/`-` segments, which is
 * what makes `check:lint`, `fix:format` and `type-check` land on the right row.
 */
const NAME_ACTIONS: Readonly<Record<string, CiAction>> = {
  install: "install",
  ci: "install",
  i: "install",
  bootstrap: "install",
  deps: "install",
  test: "test",
  tests: "test",
  spec: "test",
  e2e: "test",
  jest: "test",
  vitest: "test",
  mocha: "test",
  lint: "lint",
  eslint: "lint",
  format: "format",
  fmt: "format",
  prettier: "format",
  typecheck: "typecheck",
  types: "typecheck",
  tsc: "typecheck",
  type: "typecheck",
  build: "build",
  compile: "build",
  dev: "dev",
  start: "dev",
  serve: "dev",
  watch: "dev",
};

/**
 * Action of a script name: `lint`, `check:lint`, `fix:format`, `type-check`,
 * `test:unit`.
 *
 * Segments are tried from the end, so the trailing word decides: `check:types`
 * is a type check, not a "check", and `build:analyze` is still a build.
 */
function actionForName(name: string): CiAction | undefined {
  const whole = name.toLowerCase();
  const direct = NAME_ACTIONS[whole];
  if (direct) return direct;

  const tokens = whole.split(/[:_-]+/).filter(Boolean);
  for (let i = tokens.length - 1; i >= 0; i--) {
    // `ci` is an environment suffix, not the install action: without this,
    // `npm run lint:ci` classified as install and produced a bogus
    // "docs install vs CI lint" mismatch in the Nest run. A bare `npm ci` still
    // hits the whole-name lookup above.
    if (tokens[i] === "ci") continue;
    const action = NAME_ACTIONS[tokens[i]!];
    if (action) return action;
  }
  return undefined;
}

/** `scripts.<name>` from package.json: the name is the only honest signal. */
function actionFromContext(context: string | undefined): CiAction | undefined {
  const match = /^scripts\.(.+)$/.exec(context ?? "");
  return match ? actionForName(match[1]!) : undefined;
}

/** Task-runner words that carry no action of their own. */
const WRAPPERS = new Set(["run", "turbo", "nx", "lerna", "workspace", "workspaces", "exec"]);

/**
 * `<pm> [run] <name>` anywhere in the command, so `a && npm test` works.
 *
 * This is how almost every documented and CI command is written: `npm install`,
 * `pnpm check:lint`, `yarn test:unit`, `bun run build`. Task runners are looked
 * through, so `pnpm turbo run check:format` is a format check: the words that
 * decide are the ones with an action, not the tool that wraps them.
 */
function actionFromPackageManager(command: string): CiAction | undefined {
  const match = /\b(?:npm|yarn|pnpm|bun)\s+(.+)$/.exec(command);
  if (!match) return undefined;

  for (const token of match[1]!.trim().split(/\s+/)) {
    if (token.startsWith("-")) continue;
    if (WRAPPERS.has(token.toLowerCase())) continue;
    const action = actionForName(token);
    if (action) return action;
  }
  return undefined;
}

/**
 * Everything that is not invoked through a package manager. First match wins,
 * so `tsc --noEmit` is a type check and bare `tsc` a build.
 */
const RULES: ReadonlyArray<readonly [CiAction, RegExp]> = [
  ["typecheck", /\btsc\b[^|;&]*--noEmit|\btsc\s+(?:-p|--project)\b|\bmypy\b/],
  [
    "lint",
    // The leading boundary deliberately excludes `-`: a VS Code extension id
    // like `dbaeumer.vscode-eslint` is not an eslint invocation, but `\beslint\b`
    // matched it and made Fastify's extension install look like a lint command.
    /(?:^|[\s/;&|])(?:npx\s+)?(?:eslint|flake8|pylint|golangci-lint)\b|\bruff\s+check\b|\bbiome\s+(?:check|ci)\b|\bcargo\s+clippy\b/,
  ],
  ["format", /\bprettier\b[^|;&]*--write\b|\bblack\s|\bgofmt\b|\bcargo\s+fmt\b|\brustfmt\b/],
  [
    "test",
    /\b(?:jest|vitest|mocha|pytest|tox)\b|\bgo\s+test\b|\bcargo\s+test\b|\bmvn\b[^|;&]*\btest\b|\bdotnet\s+test\b|\bmake\s+(?:test|check)\b/,
  ],
  [
    "build",
    /\btsc\b|\b(?:vite|next|webpack|rollup|esbuild)\s+build\b|\b(?:turbo|lerna|nx)\s+(?:run\s+)?build\b|\bgo\s+build\b|\bcargo\s+build\b|\bmake\s+(?:build|all)\b/,
  ],
  [
    "install",
    /\byarn\s*$|\bpip3?\s+install\b|\bpoetry\s+install\b|\buv\s+sync\b|\bcomposer\s+install\b|\bbundle\s+install\b|\bgo\s+mod\s+download\b|\bmake\s+(?:install|deps|bootstrap|setup)\b/,
  ],
  [
    "dev",
    /\b(?:next\s+dev|nodemon|webpack-dev-server|uvicorn|gunicorn)\b|\bflask\s+run\b|\b(?:docker\s+compose|docker-compose)\s+up\b|\brails\s+(?:s|server)\b/,
  ],
];

/** The action a command performs, or undefined when it is not one we compare. */
export function actionOf(command: string, context?: string): CiAction | undefined {
  const flat = command.replace(/\s+/g, " ").trim();

  const named = actionFromContext(context) ?? actionFromPackageManager(flat);
  if (named) return named;

  const rule = RULES.find(([, pattern]) => pattern.test(flat));
  return rule?.[0];
}

/**
 * Literal form used for equality. Only whitespace, line continuations and
 * quoting churn are normalized — flags and the package manager are kept, because
 * `npm install` vs `npm ci` and `tsc` vs `tsc -b` are exactly the differences
 * worth reporting.
 */
export function normalizeCommand(command: string): string {
  return command
    .replace(/\\\s*\n\s*/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\s*([;&|])\s*/g, " $1 ")
    .replace(/^sudo\s+/i, "")
    .trim()
    .replace(/;$/, "")
    .toLowerCase()
    // `npm test` and `npm run test` are the same invocation, and package.json
    // entries are recorded in the `run` form, so both sides agree.
    .replace(/\b(npm|pnpm|yarn|bun)\s+(test|start|stop|restart)\b/g, "$1 run $2");
}

// ---------------------------------------------------------------------------
// Diff
// ---------------------------------------------------------------------------

/** Findings are ordered by how much a maintainer cares, not alphabetically. */
const ACTION_ORDER: readonly CiAction[] = ["install", "test", "build", "typecheck", "lint", "format"];

type DocRef = DocumentedCommand;

/** `README.md` or `README.md, docs/setup.md +1 more`. */
function docRefs(list: DocRef[]): string {
  const paths = [...new Set(list.map((entry) => entry.source))];
  const shown = paths.slice(0, 2).join(", ");
  return paths.length > 2 ? `${shown} +${paths.length - 2} more` : shown;
}

function ciRef(command: CiCommand): string {
  return command.context ? `${command.source} (${command.context})` : command.source;
}

/**
 * The command that best represents an action: the one CI runs most often.
 *
 * A repo may install with `pnpm install` in ten workflows and `pip install
 * awscli` in one deployment job; reporting the latter as "the" install would be
 * technically true and completely unhelpful. Frequency is the cheapest honest
 * proxy for "what CI does", and ties keep source order.
 */
function representative(list: CiCommand[], counts: Record<string, number>): CiCommand {
  let best = list[0]!;
  let bestCount = -1;
  for (const entry of list) {
    const count = counts[normalizeCommand(entry.command)] ?? 0;
    if (count > bestCount) {
      best = entry;
      bestCount = count;
    }
  }
  return best;
}

/**
 * Compare documented commands with the ones CI runs.
 *
 * Takes the unfiltered documented set, not the mode-filtered steps: CI is
 * contributor-facing, so a `--mode user` run would otherwise call every CI test
 * and lint command undocumented.
 *
 * One finding per action, so a repo that documents `npm install` in three files
 * still produces one row. `dev` is skipped in both directions: a documented dev
 * server is intentional, and no CI job starts one.
 */
export function diffDocsAgainstCi(docs: DocumentedCommand[], ci: CiConfig): CiFinding[] {
  // Phase 10: a repo whose docs contain no shell commands at all (a library
  // readme that only shows consumer usage, for example) has nothing to compare.
  // Every CI command would be "missing in docs", which is true and useless: the
  // project never claimed to document a contributor setup. The pilot runs on
  // koa and p-limit produced exactly that noise.
  if (docs.length === 0) return [];

  const byAction = new Map<CiAction, DocRef[]>();
  for (const doc of docs) {
    const action = actionOf(doc.command);
    if (!action || action === "dev") continue;
    const list = byAction.get(action) ?? [];
    list.push(doc);
    byAction.set(action, list);
  }

  const ciByAction = new Map<CiAction, CiCommand[]>();
  for (const command of ci.commands) {
    const action = actionOf(command.command, command.context);
    if (!action || action === "dev") continue;
    const list = ciByAction.get(action) ?? [];
    list.push(command);
    ciByAction.set(action, list);
  }

  const findings: CiFinding[] = [];

  for (const action of ACTION_ORDER) {
    const docList = byAction.get(action) ?? [];
    const ciList = ciByAction.get(action) ?? [];
    if (docList.length === 0 && ciList.length === 0) continue;

    const docCommand = docList[0];
    const ciCommand = ciList.length > 0 ? representative(ciList, ci.counts) : undefined;

    if (docCommand && !ciCommand) {
      findings.push({
        kind: "missing_in_ci",
        command: docCommand.command,
        note: brief(`documented in ${docRefs(docList)}; CI never runs ${ACTION_LABEL[action]}`),
      });
      continue;
    }

    if (!docCommand && ciCommand) {
      findings.push({
        kind: "missing_in_docs",
        command: ciCommand.command,
        ciSource: ciCommand.source,
        note: brief(
          `${ciRef(ciCommand)} has ${ACTION_LABEL[action]}; no documented step asks for it`,
        ),
      });
      continue;
    }

    if (!docCommand || !ciCommand) continue;

    // Both sides do it — is it literally the same command anywhere?
    const documented = new Set(docList.map((entry) => normalizeCommand(entry.command)));
    if (ciList.some((entry) => documented.has(normalizeCommand(entry.command)))) continue;

    findings.push({
      kind: "mismatch",
      command: docCommand.command,
      ciSource: ciCommand.source,
      // Plain text, no code spans: the report flattens and truncates this note
      // for its table cell, and a cut inside a span would leave the backtick
      // open. The command being compared has its own column; this is prose.
      note: brief(
        `docs: ${brief(docCommand.command, 90)} (${docRefs(docList)}) · ` +
          `CI: ${brief(ciCommand.command, 90)} (${ciRef(ciCommand)})`,
      ),
    });
  }

  return findings;
}

// ---------------------------------------------------------------------------
// Summary (the only model call)
// ---------------------------------------------------------------------------

export const CiSummarySchema = z.object({
  summary: z
    .string()
    .describe(
      "Two to four plain sentences about the disagreements listed below. No headings, no lists, no new findings.",
    ),
});
export type CiSummary = z.infer<typeof CiSummarySchema>;

export interface SummaryInput {
  findings: CiFinding[];
  docCommands: DocumentedCommand[];
  commands: CiCommand[];
}

/** Test seam, same shape as the extract classifier and the fix proposer. */
export type CiSummarizer = (input: SummaryInput, attempt: number) => Promise<CiSummary>;

/** Build the summary prompt. The findings are fixed; the prose is not. */
export function buildSummaryPrompt(input: SummaryInput, attempt = 1): string {
  const findingLines = input.findings
    .map((finding) => `- [${finding.kind}] ${finding.command} — ${finding.note}`)
    .join("\n");

  const docLines = input.docCommands
    .slice(0, MAX_PROMPT_COMMANDS)
    .map(
      (doc) =>
        `- ${brief(doc.command, 120)} (${doc.source}${doc.heading ? `, "${doc.heading}"` : ""})`,
    )
    .join("\n");

  const ciLines = input.commands
    .slice(0, MAX_PROMPT_COMMANDS)
    .map((command) => `- ${brief(command.command, 120)} (${ciRef(command)})`)
    .join("\n");

  const retryNote =
    attempt > 1
      ? "\nYour previous answer was rejected. Answer with plain prose and valid JSON.\n"
      : "";

  return `You are summarising a comparison between a repository's documentation and its CI configuration, for a maintainer who has to decide what to fix.\n\nThe differences below were computed mechanically by comparing what the docs tell a reader to run against what CI actually runs. They are complete; your job is to explain them, not to find more.\n\nDETERMINISTIC DIFF:\n${findingLines}\n\nDOCUMENTED COMMANDS:\n${docLines}\n\nCI COMMANDS:\n${ciLines}\n\nRules:\n1. Use only the differences listed above. Never invent a finding, a file or a command, and never name a command or file that is not in DETERMINISTIC DIFF.\n2. Two to four sentences of plain prose: no headings, no bullet points, no markdown tables.\n3. Name the disagreement that would actually break a new contributor first (a documented install that CI does differently is worse than a missing lint step).\n4. Do not speculate about why the two drifted.\n${retryNote}`;
}

/** The production summarizer: fast model, structured output. */
function llmSummarizer(): CiSummarizer {
  const structured = getFast().withStructuredOutput(CiSummarySchema, {
    name: "summarise_ci_mismatches",
  });

  return async (input, attempt) => {
    const result = await callModel("ci_crosscheck", () =>
      structured.invoke(buildSummaryPrompt(input, attempt)),
    );
    return CiSummarySchema.parse(result);
  };
}

/**
 * Ask for one paragraph about the findings.
 *
 * Best-effort: the findings are already decided, so a rate limit or an
 * unparseable answer costs the sentence and nothing else.
 */
export async function summarizeMismatches(
  input: SummaryInput,
  opts: { summarize?: CiSummarizer } = {},
): Promise<{ summary: string; llmRetries: number }> {
  const summary = opts.summarize ?? llmSummarizer();
  let lastError: string | undefined;

  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const answer = await summary(input, attempt);
      return { summary: asProse(answer.summary, 1200), llmRetries: attempt - 1 };
    } catch (err) {
      lastError = message(err);
      if (err instanceof LlmBudgetExhausted) break;
      if (attempt === 1) warnLine(`[ci_crosscheck] summary failed (${lastError}); retrying once`);
    }
  }

  warnLine(
    `[ci_crosscheck] summary unavailable (${brief(lastError ?? "no answer", 160)}); keeping the deterministic findings`,
  );
  return { summary: "", llmRetries: 1 };
}
