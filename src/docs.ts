/**
 * Phase 3 + 4, deterministic half: find the docs, pull the shell commands out
 * of them. No LLM anywhere in this file, so it is cheap to test and the
 * fallback path in extract.ts reuses it verbatim.
 *
 * Everything here is pure except `discoverDocs`, which takes a `read` function
 * instead of a container so it can be exercised against fixtures.
 */
import { posix } from "node:path";
import type { FileEntry } from "./docker.js";
import type { DocPage, Mode } from "./state.js";

/** How much of one doc we keep in state (checkpoints carry all of it). */
export const MAX_DOC_CHARS = 128 * 1024;

/** Docs read per run, before link following. */
const MAX_DOCS = 12;

/** Extra docs pulled in by following links from the named ones. */
const MAX_LINKED_DOCS = 6;

/**
 * A single fenced block longer than this is a file dump, not a command. Running
 * a truncated 40 KB heredoc would test nothing.
 */
const MAX_COMMAND_CHARS = 4000;

/** Fence languages whose bodies are meant to be run in a shell. */
const SHELL_LANGS = new Set([
  "bash",
  "sh",
  "shell",
  "zsh",
  "console",
  "shell-session",
  "sh-session",
  "bash-session",
  "terminal",
]);

/** Languages where `$ ` is a prompt rather than part of the command. */
const CONSOLE_LANGS = new Set(["console", "shell-session", "sh-session", "bash-session", "terminal"]);

const SKIP_PATH_SEGMENTS = /(^|\/)(node_modules|\.git|vendor)\//;

/**
 * Docs that are linked from a README but never contain setup commands.
 * Only applied to link following: a ranked name is read even if it matches.
 */
const IRRELEVANT_DOC =
  /(^|\/)(code[-_]of[-_]conduct|licen[cs]e|licence|changelog|changes|history|security|support|funding|authors|contributors|releases?|governance)(\.[a-z]+)?$/i;

/** One shell block lifted from a doc, before any labelling. */
export interface CodeBlock {
  /** Stable id `<docIndex>-<blockIndex>`, e.g. `"0-3"`. Also the LLM's handle. */
  blockId: string;
  /** Repo-relative doc path. */
  source: string;
  /** Nearest heading above the block. */
  heading?: string;
  language: string;
  /** The command text, prompt-stripped and trimmed. */
  command: string;
}

// ---------------------------------------------------------------------------
// Code blocks
// ---------------------------------------------------------------------------

const isClosingFence = (line: string, marker: string): boolean => {
  const m = /^ {0,3}(`{3,}|~{3,})[ \t]*$/.exec(line);
  if (!m) return false;
  return m[1]![0] === marker[0] && m[1]!.length >= marker.length;
};

/**
 * Strip shell prompts / pure-output lines and reject blocks that are not
 * commands.
 *
 * - In `console`-style fences, if any line starts with `$ `, everything else is
 *   program output and is dropped.
 * - A leading `$ ` is stripped in every shell fence, because plenty of READMEs
 *   write prompts inside a plain `bash` block.
 * - A block with no non-comment line is documentation, not a step.
 */
function cleanShellBody(lines: string[], language: string): string | undefined {
  const prompted = lines.filter((line) => /^\s*\$\s+\S/.test(line));
  const source = CONSOLE_LANGS.has(language) && prompted.length > 0 ? prompted : lines;

  const out = source.map((line) => line.replace(/^\s*\$\s+/, "").trimEnd());

  while (out.length > 0 && out[0]!.trim() === "") out.shift();
  while (out.length > 0 && out[out.length - 1]!.trim() === "") out.pop();
  if (out.length === 0) return undefined;

  // Only comments left => there is nothing to execute.
  if (!out.some((line) => line.trim() !== "" && !/^\s*#/.test(line))) return undefined;

  const command = out.join("\n").trim();
  return command.length > MAX_COMMAND_CHARS ? undefined : command;
}

/**
 * Every shell block in one markdown doc, in document order.
 *
 * Line-based rather than one regex: fenced blocks can contain backticks and
 * `#`-lines, and we need the heading that was current when the fence opened.
 */
export function extractCodeBlocks(
  markdown: string,
  source: string,
  docIndex = 0,
): CodeBlock[] {
  const lines = markdown.split(/\r?\n/);
  const blocks: CodeBlock[] = [];
  let heading: string | undefined;
  let index = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;

    const fence = /^ {0,3}(`{3,}|~{3,})[ \t]*([^\s`]*)/.exec(line);
    if (!fence) {
      const h = /^ {0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$/.exec(line);
      if (h) heading = h[1]!.trim();
      continue;
    }

    const marker = fence[1]!;
    const language = fence[2]!.toLowerCase();
    const body: string[] = [];
    for (i++; i < lines.length && !isClosingFence(lines[i]!, marker); i++) {
      body.push(lines[i]!);
    }

    if (!SHELL_LANGS.has(language)) continue;
    const command = cleanShellBody(body, language);
    if (command === undefined) continue;

    blocks.push({
      blockId: `${docIndex}-${index++}`,
      source,
      ...(heading ? { heading } : {}),
      language,
      command,
    });
  }

  return blocks;
}

/** First `# heading` of a page, used as the doc's display title. */
export function headingTitle(markdown: string): string | undefined {
  const match = /^ {0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$/m.exec(markdown);
  return match?.[1]?.trim();
}

// ---------------------------------------------------------------------------
// Heuristic labels (fallback path only — the LLM normally decides these)
// ---------------------------------------------------------------------------

/** Commands that block on a human: credentials, editors, pagers. */
const INTERACTIVE =
  /(^|\s)(sudo|ssh|vim?|nano|less|man)\s|(^|\s)(npm|yarn|pnpm|bun)\s+(login|adduser|add-user)|docker\s+login|gh\s+auth\s+login|(heroku|vercel|firebase|az|gcloud|aws)\s+(login|auth|configure)/;

/** Commands that never exit: they would burn a full exec timeout each. */
const LONG_RUNNING =
  /(^|\s)(npm|yarn|pnpm|bun)\s+(run\s+)?(dev|start|serve|watch)\b|(^|\s)(vite|nodemon|webpack-dev-server)\b|\bnext\s+dev\b|\b(flask\s+run|uvicorn|gunicorn|rails\s+s(erver)?|jekyll\s+serve|hugo\s+server)\b|(^|\s)(docker(-|\s)compose\s+up|docker\s+run\s+-(it|ti))\b/;

/** Obvious placeholders that would fail for reasons unrelated to the docs. */
const PLACEHOLDER = /<[^<>\s]{1,40}>|\bYOUR_[A-Z_]+\b|\byour-api-key\b/;

/** Headings/sources that mean "you are developing this repo". */
const CONTRIBUTOR_HINT =
  /(^|\/)(contribut\w*|hacking)\b|\bdevelop(ing|ment)?\b|\btest(ing|s)?\b|\blint(ing)?\b|\bbuild(ing)?\b|\bfrom source\b/i;

/** Labels the deterministic fallback assigns when the LLM is unavailable. */
export function heuristicLabels(
  command: string,
  source: string,
  heading?: string,
): { runnable: boolean; mode: Mode } {
  const runnable =
    !INTERACTIVE.test(command) && !LONG_RUNNING.test(command) && !PLACEHOLDER.test(command);
  const context = `${source} ${heading ?? ""}`;
  return { runnable, mode: CONTRIBUTOR_HINT.test(context) ? "contributor" : "user" };
}

// ---------------------------------------------------------------------------
// Doc discovery
// ---------------------------------------------------------------------------

/**
 * Docs the spec names, and how early to read them. Lower wins.
 *
 * `docs/` is matched by name on purpose: reading every markdown file in a
 * monorepo's docs folder would cost far more than it finds. Links from the
 * files below pull in the rest (see `discoverDocs`).
 */
const DOC_RANK: ReadonlyArray<readonly [RegExp, number]> = [
  [/^readme(\.[a-z]+)?$/i, 0],
  [/^install(ation|ing)?(\.[a-z]+)?$/i, 1],
  [
    /^docs\/(?:[^/]+\/)?(?:setup|install|installation|getting[-_]?started|quick[-_]?start|prerequisites|requirements)[^/]*\.(?:md|markdown)$/i,
    2,
  ],
  [/^contributing(\.[a-z]+)?$/i, 3],
  [
    /^docs\/(?:[^/]+\/)?(?:develop\w*|contribut\w*|hacking|testing|tests|build)[^/]*\.(?:md|markdown)$/i,
    4,
  ],
];

/** Rank of a repo-relative path as a doc to read, or undefined if it is not one. */
export function rankDocPath(path: string): number | undefined {
  const normalized = normalizePath(path);
  if (SKIP_PATH_SEGMENTS.test(normalized)) return undefined;
  let best: number | undefined;
  for (const [pattern, rank] of DOC_RANK) {
    if (pattern.test(normalized) && (best === undefined || rank < best)) best = rank;
  }
  return best;
}

const normalizePath = (path: string): string =>
  posix.normalize(path.replace(/\\/g, "/").replace(/^\.?\//, ""));

/** Inline markdown link targets, e.g. `[Setup](docs/setup.md)`; images skipped. */
export function extractLocalLinks(markdown: string): string[] {
  const links: string[] = [];
  const pattern = /(^|[^!\\])\[[^\]\n]*\]\(\s*<?([^)\s>]+)>?(?:\s+["'][^"']*["'])?\s*\)/g;
  for (const match of markdown.matchAll(pattern)) {
    const target = match[2];
    if (target) links.push(target);
  }
  return links;
}

/**
 * Resolve a link target to a repo-relative markdown path.
 *
 * Absolute URLs, mail links, anchors and anything escaping the repo are
 * dropped: this only follows docs, and only inside the checkout.
 */
export function resolveLink(fromPath: string, target: string): string | undefined {
  const raw = target.trim();
  if (!raw || raw.startsWith("#")) return undefined;
  if (/^[a-z][a-z0-9+.-]*:/i.test(raw)) return undefined;
  if (SKIP_PATH_SEGMENTS.test(raw)) return undefined;

  let decoded: string;
  try {
    decoded = decodeURIComponent(raw.split("#")[0]!.split("?")[0]!);
  } catch {
    decoded = raw.split("#")[0]!.split("?")[0]!;
  }
  const clean = decoded.trim();
  if (!clean) return undefined;

  const joined = clean.startsWith("/")
    ? posix.normalize(clean.replace(/^\/+/, ""))
    : posix.normalize(posix.join(posix.dirname(normalizePath(fromPath)), clean));

  if (!joined || joined.startsWith("..") || posix.isAbsolute(joined)) return undefined;
  if (!/\.(md|markdown)$/i.test(joined)) return undefined;
  return normalizePath(joined);
}

export interface DiscoverOptions {
  /** One level of the checkout, as returned by `listFiles`. */
  entries: FileEntry[];
  /** Read a repo-relative file from the container. */
  read: (path: string) => Promise<string>;
  maxDocs?: number;
  maxLinked?: number;
}

/**
 * Phase 3's `discover_docs`: named docs first, then docs they link to.
 *
 * A missing or unreadable doc is logged and skipped — a broken symlink should
 * not abort a run that could still test the README.
 */
export async function discoverDocs(opts: DiscoverOptions): Promise<DocPage[]> {
  const maxDocs = opts.maxDocs ?? MAX_DOCS;
  const maxLinked = opts.maxLinked ?? MAX_LINKED_DOCS;

  const files = new Set(
    opts.entries.filter((entry) => entry.type === "file").map((entry) => normalizePath(entry.path)),
  );

  const named = [...files]
    .map((path) => ({ path, rank: rankDocPath(path) }))
    .filter((entry): entry is { path: string; rank: number } => entry.rank !== undefined)
    .sort((a, b) => a.rank - b.rank || a.path.localeCompare(b.path))
    .map((entry) => entry.path);

  const docs: DocPage[] = [];
  const enqueued = new Set(named);
  const queue = [...named];
  let linked = 0;

  while (queue.length > 0 && docs.length < maxDocs) {
    const path = queue.shift()!;

    let content: string;
    try {
      content = await opts.read(path);
    } catch (err) {
      console.warn(`[docs] skipping ${path}: ${err instanceof Error ? err.message : err}`);
      continue;
    }

    if (content.length > MAX_DOC_CHARS) {
      content = `${content.slice(0, MAX_DOC_CHARS)}\n\n…[truncated: doc longer than ${MAX_DOC_CHARS} chars]`;
    }

    const title = headingTitle(content);
    docs.push({ path, ...(title ? { title } : {}), content });

    for (const target of extractLocalLinks(content)) {
      if (linked >= maxLinked) break;
      const resolved = resolveLink(path, target);
      if (!resolved || enqueued.has(resolved) || !files.has(resolved)) continue;
      if (IRRELEVANT_DOC.test(resolved)) continue;
      enqueued.add(resolved);
      queue.push(resolved);
      linked++;
    }
  }

  return docs;
}
