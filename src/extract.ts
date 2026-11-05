/**
 * Phase 4: turn doc code blocks into ordered, labelled steps.
 *
 * Hybrid on purpose. `docs.ts` already pulled the blocks out deterministically;
 * the strong model only *labels* them (order, mode, runnable) — it never writes
 * a command. That means a hallucinating model cannot invent `rm -rf` and get it
 * executed, and the worst case degrades to doc order with heuristic labels.
 *
 * Failure handling mirrors the plan: zod-validated structured output, one
 * retry, then a deterministic fallback that runs every block in doc order.
 */
import { z } from "zod";
import { extractCodeBlocks, heuristicLabels, type CodeBlock } from "./docs.js";
import { callModel, getStrong, LlmBudgetExhausted } from "./llm.js";
import { infoLine, warnLine } from "./log.js";
import {
  ModeSchema,
  SELF_CLONE_SKIP_REASON,
  type DocPage,
  type DocumentedCommand,
  type Mode,
  type Step,
} from "./state.js";

/**
 * Upper bound on runnable steps per run.
 *
 * Phase 9 caps a run at ~30 commands; extracting more would only burn
 * recursion budget on steps that can never execute.
 */
export const MAX_STEPS = 30;

/** Blocks per LLM call: small enough to leave room for the answer. */
const CHUNK_CHARS = 12_000;
const CHUNK_BLOCKS = 40;

/** Non-runnable blocks kept for the report (skipped steps, no commands run). */
const MAX_SKIPPED_BLOCKS = 10;

/** What the model may answer: labels for blocks that already exist. */
export const BlockLabelSchema = z.object({
  blockId: z.string().describe('Id of a listed block, e.g. "0-3".'),
  runnable: z
    .boolean()
    .describe("false for interactive prompts, long-running servers, or placeholder commands."),
  mode: ModeSchema.describe('"user" for install/use steps, "contributor" for repo-development steps.'),
  // Providers routinely send `null` for an absent optional field. Zod's
  // `.optional()` accepts `undefined` only; without `.nullable()` the whole
  // classification fails and the run falls back to document order (Mocha's
  // first c3 extraction hit exactly this).
  reason: z.string().nullable().optional().describe("Short justification, used in the report."),
});

export const ClassificationSchema = z.object({
  steps: z
    .array(BlockLabelSchema)
    .describe("The listed blocks, in the order a reader would run them."),
});
export type Classification = z.infer<typeof ClassificationSchema>;

/** Labels one chunk of blocks. Injectable so the fallback path is testable. */
export type Classifier = (blocks: CodeBlock[], attempt: number) => Promise<Classification>;

const reason = (err: unknown): string => (err instanceof Error ? err.message : String(err));

// ---------------------------------------------------------------------------
// Prompt + model
// ---------------------------------------------------------------------------

/**
 * Build the labelling prompt.
 *
 * Blocks are wrapped in pseudo-XML so a block whose command contains markdown
 * or prose cannot be confused with the instructions.
 */
export function buildPrompt(blocks: CodeBlock[], attempt: number): string {
  const body = blocks
    .map((block) => {
      const attributes = [
        `id="${block.blockId}"`,
        `source="${block.source}"`,
        block.heading ? `heading="${block.heading.replace(/"/g, "'")}"` : undefined,
        `lang="${block.language}"`,
      ]
        .filter(Boolean)
        .join(" ");
      return `<block ${attributes}>\n${block.command}\n</block>`;
    })
    .join("\n\n");

  const retryNote =
    attempt > 1
      ? "\n5. Your previous answer was rejected. Answer with valid JSON: only block ids from the list below, each at most once.\n"
      : "";

  return `You are auditing the setup instructions of a software repository.
Below are shell commands copied verbatim from its documentation, each in a
<block> element with an id.

Order the blocks the way a reader following the docs would run them, and label
each one.

Rules:
1. Only reference the ids listed below. Never invent, rewrite, merge or split a
   command — a block you return is run exactly as written.
2. Omit blocks a reader should not run at all: sample output, unrelated
   examples, and usage examples that invoke the finished tool with example
   arguments (they demonstrate the API, they do not set up the repository).
   In particular, omit commands that install or invoke *this package* from
   another project ('npm install my-package', 'npx my-package') — those are
   usage examples for consumers, not instructions for setting up this repository.
   When two blocks are alternative forms of the same setup command (for example
   'npm install' and 'npm install --ignore-scripts'), keep ONE - preferably the
   plain form - and do not drop both. A missing install block makes every later
   test fail.
3. runnable=false when a block cannot run unattended: it prompts for input,
   needs credentials (logins, tokens), starts a long-running server/watcher, or
   assumes a service is already running (curl against localhost, opening a URL).
   Commands that need Docker or a terminal multiplexer such as tmux are not
   runnable in this sandbox either. A command with '--watch' (or one that says
   it is waiting for changes) is a long-running watcher, not a one-shot test.
   Interactive commands such as 'git commit' without '-m', 'npm init' without
   '-y', an editor, or 'git push'/remote operations that need GitHub auth are
   not runnable. So are OS-specific instructions for a different platform
   (macOS '/Applications', 'xattr', 'brew'; Windows paths) when the container is
   Linux, and quoted examples whose line starts with '>'. Also mark a block
   runnable=false when it depends on an earlier block that cannot run (for
   example 'npm install' after a skipped interactive 'npm init').
   A block that mixes one such command with a setup command is still not
   runnable as written; do not mark it runnable just because the first line is.
4. mode="user" for steps that install and use the tool. mode="contributor" for
   steps only someone developing in this repository needs: running tests,
   linting, building from source, releasing.
${retryNote}
BLOCKS:

${body}`;
}

/**
 * The production classifier. The model is built here, not per call, and this
 * throws immediately when GROQ_API_KEY is missing — a config error should fail
 * the run loudly rather than silently degrade to heuristic labels.
 */
function llmClassifier(): Classifier {
  const structured = getStrong().withStructuredOutput(ClassificationSchema, {
    name: "classify_readme_blocks",
  });

  return async (blocks, attempt) => {
    // Phase 9: budget + backoff live in one place; validation retries stay here.
    const result = await callModel("extract_steps", () =>
      structured.invoke(buildPrompt(blocks, attempt)),
    );
    // Defensive: a provider that returns an object instead of parsing it.
    return ClassificationSchema.parse(result);
  };
}

// ---------------------------------------------------------------------------
// Chunking
// ---------------------------------------------------------------------------

/** Split blocks into prompt-sized groups, preserving document order. */
export function chunkBlocks(blocks: CodeBlock[]): CodeBlock[][] {
  const chunks: CodeBlock[][] = [];
  let current: CodeBlock[] = [];
  let size = 0;

  for (const block of blocks) {
    const cost = block.command.length + 120;
    if (current.length > 0 && (size + cost > CHUNK_CHARS || current.length >= CHUNK_BLOCKS)) {
      chunks.push(current);
      current = [];
      size = 0;
    }
    current.push(block);
    size += cost;
  }
  if (current.length > 0) chunks.push(current);
  return chunks;
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

export interface ExtractOptions {
  mode: Mode;
  /** `name` from the repo's package.json, used to spot consumer-only installs. */
  ownPackage?: string;
  /** The clone URL from the CLI, used to spot steps that clone this repo again. */
  repoUrl?: string;
  /** Test seam; defaults to the Groq classifier. */
  classify?: Classifier;
  maxSteps?: number;
}

export interface ExtractResult {
  steps: Step[];
  /**
   * Every shell block in the docs, before labelling and mode filtering — used by
   * the Phase 7 diff.
   *
   * Deliberately not the labelled subset: whether a command is *in the docs* is a
   * deterministic fact, and a model that labels only half the blocks must not turn
   * the other half into "undocumented in CI" findings.
   */
  documented: DocumentedCommand[];
  /** Retries actually spent — added to the run's `llmRetries`. */
  llmRetries: number;
  /** True when the model could not deliver and doc order was used instead. */
  usedFallback: boolean;
  blocksFound: number;
}

interface Labelled {
  block: CodeBlock;
  runnable: boolean;
  mode: Mode;
}

/** Contributor runs are a superset: contributors install first, then test. */
const modeAllows = (wanted: Mode, label: Mode): boolean => wanted === "contributor" || label === "user";

/** Commands kept for the CI diff; enough to describe a repo, bounded for state. */
const MAX_DOCUMENTED = 60;

const escapeRegExp = (text: string): string => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Is this command telling a *consumer* of the package how to install it?
 *
 * `npm install p-limit` at the top of p-limit's own readme is not repository
 * setup — running it inside the checkout installs the package into its own
 * project, and on a slow network that looks exactly like a broken install.
 * Found by the Phase 10 runs, where it was the only "step" in the readme.
 */
export function isConsumerInstall(command: string, packageName: string | undefined): boolean {
  if (!packageName) return false;
  const name = escapeRegExp(packageName);

  // End-anchored on purpose: `npm install p-limit && npm test` does repo work
  // after the consumer part, so it is not a pure usage example.
  const patterns = [
    new RegExp(
      `^(?:npm\\s+(?:i|install)|yarn\\s+(?:add|install)|pnpm\\s+(?:add|install)|bun\\s+(?:add|install))\\s+(?:-\\S+\\s+)*${name}(?:@\\S+)?\\s*$`,
      "i",
    ),
    new RegExp(`^npx\\s+(?:-\\S+\\s+)*${name}(?:@\\S+)?\\s*$`, "i"),
  ];

  const lines = command
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "" && !line.startsWith("#"));

  // Every line has to be a consumer install: a block that also does repo work is
  // left alone rather than half-dropped.
  return lines.length > 0 && lines.every((line) => patterns.some((pattern) => pattern.test(line)));
}

/**
 * `https://github.com/Owner/Repo.git` -> `github.com/owner/repo`.
 *
 * Handles the URL, scp-style (`git@host:owner/repo`) and `owner/repo` forms so
 * a readme's clone line can be compared with the URL the run was given.
 */
export function repoKey(url: string): string {
  const text = url.trim().replace(/\/+$/, "").replace(/\.git$/i, "");
  const scp = /^[\w.-]+@([^:/]+):(.+)$/.exec(text);
  return (scp ? `${scp[1]}/${scp[2]}` : text)
    .replace(/^[a-z][a-z0-9+.-]*:\/\//i, "")
    .replace(/^[^/]+@/, "")
    .replace(/^www\./i, "")
    .toLowerCase();
}

const repoTail = (key: string): string => key.split("/").filter(Boolean).slice(-2).join("/");

/**
 * Is this block the docs telling a reader to clone the repository under test?
 *
 * The run already has that checkout, so executing the line again clones a second
 * copy from inside the container — which on a big repo means a two-minute
 * timeout and a failure that says nothing about the docs. Phase 10 found it in a
 * real readme, where it was the first and most expensive false failure.
 *
 * Only a block that is *nothing but* clone-and-cd qualifies: `git clone` of some
 * other repository is a genuine setup step.
 */
export function isSelfClone(command: string, repoUrl: string | undefined): boolean {
  if (!repoUrl) return false;

  const steps = command
    .split(/\n|&&|;/)
    .map((part) => part.trim())
    .filter((part) => part !== "" && !part.startsWith("#"));
  if (steps.length === 0) return false;

  const wanted = repoKey(repoUrl);
  let cloned = false;

  for (const step of steps) {
    const clone = /^git\s+clone\b(.*)$/.exec(step);
    if (clone) {
      // The URL is whichever argument looks like one: `--depth 1` puts a bare
      // `1` where a first-non-flag-token regex would expect the repository.
      const arg = clone[1]!
        .trim()
        .split(/\s+/)
        .find(
          (token) =>
            !token.startsWith("-") && /:\/\/|^[^/]+@[^:]+:|^[\w.-]+\/[\w.-]+$/.test(token),
        );
      if (!arg) return false;

      const key = repoKey(arg);
      if (key !== wanted && repoTail(key) !== repoTail(wanted)) return false;
      cloned = true;
      continue;
    }
    // `cd paradev` after the clone is part of the same acquisition step.
    if (/^cd\s+\S+/.test(step)) continue;
    return false;
  }

  return cloned;
}

/** All blocks -> documented commands, in doc order, with no mode filter. */
function toDocumented(blocks: CodeBlock[]): DocumentedCommand[] {
  return blocks.slice(0, MAX_DOCUMENTED).map((block) => ({
    command: block.command,
    source: block.source,
    ...(block.heading ? { heading: block.heading } : {}),
  }));
}

/**
 * Labels -> ordered steps, applying the mode filter and the step caps.
 *
 * Non-runnable blocks are kept (up to MAX_SKIPPED_BLOCKS) so the report can say
 * "the docs contain a command nobody can run unattended". They cost no
 * commands: `run_step` must mark them `skip` without exec (Phase 5).
 */
function toSteps(
  labelled: Labelled[],
  mode: Mode,
  maxSteps: number,
  repoUrl?: string,
): Step[] {
  let candidates = labelled.filter((item) => modeAllows(mode, item.mode));

  // A model that labels everything "contributor" would leave a user run empty;
  // an empty run tests nothing, which is worse than testing too much.
  if (candidates.length === 0 && labelled.length > 0) {
    warnLine(
      `[extract_steps] mode=${mode} filtered out all ${labelled.length} block(s); keeping them all`,
    );
    candidates = labelled;
  } else if (candidates.length < labelled.length) {
    infoLine(
      `[extract_steps] mode=${mode} kept ${candidates.length}/${labelled.length} block(s)`,
    );
  }

  const steps: Step[] = [];
  let runnable = 0;
  let skipped = 0;

  for (const item of candidates) {
    // A documented clone of the repo under test is recorded as a skip, whatever
    // the model said: the checkout exists, so running it tests the network.
    const selfClone = isSelfClone(item.block.command, repoUrl);
    const canRun = item.runnable && !selfClone;

    if (canRun) {
      if (runnable >= maxSteps) continue;
      runnable++;
    } else {
      if (skipped >= MAX_SKIPPED_BLOCKS) continue;
      skipped++;
    }
    steps.push({
      id: `s${steps.length + 1}`,
      command: item.block.command,
      mode: item.mode,
      runnable: canRun,
      source: item.block.source,
      ...(item.block.heading ? { heading: item.block.heading } : {}),
      ...(selfClone ? { skipReason: SELF_CLONE_SKIP_REASON } : {}),
    });
  }

  return steps;
}

/**
 * Phase 4's node body.
 *
 * Throws only for configuration errors (no API key); a model that fails twice
 * degrades to the deterministic path and reports that it did.
 */
export async function extractSteps(docs: DocPage[], opts: ExtractOptions): Promise<ExtractResult> {
  const all = docs.flatMap((doc, index) => extractCodeBlocks(doc.content, doc.path, index));

  // Phase 10: consumer usage examples are not setup steps. They would also pollute
  // the CI diff, because `docCommands` is built from these same blocks.
  const blocks = all.filter((block) => !isConsumerInstall(block.command, opts.ownPackage));
  if (blocks.length < all.length) {
    infoLine(
      `[extract_steps] dropped ${all.length - blocks.length} consumer install example(s)`,
    );
  }

  if (blocks.length === 0) {
    return { steps: [], documented: [], llmRetries: 0, usedFallback: false, blocksFound: 0 };
  }

  const maxSteps = opts.maxSteps ?? MAX_STEPS;
  const classify = opts.classify ?? llmClassifier();
  const byId = new Map(blocks.map((block) => [block.blockId, block]));

  let llmRetries = 0;
  let usedFallback = false;
  const labelled: Labelled[] = [];
  const claimed = new Set<string>();

  for (const chunk of chunkBlocks(blocks)) {
    let answer: Classification | undefined;

    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        answer = await classify(chunk, attempt);
        break;
      } catch (err) {
        // Out of budget is not a failed attempt: retrying cannot succeed, and
        // counting it as a retry would misreport what happened in the report.
        if (err instanceof LlmBudgetExhausted) {
          warnLine("[extract_steps] LLM budget exhausted; using doc order");
          usedFallback = true;
          break;
        }
        if (attempt === 2) {
          warnLine(`[extract_steps] LLM extraction failed twice (${reason(err)}); using doc order`);
          usedFallback = true;
          break;
        }
        llmRetries++;
        warnLine(`[extract_steps] LLM extraction failed (${reason(err)}); retrying once`);
      }
    }

    if (!answer) {
      usedFallback = true;
      break;
    }

    let unknown = 0;
    let duplicates = 0;
    for (const label of answer.steps) {
      const block = byId.get(label.blockId);
      if (!block) {
        unknown++;
        continue;
      }
      if (claimed.has(block.blockId)) {
        duplicates++;
        continue;
      }
      claimed.add(block.blockId);
      labelled.push({ block, runnable: label.runnable, mode: label.mode });
    }

    const missing = chunk.length - answer.steps.length + unknown;
    if (unknown > 0 || duplicates > 0 || missing > 0) {
      warnLine(
        `[extract_steps] model answer: ${answer.steps.length} label(s), ` +
          `${unknown} unknown id(s), ${duplicates} duplicate(s), ~${Math.max(0, missing)} block(s) not labelled`,
      );
    }
  }

  if (usedFallback) {
    const fallback = blocks.map<Labelled>((block) => ({
      block,
      ...heuristicLabels(block.command, block.source, block.heading),
    }));
    return {
      steps: toSteps(fallback, opts.mode, maxSteps, opts.repoUrl),
      documented: toDocumented(blocks),
      llmRetries,
      usedFallback: true,
      blocksFound: blocks.length,
    };
  }

  return {
    steps: toSteps(labelled, opts.mode, maxSteps, opts.repoUrl),
    documented: toDocumented(blocks),
    llmRetries,
    usedFallback: false,
    blocksFound: blocks.length,
  };
}
