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
import { getStrong } from "./llm.js";
import { ModeSchema, type DocPage, type Mode, type Step } from "./state.js";

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
  reason: z.string().optional().describe("Short justification, used in the report."),
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
   examples, alternative install methods that duplicate another block.
3. runnable=false when a block cannot run unattended: it prompts for input,
   needs credentials (logins, tokens), or starts a long-running server/watcher.
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
    const result = await structured.invoke(buildPrompt(blocks, attempt));
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
  /** Test seam; defaults to the Groq classifier. */
  classify?: Classifier;
  maxSteps?: number;
}

export interface ExtractResult {
  steps: Step[];
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

/**
 * Labels -> ordered steps, applying the mode filter and the step caps.
 *
 * Non-runnable blocks are kept (up to MAX_SKIPPED_BLOCKS) so the report can say
 * "the docs contain a command nobody can run unattended". They cost no
 * commands: `run_step` must mark them `skip` without exec (Phase 5).
 */
function toSteps(labelled: Labelled[], mode: Mode, maxSteps: number): Step[] {
  let candidates = labelled.filter((item) => modeAllows(mode, item.mode));

  // A model that labels everything "contributor" would leave a user run empty;
  // an empty run tests nothing, which is worse than testing too much.
  if (candidates.length === 0 && labelled.length > 0) {
    console.warn(
      `[extract_steps] mode=${mode} filtered out all ${labelled.length} block(s); keeping them all`,
    );
    candidates = labelled;
  } else if (candidates.length < labelled.length) {
    console.log(
      `[extract_steps] mode=${mode} kept ${candidates.length}/${labelled.length} block(s)`,
    );
  }

  const steps: Step[] = [];
  let runnable = 0;
  let skipped = 0;

  for (const item of candidates) {
    if (item.runnable) {
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
      runnable: item.runnable,
      source: item.block.source,
      ...(item.block.heading ? { heading: item.block.heading } : {}),
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
  const blocks = docs.flatMap((doc, index) => extractCodeBlocks(doc.content, doc.path, index));
  if (blocks.length === 0) {
    return { steps: [], llmRetries: 0, usedFallback: false, blocksFound: 0 };
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
        if (attempt === 2) {
          console.warn(`[extract_steps] LLM extraction failed twice (${reason(err)}); using doc order`);
          usedFallback = true;
          break;
        }
        llmRetries++;
        console.warn(`[extract_steps] LLM extraction failed (${reason(err)}); retrying once`);
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
      console.warn(
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
      steps: toSteps(fallback, opts.mode, maxSteps),
      llmRetries,
      usedFallback: true,
      blocksFound: blocks.length,
    };
  }

  return {
    steps: toSteps(labelled, opts.mode, maxSteps),
    llmRetries,
    usedFallback: false,
    blocksFound: blocks.length,
  };
}
