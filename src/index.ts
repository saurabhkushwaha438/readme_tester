#!/usr/bin/env node
import "./env.js";
import { randomUUID } from "node:crypto";
import { destroyAll } from "./docker.js";
import { buildGraph, MAX_COMMANDS, RECURSION_LIMIT } from "./graph.js";
import { llmCallCap, llmCallCount, MAX_LLM_CALLS, resetLlmBudget } from "./llm.js";
import { errorLine, heading, muted } from "./log.js";
import { removeAllWorkdirs } from "./repo.js";
import { invokeWithResume } from "./run.js";
import type { Mode, RunInput } from "./state.js";

/**
 * Phase 8 CLI:
 *
 *   readme-test <url> --mode contributor
 *
 * The report goes to stdout as markdown, after the progress log the nodes
 * print; the last blank line separates them, so `... | sed -n '/^# README/,$p'`
 * extracts a clean file.
 *
 * Phase 9 adds two budgets (commands, model calls) and an in-process resume: a
 * rate limit that outlives the backoff throws, and the run is invoked again on
 * the same thread, continuing from the last checkpoint instead of re-cloning and
 * re-running every step that already passed.
 */

const DEFAULT_REPO = "https://github.com/expressjs/express";

/** A single rate-limit storm is worth waiting through; a broken provider is not. */
const MAX_RESUME_ATTEMPTS = 2;

const USAGE = [
  "usage: readme-test <repo-url> [--mode user|contributor] [--thread <id>]",
  "",
  "  <repo-url>   Git URL to clone (default: " + DEFAULT_REPO + ")",
  "  --mode       Which documented commands to run (default: user)",
  "                  user         setup instructions for someone using the project",
  "                  contributor  setup + test, lint and build steps",
  "  --thread     Checkpoint thread id (default: a fresh id, printed on start)",
  "  -h, --help   Show this message",
  "",
  "Budgets: README_TEST_MAX_COMMANDS (default " +
    MAX_COMMANDS +
    "), README_TEST_MAX_LLM_CALLS (default " +
    MAX_LLM_CALLS +
    ")",
].join("\n");

interface Args extends RunInput {
  threadId: string;
}

function parseMode(value: string | undefined): Mode {
  if (value === "user" || value === "contributor") return value;
  throw new Error(`--mode must be "user" or "contributor", got ${JSON.stringify(value)}\n${USAGE}`);
}

function parseArgs(argv: string[]): Args {
  const args = argv.slice(2);
  let repoUrl: string | undefined;
  let mode: Mode = "user";
  let threadId: string = randomUUID();

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;

    if (arg === "-h" || arg === "--help") {
      console.log(USAGE);
      process.exit(0);
    }
    if (arg === "--mode") {
      mode = parseMode(args[++i]);
      continue;
    }
    if (arg.startsWith("--mode=")) {
      mode = parseMode(arg.slice("--mode=".length));
      continue;
    }
    if (arg === "--thread") {
      const value = args[++i];
      if (!value || value.startsWith("-")) throw new Error(`--thread needs an id\n${USAGE}`);
      threadId = value;
      continue;
    }
    if (arg.startsWith("--thread=")) {
      threadId = arg.slice("--thread=".length) || threadId;
      continue;
    }
    if (arg.startsWith("-")) {
      throw new Error(`unknown option ${arg}\n${USAGE}`);
    }
    if (repoUrl !== undefined) {
      throw new Error(`expected one repo URL, got "${repoUrl}" and "${arg}"\n${USAGE}`);
    }
    repoUrl = arg;
  }

  return { repoUrl: repoUrl ?? DEFAULT_REPO, mode, threadId };
}

async function main(): Promise<void> {
  const { threadId, ...input } = parseArgs(process.argv);

  resetLlmBudget();
  heading(`Testing docs for ${input.repoUrl} (mode=${input.mode})`);
  const commandCap = `${MAX_COMMANDS} command${MAX_COMMANDS === 1 ? "" : "s"}`;
  muted(
    `Run id ${threadId}; budgets: ${commandCap}, ${llmCallCap()} LLM call${llmCallCap() === 1 ? "" : "s"}\n`,
  );

  try {
    const graph = buildGraph();
    const config = { recursionLimit: RECURSION_LIMIT, configurable: { thread_id: threadId } };

    const final = await invokeWithResume(
      (next) => graph.invoke(next as unknown as RunInput, config),
      input,
      { maxAttempts: MAX_RESUME_ATTEMPTS },
    );

    console.log(`\n${final.reportMarkdown}`);
    muted(
      `\nRun: ${final.docs.length} doc(s), ${final.steps.length} step(s), ` +
        `${final.cmdCount} command(s), ${llmCallCount()}/${llmCallCap()} LLM call(s), ` +
        `${final.llmRetries} retr(ies).`,
    );
  } finally {
    // Both are no-ops when setup never got that far, and neither throws: a
    // cleanup failure must not replace the error the run is unwinding from.
    await destroyAll();
    await removeAllWorkdirs();
  }
}

main().catch((err) => {
  errorLine(err instanceof Error ? err.message : String(err));
  process.exitCode = 1;
});
