import "./env.js";
import { destroyAll } from "./docker.js";
import { buildGraph, RECURSION_LIMIT } from "./graph.js";
import { removeAllWorkdirs } from "./repo.js";
import type { Mode, RunInput } from "./state.js";

/**
 * Phase 1 entry point, still standing in for Phase 8's real CLI:
 *   readme-test <url> --mode contributor
 */

const DEFAULT_REPO = "https://github.com/expressjs/express";

const USAGE = 'usage: npm start -- <repo-url> [--mode user|contributor]';

function parseMode(value: string | undefined): Mode {
  if (value === "user" || value === "contributor") return value;
  throw new Error(`--mode must be "user" or "contributor", got ${JSON.stringify(value)}\n${USAGE}`);
}

function parseArgs(argv: string[]): RunInput {
  const args = argv.slice(2);
  let repoUrl: string | undefined;
  let mode: Mode = "user";

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;

    if (arg === "--mode") {
      mode = parseMode(args[++i]);
      continue;
    }
    if (arg.startsWith("--mode=")) {
      mode = parseMode(arg.slice("--mode=".length));
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

  return { repoUrl: repoUrl ?? DEFAULT_REPO, mode };
}

async function main(): Promise<void> {
  const input = parseArgs(process.argv);
  console.log(`Testing docs for ${input.repoUrl} (mode=${input.mode})\n`);

  try {
    const graph = buildGraph();
    // recursionLimit is a per-invoke safety net for the run_step loop.
    const final = await graph.invoke(input, { recursionLimit: RECURSION_LIMIT });

    console.log(`\n${final.reportMarkdown}`);
    console.log(
      `\nSummary: ${final.docs.length} doc(s), ${final.steps.length} step(s), ` +
        `${final.results.length} result(s), ${final.cmdCount} command(s), ` +
        `${final.llmRetries} LLM retr(ies).`,
    );
  } finally {
    // Both are no-ops when setup never got that far, and neither throws: a
    // cleanup failure must not replace the error the run is unwinding from.
    await destroyAll();
    await removeAllWorkdirs();
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exitCode = 1;
});
