import { buildGraph, RECURSION_LIMIT } from "./graph.js";
import { initialValues, type Mode, type RunInput } from "./state.js";

/**
 * Phase 1 entry point. Phase 8 replaces this with a real CLI:
 *   readme-test <url> --mode contributor
 */

function parseArgs(argv: string[]): RunInput {
  const args = argv.slice(2);
  const repoUrl = args.find((a) => !a.startsWith("-")) ?? "https://github.com/expressjs/express";

  const modeFlag = args.indexOf("--mode");
  const rawMode = modeFlag >= 0 ? args[modeFlag + 1] : undefined;
  if (rawMode !== undefined && rawMode !== "user" && rawMode !== "contributor") {
    throw new Error(`--mode must be "user" or "contributor", got "${rawMode}"`);
  }

  return { repoUrl, mode: (rawMode ?? "user") as Mode };
}

async function main() {
  const input = parseArgs(process.argv);
  console.log(`Testing docs for ${input.repoUrl} (mode=${input.mode})\n`);

  const graph = buildGraph();
  // recursionLimit is a per-invoke safety net for the run_step loop.
  const final = await graph.invoke(
    { ...input, ...initialValues },
    { recursionLimit: RECURSION_LIMIT },
  );

  console.log(`\n${final.reportMarkdown}`);
  console.log(
    `\nSummary: ${final.steps.length} step(s), ${final.results.length} result(s), ` +
      `${final.cmdCount} command(s), ${final.llmRetries} LLM retr(ies).`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
