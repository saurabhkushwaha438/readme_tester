import { existsSync } from "node:fs";
import { join } from "node:path";
import { END, START, StateGraph } from "@langchain/langgraph";
import {
  exec,
  isAlive,
  listFiles,
  pingDocker,
  readFile,
  startContainer,
  type ExecResult,
  type FileEntry,
} from "./docker.js";
import { diagnoseStep } from "./diagnose.js";
import { discoverDocs } from "./docs.js";
import { extractSteps, MAX_STEPS } from "./extract.js";
import { CLONE_DIR_NAME, cloneRepo } from "./repo.js";
import {
  RunState,
  type RunStateType,
  type RunStateUpdate,
  type StepResult,
  type StepStatus,
} from "./state.js";

/**
 * The graph topology, with real nodes through Phase 6.
 *
 * Real: setup, discover_docs (Phase 3), extract_steps (Phase 4), run_step (5),
 * diagnose (6). Still stubs: ci_crosscheck (7), report (8). The wiring and the
 * routing are real either way.
 */

/**
 * Phase 5 makes this essential: the loop costs up to 3 supersteps per step
 * (run_step -> diagnose -> run_step), so the default of 25 is already too low
 * for a handful of steps. The spec asks for 100 or higher; 250 leaves headroom
 * for the Phase 9 cap of ~30 commands (~96 supersteps worst case) without
 * touching the limit again.
 */
export const RECURSION_LIMIT = 250;

const log = (node: string, msg: string) => console.log(`[${node}] ${msg}`);

const message = (err: unknown): string => (err instanceof Error ? err.message : String(err));

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

/**
 * Clone on the host and start the container that will run the commands.
 *
 * Docker is checked first so a missing daemon fails in a second instead of
 * after a clone. A checkpointed run (Phase 9) resumes with its container if it
 * is still alive, and reuses its checkout instead of cloning again.
 */
async function setup(state: RunStateType): Promise<RunStateUpdate> {
  await pingDocker();

  if (state.containerId) {
    if (await isAlive(state.containerId)) {
      log("setup", `reusing live container ${state.containerId.slice(0, 12)}`);
      return {};
    }
    log("setup", `container ${state.containerId.slice(0, 12)} is gone; starting a new one`);
  }

  const cached = state.workdir ? join(state.workdir, CLONE_DIR_NAME) : undefined;
  let workdir: string;
  let cloneDir: string;

  if (cached && existsSync(join(cached, ".git"))) {
    workdir = state.workdir!;
    cloneDir = cached;
    log("setup", `reusing existing clone at ${cloneDir}`);
  } else {
    const clone = await cloneRepo(state.repoUrl);
    workdir = clone.workdir;
    cloneDir = clone.cloneDir;
  }

  const containerId = await startContainer({ hostWorkdir: cloneDir });
  log("setup", `container ${containerId.slice(0, 12)} mounting ${cloneDir}`);
  return { workdir, containerId };
}

/** Every node that touches the filesystem needs setup to have succeeded. */
function requireContainer(state: RunStateType): string {
  if (!state.containerId) {
    throw new Error("setup did not produce a container — check the Docker daemon and the clone URL");
  }
  return state.containerId;
}

/**
 * The checkout listing that doc discovery needs, kept small on purpose.
 *
 * Root files plus two levels inside a docs folder. A recursive `find` over a
 * monorepo would blow through the read cap on `node_modules` alone.
 */
async function listRepoEntries(containerId: string): Promise<FileEntry[]> {
  const root = await listFiles(containerId, ".", { maxDepth: 1 });
  const docDirs = root
    .filter((entry) => entry.type === "dir" && /^(docs?|documentation)$/i.test(entry.path))
    .map((entry) => entry.path);

  const nested: FileEntry[] = [];
  for (const dir of docDirs) {
    const entries = await listFiles(containerId, dir, { maxDepth: 2 });
    for (const entry of entries) {
      nested.push({ path: `${dir}/${entry.path}`, type: entry.type });
    }
  }

  return [...root, ...nested];
}

/** Phase 3: README / INSTALL / CONTRIBUTING / docs/*setup*, plus linked docs. */
async function discover_docs(state: RunStateType): Promise<RunStateUpdate> {
  const containerId = requireContainer(state);
  const entries = await listRepoEntries(containerId);
  const docs = await discoverDocs({
    entries,
    read: (path) => readFile(containerId, path),
  });

  log(
    "discover_docs",
    docs.length > 0
      ? `${docs.length} doc(s): ${docs.map((doc) => doc.path).join(", ")}`
      : `no docs found among ${entries.length} entr(ies); nothing to extract`,
  );
  return { docs };
}

/** Phase 4: deterministic block extraction, strong-model labelling. */
async function extract_steps(state: RunStateType): Promise<RunStateUpdate> {
  const result = await extractSteps(state.docs, { mode: state.mode, maxSteps: MAX_STEPS });
  const runnable = result.steps.filter((step) => step.runnable).length;

  log(
    "extract_steps",
    result.steps.length > 0
      ? `${result.steps.length} step(s) from ${result.blocksFound} block(s): ` +
          `${runnable} runnable, ${result.steps.length - runnable} not (skipped), ` +
          `${result.usedFallback ? "doc order (LLM fell back)" : "LLM ordered"}`
      : `no shell blocks in ${state.docs.length} doc(s); nothing to run`,
  );

  return { steps: result.steps, idx: 0, llmRetries: state.llmRetries + result.llmRetries };
}

/** One-line form of a command, for logs and the report table. */
function oneLine(command: string, max = 140): string {
  const flat = command.replace(/\s*\n\s*/g, " ⏎ ").trim();
  return flat.length > max ? `${flat.slice(0, max)}…` : flat;
}

/** Last non-empty stderr line: the part a human reads first. */
function errorLine(out: ExecResult): string | undefined {
  const lines = out.stderr
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const last = lines.at(-1);
  return last === undefined ? undefined : last.length > 160 ? `${last.slice(0, 160)}…` : last;
}

/**
 * Phase 5: run steps[idx] in the container and record the outcome.
 *
 * No LLM here — this node only executes and classifies. Routing (pass → next,
 * fail → diagnose, done → ci_crosscheck) lives in `routeAfterRun`.
 *
 * A failure keeps `idx` in place so `diagnose` can retry the same step (that is
 * its Phase 6 job); every other outcome advances. A non-runnable step is
 * recorded as `skip` without an exec, so it costs no command budget.
 */
async function run_step(state: RunStateType): Promise<RunStateUpdate> {
  const step = state.steps[state.idx];
  if (!step) {
    log("run_step", `no step at idx=${state.idx}, nothing to run`);
    return {};
  }

  // Phase 4 marks interactive/long-running blocks as not runnable: record them
  // and move on. No exec, so no command budget is spent.
  if (!step.runnable) {
    const skipped: StepResult = {
      stepId: step.id,
      command: step.command,
      status: "skip",
      exitCode: null,
      stdout: "",
      stderr: "",
      durationMs: 0,
      reason: "not runnable unattended (interactive or long-running)",
    };
    log("run_step", `${step.id} -> skip (not runnable)`);
    return { results: [skipped], idx: state.idx + 1 };
  }

  const containerId = requireContainer(state);
  log("run_step", `${step.id} $ ${oneLine(step.command)} (idx=${state.idx})`);

  let out: ExecResult;
  try {
    out = await exec(containerId, step.command);
  } catch (err) {
    // A dead container makes every remaining step meaningless; anything else
    // is recorded as a failure so one bad exec does not lose the whole run.
    if (!(await isAlive(containerId))) {
      throw new Error(
        `container ${containerId.slice(0, 12)} stopped while running ${step.id}: ${message(err)}`,
      );
    }
    const broken: StepResult = {
      stepId: step.id,
      command: step.command,
      status: "fail",
      exitCode: null,
      stdout: "",
      stderr: "",
      durationMs: 0,
      reason: `could not run: ${message(err)}`,
    };
    log("run_step", `${step.id} -> fail (exec error: ${message(err)})`);
    return { results: [broken], cmdCount: state.cmdCount + 1 };
  }

  const status: StepStatus = out.exitCode === 0 && !out.timedOut ? "pass" : "fail";
  const seconds = (out.durationMs / 1000).toFixed(1);

  // Distinguish the three ways a step can fail, because Phase 6 needs them:
  // a real exit code, a killed timeout, and a stream that died without one.
  const reason = out.timedOut
    ? `timed out after ${Math.round(out.durationMs / 1000)}s (killed, exit ${out.exitCode ?? "?"})`
    : out.exitCode === null
      ? "container never reported an exit code (the exec stream ended early)"
      : undefined;

  const result: StepResult = {
    stepId: step.id,
    command: step.command,
    status,
    exitCode: out.exitCode,
    stdout: out.stdout,
    stderr: out.stderr,
    durationMs: out.durationMs,
    ...(reason ? { reason } : {}),
  };

  const err = status === "fail" ? errorLine(out) : undefined;
  log(
    "run_step",
    `${step.id} -> ${status} (exit ${out.exitCode ?? "?"}, ${seconds}s)` + (err ? ` — ${err}` : ""),
  );

  return {
    results: [result],
    // Advance only on success; a failure stays put for diagnose to retry.
    idx: status === "pass" ? state.idx + 1 : state.idx,
    cmdCount: state.cmdCount + 1,
  };
}

/**
 * Phase 6: one model-proposed fix for the step that just failed.
 *
 * The heavy lifting (prompt, denylist, execution) lives in `diagnose.ts`; this
 * node only adapts the result into graph state. It always advances `idx`,
 * whether the fix worked or not — a diagnosed step is done either way.
 */
async function diagnose(state: RunStateType): Promise<RunStateUpdate> {
  const failed = state.steps[state.idx];
  if (!failed) {
    log("diagnose", `no step at idx=${state.idx}, skipping`);
    return {};
  }

  const failure = state.results.at(-1);
  if (!failure || failure.status !== "fail") {
    log("diagnose", `no failure recorded for ${failed.id}; advancing`);
    return { idx: state.idx + 1 };
  }

  const containerId = requireContainer(state);
  log("diagnose", `${failed.id} failed (exit ${failure.exitCode ?? "?"}); asking for one fix`);

  const outcome = await diagnoseStep({ containerId, step: failed, failure });
  log(
    "diagnose",
    `${failed.id} -> ${outcome.status}` +
      (outcome.attemptedFix
        ? ` (tried \`${oneLine(outcome.attemptedFix)}\`, ${outcome.commandsRun} command(s))`
        : " (nothing ran)") +
      ` — ${outcome.reason}`,
  );

  const result: StepResult = {
    stepId: failed.id,
    command: outcome.command,
    status: outcome.status,
    exitCode: outcome.exitCode,
    stdout: outcome.stdout,
    stderr: outcome.stderr,
    durationMs: outcome.durationMs,
    reason: outcome.reason,
  };

  return {
    results: [result],
    idx: state.idx + 1,
    // Only commands that actually reached the container spend the budget:
    // the fix, plus the re-run of the documented command when it exited 0.
    cmdCount: state.cmdCount + outcome.commandsRun,
    llmRetries: state.llmRetries + outcome.llmRetries,
  };
}

/** Phase 7 stub: diff docs against CI config. */
async function ci_crosscheck(state: RunStateType): Promise<RunStateUpdate> {
  log("ci_crosscheck", `would diff docs against CI config (phase 7) with ${state.cmdCount} commands run`);
  return { ciFindings: [] };
}

/** Phase 8 stub: templated markdown report. */
async function report(state: RunStateType): Promise<RunStateUpdate> {
  const lines = [
    `# README test: ${state.repoUrl}`,
    "",
    `Mode: ${state.mode} · ${state.cmdCount} command(s) run · ${state.llmRetries} LLM retr(ies)`,
    "",
    "| Step | Command | Status | Exit | Note |",
    "| --- | --- | --- | --- | --- |",
    ...state.results.map(
      (r) =>
        `| ${r.stepId} | \`${oneLine(r.command, 100)}\` | ${r.status} | ${r.exitCode ?? "-"} | ${r.reason ?? ""} |`,
    ),
    "",
    "(Report stub — phases 6–8 replace this.)",
  ];
  log("report", `built stub report with ${state.results.length} result(s)`);
  return { reportMarkdown: lines.join("\n") };
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

type NextNode = "run_step" | "diagnose" | "ci_crosscheck";

/**
 * pass -> next step, fail -> diagnose, done -> ci_crosscheck.
 *
 * Exhaustion is checked first, and that order is load-bearing: a failure on the
 * last step leaves `idx === steps.length` with a `fail` at the tail of
 * `results`, and looking at the status first would send the graph back to
 * diagnose with nothing left to diagnose — round and round until the recursion
 * limit. Phase 6's diagnose advances `idx` too, so the same guard covers it.
 */
function routeAfterRun(state: RunStateType): NextNode {
  if (state.idx >= state.steps.length) return "ci_crosscheck";
  if (state.results.at(-1)?.status === "fail") return "diagnose";
  return "run_step";
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

export function buildGraph() {
  return new StateGraph(RunState)
    .addNode("setup", setup)
    .addNode("discover_docs", discover_docs)
    .addNode("extract_steps", extract_steps)
    .addNode("run_step", run_step)
    .addNode("diagnose", diagnose)
    .addNode("ci_crosscheck", ci_crosscheck)
    .addNode("report", report)
    .addEdge(START, "setup")
    .addEdge("setup", "discover_docs")
    .addEdge("discover_docs", "extract_steps")
    .addEdge("extract_steps", "run_step")
    .addConditionalEdges("run_step", routeAfterRun, ["run_step", "diagnose", "ci_crosscheck"])
    .addEdge("diagnose", "run_step")
    .addEdge("ci_crosscheck", "report")
    .addEdge("report", END)
    .compile();
}
