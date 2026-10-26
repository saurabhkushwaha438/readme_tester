import { existsSync } from "node:fs";
import { join } from "node:path";
import { END, START, StateGraph } from "@langchain/langgraph";
import {
  isAlive,
  listFiles,
  pingDocker,
  readFile,
  startContainer,
  type FileEntry,
} from "./docker.js";
import { discoverDocs } from "./docs.js";
import { extractSteps, MAX_STEPS } from "./extract.js";
import { CLONE_DIR_NAME, cloneRepo } from "./repo.js";
import { RunState, type RunStateType, type RunStateUpdate, type StepResult } from "./state.js";
import { stubExec, stubFix } from "./stub-exec.js";

/**
 * The graph topology, with real nodes through Phase 4.
 *
 * Real: setup, discover_docs (Phase 3), extract_steps (Phase 4). Still stubs:
 * run_step/diagnose execution (Phases 5–6), ci_crosscheck (7), report (8).
 * The wiring and the routing are real either way.
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

/** Phase 5 stub: run steps[idx] and record the outcome. */
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

  log("run_step", `running ${step.id} ${step.command} (idx=${state.idx})`);
  const started = Date.now();
  const out = await stubExec(step.command);
  const result: StepResult = {
    stepId: step.id,
    command: step.command,
    status: out.exitCode === 0 ? "pass" : "fail",
    exitCode: out.exitCode,
    stdout: out.stdout,
    stderr: out.stderr,
    durationMs: Date.now() - started,
  };
  log("run_step", `${step.id} -> ${result.status} (exit ${out.exitCode})`);

  return {
    results: [result],
    // Advance only on success; a failure stays put for diagnose to retry.
    idx: out.exitCode === 0 ? state.idx + 1 : state.idx,
    cmdCount: state.cmdCount + 1,
  };
}

/** Phase 6 stub: propose one fix for the failed step and run it. */
async function diagnose(state: RunStateType): Promise<RunStateUpdate> {
  const failed = state.steps[state.idx];
  if (!failed) {
    log("diagnose", `no step at idx=${state.idx}, skipping`);
    return {};
  }

  // No LLM in phase 1: pretend the strong model fixed it.
  const fix = stubFix(failed.command);
  log("diagnose", `would ask the strong model for a fix (phase 6); trying ${fix}`);

  const started = Date.now();
  const out = await stubExec(fix);
  const result: StepResult = {
    stepId: failed.id,
    command: fix,
    status: out.exitCode === 0 ? "workaround" : "fail",
    exitCode: out.exitCode,
    stdout: out.stdout,
    stderr: out.stderr,
    durationMs: Date.now() - started,
    reason:
      out.exitCode === 0
        ? `[stub] proposed fix \`${fix}\` succeeded`
        : "[stub] proposed fix did not work",
  };
  log("diagnose", `${failed.id} -> ${result.status}`);

  // Always advance: leaving idx put after a failed fix would loop forever.
  return { results: [result], idx: state.idx + 1, cmdCount: state.cmdCount + 1 };
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
    "| Step | Command | Status | Exit |",
    "| --- | --- | --- | --- |",
    ...state.results.map(
      (r) => `| ${r.stepId} | \`${r.command}\` | ${r.status} | ${r.exitCode ?? "-"} |`,
    ),
    "",
    "(Phase 1 stub report — phases 6–8 replace this.)",
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
 * Reads the result the current run_step just appended.
 */
function routeAfterRun(state: RunStateType): NextNode {
  const last = state.results.at(-1);
  if (last?.status === "fail") return "diagnose";
  if (state.idx < state.steps.length) return "run_step";
  return "ci_crosscheck";
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
