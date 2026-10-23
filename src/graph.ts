import { END, START, StateGraph } from "@langchain/langgraph";
import { RunState, type RunStateType, type RunStateUpdate, type Step, type StepResult } from "./state.js";
import { STUB_STEPS, stubExec, stubFix } from "./stub-exec.js";

/**
 * Phase 1: the full graph topology with stub node bodies.
 *
 * Every node here is a placeholder for a later phase (noted per node). What is
 * real is the wiring: START -> setup -> discover_docs -> extract_steps ->
 * (run_step <-> diagnose) -> ci_crosscheck -> report -> END, including the
 * conditional edges and the two counters later phases rely on.
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

/** Phase 3 stub: clone the repo and start the container. */
async function setup(state: RunStateType): Promise<RunStateUpdate> {
  log("setup", `would clone ${state.repoUrl} and start a container (phase 3)`);
  return {};
}

/** Phase 3 stub: find README / CONTRIBUTING / INSTALL / docs/*setup*. */
async function discover_docs(state: RunStateType): Promise<RunStateUpdate> {
  log("discover_docs", `would scan the clone for docs (phase 3), mode=${state.mode}`);
  return {
    docs: [{ path: "README.md", title: "Stub README", content: "[stub doc]" }],
  };
}

/** Phase 4 stub: label fenced code blocks into ordered steps. */
async function extract_steps(state: RunStateType): Promise<RunStateUpdate> {
  const steps: Step[] = STUB_STEPS.map((s) => ({
    ...s,
    runnable: true,
    source: "README.md",
  }));
  log("extract_steps", `would label ${state.docs.length} doc(s) (phase 4); using ${steps.length} stub steps`);
  return { steps, idx: 0 };
}

/** Phase 5 stub: run steps[idx] and record the outcome. */
async function run_step(state: RunStateType): Promise<RunStateUpdate> {
  const step = state.steps[state.idx];
  if (!step) {
    log("run_step", `no step at idx=${state.idx}, nothing to run`);
    return {};
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
