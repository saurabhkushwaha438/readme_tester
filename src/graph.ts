import { existsSync } from "node:fs";
import { join, posix as pathPosix } from "node:path";
import {
  END,
  MemorySaver,
  START,
  StateGraph,
  type BaseCheckpointSaver,
} from "@langchain/langgraph";
import {
  CONTAINER_WORKDIR,
  DEFAULT_IMAGE,
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
import { diffDocsAgainstCi, readCiConfig, summarizeMismatches } from "./ci.js";
import { discoverDocs } from "./docs.js";
import { extractSteps, MAX_STEPS } from "./extract.js";
import { llmCallCap, llmCallCount } from "./llm.js";
import { infoLine } from "./log.js";
import { CLONE_DIR_NAME, cloneRepo } from "./repo.js";
import { buildReport, finalResults, planPatches, suggestPatchWording } from "./report.js";
import {
  BUDGET_SKIP_REASON,
  RunState,
  type RunStateType,
  type RunStateUpdate,
  type StepResult,
  type StepStatus,
} from "./state.js";

/**
 * The graph topology, with every node real through Phase 8.
 *
 * setup (2/3), discover_docs (3), extract_steps (4), run_step (5), diagnose (6),
 * ci_crosscheck (7), report (8). The heavy lifting lives in the matching
 * modules; these nodes adapt state and log.
 */

/**
 * Phase 5 makes this essential: the loop costs up to 3 supersteps per step
 * (run_step -> diagnose -> run_step), so the default of 25 is already too low
 * for a handful of steps. The spec asks for 100 or higher; 250 leaves headroom
 * for the Phase 9 cap of ~30 commands (~96 supersteps worst case) without
 * touching the limit again.
 */
export const RECURSION_LIMIT = 250;

/**
 * Phase 9: commands one run may execute, diagnosis and re-checks included.
 *
 * A documented setup that needs more than 30 commands is a repo this tool should
 * stop testing, not a budget to raise mid-run. Steps past the cap are recorded as
 * `skip` with the reason, so the report accounts for every step instead of
 * quietly shortening the list.
 */
export const MAX_COMMANDS = Number(process.env.README_TEST_MAX_COMMANDS ?? 30);

const NOT_RUNNABLE = "not runnable unattended (interactive or long-running)";

const log = (node: string, msg: string) => infoLine(`[${node}] ${msg}`);

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
 * Root files, two levels inside a docs folder, and `.github/` — the last one is
 * where GitHub's own contributing-guide convention puts CONTRIBUTING.md, which
 * is otherwise invisible to a root-only listing.
 */
async function listRepoEntries(containerId: string): Promise<FileEntry[]> {
  const root = await listFiles(containerId, ".", { maxDepth: 1 });
  const docDirs = root
    .filter(
      (entry) =>
        entry.type === "dir" &&
        (/^(docs?|documentation)$/i.test(entry.path) || entry.path === ".github"),
    )
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

/** The package's own name, if it has one — consumer installs are filtered by it. */
async function repoPackageName(containerId: string): Promise<string | undefined> {
  try {
    const parsed = JSON.parse(await readFile(containerId, "package.json")) as { name?: unknown };
    return typeof parsed.name === "string" && parsed.name !== "" ? parsed.name : undefined;
  } catch {
    // Absent, unreadable or not JSON: nothing to filter by, which is fine.
    return undefined;
  }
}

/** Phase 4: deterministic block extraction, strong-model labelling. */
async function extract_steps(state: RunStateType): Promise<RunStateUpdate> {
  const ownPackage = await repoPackageName(requireContainer(state));
  const result = await extractSteps(state.docs, {
    mode: state.mode,
    maxSteps: MAX_STEPS,
    repoUrl: state.repoUrl,
    ...(ownPackage ? { ownPackage } : {}),
  });
  const runnable = result.steps.filter((step) => step.runnable).length;

  log(
    "extract_steps",
    result.steps.length > 0
      ? `${result.steps.length} step(s) from ${result.blocksFound} block(s): ` +
          `${runnable} runnable, ${result.steps.length - runnable} not (skipped), ` +
          `${result.usedFallback ? "doc order (LLM fell back)" : "LLM ordered"}`
      : result.blocksFound > 0
        ? `${result.blocksFound} shell block(s), but no step survived labelling; nothing to run`
        : `no shell blocks in ${state.docs.length} doc(s); nothing to run`,
  );

  return {
    steps: result.steps,
    // The CI diff needs every documented command, including the contributor
    // ones the mode filter kept out of `steps`.
    docCommands: result.documented,
    idx: 0,
    llmRetries: state.llmRetries + result.llmRetries,
  };
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
 * Carry a successful plain `cd` into the next step.
 *
 * Docker exec starts every command in the same directory. The docs, however,
 * write setup as a sequence (`mkdir my-app`, `cd my-app`, `npm i`), so without
 * this the install runs in the repository root and fails for the wrong reason.
 * Only simple, standalone `cd` segments are followed; a `cd` inside a subshell
 * or command substitution belongs to that shell and is deliberately ignored.
 */
export function nextCwd(command: string, current: string = CONTAINER_WORKDIR): string {
  let cwd = current;

  for (const raw of command.split(/(?:\r?\n|&&|\|\||;)/)) {
    const match = /^\s*cd\s+(.+?)\s*$/.exec(raw);
    if (!match) continue;

    let target = match[1]!.trim();
    if (
      (target.startsWith('"') && target.endsWith('"')) ||
      (target.startsWith("'") && target.endsWith("'"))
    ) {
      target = target.slice(1, -1);
    }
    target = target.replace(/\\(.)/g, "$1");

    if (target === "" || target === "-" || target.startsWith("$")) continue;
    if (target === "~") target = "/root";
    else if (target.startsWith("~/")) target = `/root/${target.slice(2)}`;

    cwd = pathPosix.normalize(
      target.startsWith("/") ? target : pathPosix.join(cwd, target),
    );
  }

  return cwd;
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

  // A `cd` belongs to the recipe it appears in. Keep it while the reader is in
  // the same doc section, but start a new section from the repository root so
  // one guide's `cd my-app` does not send the next guide's install into it.
  const previous = state.steps[state.idx - 1];
  const sameSection =
    previous !== undefined &&
    previous.source === step.source &&
    previous.heading === step.heading;
  const workdir = sameSection ? state.cwd : CONTAINER_WORKDIR;

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
      reason: step.skipReason ?? NOT_RUNNABLE,
    };
    log("run_step", `${step.id} -> skip (${step.skipReason ?? "not runnable"})`);
    return { results: [skipped], idx: state.idx + 1, cwd: workdir };
  }

  // Phase 9: stop executing once the budget is gone. Every remaining step still
  // gets a row, so the report never looks like the docs ended early.
  if (state.cmdCount >= MAX_COMMANDS) {
    const remaining = state.steps.slice(state.idx);
    log(
      "run_step",
      `command budget reached (${state.cmdCount}/${MAX_COMMANDS}); skipping ${remaining.length} step(s)`,
    );
    const skipped = remaining.map<StepResult>((pending) => ({
      stepId: pending.id,
      command: pending.command,
      status: "skip",
      exitCode: null,
      stdout: "",
      stderr: "",
      durationMs: 0,
      reason: pending.runnable
        ? `${BUDGET_SKIP_REASON} (limit ${MAX_COMMANDS}); not run`
        : NOT_RUNNABLE,
    }));
    return { results: skipped, idx: state.steps.length };
  }

  const containerId = requireContainer(state);
  log("run_step", `${step.id} $ ${oneLine(step.command)} (idx=${state.idx})`);

  let out: ExecResult;
  try {
    out = await exec(containerId, step.command, { workdir });
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
    return { results: [broken], cmdCount: state.cmdCount + 1, cwd: workdir };
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
    // A successful command that ended in a plain `cd` moves the next step too.
    cwd: status === "pass" ? nextCwd(step.command, workdir) : workdir,
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

  // A diagnosis costs up to two commands (the fix, then the documented command
  // again to verify it). Without room for both, a "workaround" would be a claim
  // nobody checked, so leave the failure as recorded.
  const remaining = MAX_COMMANDS - state.cmdCount;
  if (remaining < 2) {
    log(
      "diagnose",
      `skipping diagnosis of ${failed.id}: command budget reached (${state.cmdCount}/${MAX_COMMANDS})`,
    );
    return { idx: state.idx + 1 };
  }

  const containerId = requireContainer(state);
  log("diagnose", `${failed.id} failed (exit ${failure.exitCode ?? "?"}); asking for one fix`);

  const outcome = await diagnoseStep({
    containerId,
    step: failed,
    failure,
    image: DEFAULT_IMAGE,
    workdir: state.cwd,
  });
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
    ...(outcome.attemptedFix ? { attemptedFix: outcome.attemptedFix } : {}),
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

/**
 * Phase 7: diff the documented commands against what CI runs.
 *
 * Nothing here executes a command: the CI files are read and parsed, the diff is
 * computed in code, and the fast model only writes the summary sentence. A repo
 * with no CI config is a normal result, not a failure.
 */
async function ci_crosscheck(state: RunStateType): Promise<RunStateUpdate> {
  const containerId = requireContainer(state);

  // Phase 10: with no documented commands there is no diff to compute, and
  // listing every CI command as "missing in docs" is noise (the p-limit/koa
  // pilot runs produced four such findings for readmes that never claim a
  // contributor setup). Skip the file reads too.
  if (state.docCommands.length === 0) {
    log("ci_crosscheck", "no documented commands; skipping the CI comparison");
    return { ciFindings: [], ciSummary: "" };
  }

  const ci = await readCiConfig({
    list: (dir) => listFiles(containerId, dir, { maxDepth: 1 }),
    read: (path) => readFile(containerId, path),
  });

  if (ci.sources.length === 0) {
    log("ci_crosscheck", "no workflows, Makefile or package.json found; nothing to compare");
    return { ciFindings: [], ciSummary: "" };
  }

  const findings = diffDocsAgainstCi(state.docCommands, ci);
  const counts = findings.reduce<Record<string, number>>((acc, finding) => {
    acc[finding.kind] = (acc[finding.kind] ?? 0) + 1;
    return acc;
  }, {});

  let summary = "";
  let llmRetries = 0;
  if (findings.length > 0) {
    const result = await summarizeMismatches({
      findings,
      docCommands: state.docCommands,
      commands: ci.commands,
    });
    summary = result.summary;
    llmRetries = result.llmRetries;
  }

  log(
    "ci_crosscheck",
    `${findings.length} finding(s) from ${ci.sources.length} source(s) ` +
      `(${ci.sources.join(", ")})` +
      (findings.length > 0
        ? `: ${Object.entries(counts)
            .map(([kind, count]) => `${count} ${kind}`)
            .join(", ")}`
        : ": docs and CI agree"),
  );
  if (summary) log("ci_crosscheck", summary);

  return { ciFindings: findings, ciSummary: summary, llmRetries: state.llmRetries + llmRetries };
}

/**
 * Phase 8: templated markdown plus model-written patch wording.
 *
 * The patches are decided in `planPatches` — a `workaround` step is the only
 * thing that earns a diff — and the model only supplies the sentences under
 * each one. Everything else in the report is code, so a failed model call costs
 * prose rather than the run's findings.
 */
async function report(state: RunStateType): Promise<RunStateUpdate> {
  const patches = planPatches(state.steps, state.results);
  const { wording, llmRetries } = await suggestPatchWording(patches);

  const markdown = buildReport({
    repoUrl: state.repoUrl,
    mode: state.mode,
    steps: state.steps,
    results: state.results,
    ciFindings: state.ciFindings,
    ciSummary: state.ciSummary,
    cmdCount: state.cmdCount,
    llmCalls: llmCallCount(),
    llmCallBudget: llmCallCap(),
    commandBudget: MAX_COMMANDS,
    llmRetries: state.llmRetries + llmRetries,
    wording,
  });

  const results = finalResults(state.results);
  const failed = results.filter((result) => result.status === "fail").length;
  log(
    "report",
    `${results.length} step(s), ${failed} failure(s), ${patches.length} patch(es)` +
      (llmRetries > 0 ? ", wording fallback" : ""),
  );
  return { reportMarkdown: markdown, llmRetries: state.llmRetries + llmRetries };
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

/**
 * After a diagnosis, only the exhaustion check applies.
 *
 * Reusing `routeAfterRun` here would look at the tail result — still the `fail`
 * the diagnosis just wrote — and send the graph back to `diagnose`, which would
 * then judge the *next* step on the previous step's failure. Routing on `idx`
 * alone is both correct and one node visit shorter when the last step failed.
 */
function routeAfterDiagnose(state: RunStateType): NextNode {
  return state.idx >= state.steps.length ? "ci_crosscheck" : "run_step";
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

export interface GraphOptions {
  /**
   * Defaults to a fresh MemorySaver.
   *
   * MemorySaver is in-process state, which is enough for the Phase 9 case:
   * a rate-limited invoke throws, the CLI waits and invokes the same thread
   * again, and completed nodes are not re-run (the container and clone are
   * reused, and finished steps keep their rows). It does not survive a process
   * restart by design — persisting checkpoints is a later phase's problem.
   */
  checkpointer?: BaseCheckpointSaver;
}

export function buildGraph(opts: GraphOptions = {}) {
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
    .addConditionalEdges("diagnose", routeAfterDiagnose, ["run_step", "ci_crosscheck"])
    .addEdge("ci_crosscheck", "report")
    .addEdge("report", END)
    .compile({ checkpointer: opts.checkpointer ?? new MemorySaver() });
}
