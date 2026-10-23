import { Annotation } from "@langchain/langgraph";

/** Who a documented step is meant for. Tagged in extract_steps (Phase 4). */
export type Mode = "user" | "contributor";

/**
 * Outcome of running a step.
 * - pass: exit code 0 on the documented command.
 * - workaround: the documented command failed, but the fix proposed by diagnose
 *   (Phase 6) succeeded.
 * - fail: failed, and no working fix was found.
 * - skip: not runnable (interactive prompt, long-running server, ...).
 */
export type StepStatus = "pass" | "workaround" | "fail" | "skip";

/** A doc file discovered in the cloned repo. */
export interface DocPage {
  /** Repo-relative path, e.g. "README.md" or "docs/setup.md". */
  path: string;
  /** Leading `#` heading of the page, if any. */
  title?: string;
  content: string;
}

/** One runnable command taken from a doc code block. */
export interface Step {
  /** Stable id used in the report, e.g. "s3". */
  id: string;
  command: string;
  mode: Mode;
  /** False for blocks that are clearly illustrative (e.g. `pip install foo` placeholders). */
  runnable: boolean;
  /** Repo-relative path of the doc the block came from. */
  source: string;
  /** Nearest heading above the block — context for the LLM labeller. */
  heading?: string;
}

/** The record of one step execution in the container. */
export interface StepResult {
  stepId: string;
  /** The command actually executed — differs from the step's command when diagnosed. */
  command: string;
  status: StepStatus;
  exitCode: number | null;
  /** Truncated to the last ~2 KB before it is stored (Phase 2). */
  stdout: string;
  stderr: string;
  durationMs: number;
  /** Why this status was assigned: LLM's diagnosis reason, or a skip explanation. */
  reason?: string;
}

/** A place where the documented commands and CI disagree (Phase 7). */
export interface CiFinding {
  kind: "missing_in_ci" | "missing_in_docs" | "mismatch";
  command: string;
  /** Where CI was found lacking/inconsistent, e.g. ".github/workflows/ci.yml". */
  ciSource?: string;
  note: string;
}

/**
 * Graph state.
 *
 * `results` is append-only (reducer), so a resume after a rate-limit stop keeps
 * the work already done. Everything else is last-write-wins, which is what
 * every node body expects.
 */
export const RunState = Annotation.Root({
  /** Clone URL passed on the command line. */
  repoUrl: Annotation<string>,

  /** Filter applied to steps: user-facing vs contributor docs. */
  mode: Annotation<Mode>,

  /** Scratch directory holding the clone. */
  workdir: Annotation<string | undefined>,

  /** Running container id, once setup has started one. */
  containerId: Annotation<string | undefined>,

  /** Docs discovered by discover_docs. */
  docs: Annotation<DocPage[]>,

  /** Ordered commands to run. */
  steps: Annotation<Step[]>,

  /** Index of the step run_step is currently on. */
  idx: Annotation<number>,

  /** Append-only log of step executions. */
  results: Annotation<StepResult[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),

  /** Commands executed so far — checked against the Phase 9 global cap. */
  cmdCount: Annotation<number>,

  /** LLM retries spent so far; drives the Phase 4 one-retry fallback. */
  llmRetries: Annotation<number>,

  /** CI vs docs discrepancies (Phase 7). */
  ciFindings: Annotation<CiFinding[]>,

  /**
   * Final markdown report (Phase 8).
   *
   * Not called `report`: a state channel may not share a name with a node, and
   * `report` is the node name from the workflow spec.
   */
  reportMarkdown: Annotation<string>,
});

/** The state object nodes receive. */
export type RunStateType = typeof RunState.State;

/** A partial update any node may return. */
export type RunStateUpdate = typeof RunState.Update;

/** Input the CLI hands to `graph.invoke`. */
export interface RunInput {
  repoUrl: string;
  mode: Mode;
}

/**
 * Defaults for fields a node hasn't written yet. Applied once at invoke time so
 * the graph works without every node returning every field.
 */
export const initialValues = {
  docs: [],
  steps: [],
  idx: 0,
  results: [],
  cmdCount: 0,
  llmRetries: 0,
  ciFindings: [],
  reportMarkdown: "",
} satisfies Omit<RunStateType, "repoUrl" | "mode" | "workdir" | "containerId">;
