import type { ExtractUpdateType } from "@langchain/langgraph";
import { registry } from "@langchain/langgraph/zod";
import { z } from "zod";

/**
 * Graph state, declared as a Zod schema.
 *
 * LangGraph 1.x accepts a Zod object directly in `new StateGraph(...)`, so the
 * old `Annotation.Root` wrapper is gone. Two consequences worth knowing:
 *
 * - Channel defaults live in the schema (`register(registry, { default })`), so
 *   `invoke` only needs the real input fields.
 * - Key reducers are registered the same way (`register(registry, { reducer })`),
 *   which is also how `results` stays append-only.
 *
 * The data models below are Zod schemas first and TS types second (`z.infer`),
 * because Phase 4's `withStructuredOutput` needs the schema itself, not just
 * the type.
 */

/** Who a documented step is meant for. Tagged in extract_steps (Phase 4). */
export const ModeSchema = z.enum(["user", "contributor"]);
export type Mode = z.infer<typeof ModeSchema>;

/**
 * Outcome of running a step.
 * - pass: exit code 0 on the documented command.
 * - workaround: the documented command failed, but the fix proposed by diagnose
 *   (Phase 6) succeeded.
 * - fail: failed, and no working fix was found.
 * - skip: not runnable (interactive prompt, long-running server, ...).
 */
export const StepStatusSchema = z.enum(["pass", "workaround", "fail", "skip"]);
export type StepStatus = z.infer<typeof StepStatusSchema>;

/** A doc file discovered in the cloned repo. */
export const DocPageSchema = z.object({
  path: z.string().describe('Repo-relative path, e.g. "README.md" or "docs/setup.md".'),
  title: z.string().optional().describe("Leading `#` heading of the page, if any."),
  content: z.string(),
});
export type DocPage = z.infer<typeof DocPageSchema>;

/** One runnable command taken from a doc code block. */
export const StepSchema = z.object({
  id: z.string().describe('Stable id used in the report, e.g. "s3".'),
  command: z.string(),
  mode: ModeSchema.describe("Who the instructions are written for."),
  runnable: z
    .boolean()
    .describe(
      "False for blocks that are clearly illustrative (e.g. `pip install foo` placeholders).",
    ),
  source: z.string().describe("Repo-relative path of the doc the block came from."),
  heading: z.string().optional().describe("Nearest heading above the block, as context."),
  /**
   * Why a `runnable: false` step is not run, when a specific reason is known.
   *
   * The extractor only sets this for the self-clone case; everything else uses
   * the generic "not runnable unattended" wording.
   */
  skipReason: z.string().optional(),
});
export type Step = z.infer<typeof StepSchema>;

/**
 * A command a doc tells a reader to run, kept for the CI cross-check (Phase 7).
 *
 * Deliberately *not* mode-filtered: CI is contributor-facing, so comparing it
 * only against `--mode user` steps would report every test and lint command as
 * undocumented.
 */
export const DocumentedCommandSchema = z.object({
  command: z.string(),
  source: z.string(),
  heading: z.string().optional(),
});
export type DocumentedCommand = z.infer<typeof DocumentedCommandSchema>;

/** The record of one step execution in the container. */
export const StepResultSchema = z.object({
  stepId: z.string(),
  command: z.string().describe("The command actually executed — differs when diagnosed."),
  status: StepStatusSchema,
  exitCode: z.number().nullable(),
  stdout: z.string().describe("Truncated to the last ~2 KB by the Docker layer (phase 2)."),
  stderr: z.string(),
  durationMs: z.number(),
  reason: z.string().optional().describe("LLM diagnosis reason, or a skip explanation."),
  /**
   * A command the model proposed (Phase 6) that did not earn a `workaround`.
   *
   * Kept apart from `command` so the report can show what the docs say to run
   * *and* what was tried instead, without pretending one is the other.
   */
  attemptedFix: z.string().optional(),
});
export type StepResult = z.infer<typeof StepResultSchema>;

/** A place where the documented commands and CI disagree (Phase 7). */
export const CiFindingSchema = z.object({
  kind: z.enum(["missing_in_ci", "missing_in_docs", "mismatch"]),
  command: z.string(),
  ciSource: z.string().optional().describe('e.g. ".github/workflows/ci.yml".'),
  note: z.string(),
});
export type CiFinding = z.infer<typeof CiFindingSchema>;

/**
 * Prefix of the reason a step gets when the Phase 9 command budget ran out.
 *
 * Shared by the graph (which writes it) and the report (which counts it), so the
 * summary can tell a budget stop apart from an interactive command instead of
 * calling both "not runnable".
 */
export const BUDGET_SKIP_REASON = "command budget reached";

/**
 * Reason a step gets when the docs tell a reader to clone the repository under
 * test — which the run has already done.
 */
export const SELF_CLONE_SKIP_REASON =
  "already checked out — the run is testing a clone of this repo";

/** Append-only reducer shared by every list channel. */
const append = <T>(current: T[], update: T[]): T[] => current.concat(update);

/**
 * Graph state.
 *
 * `results` is append-only, so a resume after a rate-limit stop keeps the work
 * already done. Every other channel is last-write-wins.
 */
export const RunState = z.object({
  /** Clone URL passed on the command line. */
  repoUrl: z.string(),

  /** Filter applied to steps: user-facing vs contributor docs. */
  mode: ModeSchema,

  /** Scratch directory holding the clone. */
  workdir: z.string().optional(),

  /** Running container id, once setup has started one. */
  containerId: z.string().optional(),

  /**
   * Directory the next step should run in, relative to the container mount.
   *
   * Docker exec starts every command in the same workdir, so a documented
   * `cd my-app` would otherwise be lost before the next step. Phase 10 hit this
   * with Fastify's quick start: the docs create and enter `my-app`, then install
   * and run there. The graph carries the last successful `cd` forward.
   */
  cwd: z.string().register(registry, { default: () => "/workspace" }),

  /** Docs discovered by discover_docs. */
  docs: z.array(DocPageSchema).register(registry, { default: () => [] }),

  /** Ordered commands to run. */
  steps: z.array(StepSchema).register(registry, { default: () => [] }),

  /** Every command the docs contain, before the mode filter (Phase 7). */
  docCommands: z.array(DocumentedCommandSchema).register(registry, { default: () => [] }),

  /** Index of the step run_step is currently on. */
  idx: z.number().int().nonnegative().register(registry, { default: () => 0 }),

  /** Append-only log of step executions. */
  results: z.array(StepResultSchema).register(registry, {
    reducer: { schema: z.array(StepResultSchema), fn: append<StepResult> },
    default: () => [],
  }),

  /** Commands executed so far — checked against the Phase 9 global cap. */
  cmdCount: z.number().int().nonnegative().register(registry, { default: () => 0 }),

  /** LLM retries spent so far; drives the Phase 4 one-retry fallback. */
  llmRetries: z.number().int().nonnegative().register(registry, { default: () => 0 }),

  /** CI vs docs discrepancies (Phase 7). */
  ciFindings: z.array(CiFindingSchema).register(registry, {
    reducer: { schema: z.array(CiFindingSchema), fn: append<CiFinding> },
    default: () => [],
  }),

  /**
   * Prose summary of `ciFindings`, written by the fast model (Phase 7).
   *
   * Empty when nothing differed or the model was unavailable — the findings are
   * deterministic either way, so the report only loses the sentence.
   */
  ciSummary: z.string().register(registry, { default: () => "" }),

  /**
   * Final markdown report (Phase 8).
   *
   * Not called `report`: a state channel may not share a name with a node, and
   * `report` is the node name from the workflow spec.
   */
  reportMarkdown: z.string().register(registry, { default: () => "" }),
});

/** The state object nodes receive. */
export type RunStateType = z.infer<typeof RunState>;

/** A partial update any node may return (reducer channels accept their update type). */
export type RunStateUpdate = ExtractUpdateType<typeof RunState, RunStateType>;

/** Input the CLI hands to `graph.invoke`. */
export interface RunInput {
  repoUrl: string;
  mode: Mode;
}
