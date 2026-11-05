/**
 * Phase 9: run the graph, and resume it after a rate limit.
 *
 * Split out of the CLI so the retry policy is testable without a container, an
 * API key or a five-second wait. The one rule that matters: a retry invokes with
 * `null`, not with the original input. LangGraph treats an input object as a new
 * run and starts at START again — which re-cloned the repo and re-ran every step
 * that had already passed.
 */
import { backoffDelay, isRateLimitError } from "./llm.js";
import { warnLine } from "./log.js";
import type { RunInput } from "./state.js";

export interface ResumeOptions {
  /** Total attempts after the first failure. */
  maxAttempts?: number;
  delayFor?: (attempt: number) => number;
  sleep?: (ms: number) => Promise<void>;
}

const defaultSleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

const message = (err: unknown): string => (err instanceof Error ? err.message : String(err));

/**
 * Invoke once, then resume on a rate limit up to `maxAttempts` times.
 *
 * Anything that is not a rate limit propagates immediately: a clone failure or a
 * missing Docker daemon is not improved by waiting.
 */
export async function invokeWithResume<T>(
  invoke: (input: RunInput | null) => Promise<T>,
  input: RunInput,
  opts: ResumeOptions = {},
): Promise<T> {
  const maxAttempts = opts.maxAttempts ?? 2;
  const delayFor = opts.delayFor ?? ((attempt: number) => backoffDelay(attempt, 5_000, 60_000));
  const sleep = opts.sleep ?? defaultSleep;

  for (let attempt = 0; ; attempt++) {
    try {
      return await invoke(attempt === 0 ? input : null);
    } catch (err) {
      if (!isRateLimitError(err) || attempt >= maxAttempts) throw err;

      const delay = delayFor(attempt + 1);
      warnLine(
        `[cli] rate limit stopped the run (${message(err)}) — resuming from the last checkpoint in ` +
          `${(delay / 1000).toFixed(1)}s (attempt ${attempt + 1}/${maxAttempts})`,
      );
      await sleep(delay);
    }
  }
}
