import "./env.js";
import { ChatGroq } from "@langchain/groq";
import { warnLine } from "./log.js";

/**
 * Groq model IDs.
 *
 * The original plan named a "70B-class" model, but Groq shut down both
 * `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` on 2026-08-16. The
 * production replacements are the GPT-OSS pair, which also support JSON
 * Schema mode — needed for `withStructuredOutput` in extract_steps/diagnose.
 *
 * Overridable so the next deprecation is a config change, not a code change.
 */
export const FAST_MODEL = process.env.GROQ_FAST_MODEL ?? "openai/gpt-oss-20b";
export const STRONG_MODEL = process.env.GROQ_STRONG_MODEL ?? "openai/gpt-oss-120b";

/**
 * Models are built lazily.
 *
 * Constructing ChatGroq without GROQ_API_KEY throws, and phases 1–5 are
 * supposed to run without an LLM in the path — eager construction would make
 * `import "./graph.js"` fail on a machine with no key at all. Lazy getters mean
 * the error arrives when a node actually needs a model, with a message that
 * says what to do.
 */
function make(model: string): ChatGroq {
  if (!process.env.GROQ_API_KEY) {
    throw new Error(
      `GROQ_API_KEY is not set, but ${model} is needed for this run. ` +
        'Add GROQ_API_KEY=... to .env (or export it) — see phases.md Phase 0.',
    );
  }
  // Retries are ours from here on (Phase 9): the SDK's internal backoff is
  // invisible to the log and to the call budget, which is exactly what a
  // rate-limited run needs to see.
  return new ChatGroq({ model, temperature: 0, maxRetries: 0 });
}

let fastModel: ChatGroq | undefined;
let strongModel: ChatGroq | undefined;

/** Small model: classification, CI mismatch summaries (Phase 7). */
export function getFast(): ChatGroq {
  fastModel ??= make(FAST_MODEL);
  return fastModel;
}

/** 70B+ class model: step extraction (Phase 4), failure diagnosis (Phase 6). */
export function getStrong(): ChatGroq {
  strongModel ??= make(STRONG_MODEL);
  return strongModel;
}

export const modelFor = { fast: getFast, strong: getStrong } as const;
export type ModelTier = keyof typeof modelFor;

// ---------------------------------------------------------------------------
// Phase 9: rate limits, backoff, and a call budget
// ---------------------------------------------------------------------------

/**
 * Hard ceiling on model calls for one run, retries included.
 *
 * A run that hits this keeps going: every caller treats an LLM failure as
 * fail-soft (heuristic labels, no diagnosis, no summary), so the report still
 * comes out — it just says less. The alternative, spending an unbounded number
 * of calls on a repo whose docs are broken, is how a free tier disappears.
 */
export const MAX_LLM_CALLS = Number(process.env.README_TEST_MAX_LLM_CALLS ?? 60);

/** Attempts per call, so one rate-limit burst does not end a run's LLM phase. */
const MAX_ATTEMPTS = Number(process.env.README_TEST_LLM_ATTEMPTS ?? 5);

const BASE_DELAY_MS = Number(process.env.README_TEST_LLM_BASE_DELAY_MS ?? 2_000);
const MAX_DELAY_MS = Number(process.env.README_TEST_LLM_MAX_DELAY_MS ?? 60_000);

let calls = 0;
let cap = MAX_LLM_CALLS;

/** Called once at the start of a run. */
export function resetLlmBudget(limit = MAX_LLM_CALLS): void {
  calls = 0;
  cap = limit;
}

export const llmCallCount = (): number => calls;
export const llmCallCap = (): number => cap;

/** Thrown instead of calling the model once the run is out of budget. */
export class LlmBudgetExhausted extends Error {
  constructor() {
    super(`LLM call budget exhausted (${cap}/${cap}) — see README_TEST_MAX_LLM_CALLS`);
    this.name = "LlmBudgetExhausted";
  }
}

/** A provider failure worth retrying: a rate limit, a 5xx, a dropped socket. */
export function isRateLimitError(err: unknown): boolean {
  const seen = new Set<unknown>();
  const queue: unknown[] = [err];

  while (queue.length > 0) {
    const current = queue.shift();
    if (current === null || typeof current !== "object" || seen.has(current)) continue;
    seen.add(current);

    const record = current as Record<string, unknown>;
    const status = record["status"] ?? record["statusCode"] ?? (record["response"] as Record<string, unknown> | undefined)?.["status"];
    if (status === 429) return true;

    const code = String(record["code"] ?? "");
    if (/^(rate_limit_exceeded|rate_limit|too_many_requests|insufficient_quota)$/i.test(code)) return true;

    const text = current instanceof Error ? current.message : String(record["message"] ?? "");
    if (/\b429\b|rate[ _-]?limit|too many requests|tokens per (minute|day)|\bTPM\b|\bRPM\b|quota/i.test(text)) {
      return true;
    }

    // LangChain wraps SDK errors in `cause`/`error`/`response.data`.
    queue.push(record["cause"], record["error"], record["response"]);
  }

  return false;
}

function isTransient(err: unknown): boolean {
  if (isRateLimitError(err)) return true;

  const record = err as Record<string, unknown> | null;
  const status = Number(record?.["status"] ?? record?.["statusCode"] ?? 0);
  if (status >= 500 && status < 600) return true;

  const code = String(record?.["code"] ?? "");
  return /^(ECONNRESET|ETIMEDOUT|EPIPE|EAI_AGAIN|UND_ERR_.*|ENETUNREACH|EHOSTUNREACH)$/.test(code);
}

/**
 * Exponential backoff with jitter, in milliseconds.
 *
 * Jitter is asymmetric (75–125%) so a burst of parallel callers does not
 * synchronise, and the whole thing is a pure function of the attempt number so
 * the tests can assert the schedule instead of waiting for it.
 */
export function backoffDelay(
  attempt: number,
  baseMs = BASE_DELAY_MS,
  maxMs = MAX_DELAY_MS,
  jitter = Math.random(),
): number {
  const raw = Math.min(maxMs, baseMs * 2 ** (attempt - 1));
  return Math.round(raw * (0.75 + jitter * 0.5));
}

export interface CallOptions {
  attempts?: number;
  baseMs?: number;
  maxMs?: number;
  sleep?: (ms: number) => Promise<void>;
  jitter?: () => number;
}

const defaultSleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Run one model call inside the budget, retrying rate limits and 5xxs.
 *
 * `what` only appears in log lines: `[llm] waiting on rate limit — extract_steps,
 * retry 2/5 in 3.6s`. That line is the point of the phase; a silent 40-second
 * pause looks like a hang.
 */
export async function callModel<T>(
  what: string,
  run: () => Promise<T>,
  opts: CallOptions = {},
): Promise<T> {
  const attempts = opts.attempts ?? MAX_ATTEMPTS;
  const baseMs = opts.baseMs ?? BASE_DELAY_MS;
  const maxMs = opts.maxMs ?? MAX_DELAY_MS;
  const sleep = opts.sleep ?? defaultSleep;
  const jitter = opts.jitter ?? Math.random;

  let lastError: unknown;

  for (let attempt = 1; attempt <= attempts; attempt++) {
    if (calls >= cap) {
      warnLine(
        `[llm] budget exhausted (${cap} ${cap === 1 ? "call" : "calls"}); skipping ${what}`,
      );
      throw new LlmBudgetExhausted();
    }
    calls++;

    try {
      return await run();
    } catch (err) {
      lastError = err;
      if (attempt >= attempts || !isTransient(err)) throw err;

      const delay = backoffDelay(attempt, baseMs, maxMs, jitter());
      const label = isRateLimitError(err) ? "waiting on rate limit" : "provider error";
      warnLine(
        `[llm] ${label} — ${what}, retry ${attempt + 1}/${attempts} in ${(delay / 1000).toFixed(1)}s`,
      );
      await sleep(delay);
    }
  }

  throw lastError;
}
