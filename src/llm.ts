import "./env.js";
import { ChatGroq } from "@langchain/groq";

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
  return new ChatGroq({ model, temperature: 0, maxRetries: 3 });
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
