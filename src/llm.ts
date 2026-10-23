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

const build = (model: string) =>
  new ChatGroq({
    model,
    temperature: 0,
    maxRetries: 3,
  });

/** Small model: classification, CI mismatch summaries. */
export const fast = build(FAST_MODEL);

/** 70B+ class model: step extraction, failure diagnosis, patch wording. */
export const strong = build(STRONG_MODEL);

export const models = { fast, strong } as const;
export type ModelTier = keyof typeof models;
