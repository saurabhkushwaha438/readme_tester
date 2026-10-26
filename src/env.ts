/**
 * `.env` loading (Phase 0 specified it, nothing loaded it until now).
 *
 * Side effect on import: Node evaluates an importing module's imports before
 * its own body, so a module that does `import "./env.js";` at the top has the
 * values available by the time its `const`s are initialised — which matters for
 * `DEFAULT_IMAGE` in docker.ts and the model ids in llm.ts.
 *
 * Variables already present in the real environment win over the file, so
 * `env -u GROQ_API_KEY npm start` still means "no key".
 */
import { existsSync } from "node:fs";

let loaded = false;

export function loadEnv(): void {
  if (loaded) return;
  loaded = true;
  // Absent .env is normal: CI and `npm start -- <url>` with real env vars work.
  if (!existsSync(".env")) return;
  try {
    process.loadEnvFile(".env");
  } catch (err) {
    console.warn(`[env] ignoring unreadable .env: ${err instanceof Error ? err.message : err}`);
  }
}

loadEnv();
