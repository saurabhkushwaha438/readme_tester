Phase 0: Setup (½ day)

npm i @langchain/langgraph @langchain/core @langchain/groq dockerode zod
.env: GROQ_API_KEY
src/llm.ts exports two models: fast (small model, for classification) and strong (70B-class, for extract and diagnose). Both use temperature: 0 and maxRetries: 3.

Phase 1: State + stub graph (½ day)
Same as before, plus llmRetries and cmdCount fields. Get START → END running with stubs and no LLM.

Phase 2: Docker tools (1 day)
Same as before: startContainer, exec with a timeout, readFile, listFiles, destroy. Add output truncation: return only the last ~2 KB of stdout/stderr to the LLM. This saves Groq tokens, so it isn’t optional.

Phase 3: setup + discover_docs (½ day)
No LLM. Find README, CONTRIBUTING, INSTALL, and docs/*setup*, and follow links.

Phase 4: extract_steps, hybrid (1.5 days)

Deterministic first: regex out fenced code blocks (bash, sh, shell) from each doc, with the heading above each block as context.
LLM second (strong): classify and order the blocks into {id, command, mode: user|contributor, runnable: bool} via withStructuredOutput(zodSchema). The LLM never invents commands, it only labels the ones that exist.
Validate: a zod parse, and one retry node if it fails. If the retry fails too, fall back to running all blocks in README order.
Chunk long docs so each call stays well under the model’s context and token limits.

Phase 5: run_step + conditional edges (1 day)
Run steps[idx] in the container and record the result. Route on the outcome: pass → next step, fail → diagnose, done → ci_crosscheck. Set recursionLimit to 100 or higher, since the default of 25 is too low for a loop. This node needs no LLM. 
tessl

Phase 6: diagnose (1 day)
Input is the failed command plus the truncated stderr. The strong model proposes one fix as a structured {command, reason}, and you run it. If it works, mark the step workaround. If it fails, mark it fail. Add a small denylist (rm -rf /, curl | sh from unknown hosts) before running any LLM-proposed command.

Phase 7: ci_crosscheck (½ day)
Read .github/workflows/*.yml, Makefile, and package.json scripts. A deterministic diff of the commands is first, and the fast model only summarizes the mismatches.

Phase 8: report + CLI (1 day)
Markdown report of steps, statuses, error excerpts, and suggested README patch. Template it in code, and use the LLM only for the patch wording. CLI: readme-test <url> --mode contributor.

Phase 9: Rate-limit + guardrails (1 day) (new)

Wrap LLM calls with exponential backoff on 429s, and show a “waiting on rate limit” log line.
Global caps: ~30 commands and ~N LLM calls per run.
A MemorySaver checkpointer so a rate-limited run can resume instead of restarting.
Always destroy the container in a finally.

Phase 10: Real repos + PRs (ongoing)
Run on 10 repos from your target list. Log false positives, tune prompts, and open PRs for real doc bugs.