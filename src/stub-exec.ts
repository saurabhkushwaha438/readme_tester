/**
 * Phase 1 stub executor.
 *
 * Phase 2 replaces this with real Docker exec. Until then, "running" a command
 * is simulated deterministically so the graph's loop and failure branch are
 * exercised without a daemon: every command succeeds except the ones listed in
 * STUB_FAILURES, which let the run_step -> diagnose path be tested.
 */
const STUB_FAILURES = [/process\.exit\(1\)/];

/** Commands the stub extractor pretends it found in the README. */
export const STUB_STEPS = [
  { id: "s1", command: "npm install", mode: "user" as const, heading: "Install" },
  {
    id: "s2",
    command: `node -e "process.exit(1)"`,
    mode: "user" as const,
    heading: "Install",
  },
  { id: "s3", command: "npm test", mode: "contributor" as const, heading: "Testing" },
];

export interface StubExecResult {
  exitCode: number;
  stdout: string;
  stderr: string;
}

/**
 * What the stub stands in for the strong model's one proposed fix (Phase 6).
 * A failing command gets a plausible corrected version.
 */
export function stubFix(command: string): string {
  if (/process\.exit\(1\)/.test(command)) {
    return command.replace("process.exit(1)", "process.exit(0)");
  }
  return `${command} --fix`;
}

export async function stubExec(command: string): Promise<StubExecResult> {
  const shouldFail = STUB_FAILURES.some((re) => re.test(command));
  return shouldFail
    ? { exitCode: 1, stdout: "", stderr: `[stub] simulated failure for: ${command}` }
    : { exitCode: 0, stdout: `[stub] ok: ${command}`, stderr: "" };
}
