/**
 * Colour for the live progress log.
 *
 * The log is the tool's user interface: one bracketed line per thing that
 * happened. In a terminal those lines are worth reading at a glance, so the tag
 * carries its phase's colour and the tokens that matter — the command about to
 * run, the status of a step, a non-zero exit code — carry their own. Everything
 * else stays default, which is what keeps the log scannable instead of a
 * rainbow.
 *
 * The report on stdout is deliberately never coloured: it is markdown meant to
 * be pasted into an issue. Colour is also off whenever the stream is not a
 * terminal, so `npm start -- … | sed -n '/^# README/,$p'` sees the same bytes it
 * always did, and `NO_COLOR` is honoured.
 */

type Tone =
  | "red"
  | "green"
  | "yellow"
  | "blue"
  | "magenta"
  | "cyan"
  | "gray"
  | "bold"
  | "boldCyan";

/** Foreground/attribute codes, and the code that resets exactly that one. */
const TONE: Record<Tone, { on: string; off: string }> = {
  red: { on: "31", off: "39" },
  green: { on: "32", off: "39" },
  yellow: { on: "33", off: "39" },
  blue: { on: "34", off: "39" },
  magenta: { on: "35", off: "39" },
  cyan: { on: "36", off: "39" },
  gray: { on: "90", off: "39" },
  bold: { on: "1", off: "22" },
  boldCyan: { on: "1;36", off: "22;39" },
};

/** One hue per phase, so a logged run reads as a sequence of stages. */
const PHASE: Record<string, Tone> = {
  repo: "magenta",
  setup: "blue",
  docker: "blue",
  discover_docs: "cyan",
  extract_steps: "cyan",
  run_step: "cyan",
  diagnose: "magenta",
  ci_crosscheck: "blue",
  report: "green",
  docs: "gray",
  env: "gray",
  llm: "gray",
  cli: "magenta",
};

const STATUS: Record<string, Tone> = {
  pass: "green",
  fail: "red",
  skip: "yellow",
  workaround: "yellow",
};

function enabled(stream: NodeJS.WriteStream): boolean {
  // FORCE_COLOR is what npm sets for a script, and both spellings of "off" are
  // in the wild. NO_COLOR counts when it is merely present, empty included.
  const force = process.env.FORCE_COLOR;
  if (force === "0" || force === "false") return false;
  if (force) return true;
  if (process.env.NO_COLOR !== undefined) return false;
  return Boolean(stream.isTTY);
}

/** Wrap text in one ANSI attribute; a no-op when the stream has no colour. */
export function paint(
  text: string,
  tone: Tone,
  stream: NodeJS.WriteStream = process.stdout,
): string {
  if (!enabled(stream)) return text;
  const { on, off } = TONE[tone];
  return `\u001b[${on}m${text}\u001b[${off}m`;
}

/** `[tag] message` split into its tag and the rest. */
const TAGGED = /^\[([a-z_]+)\] /;

/**
 * A labelled progress line.
 *
 * All progress lines look like `[phase] …`, so the shape is parsed rather than
 * passed in: every existing call site keeps its message string, and the colours
 * come from the phase name and from the tokens the message already contains.
 */
export function infoLine(text: string): void {
  console.log(decorate(text, process.stdout));
}

/** A warning: the whole line yellow, on stderr, where it can be filtered. */
export function warnLine(text: string): void {
  console.warn(paint(text, "yellow", process.stderr));
}

/** A fatal error: red, on stderr. */
export function errorLine(text: string): void {
  console.error(paint(text, "red", process.stderr));
}

/** The two header lines that introduce a run. */
export function heading(text: string): void {
  console.log(paint(text, "boldCyan"));
}

/** Secondary output that should not compete with the progress log. */
export function muted(text: string): void {
  console.log(paint(text, "gray"));
}

function decorate(text: string, stream: NodeJS.WriteStream): string {
  if (!enabled(stream)) return text;

  const match = TAGGED.exec(text);
  if (!match) return text;

  const tag = match[0].trimEnd();
  const body = text.slice(match[0].length);
  return `${paint(tag, PHASE[match[1]!] ?? "gray")} ${detail(body)}`;
}

/**
 * The message after its tag.
 *
 * Three shapes carry colour; anything else is left alone. A step result gets a
 * coloured status word and a quiet parenthetical (with a red exit code when it
 * is not zero); a command about to run is cyan; a summary that counts failures
 * turns the number red when there are any.
 */
function detail(text: string): string {
  const status = /\s->\s(pass|fail|skip|workaround)\b/.exec(text);
  if (status) {
    const head = text.slice(0, status.index + 1);
    const tail = text.slice(status.index + status[0].length);
    const word = status[1]!;
    return `${head}${paint(`-> ${word}`, STATUS[word]!)}${parenthetical(tail)}`;
  }

  const command = / \$ /.exec(text);
  if (command) {
    const at = command.index + command[0].length;
    return `${text.slice(0, command.index)}${paint(" $", "gray")} ${paint(text.slice(at), "cyan")}`;
  }

  return text.replace(
    /\b(\d+) (failure|error)\(s\)/g,
    (whole, count: string) => (Number(count) > 0 ? paint(whole, "red") : whole),
  );
}

/** `(exit 127, 0.1s) — …`: code coloured by outcome, the rest quiet. */
function parenthetical(text: string): string {
  const exit = /\(exit (\d+)([,)])/.exec(text);
  if (!exit) return paint(text, "gray");

  const before = text.slice(0, exit.index);
  const after = text.slice(exit.index + exit[0].length);
  const code = exit[1]!;
  const tone: Tone = Number(code) === 0 ? "green" : "red";

  return `${paint(`${before}(exit `, "gray")}${paint(code, tone)}${paint(exit[2]!, "gray")}${paint(
    after,
    "gray",
  )}`;
}
