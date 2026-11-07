/** Colours and the terminal-background check (spec 6.4).
 *  tokens/s always gets the strongest contrast; prefill gets less:
 *   - dark background  → tokens/s bright white, prefill a dimmer grey
 *   - light background → tokens/s black, prefill a lighter grey
 *   - unknown          → tokens/s bold, prefill uses the terminal's own "dim" style */
export type Bg = "dark" | "light" | "unknown";

let bg: Bg = "unknown";
export const background = () => bg;

function fromColorFgBg(): Bg | undefined {
  const v = process.env.COLORFGBG;
  if (!v) return undefined;
  const last = Number(v.split(";").pop());
  if (Number.isNaN(last)) return undefined;
  return last === 7 || last === 15 || (last >= 9 && last <= 14) ? "light" : "dark";
}

/** Ask the terminal for its background colour (OSC 11). Gives up after 200 ms. */
function queryOsc11(): Promise<Bg | undefined> {
  const { stdin, stdout } = process;
  if (!stdin.isTTY || !stdout.isTTY || process.env.TERM === "dumb") return Promise.resolve(undefined);
  return new Promise((resolve) => {
    let buf = "";
    const wasRaw = stdin.isRaw;
    const done = (v: Bg | undefined) => {
      clearTimeout(t);
      stdin.off("data", onData);
      try { stdin.setRawMode(wasRaw); } catch { /* ignore */ }
      stdin.pause();
      resolve(v);
    };
    const onData = (d: Buffer) => {
      buf += d.toString("latin1");
      const m = /rgb:([0-9a-f]+)\/([0-9a-f]+)\/([0-9a-f]+)/i.exec(buf);
      if (!m) return;
      const ch = (h: string) => parseInt(h.slice(0, 2), 16) / 255;
      const lum = 0.2126 * ch(m[1]) + 0.7152 * ch(m[2]) + 0.0722 * ch(m[3]);
      done(lum > 0.5 ? "light" : "dark");
    };
    const t = setTimeout(() => done(undefined), 200);
    try {
      stdin.setRawMode(true);
      stdin.resume();
      stdin.on("data", onData);
      stdout.write("\x1b]11;?\x07");
    } catch {
      done(undefined);
    }
  });
}

export async function detectBackground(): Promise<Bg> {
  bg = fromColorFgBg() || (await queryOsc11()) || "unknown";
  return bg;
}

/** Props for Ink <Text> */
export const strongSpeed = () => (bg === "dark" ? { color: "whiteBright", bold: true } : bg === "light" ? { color: "black", bold: true } : { bold: true });
// "gray" is ANSI bright black, which some dark themes (Solarized Dark) draw in the background colour; dim is always dimmer.
export const softSpeed = () => (bg === "light" ? { color: "gray" } : { dimColor: true });

export const C = {
  main: "whiteBright",
  alive: "green",
  dead: "gray",
  warn: "yellow",
  error: "red",
  accent: "cyan",
  editable: "cyan",
  teal: "#2ec4b6",
};

export function fmtInt(n: number): string {
  return Math.round(n).toLocaleString("en-GB");
}
