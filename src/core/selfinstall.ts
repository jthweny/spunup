import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { HOME, PKG_ROOT, IS_WIN } from "./paths.js";

/** `npx spunup` runs from npm's temporary cache. The first time (and after an update) it also installs
 *  the spunup command for good, with the same installer as the one-line install. */
export const fromNpx = (): boolean => /[\\/]_npx[\\/]/.test(PKG_ROOT);

const installedRoot = () => path.join(HOME, "app", ...(IS_WIN ? [] : ["lib"]), "node_modules", "spunup");

/** dist/cli.js of the installed copy, if there is one. */
export function installedCli(): string | undefined {
  const p = path.join(installedRoot(), "dist", "cli.js");
  return fs.existsSync(p) ? p : undefined;
}

export function installedVersion(): string | undefined {
  try { return JSON.parse(fs.readFileSync(path.join(installedRoot(), "package.json"), "utf8")).version; } catch { return undefined; }
}

let note: string | undefined;
/** The start screen shows the result once. */
export const takeInstallNote = () => { const n = note; note = undefined; return n; };

export async function selfInstall(version: string): Promise<void> {
  // npx puts its own copy on PATH and sets npm_* variables; the installer must see neither, or it would
  // take npx's spunup for a different program and install itself as "spunup-ai".
  const env: NodeJS.ProcessEnv = {};
  for (const [k, v] of Object.entries(process.env)) if (!/^npm_/i.test(k)) env[k] = v;
  const pathKey = Object.keys(env).find((k) => k.toUpperCase() === "PATH") || "PATH";
  env[pathKey] = (env[pathKey] || "").split(path.delimiter).filter((d) => !/[\\/]_npx[\\/]|node_modules[\\/]\.bin/.test(d)).join(path.delimiter);
  env.SPUNUP_PACKAGE ||= `spunup@${version}`;   // set by hand only for testing with a local tarball
  const [cmd, args] = IS_WIN
    ? ["powershell.exe", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path.join(PKG_ROOT, "install", "install.ps1")]]
    : ["sh", [path.join(PKG_ROOT, "install", "install.sh")]];
  const ok = await new Promise<boolean>((resolve) => {
    const p = spawn(cmd, args, { env, stdio: "ignore", windowsHide: true });
    p.on("error", () => resolve(false));
    p.on("exit", (code) => resolve(code === 0));
  });
  const bin = path.join(HOME, "bin");
  const name = ["spunup", "spunup-ai"].find((n) => fs.existsSync(path.join(bin, IS_WIN ? `${n}.cmd` : n)));
  note = ok && installedVersion() === version && name
    ? `✓ spunup is installed. Next time, just type: ${name}`
    : "Setting up the spunup command did not work, but npx spunup still works.";
}
