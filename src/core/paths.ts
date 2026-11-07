import os from "node:os";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

/** Everything spunup keeps lives in one folder: ~/.spunup */
export const HOME = process.env.SPUNUP_HOME || path.join(os.homedir(), ".spunup");

export const P = {
  home: HOME,
  config: path.join(HOME, "config.json"),
  history: path.join(HOME, "history.json"),
  secrets: path.join(HOME, "secrets.json"),
  keeperState: path.join(HOME, "keeper.json"),
  keeperStatus: path.join(HOME, "keeper-status.json"),
  keeperLog: path.join(HOME, "logs", "keeper.log"),
  logs: path.join(HOME, "logs"),
  runtimes: path.join(HOME, "runtimes"),
  models: path.join(HOME, "models"),
  recipesCache: path.join(HOME, "recipes"),
  ssh: path.join(HOME, "ssh"),
  sshKey: path.join(HOME, "ssh", "spunup_ed25519"),
  bin: path.join(HOME, "bin"),
};

export function ensureDirs(): void {
  for (const d of [P.home, P.logs, P.runtimes, P.models, P.recipesCache, P.ssh, P.bin]) {
    fs.mkdirSync(d, { recursive: true, mode: 0o700 });
  }
}

/** Folder of the installed package (holds the shipped recipes/). */
export const PKG_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

export const IS_WIN = process.platform === "win32";
export const IS_MAC = process.platform === "darwin";
export const IS_LINUX = process.platform === "linux";

export function exe(name: string): string {
  return IS_WIN ? `${name}.exe` : name;
}
