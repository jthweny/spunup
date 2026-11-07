import { spawnSync, type SpawnSyncOptions, type SpawnSyncReturns } from "node:child_process";

/** spawnSync, then put the terminal back into raw mode on Windows. A console program (PowerShell,
 *  nvidia-smi, wsl.exe…) shares spunup's console and can leave it in line mode, after which arrow
 *  keys and Esc never reach the screens (only Enter does). Seen on Windows 10 22H2. */
export function runSync(cmd: string, args: string[], opts: SpawnSyncOptions & { encoding: BufferEncoding }): SpawnSyncReturns<string>;
export function runSync(cmd: string, args: string[], opts?: SpawnSyncOptions): SpawnSyncReturns<string | Buffer>;
export function runSync(cmd: string, args: string[], opts: SpawnSyncOptions = {}): SpawnSyncReturns<string | Buffer> {
  const r = spawnSync(cmd, args, { windowsHide: true, ...opts });
  restoreRawMode();
  return r;
}

export function restoreRawMode(): void {
  const s = process.stdin;
  if (process.platform !== "win32" || !s.isTTY || !s.isRaw) return;
  // libuv skips setting a mode it thinks is already set, so switch off and on again.
  s.setRawMode(false);
  s.setRawMode(true);
}
