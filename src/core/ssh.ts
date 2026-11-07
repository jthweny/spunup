import fs from "node:fs";
import path from "node:path";
import { spawn, type ChildProcess } from "node:child_process";
import { runSync, restoreRawMode } from "./proc.js";
import { P, ensureDirs } from "./paths.js";
import { hasCommand } from "../runtimes/install.js";
import { libGenerateKey, libExec, libTunnel } from "./sshlib.js";

export interface Remote { host: string; port: number; user: string }

/** spunup's own SSH key (ed25519), made once. Returns the public key line. */
export function ensureSshKey(): string {
  ensureDirs();
  if (!fs.existsSync(P.sshKey)) {
    if (!hasCommand("ssh-keygen")) libGenerateKey(P.sshKey);
    else {
      const r = runSync("ssh-keygen", ["-t", "ed25519", "-N", "", "-C", "spunup", "-f", P.sshKey], { stdio: "ignore" });
      if (r.status !== 0) throw new Error("Could not create an SSH key.");
    }
  }
  return fs.readFileSync(P.sshKey + ".pub", "utf8").trim();
}

let sysSsh: boolean | undefined;
export function sshAvailable(): boolean {
  return (sysSsh ??= hasCommand("ssh"));
}

function baseArgs(r: Remote): string[] {
  return [
    "-i", P.sshKey, "-p", String(r.port),
    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
    // One known_hosts per address: a re-rented Vast host or a restarted RunPod pod comes back with a new
    // host key on the same address, and accept-new would refuse it with one shared file.
    "-o", "StrictHostKeyChecking=accept-new", "-o", `UserKnownHostsFile=${path.join(P.ssh, `known_hosts-${r.host.replace(/[^\w.-]/g, "_")}-${r.port}`)}`,
    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "-o", "ConnectTimeout=15",
  ];
}

/** Run a command on the machine. Resolves with stdout; rejects with stderr on failure. */
export function sshExec(r: Remote, command: string, timeoutMs = 60_000): Promise<string> {
  if (!sshAvailable()) return libExec(r, command, undefined, timeoutMs);
  return new Promise((resolve, reject) => {
    const p = spawn("ssh", [...baseArgs(r), `${r.user}@${r.host}`, command], { windowsHide: true });
    p.on("exit", restoreRawMode);
    let out = "", err = "";
    const t = setTimeout(() => { p.kill(); reject(new Error("The machine took too long to answer over SSH.")); }, timeoutMs);
    p.stdout.on("data", (b) => (out += b));
    p.stderr.on("data", (b) => (err += b));
    p.on("error", (e) => { clearTimeout(t); reject(e); });
    p.on("close", (c) => {
      clearTimeout(t);
      if (c === 0) resolve(out);
      else reject(new Error(err.trim().split("\n").slice(-3).join(" ") || `ssh exited with code ${c}`));
    });
  });
}

/** Write a file on the machine through stdin (no scp needed). */
export function sshWrite(r: Remote, remotePath: string, content: string, mode = "600"): Promise<void> {
  const cmd = `mkdir -p "$(dirname '${remotePath}')" && cat > '${remotePath}' && chmod ${mode} '${remotePath}'`;
  if (!sshAvailable()) return libExec(r, cmd, content).then(() => undefined);
  return new Promise((resolve, reject) => {
    const p = spawn("ssh", [...baseArgs(r), `${r.user}@${r.host}`, cmd], { windowsHide: true });
    p.on("exit", restoreRawMode);
    let err = "";
    p.stderr.on("data", (b) => (err += b));
    p.on("close", (c) => (c === 0 ? resolve() : reject(new Error(err.trim() || "Could not write the file on the machine."))));
    p.stdin.end(content);
  });
}

export async function waitForSsh(r: Remote, timeoutMs: number, isCancelled: () => boolean): Promise<void> {
  const until = Date.now() + timeoutMs;
  let last = "";
  while (Date.now() < until) {
    if (isCancelled()) throw new Error("cancelled");
    try { await sshExec(r, "echo ok", 20_000); return; } catch (e) { last = (e as Error).message; }
    await new Promise((res) => setTimeout(res, 5000));
  }
  throw new Error(`Could not connect to the machine over SSH (${last}).`);
}

/** Private tunnel: 127.0.0.1:localPort on this computer → 127.0.0.1:remotePort on the machine. */
export function openTunnel(r: Remote, localPort: number, remotePort: number): ChildProcess {
  if (!sshAvailable()) return libTunnel(r, localPort, remotePort) as unknown as ChildProcess;
  return spawn("ssh", [...baseArgs(r), "-N", "-o", "ExitOnForwardFailure=yes",
    "-L", `127.0.0.1:${localPort}:127.0.0.1:${remotePort}`, `${r.user}@${r.host}`], { windowsHide: true, stdio: "ignore" });
}
