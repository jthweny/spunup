/** Fallback for computers without the system `ssh` client (spec 13).
 *  The ssh2 library (pure JavaScript) is downloaded into ~/.spunup the first time it is needed,
 *  so the normal install stays small. */
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { EventEmitter } from "node:events";
import { createRequire } from "node:module";
import { runSync } from "./proc.js";
import { P, IS_WIN } from "./paths.js";

const DIR = path.join(P.runtimes, "ssh2");
const VERSION = "1.17.0";

function load(): any {
  const entry = path.join(DIR, "node_modules", "ssh2");
  if (!fs.existsSync(entry)) {
    fs.mkdirSync(DIR, { recursive: true });
    // npm's own script, run by this Node with no shell: npm's launcher needs "node" on PATH (not there with
    // spunup's private Node), and a shell splits paths with spaces (C:\Program Files\nodejs).
    const nodeDir = path.dirname(process.execPath);
    const npmCli = IS_WIN ? path.join(nodeDir, "node_modules", "npm", "bin", "npm-cli.js") : path.join(nodeDir, "..", "lib", "node_modules", "npm", "bin", "npm-cli.js");
    const args = ["install", "--prefix", DIR, "--no-fund", "--no-audit", "--omit=optional", `ssh2@${VERSION}`];
    const npm = path.join(nodeDir, IS_WIN ? "npm.cmd" : "npm");
    const r = fs.existsSync(npmCli) ? runSync(process.execPath, [npmCli, ...args], { stdio: "ignore" })
      : runSync(fs.existsSync(npm) ? npm : "npm", args, { stdio: "ignore", shell: IS_WIN });
    if (r.status !== 0) throw new Error("No ssh program was found, and spunup could not download its built-in SSH helper.");
  }
  return createRequire(path.join(DIR, "index.js"))("ssh2");
}

export function libGenerateKey(keyPath: string): void {
  const { utils } = load();
  const k = utils.generateKeyPairSync("ed25519", { comment: "spunup" });
  fs.writeFileSync(keyPath, k.private, { mode: 0o600 });
  fs.writeFileSync(keyPath + ".pub", k.public + "\n", { mode: 0o644 });
}

interface Remote { host: string; port: number; user: string }

function connect(r: Remote): Promise<any> {
  const { Client } = load();
  return new Promise((resolve, reject) => {
    const c = new Client();
    c.on("ready", () => resolve(c)).on("error", reject)
      .connect({ host: r.host, port: r.port, username: r.user, privateKey: fs.readFileSync(P.sshKey), readyTimeout: 15000, keepaliveInterval: 15000, keepaliveCountMax: 3 });
  });
}

export async function libExec(r: Remote, command: string, input?: string, timeoutMs = 60_000): Promise<string> {
  const c = await connect(r);
  try {
    return await new Promise<string>((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("The machine took too long to answer over SSH.")), timeoutMs);
      c.exec(command, (err: Error, stream: any) => {
        if (err) { clearTimeout(t); return reject(err); }
        let out = "", e = "", status: number | undefined;
        stream.on("data", (d: Buffer) => (out += d)).stderr.on("data", (d: Buffer) => (e += d));
        // The exit status arrives in its own message; "close" only carries it when it came first.
        stream.on("exit", (code: number | null) => { status = code ?? 255; });
        stream.on("close", (code?: number) => {
          clearTimeout(t);
          const c = status ?? code;
          c === 0 ? resolve(out) : reject(new Error(e.trim() || (c === undefined ? "The SSH connection closed before the command finished." : `exit ${c}`)));
        });
        if (input !== undefined) stream.end(input);
      });
    });
  } finally {
    c.end();
  }
}

/** Behaves like the ssh child process the keeper expects: has kill(), exitCode and an "exit" event. */
export function libTunnel(r: Remote, localPort: number, remotePort: number): EventEmitter & { kill: () => void; exitCode: number | null } {
  const ev = new EventEmitter() as EventEmitter & { kill: () => void; exitCode: number | null };
  ev.exitCode = null;
  let client: any;
  const server = net.createServer((sock) => {
    if (!client) return sock.destroy();
    client.forwardOut("127.0.0.1", localPort, "127.0.0.1", remotePort, (err: Error, stream: any) => {
      if (err) return sock.destroy();
      sock.pipe(stream).pipe(sock);
    });
  });
  const finish = () => { if (ev.exitCode !== null) return; ev.exitCode = 1; server.close(); try { client?.end(); } catch { /* */ } ev.emit("exit", 1); };
  ev.kill = finish;
  connect(r).then((c) => {
    client = c;
    c.on("close", finish).on("error", finish);
    server.on("error", finish);
    server.listen(localPort, "127.0.0.1");
  }).catch(finish);
  return ev;
}
