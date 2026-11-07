/** The keeper: a tiny helper process that only runs while a cloud model (or an MLX model) is up.
 *  - holds each cloud endpoint's SSH tunnel and reconnects it quietly when it drops
 *  - serves the fixed local port (apps never change settings), forwarding to the tunnel
 *  - checks the API key in front of MLX, which has no key option of its own
 *  - stops or deletes idle machines when auto-stop is on
 *  It exits by itself once nothing needs it. */
import http from "node:http";
import fs from "node:fs";
import { spawn, type ChildProcess } from "node:child_process";
import { P, ensureDirs } from "./paths.js";
import { loadHistory, upsertEndpoint, readJson, writeJson, type Endpoint } from "./store.js";
import { openTunnel } from "./ssh.js";
import { provider } from "../providers/index.js";
import { refreshDownloadCost } from "./launch.js";

interface Status {
  pid: number;
  at: number;
  endpoints: Record<string, { tunnelUp: boolean; lastRequest: number; reconnects: number; error?: string }>;
}

export function keeperNeeded(e: Endpoint): boolean {
  return (e.status === "running" || e.status === "starting") && (e.where !== "local" || e.runtime === "mlx") && !!e.innerPort;
}

export function keeperRunning(): number | undefined {
  const s = readJson<Status | null>(P.keeperStatus, null);
  if (!s?.pid) return undefined;
  try { process.kill(s.pid, 0); } catch { return undefined; }
  return Date.now() - s.at < 30_000 ? s.pid : undefined;
}

/** Start the keeper in the background if any endpoint needs it. */
export function ensureKeeper(cliPath: string): void {
  if (!loadHistory().endpoints.some(keeperNeeded)) return;
  if (keeperRunning()) return;
  ensureDirs();
  const log = fs.openSync(P.keeperLog, "a");
  const p = spawn(process.execPath, [cliPath, "__keeper"], { detached: true, stdio: ["ignore", log, log], windowsHide: true });
  p.unref();
}

export function keeperStatus(): Status | null {
  return readJson<Status | null>(P.keeperStatus, null);
}

// ------------------------------------------------------------------ the process itself
export async function runKeeper(): Promise<void> {
  if (keeperRunning() && keeperRunning() !== process.pid) return;
  // Lock before the first round (which can wait on a provider for a while), so a second start backs off.
  const lock = `${P.keeperStatus}.lock`;
  try {
    const fd = fs.openSync(lock, "wx");
    fs.writeSync(fd, String(process.pid)); fs.closeSync(fd);
  } catch {
    const other = Number(fs.readFileSync(lock, "utf8")) || 0;
    let alive = false;
    try { if (other && other !== process.pid) { process.kill(other, 0); alive = true; } } catch { /* gone */ }
    if (alive) return;
    fs.writeFileSync(lock, String(process.pid));   // stale lock from a keeper that died
  }
  const unlock = () => { try { if (Number(fs.readFileSync(lock, "utf8")) === process.pid) fs.unlinkSync(lock); } catch { /* gone */ } };
  process.on("exit", unlock);
  const status: Status = { pid: process.pid, at: Date.now(), endpoints: {} };
  const prev = readJson<Status | null>(P.keeperStatus, null);
  const servers = new Map<string, http.Server>();
  const tunnels = new Map<string, { child?: ChildProcess; backoff: number; nextTry: number }>();
  let idleSince = 0;
  const log = (m: string) => console.log(`${new Date().toISOString()} ${m}`);
  log(`keeper started (pid ${process.pid})`);

  const touch = (id: string) => { if (status.endpoints[id]) status.endpoints[id].lastRequest = Date.now(); };

  function serve(e: Endpoint) {
    if (servers.has(e.id)) return;
    const srv = http.createServer((req, res) => {
      const cur = loadHistory().endpoints.find((x) => x.id === e.id) || e;
      if (cur.runtime === "mlx" && cur.where === "local") {
        const auth = req.headers["authorization"] || "";
        if (auth !== `Bearer ${cur.apiKey}`) {
          res.writeHead(401, { "content-type": "application/json" });
          res.end(JSON.stringify({ error: { message: "Missing or wrong API key.", type: "invalid_api_key" } }));
          return;
        }
      }
      touch(e.id);
      const forward = (body?: Buffer) => {
        const headers = body ? { ...req.headers, "content-length": String(body.length) } : req.headers;
        const up = http.request({ host: "127.0.0.1", port: cur.innerPort, method: req.method, path: req.url, headers }, (ur) => {
          res.writeHead(ur.statusCode || 502, ur.headers);
          ur.pipe(res);
          ur.on("data", () => touch(e.id));
        });
        up.on("error", () => {
          if (!res.headersSent) res.writeHead(503, { "content-type": "application/json", "retry-after": "5" });
          res.end(JSON.stringify({ error: { message: "spunup is reconnecting the private connection to your model. Try again in a few seconds.", type: "unavailable" } }));
        });
        if (body) up.end(body); else req.pipe(up);
      };
      // mlx-lm reads a request's "model" as a model to load (its server docs: "a path to a local model or Hugging
      // Face repo id"), so the endpoint's own name is dropped and mlx-lm answers with the model it has loaded.
      if (cur.runtime === "mlx" && cur.where === "local" && req.method === "POST" && /json/i.test(req.headers["content-type"] || "")) {
        const chunks: Buffer[] = [];
        req.on("data", (c: Buffer) => chunks.push(c));
        req.on("end", () => {
          let body = Buffer.concat(chunks);
          try {
            const j = JSON.parse(body.toString("utf8"));
            if (j && typeof j === "object" && j.model === cur.name) { delete j.model; body = Buffer.from(JSON.stringify(j)); }
          } catch { /* not JSON after all: pass it on as it is */ }
          forward(body);
        });
        return;
      }
      forward();
    });
    srv.on("error", (err) => { status.endpoints[e.id] = { ...(status.endpoints[e.id] || { lastRequest: Date.now(), reconnects: 0, tunnelUp: false }), error: `Port ${e.port} is busy: ${err.message}` }; });
    srv.listen(e.port, "127.0.0.1");
    servers.set(e.id, srv);
  }

  function tunnel(e: Endpoint) {
    if (e.where === "local" || !e.ssh || !e.remotePort || !e.innerPort) return;
    const t = tunnels.get(e.id) || { backoff: 2000, nextTry: 0 };
    tunnels.set(e.id, t);
    if (t.child || Date.now() < t.nextTry) return;
    const c = openTunnel(e.ssh, e.innerPort, e.remotePort);
    t.child = c;
    const st = status.endpoints[e.id];
    const openedAt = Date.now();
    setTimeout(() => { if (t.child === c && c.exitCode === null) { st.tunnelUp = true; t.backoff = 2000; } }, 3000);
    c.on("exit", () => {
      t.child = undefined;
      st.tunnelUp = false;
      st.reconnects++;
      if (Date.now() - openedAt > 60_000) t.backoff = 2000;
      t.nextTry = Date.now() + t.backoff;
      t.backoff = Math.min(30_000, t.backoff * 2);
      log(`tunnel for ${e.name} dropped; retrying in ${Math.round(t.backoff / 1000)}s`);
    });
  }

  function close(id: string) {
    servers.get(id)?.close();
    servers.delete(id);
    const t = tunnels.get(id);
    if (t?.child) { t.child.removeAllListeners("exit"); t.child.kill(); }
    tunnels.delete(id);
    delete status.endpoints[id];
  }

  async function autoStop(e: Endpoint) {
    const st = status.endpoints[e.id];
    if (!e.autoStop?.enabled || e.where === "local" || !e.instanceId || e.status !== "running") return;
    if (Date.now() - st.lastRequest < e.autoStop.minutes * 60_000) return;
    log(`auto-stop: ${e.name} idle for ${e.autoStop.minutes} min → ${e.autoStop.action}`);
    const p = provider(e.where);
    try {
      if (e.autoStop.action === "delete") {
        await p.destroy(e.instanceId, { volumeId: e.volumeId, keepDisk: false });
        upsertEndpoint({ ...e, status: "deleted", costBefore: costSoFar(e) - (e.oneOffCost || 0), stoppedAt: Date.now() });
      } else {
        await p.stop(e.instanceId);
        upsertEndpoint({ ...e, status: "stopped", costBefore: costSoFar(e) - (e.oneOffCost || 0), stoppedAt: Date.now() });
      }
      close(e.id);
    } catch (err) {
      st.error = `Auto-stop failed: ${(err as Error).message}`;
    }
  }

  // When a tunnel will not come back, ask the provider why: the machine may have been stopped or
  // deleted somewhere else (another spunup, the provider's website). Then stop retrying.
  const lastCheck = new Map<string, number>();
  async function checkGone(e: Endpoint) {
    const st = status.endpoints[e.id];
    if (e.where === "local" || !e.instanceId || st.tunnelUp || st.reconnects < 2) return;
    if (Date.now() - (lastCheck.get(e.id) || 0) < 60_000) return;
    lastCheck.set(e.id, Date.now());
    try {
      const inst = await provider(e.where).get(e.instanceId);
      // A Community Cloud pod can come back on a new address (RunPod docs): follow it.
      if (inst.status === "running" && inst.ssh && (inst.ssh.host !== e.ssh?.host || inst.ssh.port !== e.ssh?.port)) {
        log(`${e.name}: the machine has a new address; reconnecting`);
        upsertEndpoint({ ...e, ssh: inst.ssh });
        close(e.id);
        return;
      }
      if (inst.status === "gone") {
        log(`${e.name}: the machine no longer exists; stopping its tunnel`);
        upsertEndpoint({ ...e, status: "deleted", costBefore: costSoFar(e) - (e.oneOffCost || 0), stoppedAt: undefined, stoppedPricePerHour: 0 });
        close(e.id);
      } else if (inst.status === "stopped") {
        log(`${e.name}: the machine was stopped; stopping its tunnel`);
        upsertEndpoint({ ...e, status: "stopped", costBefore: costSoFar(e) - (e.oneOffCost || 0), stoppedAt: Date.now(), stoppedPricePerHour: inst.stoppedPricePerHour ?? e.stoppedPricePerHour });
        close(e.id);
      }
    } catch { /* offline: keep trying */ }
  }

  const lastBill = new Map<string, number>();   // when each Vast machine's download charges were last read
  const tick = async () => {
    const want = loadHistory().endpoints.filter(keeperNeeded);
    const ids = new Set(want.map((e) => e.id));
    for (const id of [...servers.keys()]) if (!ids.has(id)) close(id);
    for (const e of want) {
      status.endpoints[e.id] ||= { tunnelUp: e.where === "local", lastRequest: prev?.endpoints[e.id]?.lastRequest || Date.now(), reconnects: 0 };
      serve(e);
      tunnel(e);
      await checkGone(e);
      if (e.where === "vast" && Date.now() - (lastBill.get(e.id) || 0) > 5 * 60_000) { lastBill.set(e.id, Date.now()); await refreshDownloadCost(e); }
      await autoStop(e);
    }
    status.at = Date.now();
    writeJson(P.keeperStatus, status);
    if (!want.length) {
      idleSince ||= Date.now();
      if (Date.now() - idleSince > 60_000) { log("nothing to keep; exiting"); process.exit(0); }
    } else idleSince = 0;
  };
  // The next round starts 5 s after this one ends, so slow provider calls never make rounds overlap.
  const loop = async () => { await tick().catch((e) => log(`tick error: ${e.message}`)); setTimeout(loop, 5000); };
  await loop();
}

/** Cost so far for a cloud endpoint, from the listed prices and start/stop times (USD). */
export function costSoFar(e: Endpoint, now = Date.now()): number {
  if (e.where === "local") return 0;
  let c = (e.costBefore || 0) + (e.oneOffCost || 0);
  // A rented machine costs from the moment it is rented, booting included ("starting").
  if ((e.status === "running" || (e.status === "starting" && e.instanceId)) && e.startedAt) c += ((now - e.startedAt) / 3600_000) * (e.pricePerHour || 0);
  if (e.status === "deleted" && e.diskKept && e.deletedAt) c += ((now - e.deletedAt) / 3600_000) * (e.stoppedPricePerHour || 0);
  if (e.status === "stopped" && e.stoppedAt) c += ((now - e.stoppedAt) / 3600_000) * (e.stoppedPricePerHour || 0);
  return c;
}
