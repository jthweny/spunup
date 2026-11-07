import fs from "node:fs";
import net from "node:net";
import nodeHttp from "node:http";
import https from "node:https";
import { pipeline } from "node:stream/promises";

export class HttpError extends Error {
  constructor(public status: number, public body: string, message?: string) {
    super(message || HttpError.describe(status, body));
  }
  /** "HTTP 400: <the provider's own message>" so failures are never a bare status code. */
  static describe(status: number, body: string): string {
    let msg = "";
    try { const j = JSON.parse(body); msg = j.msg || j.message || j.detail || j.error?.message || j.error || ""; } catch { msg = body.replace(/<[^>]*>/g, " ").trim(); }
    return msg ? `HTTP ${status}: ${String(msg).slice(0, 240)}` : `HTTP ${status}`;
  }
}

export async function http<T = any>(
  url: string,
  opts: { method?: string; headers?: Record<string, string>; body?: unknown; timeoutMs?: number; raw?: boolean } = {},
): Promise<T> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), opts.timeoutMs ?? 30_000);
  const dbg = process.env.SPUNUP_DEBUG ? (m: string) => fs.appendFileSync(process.env.SPUNUP_DEBUG!, `${new Date().toISOString()} ${m} ${url}\n`) : () => {};
  dbg("start");
  try {
    const res = await fetch(url, {
      method: opts.method || (opts.body !== undefined ? "POST" : "GET"),
      headers: { "user-agent": "spunup", ...(opts.body !== undefined ? { "content-type": "application/json" } : {}), ...opts.headers },
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
      signal: ctrl.signal,
    });
    dbg(`headers ${res.status}`);
    const text = await res.text();
    dbg(`body ${text.length}`);
    if (!res.ok) throw new HttpError(res.status, text);
    if (opts.raw) return text as T;
    return (text ? JSON.parse(text) : {}) as T;
  } finally {
    clearTimeout(t);
  }
}

/** A streaming GET for big files, following redirects. Uses node:https, not fetch: on a Windows 10
 *  machine fetch read a Hugging Face file at 2 MB/s while node:https and curl got 22 MB/s on the
 *  same link (RESEARCH.md §7). The Authorization header is dropped when a redirect leaves the
 *  host (Hugging Face sends files from its CDN with a signed URL). */
export function getStream(url: string, opts: { headers?: Record<string, string>; signal?: AbortSignal } = {}, hops = 0): Promise<nodeHttp.IncomingMessage> {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const req = (u.protocol === "http:" ? nodeHttp : https).get(u, { headers: { "user-agent": "spunup", ...opts.headers }, signal: opts.signal }, (res) => {
      const code = res.statusCode || 0;
      if (code >= 300 && code < 400 && res.headers.location) {
        res.resume();
        if (hops >= 8) return reject(new Error(`Too many redirects: ${url}`));
        const next = new URL(res.headers.location, u);
        const headers = { ...opts.headers };
        if (next.host !== u.host) delete headers.authorization;
        getStream(next.toString(), { ...opts, headers }, hops + 1).then(resolve, reject);
        return;
      }
      resolve(res);
    });
    req.on("error", reject);
  });
}

const bodyText = (res: nodeHttp.IncomingMessage) => new Promise<string>((r) => { let t = ""; res.setEncoding("utf8"); res.on("data", (c) => { if (t.length < 4000) t += c; }); res.on("end", () => r(t)); res.on("error", () => r(t)); });
export async function readBody(res: nodeHttp.IncomingMessage): Promise<string> { return bodyText(res); }

/** Download to `dest`, carrying on from a partial `.part` file left by an earlier try. */
export async function download(url: string, dest: string, onProgress?: (done: number, total: number) => void): Promise<void> {
  const tmp = dest + ".part";
  for (let attempt = 1; ; attempt++) {
    const have = fs.existsSync(tmp) ? fs.statSync(tmp).size : 0;
    // A connection that stops sending data is dropped after 30 s and resumed (up to 5 tries).
    const ctl = new AbortController();
    let timer = setTimeout(() => ctl.abort(), 30_000);
    try {
      const res = await getStream(url, { headers: have ? { range: `bytes=${have}-` } : {}, signal: ctl.signal });
      const code = res.statusCode || 0;
      if (code < 200 || code >= 300) throw new HttpError(code, await bodyText(res), `Download failed (${code}): ${url}`);
      const resumed = code === 206;
      let done = resumed ? have : 0;
      const total = Number(res.headers["content-length"] || 0) + done;
      const src = res;
      src.on("data", (c: Buffer) => {
        clearTimeout(timer); timer = setTimeout(() => ctl.abort(), 30_000);
        done += c.length;
        onProgress?.(done, total);
      });
      await pipeline(src, fs.createWriteStream(tmp, { flags: resumed ? "a" : "w" }));
      break;
    } catch (e) {
      if (e instanceof HttpError || attempt >= 5) throw e instanceof HttpError ? e : new Error(`The download keeps stopping: ${url}. Check your internet connection or VPN, then try again.`);
    } finally {
      clearTimeout(timer);
    }
  }
  fs.renameSync(tmp, dest);
}

export function portFree(port: number, host = "127.0.0.1"): Promise<boolean> {
  return new Promise((resolve) => {
    const s = net.createServer();
    s.once("error", () => resolve(false));
    s.once("listening", () => s.close(() => resolve(true)));
    s.listen(port, host);
  });
}

export async function freePort(start: number, avoid: Set<number> = new Set()): Promise<number> {
  for (let p = start; p < start + 2000; p++) {
    if (avoid.has(p)) continue;
    if (await portFree(p)) return p;
  }
  throw new Error("No free port found");
}

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
