import fs from "node:fs";
import path from "node:path";
import { P } from "./paths.js";
import { getKey } from "./keys.js";
import { hfTree, type Quant } from "./recipes.js";
import { getStream, readBody } from "./net.js";

export interface FileItem { path: string; size: number; url: string; out?: string /* remote: full target path */ }

const SKIP = /^(original|metal|onnx|coreml|openvino|figures|assets|images|evaluation|inference|encoding)\//i;

/** Files spunup needs for one quantization (weights + tokenizer/config files). */
export async function filesFor(q: Quant): Promise<FileItem[]> {
  const tree = (await hfTree(q.repo, { strict: true })).filter((f) => f.size !== undefined && (f as any).type !== "directory");
  const url = (p: string) => `https://huggingface.co/${q.repo}/resolve/main/${p.split("/").map(encodeURIComponent).join("/")}`;
  if (q.format === "gguf") {
    const first = q.file || "";
    const stem = first.replace(/-0*1-of-\d+\.gguf$/, "");
    // Only this file's own parts (Model-Q2_K-0000x-of-…, not Model-Q2_K_L-…).
    const own = new RegExp(`^${stem.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}-\\d+-of-\\d+\\.gguf$`);
    const parts = q.parts && q.parts > 1 ? tree.filter((f) => own.test(f.path)) : tree.filter((f) => f.path === first);
    return parts.map((f) => ({ path: f.path, size: f.size, url: url(f.path) }));
  }
  const hasSt = tree.some((f) => f.path.endsWith(".safetensors") && !f.path.includes("/"));
  // Mistral repos ship the weights twice (consolidated.safetensors and the sharded model-0000x files);
  // vLLM and SGLang load the sharded ones.
  const sharded = tree.some((f) => /^model-\d+-of-\d+\.safetensors$/.test(f.path));
  return tree
    .filter((f) => !(sharded && /^consolidated.*\.safetensors$/.test(f.path)))
    .filter((f) => !SKIP.test(f.path))
    .filter((f) => !f.path.endsWith(".gguf") && !(hasSt && /\.(bin|pt|pth)$/.test(f.path)))
    .filter((f) => !/\.(png|jpe?g|gif|mp4|pdf|md)$/i.test(f.path) || /chat_template/.test(f.path))
    .map((f) => ({ path: f.path, size: f.size, url: url(f.path) }));
}

export function localModelDir(q: Quant): string {
  return path.join(P.models, q.repo.replace("/", "__"));
}

/** A connection that stops sending data (VPN, flaky Wi-Fi) is dropped after this long and resumed. */
const STALL_MS = 30_000;
const TRIES = 5;

/** Download with resume. onProgress(doneBytes, totalBytes). */
export async function downloadFiles(files: FileItem[], dest: string, onProgress: (d: number, t: number) => void, isCancelled: () => boolean): Promise<void> {
  const total = files.reduce((a, f) => a + f.size, 0);
  let done = 0;
  const token = getKey("hf");
  for (const f of files) {
    const out = path.join(dest, f.path);
    fs.mkdirSync(path.dirname(out), { recursive: true });
    if (fs.existsSync(out) && fs.statSync(out).size === f.size) { done += f.size; onProgress(done, total); continue; }
    const part = out + ".part";
    const before = done;
    for (let attempt = 1; ; attempt++) {
      const have = fs.existsSync(part) ? fs.statSync(part).size : 0;
      done = before;
      const ctl = new AbortController();
      let timer = setTimeout(() => ctl.abort(), STALL_MS);
      const kick = () => { clearTimeout(timer); timer = setTimeout(() => ctl.abort(), STALL_MS); };
      try {
        const res = await getStream(f.url, { headers: { ...(token ? { authorization: `Bearer ${token}` } : {}), ...(have ? { range: `bytes=${have}-` } : {}) }, signal: ctl.signal });
        const code = res.statusCode || 0;
        if (code === 401 || code === 403) { res.resume(); throw new Error("Hugging Face refused the download. This model is gated: accept its license on the model page and check your token in Keys & tokens."); }
        if (code === 416 && have === f.size) { res.resume(); break; }   // the .part is already complete
        if (code < 200 || code >= 300) { await readBody(res); throw new Error(`Download failed (${code}) for ${f.path}.`); }
        const append = code === 206;
        done += append ? have : 0;
        const ws = fs.createWriteStream(part, { flags: append ? "a" : "w" });
        try {
          for await (const value of res as AsyncIterable<Buffer>) {
            if (isCancelled()) { res.destroy(); throw new Error("cancelled"); }
            kick();
            done += value.length;
            if (!ws.write(value)) await new Promise<void>((r) => ws.once("drain", () => r()));
            onProgress(done, total);
          }
        } finally {
          await new Promise<void>((r) => ws.end(() => r()));
        }
        break;
      } catch (e) {
        const stalled = ctl.signal.aborted || /fetch failed|ECONNRESET|ETIMEDOUT|EPIPE|ECONNREFUSED|EAI_AGAIN|socket|terminated|timeout|aborted/i.test(String((e as Error).message) + String((e as any).code || "") + String((e as any).cause?.code || ""));
        if (!stalled || isCancelled() || attempt >= TRIES) throw stalled && !isCancelled() ? new Error(`The download of ${f.path} keeps stopping (no data for ${STALL_MS / 1000} s, ${TRIES} tries). Check your internet connection or VPN, then start again: it carries on where it stopped.`) : e;
      } finally {
        clearTimeout(timer);
      }
    }
    fs.renameSync(part, out);
  }
}

/** Shell script that downloads the same files on a rented machine (curl or wget, whichever exists). */
export function remoteDownloadScript(files: FileItem[], dir: string, withToken: boolean): string {
  // Single-quoted for the shell, with any ' in a file name escaped (a name must never run as a command).
  const q = (v: string) => `'${v.replace(/'/g, `'\\''`)}'`;
  const lines = files.map((f) => {
    const o = f.out || `${dir}/${f.path}`;
    return `mkdir -p "$(dirname ${q(o)})"; dl ${q(f.url)} ${q(o)}`;
  });
  return `#!/bin/sh
set -e
dl() {
  if [ -f "$2" ]; then return 0; fi
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 -C - ${withToken ? '-H "Authorization: Bearer $HF_TOKEN"' : ""} -o "$2.part" "$1"
  else
    wget -q -c ${withToken ? '--header="Authorization: Bearer $HF_TOKEN"' : ""} -O "$2.part" "$1"
  fi
  mv "$2.part" "$2"
}
${lines.join("\n")}
echo SPUNUP_DOWNLOAD_DONE
`;
}

/** Measure this computer's download speed from Hugging Face (MB/s) by fetching ~3 s of the real file. */
export async function measureDownloadMBps(q: Quant): Promise<number | undefined> {
  try {
    const files = await filesFor(q);
    const big = [...files].sort((a, b) => b.size - a.size)[0];
    if (!big) return undefined;
    const token = getKey("hf");
    const t0 = performance.now();
    const res = await getStream(big.url, { headers: { range: "bytes=0-268435455", ...(token ? { authorization: `Bearer ${token}` } : {}) }, signal: AbortSignal.timeout(10_000) });
    if ((res.statusCode || 0) < 200 || (res.statusCode || 0) >= 300) { res.resume(); return undefined; }
    let got = 0, start = 0;
    for await (const value of res as AsyncIterable<Buffer>) {
      if (!start) start = performance.now();     // skip connection setup time
      got += value.length;
      if (performance.now() - start > 3000) break;
    }
    res.destroy();
    const s = (performance.now() - (start || t0)) / 1000;
    return s > 0.2 ? got / 1e6 / s : undefined;
  } catch {
    return undefined;
  }
}
