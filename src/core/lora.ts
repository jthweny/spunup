/** LoRA adapters: find them in a Hugging Face repo, work out which runtimes can load them,
 *  and what they cost in speed. Formats checked 2026-11-06 (RESEARCH.md §LoRA):
 *  - PEFT (adapter_config.json + adapter_model.safetensors): vLLM, SGLang
 *  - GGUF adapter file: llama.cpp (--lora), on top of any GGUF size of the same base model
 *  - MLX (adapter_config.json with lora_parameters + adapters.safetensors): mlx-lm only */
import { hfTree, type Quant } from "./recipes.js";
import { http } from "./net.js";
import { getKey } from "./keys.js";
import type { RuntimeId } from "./store.js";

export type AdapterKind = "peft" | "gguf" | "mlx";

export interface Adapter {
  repo: string;
  dir: string;                 // folder inside the repo ("" = top level)
  kind: AdapterKind;
  file?: string;               // GGUF adapter file
  files: { path: string; size: number }[];
  sizeGb: number;
  rank?: number;
  base?: string;               // base model the adapter was trained on, from its config
}

/** The adapter the user chose: one found in a repo, or a folder/file on this computer (Advanced). */
export interface LoraUse { name: string; adapter?: Adapter; localPath?: string }

export const ADAPTER_RUNTIMES: Record<AdapterKind, RuntimeId[]> = { peft: ["vllm", "sglang"], gguf: ["llamacpp"], mlx: ["mlx"] };
const FORMAT_KIND: Record<Quant["format"], AdapterKind> = { safetensors: "peft", gguf: "gguf", mlx: "mlx" };

export function adapterFitsQuant(a: Adapter, q: Quant): boolean { return FORMAT_KIND[q.format] === a.kind; }

export const KIND_NAME: Record<AdapterKind, string> = { peft: "Hugging Face (PEFT)", gguf: "GGUF", mlx: "MLX" };

const dirOf = (p: string) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "");
const raw = (repo: string, p: string) => `https://huggingface.co/${repo}/resolve/main/${p.split("/").map(encodeURIComponent).join("/")}`;
const hfHeaders = (): Record<string, string> => { const t = getKey("hf"); return t ? { authorization: `Bearer ${t}` } : {}; };

/** Every LoRA adapter in a repo (top level or in sub-folders). */
export async function findAdapters(repo: string): Promise<Adapter[]> {
  const tree = (await hfTree(repo)).filter((f) => f.size !== undefined);
  const out: Adapter[] = [];
  for (const cfg of tree.filter((f) => f.path.endsWith("adapter_config.json"))) {
    const dir = dirOf(cfg.path);
    const inDir = tree.filter((f) => dirOf(f.path) === dir);
    const peft = inDir.find((f) => /(^|\/)adapter_model\.(safetensors|bin)$/.test(f.path));
    const mlx = inDir.find((f) => /(^|\/)adapters\.safetensors$/.test(f.path));
    const weights = peft || mlx;
    if (!weights) continue;
    let conf: any = {};
    try { conf = await http<any>(raw(repo, cfg.path), { headers: hfHeaders(), timeoutMs: 15000 }); } catch { /* size-only */ }
    const files = [cfg, weights].map((f) => ({ path: f.path, size: f.size! }));
    out.push({
      repo, dir, kind: peft ? "peft" : "mlx", files, sizeGb: files.reduce((a, f) => a + f.size, 0) / 1e9,
      rank: Number(peft ? conf.r : conf.lora_parameters?.rank) || undefined,
      base: conf.base_model_name_or_path || undefined,
    });
  }
  // GGUF adapters are single files named "*lora*" / "*adapter*" by llama.cpp's converter. Merged models
  // are often named the same way ("…-lora-Q8_0.gguf"), so the file header decides: adapters say
  // general.type = "adapter".
  for (const f of tree.filter((x) => x.path.endsWith(".gguf") && /lora|adapter/i.test(x.path.split("/").pop()!))) {
    const type = await ggufType(repo, f.path);
    if (type && type !== "adapter") continue;
    out.push({ repo, dir: dirOf(f.path), kind: "gguf", file: f.path, files: [{ path: f.path, size: f.size! }], sizeGb: f.size! / 1e9 });
  }
  return out;
}

/** general.type from a GGUF file's header ("model", "adapter", …), read with one small range request.
 *  Undefined when it cannot be read. Format: github.com/ggml-org/ggml/blob/master/docs/gguf.md */
export async function ggufType(repo: string, file: string): Promise<string | undefined> {
  let buf: Buffer;
  try {
    const r = await fetch(raw(repo, file), { headers: { ...hfHeaders(), range: "bytes=0-65535" }, signal: AbortSignal.timeout(15000) });
    if (!r.ok) return undefined;
    buf = Buffer.from(await r.arrayBuffer());
  } catch { return undefined; }
  return ggufHeaderValue(buf, "general.type");
}

/** A string value from the key/value part of a GGUF header (undefined if not in the bytes given). */
export function ggufHeaderValue(buf: Buffer, want: string): string | undefined {
  if (buf.length < 24 || buf.toString("latin1", 0, 4) !== "GGUF") return undefined;
  const SIZES: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
  let o = 24; // magic, version (u32), tensor count (u64), then the key/value count (u64)
  const kvs = Number(buf.readBigUInt64LE(16));
  const str = () => { const n = Number(buf.readBigUInt64LE(o)); o += 8; const v = buf.toString("utf8", o, o + n); o += n; return v; };
  const skip = (t: number): void => {
    if (t === 8) { str(); return; }
    if (t === 9) { const et = buf.readUInt32LE(o); const n = Number(buf.readBigUInt64LE(o + 4)); o += 12; for (let i = 0; i < n; i++) skip(et); return; }
    if (!(t in SIZES)) throw new Error("bad type");
    o += SIZES[t];
  };
  try {
    for (let i = 0; i < kvs && o < buf.length; i++) {
      const key = str();
      const t = buf.readUInt32LE(o); o += 4;
      if (key === want && t === 8) return str();
      skip(t);
    }
  } catch { /* ran past the bytes we have */ }
  return undefined;
}

/** True when the repo is only an adapter (no model weights of its own). */
export async function isAdapterOnly(repo: string, adapters: Adapter[]): Promise<boolean> {
  if (!adapters.length) return false;
  const tree = await hfTree(repo);
  const own = new Set(adapters.flatMap((a) => a.files.map((f) => f.path)));
  return !tree.some((f) => !own.has(f.path) && (/\.(safetensors|gguf)$/.test(f.path) || /(^|\/)model.*\.bin$/.test(f.path)));
}

/** The full-precision model an adapter belongs to. Cards often point one step short: at a quantized
 *  copy ("…-bnb-4bit") or, for converted adapters ("…-F16-GGUF"), at the adapter they came from.
 *  Follow base_model back (up to 3 steps) until it is a model of its own. */
export async function baseModelOf(a: Adapter): Promise<string | undefined> {
  const card = async (repo: string) => {
    const m = await http<any>(`https://huggingface.co/api/models/${repo}`, { headers: hfHeaders(), timeoutMs: 15000 }).catch(() => undefined);
    const b = m?.cardData?.base_model;
    const up = Array.isArray(b) ? b[0] : b;
    return { up: typeof up === "string" && up.includes("/") ? up : undefined, adapter: !!m?.siblings?.some((f: any) => /(^|\/)adapter_config\.json$/.test(f.rfilename)) };
  };
  let base = a.base || (await card(a.repo)).up;
  for (let i = 0; i < 3 && base; i++) {
    const c = await card(base);
    if (!c.adapter && !/bnb|4bit|8bit|gguf|awq|gptq|fp8|mlx/i.test(base)) break;
    if (!c.up) break;
    base = c.up;
  }
  return base && base.includes("/") ? base : undefined;
}

/** vLLM only accepts these values for --max-lora-rank (vllm/config/lora.py, v0.33.0). */
const VLLM_RANKS = [1, 8, 16, 32, 64, 128, 256, 320, 512];
export function vllmRank(r?: number): number {
  if (!r) return 64; // unknown: large enough for most adapters, small memory cost
  return VLLM_RANKS.find((x) => x >= r) || 512;
}

/** Generation reads every weight once per token, so an unmerged adapter slows it by about its
 *  share of the bytes read (adapter / (active weights + adapter)). Small adapters: a few percent. */
export function adapterSpeedFactor(activeWeightsGb: number, adapterGb: number): number {
  return activeWeightsGb > 0 ? activeWeightsGb / (activeWeightsGb + adapterGb) : 1;
}

export function adapterUrl(a: Adapter, p: string): string { return raw(a.repo, p); }

/** Short name apps use as "model" to get the adapter (vLLM / SGLang serve it by name). */
export function adapterName(a: Adapter): string {
  const n = (a.dir ? a.dir.split("/").pop()! : a.file ? a.file.split("/").pop()!.replace(/\.gguf$/, "") : a.repo.split("/")[1]) || "lora";
  return n.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").slice(0, 40).replace(/[-._]+$/, "") || "lora";
}

export function adapterLabel(a: Adapter): string {
  return `${a.repo}${a.dir ? `/${a.dir}` : ""}${a.file && !a.dir ? `/${a.file}` : ""}`;
}
