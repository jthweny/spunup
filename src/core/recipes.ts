import fs from "node:fs";
import path from "node:path";
import { P, PKG_ROOT } from "./paths.js";
import { readJson, writeJson, loadConfig, saveConfig, type RuntimeId, type Setup } from "./store.js";
import { http } from "./net.js";
import { kvFromConfig, kvFallback, type KvInfo } from "./kv.js";
import { getKey } from "./keys.js";

/** Where updated recipes are published. Change this to the project's repo before publishing
 *  (see RESEARCH.md, open question 1). The tool works offline with the shipped recipes. */
export const DEFAULT_RECIPES_URL = "https://raw.githubusercontent.com/jthweny/spunup/main/recipes";

export interface Quant {
  id: string;
  label: string;              // "BF16", "FP8", "Q4", "MLX 4-bit" ...
  format: "safetensors" | "gguf" | "mlx";
  repo: string;
  file?: string;              // first GGUF file (split files: part 1)
  parts?: number;
  quant?: string;             // exact GGUF quant name, e.g. UD-Q4_K_XL
  size_gb: number | null;
  bits?: number | null;
  runtimes: RuntimeId[];
  extra_args?: string[];
  gated?: boolean | string | null;
}

export interface RuntimeRecipe {
  supported: boolean;
  args: string[];
  env?: Record<string, string>;
  tool_parser?: string | null;
  reasoning_parser?: string | null;
  reasoning_plugin?: string | null;
  chat_template?: string | null;
  min_version?: string | null;
  nightly?: boolean;
  image?: string | null;
  hardware?: Record<string, { args: string[]; env: Record<string, string> }>;
  deps?: string[];
  lora?: { supported?: boolean; args: string[]; note?: string };
  caching?: string;
  long_context?: { args: string[]; env: Record<string, string> } | null;
  repo?: string | null;
  repos?: string[];
  note?: string;
}

export interface SetupRecipe { best: RuntimeId | null; others: RuntimeId[]; command?: string | null; note?: string }

export interface Recipe {
  schema: number;
  id: string;
  name: string;
  family: string | null;
  popular: boolean;
  checked: string;
  architecture?: string | null;
  model_type?: string | null;
  moe: boolean;
  params_b: number | null;
  active_b: number | null;
  layers?: number | null;
  experts?: number | null;
  max_context: number | null;
  max_context_extended?: number | null;
  gated: boolean;
  license?: string | null;
  multimodal?: boolean;
  kv: { mb_per_1k: number; fixed_mb: number; method: string; estimate: boolean } | null;
  quants: Quant[];
  runtimes: Record<RuntimeId, RuntimeRecipe>;
  setups: Record<Setup, SetupRecipe>;
  known_issues: string[];
  benchmarks: { gpu: string; quant: string; tps: number; prefill?: number; source: string }[];
  sources: string[];
  layer: "user" | "model" | "family" | "generic";   // which layer the settings came from
}

export interface Family {
  schema: number;
  id: string;
  name: string;
  hf_orgs: string[];
  model_types: string[];
  checked: string;
  runtimes: Record<RuntimeId, RuntimeRecipe>;
  setups: Record<Setup, SetupRecipe>;
  notes: string[];
  members: string[];
  sources: string[];
  gguf_publishers: string[];
  mlx_publishers: string[];
}

export interface IndexEntry {
  id: string; family: string; params_b: number | null; active_b: number | null; moe: boolean;
  max_context: number | null; gated: boolean; popular: boolean; file: string; isNew?: boolean; downloads?: number;
}
interface Index { schema: number; checked: string; families: string[]; family_names?: Record<string, string>; models: IndexEntry[] }

function recipeRoots(): string[] {
  // Downloaded updates first (if newer), then the copy that ships with the tool.
  const shipped = path.join(PKG_ROOT, "recipes");
  const cached = P.recipesCache;
  const a = readJson<Index | null>(path.join(cached, "index.json"), null);
  const b = readJson<Index | null>(path.join(shipped, "index.json"), null);
  if (a && (!b || a.checked >= b.checked)) return [cached, shipped];
  return [shipped, cached];
}

function readRecipeFile<T>(rel: string): T | undefined {
  for (const root of recipeRoots()) {
    const f = path.join(root, rel);
    if (fs.existsSync(f)) return readJson<T | undefined>(f, undefined);
  }
  return undefined;
}

export function loadIndex(): Index {
  return readRecipeFile<Index>("index.json") || { schema: 1, checked: "", families: [], models: [] };
}

export function loadFamily(id: string): Family | undefined {
  return readRecipeFile<Family>(`families/${id}.json`);
}

export function allFamilies(): Family[] {
  return loadIndex().families.map(loadFamily).filter(Boolean) as Family[];
}

export function loadModelRecipe(id: string): Recipe | undefined {
  const r = readRecipeFile<Recipe>(`models/${id.replace("/", "__")}.json`);
  if (r) r.layer = "model";
  return r;
}

/** Downloads updated recipes from the project's repo, at most once a day. Quiet on failure. */
export async function updateRecipes(force = false): Promise<boolean> {
  const cfg = loadConfig();
  if (!force && cfg.lastRecipeUpdate && Date.now() - cfg.lastRecipeUpdate < 24 * 3600_000) return false;
  const base = cfg.recipesUrl || DEFAULT_RECIPES_URL;
  try {
    const idx = await http<Index>(`${base}/index.json`, { timeoutMs: 8000 });
    const shipped = readJson<Index | null>(path.join(PKG_ROOT, "recipes", "index.json"), null);
    cfg.lastRecipeUpdate = Date.now();
    saveConfig(cfg);
    if (!idx?.models || (shipped && idx.checked <= shipped.checked)) return false;
    fs.mkdirSync(path.join(P.recipesCache, "models"), { recursive: true });
    fs.mkdirSync(path.join(P.recipesCache, "families"), { recursive: true });
    for (const f of idx.families) {
      const fam = await http(`${base}/families/${f}.json`, { timeoutMs: 8000 });
      writeJson(path.join(P.recipesCache, "families", `${f}.json`), fam, 0o644);
    }
    for (const m of idx.models) {
      const r = await http(`${base}/${m.file}`, { timeoutMs: 8000 });
      writeJson(path.join(P.recipesCache, m.file), r, 0o644);
    }
    writeJson(path.join(P.recipesCache, "index.json"), idx, 0o644);
    return true;
  } catch {
    return false;
  }
}

// ------------------------------------------------------------------ Hugging Face helpers
function hfHeaders(): Record<string, string> {
  const t = getKey("hf");
  return t ? { authorization: `Bearer ${t}` } : {};
}

export async function hfModelInfo(id: string): Promise<any> {
  return http(`https://huggingface.co/api/models/${id}?expand[]=safetensors&expand[]=gated&expand[]=config&expand[]=cardData&expand[]=downloads&expand[]=createdAt`, { headers: hfHeaders(), timeoutMs: 15000 });
}

export async function hfConfig(id: string): Promise<any | undefined> {
  try {
    return await http(`https://huggingface.co/${id}/resolve/main/config.json`, { headers: hfHeaders(), timeoutMs: 15000 });
  } catch {
    return undefined;
  }
}

/** The model's chat template (tokenizer_config.json, or chat_template.jinja in newer repos). */
async function hfChatTemplate(id: string): Promise<string | undefined> {
  const tc = await http<any>(`https://huggingface.co/${id}/resolve/main/tokenizer_config.json`, { headers: hfHeaders(), timeoutMs: 15000 }).catch(() => undefined);
  const t = tc?.chat_template;
  if (typeof t === "string") return t;
  if (Array.isArray(t)) return t.map((x: any) => x?.template || "").join("\n");
  const r = await fetch(`https://huggingface.co/${id}/resolve/main/chat_template.jinja`, { headers: hfHeaders(), signal: AbortSignal.timeout(15000) }).catch(() => undefined);
  return r?.ok ? r.text() : undefined;
}

export async function hfTree(id: string, opts: { strict?: boolean } = {}): Promise<{ path: string; size: number }[]> {
  let last: unknown;
  for (let i = 0; i < 3; i++) {
    try {
      return await http(`https://huggingface.co/api/models/${id}/tree/main?recursive=true`, { headers: hfHeaders(), timeoutMs: 30000 });
    } catch (e) {
      last = e;
      if ((e as any).status === 401 || (e as any).status === 403 || (e as any).status === 404) break;
      await new Promise((r) => setTimeout(r, 1500 * (i + 1)));
    }
  }
  if (opts.strict) throw new Error(`Could not list the files of ${id} on Hugging Face (${(last as Error)?.message}).`);
  return [];
}

/** "https://huggingface.co/org/name/tree/main" -> "org/name" */
export function parseHfId(input: string): string | undefined {
  const s = input.trim().replace(/^https?:\/\/(www\.)?(huggingface\.co|hf\.co)\//, "").replace(/\/(tree|blob|resolve)\/.*$/, "").replace(/\/$/, "");
  return /^[\w.-]+\/[\w.-]+$/.test(s) ? s : undefined;
}

export const QRE = /(?<![A-Za-z0-9])(UD-)?(IQ[1-4]_[A-Z]+|Q[1-8]_K(_[A-Z]+)?|Q[1-8]_[01]|MXFP4(_MOE)?|BF16|F16|TQ[12]_0)(?![A-Za-z0-9])/i;
const SIDE = /mmproj|mtp|draft|assistant|dflash|dspark|eagle|imatrix/i;
const LADDER: [string, string[]][] = [
  ["Q8", ["Q8_0", "UD-Q8_K_XL"]], ["Q6", ["Q6_K", "UD-Q6_K_XL"]], ["Q5", ["Q5_K_M", "UD-Q5_K_XL"]],
  ["Q4", ["Q4_K_M", "UD-Q4_K_XL", "Q4_K", "MXFP4_MOE", "MXFP4", "Q4_0", "Q4_K_S"]], ["Q3", ["Q3_K_M", "UD-Q3_K_XL", "IQ3_M", "UD-IQ3_S", "IQ3_S", "UD-IQ3_XXS", "IQ3_XXS"]],
  ["Q2", ["UD-Q2_K_XL", "Q2_K_L", "Q2_K", "UD-IQ2_M", "IQ2_M", "UD-IQ2_XXS", "IQ2_XXS"]], ["Q1", ["UD-IQ1_M", "UD-IQ1_S", "IQ1_M"]],
];

/** MLX repos say their precision in the name (…-4bit, …-q4-mlx, …-mxfp4, …-bf16); otherwise use bits per weight.
 *  Same rule as research/scripts/build_recipes.py. */
export function mlxLabel(repo: string, sizeGb: number, paramsB: number | null): string {
  const n = repo.toLowerCase();
  if (/mxfp4|nvfp4/.test(n)) return "MLX 4-bit"; // before "-q8": gpt-oss "MXFP4-Q8" is 4-bit weights
  const m = /(\d+)[-_]?bit/.exec(n) || /[-_]q(\d)(?!\d)/.exec(n);
  if (m) return `MLX ${m[1]}-bit`;
  if (/bf16|fp16|f16/.test(n)) return "MLX 16-bit";
  if (sizeGb && paramsB) { const b = (sizeGb * 8) / paramsB; return b >= 12 ? "MLX 16-bit" : b >= 6.5 ? "MLX 8-bit" : b >= 5.5 ? "MLX 6-bit" : "MLX 4-bit"; }
  return "MLX";
}

/** List GGUF quants in a repo, grouped (split files summed). */
export async function ggufQuants(repo: string, paramsB: number | null): Promise<Quant[]> {
  const files = new Map<string, { size: number; files: string[] }>();
  for (const f of await hfTree(repo)) {
    if (!f.path.endsWith(".gguf")) continue;
    const base = f.path.split("/").pop()!;
    if (SIDE.test(base)) continue;
    const m = QRE.exec(base) || QRE.exec(f.path);
    if (!m) continue;
    const q = m[0].toUpperCase();
    const e = files.get(q) || { size: 0, files: [] };
    e.size += f.size; e.files.push(f.path); files.set(q, e);
  }
  const out: Quant[] = [];
  // Per step, the first file that is really smaller (>7%) than the step above (same rule as the recipe
  // builder): a "Q4" as big as the Q8 is no use, and the step's next files can hold the real middle size.
  for (const [label, names] of LADDER) {
    const n = names.find((x) => files.has(x) && (!out.length || files.get(x)!.size / 1e9 < out[out.length - 1].size_gb! * 0.93));
    if (!n) continue;
    const f = files.get(n)!;
    f.files.sort();
    out.push({ id: `gguf:${n}`, label, format: "gguf", repo, file: f.files[0], parts: f.files.length, quant: n,
      size_gb: Math.round(f.size / 1e8) / 10, bits: paramsB ? Math.round((f.size * 8) / (paramsB * 1e9) * 100) / 100 : null, runtimes: ["llamacpp"] });
  }
  return out;
}

/** Words a GGUF / MLX repo adds after the model's own name. */
const QUANT_WORD = /^(gguf|mlx|ud|dwq|dynamic|imatrix|i?q\d\w*|\d+-?bits?|bits?|bf16|fp16|f16|fp8|mxfp4|nvfp4|int[48]|awq|gptq|quantized|quants?)$/i;

async function findRepos(name: string, filter: "gguf" | "mlx", publishers: string[]): Promise<string[]> {
  try {
    const r = await http<any[]>(`https://huggingface.co/api/models?search=${encodeURIComponent(name)}&filter=${filter}&sort=downloads&limit=20`, { timeoutMs: 15000 });
    const low = name.toLowerCase();
    return r.map((m) => m.id as string).filter((id) => {
      const [org, n] = id.split("/");
      // The same model, only with format / quant words after its name (Qwen3-4B-GGUF, Qwen3-4B-MLX-4bit), not
      // another variant that starts the same (Qwen3-4B-Instruct-2507, Qwen3-4B-Thinking-2507).
      const rest = n.toLowerCase().replace(/^[^_]+_/, "");
      if (!rest.startsWith(low)) return false;
      const extra = rest.slice(low.length).split(/[-_.]/).filter(Boolean);
      return publishers.includes(org) && extra.every((w) => QUANT_WORD.test(w)) && !/abliter|uncensor|heretic|reap|distill/i.test(n);
    }).sort((a, b) => publishers.indexOf(a.split("/")[0]) - publishers.indexOf(b.split("/")[0]));
  } catch {
    return [];
  }
}

function familyFor(id: string, modelType: string | undefined, families: Family[]): Family | undefined {
  const org = id.split("/")[0];
  return families.find((f) => f.hf_orgs.includes(org)) || families.find((f) => modelType && f.model_types.includes(modelType));
}

/** Resolve any Hugging Face model into a Recipe: model recipe, else family recipe, else generic fallback. */
export async function resolveRecipe(id: string): Promise<Recipe> {
  const r = loadModelRecipe(id);
  if (r) return r;
  const [info, cfg, template] = await Promise.all([hfModelInfo(id), hfConfig(id), hfChatTemplate(id)]);
  const families = allFamilies();
  const modelType: string | undefined = cfg?.model_type || info?.config?.model_type;
  const fam = familyFor(id, modelType, families);
  const st = info?.safetensors?.total as number | undefined;
  const paramsB = st ? Math.round(st / 1e8) / 10 : guessParams(id);
  const tc = cfg?.text_config || {};
  const experts = tc.num_experts || tc.n_routed_experts || cfg?.num_experts || cfg?.n_routed_experts || cfg?.num_local_experts || null;
  const actMatch = /-A(\d+(?:\.\d+)?)B/i.exec(id);
  const activeB = experts ? (actMatch ? Number(actMatch[1]) : paramsB ? Math.round(paramsB * 0.1 * 10) / 10 : null) : paramsB;
  const kv: KvInfo = kvFromConfig(cfg) || kvFallback(paramsB || 8);
  const maxCtx = tc.max_position_embeddings || cfg?.max_position_embeddings || 32768;
  const name = id.split("/")[1];
  const ggPubs = fam?.gguf_publishers || ["ggml-org", "unsloth", "bartowski", "lmstudio-community", id.split("/")[0]];
  const mlxPubs = fam?.mlx_publishers || ["mlx-community", "lmstudio-community", id.split("/")[0]];
  const [ggRepos, mlxRepos] = await Promise.all([findRepos(name, "gguf", ggPubs), findRepos(name, "mlx", mlxPubs)]);
  const quants: Quant[] = [];
  const tree = await hfTree(id);
  // Some repos (Mistral) ship the same weights twice: consolidated.safetensors and the sharded model-0000x files.
  const stFiles = tree.filter((f) => f.path.endsWith(".safetensors") && !f.path.includes("/"));
  const stBytes = (stFiles.some((f) => /^model-\d+-of-\d+\.safetensors$/.test(f.path)) ? stFiles.filter((f) => !/^consolidated/.test(f.path)) : stFiles).reduce((a, f) => a + f.size, 0);
  const isGguf = tree.some((f) => f.path.endsWith(".gguf"));
  if (stBytes) quants.push({ id: `bf16:${id}`, label: cfg?.quantization_config?.quant_method ? String(cfg.quantization_config.quant_method).toUpperCase() : "BF16", format: "safetensors", repo: id, size_gb: Math.round(stBytes / 1e8) / 10, runtimes: ["vllm", "sglang"] });
  if (isGguf) quants.push(...(await ggufQuants(id, paramsB)));
  else if (ggRepos[0]) quants.push(...(await ggufQuants(ggRepos[0], paramsB)));
  for (const m of mlxRepos.slice(0, 2)) {
    const t = await hfTree(m);
    const b = t.filter((f) => f.path.endsWith(".safetensors")).reduce((a, f) => a + f.size, 0);
    quants.push({ id: `mlx:${m}`, label: mlxLabel(m, b / 1e9, paramsB), format: "mlx", repo: m, size_gb: b ? Math.round(b / 1e8) / 10 : null, runtimes: ["mlx"] });
  }
  const hasGguf = quants.some((q) => q.format === "gguf");
  const hasMlx = quants.some((q) => q.format === "mlx");
  const generic = genericRuntimes();
  const runtimes = fam ? structuredClone(fam.runtimes) : generic;
  runtimes.llamacpp.supported = hasGguf;
  runtimes.mlx.supported = hasMlx;
  runtimes.vllm.supported = !!stBytes && !!fam;
  runtimes.sglang.supported = !!stBytes;
  if (ggRepos[0] || isGguf) runtimes.llamacpp.repo = isGguf ? id : ggRepos[0];
  const gpuBest: RuntimeId | null = runtimes.vllm.supported ? "vllm" : runtimes.sglang.supported ? "sglang" : hasGguf ? "llamacpp" : null;
  // Family notes are written for the family's current models. An older member matched by its
  // publisher (Qwen2.5 in the Qwen family) may have no thinking mode: keep thinking notes only when
  // the chat template has one (or could not be read).
  const thinks = template === undefined || /enable_thinking|<think>|reasoning/i.test(template);
  const issues = (fam?.notes || []).filter((n) => thinks || !/thinking/i.test(n));
  if (!fam) issues.push("Unknown model family: spunup uses safe general settings and lets the runtime detect tool calling from the chat template.");
  if (info?.gated) issues.push("Gated on Hugging Face: accept the license on the model page, then use a Read token.");
  return {
    schema: 1, id, name, family: fam?.id || null, popular: false, checked: new Date().toISOString().slice(0, 10),
    architecture: cfg?.architectures?.[0] || null, model_type: modelType || null, moe: !!experts,
    params_b: paramsB, active_b: activeB, layers: tc.num_hidden_layers || cfg?.num_hidden_layers || null, experts,
    max_context: maxCtx, gated: !!info?.gated, license: info?.cardData?.license || null,
    kv: { mb_per_1k: kv.mbPer1k, fixed_mb: kv.fixedMb, method: kv.method, estimate: kv.estimate },
    quants, runtimes,
    setups: {
      gpu: { best: gpuBest, others: (["vllm", "sglang", "llamacpp"] as RuntimeId[]).filter((x) => x !== gpuBest && runtimes[x].supported) },
      cpu: { best: hasGguf ? "llamacpp" : null, others: [] },
      mixed: { best: hasGguf ? "llamacpp" : null, others: [] },
      mac: { best: hasGguf ? "llamacpp" : hasMlx ? "mlx" : null, others: hasGguf && hasMlx ? ["mlx"] : [] },
    },
    known_issues: issues, benchmarks: [], sources: [`https://huggingface.co/${id}`, ...(fam?.sources || [])],
    layer: fam ? "family" : "generic",
  };
}

/** Generic fallback: the runtimes work out the tool format from the chat template themselves. */
function genericRuntimes(): Record<RuntimeId, RuntimeRecipe> {
  return {
    vllm: { supported: false, args: [], tool_parser: "hermes", reasoning_parser: null, caching: "Prefix caching is on by default." },
    sglang: { supported: true, args: [], tool_parser: "auto", reasoning_parser: "auto", caching: "Radix cache is on by default.",
      lora: { args: ["--enable-lora", "--lora-paths", "{lora_name}={lora_path}"] } },
    llamacpp: { supported: true, args: ["--jinja", "--reasoning-format", "deepseek"], caching: "--cache-prompt on by default.",
      lora: { args: ["--lora", "{lora_path}"] } },
    mlx: { supported: true, args: [], caching: "Prompt cache built in.", lora: { args: ["--adapter-path", "{lora_path}"] } },
  };
}

function guessParams(id: string): number | null {
  const m = /(\d+(?:\.\d+)?)\s*([BT])(?![a-z])/i.exec(id.split("/")[1]);
  if (!m) return null;
  return Number(m[1]) * (m[2].toUpperCase() === "T" ? 1000 : 1);
}

// ------------------------------------------------------------------ new releases (spec 5.3)
interface NewCache { at: number; models: IndexEntry[] }
const NEW_FILE = () => path.join(P.recipesCache, "new-models.json");

/** New models from known families' Hugging Face orgs, not yet in the recipes. Checked at most daily. */
export async function checkNewModels(): Promise<IndexEntry[]> {
  const cache = readJson<NewCache>(NEW_FILE(), { at: 0, models: [] });
  if (Date.now() - cache.at < 24 * 3600_000) return cache.models;
  const idx = loadIndex();
  const known = new Set(idx.models.map((m) => m.id));
  const out: IndexEntry[] = [];
  const since = idx.checked || "2026-01-01";
  for (const fam of allFamilies()) {
    for (const org of fam.hf_orgs) {
      try {
        const list = await http<any[]>(`https://huggingface.co/api/models?author=${org}&sort=createdAt&direction=-1&limit=15&expand[]=createdAt&expand[]=pipeline_tag&expand[]=downloads`, { timeoutMs: 10000 });
        for (const m of list) {
          if (known.has(m.id) || (m.createdAt || "") < since) continue;
          if (!["text-generation", "image-text-to-text", "any-to-any"].includes(m.pipeline_tag)) continue;
          if (/GGUF|MLX|AWQ|GPTQ|FP8|NVFP4|-Base|DSpark|DFlash|Eagle/i.test(m.id)) continue;
          out.push({ id: m.id, family: fam.id, params_b: guessParams(m.id), active_b: null, moe: /-A\d/.test(m.id), max_context: null, gated: false, popular: false, file: "", isNew: true });
        }
      } catch { /* offline: skip */ }
    }
  }
  writeJson(NEW_FILE(), { at: Date.now(), models: out }, 0o644);
  return out;
}
