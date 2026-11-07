import type { Recipe, Quant } from "../core/recipes.js";
import type { RuntimeId, Setup } from "../core/store.js";

/** Versions spunup installs / runs. Checked 2026-11-06 (RESEARCH.md §Runtimes). */
export const VERSIONS = {
  vllm: "0.33.0",
  sglang: "0.5.23",
  llamacppBuild: "b12110",       // build behind llama.cpp release v0.9.0
  mlxLm: "0.32.2",
  uv: "0.13.7",
};

export const IMAGES = {
  vllm: `vllm/vllm-openai:v${VERSIONS.vllm}`,
  vllmNightly: "vllm/vllm-openai:nightly",
  sglang: `lmsysorg/sglang:v${VERSIONS.sglang}`,
  llamacppCuda12: `ghcr.io/ggml-org/llama.cpp:server-cuda-${VERSIONS.llamacppBuild}`,
  llamacppCuda13: `ghcr.io/ggml-org/llama.cpp:server-cuda13-${VERSIONS.llamacppBuild}`,
};

/** Compressed linux/amd64 image sizes (GB), read from the registries' manifests on 2026-11-06.
 *  On Vast.ai the image pull counts as downloaded bandwidth unless the host already has it cached. */
export const IMAGE_GB: Record<"vllm" | "sglang" | "llamacpp", number> = { vllm: 9.74, sglang: 16.11, llamacpp: 2.6 };

export const RUNTIME_NAMES: Record<RuntimeId, string> = {
  vllm: "vLLM",
  sglang: "SGLang",
  llamacpp: "llama.cpp",
  mlx: "MLX",
};

export const RUNTIME_BLURB: Record<RuntimeId, string> = {
  vllm: "fastest on NVIDIA GPUs, Linux only",
  sglang: "fast on NVIDIA GPUs, Linux only",
  llamacpp: "runs anywhere: GPU, CPU, or both together",
  mlx: "made for Apple Silicon Macs",
};

/** OpenAI-compatible paths every runtime serves (checked in each runtime's docs). */
export const OPENAI_PATHS = ["/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/embeddings"];

export interface LaunchOptions {
  modelPath: string;          // HF repo id or local path (GGUF file for llama.cpp)
  servedName: string;
  host: string;               // always 127.0.0.1
  port: number;
  apiKey: string;
  context: number;
  gpus: number;
  setup: Setup;
  gpuLayers?: string;         // llama.cpp -ngl value
  cpuMoeLayers?: number;      // llama.cpp --n-cpu-moe
  chatTemplatePath?: string;  // downloaded template file
  pluginPath?: string;        // vLLM reasoning parser plugin file
  gpuGen?: "hopper" | "blackwell" | "ampere" | "ada" | "other";
  lora?: { name: string; path: string; rank?: number };
  extraArgs?: string[];       // Advanced: user's own flags (win over recipe)
  overrideArgs?: string[];    // Advanced: replace recipe args entirely
}

export interface Launch { bin: string; args: string[]; env: Record<string, string> }

function fill(args: string[], o: LaunchOptions): string[] {
  return args.map((a) => a
    .replace("{lora_name}", o.lora?.name || "")
    .replace("{lora_path}", o.lora?.path || "")
    .replace("{lora_rank}", String(o.lora?.rank || 64)));
}

/** Merge user flags over recipe flags: a user flag replaces the recipe flag with the same name. */
export function mergeArgs(base: string[], user: string[]): string[] {
  const userFlags = new Set(user.filter((a) => a.startsWith("--") || /^-[a-zA-Z]+$/.test(a)).map((a) => a.split("=")[0]));
  const out: string[] = [];
  for (let i = 0; i < base.length; i++) {
    const a = base[i];
    const flag = a.split("=")[0];
    if ((a.startsWith("-")) && userFlags.has(flag)) {
      if (!a.includes("=") && base[i + 1] !== undefined && !base[i + 1].startsWith("-")) i++;
      continue;
    }
    out.push(a);
  }
  return [...out, ...user];
}

/** Flags spunup sets itself. A recipe's own copy would win (the last one counts) and break the endpoint
 *  (its name, its context, the GPU count of the machine actually rented), so recipe copies are dropped. */
const OWNED: Record<RuntimeId, string[]> = {
  vllm: ["--host", "--port", "--api-key", "--served-model-name", "--max-model-len", "--tensor-parallel-size", "-tp"],
  sglang: ["--host", "--port", "--api-key", "--served-model-name", "--context-length", "--tp-size", "--tp", "--model-path"],
  llamacpp: ["--host", "--port", "--api-key", "--alias", "-c", "--ctx-size", "-m", "--model"],
  mlx: ["--host", "--port", "--model"],
};
function dropOwned(args: string[], rt: RuntimeId): string[] {
  // "--max-model-len auto" written as one argument is two.
  const flat = args.flatMap((a) => (a.startsWith("-") && /\s/.test(a.trim()) ? a.trim().split(/\s+/) : [a]));
  const out: string[] = [];
  for (let i = 0; i < flat.length; i++) {
    if (OWNED[rt].includes(flat[i].split("=")[0])) {
      if (!flat[i].includes("=") && flat[i + 1] !== undefined && !flat[i + 1].startsWith("-")) i++;
      continue;
    }
    out.push(flat[i]);
  }
  return out;
}

export function buildLaunch(rt: RuntimeId, r: Recipe, q: Quant, o: LaunchOptions): Launch {
  const rr = r.runtimes[rt];
  let recipeArgs = [...(rr.args || []), ...(q.extra_args || [])];
  const env: Record<string, string> = { ...(rr.env || {}) };
  const hwo = o.gpuGen && rr.hardware?.[o.gpuGen];
  if (hwo) { recipeArgs = mergeArgs(recipeArgs, hwo.args); Object.assign(env, hwo.env); }
  recipeArgs = dropOwned(recipeArgs, rt);
  if (o.overrideArgs) recipeArgs = o.overrideArgs;
  let args: string[];
  let bin: string;
  if (rt === "vllm") {
    bin = "vllm";
    args = ["serve", o.modelPath, "--host", o.host, "--port", String(o.port), "--api-key", o.apiKey,
      "--served-model-name", o.servedName, "--max-model-len", String(o.context), "--gpu-memory-utilization", "0.90"];
    if (o.gpus > 1) args.push("--tensor-parallel-size", String(o.gpus));
    args.push(...recipeArgs);
    if (rr.tool_parser) args.push("--enable-auto-tool-choice", "--tool-call-parser", rr.tool_parser);
    if (rr.reasoning_parser) {
      if (rr.reasoning_plugin && o.pluginPath) args.push("--reasoning-parser-plugin", o.pluginPath);
      args.push("--reasoning-parser", rr.reasoning_parser);
    }
    if (rr.chat_template && o.chatTemplatePath) args.push("--chat-template", o.chatTemplatePath);
    if (o.context > (r.max_context || 0) && rr.long_context) {
      args = mergeArgs(args, rr.long_context.args.filter((_, i, a) => a[i - 1] !== "--max-model-len" && _ !== "--max-model-len"));
      Object.assign(env, rr.long_context.env);
    }
    if (o.lora && rr.lora) args.push(...fill(rr.lora.args, o));
  } else if (rt === "sglang") {
    bin = "python3";
    args = ["-m", "sglang.launch_server", "--model-path", o.modelPath, "--host", o.host, "--port", String(o.port),
      "--api-key", o.apiKey, "--served-model-name", o.servedName, "--context-length", String(o.context)];
    if (o.gpus > 1) args.push("--tp-size", String(o.gpus));
    args.push(...(o.overrideArgs ? recipeArgs : (r.runtimes.sglang.args || [])));
    if (rr.tool_parser) args.push("--tool-call-parser", rr.tool_parser);
    if (rr.reasoning_parser) args.push("--reasoning-parser", rr.reasoning_parser);
    if (o.lora && rr.lora) args.push(...fill(rr.lora.args, o));
  } else if (rt === "llamacpp") {
    bin = "llama-server";
    args = ["-m", o.modelPath, "--host", o.host, "--port", String(o.port), "--api-key", o.apiKey,
      "--alias", o.servedName, "-c", String(o.context), "-np", "1", "--cache-reuse", "256", "--fit", "on",
      "--metrics", ...(o.overrideArgs ? [] : dropOwned(rr.args || ["--jinja"], rt))];
    if (o.overrideArgs) args.push(...o.overrideArgs);
    if (o.setup === "cpu") args.push("-ngl", "0");
    else if (o.setup === "gpu" || o.setup === "mac") args.push("-ngl", "all");
    else if (o.setup === "mixed") {
      // Let llama.cpp's --fit place layers, but tell it to put MoE experts in RAM first.
      if (r.moe && o.cpuMoeLayers) args.push("-ngl", "all", "--n-cpu-moe", String(o.cpuMoeLayers));
    }
    if (o.chatTemplatePath) args.push("--chat-template-file", o.chatTemplatePath);
    if (o.lora) args.push(...fill(rr.lora?.args || ["--lora", "{lora_path}"], o));
  } else {
    bin = "mlx_lm.server";
    args = ["--model", o.modelPath, "--host", o.host, "--port", String(o.port), "--trust-remote-code"];
    if (o.lora) args.push(...fill(rr.lora?.args || ["--adapter-path", "{lora_path}"], o));
    if (o.overrideArgs) args.push(...o.overrideArgs);
  }
  if (o.extraArgs?.length) args = mergeArgs(args, o.extraArgs);
  return { bin, args, env };
}

export function shellQuote(a: string): string {
  return /^[\w@%+=:,./-]+$/.test(a) ? a : `'${a.replace(/'/g, `'\\''`)}'`;
}

export function commandString(l: Launch): string {
  const env = Object.entries(l.env).map(([k, v]) => `${k}=${shellQuote(v)}`).join(" ");
  return `${env ? env + " " : ""}${l.bin} ${l.args.map(shellQuote).join(" ")}`;
}

/** Hide the API key when a command is shown on screen. */
export function redact(cmd: string, key: string): string {
  return key ? cmd.split(key).join("sk-spunup-••••••••") : cmd;
}

/** How many MoE layers' experts go to RAM so the rest fits on the GPU. */
export function cpuMoeLayers(r: Recipe, q: Quant, gpuShare: number): number {
  const L = r.layers || 48;
  return Math.min(L, Math.max(1, Math.ceil(L * (1 - gpuShare) + 1)));
}
