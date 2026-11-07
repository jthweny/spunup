import type { Recipe, Quant } from "./recipes.js";
import type { Hardware } from "./hardware.js";
import type { RuntimeId, Setup, Where } from "./store.js";
import type { Offer } from "../providers/index.js";
import { IS_LINUX, IS_WIN, IS_MAC } from "./paths.js";
import { computeFit, maxContextThatFits, estimateSpeed, localDevices, calibrate, kvGb, runtimeOverheadGb, tensorParallelGpus, type Fit, type Speed } from "./fit.js";
import { gpuSpec } from "./gpus.js";
import { wslReady } from "../runtimes/install.js";
import { IMAGE_GB } from "../runtimes/index.js";
import { adapterSpeedFactor } from "./lora.js";

/** Each adapted weight adds two small matrix products per token, and on a GPU each has a fixed
 *  launch cost, which follows the device's own fixed cost per token. Measured on an RTX PRO 6000
 *  under Windows (llama.cpp, Qwen2.5-7B Q8, a 0.36 GB adapter on all 7 weights of 28 layers):
 *  150.5 tokens/s with the adapter at scale 0, 88.5–91 with it on, so 4.5 ms more per token, about
 *  23 µs per adapted weight = 0.0135 × that GPU's fixed 1.7 ms (RESEARCH.md §LoRA). */
const LORA_LAUNCH_SHARE = 0.0135;
const ADAPTED_PER_LAYER = 7;   // q, k, v, o, gate, up, down: what most adapters ("all-linear") change

/** Speed with an unmerged LoRA adapter loaded (adapterGb = its file size). gpuOverheadS: the
 *  device's fixed cost per token when the model runs on a GPU (undefined on the CPU, where the
 *  extra bytes read are the cost), gpuShare: the part of the layers on that GPU. */
export function withAdapter(s: Speed, r: Recipe, q: Quant, adapterGb = 0, gpuOverheadS?: number, gpuShare = 1): Speed {
  if (!adapterGb) return s;
  const active = (q.size_gb || 0) * ((r.active_b || r.params_b || 1) / (r.params_b || 1));
  const f = adapterSpeedFactor(active, adapterGb);
  const extraS = gpuOverheadS ? (r.layers || 32) * ADAPTED_PER_LAYER * gpuOverheadS * LORA_LAUNCH_SHARE * gpuShare : 0;
  const slow = (tps: number) => (tps > 0 ? 1 / (1 / (tps * f) + extraS) : tps);
  const g = slow(s.tpsMax) / Math.max(1e-9, s.tpsMax);
  return { ...s, tpsMin: Math.round(s.tpsMin * g * 10) / 10, tpsMax: Math.round(s.tpsMax * g * 10) / 10, prefill: Math.round(s.prefill * f) };
}

export const FORMAT_FOR: Record<RuntimeId, Quant["format"]> = { vllm: "safetensors", sglang: "safetensors", llamacpp: "gguf", mlx: "mlx" };

/** Higher = better quality. A file picked from "More GGUF files…" is labelled with its file name, so its
 *  quant name (Q8_0, UD-Q4_K_XL, IQ3_XXS, BF16…) gives the step instead. */
export function quality(q: Quant): number {
  const n = (q.quant || "").toUpperCase();
  const step = /(?:^|-)I?Q(\d)_/.exec(n)?.[1];
  const fromName = step ? `Q${step}` : /(?:^|-)(BF16|F16)$/.test(n) ? "BF16" : /MXFP4/.test(n) ? "MXFP4" : undefined;
  const l = fromName && !/^(Q\d|IQ\d|UD-|BF16|F16|FP\d|MXFP|NVFP|INT\d|MLX)/i.test(q.label) ? fromName : q.label.toUpperCase();
  const table: [RegExp, number][] = [
    [/^BF16|^FP16|^F16|MLX 16/, 100], [/^FP8|^MXFP8/, 92], [/^INT8/, 90], [/^Q8|MLX 8/, 90], [/^Q6|MLX 6/, 85], [/^Q5|MLX 5/, 80],
    [/^NVFP4|^MXFP4|^FP4|^INT4|^AWQ|^GPTQ|^QAT|^Q4|MLX 4/, 72], [/^Q3|MLX 3/, 60], [/^Q2|MLX 2/, 45], [/^Q1/, 30],
  ];
  for (const [re, v] of table) if (re.test(l)) return v;
  return 50;
}

/** Oldest NVIDIA generation (compute capability) a format runs on in vLLM / SGLang. */
export function minCc(q: Quant): number {
  const l = q.label.toUpperCase();
  if (/NVFP4|^FP4|MXFP8/.test(l)) return 10.0;
  if (/FP8/.test(l)) return 8.0;
  return 7.0;
}

export function quantsFor(r: Recipe, rt: RuntimeId): Quant[] {
  return r.quants.filter((q) => q.format === FORMAT_FOR[rt] && q.size_gb).sort((a, b) => quality(b) - quality(a));
}

export interface RuntimeChoice { id: RuntimeId; ok: boolean; why?: string }

/** Which runtimes can run this model in this place. */
export function runtimeChoices(r: Recipe, where: Where, hw?: Hardware): RuntimeChoice[] {
  const all: RuntimeId[] = ["vllm", "sglang", "llamacpp", "mlx"];
  return all.map((id) => {
    const rr = r.runtimes[id];
    const hasFiles = quantsFor(r, id).length > 0;
    if (!rr?.supported || !hasFiles) return { id, ok: false, why: id === "llamacpp" ? "no GGUF version of this model yet" : id === "mlx" ? "no MLX version of this model yet" : "not supported for this model" };
    if (where !== "local") {
      if (id === "mlx") return { id, ok: false, why: "Apple Silicon only" };
      return { id, ok: true };
    }
    if (id === "mlx") return IS_MAC && hw?.apple ? { id, ok: true } : { id, ok: false, why: "Apple Silicon Macs only" };
    if (id === "vllm" || id === "sglang") {
      const nv = hw?.gpus.some((g) => g.vendor === "nvidia");
      if (IS_MAC) return { id, ok: false, why: "not available on macOS" };
      if (!nv) return { id, ok: false, why: "needs an NVIDIA GPU" };
      if (IS_WIN && !wslReady(hw!)) return { id, ok: true, why: "needs WSL (Linux inside Windows); spunup can set it up" };
      if (IS_LINUX || IS_WIN) return { id, ok: true };
    }
    return { id, ok: true };
  });
}

/** Best runtime for the place (spec 7.1 "Choose best"). */
export function bestRuntime(r: Recipe, where: Where, hw?: Hardware): RuntimeId | undefined {
  const ok = new Set(runtimeChoices(r, where, hw).filter((c) => c.ok && !c.why).map((c) => c.id));
  if (where !== "local") {
    const b = r.setups.gpu.best;
    if (b && ok.has(b)) return b;
    return r.setups.gpu.others.find((x) => ok.has(x)) || [...ok][0];
  }
  if (hw?.apple) {
    const b = r.setups.mac.best;
    if (b && ok.has(b)) return b;
    return ok.has("llamacpp") ? "llamacpp" : ok.has("mlx") ? "mlx" : undefined;
  }
  const gpuBest = r.setups.gpu.best;
  if (gpuBest && (gpuBest === "vllm" || gpuBest === "sglang") && ok.has(gpuBest) && hw) {
    const cc = Number(hw.gpus.find((g) => g.vendor === "nvidia")?.computeCap || 0);
    const fits = quantsFor(r, gpuBest).some((q) => cc >= minCc(q) && computeFit(r, q, Math.min(8192, r.max_context || 8192), hw, gpuBest).setup === "gpu");
    if (fits) return gpuBest;
  }
  return ok.has("llamacpp") ? "llamacpp" : [...ok][0];
}

/** Best quality quant that runs here with at least `ctx` tokens (spec 6.2). */
export function preferredQuant(r: Recipe, rt: RuntimeId, hw: Hardware, ctx: number): { quant: Quant; fit: Fit } | undefined {
  const cc = Number(hw.gpus.find((g) => g.vendor === "nvidia")?.computeCap || 0);
  const cands = quantsFor(r, rt).filter((q) => rt === "llamacpp" || rt === "mlx" || cc >= minCc(q));
  const fits = cands.map((q) => ({ quant: q, fit: computeFit(r, q, ctx, hw, rt) })).filter((x) => x.fit.ok);
  // Prefer: fully on the GPU at Q3 or better; then Q3 or better split with RAM; only then Q2/Q1 (on the GPU first).
  // A Q1 that fits the GPU is worse than a Q4 that spills into RAM: below Q3 answers fall apart.
  const tier = (x: { quant: Quant; fit: Fit }) => (quality(x.quant) >= 60 ? 2 : 0) + (x.fit.setup === "gpu" || x.fit.setup === "mac" ? 1 : 0);
  fits.sort((a, b) => tier(b) - tier(a) || quality(b.quant) - quality(a.quant));
  return fits[0];
}

export function localMaxContext(r: Recipe, rt: RuntimeId, hw: Hardware): { ctx: number; quant?: Quant } {
  const p = preferredQuant(r, rt, hw, 8192) || preferredQuant(r, rt, hw, 2048);
  if (!p) return { ctx: 0 };
  return { ctx: maxContextThatFits(r, p.quant, hw, rt), quant: p.quant };
}

export interface LocalOption { quant: Quant; fit: Fit; speed: Speed }

export function localOptions(r: Recipe, rt: RuntimeId, hw: Hardware, ctx: number, adapterGb = 0): LocalOption[] {
  const cc = Number(hw.gpus.find((g) => g.vendor === "nvidia")?.computeCap || 0);
  const dev = localDevices(hw);
  return quantsFor(r, rt)
    .filter((q) => rt === "llamacpp" || rt === "mlx" || cc >= minCc(q))
    .map((q) => {
      const fit = computeFit(r, q, ctx, hw, rt);
      const share = fit.setup === "cpu" ? 0 : fit.setup === "mixed" ? fit.gpuShare : 1;
      return { quant: q, fit, speed: withAdapter(estimateSpeed(r, q, { ...dev, layerSplit: rt === "llamacpp" }, share, 0.12), r, q, adapterGb, share > 0 ? dev.gpu?.overheadS : undefined, share) };
    });
}

// ------------------------------------------------------------------ cloud
export interface CloudOption { offer: Offer; quant: Quant; speed: Speed; downloadMin: number; downloadCost: number }

export function offerDevice(o: Offer) {
  const s = gpuSpec(o.gpuName);
  const bw = o.gpuMemBw || s?.bw;
  // Vast's total_flops is FP32 TFLOPS; the scout table's tflops are FP16.
  const compute = o.tflops ? { fp32: o.tflops } : s?.fp32 ? { fp32: s.fp32 } : { fp16: s?.tflops };
  return { gpu: calibrate(bw, s?.tg || 0, s?.pp || 0, compute), gpus: o.numGpus, cpuBwGb: 50, pcieGbps: o.pcieGbps || 25 };
}

/** For each machine: the best quant that fits fully in its GPUs at this context. */
export function cloudOptions(r: Recipe, rt: RuntimeId, ctx: number, offers: Offer[], adapterGb = 0): CloudOption[] {
  const out: CloudOption[] = [];
  for (const offer of offers) {
    const best = bestValue(cloudFits(r, rt, ctx, offer, adapterGb));
    if (best) out.push(best);
  }
  return out;
}

/** Every size that fits this machine, best quality first (the price can differ per size on RunPod: disk). */
export function cloudFits(r: Recipe, rt: RuntimeId, ctx: number, offer: Offer, adapterGb = 0): CloudOption[] {
  return quantsFor(r, rt).map((q) => cloudOption(r, rt, ctx, offer, q, adapterGb)).filter((c): c is CloudOption => !!c);
}

/** Below this a chat is too slow to use, so Smart squeeze only goes there when nothing is faster
 *  (e.g. BF16 of a 27B model on one GB10 runs at 1–2 tokens/s and is the cheapest machine it fits). */
export const USABLE_TPS = 10;
const usable = (opts: CloudOption[]) => { const fast = opts.filter((o) => o.speed.tpsMax >= USABLE_TPS); return fast.length ? fast : opts; };

/** The size with the most quality for the price; ties go to higher quality. */
export function bestValue(fits: CloudOption[]): CloudOption | undefined {
  return usable(fits).reduce<CloudOption | undefined>((b, c) => (!b || valueScore(c) > valueScore(b) ? c : b), undefined);
}

/** VRAM spunup lets the model use on this machine, and what is left for the KV cache with this size. */
function offerRoom(r: Recipe, rt: RuntimeId, o: Offer, q: Quant): number {
  const vram = o.vramPerGpuGb * o.numGpus * (rt === "llamacpp" ? 0.97 : 0.9);
  return vram - (q.size_gb || 0) - runtimeOverheadGb(rt, q.size_gb || 0) * Math.max(1, o.numGpus * 0.6);
}

/** Longest context that fits on this machine with this size (capped at the model's maximum, or `cap`). */
export function maxContextOnOffer(r: Recipe, rt: RuntimeId, q: Quant, o: Offer, cap?: number): number {
  const maxM = cap || r.max_context || 32768;
  const kv = r.kv || { mb_per_1k: Math.max(16, (r.params_b || 8) * 8), fixed_mb: 0 };
  const ctx = ((offerRoom(r, rt, o, q) * 1000 - kv.fixed_mb) / kv.mb_per_1k) * 1024;
  return Math.max(0, Math.min(maxM, Math.floor(ctx / 1024) * 1024));
}

/** Why this size can't run on this machine (undefined = it can): its GPUs are too old for the
 *  format, it doesn't fit their memory with this much context, or (Vast) the host's disk is too small. */
export function cloudMiss(r: Recipe, rt: RuntimeId, ctx: number, o: Offer, q: Quant): "format" | "gpus" | "memory" | "disk" | undefined {
  const cc = o.computeCap || gpuSpec(o.gpuName)?.cc || 8.0;
  if ((rt === "vllm" || rt === "sglang") && tensorParallelGpus(o.numGpus) !== o.numGpus) return "gpus";
  if (rt !== "llamacpp" && cc < minCc(q)) return "format";
  if (kvGb(r, ctx) > offerRoom(r, rt, o, q)) return "memory";
  // Vast hosts have a fixed amount of free disk. On RunPod spunup sets the disk size itself.
  if (o.provider !== "runpod" && o.diskGb && diskNeeded(q) > o.diskGb) return "disk";
  return undefined;
}

function cloudOption(r: Recipe, rt: RuntimeId, ctx: number, o: Offer, q: Quant, adapterGb: number): CloudOption | undefined {
  if (cloudMiss(r, rt, ctx, o, q)) return undefined;
  // On RunPod spunup sets the disk size (same formula as launch), and the price follows that size.
  const needDisk = Math.ceil((q.size_gb || 50) * 1.15 + 25);
  if (o.provider === "runpod") {
    o = { ...o, pricePerHour: o.pricePerHour + ((needDisk - (o.diskGb || 0)) * (o.storagePerGbMonth || 0)) / 720, diskGb: needDisk };
  }
  const odev = offerDevice(o);
  const speed = withAdapter(estimateSpeed(r, q, { ...odev, layerSplit: rt === "llamacpp" }, 1, 0.25), r, q, adapterGb, odev.gpu?.overheadS);
  const mbps = o.inetDownMbps || 500;
  // Time: the model plus (if the host does not have it yet) the runtime image.
  // Money: Vast bills the container's own traffic at each host's rate, which is the model; the
  // image pull is not billed (invoice for a 4.6 GB model + vLLM image: "download charge 4.600").
  const image = IMAGE_GB[rt as keyof typeof IMAGE_GB] || 0;
  const downloadMin = Math.max(1, Math.round((((q.size_gb || 0) + adapterGb + image) * 8000) / (mbps * 0.6) / 60));
  const downloadCost = ((q.size_gb || 0) + adapterGb) * (o.downloadPerGb || 0);
  return { offer: o, quant: q, speed, downloadMin, downloadCost };
}

/** Quality on the quality() scale below which answers fall apart (Q2 is 45), so it is worth nothing. */
const USEFUL_QUALITY = 40;

/** Quality for the price: quality above "falls apart", per $/hr. A size twice as good for under
 *  twice the price wins; a much bigger machine for a small step up in quality does not. Sizes at or
 *  below the line (Q1) count as a sliver above it: a negative score would make the dearest machine
 *  look best, as dividing by a bigger price brings a negative number closer to zero. */
export function valueScore(o: CloudOption): number {
  return Math.max(1, quality(o.quant) - USEFUL_QUALITY) / Math.max(0.01, o.offer.pricePerHour);
}

/** The machine and size spunup recommends: most quality for the price; then higher quality; then faster. */
export function recommendedOption(opts: CloudOption[]): CloudOption | undefined {
  return [...usable(opts)].sort((a, b) => valueScore(b) - valueScore(a) || quality(b.quant) - quality(a.quant) || b.speed.tpsMax - a.speed.tpsMax)[0];
}

/** VRAM one size needs at this context, and the disk a Vast host must have free for it. */
export function needsFor(r: Recipe, rt: RuntimeId, ctx: number, q: Quant): { vramGb: number; diskGb: number } {
  // Same share of the GPUs as offerRoom uses, so the search doesn't leave out machines that fit (48 GB cards for llama.cpp).
  return { vramGb: ((q.size_gb || 0) + kvGb(r, ctx) + runtimeOverheadGb(rt, q.size_gb || 0)) / (rt === "llamacpp" ? 0.97 : 0.9), diskGb: diskNeeded(q) };
}

/** Disk a machine needs for this size: the same amount launch asks for. */
export function diskNeeded(q: Quant): number {
  return Math.ceil((q.size_gb || 50) * 1.15 + 25);
}

/** Smallest VRAM any quant needs at this context (for the provider search filter). */
export function minVramNeeded(r: Recipe, rt: RuntimeId, ctx: number): number {
  const qs = quantsFor(r, rt);
  if (!qs.length) return 0;
  return Math.min(...qs.map((q) => needsFor(r, rt, ctx, q).vramGb));
}

export function setupLabel(s: Setup | "none"): string {
  return { gpu: "all on GPU", cpu: "CPU + RAM", mixed: "GPU + RAM mixed", mac: "Apple GPU", none: "does not fit" }[s];
}

export const PRETTY_GPU = (n: string) => n.replace(/NVIDIA |GeForce /g, "").replace(/ PCIe| SXM\d?/i, (m) => m);
