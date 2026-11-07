import type { Recipe, Quant } from "./recipes.js";
import type { Hardware } from "./hardware.js";
import type { RuntimeId, Setup } from "./store.js";
import { totalVram } from "./hardware.js";
import { gpuSpec, appleSpec, cpuBw, BENCH_MODEL_GB, BENCH_ACTIVE_B, GPU_SPECS } from "./gpus.js";

/** Memory the runtime needs on top of weights + KV (GB): CUDA context, compute buffers, graphs. */
export function runtimeOverheadGb(rt: RuntimeId, weightsGb: number): number {
  if (rt === "vllm" || rt === "sglang") return 2 + weightsGb * 0.03;
  if (rt === "mlx") return 1 + weightsGb * 0.02;
  return 0.8 + weightsGb * 0.02; // llama.cpp compute buffers
}

export function kvGb(r: Recipe, ctx: number): number {
  const kv = r.kv || { mb_per_1k: Math.max(16, (r.params_b || 8) * 8), fixed_mb: 0 };
  return (kv.mb_per_1k * (ctx / 1024) + kv.fixed_mb) / 1000;
}

export interface Budget { vram: number; ram: number }

/** Memory spunup lets a model use on this machine. */
/** GPUs vLLM / SGLang split a model across: 1, 2, 4 or 8 (the count has to divide the model's attention
 *  heads evenly, which these do for the usual 32/40/64/… heads; 3 or 6 make them refuse to start). */
export function tensorParallelGpus(n: number): number {
  return n >= 8 ? 8 : n >= 4 ? 4 : n >= 2 ? 2 : 1;
}

export function budget(hw: Hardware, rt: RuntimeId): Budget {
  // vLLM / SGLang give every GPU an equal share, so the smallest card sets the room on each.
  const nv = hw.gpus.filter((g) => g.vendor === "nvidia");
  const vramRaw = (rt === "vllm" || rt === "sglang") && nv.length > 1
    ? Math.min(...nv.map((g) => g.vramGb)) * tensorParallelGpus(nv.length) : totalVram(hw);
  const vram = hw.apple ? vramRaw : rt === "vllm" ? vramRaw * 0.9 : rt === "sglang" ? vramRaw * 0.88 : Math.max(0, vramRaw - 0.6);
  const ram = Math.max(0, hw.ramGb - Math.max(4, hw.ramGb * 0.15));
  return { vram, ram: hw.apple ? 0 : ram };
}

export interface Fit {
  setup: Setup | "none";
  totalGb: number;
  weightsGb: number;
  kvGb: number;
  gpuGb: number;           // part on the GPU
  ramGb: number;           // part in system RAM
  gpuShare: number;        // 0..1 of weights on GPU
  ok: boolean;
  slow: boolean;           // fits, but mostly in RAM
  reason?: string;
}

export function computeFit(r: Recipe, q: Quant, ctx: number, hw: Hardware, rt: RuntimeId): Fit {
  const weightsGb = q.size_gb ?? ((r.params_b || 8) * 2);
  const kv = kvGb(r, ctx);
  const over = runtimeOverheadGb(rt, weightsGb);
  const total = weightsGb + kv + over;
  const b = budget(hw, rt);
  const hasGpu = b.vram > 1;
  const base = { totalGb: round1(total), weightsGb: round1(weightsGb), kvGb: round1(kv) };
  if (hw.apple) {
    if (total <= b.vram) return { ...base, setup: "mac", gpuGb: round1(total), ramGb: 0, gpuShare: 1, ok: true, slow: false };
    return { ...base, setup: "none", gpuGb: 0, ramGb: 0, gpuShare: 0, ok: false, slow: false,
      reason: `Needs about ${round1(total)} GB but this Mac lets the GPU use about ${round1(b.vram)} GB.` };
  }
  if (hasGpu && total <= b.vram) return { ...base, setup: "gpu", gpuGb: round1(total), ramGb: 0, gpuShare: 1, ok: true, slow: false };
  // vLLM / SGLang cannot spill weights into RAM well; mixed mode is llama.cpp's job.
  // The KV cache stays on the GPU (spunup keeps attention there and moves weights/experts to RAM).
  if (hasGpu && rt === "llamacpp" && total <= b.vram + b.ram && kv + over <= b.vram) {
    const gpuForWeights = Math.max(0, b.vram - kv - over);
    const share = Math.min(1, gpuForWeights / weightsGb);
    return { ...base, setup: "mixed", gpuGb: round1(b.vram), ramGb: round1(total - b.vram), gpuShare: share, ok: true, slow: share < 0.3 };
  }
  if (!hasGpu && rt === "llamacpp" && total <= b.ram) return { ...base, setup: "cpu", gpuGb: 0, ramGb: round1(total), gpuShare: 0, ok: true, slow: false };
  const room = hasGpu ? (rt === "llamacpp" ? b.vram + b.ram : b.vram) : b.ram;
  return { ...base, setup: "none", gpuGb: 0, ramGb: 0, gpuShare: 0, ok: false, slow: false,
    reason: `Needs about ${round1(total)} GB but there is only about ${round1(room)} GB free for it.` };
}

/** Largest context that fits (and the model supports), rounded down to 1,024. */
export function maxContextThatFits(r: Recipe, q: Quant, hw: Hardware, rt: RuntimeId): number {
  const maxM = r.max_context || 32768;
  if (computeFit(r, q, maxM, hw, rt).ok) return maxM;
  let lo = 0, hi = maxM;
  while (hi - lo > 1024) {
    const mid = Math.floor((lo + hi) / 2);
    if (computeFit(r, q, mid, hw, rt).ok) lo = mid; else hi = mid;
  }
  return Math.floor(lo / 1024) * 1024;
}

// ------------------------------------------------------------------ speed (spec 6.3)

export interface Speed { tpsMin: number; tpsMax: number; prefill: number; basis: string }

/** Bytes read per generated token (GB): active weights at this quant + a little KV. */
function activeGb(r: Recipe, q: Quant): number {
  const total = r.params_b || 8;
  const active = r.active_b || (r.moe ? total * 0.1 : total);
  const weights = q.size_gb ?? total * 2;
  return weights * (active / total);
}

export interface Device { bwGb: number; overheadS: number; prefill7b: number }

/** Calibrate a device from a measured 7B benchmark: t = bytes/bw + fixed overhead. */
export interface Compute { fp32?: number; fp16?: number }

/** Calibrate one GPU so that  time per token = bytes read / bandwidth + fixed cost.
 *  - bw: memory bandwidth in GB/s (Vast's measured gpu_mem_bw, or the spec table)
 *  - tg7b / pp7b: measured llama.cpp tg128 / pp512 on Llama-2-7B Q4_0, when the GPU is in the scoreboards
 *  - compute: FP32 TFLOPS (what Vast's total_flops reports) or FP16 TFLOPS (the scout table), used when
 *    there is no pp7b. On the nine scoreboard GPUs Vast also lists, pp7b ≈ 175 × FP32 TFLOPS (136–197);
 *    on H100 SXM, pp7b ≈ 11.4 × FP16 TFLOPS.
 *  - apple: Metal pays a larger fixed cost per token (~4 ms, measured on an M2: Qwen3.5-2B Q4 41.6 tokens/s
 *    vs 7B Q4_0 21.9 tokens/s); CUDA's is far smaller. */
export function calibrate(bw: number | undefined, tg7b: number, pp7b: number, compute: Compute = {}, apple = false, floorS?: number): Device {
  const FLOOR = floorS ?? (apple ? 0.004 : 0.0008);
  let effBw = bw ? bw * measuredShare(bw) : 300;
  if (tg7b) {
    effBw = BENCH_MODEL_GB / Math.max(1e-4, 1 / tg7b - FLOOR);
    if (bw) effBw = Math.min(effBw, bw * 0.9);       // never faster than the memory itself
  }
  // A measured benchmark can show a smaller fixed cost than the floor (Apple M5 chips): trust it, down to 1.5 ms.
  const overhead = tg7b ? Math.max(apple ? 0.0015 : FLOOR, 1 / tg7b - BENCH_MODEL_GB / effBw) : FLOOR;
  const pp = pp7b || (compute.fp32 ? compute.fp32 * 175 : compute.fp16 ? compute.fp16 * 11.4 : 1000);
  return { bwGb: effBw, overheadS: overhead, prefill7b: pp };
}

/** Share of its memory speed a GPU without its own llama.cpp benchmark is likely to reach: the share the
 *  benchmarked GPU nearest to it in memory speed reached. Desktop cards reach about 70–80%, but the fast
 *  HBM cards only about 40% (A100 200.9, H100 280.7 tokens/s on the 3.56 GB benchmark model), so a flat
 *  80% made an H200 look about twice as fast as its nearest measured relative, the H100. Below the
 *  slowest benchmarked card the old 80% stays: small cards reach about that (RTX 3060: ~81% of 360 GB/s). */
function measuredShare(bw: number): number {
  let best: { d: number; share: number } | undefined;
  let slowest = Infinity;
  for (const g of GPU_SPECS) {
    if (!g.tg || !g.bw) continue;
    slowest = Math.min(slowest, g.bw);
    const eff = Math.min(BENCH_MODEL_GB / Math.max(1e-4, 1 / g.tg - 0.0008), g.bw * 0.9);
    const d = Math.abs(Math.log(g.bw / bw));
    if (!best || d < best.d) best = { d, share: eff / g.bw };
  }
  return best && bw >= slowest ? best.share : 0.8;
}

export function localDevices(hw: Hardware): { gpu?: Device; gpus: number; cpuBwGb: number; pcieGbps: number; cpuCores: number } {
  // Measured RAM speed when we have it, otherwise the CPU family table. llama.cpp on the CPU used
  // about half the measured speed in a real run (18 tokens/s for a 0.8 GB model at 29 GB/s measured):
  // with few cores the matrix maths, not the memory, sets the pace.
  const cpu = hw.cpu.bwGb ? hw.cpu.bwGb * 0.5 : cpuBw(hw.cpu.model) * 0.4;
  if (hw.apple) {
    const a = appleSpec(hw.apple.chip);
    return { gpu: a ? calibrate(a.bw, a.tg, a.pp, {}, true) : calibrate(100, 21, 180, {}, true), gpus: 1, cpuBwGb: cpu, pcieGbps: 0, cpuCores: hw.cpu.cores };
  }
  const g = hw.gpus.find((x) => x.vendor === "nvidia" || x.vendor === "amd" || x.vendor === "intel");
  if (!g) return { gpus: 0, cpuBwGb: cpu, pcieGbps: 0, cpuCores: hw.cpu.cores };
  const s = gpuSpec(g.name);
  // Each token has a fixed cost besides reading the weights, and it is larger on Windows: Qwen3.5-0.8B
  // Q8 on an RTX PRO 6000 under Windows 10 ran at 443 tokens/s (2.26 ms a token, of which reading
  // 0.8 GB takes about 0.5 ms), so about 1.7 ms is fixed.
  const floor = hw.os === "windows" ? 0.0017 : undefined;
  const dev = s ? calibrate(s.bw, s.tg, s.pp, { fp16: s.tflops, fp32: s.fp32 }, false, floor) : calibrate(Math.max(150, g.vramGb * 25), 0, 0, {}, false, floor);
  return { gpu: dev, gpus: hw.gpus.filter((x) => x.vendor === g.vendor).length, cpuBwGb: cpu, pcieGbps: 25, cpuCores: hw.cpu.cores };
}

/** Fixed cost per layer of a mixture-of-experts model, per token (see estimateSpeed). */
const MOE_LAYER_S = 0.000115;

/** Estimate generation and prefill speed. `share` = fraction of weights on the GPU. */
export function estimateSpeed(r: Recipe, q: Quant, dev: { gpu?: Device; gpus: number; cpuBwGb: number; pcieGbps: number; cpuCores?: number; layerSplit?: boolean }, share: number, spread = 0.2): Speed {
  const act = activeGb(r, q);
  const total = r.params_b || 8;
  const activeB = r.active_b || (r.moe ? total * 0.1 : total);
  const n = Math.max(1, dev.gpus);
  let t: number, pre: number, basis: string;
  if (!dev.gpu || share <= 0) {
    t = act / dev.cpuBwGb;
    // Prefill on a CPU is compute-bound, so it grows with cores. Calibrated on a real run: Qwen3.5-0.8B
    // Q8 with llama.cpp on a 4-core Cascade Lake VM did prefill 146 tokens/s (RESEARCH.md §7).
    pre = Math.max(5, ((dev.cpuCores || 8) * 30) / Math.max(0.3, activeB));
    basis = "CPU memory speed";
  } else {
    // MoE in mixed mode: attention + shared weights stay on GPU, routed experts split (scout logic).
    const gpuBytes = r.moe && share < 1 ? act * (0.15 + 0.85 * share) : act * share;
    const cpuBytes = act - gpuBytes;
    // vLLM / SGLang split each layer across the GPUs (tensor parallel), so they read the weights together.
    // llama.cpp (by default) gives each GPU whole layers and they take turns on every token: more room, not
    // a faster single chat.
    const bw = dev.gpu.bwGb * (dev.layerSplit ? 1 : Math.pow(n, 0.85));
    // Mixture-of-experts models pay a fixed cost on every layer (routing, many small expert kernels, syncs
    // between GPUs) that reading the weights doesn't explain. SGLang at one request, no speculation (cookbook
    // pages in research/raw/sglang-docs): Kimi-K3.1-Pro on 8× B300 9.6 ms a token, MiniMax-M2.5 on 8× B200
    // 9.89 ms and on 8× H100 8.08 ms, Qwen3.8-Flash-Next on 1× RTX PRO 6000 11.4 ms. After the weights that
    // is 70–150 µs a layer, ~115 µs on average. (Dense models don't show it: Llama-2-7B on an H100 does a
    // whole token, 32 layers, in 3.56 ms with llama.cpp.)
    const fixed = r.moe && r.layers ? Math.max(dev.gpu.overheadS, r.layers * MOE_LAYER_S) : dev.gpu.overheadS;
    t = gpuBytes / bw + cpuBytes / dev.cpuBwGb + fixed;
    // Prefill does not speed up linearly for smaller models (fixed costs); exponent 0.6 fits the M2 test.
    pre = dev.gpu.prefill7b * Math.pow(BENCH_ACTIVE_B / Math.max(0.5, activeB), 0.6) * Math.pow(n, 0.7);
    if (share < 1 && dev.pcieGbps) {
      const offloaded = (q.size_gb ?? total) * (1 - share);
      pre = Math.min(pre, (dev.pcieGbps * 2048) / Math.max(0.1, offloaded));
    }
    basis = share < 1 ? "GPU + RAM split" : "GPU memory speed, calibrated to llama.cpp benchmarks";
  }
  const tps = 1 / t;
  return { tpsMin: round1(tps * (1 - spread)), tpsMax: round1(tps), prefill: Math.round(pre), basis };
}

export function setupFor(fit: Fit): Setup | null {
  return fit.setup === "none" ? null : fit.setup;
}

const round1 = (x: number) => Math.round(x * 10) / 10;
