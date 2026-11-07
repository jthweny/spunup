/** GPU and Apple chip numbers used for speed estimates.
 *
 *  tg / pp: llama.cpp llama-bench on Llama-2-7B Q4_0 (3.56 GB of weights), flash attention on,
 *  from the community scoreboards:
 *    CUDA:  https://github.com/ggml-org/llama.cpp/discussions/15013
 *    Apple: https://github.com/ggml-org/llama.cpp/discussions/4167
 *  bw (GB/s) / tflops (dense FP16): the owner's vast_moe_scout.py table where listed; Apple
 *  bandwidth from the Apple scoreboard. Other GPUs have no bw entry and are derived from tg.
 *  fp32 (TFLOPS) + bw for GPUs RunPod rents that the scoreboards miss: NVIDIA product pages /
 *  datasheets, checked 2026-11-06 (RTX PRO 4000: 672 GB/s, 37 TFLOPS; RTX PRO 4500: 896 GB/s,
 *  server edition 800 GB/s, 51 TFLOPS; L4: 300 GB/s, 30.3; A40: 696 GB/s; RTX 4000 Ada: 360 GB/s, 26.7;
 *  RTX PRO 6000: 1,792 GB/s, 125 TFLOPS; RTX PRO 5000: 1,344 GB/s, 65 TFLOPS; RTX 5000 Ada: 576 GB/s,
 *  65.3; RTX 2000 Ada: 224 GB/s, 12.0; RTX A4500: 640 GB/s, 23.7; RTX A2000: 288 GB/s, 8.0;
 *  RTX 4070 Ti: 504 GB/s, 40.1; RTX 3070: 448 GB/s, 20.3). A MIG slice ("PRO 6000 MIG 24GB") gets the full card's
 *  numbers times its share of the memory (24 GB of 96 GB = 1/4): MIG splits memory and compute together.
 */
export interface GpuSpec { match: string; bw?: number; tflops?: number; fp32?: number; tg: number; pp: number; cc?: number; vram?: number }

export const GPU_SPECS: GpuSpec[] = [
  { match: "B300", bw: 8000, tflops: 2250, tg: 0, pp: 0, cc: 10.3 },
  { match: "B200", bw: 8000, tflops: 2250, tg: 0, pp: 0, cc: 10.0 },
  { match: "H200", bw: 4800, tflops: 990, tg: 0, pp: 0, cc: 9.0 },
  { match: "MI300X", bw: 5300, tflops: 1300, tg: 0, pp: 0 },
  { match: "H100 NVL", bw: 3900, tflops: 835, tg: 0, pp: 0, cc: 9.0 },
  { match: "H100 SXM", bw: 3350, tflops: 990, tg: 280.74, pp: 11263.29, cc: 9.0 },
  { match: "H100 PCIE", bw: 2000, tflops: 756, tg: 0, pp: 0, cc: 9.0 },
  { match: "H100", bw: 2000, tflops: 756, tg: 280.74, pp: 11263.29, cc: 9.0 },
  { match: "A100 SXM", bw: 2039, tflops: 312, tg: 200.9, pp: 5285.96, cc: 8.0 },
  { match: "A100 PCIE", bw: 1935, tflops: 312, tg: 200.9, pp: 5285.96, cc: 8.0 },
  { match: "A100", bw: 1935, tflops: 312, tg: 200.9, pp: 5285.96, cc: 8.0 },
  { match: "RTX PRO 6000", bw: 1792, fp32: 125, tg: 281.11, pp: 16618.98, cc: 12.0, vram: 96 },
  { match: "RTX PRO 5000", bw: 1344, fp32: 65, tg: 0, pp: 0, cc: 12.0, vram: 48 },
  { match: "RTX PRO 4500 SE", bw: 800, fp32: 51, tg: 0, pp: 0, cc: 12.0 },
  { match: "RTX PRO 4500", bw: 896, fp32: 51, tg: 0, pp: 0, cc: 12.0 },
  { match: "RTX PRO 4000", bw: 672, fp32: 37, tg: 0, pp: 0, cc: 12.0 },
  { match: "RTX 5090", bw: 1792, tflops: 419, tg: 300.4, pp: 14970.15, cc: 12.0 },
  { match: "RTX 5080", tg: 184.68, pp: 9487.7, cc: 12.0 },
  { match: "RTX 5070 TI", tg: 182.43, pp: 8419.56, cc: 12.0 },
  { match: "RTX 5070", tg: 128.21, pp: 5783.44, cc: 12.0 },
  { match: "RTX 5060 TI", tg: 93.46, pp: 4195.53, cc: 12.0 },
  { match: "RTX 4090", bw: 1008, tflops: 330, tg: 188.96, pp: 14770.63, cc: 8.9 },
  { match: "RTX 6000 ADA", bw: 960, tflops: 364, tg: 179.47, pp: 10576.85, cc: 8.9 },
  { match: "RTX 5000 ADA", bw: 576, fp32: 65.3, tg: 0, pp: 0, cc: 8.9 },
  { match: "RTX 2000 ADA", bw: 224, fp32: 12.0, tg: 0, pp: 0, cc: 8.9 },
  { match: "RTX 4000 ADA", bw: 360, fp32: 26.7, tg: 0, pp: 0, cc: 8.9 },
  { match: "L40S", bw: 864, tflops: 362, tg: 153.76, pp: 10097.64, cc: 8.9 },
  { match: "L40", bw: 864, tflops: 181, tg: 153.76, pp: 10097.64, cc: 8.9 },
  { match: "RTX 4080 SUPER", tg: 147.48, pp: 9439.01, cc: 8.9 },
  { match: "RTX 4080", tg: 143.47, pp: 9205.93, cc: 8.9 },
  { match: "RTX 4070 TI SUPER", tg: 132.85, pp: 7612.32, cc: 8.9 },
  { match: "RTX 4070 TI", bw: 504, fp32: 40.1, tg: 0, pp: 0, cc: 8.9 },
  { match: "RTX 4060 TI", tg: 64.03, pp: 3803.45, cc: 8.9 },
  { match: "RTX 3090 TI", tg: 172.26, pp: 6924.01, cc: 8.6 },
  { match: "RTX 3090", bw: 936, tflops: 142, tg: 161.89, pp: 5560.06, cc: 8.6 },
  { match: "RTX 3080", tg: 139.95, pp: 5569.56, cc: 8.6 },
  { match: "RTX 3070", bw: 448, fp32: 20.3, tg: 0, pp: 0, cc: 8.6 },
  { match: "RTX 3060", tg: 76.92, pp: 2407.67, cc: 8.6 },
  { match: "A6000", bw: 768, tflops: 155, tg: 144.87, pp: 5662.39, cc: 8.6 },
  { match: "A5000", tg: 135.83, pp: 4552.15, cc: 8.6 },
  { match: "A4500", bw: 640, fp32: 23.7, tg: 0, pp: 0, cc: 8.6 },
  { match: "A2000", bw: 288, fp32: 8.0, tg: 0, pp: 0, cc: 8.6 },
  { match: "A4000", tg: 85.17, pp: 2807.83, cc: 8.6 },
  { match: "A40", bw: 696, tg: 126.24, pp: 5256.38, cc: 8.6 },
  { match: "L4", bw: 300, fp32: 30.3, tg: 0, pp: 0, cc: 8.9 },
  { match: "A10G", tg: 111.51, pp: 2190.43, cc: 8.6 },
  { match: "V100", tg: 134.76, pp: 2973.78, cc: 7.0 },
  { match: "T4", tg: 44.03, pp: 1309.73, cc: 7.5 },
  { match: "RTX 2080 TI", tg: 109.17, pp: 3107.61, cc: 7.5 },
];

export interface AppleSpec { match: string; bw: number; tg: number; pp: number }

/** Highest-core variant per chip from the Apple scoreboard (Q4_0 columns). */
export const APPLE_SPECS: AppleSpec[] = [
  { match: "M5 ULTRA", bw: 1228, tg: 179.1, pp: 4944.66 },
  { match: "M5 MAX", bw: 614, tg: 119.92, pp: 3219.99 },
  { match: "M5 PRO", bw: 307, tg: 66.33, pp: 1620.64 },
  { match: "M5", bw: 154, tg: 31.88, pp: 722.79 },
  { match: "M4 MAX", bw: 546, tg: 83.06, pp: 885.68 },
  { match: "M4 PRO", bw: 273, tg: 50.74, pp: 439.78 },
  { match: "M4", bw: 120, tg: 24.11, pp: 221.29 },
  { match: "M3 ULTRA", bw: 800, tg: 92.14, pp: 1471.24 },
  { match: "M3 MAX", bw: 400, tg: 66.31, pp: 759.7 },
  { match: "M3 PRO", bw: 150, tg: 30.74, pp: 341.67 },
  { match: "M3", bw: 100, tg: 21.34, pp: 186.75 },
  { match: "M2 ULTRA", bw: 800, tg: 94.27, pp: 1238.48 },
  { match: "M2 MAX", bw: 400, tg: 65.95, pp: 671.31 },
  { match: "M2 PRO", bw: 200, tg: 38.86, pp: 341.19 },
  { match: "M2", bw: 100, tg: 21.91, pp: 179.57 },
  { match: "M1 ULTRA", bw: 800, tg: 83.73, pp: 1030.04 },
  { match: "M1 MAX", bw: 400, tg: 61.19, pp: 530.06 },
  { match: "M1 PRO", bw: 200, tg: 36.41, pp: 266.25 },
  { match: "M1", bw: 68, tg: 14.15, pp: 117.96 },
];

/** RAM bandwidth (GB/s) by CPU family, from the owner's vast_moe_scout.py. Desktop default is a guess. */
export const CPU_BW: [string, number][] = [
  ["EPYC 9", 460], ["EPYC 7", 205], ["THREADRIPPER PRO 7", 330], ["THREADRIPPER PRO 5", 200],
  ["THREADRIPPER PRO 3", 180], ["THREADRIPPER", 120], ["XEON PLATINUM 8", 300], ["XEON GOLD 6", 280], ["XEON", 150],
];
export const CPU_BW_DEFAULT = 60; // typical dual-channel desktop; not measured — shown as an estimate

/** Match ignoring spaces and dashes: providers write "RTX 6000Ada", "RTX_4090", "H100 80GB HBM3"… */
function longestMatch<T extends { match: string }>(table: T[], name: string): T | undefined {
  const norm = (s: string) => s.toUpperCase().replace(/NVIDIA|GEFORCE|[\s_-]+/g, "");
  const up = norm(name);
  return [...table].sort((a, b) => norm(b.match).length - norm(a.match).length).find((g) => up.includes(norm(g.match)));
}

export const BENCH_MODEL_GB = 3.56; // Llama-2-7B Q4_0 weights read per token
export const BENCH_ACTIVE_B = 6.74;

export function gpuSpec(name: string): GpuSpec | undefined {
  const mig = /^(.*?)\s*MIG\s*(\d+)\s*GB/i.exec(name);
  if (mig) {
    const full = longestMatch(GPU_SPECS, mig[1]) || longestMatch(GPU_SPECS, `RTX ${mig[1]}`);
    if (!full?.vram) return undefined;
    const f = Number(mig[2]) / full.vram;
    return { match: name, bw: full.bw && full.bw * f, fp32: full.fp32 && full.fp32 * f, tg: 0, pp: 0, cc: full.cc };
  }
  return longestMatch(GPU_SPECS, name);
}
export function appleSpec(name: string): AppleSpec | undefined { return longestMatch(APPLE_SPECS, name); }
export function cpuBw(name: string): number {
  const up = name.toUpperCase();
  for (const [k, v] of [...CPU_BW].sort((a, b) => b[0].length - a[0].length)) if (up.includes(k)) return v;
  return CPU_BW_DEFAULT;
}
