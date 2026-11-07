import os from "node:os";
import fs from "node:fs";
import { runSync } from "./proc.js";
import { Worker } from "node:worker_threads";
import { HOME, IS_MAC, IS_WIN, IS_LINUX } from "./paths.js";
import { loadConfig, saveConfig } from "./store.js";

export interface Gpu {
  vendor: "nvidia" | "amd" | "intel" | "apple" | "other";
  name: string;
  vramGb: number;          // usable memory for the model (Apple: share of unified RAM)
  driver?: string;
  cuda?: string;           // max CUDA version the driver supports (NVIDIA)
  computeCap?: string;
}

export interface Hardware {
  os: "linux" | "macos" | "windows";
  arch: string;
  cpu: { model: string; cores: number; avx2: boolean | "unknown"; avx512: boolean | "unknown"; bwGb?: number /* measured */ };
  ramGb: number;
  diskFreeGb: number;
  gpus: Gpu[];
  apple?: { chip: string; unifiedGb: number };
  wsl?: WslInfo;
  notes: string[];
}

function run(cmd: string, args: string[], timeout = 8000): string | undefined {
  try {
    const r = runSync(cmd, args, { encoding: "utf8", timeout, windowsHide: true });
    return r.status === 0 ? r.stdout : undefined;
  } catch {
    return undefined;
  }
}

function nvidia(): Gpu[] {
  const out = run("nvidia-smi", ["--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"]);
  if (!out) return [];
  const head = run("nvidia-smi", []) || "";
  const cuda = /CUDA Version:\s*([\d.]+)/.exec(head)?.[1];
  return out.trim().split(/\r?\n/).filter(Boolean).map((l) => {
    const [name, mem, driver, cc] = l.split(",").map((s) => s.trim());
    // Unified-memory boards (GB10 / DGX Spark, Jetson) can report "[N/A]": count 0 rather than NaN.
    const mib = Number(mem);
    return { vendor: "nvidia" as const, name, vramGb: Number.isFinite(mib) ? Math.round((mib / 1024) * 10) / 10 : 0, driver, cuda, computeCap: cc };
  });
}

function amdLinux(): Gpu[] {
  const out = run("rocm-smi", ["--showproductname", "--showmeminfo", "vram", "--json"]);
  if (!out) return [];
  try {
    const j = JSON.parse(out);
    return Object.values<any>(j).filter((c) => c && typeof c === "object" && (c["Card series"] || c["Card SKU"])).map((c) => ({
      vendor: "amd" as const,
      name: String(c["Card series"] || c["Card SKU"] || "AMD GPU"),
      vramGb: Math.round(Number(c["VRAM Total Memory (B)"] || 0) / 1e9 * 10) / 10,
    }));
  } catch {
    return [];
  }
}

function windowsGpus(): Gpu[] {
  // Win32_VideoController.AdapterRAM caps at 4 GB, so read qwMemorySize from the display class registry keys.
  const ps = `$k='HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}';
Get-ChildItem $k -ErrorAction SilentlyContinue | ForEach-Object { $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue;
 if ($p.DriverDesc) { "$($p.DriverDesc)|$($p.'HardwareInformation.qwMemorySize')" } }`;
  const out = run("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", ps], 15000);
  if (!out) return [];
  const seen = new Set<string>();
  const gpus: Gpu[] = [];
  for (const line of out.trim().split(/\r?\n/)) {
    const [name, mem] = line.split("|");
    // Integrated graphics share system RAM; adding them to a real GPU's memory makes models look like they fit.
    if (!name || seen.has(name) || /basic display|remote|virtual/i.test(name) || /radeon\(tm\) graphics|radeon graphics|uhd graphics|iris|hd graphics|vega \d+ graphics/i.test(name)) continue;
    seen.add(name);
    const vendor = /nvidia|geforce|rtx|quadro/i.test(name) ? "nvidia" : /amd|radeon/i.test(name) ? "amd" : /intel|arc/i.test(name) ? "intel" : "other";
    if (vendor === "nvidia") continue; // nvidia-smi covers these with exact numbers
    gpus.push({ vendor, name: name.trim(), vramGb: Math.round(Number(mem || 0) / 1e9 * 10) / 10 });
  }
  return gpus;
}

function intelLinux(): Gpu[] {
  const out = run("lspci", []);
  if (!out) return [];
  return out.split("\n").filter((l) => /VGA|Display|3D/.test(l) && /Intel.*Arc/i.test(l)).map((l) => ({
    vendor: "intel" as const, name: l.split(": ").pop()!.trim(), vramGb: 0,
  }));
}

function cpuFlags(): { avx2: boolean | "unknown"; avx512: boolean | "unknown" } {
  if (IS_LINUX) {
    try {
      const f = fs.readFileSync("/proc/cpuinfo", "utf8");
      return { avx2: /\bavx2\b/.test(f), avx512: /\bavx512f\b/.test(f) };
    } catch { /* fall through */ }
  }
  if (IS_MAC && os.arch() === "x64") {
    const f = (run("sysctl", ["-n", "machdep.cpu.leaf7_features"]) || "").toUpperCase();
    return { avx2: f.includes("AVX2"), avx512: f.includes("AVX512F") };
  }
  if (os.arch() === "arm64") return { avx2: false, avx512: false };
  return { avx2: "unknown", avx512: "unknown" };
}

/** Share of unified memory macOS lets the GPU use by default
 *  (Metal recommendedMaxWorkingSetSize is about 2/3 on small Macs, 3/4 on larger ones). */
export function appleGpuShare(ramGb: number): number {
  return ramGb <= 36 ? 0.67 : 0.75;
}

/** How far WSL is set up: off (Windows feature not on), nodistro (on, but no Linux installed),
 *  nogpu (Linux installed, but it cannot see the NVIDIA GPU), or ready. */
export interface WslInfo { installed: boolean; gpu: boolean; state: "off" | "nodistro" | "nogpu" | "ready" }

export function wslInfo(): WslInfo {
  // WSL_UTF8 makes wsl.exe print UTF-8 instead of UTF-16.
  const wsl = (args: string[], timeout: number) => {
    try {
      const r = runSync("wsl.exe", args, { encoding: "utf8", timeout, windowsHide: true, env: { ...process.env, WSL_UTF8: "1" } });
      return { ok: r.status === 0, out: `${r.stdout || ""}${r.stderr || ""}` };
    } catch { return { ok: false, out: "" }; }
  };
  const st = wsl(["--status"], 8000);
  if (!st.ok || /Optional Component|not installed|is not enabled/i.test(st.out)) return { installed: false, gpu: false, state: "off" };
  const list = wsl(["--list", "--quiet"], 8000);
  if (!list.ok || !list.out.trim()) return { installed: true, gpu: false, state: "nodistro" };
  const smi = wsl(["-e", "sh", "-c", "PATH=$PATH:/usr/lib/wsl/lib nvidia-smi -L"], 30000);
  const gpu = smi.ok && /GPU \d/.test(smi.out);
  return { installed: true, gpu, state: gpu ? "ready" : "nogpu" };
}

/** RAM read speed (GB/s), measured once per computer and saved: one worker per core scans its own
 *  buffer with native memchr. CPU generation speed follows this number, and a CPU name says little
 *  about it (a 4-core cloud VM on a big Xeon measured 29 GB/s; the whole socket would do ~150). */
export async function measureCpuBw(hw: Hardware): Promise<number | undefined> {
  const key = `${hw.cpu.model}|${hw.cpu.cores}|${hw.ramGb}`;
  const cfg = loadConfig();
  if (cfg.cpuBw?.key === key) return cfg.cpuBw.gbps;
  const n = Math.min(hw.cpu.cores, 32);
  const mb = Math.max(32, Math.min(128, Math.floor((os.totalmem() * 0.1) / n / 2 ** 20)));
  // Buffers are written first: untouched pages all map to one zero page and read from cache.
  const src = `const { parentPort, workerData } = require("node:worker_threads");
const b = Buffer.allocUnsafe(workerData.mb * 1048576); b.fill(7); b.indexOf(1);
parentPort.once("message", () => { let bytes = 0; const t0 = process.hrtime.bigint(); let t;
  do { b.indexOf(1); bytes += b.length; t = Number(process.hrtime.bigint() - t0) / 1e9; } while (t < 0.25);
  parentPort.postMessage(bytes / t); });`;
  const ws: Worker[] = [];
  try {
    for (let i = 0; i < n; i++) ws.push(new Worker(src, { eval: true, workerData: { mb } }));
    await Promise.all(ws.map((w) => new Promise((r) => w.once("online", r))));
    await new Promise((r) => setTimeout(r, 300));
    const rates = await Promise.all(ws.map((w) => new Promise<number>((r) => { w.once("message", r); w.postMessage(0); })));
    const gbps = Math.round(rates.reduce((a, b) => a + b, 0) / 1e8) / 10;
    if (!(gbps > 1)) return undefined;
    saveConfig({ ...loadConfig(), cpuBw: { key, gbps } });
    return gbps;
  } catch { return undefined; } finally { ws.forEach((w) => w.terminate()); }
}

export function detectHardware(): Hardware {
  const notes: string[] = [];
  const ramGb = Math.round(os.totalmem() / 2 ** 30);
  const cpus = os.cpus();
  let diskFreeGb = 0;
  try {
    fs.mkdirSync(HOME, { recursive: true });
    const s = fs.statfsSync(HOME);
    diskFreeGb = Math.round((s.bavail * s.bsize) / 1e9);
  } catch { /* ignore */ }
  const hw: Hardware = {
    os: IS_MAC ? "macos" : IS_WIN ? "windows" : "linux",
    arch: os.arch(),
    cpu: { model: (cpus[0]?.model || "Unknown CPU").replace(/\s+/g, " ").trim(), cores: cpus.length, ...cpuFlags() },
    ramGb,
    diskFreeGb,
    gpus: [],
    notes,
  };
  if (IS_MAC) {
    if (os.arch() === "arm64") {
      const chip = (run("sysctl", ["-n", "machdep.cpu.brand_string"]) || "Apple Silicon").trim();
      const share = appleGpuShare(ramGb);
      hw.apple = { chip, unifiedGb: ramGb };
      hw.gpus.push({ vendor: "apple", name: chip, vramGb: Math.floor(ramGb * share) });
    } else {
      notes.push("Intel Mac: models run on the CPU (the GPU is not used).");
    }
    return hw;
  }
  hw.gpus.push(...nvidia());
  if (IS_LINUX) hw.gpus.push(...amdLinux(), ...intelLinux());
  if (IS_WIN) {
    hw.gpus.push(...windowsGpus());
    hw.wsl = wslInfo();
  }
  const nv = hw.gpus.find((g) => g.vendor === "nvidia");
  if (nv?.cuda && Number(nv.cuda) < 12.4) notes.push(`Your NVIDIA driver is old (supports CUDA ${nv.cuda}). Update it from nvidia.com for the fastest runtimes.`);
  if (!hw.gpus.length) notes.push("No usable GPU found. Models will run on the CPU, which is slower.");
  if (hw.gpus.some((g) => g.vendor === "intel")) notes.push("Intel GPUs run through llama.cpp's Vulkan build.");
  return hw;
}

export function totalVram(hw: Hardware): number {
  return hw.gpus.filter((g) => g.vendor !== "other").reduce((a, g) => a + g.vramGb, 0);
}

export function hardwareSummary(hw: Hardware): string {
  const g = hw.gpus.filter((x) => x.vendor !== "other");
  if (hw.apple) return `${hw.apple.chip} · ${hw.ramGb}GB unified memory`;
  if (!g.length) return `${hw.cpu.model.split("@")[0].trim()} · ${hw.ramGb}GB RAM · no GPU`;
  const first = g[0];
  const name = first.name.replace(/NVIDIA |GeForce |AMD |Radeon /g, "");
  const count = g.length > 1 ? `${g.length}× ` : "";
  return `${count}${name} ${Math.round(first.vramGb)}GB · ${hw.ramGb}GB RAM`;
}
