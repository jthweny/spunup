import fs from "node:fs";
import path from "node:path";
import { P, ensureDirs } from "./paths.js";

export function readJson<T>(file: string, fallback: T): T {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8")) as T;
  } catch {
    return fallback;
  }
}

/** Write atomically so a crash never leaves a half-written file. */
export function writeJson(file: string, data: unknown, mode = 0o600): void {
  ensureDirs();
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2), { mode });
  fs.renameSync(tmp, file);
}

export type Where = "local" | "vast" | "runpod";
export type RuntimeId = "vllm" | "sglang" | "llamacpp" | "mlx";
export type Setup = "gpu" | "cpu" | "mixed" | "mac";

export interface Endpoint {
  id: string;                 // spunup id, e.g. "e-1730900000-ab12"
  model: string;              // HF id
  name: string;               // served model name
  where: Where;
  runtime: RuntimeId;
  setup: Setup;
  quant: string;              // quant label, e.g. "Q4" / "FP8"
  quantId: string;
  context: number;
  port: number;               // fixed local port the user's apps talk to
  apiKey: string;
  createdAt: number;          // ms
  startedAt?: number;         // ms, machine start (for cost)
  status: "starting" | "running" | "stopped" | "deleted" | "failed";
  // cloud only
  instanceId?: string;
  machine?: string;           // "1× RTX 4090"
  pricePerHour?: number;      // USD, compute while running
  stoppedPricePerHour?: number; // USD, while stopped (disk)
  costBefore?: number;        // USD accumulated in earlier running periods
  oneOffCost?: number;        // USD for data downloaded/uploaded so far (Vast: measured on the instance)
  downloadPerGb?: number;     // USD per GB this host charges for downloads (from the listing)
  stoppedAt?: number;
  remotePort?: number;
  ssh?: { host: string; port: number; user: string };
  volumeId?: string;          // Vast volume kept separately from the machine
  diskKept?: boolean;         // deleted with "Machine only": the disk is still there, and still charged
  deletedAt?: number;
  // local only
  pid?: number;
  innerPort?: number;         // MLX: runtime port behind the key proxy
  logFile?: string;
  launch?: { bin: string; args: string[]; env: Record<string, string> };   // how to start it again (no shell, so no quoting)
  // settings
  autoStop?: { enabled: boolean; minutes: number; action: "stop" | "delete" };
  measured?: { tps?: number; prefill?: number };
  estimate?: { tpsMin: number; tpsMax: number; prefill: number };
  lora?: { name: string; path: string };
  command?: string;
  lastError?: string;         // why the last start failed (shown in History)
}

export interface History {
  endpoints: Endpoint[];
}

export function loadHistory(): History {
  return readJson<History>(P.history, { endpoints: [] });
}

export function saveHistory(h: History): void {
  writeJson(P.history, h);
}

export function upsertEndpoint(e: Endpoint): void {
  const h = loadHistory();
  const i = h.endpoints.findIndex((x) => x.id === e.id);
  if (i >= 0) h.endpoints[i] = e;
  else h.endpoints.unshift(e);
  saveHistory(h);
}

export function getEndpoint(id: string): Endpoint | undefined {
  return loadHistory().endpoints.find((x) => x.id === id);
}

export interface Config {
  firstPort: number;
  ports: Record<string, number>;   // model+where key -> fixed port (reused)
  recipesUrl?: string;
  lastRecipeUpdate?: number;
  currency: "USD" | "GBP" | "EUR";
  fx?: { rate: number; at: number };
  cpuBw?: { key: string; gbps: number };   // measured RAM read speed of this computer (see measureCpuBw)
}

export function loadConfig(): Config {
  return readJson<Config>(P.config, { firstPort: 8000, ports: {}, currency: "USD" });
}

export function saveConfig(c: Config): void {
  writeJson(P.config, c);
}

export function logFile(name: string): string {
  ensureDirs();
  return path.join(P.logs, name);
}
