import type { Remote } from "../core/ssh.js";

export interface Offer {
  provider: "vast" | "runpod";
  id: string;                  // Vast ask id, or RunPod "gpuTypeId|count|cloud"
  gpuName: string;
  numGpus: number;
  vramPerGpuGb: number;
  ramGb: number;
  diskGb: number;
  cpuName?: string;
  pricePerHour: number;        // USD, compute + disk while running
  storagePerGbMonth?: number;  // USD
  downloadPerGb?: number;      // USD per GB downloaded (Vast inet_down_cost); RunPod charges none
  inetDownMbps?: number;
  reliability?: number;
  location?: string;
  cudaMax?: number;
  computeCap?: number;         // e.g. 8.9
  gpuMemBw?: number;           // GB/s (Vast measures this)
  tflops?: number;
  pcieGbps?: number;
  machineId?: number;
  stock?: "LOW" | "MEDIUM" | "HIGH";  // RunPod: how many of this GPU are free right now
  cloud?: "SECURE" | "COMMUNITY";      // RunPod cloud tier the offer is on
}

/** RunPod's two clouds: Secure (RunPod's data centres) and Community (vetted hosts, usually cheaper). */
export type RunpodCloud = "SECURE" | "COMMUNITY" | "ALL";

/** The machine picked from the list was taken before it could be rented (stock changes by the minute). */
export class NoCapacity extends Error {}

export interface Instance {
  id: string;
  status: "starting" | "running" | "stopped" | "gone" | "error";
  statusText?: string;
  ssh?: Remote;
  pricePerHour?: number;
  stoppedPricePerHour?: number;
  startedAt?: number;          // ms
  label?: string;
  gpuName?: string;
  numGpus?: number;
  volumeId?: string;          // Vast: volume that can outlive the machine
  downloadCost?: number;      // USD billed so far for data in and out (Vast measures it per instance)
}

export interface CreateRequest {
  offer: Offer;
  image: string;
  diskGb: number;
  label: string;
  publicKey: string;
  keepDiskSeparately?: boolean;   // Vast: put the model on a volume that can outlive the machine
}

export interface Provider {
  id: "vast" | "runpod";
  name: string;
  /** Whether a disk can be kept or deleted separately from the machine through the API. */
  separateDisk: boolean;
  stopNote: string;
  deleteNote: string;
  /** `fastest`: machines with the fastest GPU memory first, instead of the cheapest first. */
  search(req: { minVramGb: number; minDiskGb: number; minCc?: number; maxGpus?: number; cloud?: RunpodCloud; fastest?: boolean }): Promise<Offer[]>;
  create(req: CreateRequest): Promise<{ id: string; volumeId?: string }>;
  get(id: string): Promise<Instance>;
  list(): Promise<Instance[]>;
  stop(id: string): Promise<void>;
  start(id: string): Promise<void>;
  destroy(id: string, opts?: { volumeId?: string; keepDisk?: boolean }): Promise<void>;
  /** Delete a disk kept after "Machine only" (providers with separateDisk). */
  deleteDisk?(volumeId: string): Promise<void>;
  /** Let spunup's SSH key into a machine that is already running ("Use a machine I already have"). */
  attachKey?(id: string, publicKey: string): Promise<void>;
  /** Cost so far from the provider's billing API, if it has one. */
  billed?(id: string, sinceMs: number): Promise<number | undefined>;
  registerKey(publicKey: string): Promise<void>;
  /** Root folder on the machine where models and logs go. */
  workDir: string;
}
