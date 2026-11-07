import { http, HttpError } from "../core/net.js";
import { getKey } from "../core/keys.js";
import { NoCapacity, type Provider, type Offer, type Instance, type CreateRequest } from "./types.js";

/** RunPod REST v2 (https://api.runpod.io/v2/openapi.json, checked 2026-11-06).
 *  Port forwarding needs `ssh.direct`: the proxy SSH "carries an interactive shell only",
 *  so every pod exposes 22/tcp and runs sshd (RunPod docs: Connect to a Pod with SSH). */
const API = "https://api.runpod.io/v2";

function headers(): Record<string, string> {
  const k = getKey("runpod");
  if (!k) throw new Error("No RunPod API key saved.");
  return { authorization: `Bearer ${k}` };
}
const call = <T = any>(path: string, opts: { method?: string; body?: unknown } = {}) =>
  http<T>(`${API}${path}`, { ...opts, headers: headers(), timeoutMs: 30_000 });

// RunPod's catalog does not publish memory bandwidth or CUDA compute capability, so these come from the GPU table.
const VOLUME_RUNNING_GB_MONTH = 0.10; // docs.runpod.io/pods/pricing: volume disk while running
const VOLUME_STOPPED_GB_MONTH = 0.20; // ... and while stopped
const CONTAINER_GB_MONTH = 0.10;
const CONTAINER_DISK_GB = 20;
const MIN_CUDA = "12.8";   // RunPod's documented floor for current images (runpod-usage gpu-selection)

/** sshd for any Debian/Ubuntu image, as RunPod documents for custom templates. */
const SSHD_START = "apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server curl >/dev/null; " +
  "mkdir -p ~/.ssh /run/sshd; chmod 700 ~/.ssh; echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; " +
  "service ssh start || /usr/sbin/sshd; mkdir -p /workspace/spunup; sleep infinity";

function toInstance(p: any): Instance {
  const st = String(p.status || "");
  const status: Instance["status"] = st === "RUNNING" ? "running" : st === "EXITED" ? "stopped" : st === "TERMINATED" ? "gone" : st === "ERROR" ? "error" : "starting";
  const d = p.ssh?.direct;
  const volGb = Number(p.mounts?.persistent?.size || 0);
  return {
    id: String(p.id),
    status,
    statusText: st,
    ssh: d?.host ? { host: d.host, port: Number(d.port), user: d.username || "root" } : undefined,
    pricePerHour: Number(p.cost || 0) + ((volGb * VOLUME_RUNNING_GB_MONTH + CONTAINER_DISK_GB * CONTAINER_GB_MONTH) / (30 * 24)),
    stoppedPricePerHour: (volGb * VOLUME_STOPPED_GB_MONTH) / (30 * 24),
    startedAt: p.startedAt ? Date.parse(p.startedAt) : p.createdAt ? Date.parse(p.createdAt) : undefined,
    label: p.name,
    gpuName: p.gpu?.id || p.gpu?.displayName,
    numGpus: p.gpu?.count,
  };
}

export const runpod: Provider = {
  id: "runpod",
  name: "RunPod",
  separateDisk: false, // the pod's volume disk is deleted with the pod; spunup does not use network volumes
  stopNote: "The GPU charge stops. You keep paying for the disk ($0.20 per GB per month while stopped). Someone else may take the GPU meanwhile, and then the pod cannot restart.",
  deleteNote: "The pod and its disk are gone. No more charges.",
  workDir: "/workspace/spunup",

  async search({ minVramGb, minDiskGb, maxGpus, cloud = "SECURE" }) {
    const offers: Offer[] = [];
    for (const c of cloud === "ALL" ? (["SECURE", "COMMUNITY"] as const) : [cloud]) {
      // Stock and price are asked per cloud: the same GPU can be free on one and gone on the other.
      const r = await call<any>(`/catalog/gpus?include=AVAILABILITY&product=POD&cloud=${c}`);
      const key = c === "SECURE" ? "secure" : "community";
      for (const g of Array.isArray(r) ? r : r.gpus || []) {
        if (!g[key] || g.availability === "NONE") continue;
        const per = Number(g.memory || 0);
        const price = Number(g.price?.[key] ?? 0);
        if (!per || !price) continue;
        const maxC = Number(g.maxCount?.[key] ?? 8);
        for (let n = 1; n <= Math.min(maxC, maxGpus || 8); n *= 2) {
          if (per * n < minVramGb) continue;
          offers.push({
            provider: "runpod", id: `${g.id}|${n}|${c}`, gpuName: g.name || g.id, numGpus: n, vramPerGpuGb: per,
            ramGb: Number(g.minRam || 0), diskGb: minDiskGb, pricePerHour: price * n + (minDiskGb * VOLUME_RUNNING_GB_MONTH) / 720,
            storagePerGbMonth: VOLUME_RUNNING_GB_MONTH, location: (g.dataCenters || []).map((d: any) => d.id).slice(0, 2).join(", "),
            cudaMax: undefined, stock: g.availability, cloud: c,
          });
          break; // smallest count that fits
        }
      }
    }
    return offers.sort((a, b) => a.pricePerHour - b.pricePerHour);
  },

  async create(req: CreateRequest) {
    const [gpuId, count, cloud] = req.offer.id.split("|");
    // Create places exactly this GPU type and never falls back (REST v2 "Checking what you can
    // deploy"). Stock is often LOW and changes by the minute, so read it again right before renting.
    const now = await call<any>(`/catalog/gpus/${encodeURIComponent(gpuId)}?include=AVAILABILITY&product=POD&cloud=${cloud}&count=${count}&minCudaVersion=${MIN_CUDA}`).catch(() => undefined);
    const g = now?.gpus?.[0] || now;
    if (g?.availability === "NONE") throw new NoCapacity(`RunPod has no free ${req.offer.gpuName} right now (someone took the last one after the list loaded).`);
    const body = {
      name: req.label, image: req.image, cloud,
      gpu: { id: gpuId, count: Number(count), minCudaVersion: MIN_CUDA },
      disk: CONTAINER_DISK_GB,
      mounts: { persistent: { size: Math.max(10, Math.ceil(req.diskGb)), path: "/workspace" } },
      ports: ["22/tcp"], startSsh: true,
      env: { PUBLIC_KEY: req.publicKey },
      entrypoint: ["bash", "-c"], cmd: [SSHD_START],
    };
    try {
      const p = await call<any>("/pods", { method: "POST", body });
      return { id: String(p.id) };
    } catch (e) {
      if (e instanceof HttpError) {
        const detail = (() => { try { return JSON.parse(e.body).detail || e.body; } catch { return e.body; } })();
        if (e.status === 402) throw new Error("Your RunPod balance is too low to rent this pod.");
        if (e.status === 400 && /no longer any instances|not enough|no available|capacity/i.test(String(detail))) throw new NoCapacity(`RunPod has no free ${req.offer.gpuName} right now (someone took the last one after the list loaded).`);
        if (e.status === 400) throw new Error(`RunPod could not place this pod right now (${String(detail).slice(0, 160)}). Try another machine.`);
      }
      throw e;
    }
  },

  async get(id) {
    try { return toInstance(await call(`/pods/${id}`)); }
    catch (e) { if (e instanceof HttpError && e.status === 404) return { id, status: "gone" }; throw e; }
  },

  async list() {
    const r = await call<any>("/pods");
    return (Array.isArray(r) ? r : r.pods || []).map(toInstance);
  },

  async stop(id) { await call(`/pods/${id}/action`, { method: "POST", body: { action: "stop" } }); },
  async start(id) {
    try { await call(`/pods/${id}/action`, { method: "POST", body: { action: "start" } }); }
    catch (e) {
      if (e instanceof HttpError && (e.status === 400 || e.status === 409 || e.status === 500)) {
        throw new Error("RunPod could not restart this pod. Someone else is probably using its GPU now. Delete it and start a new one.");
      }
      throw e;
    }
  },
  async destroy(id) {
    try { await call(`/pods/${id}`, { method: "DELETE" }); }
    catch (e) { if (!(e instanceof HttpError && e.status === 404)) throw e; }
  },

  async billed(id, sinceMs) {
    const start = new Date(sinceMs - 3600_000).toISOString();
    const r = await call<any>(`/billing/pods?podId=${encodeURIComponent(id)}&bucketSize=hour&startTime=${encodeURIComponent(start)}`);
    const recs: any[] = r.records || r.data || (Array.isArray(r) ? r : []);
    if (!recs.length) return undefined;
    return recs.reduce((a, x) => a + Number(x.totalAmount || 0), 0);
  },

  async registerKey(publicKey) {
    const cur = await call<{ keys: string[] }>("/account/ssh-keys");
    const keys = cur.keys || [];
    if (keys.some((k) => k.trim() === publicKey.trim())) return;
    // PUT replaces the whole list, so send the existing keys too.
    await call("/account/ssh-keys", { method: "PUT", body: { keys: [...keys, publicKey] } });
  },
};
