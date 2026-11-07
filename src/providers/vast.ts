import { http, HttpError } from "../core/net.js";
import { getKey, getVastSession } from "../core/keys.js";
import { NoCapacity, type Provider, type Offer, type Instance, type CreateRequest } from "./types.js";

/** Vast.ai REST API (paths and fields from vast-ai/vast-cli vast.py and docs.vast.ai, checked 2026-11-06). */
const API = "https://console.vast.ai/api/v0";

function headers(optional = false): Record<string, string> {
  const k = getVastSession() || getKey("vast");
  if (!k) { if (optional) return {}; throw new Error("No Vast.ai API key saved."); }
  return { authorization: `Bearer ${k}` };
}

export class VastTfaNeeded extends Error {}

async function call<T = any>(path: string, opts: { method?: string; body?: unknown } = {}): Promise<T> {
  try {
    return await http<T>(`${API}${path}`, { ...opts, headers: headers(), timeoutMs: 30_000 });
  } catch (e) {
    if (e instanceof HttpError && /two factor/i.test(e.body)) throw new VastTfaNeeded("Vast.ai asks for your two-factor code again. Open Keys & tokens → Vast.ai API key to log in.");
    throw e;
  }
}

function toOffer(o: any): Offer {
  const n = Number(o.num_gpus || 1);
  return {
    provider: "vast",
    id: String(o.id),
    gpuName: String(o.gpu_name || "GPU"),
    numGpus: n,
    vramPerGpuGb: Math.round((Number(o.gpu_ram || 0) / 1024) * 10) / 10,
    ramGb: Math.round(Number(o.cpu_ram || 0) / 1024),
    diskGb: Math.round(Number(o.disk_space || 0)),
    cpuName: o.cpu_name,
    pricePerHour: Number(o.dph_total || 0),
    storagePerGbMonth: Number(o.storage_cost || 0),
    downloadPerGb: Number(o.inet_down_cost || 0),   // $/GB (vast-cli: "internet download bandwidth cost in $/GB")
    inetDownMbps: Number(o.inet_down || 0),
    reliability: Number(o.reliability2 ?? o.reliability ?? 0),
    location: o.geolocation,
    cudaMax: Number(o.cuda_max_good || 0),
    computeCap: o.compute_cap ? Number(o.compute_cap) / 100 : undefined,
    gpuMemBw: o.gpu_mem_bw ? Number(o.gpu_mem_bw) : undefined,
    tflops: o.total_flops ? Number(o.total_flops) / n : undefined,
    pcieGbps: o.pcie_bw ? Number(o.pcie_bw) : undefined,
    machineId: o.machine_id,
  };
}

function toInstance(r: any): Instance {
  // actual_status is what the machine is doing now (null/"loading" while the image is pulled);
  // cur_state only says what was asked for, so it is not used here.
  const s = String(r.actual_status || "");
  const status: Instance["status"] =
    s === "running" ? "running" : s === "exited" || s === "stopped" || r.intended_status === "stopped" ? "stopped"
      : s === "offline" || s === "error" ? "error" : "starting";
  const ports = r.ports || {};
  const direct = ports["22/tcp"]?.[0]?.HostPort;
  const ssh = direct && r.public_ipaddr
    ? { host: String(r.public_ipaddr).trim(), port: Number(direct), user: "root" }
    : r.ssh_host ? { host: r.ssh_host, port: Number(r.ssh_port) + (String(r.image_runtype || "").includes("jupyter") ? 1 : 0), user: "root" } : undefined;
  const diskGb = Number(r.disk_space || 0);
  return {
    id: String(r.id),
    status,
    statusText: r.status_msg,
    ssh,
    pricePerHour: Number(r.dph_total || 0),
    stoppedPricePerHour: (Number(r.storage_cost || 0) * diskGb) / (30 * 24),
    startedAt: r.start_date ? Number(r.start_date) * 1000 : undefined,
    label: r.label,
    gpuName: r.gpu_name,
    numGpus: r.num_gpus,
    // inet_down_billed / inet_up_billed are kB (an instance that downloaded a 4.6 GB model showed
    // 4,843,853, and its invoice line read "download charge 4.600"); inet_*_cost is $/GB.
    downloadCost: r.inet_down_billed === undefined ? undefined
      : (Number(r.inet_down_billed || 0) / 1e6) * Number(r.inet_down_cost || 0) + (Number(r.inet_up_billed || 0) / 1e6) * Number(r.inet_up_cost || 0),
  };
}

/** The volume attached to a machine (GET /volumes/?owner=me&type=all_volume, matched on its "instances"). */
async function volumeFor(instanceId: string, tries = 6): Promise<string | undefined> {
  for (let i = 0; i < tries; i++) {
    try {
      const r = await call<{ volumes: any[] }>("/volumes/?owner=me&type=all_volume");
      const v = (r.volumes || []).find((x) => (x.instances || []).map(String).includes(String(instanceId)));
      if (v) return String(v.id);
    } catch { /* try again */ }
    if (i < tries - 1) await new Promise((res) => setTimeout(res, 3000));
  }
  return undefined;
}

export const vast: Provider = {
  id: "vast",
  name: "Vast.ai",
  separateDisk: true,
  stopNote: "The GPU charge stops. You keep paying for the disk, usually at a higher rate while stopped. Someone else may rent the GPU meanwhile, and then the machine cannot restart.",
  deleteNote: "The machine and its data are gone, and its charges stop.",
  workDir: "/workspace/spunup",

  async search({ minVramGb, minDiskGb, minCc, maxGpus, fastest }) {
    const q: any = {
      verified: { eq: true }, external: { eq: false }, rentable: { eq: true }, rented: { eq: false },
      gpu_total_ram: { gte: Math.ceil(minVramGb * 1000) },  // query units: GB × 1000 (offers_mult)
      disk_space: { gte: Math.ceil(minDiskGb) },
      reliability2: { gte: 0.95 },
      cuda_max_good: { gte: 12.4 },
      num_gpus: { lte: maxGpus || 8 },
      // Any offer field can be sorted on (vast.py: `-o 'total_flops-'`); gpu_mem_bw is GPU memory speed in GB/s.
      // Rows count toward a daily search quota (vast.py: a quota 429 "lasts until UTC midnight"), so ask for few:
      // the 64 cheapest are plenty to pick from, and the 16 fastest bring in the top end.
      order: fastest ? [["gpu_mem_bw", "desc"]] : [["dph_total", "asc"]], type: "on-demand", limit: fastest ? 16 : 64, allocated_storage: Math.ceil(minDiskGb),
    };
    if (minCc) q.compute_cap = { gte: Math.round(minCc * 100) };
    // Offer search works without a key; the key is only needed to rent.
    const r = await http<{ offers: any[] }>(`${API}/bundles/`, { method: "POST", body: q, headers: headers(true), timeoutMs: 30_000 }).catch((e) => {
      if (e instanceof HttpError && e.status === 429 && /quota/i.test(e.body))
        throw new Error("Vast.ai's daily limit on machine searches is used up. It resets at midnight UTC; RunPod works meanwhile.");
      throw e;
    });
    return (r.offers || []).map(toOffer);
  },

  async create(req: CreateRequest) {
    const body: any = {
      client_id: "me", image: req.image, env: {}, disk: req.diskGb, label: req.label,
      runtype: "ssh_direc ssh_proxy", onstart: "mkdir -p /workspace/spunup",
      cancel_unavail: true,
    };
    let volumeId: string | undefined;
    if (req.keepDiskSeparately && req.offer.machineId) {
      try {
        const vs = await call<{ offers: any[] }>("/volumes/search/", { method: "POST", body: {
          machine_id: { eq: req.offer.machineId }, disk_space: { gte: req.diskGb }, order: [["storage_cost", "asc"]], limit: 5,
        } });
        const v = vs.offers?.[0];
        if (v) {
          // No "name": Vast rejects some names and the field is optional (vast-cli --volume-label).
          body.volume_info = { create_new: true, volume_id: v.id, size: req.diskGb, mount_path: "/workspace/spunup" };
          body.disk = 16; // the model lives on the volume; the machine disk only holds the image
        }
      } catch { /* volumes not offered on this machine: plain disk */ }
    }
    const taken = (why: string) => new NoCapacity(`That Vast.ai machine was rented by someone else after the list loaded${why ? ` (${why})` : ""}.`);
    const TAKEN = /taken|no_such_ask|not available|unavailable|already rented/i;
    let r: any;
    try { r = await call<any>(`/asks/${req.offer.id}/`, { method: "PUT", body }); }
    catch (e) { if (e instanceof HttpError && TAKEN.test(e.body)) throw taken(e.message); throw e; }
    if (!r?.success || !r.new_contract) {
      const why = String(r?.msg || r?.error || "");
      if (!why || TAKEN.test(why)) throw taken(why);
      throw new Error(why);
    }
    const id = String(r.new_contract);
    if (body.volume_info) volumeId = await volumeFor(id);
    try { await call(`/instances/${id}/ssh/`, { method: "POST", body: { ssh_key: req.publicKey } }); } catch { /* account key also applies */ }
    return { id, volumeId };
  },

  async get(id) {
    try {
      const r = await call<{ instances: any }>(`/instances/${id}/?owner=me`);
      if (!r.instances) return { id, status: "gone" };
      const inst = toInstance(r.instances);
      const vol = r.instances.volume_info || r.instances.volumes;
      if (Array.isArray(vol) && vol[0]?.id) inst.volumeId = String(vol[0].id);
      return inst;
    } catch (e) {
      if (e instanceof HttpError && e.status === 404) return { id, status: "gone" };
      throw e;
    }
  },

  async list() {
    const out: Instance[] = [];
    let after: string | undefined;
    for (let i = 0; i < 20; i++) {
      const params = new URLSearchParams({ select_filters: "{}", order_by: JSON.stringify([{ col: "id", dir: "asc" }]), limit: "25", ...(after ? { after_token: after } : {}) });
      const r = await http<any>(`https://console.vast.ai/api/v1/instances/?${params}`, { headers: headers(), timeoutMs: 20000 });
      out.push(...(r.instances || []).map(toInstance));
      after = r.next_token;
      if (!after) break;
    }
    return out;
  },

  async stop(id) { await call(`/instances/${id}/`, { method: "PUT", body: { state: "stopped" } }); },
  async start(id) { await call(`/instances/${id}/`, { method: "PUT", body: { state: "running" } }); },

  async destroy(id, opts) {
    // Find the machine's volume before deleting it: afterwards the volume no longer lists the machine.
    const volumeId = opts?.volumeId || (opts?.keepDisk ? undefined : await volumeFor(id, 1));
    // Already gone (an earlier try deleted it): carry on to the volume, or it would be left behind, still billing.
    await call(`/instances/${id}/`, { method: "DELETE", body: {} }).catch((e) => { if (!(e instanceof HttpError && e.status === 404)) throw e; });
    if (volumeId && !opts?.keepDisk) await this.deleteDisk!(volumeId);
  },

  async attachKey(id, publicKey) {
    // Account keys only reach machines created after they were added, so an existing machine gets it directly.
    await call(`/instances/${id}/ssh/`, { method: "POST", body: { ssh_key: publicKey } });
  },

  async deleteDisk(volumeId) {
    // Right after its machine is deleted the volume can still be attached for a moment.
    for (let i = 0; ; i++) {
      try { await call(`/volumes/?id=${encodeURIComponent(volumeId)}`, { method: "DELETE" }); return; }
      catch (e) { if (e instanceof HttpError && e.status === 404) return; if (i >= 4) throw e; await new Promise((r) => setTimeout(r, 5000)); }
    }
  },

  async registerKey(publicKey) {
    try {
      const keys = await call<any>("/ssh/");
      const list: any[] = Array.isArray(keys) ? keys : keys?.ssh_keys || [];
      if (list.some((k) => String(k.public_key || k.ssh_key || "").trim() === publicKey.trim())) return;
    } catch { /* fall through and add */ }
    await call("/ssh/", { method: "POST", body: { ssh_key: publicKey } });
  },
};
