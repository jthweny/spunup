/** The set-up page: every setting on one page, what each one depends on, and how the rest re-tunes
 *  when one changes.
 *
 *  Dependence, top to bottom (changing one re-tunes the ones below it):
 *    Model (+ its LoRA adapter)
 *      → Where: Local / Vast.ai / RunPod (+ RunPod's cloud)
 *        → Runtime ⇄ Size       tied by file format: GGUF → llama.cpp, safetensors → vLLM / SGLang, MLX → MLX
 *          → Machine            cloud: the live listings that can run that runtime and size
 *            → Context          Max = what fits on that machine, or on this computer
 *              → tokens/s, prefill, price, download
 *
 *  Each setting is either ✦ smart (spunup picks it, and picks again when something it depends on
 *  changes) or set by you (it stays put). An option that clashes with something you set is greyed
 *  out, with the reason. An option that only moves ✦ settings says what it will move. */
import type { Recipe, Quant } from "./recipes.js";
import { hardwareSummary, type Hardware } from "./hardware.js";
import type { RuntimeId, Where } from "./store.js";
import { computeFit, maxContextThatFits, type Fit, type Speed } from "./fit.js";
import { provider, type Offer, type RunpodCloud } from "../providers/index.js";
import { RUNTIME_NAMES } from "../runtimes/index.js";
import { perHour } from "./cost.js";
import {
  bestRuntime, runtimeChoices, preferredQuant, localOptions, quantsFor, cloudFits, bestValue, recommendedOption,
  maxContextOnOffer, needsFor, quality, valueScore, setupLabel, cloudMiss, FORMAT_FOR, PRETTY_GPU, type CloudOption,
} from "./plan.js";

export type Sort = "cheapest" | "fastest" | "value";
export const WHERE_NAME: Record<Where, string> = { local: "Local", vast: "Vast.ai", runpod: "RunPod" };

/** What you set yourself. Anything left out is ✦ smart. */
export interface Locks {
  runtime?: RuntimeId;   // ✦ = the best runtime for this model here (spec 7.1 "Choose best")
  size?: string;         // a quant id; ✦ = Smart squeeze
  machine?: Offer;       // cloud: the machine you picked, pinned (it stays even when a new search leaves it out); ✦ = Smart squeeze's machine
  context?: number;      // ✦ = Max: the most that fits
}

/** Everything the settings are worked out from. */
export interface Setting {
  r: Recipe;              // the model; its sizes are only the ones the LoRA adapter in use (if any) works with
  hw: Hardware;
  where: Where;
  locks: Locks;
  allowed?: RuntimeId[];  // runtimes that can load the LoRA adapter in use
  adapterGb: number;
}

export const fmt = (n: number) => Math.round(n).toLocaleString("en-GB");
/** "Q5_K_M" where the file has a precise name, else the size label. */
export const quantName = (q: Quant) => q.quant || q.label;
export const sizeById = (r: Recipe, id?: string) => (id ? r.quants.find((q) => q.id === id) : undefined);
export const gpuName = (o: Offer) => `${o.numGpus}× ${PRETTY_GPU(o.gpuName)}`;
const takes = (q: Quant, rt: RuntimeId) => q.format === FORMAT_FOR[rt];
const orList = (xs: string[]) => (xs.length > 1 ? `${xs.slice(0, -1).join(", ")} or ${xs[xs.length - 1]}` : xs[0] || "");

/** Runtimes that work in this place and can load the adapter, ready-to-use ones first. */
export function usableRuntimes(s: Setting, where: Where = s.where): RuntimeId[] {
  return runtimeChoices(s.r, where, s.hw).filter((c) => c.ok && (!s.allowed || s.allowed.includes(c.id)))
    .sort((a, b) => Number(!!a.why) - Number(!!b.why)).map((c) => c.id);
}

/** The runtime in use: yours if it still works here with the size, else ✦ the best for the model
 *  (on a machine you picked: the best one that can run something on it). */
export function runtimeFor(s: Setting): RuntimeId | undefined {
  const ok = usableRuntimes(s);
  const size = sizeById(s.r, s.locks.size);
  const loads = (rt: RuntimeId) => !size || takes(size, rt);
  if (s.locks.runtime && ok.includes(s.locks.runtime) && loads(s.locks.runtime)) return s.locks.runtime;
  const pin = s.where !== "local" ? s.locks.machine : undefined;
  const runs = (rt: RuntimeId) => !pin || onMachine(s, rt, pin).some((c) => !size || c.quant.id === size.id);
  const best = bestRuntime(size ? { ...s.r, quants: [size] } : s.r, s.where, s.hw);
  if (best && ok.includes(best) && loads(best) && runs(best)) return best;
  return ok.find((rt) => loads(rt) && runs(rt)) ?? (best && ok.includes(best) && loads(best) ? best : ok.find(loads));
}

/** Every size that fits the machine you picked, with this runtime (at the context the search checks). */
const onMachine = (s: Setting, rt: RuntimeId, o: Offer, ctx = searchCtx(s)) => cloudFits(s.r, rt, ctx, o, s.adapterGb);

/** Why a size can't run on the machine you picked, in plain words. */
function missOnMachine(s: Setting, rt: RuntimeId, o: Offer, q: Quant, ctx = searchCtx(s)): string {
  const m = cloudMiss(s.r, rt, ctx, o, q);
  return m === "format" ? `the machine you picked is too old for ${q.label}` : m === "disk" ? "not enough disk on the machine you picked"
    : m === "gpus" ? `${RUNTIME_NAMES[rt]} needs 1, 2, 4 or 8 GPUs; the machine you picked has ${o.numGpus}` : "too big for the machine you picked";
}

/** Only MLX files are left (the model ships only those, or the LoRA adapter only works with them). */
const onlyMlx = (s: Setting) => s.r.quants.length > 0 && s.r.quants.every((q) => q.format === "mlx");
const MLX_ONLY = "This model only has MLX files, which run on Apple Silicon Macs. On this computer and on rented GPUs it can't run yet.";

// ---------------------------------------------------------------- live machines
/** A machine and every size that fits on it (best quality first). */
export interface MachineRow { offer: Offer; fits: CloudOption[] }
/** A machine with the size it would run: the one you set, or the best value on that machine. */
export type Row = MachineRow & { option: CloudOption };

/** Context the machine search checks against: what you set, or 32K, which keeps machines on the list
 *  that coding agents can use. Max context is then worked out on the machine itself. */
export const baseCtx = (r: Recipe) => Math.min(32768, r.max_context || 32768);
export const searchCtx = (s: Pick<Setting, "r" | "locks">) => s.locks.context ?? baseCtx(s.r);

/** Live machines for this model. One search per size: each returns the cheapest machines that size
 *  fits on, so the bigger machines only the larger sizes need are not cut off by the provider's result
 *  limit. Kept for 10 minutes, so flipping between settings is instant and the picks hold still. */
const searches = new Map<string, { at: number; rows: Promise<MachineRow[]> }>();
export function findMachines(r: Recipe, rt: RuntimeId, ctx: number, where: "vast" | "runpod", cloud: RunpodCloud, adapterGb: number): Promise<MachineRow[]> {
  const key = JSON.stringify([r.id, r.quants.map((q) => q.id), rt, ctx, where, cloud, adapterGb]);
  const hit = searches.get(key);
  if (hit && Date.now() - hit.at < 10 * 60_000) return hit.rows;
  const rows = (async () => {
    const needs = new Map<number, ReturnType<typeof needsFor>>();
    for (const q of quantsFor(r, rt)) { const n = needsFor(r, rt, ctx, q); needs.set(Math.ceil(n.vramGb), n); }
    // For each size: the cheapest machines, and (Vast) the ones with the fastest GPU memory, so the top end
    // is always on the list too. RunPod lists every GPU type in one go, so it needs no second search.
    const found = await Promise.allSettled([...needs.values()].flatMap((n) => (where === "vast" ? [false, true] : [false]).map((fastest) =>
      provider(where).search({ minVramGb: n.vramGb, minDiskGb: n.diskGb, maxGpus: 8, minCc: rt === "llamacpp" ? 6.1 : 8.0, cloud, fastest }))));
    const offers = new Map<string, Offer>();
    for (const f of found) if (f.status === "fulfilled") for (const o of f.value) {
      const seen = offers.get(o.id);
      if (!seen || o.pricePerHour < seen.pricePerHour) offers.set(o.id, o);
    }
    const failed = found.find((f) => f.status === "rejected");
    if (!offers.size && failed) throw (failed as PromiseRejectedResult).reason;
    return [...offers.values()].map((offer) => ({ offer, fits: cloudFits(r, rt, ctx, offer, adapterGb) })).filter((m) => m.fits.length);
  })();
  searches.set(key, { at: Date.now(), rows });
  rows.catch(() => { if (searches.get(key)?.rows === rows) searches.delete(key); });
  return rows;
}

// ---------------------------------------------------------------- the settings in use
export interface Resolved {
  runtime?: RuntimeId;
  quant?: Quant;              // the size in use
  speed?: Speed;
  fit?: Fit;                  // Local: how it sits in memory
  gpuMax?: number;            // Local: most context that keeps all of it on the GPU (when Max spills into RAM)
  rows?: Row[];               // cloud: machines that can run it, each with the size it would use
  smart?: CloudOption;        // cloud: Smart squeeze's machine and size
  chosen?: CloudOption;       // cloud: the machine in use (yours, or Smart squeeze's)
  machineGone?: boolean;      // cloud: the machine you picked is no longer listed
  ctxMax: number;             // the most context that fits (up to the model's own maximum)
  smartCtx?: number;          // Local: ✦ context: the most that keeps the model all on the GPU (when that is enough)
  context: number;            // context in use
  clamped?: boolean;          // the context you set does not fit here, so it was shortened
  problem?: string;           // why nothing can run yet, in plain words
}

export function resolveLocal(s: Setting): Resolved {
  const rt = runtimeFor(s);
  if (!rt) return { ctxMax: 0, context: 0, problem: onlyMlx(s) ? MLX_ONLY : "No runtime can run this model on this computer. Try Vast.ai or RunPod." };
  const want = Math.min(s.locks.context ?? 8192, s.r.max_context || 8192);
  const q = s.locks.size ? sizeById(s.r, s.locks.size) : preferredQuant(s.r, rt, s.hw, want)?.quant;
  if (!q) {
    return { runtime: rt, ctxMax: 0, context: 0, problem: s.locks.context
      ? `No size fits this computer with ${fmt(s.locks.context)} tokens of context. Pick a shorter context, or rent a machine.`
      : "Even the smallest size needs more memory than this computer has free. Try Vast.ai or RunPod." };
  }
  const ctxMax = maxContextThatFits(s.r, q, s.hw, rt);
  // ✦ context keeps a size that fits the GPU all on the GPU: spilling the cache into RAM runs at CPU speed.
  // The longer, slower lengths are still there (Max, or a number you pick).
  const onGpu = !s.hw.apple ? gpuContext(s, q, rt) : 0;
  const smartCtx = onGpu >= Math.min(8192, ctxMax) && onGpu < ctxMax ? onGpu : ctxMax;
  const context = Math.min(s.locks.context ?? smartCtx, ctxMax);
  const o = ctxMax ? localOptions({ ...s.r, quants: [q] }, rt, s.hw, context, s.adapterGb)[0] : undefined;
  if (!o?.fit.ok) return { runtime: rt, quant: q, ctxMax, context, problem: `${q.label} doesn't fit this computer. Pick a smaller size, or rent a machine.` };
  return { runtime: rt, quant: q, speed: o.speed, fit: o.fit, ctxMax, smartCtx, context, clamped: !!s.locks.context && s.locks.context > ctxMax,
    gpuMax: o.fit.setup === "mixed" ? gpuContext(s, q, rt) : undefined };
}

/** Most context that keeps the whole model on the GPU. */
function gpuContext(s: Setting, q: Quant, rt: RuntimeId): number {
  const onGpu = (c: number) => { const f = computeFit(s.r, q, c, s.hw, rt); return f.ok && f.setup === "gpu"; };
  if (!onGpu(1024)) return 0;
  let lo = 1024, hi = s.r.max_context || 32768;
  while (hi - lo > 1024) { const mid = Math.floor((lo + hi) / 2); if (onGpu(mid)) lo = mid; else hi = mid; }
  return Math.floor(lo / 1024) * 1024;
}

export function resolveCloud(s: Setting, machines: MachineRow[] | undefined, skip?: ReadonlySet<string>): Resolved {
  const rt = runtimeFor(s);
  if (!rt) return { ctxMax: 0, context: 0, problem: onlyMlx(s) ? MLX_ONLY : "This model can't run on rented GPUs with these settings (MLX runs on Macs only)." };
  if (!machines) return { runtime: rt, quant: sizeById(s.r, s.locks.size), ctxMax: 0, context: 0 };
  const size = s.locks.size;
  // The machine you picked stays on the list: a new search (another runtime or context) only returns
  // the cheapest machines for each size, which can leave a big one out.
  const pin = s.locks.machine;
  let pool = machines;
  if (pin && !skip?.has(pin.id) && !machines.some((m) => m.offer.id === pin.id)) {
    const fits = onMachine(s, rt, pin);
    if (fits.length) pool = [...machines, { offer: pin, fits }];
  }
  const rows = pool.filter((m) => !skip?.has(m.offer.id))
    .map((m) => ({ ...m, option: size ? m.fits.find((c) => c.quant.id === size) : bestValue(m.fits) }))
    .filter((m): m is Row => !!m.option);
  const smart = recommendedOption(rows.map((m) => m.option));
  const mine = pin ? rows.find((m) => m.offer.id === pin.id) : undefined;
  const chosen = mine?.option ?? smart;
  const machineGone = !!s.locks.machine && !mine;
  if (!chosen) {
    const tries = [size ? "a smaller size" : "", s.locks.context ? "a shorter context" : "", s.where === "vast" ? "RunPod" : "Vast.ai"].filter(Boolean);
    return { runtime: rt, rows, machineGone, ctxMax: 0, context: 0,
      problem: `No machine on ${WHERE_NAME[s.where]} can run ${size ? `size ${sizeById(s.r, size)?.label}` : "this model"}${s.locks.context ? ` with ${fmt(s.locks.context)} tokens of context` : ""} right now. Try ${orList(tries)}.` };
  }
  const native = s.r.max_context || 32768;
  // A context you typed may go past the model's own maximum (extended context); check it against the machine too.
  const room = maxContextOnOffer(s.r, rt, chosen.quant, chosen.offer, Math.max(native, s.locks.context || 0));
  const ctxMax = Math.min(room, native);
  const context = s.locks.context ? Math.min(s.locks.context, room) : ctxMax;
  return { runtime: rt, quant: chosen.quant, speed: chosen.speed, rows, smart, chosen, machineGone, ctxMax, context, clamped: !!s.locks.context && s.locks.context > room };
}

// ---------------------------------------------------------------- drop-down options
/** One option in a drop-down. `why` = why it is greyed out; `note` = what picking it also does. */
export interface Choice<T> { value: T; label: string; ok: boolean; why?: string; note?: string; speed?: Speed; price?: number; preview?: Quant }

export function runtimeOptions(s: Setting): Choice<RuntimeId | undefined>[] {
  const best = runtimeFor({ ...s, locks: { ...s.locks, runtime: undefined } });
  const size = sizeById(s.r, s.locks.size);
  const out: Choice<RuntimeId | undefined>[] = [{ value: undefined, label: best ? `Choose best (${RUNTIME_NAMES[best]})` : "Choose best", ok: !!best, note: "recommended", why: "nothing can run this here" }];
  for (const c of runtimeChoices(s.r, s.where, s.hw)) {
    const label = RUNTIME_NAMES[c.id];
    if (!c.ok) out.push({ value: c.id, label, ok: false, why: c.why });
    else if (s.allowed && !s.allowed.includes(c.id)) out.push({ value: c.id, label, ok: false, why: "can't load this LoRA adapter" });
    else if (size && !takes(size, c.id)) out.push({ value: c.id, label, ok: false, why: `no ${size.label} file for it (you picked ${size.label})` });
    else if (s.where !== "local" && s.locks.machine && !onMachine(s, c.id, s.locks.machine).some((x) => !size || x.quant.id === size.id))
      out.push({ value: c.id, label, ok: false, why: "can't run this model on the machine you picked" });
    else out.push({ value: c.id, label, ok: true, note: c.why });
  }
  return out;
}

/** Sizes that can run here. Each says how it would run: on this computer, or on the best machine for it. */
export function sizeOptions(s: Setting, res: Resolved, machines?: MachineRow[], others: Partial<Record<RuntimeId, MachineRow[]>> = {}, skip?: ReadonlySet<string>): Choice<string | undefined>[] {
  const rt = res.runtime;
  const pool = (r?: RuntimeId) => (!r ? undefined : r === rt ? machines : others[r])?.filter((m) => !skip?.has(m.offer.id));
  // Smart squeeze's pick, with the size left to it (the runtime may move with it).
  const free: Setting = { ...s, locks: { ...s.locks, size: undefined } };
  const freeRes = s.where === "local" ? resolveLocal(free) : resolveCloud(free, pool(runtimeFor(free)), skip);
  const out: Choice<string | undefined>[] = [{ value: undefined, label: "Smart squeeze", ok: true, note: "this model's sweet spot",
    preview: freeRes.quant, speed: freeRes.speed, price: freeRes.chosen?.offer.pricePerHour }];
  const here = usableRuntimes(s);
  const sizes = s.r.quants.filter((q) => q.size_gb && here.some((x) => takes(q, x)))
    .sort((a, b) => quality(b) - quality(a) || (b.size_gb || 0) - (a.size_gb || 0));
  for (const q of sizes) {
    const twin = sizes.some((x) => x !== q && x.label === q.label);
    const label = `${q.label}${q.quant && q.quant !== q.label ? ` (${q.quant})` : ""}${twin ? ` · ${q.format === "gguf" ? "GGUF" : q.format === "mlx" ? "MLX" : "safetensors"}` : ""}`;
    const base = { value: q.id, label };
    const qrt = rt && takes(q, rt) ? rt : runtimeFor({ ...s, locks: { ...s.locks, runtime: undefined, size: q.id } });
    const own = s.locks.runtime;
    if (own && !takes(q, own)) { out.push({ ...base, ok: false, why: `needs ${qrt ? RUNTIME_NAMES[qrt] : "another runtime"} (you picked ${RUNTIME_NAMES[own]})` }); continue; }
    if (!qrt) { out.push({ ...base, ok: false, why: "no runtime here can load it" }); continue; }
    const swap = qrt !== rt ? `switches the runtime to ${RUNTIME_NAMES[qrt]}` : undefined;
    if (s.where === "local") {
      const ctx = Math.min(s.locks.context ?? 8192, s.r.max_context || 8192);
      const o = localOptions({ ...s.r, quants: [q] }, qrt, s.hw, ctx, s.adapterGb)[0];
      if (!o) out.push({ ...base, ok: false, why: "this computer's GPU is too old for it" });
      else if (!o.fit.ok) out.push({ ...base, ok: false, why: s.locks.context ? `too big for this computer with ${fmt(ctx)} tokens` : "too big for this computer" });
      else out.push({ ...base, ok: true, speed: o.speed, note: [setupLabel(o.fit.setup), swap].filter(Boolean).join(" · ") });
      continue;
    }
    const pin = s.locks.machine;
    if (pin) {   // the machine you picked: every size that fits it, whichever runtime that size needs
      const c = onMachine(s, qrt, pin).find((x) => x.quant.id === q.id);
      out.push(c ? { ...base, ok: true, speed: c.speed, price: c.offer.pricePerHour, note: swap } : { ...base, ok: false, why: missOnMachine(s, qrt, pin, q) });
      continue;
    }
    const live = pool(qrt);
    if (!live) { out.push({ ...base, ok: true, note: swap }); continue; }
    const best = recommendedOption(live.map((m) => m.fits.find((x) => x.quant.id === q.id)).filter((x): x is CloudOption => !!x));
    out.push(best
      ? { ...base, ok: true, speed: best.speed, price: best.offer.pricePerHour, note: [`on ${gpuName(best.offer)}`, swap].filter(Boolean).join(" · ") }
      : { ...base, ok: false, why: `no machine on ${WHERE_NAME[s.where]} can run it${s.locks.context ? " with this context" : ""} right now` });
  }
  return out;
}

const PRESETS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
const short = (n: number) => (n % 1048576 === 0 ? `${n / 1048576}M` : n % 1024 === 0 ? `${n / 1024}K` : fmt(n));

/** The real Max: the longest context you can get here with what you set; Smart squeeze re-tunes the rest.
 *  This computer: the longest any allowed size fits. A machine you picked: the most it holds with a size that
 *  fits. Otherwise: the model's own maximum if a machine on the list holds it (`atFull` = the machines found
 *  for the full context), else the most any listed machine holds. */
export function contextMax(s: Setting, res: Resolved, atFull?: MachineRow[]): number | undefined {
  const rt = res.runtime;
  if (!rt) return undefined;
  const size = s.locks.size;
  if (s.where === "local") {
    let best = 0;
    for (const q of quantsFor(s.r, rt).filter((x) => !size || x.id === size))
      if (localOptions({ ...s.r, quants: [q] }, rt, s.hw, 1024, s.adapterGb).length) best = Math.max(best, maxContextThatFits(s.r, q, s.hw, rt));
    return best || undefined;
  }
  const room = (o: Offer, fits: CloudOption[]) => Math.max(0, ...fits.filter((c) => !size || c.quant.id === size).map((c) => maxContextOnOffer(s.r, rt, c.quant, o)));
  if (s.locks.machine && !res.machineGone) return room(s.locks.machine, onMachine(s, rt, s.locks.machine, 1024)) || undefined;
  if (atFull?.some((m) => m.fits.some((c) => !size || c.quant.id === size))) return s.r.max_context || 32768;
  if (!res.rows) return undefined;
  return Math.max(0, ...res.rows.map((m) => room(m.offer, m.fits))) || undefined;
}

export function contextOptions(s: Setting, res: Resolved, trueMax?: number): Choice<number | undefined | "type">[] {
  const native = s.r.max_context || 32768;
  const here = s.where === "local" ? "this computer" : "the picked machine";
  const rt = res.runtime;
  const one = (n: number) => contextChoice(s, res, n, trueMax);
  const smart = res.smartCtx ?? res.ctxMax;
  const out: Choice<number | undefined | "type">[] = [{ value: undefined, label: smart ? `Smart squeeze (${fmt(smart)})` : "Smart squeeze",
    ok: smart > 0 || (s.where !== "local" && !res.problem), note: s.where === "local" && smart < res.ctxMax ? "the most that stays all on the GPU" : `the most ${here} holds`, why: "nothing fits yet" }];
  if (trueMax) { const m = one(trueMax); out.push({ ...m, label: `Max (${fmt(trueMax)})`, note: m.note || "the longest it can go here" }); }
  for (const n of [...PRESETS.filter((x) => x < native), native]) if (n !== trueMax) out.push({ ...one(n), label: `${short(n)} (${fmt(n)})` });
  out.push({ value: "type", label: "Type a number…", ok: true });
  return out;
}

/** What picking this context length does: fine as it is, re-tunes the ✦ settings, or can't be done (and why).
 *  Used for the list and for a number you type, so both follow the same rules. */
export function contextChoice(s: Setting, res: Resolved, n: number, trueMax?: number): Omit<Choice<number>, "label"> {
  const rt = res.runtime;
  {
    const base = { value: n };
    if (!rt) return { ...base, ok: false, why: "nothing can run here yet" };
    if (s.where === "local") {
      if (s.locks.size && res.quant) {
        const f = computeFit(s.r, res.quant, n, s.hw, rt);
        return f.ok ? { ...base, ok: true, note: f.setup === "mixed" ? "part of it sits in RAM: slower" : undefined } : { ...base, ok: false, why: `doesn't fit this computer with ${res.quant.label}` };
      }
      const p = preferredQuant(s.r, rt, s.hw, n);
      return !p ? { ...base, ok: false, why: "too long for this computer" }
        : { ...base, ok: true, note: [p.quant.id !== res.quant?.id ? `Smart squeeze switches to ${p.quant.label}` : "", p.fit.setup === "mixed" ? "part of it sits in RAM: slower" : ""].filter(Boolean).join(" · ") || undefined };
    }
    if (!res.chosen || n <= res.ctxMax) return { ...base, ok: true };
    if (s.locks.machine && !res.machineGone) {
      // A smaller size can leave room for more context on the same machine.
      const fits = onMachine(s, rt, s.locks.machine, n);
      const want = s.locks.size ? fits.find((c) => c.quant.id === s.locks.size) : bestValue(fits);
      return want
        ? { ...base, ok: true, note: want.quant.id !== res.quant?.id ? `Smart squeeze switches to ${want.quant.label}` : undefined }
        : { ...base, ok: false, why: `doesn't fit the machine you picked${s.locks.size && res.quant ? ` with ${res.quant.label}` : ""} (max ${fmt(res.ctxMax)})` };
    }
    if (trueMax && n > trueMax) return { ...base, ok: false, why: `longer than any machine on ${WHERE_NAME[s.where]} holds right now` };
    return { ...base, ok: true, note: "needs a bigger machine: Smart squeeze picks one" };
  }
}

export function whereOptions(s: Setting, hasKey: (w: "vast" | "runpod") => boolean, best: Partial<Record<"vast" | "runpod", CloudOption>> = {}): Choice<Where>[] {
  const hw = hardwareSummary(s.hw);
  const rtL = runtimeFor({ ...s, where: "local", locks: {} });
  const pq = rtL ? preferredQuant(s.r, rtL, s.hw, Math.min(8192, s.r.max_context || 8192)) || preferredQuant(s.r, rtL, s.hw, 2048) : undefined;
  const out: Choice<Where>[] = [pq
    ? { value: "local", label: "Local (this computer)", ok: true, note: `${hw} · fits (${pq.quant.label}${pq.fit.setup === "mixed" ? ", GPU + RAM" : ""}) · free` }
    : { value: "local", label: "Local (this computer)", ok: false, why: `${hw} · ${rtL ? "too big for this computer" : "no runtime can run it here"}` }];
  for (const w of ["vast", "runpod"] as const) {
    const b = best[w];
    out.push(runtimeFor({ ...s, where: w, locks: {} })
      ? { value: w, label: WHERE_NAME[w], ok: true, note: ["rented GPUs", b ? `from ${perHour(b.offer.pricePerHour)} (${b.quant.label})` : "", hasKey(w) ? "" : "needs your API key"].filter(Boolean).join(" · ") }
      : { value: w, label: WHERE_NAME[w], ok: false, why: "rented GPUs · can't run this model (MLX runs on Macs only)" });
  }
  return out;
}

/** Where Smart squeeze starts: this computer when the model fits it well (all on the GPU, at Q4 or
 *  better), as that is free; else the provider with the most quality for the price. */
export function smartWhere(s: Setting, found: Partial<Record<"vast" | "runpod", MachineRow[]>>): { where: Where; why: string; best: Partial<Record<"vast" | "runpod", CloudOption>> } {
  const best: Partial<Record<"vast" | "runpod", CloudOption>> = {};
  for (const w of ["vast", "runpod"] as const) {
    const b = recommendedOption((found[w] || []).map((m) => bestValue(m.fits)!));
    if (b) best[w] = b;
  }
  const rtL = runtimeFor({ ...s, where: "local", locks: {} });
  const pq = rtL ? preferredQuant(s.r, rtL, s.hw, Math.min(8192, s.r.max_context || 8192)) : undefined;
  if (pq && (pq.fit.setup === "gpu" || pq.fit.setup === "mac") && quality(pq.quant) >= 72)
    return { where: "local", why: `It fits this computer at ${quantName(pq.quant)}, all on the GPU, and it's free`, best };
  const cloud = (["vast", "runpod"] as const).filter((w) => best[w]).sort((a, b) => valueScore(best[b]!) - valueScore(best[a]!))[0];
  if (cloud) {
    const b = best[cloud]!;
    const n = (found.vast?.length || 0) + (found.runpod?.length || 0);
    return { where: cloud, why: `Best quality for the price: ${quantName(b.quant)} on ${gpuName(b.offer)} at ${perHour(b.offer.pricePerHour)}, out of ${fmt(n)} machines`, best };
  }
  if (pq) return { where: "local", why: `It fits this computer at ${quantName(pq.quant)}${pq.fit.setup === "mixed" ? " (part of it in RAM, so slower)" : ""}, and it's free`, best };
  if (!rtL && onlyMlx(s)) return { where: "local", why: MLX_ONLY.replace(/\.$/, ""), best };
  return { where: "vast", why: "It's too big for this computer, so it runs on a rented GPU", best };
}

/** After Where or the LoRA adapter changes: keep what you set where it still works, and say what was let go. */
export function keepLocks(s: Setting): { locks: Locks; dropped: string[] } {
  const l: Locks = { ...s.locks };
  const dropped: string[] = [];
  const ok = usableRuntimes(s);
  if (l.runtime && !ok.includes(l.runtime)) { dropped.push(RUNTIME_NAMES[l.runtime]); l.runtime = undefined; }
  const size = sizeById(s.r, l.size);
  if (l.size && (!size || !ok.some((rt) => takes(size, rt) && (!l.runtime || rt === l.runtime)))) { dropped.push(size ? `size ${size.label}` : "your size"); l.size = undefined; }
  return { locks: l, dropped };
}

// ---------------------------------------------------------------- what moved
/** What the page shows, to tell what moved after a change. */
export interface Snap { where?: string; runtime?: string; size?: string; machine?: string; context?: string; speed?: string; price?: string }
export const SNAP_NAME: Record<keyof Snap, string> = { where: "Where", runtime: "Runtime", size: "Size", machine: "Machine", context: "Context", speed: "tokens/s", price: "Price" };

export function snap(where: Where, res: Resolved): Snap {
  return {
    where: WHERE_NAME[where],
    runtime: res.runtime && RUNTIME_NAMES[res.runtime],
    size: res.quant && quantName(res.quant),
    // Same-named machines are common (several 1× RTX 4090), so the price tells them apart.
    machine: where === "local" ? "this computer" : res.chosen && `${gpuName(res.chosen.offer)} (${perHour(res.chosen.offer.pricePerHour)})`,
    context: res.context ? fmt(res.context) : undefined,
    speed: res.speed ? `${Math.round(res.speed.tpsMin)}–${Math.round(res.speed.tpsMax)}` : undefined,
    price: where === "local" ? "free" : res.chosen && perHour(res.chosen.offer.pricePerHour),
  };
}

export const moved = (a: Snap, b: Snap) => (Object.keys(SNAP_NAME) as (keyof Snap)[]).filter((k) => a[k] !== b[k]);
