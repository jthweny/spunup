/** The set-up page, right after the model: every setting on one page, ✦ smart-tuned to start with.
 *  Change anything in [ ▾ ] and the ✦ settings re-tune to match (the rules live in core/tune.ts). */
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import { C } from "../ui/theme.js";
import { Tps, Prefill, Spinner, TextInput } from "../ui/components.js";
import type { Recipe, Quant } from "../core/recipes.js";
import { hardwareSummary, type Hardware } from "../core/hardware.js";
import type { RuntimeId, Where } from "../core/store.js";
import { computeFit, type Fit, type Speed } from "../core/fit.js";
import type { Offer, RunpodCloud } from "../providers/index.js";
import { getKey } from "../core/keys.js";
import { perHour, money } from "../core/cost.js";
import { RUNTIME_NAMES, RUNTIME_BLURB } from "../runtimes/index.js";
import { quality, setupLabel, preferredQuant, maxContextOnOffer, type CloudOption } from "../core/plan.js";
import { adapterName, adapterLabel, KIND_NAME, type Adapter, type LoraUse } from "../core/lora.js";
import {
  findMachines, resolveLocal, resolveCloud, runtimeFor, usableRuntimes, runtimeOptions, sizeOptions, contextOptions, contextChoice, contextMax, whereOptions,
  keepLocks, searchCtx, snap, moved, SNAP_NAME, quantName, sizeById, gpuName, fmt, WHERE_NAME,
  type Locks, type Sort, type Setting, type MachineRow, type Row, type Choice, type Resolved, type Snap,
} from "../core/tune.js";

export interface TuneState { where: Where; whereSmart: boolean; cloud: RunpodCloud; locks: Locks; sort: Sort }
/** What the page hands on when you continue. */
export interface Final { where: Where; runtime: RuntimeId; quant: Quant; context: number; speed?: Speed; fit?: Fit; offer?: Offer; downloadMin?: number; downloadCost?: number }

const RUNTIME_NOTE = "Runtime = the software the model runs on.";
const SIZE_NOTE = "Size = how compressed the model is. Smaller is cheaper and faster but a bit less accurate.";
const RP_CLOUD: { value: RunpodCloud; label: string; note: string }[] = [
  { value: "SECURE", label: "Secure Cloud", note: "RunPod's own data centres · steadier, costs more" },
  { value: "COMMUNITY", label: "Community Cloud", note: "machines from vetted hosts · usually cheaper" },
  { value: "ALL", label: "Both", note: "machines from both, side by side" },
];
const SORTS: { value: Sort; label: string; note: string }[] = [
  { value: "cheapest", label: "cheapest", note: "lowest price per hour first" },
  { value: "fastest", label: "fastest", note: "most tokens/s first" },
  { value: "value", label: "best value", note: `most tokens/s per ${money(1).replace(/[\d.,]/g, "")}` },
];

type Field = "model" | "lora" | "where" | "cloud" | "runtime" | "size" | "context" | "sort";
const LABEL: Record<Field, string> = { model: "Model", lora: "LoRA", where: "Where", cloud: "Cloud", runtime: "Runtime", size: "Size", context: "Context", sort: "Sort by" };
/** Width of the option names in each drop-down. */
const OPT_W: Record<Field, number> = { model: 0, lora: 28, where: 24, cloud: 18, runtime: 22, size: 22, context: 20, sort: 13 };
const HEAD: Partial<Record<Field, string>> = {
  where: "Rented GPUs are reached only through a private SSH tunnel from this computer.",
  runtime: RUNTIME_NOTE,
  size: SIZE_NOTE,
  context: "1,000 tokens ≈ 750 words. Max = the longest it can go here; Smart squeeze = the most the picked machine holds.",
};

/** One option in an open drop-down. `act(enter)`: enter = picked with Enter (not flipped through with ←→). */
interface Opt { key: string; label: string; ok: boolean; current: boolean; cycle: boolean; detail: React.ReactNode; act: (enter: boolean) => void }

function qualityWords(q: Quant): string {
  const v = quality(q);
  return v >= 100 ? "full quality" : v >= 90 ? "near full" : v >= 80 ? "very good" : v >= 72 ? "good" : v >= 60 ? "fair" : "lower";
}
const gb = (q: Quant) => { const v = q.size_gb || 0; return `${v < 10 ? v.toFixed(1) : Math.round(v)} GB`; };
function paramsText(r: Recipe): string {
  const f = (b: number) => (b >= 1000 ? `${Math.round(b / 100) / 10}T` : `${b >= 10 ? Math.round(b) : Math.round(b * 10) / 10}B`);
  if (!r.params_b) return "";
  return r.moe && r.active_b ? `${f(r.params_b)} (${f(r.active_b)} active)` : f(r.params_b);
}
/** "32768", "32,768" or "32k" (= 32,768: context lengths are counted in 1,024s). */
const parseCtx = (v: string) => { const t = v.replace(/[,_\s]/g, ""); return /k$/i.test(t) ? Number(t.slice(0, -1)) * 1024 : Number(t); };

export function TunePage(p: {
  recipe: Recipe; hw: Hardware; state: TuneState; setState: (f: (s: TuneState) => TuneState) => void;
  allowed?: RuntimeId[]; adapterGb: number; adapters: Adapter[]; lora?: LoraUse; onLora: (l: LoraUse | undefined) => void;
  banner?: string; onTouched: () => void; best: Partial<Record<"vast" | "runpod", CloudOption>>; skip: ReadonlySet<string>;
  onModel: () => void; onKey: (w: "vast" | "runpod") => void; onExisting: (rt: RuntimeId) => void; onMoreFiles: () => void;
  onContinue: (f: Final) => void; onBack: () => void;
}) {
  const { recipe, hw, state } = p;
  const height = process.stdout.rows || 40;
  const width = process.stdout.columns || 120;
  const s: Setting = { r: recipe, hw, where: state.where, locks: state.locks, allowed: p.allowed, adapterGb: p.adapterGb };
  const cloud = state.where !== "local";
  const rt = runtimeFor(s);
  const noKey = state.where === "runpod" && !getKey("runpod");   // RunPod lists its machines only with a key
  const ctx = searchCtx(s);
  const searchKey = cloud && rt && !noKey ? JSON.stringify([state.where, state.cloud, rt, ctx, recipe.id, recipe.quants.map((q) => q.id), p.adapterGb]) : "";

  // ---------------------------------------------------------------- live machines
  const [found, setFound] = useState<{ key: string; rows?: MachineRow[]; err?: string }>({ key: "" });
  useEffect(() => {
    if (!searchKey || !rt) return;
    let live = true;
    setFound({ key: searchKey });
    findMachines(recipe, rt, ctx, state.where as "vast" | "runpod", state.cloud, p.adapterGb)
      .then((rows) => { if (live) setFound({ key: searchKey, rows }); })
      .catch((e) => { if (live) setFound({ key: searchKey, err: (e as Error).message }); });
    return () => { live = false; };
  }, [searchKey]);
  const machines = found.key === searchKey ? found.rows : undefined;
  const searchErr = found.key === searchKey ? found.err : undefined;
  const loading = !!searchKey && !machines && !searchErr;

  const [open, setOpen] = useState<Field>();
  const [opt, setOpt] = useState(0);
  const [typing, setTyping] = useState(false);

  // The other runtimes' machines too, while the Size list is open, so every size there shows its price.
  const [others, setOthers] = useState<{ key: string; rows: Partial<Record<RuntimeId, MachineRow[]>> }>({ key: "", rows: {} });
  useEffect(() => {
    if (open !== "size" || !searchKey || state.where === "vast") return;   // Vast: searches count toward a daily quota
    let live = true;
    for (const o of usableRuntimes(s).filter((x) => x !== rt && x !== "mlx")) {
      findMachines(recipe, o, ctx, state.where as "vast" | "runpod", state.cloud, p.adapterGb)
        .then((rows) => { if (live) setOthers((x) => ({ key: searchKey, rows: { ...(x.key === searchKey ? x.rows : {}), [o]: rows } })); })
        .catch(() => {});
    }
    return () => { live = false; };
  }, [open, searchKey]);
  const otherRows = others.key === searchKey ? others.rows : {};

  const res: Resolved = useMemo(() => (cloud ? resolveCloud(s, machines, p.skip) : resolveLocal(s)),
    [recipe, hw, state.where, state.locks, p.allowed, p.adapterGb, machines, p.skip]);

  // The machines that hold the model's full context, so Max can be the real maximum (cloud, machine left to ✦).
  const native = recipe.max_context || 32768;
  const fullKey = searchKey && !state.locks.machine && state.where === "runpod" ? `${searchKey}|full` : "";   // Vast: searches count toward a daily quota
  const [atFull, setAtFull] = useState<{ key: string; rows?: MachineRow[] }>({ key: "" });
  useEffect(() => {
    if (!fullKey || !rt) return;
    let live = true;
    findMachines(recipe, rt, native, state.where as "vast" | "runpod", state.cloud, p.adapterGb)
      .then((rows) => { if (live) setAtFull({ key: fullKey, rows }); }).catch(() => { if (live) setAtFull({ key: fullKey, rows: [] }); });
    return () => { live = false; };
  }, [fullKey]);
  const trueMax = useMemo(() => contextMax(s, res, atFull.key === fullKey ? atFull.rows : undefined), [res, atFull, fullKey]);

  // ---------------------------------------------------------------- what re-tuned
  const showBanner = !!p.banner;   // until the first change
  const [status, setStatus] = useState<{ tone: "retune" | "ok" | "warn"; text: string }>();
  const [flash, setFlash] = useState<ReadonlySet<keyof Snap>>(new Set());
  const pending = useRef<{ before: Snap; by: string[]; note?: string; from: TuneState } | undefined>(undefined);
  const flashTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => { if (flashTimer.current) clearTimeout(flashTimer.current); }, []);
  const change = (by: string | string[], f: (st: TuneState) => TuneState, note?: string) => {
    pending.current = { before: snap(state.where, res), by: ([] as string[]).concat(by), note, from: state };
    p.onTouched();
    p.setState(f);
  };
  // Once a change has settled (machines found), say what re-tuned to match it and light that up for a moment.
  useEffect(() => {
    const pd = pending.current;
    if (!pd || loading || state === pd.from) return;
    pending.current = undefined;
    const after = snap(state.where, res);
    const keys = moved(pd.before, after).filter((k) => !pd.by.includes(k));
    const settings = keys.filter((k) => k !== "speed" && k !== "price");
    const lead = pd.note ? `${pd.note}  ` : "";
    setStatus(res.problem ? { tone: "warn", text: res.problem }
      : settings.length ? { tone: "retune", text: `${lead}Re-tuned to match:  ${settings.map((k) => `${SNAP_NAME[k]} ${pd.before[k] ?? "—"} → ${after[k] ?? "—"}`).join("   ")}` }
      : { tone: "ok", text: pd.note || "Done. Nothing else needed to change." });
    setFlash(new Set(keys));
    if (flashTimer.current) clearTimeout(flashTimer.current);
    flashTimer.current = setTimeout(() => setFlash(new Set()), 2500);
  });
  useEffect(() => {
    if (!res.machineGone) return;
    p.setState((st) => ({ ...st, locks: { ...st.locks, machine: undefined } }));
    setStatus({ tone: "warn", text: "The machine you picked is no longer listed, so Smart squeeze picked another." });
  }, [res.machineGone]);

  // ---------------------------------------------------------------- changes
  const setWhere = (w: Where, enter: boolean) => {
    if (w !== state.where) {
      const k = keepLocks({ ...s, where: w, locks: { ...state.locks, machine: undefined } });
      change("where", (st) => ({ ...st, where: w, whereSmart: false, locks: k.locks }),
        k.dropped.length ? `Your ${k.dropped.join(" and ")} can't run on ${WHERE_NAME[w]}, so Smart squeeze picked again.` : undefined);
    }
    if (enter && w !== "local" && !getKey(w)) p.onKey(w);   // spec 10: ask for the key right when it is picked
  };
  const setSize = (id: string | undefined) => change("size", (st) => ({ ...st, locks: { ...st.locks, size: id } }));
  const setLora = (l: LoraUse | undefined) => {
    pending.current = { before: snap(state.where, res), by: ["lora"], from: state };
    p.onTouched();
    p.onLora(l);
  };

  // ---------------------------------------------------------------- machine list
  const sorted: Row[] = useMemo(() => {
    if (!res.rows) return [];
    const per = (o: CloudOption) => o.speed.tpsMax / Math.max(0.01, o.offer.pricePerHour);
    const top = res.rows.find((m) => m.option === res.smart);
    const mine = res.chosen !== res.smart ? res.rows.find((m) => m.option === res.chosen) : undefined;
    const rest = res.rows.filter((m) => m !== top && m !== mine).sort((x, y) => {
      const a = x.option, b = y.option;
      return state.sort === "cheapest" ? a.offer.pricePerHour - b.offer.pricePerHour || quality(b.quant) - quality(a.quant)
        : state.sort === "fastest" ? b.speed.tpsMax - a.speed.tpsMax : per(b) - per(a);
    });
    // One row per machine type (same GPUs, count and size): the cheapest of each, so the list spans more hardware.
    const seen = new Set([top, mine].filter(Boolean).map((m) => `${m!.offer.numGpus}|${m!.offer.gpuName}|${m!.option.quant.id}`));
    const kinds = [...rest].sort((x, y) => x.option.offer.pricePerHour - y.option.offer.pricePerHour).filter((m) => {
      const k = `${m.offer.numGpus}|${m.offer.gpuName}|${m.option.quant.id}`;
      return seen.has(k) ? false : (seen.add(k), true);
    });
    const keep = new Set(kinds);
    return [...(top ? [top] : []), ...(mine ? [mine] : []), ...rest.filter((m) => keep.has(m))].slice(0, 15);
  }, [res, state.sort]);
  const pickMachine = (id: string) => {
    const row = sorted.find((m) => m.offer.id === id);
    if (!row) return;
    const isSmart = row.option === res.smart;
    change("machine", (st) => ({ ...st, locks: { ...st.locks, machine: isSmart ? undefined : row.offer } }),
      isSmart ? "Machine back to Smart squeeze's pick." : `Using ${gpuName(row.offer)} at ${perHour(row.option.offer.pricePerHour)}.`);
  };
  const shiftSize = (id: string, d: number) => {
    const row = sorted.find((m) => m.offer.id === id);
    if (!row) return;
    const next = row.fits[row.fits.findIndex((c) => c.quant.id === row.option.quant.id) + d];   // fits: biggest first, so ← = bigger
    if (!next) return;
    change(["machine", "size"], (st) => ({ ...st, locks: { ...st.locks, machine: row.offer, size: next.quant.id } }),
      `Using ${gpuName(row.offer)} with ${next.quant.label} at ${perHour(next.offer.pricePerHour)}.`);
  };

  // ---------------------------------------------------------------- drop-down options
  const whereOpts = whereOptions(s, (w) => !!getKey(w), p.best);
  const sizeDetail = (c: Choice<string | undefined>) => {
    if (c.value === undefined) {
      return (
        <Box>
          <Box width={32}><Text dimColor>~ this model's sweet spot </Text>{c.preview ? <Text color={C.teal}>{quantName(c.preview)}</Text> : <Text dimColor>…</Text>}</Box>
          {c.speed && <Box width={19}><Tps min={c.speed.tpsMin} max={c.speed.tpsMax} /></Box>}
          {c.price !== undefined && <Text>{perHour(c.price)}</Text>}
        </Box>
      );
    }
    const q = sizeById(recipe, c.value)!;
    return (
      <Box>
        <Box width={8}><Text>{gb(q)}</Text></Box>
        <Box width={12}><Text dimColor>{qualityWords(q)}</Text></Box>
        {c.ok ? (
          <Box>
            <Box width={19}>{c.speed ? <Tps min={c.speed.tpsMin} max={c.speed.tpsMax} /> : <Text dimColor>…</Text>}</Box>
            <Box width={11}><Text>{c.price !== undefined ? perHour(c.price) : ""}</Text></Box>
            {c.note ? <Text dimColor>{c.note}</Text> : null}
          </Box>
        ) : <Text color="gray">{c.why}</Text>}
      </Box>
    );
  };
  const reason = (ok: boolean, note?: string, why?: string) => <Text color={ok ? undefined : "gray"} dimColor>{ok ? note : why}</Text>;
  const optionsFor = (f: Field): Opt[] => {
    switch (f) {
      case "model": return [];
      case "lora": return [
        ...p.adapters.map((a) => ({ key: `a:${adapterName(a)}`, label: adapterName(a), ok: true, current: p.lora?.adapter === a, cycle: true,
          detail: <Text dimColor>{adapterLabel(a)} · {a.sizeGb < 0.1 ? `${Math.round(a.sizeGb * 1000)} MB` : `${a.sizeGb.toFixed(1)} GB`} · {KIND_NAME[a.kind]}</Text>,
          act: () => setLora({ name: adapterName(a), adapter: a }) })),
        { key: "off", label: "Off", ok: true, current: !p.lora, cycle: true, detail: <Text dimColor>serve the model on its own</Text>, act: () => setLora(undefined) },
      ];
      case "where": return whereOpts.map((c) => ({ key: c.value, label: c.label, ok: c.ok, current: c.value === state.where, cycle: true,
        detail: reason(c.ok, c.note, c.why), act: (enter: boolean) => setWhere(c.value, enter) }));
      case "cloud": return RP_CLOUD.map((c) => ({ key: c.value, label: c.label, ok: true, current: c.value === state.cloud, cycle: true,
        detail: <Text dimColor>{c.note}</Text>, act: () => change("cloud", (st) => ({ ...st, cloud: c.value, locks: { ...st.locks, machine: undefined } })) }));
      case "runtime": return runtimeOptions(s).map((c) => ({ key: String(c.value), label: c.label, ok: c.ok, current: c.value === state.locks.runtime, cycle: true,
        detail: reason(c.ok, c.value ? c.note || RUNTIME_BLURB[c.value] : c.note, c.why),
        act: () => change("runtime", (st) => ({ ...st, locks: { ...st.locks, runtime: c.value, machine: undefined } })) }));
      case "size": {
        const list: Opt[] = sizeOptions(s, res, machines, otherRows, p.skip).map((c) => ({ key: String(c.value), label: c.label, ok: c.ok,
          current: c.value === state.locks.size, cycle: true, detail: sizeDetail(c), act: () => setSize(c.value) }));
        if (usableRuntimes(s).includes("llamacpp") && recipe.quants.some((q) => q.format === "gguf"))
          list.push({ key: "files", label: "More GGUF files…", ok: true, current: false, cycle: false, detail: <Text dimColor>every file in the repo, for llama.cpp</Text>, act: () => p.onMoreFiles() });
        return list;
      }
      case "context": return contextOptions(s, res, trueMax).map((c) => ({ key: String(c.value), label: c.label, ok: c.ok, current: c.value === state.locks.context,
        cycle: c.value !== "type", detail: reason(c.ok, c.note, c.why),
        act: () => (c.value === "type" ? setTyping(true) : change("context", (st) => ({ ...st, locks: { ...st.locks, context: c.value as number | undefined } }))) }));
      case "sort": return SORTS.map((c) => ({ key: c.value, label: c.label, ok: true, current: c.value === state.sort, cycle: true,
        detail: <Text dimColor>{c.note}</Text>, act: () => p.setState((st) => ({ ...st, sort: c.value })) }));
    }
  };
  const opts = open ? optionsFor(open) : [];

  // ---------------------------------------------------------------- moving around
  const fields: Field[] = ["model", ...(p.adapters.length ? ["lora" as const] : []), "where", "cloud", "runtime", "size", "context", ...(cloud ? ["sort" as const] : [])];
  const keys = [
    ...fields.filter((f) => f !== "cloud" || state.where === "runpod").map((f) => `f:${f}`),
    ...sorted.map((m) => `m:${m.offer.id}`),
    ...(cloud && !noKey ? ["existing"] : []),
    ...(noKey ? ["addkey"] : []),
    "go",
  ];
  const [focus, setFocus] = useState("go");
  const fi = keys.includes(focus) ? keys.indexOf(focus) : keys.length - 1;
  const at = keys[fi];

  const canGo = !loading && !noKey && !!res.quant && !!res.runtime && !res.problem && (!cloud || !!res.chosen);
  const goOn = () => {
    if (loading) return setStatus({ tone: "warn", text: "Still finding machines. One moment." });
    if (noKey) return p.onKey("runpod");
    if (!canGo) return setStatus({ tone: "warn", text: res.problem || "Nothing can run with these settings yet." });
    p.onContinue({ where: state.where, runtime: res.runtime!, quant: res.quant!, context: res.context, speed: res.speed, fit: res.fit,
      offer: res.chosen?.offer, downloadMin: res.chosen?.downloadMin, downloadCost: res.chosen?.downloadCost });
  };
  const cycle = (f: Field, d: number) => {
    const list = optionsFor(f).filter((o) => o.cycle);
    if (!list.length) return;
    let i = list.findIndex((o) => o.current);
    if (i < 0) i = d > 0 ? -1 : list.length;
    for (let k = 0; k < list.length; k++) { i = (i + d + list.length) % list.length; if (list[i].ok) break; }
    if (list[i]?.ok && !list[i].current) list[i].act(false);
  };
  const activate = (k: string) => {
    if (k === "go") return goOn();
    if (k === "existing") return rt && p.onExisting(rt);
    if (k === "addkey") return p.onKey("runpod");
    if (k.startsWith("m:")) return pickMachine(k.slice(2));
    const f = k.slice(2) as Field;
    if (f === "model") return p.onModel();
    const list = optionsFor(f);
    const cur = list.findIndex((o) => o.current);
    setOpt(cur >= 0 ? cur : Math.max(0, list.findIndex((o) => o.ok)));
    setOpen(f);
  };
  useInput((_input, key) => {
    if (open) {
      if (key.upArrow || key.downArrow) {
        const d = key.upArrow ? -1 : 1;
        let n = opt;
        for (let k = 0; k < opts.length; k++) { n = (n + d + opts.length) % opts.length; if (opts[n].ok) break; }
        setOpt(n);
      } else if (key.return) {
        const o = opts[opt];
        if (o?.ok) { setOpen(undefined); o.act(true); }
      } else if (key.escape || key.leftArrow) setOpen(undefined);
      return;
    }
    if (key.upArrow) setFocus(keys[Math.max(0, fi - 1)]);
    else if (key.downArrow) setFocus(keys[Math.min(keys.length - 1, fi + 1)]);
    else if ((key.leftArrow || key.rightArrow) && at.startsWith("f:")) cycle(at.slice(2) as Field, key.leftArrow ? -1 : 1);
    else if ((key.leftArrow || key.rightArrow) && at.startsWith("m:")) shiftSize(at.slice(2), key.leftArrow ? -1 : 1);
    else if (key.return) activate(at);
    else if (key.escape) p.onBack();
  }, { isActive: !typing });

  // A typed context: the same rules as the list, and the model's own limit.
  const checkTyped = (v: string): string | undefined => {
    const n = parseCtx(v);
    if (!Number.isFinite(n) || n < 512) return "Type a number of tokens, at least 512.";
    const top = cloud ? recipe.max_context_extended || recipe.max_context || n : recipe.max_context || n;
    if (n > top) return `${cloud ? "This model" : "On this computer it"} goes up to ${fmt(top)} tokens.`;
    const c = contextChoice(s, res, n, trueMax);
    return c.ok ? undefined : `That won't work: ${c.why}.`;
  };

  // ---------------------------------------------------------------- layout
  const contextWarn = !cloud && res.fit?.setup === "mixed"
    ? `Part of the model sits in RAM at this length, so it will be slower.${res.gpuMax ? ` Up to ${fmt(res.gpuMax)} tokens it all fits on the GPU.` : ""}`
    : res.clamped ? `You set ${fmt(state.locks.context!)} tokens, but only ${fmt(res.context)} fit ${cloud ? "on this machine" : "on this computer"}.` : undefined;
  // Keep the whole page inside the terminal: Ink can't redraw what has scrolled off the top.
  // Short windows (80×24) get a compact page: one-line banner and help, no blank gaps. Every row is one
  // line (long text is cut with "…"), so the height below is exact and Ink can redraw in place.
  const compact = height < 34;
  const gap = compact ? 0 : 1;
  const topLines = showBanner && p.banner && !compact ? 5 : 1;
  const base = topLines + gap + fields.length + (typing ? 3 : 0) + (contextWarn ? 1 : 0) + (cloud ? 0 : hw.notes.length)
    + (3 + gap) + (1 + gap) + ((compact ? 1 : 2) + gap) + 1;
  const vis = cloud && !open ? Math.max(2, Math.min(15, height - base - (6 + gap))) : 0;
  const dvis = open ? Math.max(3, Math.min(opts.length, height - base - (cloud ? 1 + gap : 3 + gap) - 6)) : 0;
  const dStart = Math.max(0, Math.min(opt - Math.floor(dvis / 2), opts.length - dvis));
  const win = useRef(0);
  const mk = at.startsWith("m:") ? sorted.findIndex((m) => `m:${m.offer.id}` === at) : -1;
  let start = Math.min(win.current, Math.max(0, sorted.length - vis));
  if (mk >= 0 && mk < start) start = mk;
  if (mk >= 0 && mk >= start + vis) start = mk - vis + 1;
  win.current = start;
  // Narrow windows get tighter columns; Prefill stays (spec 11.6), the one-off download column goes first.
  const narrow = width < 112;
  const W = narrow ? { gpu: 17, size: 9, price: 11, speed: 17, prefill: 16 } : { gpu: 22, size: 11, price: 12, speed: 20, prefill: 18 };
  const cols = { oneOff: width >= 116 };

  const value = (f: Field): { text: React.ReactNode; smart?: boolean; hint?: string; lit?: keyof Snap; grey?: boolean } => {
    switch (f) {
      case "model": return { text: recipe.name.length > 34 ? `${recipe.name.slice(0, 33)}…` : recipe.name, hint: [paramsText(recipe), recipe.gated ? "gated" : ""].filter(Boolean).join(" · ") };
      case "lora": return { text: p.lora ? "On" : "Off", hint: p.lora?.adapter ? `${adapterLabel(p.lora.adapter)} · ${KIND_NAME[p.lora.adapter.kind]}` : `${p.adapters.length} LoRA adapter${p.adapters.length > 1 ? "s" : ""} in this repo` };
      case "where": { const w = whereOpts.find((o) => o.value === state.where); return { text: state.where === "local" ? "Local (this computer)" : WHERE_NAME[state.where], smart: state.whereSmart, hint: w?.ok ? w.note : w?.why, lit: "where" }; }
      case "cloud": { const c = RP_CLOUD.find((x) => x.value === state.cloud)!; return state.where === "runpod" ? { text: c.label, hint: c.note } : { text: c.label, hint: "RunPod only", grey: true }; }
      case "runtime": return rt
        ? { text: state.locks.runtime ? RUNTIME_NAMES[rt] : `Choose best (${RUNTIME_NAMES[rt]})`, smart: !state.locks.runtime, hint: RUNTIME_NOTE, lit: "runtime" }
        : { text: "none can run here", hint: RUNTIME_NOTE, grey: true };
      case "size": {
        const q = res.quant;
        if (!state.locks.size) return { text: <>Smart squeeze · {q ? <Text color={C.teal}>{quantName(q)}</Text> : <Text dimColor>…</Text>}</>, smart: true, hint: `~ this model's sweet spot${q ? ` · ${gb(q)}` : ""}`, lit: "size" };
        const own = sizeById(recipe, state.locks.size);
        return { text: own ? `${own.label}${own.quant && own.quant !== own.label ? ` (${own.quant})` : ""}` : "?", hint: own ? `${gb(own)} · ${qualityWords(own)}` : undefined, lit: "size" };
      }
      case "context": return { text: state.locks.context === undefined ? (res.ctxMax ? `Smart squeeze · ${fmt(res.context)}` : "Smart squeeze")
          : state.locks.context === trueMax ? `Max · ${fmt(state.locks.context)}` : fmt(state.locks.context),
        smart: state.locks.context === undefined, hint: "text it can hold at once · 1,000 tokens ≈ 750 words", lit: "context" };
      case "sort": { const c = SORTS.find((x) => x.value === state.sort)!; return { text: c.label, hint: c.note }; }
    }
  };

  const dropdown = open && (
    <Box marginLeft={14} flexDirection="column" borderStyle="round" borderColor={C.editable} paddingX={1}>
      {HEAD[open] && <Text dimColor wrap="truncate-end">{HEAD[open]}</Text>}
      {dStart > 0 && <Text dimColor>  ↑ {dStart} more</Text>}
      {opts.slice(dStart, dStart + dvis).map((o, k) => {
        const i = dStart + k;
        return (
          <Box key={o.key} height={1} overflow="hidden">
            <Text color={C.accent}>{i === opt ? "› " : "  "}</Text>
            <Box width={OPT_W[open]}><Text color={o.ok ? undefined : "gray"} bold={i === opt && o.ok}>{o.label}</Text></Box>
            <Box width={2}><Text color={C.alive}>{o.current ? "✓" : ""}</Text></Box>
            {o.detail}
          </Box>
        );
      })}
      {dStart + dvis < opts.length && <Text dimColor>  ↓ {opts.length - dStart - dvis} more</Text>}
      <Text dimColor wrap="truncate-end">↑↓ choose · Enter pick · Esc close · grey = not possible here, and why</Text>
    </Box>
  );

  const fieldRow = (f: Field) => {
    const v = value(f);
    const focused = at === `f:${f}`;
    const lit = !!v.lit && flash.has(v.lit);
    return (
      <Box key={f} flexDirection="column">
        <Box height={1} overflow="hidden">
          <Box width={3} flexShrink={0}><Text color={C.accent}>{focused ? " › " : "   "}</Text></Box>
          <Box width={11} flexShrink={0}><Text color={v.grey ? "gray" : undefined}>{LABEL[f]}</Text></Box>
          <Box width={narrow ? 38 : 46} flexShrink={0}>
            {v.grey ? <Text color="gray">{v.text}</Text> : (
              <Text>
                <Text color={C.editable}>[ </Text>
                <Text bold={focused} color={lit ? C.teal : undefined}>{v.text}</Text>
                <Text color={C.editable}> ▾ ]</Text>
                {v.smart ? <Text color={C.teal}> ✦</Text> : null}
              </Text>
            )}
          </Box>
          {v.hint ? <Text color={v.grey ? "gray" : undefined} dimColor wrap="truncate-end">{v.hint}</Text> : null}
        </Box>
        {open === f && dropdown}
        {f === "context" && typing && (
          <Box marginLeft={14} flexDirection="column">
            <Text dimColor>Type a number of tokens (32768, or 32k). Enter sets it, Esc cancels. Most that fits {cloud ? "on this machine" : "on this computer"}: {fmt(res.ctxMax)}.</Text>
            <TextInput validate={checkTyped} onBack={() => setTyping(false)}
              onSubmit={(v) => { setTyping(false); change("context", (st) => ({ ...st, locks: { ...st.locks, context: parseCtx(v) } })); }} />
          </Box>
        )}
        {f === "context" && contextWarn && <Text color={C.warn} wrap="truncate-end">{"              "}{contextWarn}</Text>}
      </Box>
    );
  };

  const machineRow = (m: Row) => {
    const o = m.option;
    const focused = at === `m:${m.offer.id}`;
    const inUse = o === res.chosen;
    const lit = inUse && (flash.has("machine") || flash.has("size"));
    return (
      <Box key={m.offer.id} height={1} overflow="hidden">
        <Box width={3} flexShrink={0}><Text color={C.accent}>{focused ? " › " : "   "}</Text></Box>
        <Box width={3} flexShrink={0}><Text color={C.alive}>{inUse ? "●" : ""}</Text></Box>
        <Box width={W.gpu} flexShrink={0}><Text bold={focused} color={lit ? C.teal : undefined}>{gpuName(o.offer).slice(0, W.gpu - 1)}</Text></Box>
        <Box width={W.size} flexShrink={0}>{focused && m.fits.length > 1
          ? <Text><Text color={C.editable}>{m.fits[0] === o ? "  " : "‹ "}</Text><Text bold color={lit ? C.teal : undefined}>{o.quant.label}</Text><Text color={C.editable}>{m.fits[m.fits.length - 1] === o ? "" : " ›"}</Text></Text>
          : <Text color={lit ? C.teal : undefined}>{"  "}{o.quant.label}</Text>}</Box>
        <Box width={W.price} flexShrink={0}><Text>{perHour(o.offer.pricePerHour)}</Text></Box>
        <Box width={W.speed} flexShrink={0}><Tps min={o.speed.tpsMin} max={o.speed.tpsMax} /></Box>
        <Box width={W.prefill} flexShrink={0}><Prefill value={o.speed.prefill} label={false} /></Box>
        {cols.oneOff && <Box width={10} flexShrink={0}><Text dimColor>{o.downloadCost >= 0.005 ? `+ ${money(o.downloadCost)}` : "—"}</Text></Box>}
        {o === res.smart && <Text color={C.teal}>{narrow ? "✦" : "✦ smart squeeze"}</Text>}
        {!narrow && state.where === "runpod" && state.cloud === "ALL" && <Text dimColor>  {o.offer.cloud === "COMMUNITY" ? "community" : "secure"}</Text>}
        {!narrow && o.offer.stock === "LOW" && <Text color={C.warn}>  few left</Text>}
      </Box>
    );
  };

  const rowFor = (k: string, label: string, color?: string) => (
    <Box><Text color={C.accent}>{at === k ? " › " : "   "}</Text><Text>{"   "}</Text><Text bold={at === k} color={color}>{label}</Text></Box>
  );
  const machinesBlock = (
    <Box marginTop={gap} flexDirection="column">
      <Box height={1} overflow="hidden">
        <Text>{"   "}</Text>
        <Text bold>Machines on {WHERE_NAME[state.where]}{state.where === "runpod" ? ` (${RP_CLOUD.find((c) => c.value === state.cloud)!.label})` : ""}</Text>
        {loading && <Text>{"   "}<Spinner /><Text dimColor> finding the ones that can run it…</Text></Text>}
        {!loading && res.rows && <Text dimColor wrap="truncate-end">{`   ${fmt(res.rows.length)} can run ${state.locks.size && res.quant ? res.quant.label : "it"}${open ? " · close the list to see them" : " · Enter = use this one · ←→ = its size"}`}</Text>}
      </Box>
      {open ? null : noKey ? <Text color={C.warn} wrap="truncate-end">{"   "}RunPod lists its machines only with your API key.</Text>
        : searchErr ? <Text color={C.error} wrap="truncate-end">{"   "}Could not get the machine list: {searchErr}</Text>
        : !loading && res.problem ? <Text color={C.warn} wrap="truncate-end">{"   "}{res.problem}</Text>
        : !loading && (
          <Box flexDirection="column">
            <Box>
              <Text>{"      "}</Text><Box width={W.gpu} flexShrink={0}><Text dimColor>GPU</Text></Box><Box width={W.size} flexShrink={0}><Text dimColor>{"  "}Size</Text></Box><Box width={W.price} flexShrink={0}><Text dimColor>Price</Text></Box><Box width={W.speed} flexShrink={0}><Text dimColor>Speed</Text></Box>
              <Box width={W.prefill} flexShrink={0}><Text dimColor>Prefill</Text></Box>{cols.oneOff && <Box width={10}><Text dimColor>One-off</Text></Box>}
            </Box>
            {start > 0 && <Text dimColor>{"      "}↑ {start} more</Text>}
            {sorted.slice(start, start + vis).map(machineRow)}
            {sorted.length > start + vis && <Text dimColor>{"      "}↓ {sorted.length - start - vis} more</Text>}
          </Box>
        )}
      {!open && cloud && !noKey && rowFor("existing", "Use a machine I already have…")}
      {!open && noKey && rowFor("addkey", "Add your RunPod API key…", C.editable)}
    </Box>
  );
  const localBlock = (
    <Box marginTop={gap} flexDirection="column">
      <Text wrap="truncate-end">{"   "}<Text bold>This computer</Text><Text dimColor>{"   "}{hardwareSummary(hw)}</Text></Text>
      {res.problem ? <Text color={C.warn} wrap="truncate-end">{"   "}{res.problem}</Text>
        : res.fit ? <Text dimColor wrap="truncate-end">{"   "}Runs {setupLabel(res.fit.setup)}{res.fit.setup === "mixed" ? ` (${res.fit.gpuGb} GB on the GPU, ${res.fit.ramGb} GB in RAM)` : ""} · needs about {res.fit.totalGb} GB</Text> : null}
      {hw.notes.map((n, i) => <Text key={i} color={C.warn} wrap="truncate-end">{"   "}{n}</Text>)}
    </Box>
  );

  const ping = (k: keyof Snap) => (flash.has(k) ? <Text color={C.teal}>  ↻</Text> : null);
  const dash = <Text dimColor>{loading ? "…" : "—"}</Text>;
  const summary = (
    <Box marginTop={gap} flexDirection="column">
      <Box height={1} overflow="hidden"><Text>{"   "}</Text><Box width={11}><Text dimColor>Speed</Text></Box>
        {res.speed ? <><Tps min={res.speed.tpsMin} max={res.speed.tpsMax} /><Text>{"   "}</Text><Prefill value={res.speed.prefill} /><Text dimColor>{"   "}estimate</Text>{ping("speed")}</> : dash}
      </Box>
      <Box height={1} overflow="hidden"><Text>{"   "}</Text><Box width={11}><Text dimColor>Cost</Text></Box>
        {!cloud ? <Text>free<Text dimColor>{"   "}runs on this computer</Text></Text>
          : res.chosen ? <Text>{perHour(res.chosen.offer.pricePerHour)}<Text dimColor>{res.chosen.downloadCost >= 0.005 ? `   + ${money(res.chosen.downloadCost)} one-off download` : state.where === "runpod" ? "   no download charge" : ""}</Text>{ping("price")}</Text> : dash}
      </Box>
      <Box height={1} overflow="hidden"><Text>{"   "}</Text><Box width={11}><Text dimColor>Download</Text></Box>
        {res.quant ? <Text>{gb(res.quant)}<Text dimColor>{cloud && res.chosen ? `   about ${res.chosen.downloadMin} min on this machine` : !cloud ? "   the time is worked out on the next screen" : ""}</Text></Text> : dash}
      </Box>
    </Box>
  );

  return (
    <Box flexDirection="column" paddingX={1}>
      {showBanner && p.banner && compact ? (
        <Text color={C.teal} wrap="truncate-end">✦ Smartly tuned for {recipe.name}: {p.banner}. Enter to go, or change anything in [ ▾ ].</Text>
      ) : showBanner && p.banner ? (
        <Box borderStyle="round" borderColor={C.teal} paddingX={1} flexDirection="column">
          <Text color={C.teal} bold>✦ Smartly tuned for {recipe.name} on {state.where === "local" ? "this computer" : WHERE_NAME[state.where]}</Text>
          <Text>{p.banner}.</Text>
          <Text dimColor>Press Enter to go with it, or change anything in <Text color={C.editable}>[ ▾ ]</Text>: the <Text color={C.teal}>✦</Text> settings re-tune to match.</Text>
        </Box>
      ) : status ? (
        <Text color={status.tone === "warn" ? C.warn : C.teal} wrap="truncate-end">{status.tone === "warn" ? "! " : status.tone === "retune" ? "↻ " : "✓ "}{status.text}</Text>
      ) : (
        <Text bold>Set it up<Text dimColor>{"   "}{recipe.name}</Text></Text>
      )}
      <Box marginTop={gap} flexDirection="column">{fields.map(fieldRow)}</Box>
      {cloud ? machinesBlock : localBlock}
      {summary}
      <Box marginTop={gap} height={1} overflow="hidden">
        <Text color={C.accent}>{at === "go" ? " › " : "   "}</Text>
        <Text bold={at === "go"} color={canGo ? undefined : "gray"}>Continue →</Text>
        <Text dimColor>{"   "}{canGo ? "check everything, then start" : loading ? "finding machines…" : noKey ? "add your RunPod key first" : "nothing can run with these settings yet"}</Text>
      </Box>
      <Box marginTop={gap} flexDirection="column">
        <Text dimColor wrap="truncate-end">↑↓ move   ←→ change   Enter open <Text color={C.editable}>▾</Text> or pick   Esc back{"        "}<Text color={C.teal}>✦</Text> = Smart squeeze's pick: it re-tunes when you change something</Text>
        {!compact && <Text dimColor wrap="truncate-end">{SIZE_NOTE}</Text>}
      </Box>
    </Box>
  );
}
