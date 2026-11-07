import React, { useEffect, useMemo, useState } from "react";
import { Box, Text, useInput, usePaste } from "ink";
import { Screen, Select, TextInput, Busy, Tps, Prefill, type Item } from "../ui/components.js";
import { C, fmtInt } from "../ui/theme.js";
import { loadIndex, checkNewModels, resolveRecipe, parseHfId, ggufQuants, hfTree, QRE, type Recipe, type Quant, type IndexEntry } from "../core/recipes.js";
import { getKey, type KeyName } from "../core/keys.js";
import type { RuntimeId, Where } from "../core/store.js";
import { setupLabel, type CloudOption } from "../core/plan.js";
import { findMachines, smartWhere, runtimeFor, keepLocks, baseCtx, quantName, type Setting, type MachineRow } from "../core/tune.js";
import { TunePage, type TuneState, type Final } from "./tune.js";
import { RUNTIME_NAMES } from "../runtimes/index.js";
import { startWslInstall, wslReady } from "../runtimes/install.js";
import { wslInfo } from "../core/hardware.js";
import { provider, type Instance, type Offer, type RunpodCloud } from "../providers/index.js";
import { perHour, money } from "../core/cost.js";
import { KeyPrompt, VastTfa } from "./keys.js";
import { checkKey } from "../core/keys.js";
import { measureDownloadMBps } from "../core/hfdownload.js";
import { StartingScreen } from "./starting.js";
import { startJob, activeJobs, planLora, checkPort, keptApiKey, fixedPort, defaultModelName, type Plan } from "../core/launch.js";
import { findAdapters, isAdapterOnly, baseModelOf, adapterName, adapterLabel, adapterFitsQuant, ADAPTER_RUNTIMES, KIND_NAME, type Adapter, type LoraUse } from "../core/lora.js";
import { loadHistory } from "../core/store.js";
import type { Ctx } from "../app.js";
import { IS_WIN } from "../core/paths.js";

const RUNTIME_NOTE = "Runtime = the software the model runs on.";
const SIZE_NOTE = "Size = how compressed the model is. Smaller is cheaper and faster but a bit less accurate.";
const WHERE_NAME: Record<Where, string> = { local: "Local", vast: "Vast.ai", runpod: "RunPod" };
const RP_CLOUD_NAME: Record<RunpodCloud, string> = { SECURE: "Secure Cloud", COMMUNITY: "Community Cloud", ALL: "Secure + Community" };

type Step = "model" | "hfkey" | "tune" | "provkey" | "vasttfa" | "wsl" | "sizeFiles" | "existing" | "ctxExisting" | "ctxType" | "confirm" | "advanced" | "starting";

/** A promise that gives up after `ms`. */
const within = <T,>(p: Promise<T>, ms: number) => Promise.race([p, new Promise<T>((_, no) => setTimeout(() => no(new Error("timed out")), ms))]);

function paramsText(e: { params_b: number | null; active_b: number | null; moe: boolean }): string {
  const f = (b: number) => (b >= 1000 ? `${Math.round(b / 100) / 10}T` : `${b >= 10 ? Math.round(b) : Math.round(b * 10) / 10}B`);
  if (!e.params_b) return "";
  return e.moe && e.active_b ? `${f(e.params_b)} (${f(e.active_b)} active)` : f(e.params_b);
}

export function ServeFlow({ ctx }: { ctx: Ctx }) {
  const hw = ctx.hw!;
  const [step, setStep] = useState<Step>("model");
  const [busy, setBusy] = useState<string>();
  const [error, setError] = useState<string>();
  const [noVm, setNoVm] = useState(false);                   // WSL cannot start on this computer (no virtualization)
  const [recipe, setRecipe] = useState<Recipe>();
  const [adapters, setAdapters] = useState<Adapter[]>([]);   // LoRA adapters found in the model's repo
  const [lora, setLora] = useState<LoraUse>();               // the one in use (on by default)
  const [custom, setCustom] = useState<Quant>();             // a GGUF file picked from "More GGUF files…"
  // The set-up page: what you set there, and what Smart squeeze worked out when the model was picked.
  const [tune, setTune] = useState<TuneState>({ where: "local", whereSmart: true, cloud: "SECURE", locks: {}, sort: "cheapest" });
  const [banner, setBanner] = useState<string>();
  const [best, setBest] = useState<Partial<Record<"vast" | "runpod", CloudOption>>>({});
  const [skip, setSkip] = useState<ReadonlySet<string>>(new Set());   // machines taken before they could be rented
  const [keyFor, setKeyFor] = useState<"vast" | "runpod">("vast");
  // What the set-up page handed on (Confirm shows it and builds the plan from it).
  const [where, setWhere] = useState<Where>("local");
  const [runtime, setRuntime] = useState<RuntimeId>();
  const [context, setContext] = useState<number>(0);
  const [choice, setChoice] = useState<{ quant: Quant; offer?: CloudOption["offer"]; speed?: any; fit?: any; downloadMin?: number; downloadCost?: number; existing?: Instance }>();
  const [advanced, setAdvanced] = useState<Plan["advanced"]>({});
  const [endpoint, setEndpoint] = useState<NonNullable<Plan["endpoint"]>>({});   // your own port / API key
  const [history, setHistory] = useState<Step[]>([]);
  const [jobId, setJobId] = useState<string>();

  const loraUse = advanced?.lora || lora;
  const adapterGb = loraUse?.adapter?.sizeGb || 0;
  // The model as the set-up page sees it: a GGUF file picked from the list joins its sizes, and with a
  // LoRA adapter on, only the sizes that adapter works with stay (and only the runtimes that can load it).
  const withSizes = (r: Recipe, l?: LoraUse): Recipe => ({ ...r, quants: [...(custom && !r.quants.some((q) => q.id === custom.id) ? [custom] : []), ...r.quants].filter((q) => !l?.adapter || adapterFitsQuant(l.adapter, q)) });
  const base = useMemo<Recipe | undefined>(() => recipe && withSizes(recipe, loraUse), [recipe, custom, loraUse]);
  const allowed = loraUse?.adapter ? ADAPTER_RUNTIMES[loraUse.adapter.kind] : undefined;

  const go = (s: Step) => { setHistory((h) => [...h, step]); setError(undefined); setStep(s); };
  /** Leave a one-off step (token, key, login code) without keeping it in the history, so Back skips it. */
  const swap = (s: Step) => { setError(undefined); setStep(s); };
  const [keyNext, setKeyNext] = useState<Step>();   // where to go once a key is saved (else back)
  const [keyReason, setKeyReason] = useState<string>();
  const back = () => {
    setError(undefined);
    const h = [...history];
    const prev = h.pop();
    setHistory(h);
    if (prev) setStep(prev); else ctx.go({ name: "start" });
  };

  const summary: [string, string][] = [];
  if (recipe) summary.push(["Model", recipe.name]);
  if (!["model", "hfkey", "tune", "provkey", "sizeFiles"].includes(step)) {
    summary.push(["Where", where === "runpod" ? `RunPod ${RP_CLOUD_NAME[tune.cloud]}` : WHERE_NAME[where]]);
    if (choice?.quant) summary.push(["Size", quantName(choice.quant)]);
    if (loraUse) summary.push(["LoRA", loraUse.name]);
    if (runtime) summary.push(["Runtime", RUNTIME_NAMES[runtime]]);
    if (context && ["confirm", "advanced"].includes(step)) summary.push(["Context", fmtInt(context)]);
  }

  if (busy) return <Busy text={busy} />;

  // Picking a model tunes everything for it: Smart squeeze compares this computer with the best value
  // machines on Vast.ai (and RunPod, with a key saved) and starts the set-up page on the best of them.
  const tuneFor = async (r: Recipe, found: Adapter[]) => {
    const l = found[0] ? { name: adapterName(found[0]), adapter: found[0] } : undefined;
    const b: Recipe = { ...r, quants: r.quants.filter((q) => !l?.adapter || adapterFitsQuant(l.adapter, q)) };
    const st: Setting = { r: b, hw, where: "local", locks: {}, allowed: l?.adapter ? ADAPTER_RUNTIMES[l.adapter.kind] : undefined, adapterGb: l?.adapter?.sizeGb || 0 };
    const lists: Partial<Record<"vast" | "runpod", MachineRow[]>> = {};
    await Promise.all((["vast", "runpod"] as const).filter((w) => w === "vast" || getKey("runpod")).map(async (w) => {
      const rt = runtimeFor({ ...st, where: w });
      if (rt) lists[w] = await within(findMachines(b, rt, baseCtx(b), w, "SECURE", st.adapterGb), 20_000).catch(() => undefined);
    }));
    const pick = smartWhere(st, lists);
    setRecipe(r); setAdapters(found); setLora(l); setCustom(undefined); setSkip(new Set()); setEndpoint({});
    setAdvanced({}); setChoice(undefined); setRuntime(undefined); setContext(0);   // nothing carries over from the last model
    setTune({ where: pick.where, whereSmart: true, cloud: "SECURE", locks: {}, sort: "cheapest" });
    setBanner(pick.why); setBest(pick.best);
  };

  // ------------------------------------------------------------ Step 1: model
  if (step === "model") {
    return <ModelPicker onBack={back} onPick={async (id) => {
      setBusy("Looking up the model…");
      try {
        // A LoRA adapter repo: serve the model it was made for, with the adapter on.
        const found = await within(findAdapters(id), 30_000).catch(() => [] as Adapter[]);
        let modelId = id;
        if (await within(isAdapterOnly(id, found), 30_000).catch(() => false)) {
          const baseId = await within(baseModelOf(found[0]), 30_000).catch(() => undefined);
          if (!baseId) { setBusy(undefined); setError(`${id} is a LoRA adapter, but it does not say which model it was made for. Pick that model, then add the adapter under Advanced settings.`); return; }
          modelId = baseId;
        }
        const r = await within(resolveRecipe(modelId), 30_000);
        setBusy(`✦ Tuning ${r.name}: comparing this computer with ${getKey("runpod") ? "Vast.ai and RunPod" : "Vast.ai"}…`);
        await tuneFor(r, found);
        setBusy(undefined);
        if (r.gated && !getKey("hf")) go("hfkey"); else go("tune");
      } catch (e) {
        setBusy(undefined);
        setError(`Could not find "${id}" on Hugging Face. Check the name or link. (${(e as Error).message})`);
      }
    }} error={error} />;
  }
  if (step === "hfkey") {
    return <KeyPrompt name="hf" reason={`${recipe!.name} is gated: its maker asks you to accept their license first. A Hugging Face token lets spunup download it.`}
      onBack={back} onDone={() => swap("tune")} />;
  }

  // ------------------------------------------------------------ Step 2: the set-up page (everything else)
  const onContinue = (f: Final) => {
    setWhere(f.where); setRuntime(f.runtime); setContext(f.context);
    setChoice({ quant: f.quant, offer: f.offer, speed: f.speed, fit: f.fit, downloadMin: f.downloadMin, downloadCost: f.downloadCost });
    if (f.where === "local") { go(IS_WIN && (f.runtime === "vllm" || f.runtime === "sglang") && !wslReady(hw) ? "wsl" : "confirm"); return; }
    if (!getKey(f.where)) { setKeyFor(f.where); setKeyNext("confirm"); setKeyReason(undefined); go("provkey"); return; }
    if (f.where === "vast") {
      setBusy("Checking your Vast.ai key…");
      checkKey("vast", getKey("vast")!).then((r) => {
        setBusy(undefined);
        if (r.needsTfa) go("vasttfa");
        else if (!r.ok && !/^Could not check/.test(r.message)) { setKeyFor("vast"); setKeyNext("confirm"); setKeyReason(r.message); go("provkey"); }
        else go("confirm");   // works, or couldn't be checked right now: renting will say if it really doesn't work
      }).catch(() => { setBusy(undefined); go("confirm"); });
      return;
    }
    go("confirm");
  };
  if (step === "tune") {
    return <TunePage key={recipe!.id} recipe={base!} hw={hw} state={tune} setState={setTune} allowed={allowed} adapterGb={adapterGb}
      adapters={adapters} lora={lora}
      onLora={(l) => {
        setLora(l);
        setTune((t) => ({ ...t, locks: keepLocks({ r: withSizes(recipe!, advanced?.lora || l), hw, where: t.where, locks: { ...t.locks, machine: undefined },
          allowed: l?.adapter ? ADAPTER_RUNTIMES[l.adapter.kind] : undefined, adapterGb: l?.adapter?.sizeGb || 0 }).locks }));
      }}
      banner={banner} onTouched={() => setBanner(undefined)} best={best} skip={skip}
      onModel={() => go("model")}
      onKey={(w) => { setKeyFor(w); setKeyNext(undefined); setKeyReason(undefined); go("provkey"); }}
      onExisting={(rt) => {
        const w = tune.where as "vast" | "runpod";
        setWhere(w); setRuntime(rt);
        if (getKey(w)) go("existing"); else { setKeyFor(w); setKeyNext("existing"); setKeyReason(undefined); go("provkey"); }
      }}
      onMoreFiles={() => go("sizeFiles")}
      onContinue={onContinue} onBack={back} />;
  }
  if (step === "provkey") {
    return <KeyPrompt name={keyFor} reason={keyReason} onBack={back} onDone={(ok) => (ok && keyNext ? swap(keyNext) : back())} />;
  }
  if (step === "vasttfa") {
    return <VastTfa apiKey={getKey("vast")!} onBack={back} onDone={() => swap("confirm")} />;
  }
  if (step === "sizeFiles") {
    return <GgufFiles recipe={recipe!} onBack={back} onPick={(q) => {
      if (!recipe!.quants.some((x) => x.id === q.id)) setCustom(q);
      setLora((l) => (l?.adapter && !adapterFitsQuant(l.adapter, q) ? undefined : l));
      setTune((t) => ({ ...t, locks: { ...t.locks, size: q.id, runtime: t.locks.runtime === "llamacpp" ? "llamacpp" : undefined, machine: undefined } }));
      back();
    }} />;
  }
  if (step === "wsl") {
    const st = hw.wsl?.state || "off";
    const after = () => go("confirm");
    return (
      <Screen title="This runtime needs WSL" summary={summary}>
        <Text>{RUNTIME_NAMES[runtime!]} does not run directly on Windows. It needs WSL (a small Linux inside Windows) with your NVIDIA driver.</Text>
        {st === "off" && <Text color={C.warn}>WSL is not switched on yet. Switching it on needs admin rights (Windows will ask) and a restart, because Windows only loads it while starting up.</Text>}
        {st === "nodistro" && !noVm && <Text>WSL is on. spunup will install Ubuntu into it (about 1 GB, a few minutes). No admin rights or restart needed.</Text>}
        {st === "nogpu" && <Text color={C.warn}>Linux in WSL cannot see your NVIDIA GPU. Updating the NVIDIA driver on Windows (nvidia.com) usually fixes this.</Text>}
        {error && <Text color={C.error}>{error}</Text>}
        <Box marginTop={1}><Select items={[
          ...(noVm ? [] : st === "nogpu" ? [{ value: "recheck", label: "Check again" }] : [{ value: "install", label: st === "nodistro" ? "Install Ubuntu in WSL now" : "Switch on WSL now" }]),
          { value: "llama", label: "Use llama.cpp instead (works right away)" },
          { value: "back", label: "Back" },
        ]} onBack={back} onSelect={(v) => {
          if (v === "install" && st === "off") { startWslInstall("off").then((r) => setError(r.message)); }
          else if (v === "install") {
            setBusy("Installing Ubuntu in WSL…");
            startWslInstall(st, (l) => setBusy(`Installing Ubuntu in WSL…  ${l.slice(0, 70)}`)).then((r) => {
              hw.wsl = wslInfo();
              setBusy(undefined);
              if (r.ok && wslReady(hw)) after(); else { setError(r.message); if (r.noVm) setNoVm(true); }
            });
          }
          else if (v === "recheck") { hw.wsl = wslInfo(); if (wslReady(hw)) after(); else setError("Still no GPU inside WSL."); }
          else if (v === "llama") { setTune((t) => ({ ...t, locks: { ...t.locks, runtime: "llamacpp", size: undefined, machine: undefined } })); back(); }
          else back();
        }} /></Box>
      </Screen>
    );
  }

  // ------------------------------------------------------------ A machine you already have (its memory is unknown)
  if (step === "existing") {
    return <ExistingPicker recipe={base!} runtime={runtime!} where={where as "vast" | "runpod"} summary={summary} onBack={back}
      onPick={(inst, quant) => { setChoice({ quant, existing: inst }); go("ctxExisting"); }} />;
  }
  if (step === "ctxExisting" || step === "ctxType") {
    const r = base!;
    const maxCtx = r.max_context || 32768;
    const top = r.max_context_extended || maxCtx;
    const done = (c: number) => { setContext(c); go("confirm"); };
    const note = <Text dimColor>spunup can't see how much memory this machine has free, so a very long context may not fit.</Text>;
    if (step === "ctxType") {
      return (
        <Screen title="Context length (how much text the model can hold at once)" summary={summary} footer="Type a number, Enter · Esc back">
          <Text dimColor>This model supports up to {fmtInt(top)} tokens. 1,000 tokens ≈ 750 words.</Text>
          <TextInput onBack={back} placeholder="e.g. 32768"
            validate={(v) => {
              const n = Number(v.replace(/[,_\s]/g, "").replace(/k$/i, "000"));
              if (!Number.isFinite(n) || n < 512) return "Type a number of tokens, at least 512.";
              if (n > top) return `This model supports at most ${fmtInt(top)} tokens.`;
              return undefined;
            }}
            onSubmit={(v) => done(Number(v.replace(/[,_\s]/g, "").replace(/k$/i, "000")))} />
          {note}
        </Screen>
      );
    }
    return (
      <Screen title="Context length (how much text the model can hold at once)" summary={summary}>
        <Select items={[{ value: "max", label: `Max (${fmtInt(maxCtx)} tokens)` }, { value: "type", label: "Type a number…" }]} onBack={back}
          onSelect={(v) => (v === "max" ? done(maxCtx) : go("ctxType"))} />
        {note}
      </Screen>
    );
  }

  // ------------------------------------------------------------ Confirm
  const plan: Plan = {
    recipe: base!, quant: choice?.quant!, where, runtime: runtime!, context, hw, lora,
    setup: where === "local" ? (choice?.fit?.setup === "none" ? "gpu" : choice?.fit?.setup || "gpu") : "gpu",
    fit: choice?.fit, speed: choice?.speed, offer: choice?.offer, existingInstanceId: choice?.existing?.id, advanced, endpoint,
  };
  // A machine that was taken before it could be rented: back to the set-up page, without it.
  const backToMachines = () => {
    const failed = choice?.offer?.id;
    if (failed) setSkip((x) => new Set([...x, failed]));
    setTune((t) => ({ ...t, locks: { ...t.locks, machine: undefined } }));
    const i = history.lastIndexOf("tune");
    if (i < 0) { back(); return; }
    setHistory(history.slice(0, i)); setError(undefined); setStep("tune");
  };
  if (step === "starting" && jobId) return <StartingScreen key={jobId} ctx={ctx} jobId={jobId} onBack={back} onPickMachine={backToMachines} />;
  if (step === "advanced") return <Advanced plan={plan} summary={summary} onBack={back} onChange={(a) => setAdvanced(a)} onEndpoint={setEndpoint} />;
  if (step === "confirm") {
    return <Confirm recipe={recipe!} plan={plan} choice={choice!} where={where} runtime={runtime!} context={context} advanced={advanced} back={back} setEndpoint={setEndpoint} go={(s) => { if (s === "starting") setJobId(startJob(plan, ctx.cliPath)); go(s); }} />;
  }
  return null;
}

function Confirm({ recipe, plan, choice, where, runtime, context, advanced, back, go, setEndpoint }: {
  recipe: Recipe; plan: Plan; choice: { quant: Quant; offer?: CloudOption["offer"]; speed?: any; downloadMin?: number; downloadCost?: number; existing?: Instance };
  where: Where; runtime: RuntimeId; context: number; advanced: Plan["advanced"]; back: () => void; go: (s: Step) => void;
  setEndpoint: (e: NonNullable<Plan["endpoint"]>) => void;
}) {
  // Endpoint and API key: spunup's (the saved port for this model here, last time's key) unless you set your own.
  const ep = plan.endpoint || {};
  const [autoPort, setAutoPort] = useState<number>();
  useEffect(() => { fixedPort(recipe.id, where).then(setAutoPort).catch(() => {}); }, []);
  const kept = useMemo(() => keptApiKey(recipe.id, where), []);
  const port = ep.port ?? autoPort;
  const [mbps, setMbps] = useState<number | null>();   // undefined = still checking, null = could not tell
  useEffect(() => { if (where === "local") measureDownloadMBps(choice.quant).then((v) => setMbps(v ?? null)); }, []);
  const othersLocal = useMemo(() => {
    const ids = new Set(loadHistory().endpoints.filter((e) => e.where === "local" && (e.status === "running" || e.status === "starting")).map((e) => e.id));
    for (const j of activeJobs()) if (j.launcher.plan.where === "local") ids.add(j.launcher.endpoint?.id || j.id);
    return ids.size;
  }, []);
  {
    const q = choice!.quant;
    const dlCost = choice?.downloadCost || 0;
    const cost = where === "local" ? "free" : choice?.offer
      ? perHour(choice.offer.pricePerHour) + (dlCost >= 0.005 ? ` + about ${money(dlCost)} one-off download (${((q.size_gb || 0) + (planLora(plan)?.adapter?.sizeGb || 0)).toFixed(1)} GB${planLora(plan)?.adapter ? " model + LoRA" : " model"} at this host's ${money((choice.offer.downloadPerGb || 0) * 1000)}/TB)` : where === "runpod" ? " (RunPod does not charge for downloads)" : "")
      : "your machine's price";
    const dlGb = (q.size_gb || 0) + (planLora(plan)?.adapter?.sizeGb || 0);   // model + LoRA adapter
    const dlText = where !== "local"
      ? choice.downloadMin ? `~${choice.downloadMin} min (${Math.round(dlGb)} GB)` : `${Math.round(dlGb)} GB (the time depends on your machine's internet speed)`
      : mbps ? `~${Math.max(1, Math.round((dlGb * 1000) / mbps / 60))} min (${dlGb.toFixed(1)} GB at ${mbps < 10 ? mbps.toFixed(1) : Math.round(mbps)} MB/s)`
      : mbps === null ? `${dlGb.toFixed(1)} GB` : `${dlGb.toFixed(1)} GB (checking your internet speed…)`;
    const changed = [advanced?.extraArgs?.length && "extra flags", advanced?.command && "own start command", advanced?.lora && `LoRA ${advanced.lora.name}`].filter(Boolean);
    return (
      <Screen title="Ready to start">
        <Box flexDirection="column" marginBottom={1}>
          <Row k="Model" v={`${recipe!.name} (${q.label}${q.quant && q.quant !== q.label ? ` · ${q.quant}` : ""})`} />
          <Row k="Where" v={where === "local" ? `Local · ${setupLabel(plan.setup)}` : `${WHERE_NAME[where!]} · ${choice?.offer ? `${choice.offer.numGpus}× ${choice.offer.gpuName}` : `machine ${choice?.existing?.id}`}`} />
          <Row k="Runtime" v={`${RUNTIME_NAMES[runtime!]}   (${RUNTIME_NOTE})`} />
          <Row k="Context" v={fmtInt(context)} />
          {planLora(plan) && <Row k="LoRA" v={`${planLora(plan)!.adapter ? adapterLabel(planLora(plan)!.adapter!) : planLora(plan)!.localPath}  ${runtime === "vllm" || runtime === "sglang" ? `  (apps use the model name "${planLora(plan)!.name}" to get it)` : "  (used on every request)"}`} />}
          <Row k="Cost" v={cost} />
          <Row k="Download" v={dlText} />
          {choice?.speed && <Box><Box width={11}><Text dimColor>Speed</Text></Box><Tps min={choice.speed.tpsMin} max={choice.speed.tpsMax} /><Text>   </Text><Prefill value={choice.speed.prefill} /><Text dimColor>  (estimate)</Text></Box>}
          {changed.length > 0 && <Row k="Advanced" v={changed.join(", ")} />}
        </Box>
        {recipe!.known_issues.slice(0, 2).map((n, i) => <Text key={i} dimColor>· {n}</Text>)}
        {where === "local" && othersLocal > 0 && <Text color={C.warn}>Another model is already running or starting on this computer. They share its memory, so this one may not fit or may run slower. The estimates above assume this computer is free.</Text>}
        <Box flexDirection="column" marginBottom={1}>
          <Row k="Endpoint" v={`${port ? `http://localhost:${port}/v1` : "http://localhost:…/v1"}${ep.port ? "   (yours)" : ""}`} />
          <Row k="API key" v={ep.apiKey ? `${maskKey(ep.apiKey)}   (yours)` : kept ? `${maskKey(kept)}   (same as last time)` : "a new key, made when it starts"} />
          <Row k="Model name" v={`${ep.name || defaultModelName(recipe.name)}${ep.name ? "   (yours)" : ""}   apps send this as "model"`} />
          <Text dimColor>Change these in Advanced settings.</Text>
        </Box>
        <Box flexDirection="column"><Select items={[
          { value: "start", label: "Start" },
          { value: "advanced", label: "Advanced settings" },
          { value: "back", label: "Back" },
        ]} onBack={back} onSelect={(v) => (v === "start" ? go("starting") : v === "advanced" ? go("advanced") : back())} />
        </Box>
      </Screen>
    );
  }
}

/** An API key with its middle hidden. */
const maskKey = (k: string) => (k.length > 16 ? `${k.slice(0, 10)}…${k.slice(-4)}` : k);

function Row({ k, v }: { k: string; v: string }) {
  return <Box><Box width={11}><Text dimColor>{k}</Text></Box><Text>{v}</Text></Box>;
}

// ------------------------------------------------------------ Model picker
function ModelPicker({ onPick, onBack, error }: { onPick: (id: string) => void; onBack: () => void; error?: string }) {
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const [newOnes, setNewOnes] = useState<IndexEntry[]>([]);
  const [pasting, setPasting] = useState(false);
  useEffect(() => { checkNewModels().then(setNewOnes).catch(() => {}); }, []);
  const idx = useMemo(() => loadIndex(), []);
  const famOrder = useMemo(() => idx.families, [idx]);
  const all = useMemo(() => {
    const list = [...idx.models, ...newOnes];
    const byFam = (f: string) => famOrder.indexOf(f);
    // Popular models first, grouped by family; big downloads last within a family.
    return list.sort((a, b) => byFam(a.family) - byFam(b.family) || Number(!!a.isNew) - Number(!!b.isNew) || (b.downloads || 0) - (a.downloads || 0));
  }, [idx, newOnes]);
  const words = q.toLowerCase().split(/\s+/).filter(Boolean);
  const hits = all.filter((m) => words.every((w) => m.id.toLowerCase().includes(w) || m.family.includes(w)));
  const VISIBLE = Math.max(5, Math.min(14, (process.stdout.rows || 30) - 12));   // family headers add lines too
  // Paste row sits first and starts selected, so people see they can bring any model.
  const items = [{ id: "__paste", entry: undefined }, ...hits.map((m) => ({ id: m.id, entry: m }))];
  const i = Math.min(sel, items.length - 1);
  const start = Math.max(0, Math.min(i - 5, items.length - VISIBLE));
  usePaste((text) => {
    const id = parseHfId(text);
    if (id) onPick(id); else setQ((x) => x + text.trim());
  }, { isActive: !pasting });
  useInput((input, key) => {
    if (key.upArrow) setSel((s) => Math.max(0, s - 1));
    else if (key.downArrow) setSel((s) => Math.min(items.length - 1, s + 1));
    else if (key.return) {
      const it = items[i];
      if (it.id === "__paste") setPasting(true);
      else onPick(it.id);
    } else if (key.escape) { if (q) setQ(""); else onBack(); }
    else if (key.backspace || key.delete) setQ((x) => x.slice(0, -1));
    else if (input && !key.ctrl && !key.meta) { setQ((x) => x + input); setSel(1); }
  }, { isActive: !pasting });
  if (pasting) {
    return (
      <Screen title="Paste a Hugging Face link or name" footer="Enter · Esc back">
        <Text dimColor>For example: Qwen/Qwen3.8-27B or https://huggingface.co/Qwen/Qwen3.8-27B</Text>
        <TextInput onBack={() => setPasting(false)} validate={(v) => (parseHfId(v) ? undefined : "That is not a Hugging Face model name (it looks like org/model).")}
          onSubmit={(v) => onPick(parseHfId(v)!)} />
      </Screen>
    );
  }
  let lastFam = "";
  return (
    <Box flexDirection="column" paddingX={1}>
      <Text bold>Pick a model</Text>
      <Text>Search: {q}<Text color={C.accent}>_</Text></Text>
      {error && <Text color={C.error}>{error}</Text>}
      <Box flexDirection="column" marginTop={1}>
        {items.slice(start, start + VISIBLE).map((it, k) => {
          const real = start + k;
          const cur = real === i;
          if (it.id === "__paste") return <Box key="p" marginBottom={1}><Text color={cur ? C.accent : undefined}>{cur ? " › " : "   "}Paste a Hugging Face link or name…</Text></Box>;
          const e = it.entry!;
          const famHead = !q && e.family !== lastFam ? e.family : "";
          lastFam = e.family;
          return (
            <Box key={it.id} flexDirection="column">
              {famHead && <Text dimColor>   {idx.family_names?.[e.family] || e.family}</Text>}
              <Box>
                <Text color={cur ? C.accent : undefined}>{cur ? " › " : "   "}</Text>
                <Box width={44}><Text bold={cur}>{it.id.split("/")[1].slice(0, 43)}</Text></Box>
                <Box width={20}><Text dimColor>{paramsText(e)}</Text></Box>
                <Text color={e.isNew ? C.accent : C.alive}>{e.isNew ? "new" : e.popular ? "popular" : ""}</Text>
                {e.gated && <Text color={C.warn}>  gated</Text>}
              </Box>
            </Box>
          );
        })}
      </Box>
      <Box marginTop={1}><Text dimColor>Type to search · ↑↓ move · Enter pick · Esc back   {hits.length} models</Text></Box>
    </Box>
  );
}

/** Every GGUF file in the model's GGUF repo (was "Pick a particular quantization file" in Advanced). */
function GgufFiles({ recipe, onPick, onBack }: { recipe: Recipe; onPick: (q: Quant) => void; onBack: () => void }) {
  const [files, setFiles] = useState<Quant[]>();
  const repo = recipe.quants.find((q) => q.format === "gguf")?.repo;
  useEffect(() => {
    (async () => {
      const out: Quant[] = recipe.quants.filter((q) => q.format === "gguf");
      const tree = repo ? await hfTree(repo).catch(() => []) : [];
      // Split files: only part 1 (parts 2, 10, 12… are not models on their own).
      for (const f of tree.filter((x) => x.path.endsWith(".gguf") && !/mmproj|lora|adapter/i.test(x.path) && Number(/-(\d+)-of-\d+\.gguf$/i.exec(x.path)?.[1] ?? 1) === 1)) {
        if (out.some((o) => o.file === f.path)) continue;
        const parts = /-of-(\d+)/.exec(f.path) ? Number(/-of-(\d+)/.exec(f.path)![1]) : 1;
        out.push({ id: `gguf:${f.path}`, label: f.path.split("/").pop()!.replace(/(-0*1-of-\d+)?\.gguf$/, ""), quant: QRE.exec(f.path.split("/").pop()!)?.[0].toUpperCase(), format: "gguf", repo: repo!, file: f.path, parts, size_gb: Math.round(((f.size || 0) * parts) / 1e8) / 10, runtimes: ["llamacpp"] });
      }
      setFiles(out.sort((a, b) => (b.size_gb || 0) - (a.size_gb || 0)));
    })();
  }, []);
  if (!files) return <Busy text="Listing files…" />;
  return (
    <Screen title="Pick a GGUF file" note={`From ${repo}. Runs with llama.cpp.`}>
      <Select items={files.map((q, i) => ({ value: String(i), label: q.file ? q.file.split("/").pop()! : q.label, hint: `${q.size_gb ?? "?"} GB` }))}
        onBack={onBack} onSelect={(v) => onPick(files[Number(v)])} />
    </Screen>
  );
}

function ExistingPicker({ recipe, runtime, where, summary, onPick, onBack }: { recipe: Recipe; runtime: RuntimeId; where: "vast" | "runpod"; summary: [string, string][]; onPick: (i: Instance, q: Quant) => void; onBack: () => void }) {
  const [list, setList] = useState<Instance[]>();
  const [inst, setInst] = useState<Instance>();
  useEffect(() => { provider(where).list().then((l) => setList(l.filter((i) => i.status === "running"))).catch(() => setList([])); }, []);
  if (!list) return <Busy text="Getting your machines…" />;
  if (!inst) {
    return (
      <Screen title="Use a machine I already have" summary={summary} note="Only running machines are listed. spunup adds its SSH key and uses the machine as it is.">
        <Select items={[...list.map((i) => ({ value: i.id, label: `${i.label || i.id}`, hint: `${i.numGpus || 1}× ${i.gpuName || "GPU"} · ${perHour(i.pricePerHour || 0)}` })), { value: "back", label: "Back" }]}
          onBack={onBack} onSelect={(v) => (v === "back" ? onBack() : setInst(list.find((i) => i.id === v)))} />
      </Screen>
    );
  }
  const qs = recipe.quants.filter((q) => q.runtimes.includes(runtime));
  return (
    <Screen title="Pick a size" summary={summary} note={SIZE_NOTE}>
      <Select items={qs.map((q, i) => ({ value: String(i), label: q.label, hint: `${Math.round(q.size_gb || 0)} GB` }))} onBack={() => setInst(undefined)} onSelect={(v) => onPick(inst, qs[Number(v)])} />
    </Screen>
  );
}

// ------------------------------------------------------------ Advanced (spec 7.3)
function Advanced({ plan, summary, onBack, onChange, onEndpoint }: { plan: Plan; summary: [string, string][]; onBack: () => void; onChange: (a: Plan["advanced"]) => void; onEndpoint: (e: NonNullable<Plan["endpoint"]>) => void }) {
  const [mode, setMode] = useState<"menu" | "flags" | "cmd" | "lora" | "loraName" | "url" | "key" | "name">("menu");
  // Your own endpoint: port (the URL is always http://localhost:PORT/v1), API key and model name.
  const ep = plan.endpoint || {};
  const [autoPort, setAutoPort] = useState<number>();
  useEffect(() => { fixedPort(plan.recipe.id, plan.where).then(setAutoPort).catch(() => {}); }, []);
  const kept = useMemo(() => keptApiKey(plan.recipe.id, plan.where), []);
  const [epErr, setEpErr] = useState<string>();
  const url = (p?: number) => `http://localhost:${p ?? "…"}/v1`;
  // All hooks before the first early return (React needs the same hooks on every render).
  const [msg, setMsg] = useState<string>();
  const [pending, setPending] = useState<Omit<LoraUse, "name">>();
  const a = plan.advanced || {};
  const rr = plan.recipe.runtimes[plan.runtime];
  if (mode === "url") {
    return (
      <Screen title="Endpoint URL" footer="Enter save · Esc back">
        <Text dimColor>Type a port (8080) or a full URL (http://localhost:8080/v1). It stays on this computer: localhost, with the /v1 path every runtime serves. Leave it empty for spunup's pick ({url(autoPort)}).</Text>
        <TextInput initial={ep.port ? url(ep.port) : ""} placeholder={url(autoPort)} onBack={() => { setEpErr(undefined); setMode("menu"); }}
          onSubmit={async (v) => {
            const t = v.trim();
            if (!t) { onEndpoint({ ...ep, port: undefined }); setEpErr(undefined); setMode("menu"); return; }
            const m = /^(?:https?:\/\/)?(?:(localhost|127\.0\.0\.1)(?::(\d+))?|(\d+))(\/v1\/?|\/)?$/i.exec(t);
            if (!m || (m[1] && !m[2])) { setEpErr("Use a port like 8080, or a URL like http://localhost:8080/v1. spunup keeps endpoints on this computer only (localhost), so other hosts and paths can't be used."); return; }
            const n = Number(m[2] || m[3]);
            const why = n === autoPort ? undefined : await checkPort(n, plan.recipe.id, plan.where);
            if (why) { setEpErr(why); return; }
            onEndpoint({ ...ep, port: n }); setEpErr(undefined); setMode("menu");
          }} />
        {epErr && <Text color={C.warn}>{epErr}</Text>}
      </Screen>
    );
  }
  if (mode === "key") {
    return (
      <Screen title="API key" footer="Enter save · Esc back">
        <Text dimColor>Type your own API key: 8 to 128 letters, numbers, or . _ - ~. Apps send it as "Authorization: Bearer KEY". Leave it empty for spunup's ({kept ? "the same key as last time" : "a new key, made when it starts"}).</Text>
        <TextInput mask initial={ep.apiKey || ""} onBack={() => setMode("menu")}
          validate={(v) => (v.trim() && !/^[A-Za-z0-9._~-]{8,128}$/.test(v.trim()) ? "Use 8 to 128 letters, numbers, or . _ - ~ (no spaces)." : undefined)}
          onSubmit={(v) => { onEndpoint({ ...ep, apiKey: v.trim() || undefined }); setMode("menu"); }} />
      </Screen>
    );
  }
  if (mode === "name") {
    return (
      <Screen title="Model name" footer="Enter save · Esc back">
        <Text dimColor>The name apps send as "model" in their requests, and that /v1/models lists. Letters, numbers, and . _ - / : (up to 64). Leave it empty for spunup's ({defaultModelName(plan.recipe.name)}).</Text>
        <TextInput initial={ep.name || ""} placeholder={defaultModelName(plan.recipe.name)} onBack={() => setMode("menu")}
          validate={(v) => (v.trim() && !/^[A-Za-z0-9._\/:-]{1,64}$/.test(v.trim()) ? "Use up to 64 letters, numbers, or . _ - / : (no spaces)." : undefined)}
          onSubmit={(v) => { onEndpoint({ ...ep, name: v.trim() || undefined }); setMode("menu"); }} />
        {planLora(plan) && (plan.runtime === "vllm" || plan.runtime === "sglang") && <Text dimColor>The LoRA adapter keeps its own name ({planLora(plan)!.name}); this one is for the model without it.</Text>}
      </Screen>
    );
  }
  if (mode === "flags") {
    return (
      <Screen title="Edit runtime flags" footer="Enter save · Esc back">
        <Text dimColor>spunup's flags for {RUNTIME_NAMES[plan.runtime]}: {[...(rr.args || []), rr.tool_parser ? `--tool-call-parser ${rr.tool_parser}` : "", rr.reasoning_parser ? `--reasoning-parser ${rr.reasoning_parser}` : ""].filter(Boolean).join(" ") || "(none)"}</Text>
        <Text dimColor>Type flags to add or replace (a flag you type replaces spunup's flag of the same name).</Text>
        <TextInput initial={(a.extraArgs || []).join(" ")} onBack={() => setMode("menu")} onSubmit={(v) => { onChange({ ...a, extraArgs: v.trim() ? v.trim().match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g)!.map((x) => x.replace(/^["']|["']$/g, "")) : undefined }); setMode("menu"); }} />
      </Screen>
    );
  }
  if (mode === "cmd") {
    return (
      <Screen title="Set your own start command" footer="Enter save · Esc back">
        <Text dimColor>Placeholders: {"{model} {port} {api_key} {ctx} {name}"}. Keep --host 127.0.0.1 so the endpoint stays private.</Text>
        <TextInput initial={a.command || ""} onBack={() => setMode("menu")} onSubmit={(v) => {
          if (v && !/127\.0\.0\.1|localhost/.test(v)) { setMsg("Your command must bind to 127.0.0.1 (spunup never exposes a public endpoint)."); setMode("menu"); return; }
          onChange({ ...a, command: v.trim() || undefined }); setMode("menu");
        }} />
      </Screen>
    );
  }
  if (mode === "lora") {
    return (
      <Screen title="Add a LoRA adapter" footer="Enter · Esc back">
        <Text dimColor>A LoRA adapter is a small add-on that changes how the model behaves. It must be made for {plan.recipe.name}. Paste a Hugging Face adapter name{plan.where === "local" ? " or a folder / .gguf path on this computer" : ""}.</Text>
        <Text dimColor>{RUNTIME_NAMES[plan.runtime]} needs {plan.runtime === "llamacpp" ? "a GGUF adapter file" : plan.runtime === "mlx" ? "an adapter made with mlx-lm (adapters.safetensors)" : "a Hugging Face (PEFT) adapter: adapter_config.json + adapter_model.safetensors"}.</Text>
        <TextInput onBack={() => setMode("menu")} onSubmit={async (v) => {
          v = v.trim();
          const id = parseHfId(v);
          if (!id) {
            if (plan.where !== "local") { setMsg("For a rented machine, use a Hugging Face adapter name."); setMode("menu"); return; }
            setPending({ localPath: v }); setMode("loraName"); return;
          }
          const found = await findAdapters(id).catch(() => [] as Adapter[]);
          const fits = found.filter((x) => ADAPTER_RUNTIMES[x.kind].includes(plan.runtime));
          if (!found.length) { setMsg(`${id} has no LoRA adapter files.`); setMode("menu"); return; }
          if (!fits.length) { setMsg(`${id} has a ${KIND_NAME[found[0].kind]} adapter, which ${RUNTIME_NAMES[plan.runtime]} cannot load. Go back and pick ${ADAPTER_RUNTIMES[found[0].kind].map((r) => RUNTIME_NAMES[r]).join(" or ")}.`); setMode("menu"); return; }
          const base = await baseModelOf(fits[0]).catch(() => undefined);
          const same = (x: string) => x.toLowerCase().split("/").pop()!.replace(/-(instruct|it|chat|bf16|fp8)$/g, "");
          if (base && same(base) !== same(plan.recipe.id)) setMsg(`Warning: this adapter was made for ${base}, not ${plan.recipe.id}. It will most likely fail to load.`);
          else setMsg(`Adapter matches ${plan.recipe.name}.`);
          setPending({ adapter: fits[0] });
          setMode("loraName");
        }} />
      </Screen>
    );
  }
  if (mode === "loraName") {
    return (
      <Screen title="Name for the adapter" footer="Enter · Esc back">
        <Text dimColor>Apps pick the adapter by this name as the "model" in their requests.</Text>
        <TextInput initial={pending?.adapter ? adapterName(pending.adapter) : "my-lora"} onBack={() => setMode("lora")} onSubmit={(n) => { onChange({ ...a, lora: { name: n.trim() || "my-lora", ...pending } }); setMode("menu"); }} />
      </Screen>
    );
  }
  const loraOk = rr.lora?.supported !== false;
  return (
    <Screen title="Advanced settings" summary={summary} note="You can ignore this screen. spunup's settings already come from the model recipe.">
      {msg && <Text color={/Warning|not|must/.test(msg) ? C.warn : C.alive}>{msg}</Text>}
      <Select labelWidth={32} items={[
        { value: "url", label: "Endpoint URL", hint: ep.port ? `${url(ep.port)} (yours)` : `${url(autoPort)} (spunup's)` },
        { value: "key", label: "API key", hint: ep.apiKey ? `${maskKey(ep.apiKey)} (yours)` : kept ? "same as last time (spunup's)" : "a new key (spunup's)" },
        { value: "name", label: "Model name", hint: ep.name ? `${ep.name} (yours)` : `${defaultModelName(plan.recipe.name)} (spunup's)` },
        { value: "flags", label: "Edit runtime flags", hint: a.extraArgs?.length ? a.extraArgs.join(" ") : undefined, separatorBefore: true },
        { value: "cmd", label: "Set your own start command", hint: a.command ? "set" : undefined },
        { value: "lora", label: "Use a LoRA adapter from elsewhere", hint: a.lora ? `${a.lora.name} = ${a.lora.adapter ? adapterLabel(a.lora.adapter) : a.lora.localPath}` : loraOk ? (plan.lora ? `replaces ${plan.lora.name}` : undefined) : "not supported for this model", disabled: !loraOk },
        { value: "reset", label: "Reset to recipe settings" },
        { value: "back", label: "Done", separatorBefore: true },
      ]} onBack={onBack} onSelect={(v) => {
        if (v === "back") onBack();
        else if (v === "reset") { onChange({}); setMsg("Back to the recipe settings."); }
        else setMode(v as any);
      }} />
    </Screen>
  );
}

