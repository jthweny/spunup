import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import { Select, Spinner, type Item } from "../ui/components.js";
import { C } from "../ui/theme.js";
import { loadHistory, upsertEndpoint, type Endpoint } from "../core/store.js";
import { getKey } from "../core/keys.js";
import { provider, type Instance } from "../providers/index.js";
import { ensureKeeper, costSoFar } from "../core/keeper.js";
import { money, perHour } from "../core/cost.js";
import { visibleJobs, activeJobs, jobEvents, type Job } from "../core/launch.js";
import { takeInstallNote } from "../core/selfinstall.js";
import type { Ctx } from "../app.js";

const WHERE = { local: "Local", vast: "Vast.ai", runpod: "RunPod" };

const BANNER = [
  "███████╗ ██████╗  ██╗   ██╗ ███╗   ██╗ ██╗   ██╗ ██████╗",
  "██╔════╝ ██╔══██╗ ██║   ██║ ████╗  ██║ ██║   ██║ ██╔══██╗",
  "███████╗ ██████╔╝ ██║   ██║ ██╔██╗ ██║ ██║   ██║ ██████╔╝",
  "╚════██║ ██╔═══╝  ██║   ██║ ██║╚██╗██║ ██║   ██║ ██╔═══╝",
  "███████║ ██║      ╚██████╔╝ ██║ ╚████║ ╚██████╔╝ ██║",
  "╚══════╝ ╚═╝       ╚═════╝  ╚═╝  ╚═══╝  ╚═════╝  ╚═╝",
];
const RULE = "─".repeat(57);

/** One tiny request per saved endpoint → alive or not. */
async function ping(e: Endpoint): Promise<boolean> {
  try {
    const r = await fetch(`http://127.0.0.1:${e.port}/v1/models`, { headers: { authorization: `Bearer ${e.apiKey}` }, signal: AbortSignal.timeout(2500) });
    return r.ok;
  } catch {
    return false;
  }
}

export function StartScreen({ ctx }: { ctx: Ctx }) {
  const [hist, setHist] = useState<Endpoint[]>(() => loadHistory().endpoints);
  const [alive, setAlive] = useState<Record<string, boolean>>({});
  const [orphans, setOrphans] = useState<(Instance & { provider: "vast" | "runpod" })[]>([]);
  const [open, setOpen] = useState(false);   // closed until the user opens it
  const [toggled, setToggled] = useState(false);
  const [orphan, setOrphan] = useState<(Instance & { provider: "vast" | "runpod" })>();
  const [note, setNote] = useState<string | undefined>(takeInstallNote);
  const [quitAsk, setQuitAsk] = useState(false);
  const [jobs, setJobs] = useState<Job[]>(visibleJobs);
  const [, tick] = useState(0);

  // Setups running in the background: live progress, and a fresh check when one finishes.
  useEffect(() => {
    let watching = activeJobs();
    const onChange = () => {
      for (const j of watching.filter((x) => x.finished)) {
        setHist(loadHistory().endpoints);
        const e = j.result;
        if (e) {
          ping(e).then((ok) => setAlive((a) => ({ ...a, [e.id]: ok })));
          setNote(`✓ ${e.name} is ready at http://localhost:${e.port}/v1`);
        } else if (j.launcher.error) setNote(`${j.launcher.endpoint?.name || j.launcher.plan.recipe.name} did not start. Open it in History to see why.`);
      }
      watching = activeJobs();
      setJobs(visibleJobs());
    };
    jobEvents.on("change", onChange);
    return () => { jobEvents.off("change", onChange); };
  }, []);

  useEffect(() => {
    // Reopen any tunnel that should be open (spec 8), then the two cheap checks (spec 11.1).
    ensureKeeper(ctx.cliPath);
    // A local runtime whose process has gone (reboot, crash) is stopped, not running.
    for (const e of loadHistory().endpoints) {
      if (e.where === "local" && (e.status === "running" || e.status === "starting") && e.pid) {
        try { process.kill(e.pid, 0); } catch { upsertEndpoint({ ...e, status: "stopped" }); }
      }
    }
    // A cloud setup that was cut off (spunup closed) left a rented machine: show it as running so it can be deleted.
    const mine = new Set(activeJobs().map((j) => j.launcher.endpoint?.id));
    for (const e of loadHistory().endpoints) {
      if (e.where !== "local" && e.status === "starting" && !mine.has(e.id)) upsertEndpoint({ ...e, status: e.instanceId ? "running" : "failed" });
    }
    const h = loadHistory().endpoints;
    setHist(h);
    Promise.all(h.filter((e) => e.status !== "deleted").map(async (e) => [e.id, await ping(e)] as const)).then((r) => setAlive(Object.fromEntries(r)));
    (async () => {
      const found: (Instance & { provider: "vast" | "runpod" })[] = [];
      for (const pid of ["vast", "runpod"] as const) {
        if (!getKey(pid)) continue;
        try {
          const list = await provider(pid).list();
          for (const inst of list) {
            const mine = h.find((e) => e.where === pid && e.instanceId === inst.id);
            if (mine) {
              // Keep History in step with the provider (a machine stopped or deleted elsewhere).
              const status = inst.status === "stopped" ? "stopped" : inst.status === "gone" ? "deleted" : mine.status;
              if (status !== mine.status && mine.status !== "starting") upsertEndpoint({ ...mine, status, stoppedAt: status === "stopped" ? Date.now() : mine.stoppedAt, costBefore: costSoFar(mine) - (mine.oneOffCost || 0) });
              if (inst.downloadCost !== undefined) upsertEndpoint({ ...(loadHistory().endpoints.find((x) => x.id === mine.id) || mine), oneOffCost: inst.downloadCost });
            } else if (inst.status === "running" || inst.status === "starting") found.push({ ...inst, provider: pid });
          }
          for (const e of h.filter((x) => x.where === pid && x.instanceId && (x.status === "running" || x.status === "stopped"))) {
            if (!list.some((i) => i.id === e.instanceId)) upsertEndpoint({ ...e, status: "deleted", costBefore: costSoFar(e) - (e.oneOffCost || 0) });
          }
        } catch { /* offline or key problem: show what we know */ }
      }
      setOrphans(found);
      setHist(loadHistory().endpoints);
    })();
    const t = setInterval(() => tick((x) => x + 1), 5000);
    return () => clearInterval(t);
  }, []);

  // Failed starts never served anything, so History only lists endpoints that ran (or rented machines).
  const shown = hist.filter((e) => (e.status !== "failed" || !!e.instanceId) && (e.status !== "deleted" || Date.now() - (e.createdAt || 0) < 30 * 86400_000)).slice(0, 12);
  const jobIds = new Set(jobs.map((j) => j.launcher.endpoint?.id).filter(Boolean));
  const rows: Item<string>[] = [];
  const startingN = jobs.filter((j) => !j.launcher.error).length, failedN = jobs.length - startingN;
  const closedHint = [startingN && `${startingN} starting`, failedN && `${failedN} did not start`, orphans.length && `${orphans.length} running outside spunup`].filter(Boolean).join(" · ");
  rows.push({ value: "toggle", label: `History ${open ? "▾" : "▸"}`, hint: shown.length || orphans.length || jobs.length ? (open ? undefined : closedHint || undefined) : "nothing yet" });
  if (open) {
    for (const j of jobs) {
      const l = j.launcher;
      const step = l.steps.find((x) => x.state === "doing") || l.steps.find((x) => x.state === "failed");
      const progress = l.error ? <Text color={C.error}>did not start · open to see why</Text>
        : <Text>starting · {step ? step.label.toLowerCase() : "getting ready"}{step?.detail ? <Text dimColor>  {step.detail}</Text> : null}</Text>;
      rows.push({
        value: `j:${j.id}`, indent: 2, label: "",
        right: <Box>{l.error ? <Text color={C.error}>✗ </Text> : <Box marginRight={1}><Spinner /></Box>}<Box width={22}><Text>{(l.endpoint?.name || l.plan.recipe.name).slice(0, 21)}</Text></Box><Box width={9}><Text dimColor>{WHERE[l.plan.where]}</Text></Box>{progress}</Box>,
      });
    }
    for (const e of shown.filter((x) => !jobIds.has(x.id))) {
      const isAlive = alive[e.id];
      const dot = <Text color={isAlive ? C.alive : C.dead}>{isAlive ? "●" : "○"} </Text>;
      let money_: React.ReactNode;
      if (e.where === "local") money_ = <Text dimColor>free</Text>;
      else if (e.status === "running" || e.status === "starting") money_ = <Text>{perHour(e.pricePerHour || 0)}   <Text bold>{money(costSoFar(e))} so far</Text></Text>;
      else if (e.status === "stopped") money_ = <Text color={C.warn}>{perHour(e.stoppedPricePerHour || 0)} to keep · {money(costSoFar(e))} so far</Text>;
      else money_ = <Text dimColor>{money(costSoFar(e))} total</Text>;
      const place = e.status === "deleted" ? "(deleted)" : e.status === "stopped" ? "(stopped)" : `localhost:${e.port}`;
      rows.push({
        value: `e:${e.id}`, indent: 2, label: "",
        right: <Box>{dot}<Box width={22}><Text>{e.name.slice(0, 21)}</Text></Box><Box width={9}><Text dimColor>{WHERE[e.where]}</Text></Box><Box width={17}><Text>{place}</Text></Box>{money_}</Box>,
      });
    }
    for (const o of orphans) {
      rows.push({ value: `o:${o.provider}:${o.id}`, indent: 2, label: "", right: <Text color={C.warn}>● {o.label || o.id} on {o.provider === "vast" ? "Vast.ai" : "RunPod"} is running and costing {perHour(o.pricePerHour || 0)} ({/^spunup-/.test(o.label || "") ? "started by spunup on another computer or install" : "not started by spunup"})</Text> });
    }
  }
  rows.push({ value: "serve", label: "Serve a new model" });
  rows.push({ value: "keys", label: "Keys & tokens", separatorBefore: true });

  if (quitAsk) {
    const active = activeJobs();
    const rented = active.some((j) => j.launcher.endpoint?.instanceId);
    return (
      <Box flexDirection="column" paddingX={1}>
        <Text color={C.warn}>{active.length === 1 ? "1 model is" : `${active.length} models are`} still starting. Closing spunup stops {active.length === 1 ? "its" : "their"} setup.</Text>
        {rented && <Text>Machines already rented keep running and cost money until you delete them. They stay in History.</Text>}
        <Box marginTop={1}><Select items={[
          { value: "stay", label: "Keep spunup open" },
          { value: "quit", label: "Close anyway" },
        ]} onBack={() => setQuitAsk(false)} onSelect={(v) => {
          if (v === "stay") { setQuitAsk(false); return; }
          // Local runtimes that are half loaded are closed; rented machines are left for History.
          for (const j of active) {
            if (j.launcher.plan.where === "local") j.launcher.cancel();
            else if (j.launcher.endpoint?.instanceId) { j.launcher.cancelled = true; upsertEndpoint({ ...j.launcher.endpoint, status: "running" }); }
          }
          ctx.quit();
        }} /></Box>
      </Box>
    );
  }

  if (orphan) {
    const p = provider(orphan.provider);
    return (
      <Box flexDirection="column" paddingX={1}>
        <Text bold>{orphan.label || orphan.id} on {p.name}</Text>
        <Text>This machine is running and costing {perHour(orphan.pricePerHour || 0)}. {/^spunup-/.test(orphan.label || "") ? "It was started by spunup somewhere else, so it is not in this History." : "spunup did not start it."}</Text>
        <Box marginTop={1}><Select items={[
          { value: "stop", label: "Stop machine", hint: "(kept for you, lower hourly price)" },
          { value: "delete", label: "Delete machine", hint: "(its charges stop; its data is gone)" },
          { value: "leave", label: "Leave it running" },
        ]} onBack={() => setOrphan(undefined)} onSelect={async (v) => {
          const o = orphan;
          setOrphan(undefined);
          if (v === "leave") return;
          try {
            if (v === "stop") await p.stop(o.id); else await p.destroy(o.id);
            setOrphans((list) => list.filter((x) => x.id !== o.id));
            setNote(v === "stop" ? `Stopped ${o.label || o.id}. ${p.stopNote}` : `Deleted ${o.label || o.id}.`);
          } catch (e) { setNote(`That did not work: ${(e as Error).message}`); }
        }} /></Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" paddingX={1}>
      {(process.stdout.columns || 80) >= RULE.length + 2 ? (
        // The big banner only where it fits; a narrow window would wrap it into noise.
        <Box flexDirection="column" marginTop={1} marginBottom={1}>
          {BANNER.map((l, i) => <Text key={i} bold color={C.accent}>{l}</Text>)}
          {/* Rules fade from the banner's colour down to the light grey. */}
          <Text color={C.accent}>{RULE}</Text>
          <Text color={C.accent} dimColor>{RULE}</Text>
          <Text color="gray">{RULE}</Text>
          <Text dimColor>{RULE}</Text>
          <Text dimColor>run AI models with one command</Text>
        </Box>
      ) : (
        <Box marginBottom={1}><Text bold color={C.accent}>spunup</Text><Text dimColor>  run AI models with one command</Text></Box>
      )}
      <Select key={`${open}-${rows.length}`} items={rows} initial={toggled ? 0 : rows.findIndex((r) => r.value === "serve")}
        onBack={() => (activeJobs().length ? setQuitAsk(true) : ctx.quit())}
        onSelect={(v) => {
          if (v === "toggle") { setOpen((x) => !x); setToggled(true); }
          else if (v === "serve") ctx.go({ name: "serve" });
          else if (v === "keys") ctx.go({ name: "keys" });
          else if (v.startsWith("e:")) ctx.go({ name: "endpoint", id: v.slice(2) });
          else if (v.startsWith("j:")) ctx.go({ name: "starting", id: v.slice(2) });
          else if (v.startsWith("o:")) setOrphan(orphans.find((o) => `o:${o.provider}:${o.id}` === v));
        }} />
      {note && <Text color={/not work|did not start/.test(note) ? C.error : C.alive}>{note}</Text>}
      <Box marginTop={1}><Text dimColor>↑↓ move · Enter pick · Esc quit   Endpoints work on this computer only.</Text></Box>
    </Box>
  );
}
