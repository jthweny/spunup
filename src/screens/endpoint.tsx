import React, { useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import { runSync } from "../core/proc.js";
import { Select, Busy, Tps, Prefill, TextInput, type Item } from "../ui/components.js";
import { C } from "../ui/theme.js";
import { getEndpoint, upsertEndpoint, type Endpoint } from "../core/store.js";
import { costSoFar, ensureKeeper, keeperStatus } from "../core/keeper.js";
import { money, perHour, checkedCost } from "../core/cost.js";
import { stopEndpoint, deleteEndpoint, deleteKeptDisk, restartEndpoint, refreshDownloadCost } from "../core/launch.js";
import { provider } from "../providers/index.js";
import { RUNTIME_NAMES } from "../runtimes/index.js";
import { IS_MAC, IS_WIN } from "../core/paths.js";
import type { Ctx } from "../app.js";

function copy(text: string): boolean {
  const tries: [string, string[]][] = IS_MAC ? [["pbcopy", []]] : IS_WIN ? [["clip.exe", []]] : [["wl-copy", []], ["xclip", ["-selection", "clipboard"]], ["xsel", ["--clipboard", "--input"]]];
  for (const [cmd, args] of tries) {
    const r = runSync(cmd, args, { input: text, stdio: ["pipe", "ignore", "ignore"] });
    if (r.status === 0) return true;
  }
  return false;
}

const TIMES = [30, 60, 180];
const timeLabel = (m: number) => (m % 60 === 0 ? `${m / 60} hour${m === 60 ? "" : "s"}` : m > 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`);

export function EndpointScreen({ ctx, id }: { ctx: Ctx; id: string }) {
  const [e, setE] = useState<Endpoint | undefined>(() => getEndpoint(id));
  const [msg, setMsg] = useState<string>();
  const [busy, setBusy] = useState<string>();
  const [showKey, setShowKey] = useState(false);
  const [example, setExample] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [timeOpen, setTimeOpen] = useState(false);
  const [custom, setCustom] = useState<"h" | "m">();
  const [customH, setCustomH] = useState(0);
  const [onTime, setOnTime] = useState(false);   // which half of the auto-stop line is selected
  const [billed, setBilled] = useState<{ usd: number; checked: boolean }>();
  const [alive, setAlive] = useState<boolean>();
  const [, tick] = useState(0);

  useEffect(() => {
    const t = setInterval(() => { setE(getEndpoint(id)); tick((x) => x + 1); }, 1000);
    const check = async () => {
      const cur = getEndpoint(id);
      if (!cur) return;
      try {
        const r = await fetch(`http://127.0.0.1:${cur.port}/v1/models`, { headers: { authorization: `Bearer ${cur.apiKey}` }, signal: AbortSignal.timeout(3000) });
        setAlive(r.ok);
      } catch { setAlive(false); }
      if (cur.where !== "local") refreshDownloadCost(cur).then((n) => checkedCost(n)).then(setBilled).catch(() => {});
    };
    check();
    const t2 = setInterval(check, 30_000);
    return () => { clearInterval(t); clearInterval(t2); };
  }, [id]);

  useInput((input, key) => {
    if (custom || timeOpen) return;
    if (input === "c" && e) setMsg(copy(e.apiKey) ? "API key copied." : `Could not reach the clipboard. Your key: ${e.apiKey}`);
    if (input === "k") setShowKey((x) => !x);
    if (input === "x") setExample((x) => !x);
  });

  if (!e) return <Box paddingX={1} flexDirection="column"><Text>This endpoint is no longer in History.</Text><Select items={[{ value: "home", label: "Back to History" }]} onSelect={() => ctx.go({ name: "start" })} onBack={() => ctx.go({ name: "start" })} /></Box>;
  if (busy) return <Busy text={busy} />;
  const cloud = e.where !== "local";
  const prov = cloud ? provider(e.where as "vast" | "runpod") : undefined;
  const url = `http://localhost:${e.port}/v1`;
  const running = e.status === "running" || e.status === "starting";
  const st = keeperStatus()?.endpoints[e.id];
  const so = costSoFar(e);
  const shownCost = billed?.checked && billed.usd > so ? billed.usd : so;

  const doIt = async (label: string, f: () => Promise<Endpoint | void>) => {
    setBusy(label);
    try { const n = await f(); if (n) setE(n); setMsg(undefined); }
    catch (err) { setMsg(`That did not work: ${(err as Error).message}`); }
    setBusy(undefined);
  };

  const as = e.autoStop || { enabled: false, minutes: 30, action: "stop" as const };
  const setAuto = (a: Partial<NonNullable<Endpoint["autoStop"]>>) => {
    const n = { ...e, autoStop: { ...as, ...a } };
    upsertEndpoint(n);
    setE(n);
    if (n.autoStop.enabled) ensureKeeper(ctx.cliPath);
  };

  if (custom) {
    return (
      <Box flexDirection="column" paddingX={1}>
        <Text>   Auto-stop when idle: custom time</Text>
        {custom === "h"
          ? <Box><Text>   Custom time   H: </Text><TextInput onBack={() => setCustom(undefined)} validate={(v) => (/^\d+$/.test(v.trim()) ? undefined : "Hours: a whole number, 0 or more.")} onSubmit={(v) => { setCustomH(Number(v)); setCustom("m"); }} /><Text dimColor>  (0–∞)</Text></Box>
          : <Box><Text>   Custom time   H: {customH}   M: </Text><TextInput onBack={() => setCustom("h")} validate={(v) => (/^\d+$/.test(v.trim()) && Number(v) <= 60 ? (customH * 60 + Number(v) > 0 ? undefined : "Pick at least 1 minute.") : "Minutes: 0 to 60.")} onSubmit={(v) => { setAuto({ minutes: customH * 60 + Number(v) }); setCustom(undefined); setTimeOpen(false); }} /><Text dimColor>  (0–60)</Text></Box>}
      </Box>
    );
  }

  const items: Item<string>[] = [{ value: "home", label: "Back to History" }];
  if (cloud) {
    if (running) items.push({ value: "stop", label: "Stop machine", hint: "(kept for you, lower hourly price)" });
    if (e.status === "stopped") items.push({ value: "start", label: "Start machine again" });
    if (e.status !== "deleted") {
      const separate = !!prov?.separateDisk && !!e.volumeId;
      if (separate) {
        items.push({ value: "delmenu", label: `Delete machine ${deleteOpen ? "▴" : "▾"}` });
        if (deleteOpen) {
          items.push({ value: "delKeep", label: "Machine only", hint: "(disk kept, still charged)", indent: 4 });
          items.push({ value: "del", label: "Machine + disk", hint: "(no more charges)", indent: 4 });
        }
      } else items.push({ value: "del", label: "Delete machine", hint: "(no more charges)" });
    }
    if (e.status === "deleted" && e.diskKept) items.push({ value: "deldisk", label: "Delete disk", hint: "(kept with \"Machine only\"; no more charges)" });
    if (running) items.push({ value: "reconnect", label: "Reconnect", hint: "(reopen the private tunnel)" });
    if (e.status !== "deleted") {
      items.push({
        value: "auto", label: "",
        right: <Box>
          <Box width={34}><Text>Auto-stop when idle: </Text><Text bold={!onTime} inverse={!onTime}>{as.enabled ? `on (${as.action})` : "off"}</Text></Box>
          <Text color={C.editable} inverse={onTime}>{timeLabel(as.minutes)} ▾</Text>
        </Box>,
      });
      if (timeOpen) {
        for (const m of TIMES) items.push({ value: `t:${m}`, label: timeLabel(m), indent: 37, color: m === as.minutes ? C.editable : undefined });
        items.push({ value: "t:custom", label: "Custom…", indent: 37 });
      }
    }
  } else {
    if (running) items.push({ value: "stop", label: "Stop", hint: "(closes the runtime on this computer)" });
    else items.push({ value: "start", label: "Start again" });
  }

  return (
    <Box flexDirection="column" paddingX={1}>
      {running && alive !== false
        ? <Text color={C.alive} bold> ✓ Your model is running</Text>
        : e.status === "stopped" ? <Text color={C.warn} bold> ■ Stopped</Text>
        : e.status === "deleted" ? <Text dimColor bold> Deleted</Text>
        : <Text color={C.warn} bold> ○ Not answering right now{cloud ? " (spunup is reconnecting)" : ""}</Text>}
      <Box flexDirection="column" marginY={1}>
        <Line k="Endpoint" v={<Text bold>{url}</Text>} />
        <Line k="API key" v={<Text>{showKey ? e.apiKey : "sk-spunup-••••••••"}   <Text dimColor>(press c to copy, k to show)</Text></Text>} />
        <Line k="Model" v={<Text>{e.name}  <Text dimColor>{e.quant} · {RUNTIME_NAMES[e.runtime]}{e.lora ? ` · LoRA "${e.lora.name}"` : ""}</Text></Text>} />
        <Line k="Speed" v={e.measured?.tps
          ? <Text><Tps max={e.measured.tps} />   <Prefill value={e.measured.prefill} /></Text>
          : e.estimate ? <Text><Tps min={e.estimate.tpsMin} max={e.estimate.tpsMax} />   <Prefill value={e.estimate.prefill} /><Text dimColor>  (estimate)</Text></Text> : <Text dimColor>—</Text>} />
        <Line k="Cost" v={cloud
          ? <Text>{e.status === "running" ? perHour(e.pricePerHour || 0) : e.status === "stopped" ? `${perHour(e.stoppedPricePerHour || 0)} while stopped` : e.diskKept ? `${perHour(e.stoppedPricePerHour || 0)} for the kept disk` : "no more charges"} · <Text bold>{money(shownCost)} {e.status === "deleted" && !e.diskKept ? "total" : "so far"}</Text>
            {(e.oneOffCost || 0) >= 0.005 && !billed?.checked ? <Text dimColor>  ({money(so - (e.oneOffCost || 0))} machine + {money(e.oneOffCost || 0)} download, as counted by {prov!.name})</Text> : null}
            {billed?.checked ? <Text dimColor>  (checked with {prov!.name} billing)</Text> : null}</Text>
          : <Text>free <Text dimColor>(runs on this computer)</Text></Text>} />
        {cloud && <Line k="Machine" v={<Text>{e.machine} on {prov!.name}{st ? <Text color={st.tunnelUp ? C.alive : C.warn}>   tunnel {st.tunnelUp ? "up" : "reconnecting"}</Text> : null}</Text>} />}
      </Box>
      {cloud && e.status !== "deleted" && <Text dimColor>   The endpoint only works on this computer (it goes through a private SSH tunnel from here).</Text>}
      {e.status === "stopped" && cloud && <Text color={C.warn}>   {prov!.stopNote}</Text>}
      {msg && <Text color={/not|could not|did not/i.test(msg) ? C.warn : C.alive}>   {msg}</Text>}
      {example && <Example url={url} model={e.name} />}
      <Box marginTop={1}>
        <Select key={`${deleteOpen}${timeOpen}${items.length}`} items={items}
          initial={timeOpen ? ((i) => (i >= 0 ? i : items.findIndex((x) => x.value === "t:custom")))(items.findIndex((x) => x.value === `t:${as.minutes}`)) : deleteOpen ? items.findIndex((x) => x.value === "delKeep") : 0}
          onBack={() => (timeOpen ? setTimeOpen(false) : deleteOpen ? setDeleteOpen(false) : ctx.go({ name: "start" }))}
          onKey={(input, i) => { if (items[i]?.value === "auto" && (input === "left" || input === "right")) setOnTime((x) => !x); }}
          onSelect={(v) => {
            if (v === "home") ctx.go({ name: "start" });
            else if (v === "stop") doIt(cloud ? "Stopping the machine…" : "Stopping…", () => stopEndpoint(e));
            else if (v === "start") doIt("Starting again…", () => restartEndpoint(e, ctx.cliPath, setBusy));
            else if (v === "delmenu") setDeleteOpen((x) => !x);
            else if (v === "del" || v === "delKeep") doIt("Deleting the machine…", () => deleteEndpoint(e, v === "delKeep"));
            else if (v === "deldisk") doIt("Deleting the disk…", () => deleteKeptDisk(e));
            else if (v === "reconnect") { ensureKeeper(ctx.cliPath); setMsg("Reconnecting the private tunnel…"); }
            else if (v === "auto") {
              if (onTime) setTimeOpen((x) => !x);
              else setAuto(!as.enabled ? { enabled: true, action: "stop" } : as.action === "stop" ? { action: "delete" } : { enabled: false, action: "stop" });
            } else if (v.startsWith("t:")) {
              if (v === "t:custom") setCustom("h");
              else { setAuto({ minutes: Number(v.slice(2)) }); setTimeOpen(false); }
            }
          }} />
      </Box>
      <Text dimColor>   ↑↓ move · Enter pick{cloud ? " · ←→ on the auto-stop line: switch on/off ↔ time" : ""} · x example · Esc back</Text>
    </Box>
  );
}

function Line({ k, v }: { k: string; v: React.ReactNode }) {
  return <Box><Box width={13}><Text dimColor>   {k}</Text></Box>{v}</Box>;
}

function Example({ url, model }: { url: string; model: string }) {
  return (
    <Box flexDirection="column" marginTop={1} borderStyle="round" borderColor="gray" paddingX={1}>
      <Text dimColor># curl</Text>
      <Text>curl {url}/chat/completions \</Text>
      <Text>  -H "Authorization: Bearer $SPUNUP_KEY" -H "Content-Type: application/json" \</Text>
      <Text>  -d '{`{"model": "${model}", "messages": [{"role": "user", "content": "Hello!"}]}`}'</Text>
      <Text dimColor># Python (pip install openai)</Text>
      <Text>from openai import OpenAI</Text>
      <Text>client = OpenAI(base_url="{url}", api_key="YOUR_SPUNUP_KEY")</Text>
      <Text>print(client.chat.completions.create(model="{model}", messages=[{`{"role": "user", "content": "Hello!"}`}]).choices[0].message.content)</Text>
    </Box>
  );
}
