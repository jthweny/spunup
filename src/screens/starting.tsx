import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import { Select, Spinner, Busy, type Item } from "../ui/components.js";
import { C } from "../ui/theme.js";
import { getJob, dropJob, stopEndpoint, deleteEndpoint } from "../core/launch.js";
import { getEndpoint } from "../core/store.js";
import { provider } from "../providers/index.js";
import { sleep } from "../core/net.js";
import { RUNTIME_NAMES } from "../runtimes/index.js";
import type { Ctx } from "../app.js";

/** Progress of one setup. The setup itself runs in the background (see startJob), so leaving
 *  this screen does not stop it; it can be opened again from History. */
export function StartingScreen({ ctx, jobId, onBack, onPickMachine }: { ctx: Ctx; jobId: string; onBack?: () => void; onPickMachine?: () => void }) {
  const [job] = useState(() => getJob(jobId));   // kept after Cancel drops it from the list
  const [, render] = useState(0);
  const [menuOpen, setMenuOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [working, setWorking] = useState<string>();
  const [done, setDone] = useState<string>();
  const l = job?.launcher;
  const cloud = !!l && l.plan.where !== "local";
  const prov = cloud ? provider(l!.plan.where as "vast" | "runpod") : undefined;

  useEffect(() => {
    if (!job) { ctx.go({ name: "start" }); return; }
    const onUpdate = () => {
      render((x) => x + 1);
      if (job.finished && job.result && !job.launcher.cancelled) ctx.go({ name: "endpoint", id: job.result.id });
    };
    onUpdate();
    job.launcher.on("update", onUpdate);
    return () => { job.launcher.off("update", onUpdate); };
  }, []);
  if (!job || !l) return <Busy text="Getting ready…" />;
  const home = () => { if (l.error) dropJob(jobId); ctx.go({ name: "start" }); };
  if (working) return <Busy text={working} />;

  /** Stop or delete the machine (waits for its id if renting is still in flight). */
  const act = async (what: "stop" | "delete" | "deleteKeepDisk" | "cancelLocal") => {
    l.cancel();
    dropJob(jobId);
    if (what === "cancelLocal") { setDone("Stopped. Nothing is running."); return; }
    setWorking(what === "stop" ? "Stopping the machine…" : "Deleting the machine…");
    let e = l.endpoint && getEndpoint(l.endpoint.id);
    for (let i = 0; i < 30 && !e?.instanceId; i++) { await sleep(2000); e = l.endpoint && getEndpoint(l.endpoint.id); }
    try {
      if (!e?.instanceId) setDone("Nothing was rented, so nothing is charged.");
      else if (what === "stop") { await stopEndpoint(e); setDone(`Machine stopped. ${prov!.stopNote}`); }
      else { await deleteEndpoint(e, what === "deleteKeepDisk"); setDone(what === "deleteKeepDisk" ? "Machine deleted. Its disk is kept and still charged until you delete it from History." : "Machine deleted. No more charges."); }
    } catch (err) {
      setDone(`That did not work: ${(err as Error).message}. Open History to try again, or use the ${prov!.name} website.`);
    }
    setWorking(undefined);
  };

  if (done) {
    return (
      <Box flexDirection="column" paddingX={1}>
        <Text>{done}</Text>
        <Select items={[{ value: "h", label: "Back to History" }]} onSelect={() => ctx.go({ name: "start" })} onBack={() => ctx.go({ name: "start" })} />
      </Box>
    );
  }

  const steps = (
    <Box flexDirection="column" marginBottom={1}>
      {l.steps.map((s) => (
        <Box key={s.key}>
          {s.state === "done" ? <Text color={C.alive}> ✓ </Text> : s.state === "doing" ? <Text> <Spinner /> </Text> : s.state === "failed" ? <Text color={C.error}> ✗ </Text> : <Text>   </Text>}
          <Text color={s.state === "todo" ? "gray" : undefined} bold={s.state === "doing"}>{s.label}</Text>
          {s.detail && <Text dimColor>  {s.detail}</Text>}
        </Box>
      ))}
    </Box>
  );

  if (l.error) {
    const items: Item<string>[] = [];
    if (l.error.pickAnother && onPickMachine) items.push({ value: "pick", label: "Pick another machine" });
    if (l.error.machineAlive) {
      items.push({ value: "delete", label: "Delete the machine", hint: "(no more charges)" });
      items.push({ value: "keep", label: "Keep it for now", hint: "it is still charged; manage it from History" });
    }
    if (onBack) items.push({ value: "back", label: "Back to change settings" });
    items.push({ value: "home", label: "Back to History" });
    return (
      <Box flexDirection="column" paddingX={1}>
        {steps}
        <Text color={C.error}>Something went wrong: {l.error.message}</Text>
        {l.error.log && l.error.log.length > 0 && (
          <Box flexDirection="column" marginY={1} paddingLeft={2} borderStyle="single" borderLeft borderTop={false} borderRight={false} borderBottom={false} borderColor="gray">
            <Text dimColor>What {RUNTIME_NAMES[l.plan.runtime]} printed:</Text>
            {l.error.log.map((line, i) => <Text key={i} dimColor wrap="truncate-end">{line}</Text>)}
            {l.error.logPath && <Text dimColor>Full log: {l.error.logPath}</Text>}
          </Box>
        )}
        {l.error.next.map((n, i) => <Text key={i}>· {n}</Text>)}
        <Box marginTop={1}><Select items={items} onBack={() => { dropJob(jobId); (onBack || home)(); }} onSelect={(v) => {
          if (v === "delete") act("delete");
          else if (v === "pick") { dropJob(jobId); onPickMachine!(); }
          else if (v === "keep" || v === "home") home();
          else { dropJob(jobId); onBack!(); }
        }} /></Box>
      </Box>
    );
  }

  // Cancel button: plain for Local; a drop-down for cloud (spec 11.8).
  const separate = !!prov?.separateDisk && !!l.endpoint?.volumeId;
  const items: Item<string>[] = !cloud
    ? [{ value: "cancelLocal", label: "Cancel" }]
    : [{ value: "menu", label: `Cancel ${menuOpen ? "▴" : "▾"}` }];
  if (cloud && menuOpen) {
    items.push({ value: "stop", label: "Stop machine", hint: "(kept for you, lower hourly price)", indent: 4 });
    if (separate) {
      items.push({ value: "delmenu", label: `Delete machine ${deleteOpen ? "▴" : "▾"}`, indent: 4 });
      if (deleteOpen) {
        items.push({ value: "deleteKeepDisk", label: "Machine only", hint: "(disk kept, still charged)", indent: 8 });
        items.push({ value: "delete", label: "Machine + disk", hint: "(no more charges)", indent: 8 });
      }
    } else items.push({ value: "delete", label: "Delete machine", hint: "(no more charges)", indent: 4 });
  }
  // Leave this setup running: from the main page people can serve another model, or open this one again.
  items.push({ value: "another", label: "Back to main page", hint: "(this keeps starting; serve another model meanwhile, or open it again from History)", separatorBefore: true });
  return (
    <Box flexDirection="column" paddingX={1}>
      {steps}
      <Select key={`${menuOpen}${deleteOpen}`} items={items} initial={deleteOpen ? 2 : menuOpen ? 1 : 0}
        onBack={() => (deleteOpen ? setDeleteOpen(false) : menuOpen ? setMenuOpen(false) : ctx.go({ name: "start" }))}
        onSelect={(v) => {
          if (v === "menu") setMenuOpen((x) => !x);
          else if (v === "delmenu") setDeleteOpen((x) => !x);
          else if (v === "another") ctx.go({ name: "start" });
          else act(v as any);
        }} />
      {cloud && <Box marginTop={1}><Text dimColor>The endpoint will be on this computer only, through a private SSH tunnel. Nothing is public.</Text></Box>}
      <Text dimColor>Keep spunup open until it is ready. Closing spunup stops the setup.</Text>
    </Box>
  );
}
