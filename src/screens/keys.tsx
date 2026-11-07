import React, { useState } from "react";
import { Box, Text } from "ink";
import { Screen, Select, TextInput, Busy } from "../ui/components.js";
import { C } from "../ui/theme.js";
import { KEY_INFO, getKey, setKey, removeKey, checkKey, vastSendSms, vastSendEmail, vastTfaLogin, type KeyName } from "../core/keys.js";
import type { Ctx } from "../app.js";

/** Ask for one key, with exact instructions, and check it straight away (spec 10). */
export function KeyPrompt({ name, onDone, onBack, reason }: { name: KeyName; onDone: (ok: boolean) => void; onBack: () => void; reason?: string }) {
  const info = KEY_INFO[name];
  const [state, setState] = useState<{ checking?: boolean; error?: string; ok?: string; tfa?: string }>({});
  if (state.checking) return <Busy text="Checking the key…" />;
  if (state.tfa) {
    return <VastTfa apiKey={state.tfa} onBack={onBack} onDone={(msg) => setState({ ok: msg })} />;
  }
  if (state.ok) {
    return (
      <Screen title={info.title} footer="Enter continue">
        <Text color={C.alive}>✓ {state.ok}</Text>
        <Select items={[{ value: "ok", label: "Continue" }]} onSelect={() => onDone(true)} onBack={() => onDone(true)} />
      </Screen>
    );
  }
  return (
    <Screen title={info.title} note={reason || info.why} footer="Paste the key, then Enter · Esc back">
      <Box flexDirection="column">
        <Text> Get it here: <Text color={C.accent}>{info.link}</Text></Text>
        <Text> Use: {info.use}</Text>
        <Text> Not: <Text color={C.warn}>{info.not}</Text></Text>
        {state.error && <Box marginTop={1}><Text color={C.error}> {state.error}</Text></Box>}
        <Box marginTop={1}>
          <Text> Paste key: </Text>
          <TextInput mask onBack={onBack} validate={(v) => (v.trim() ? undefined : "Paste the key first.")}
            onSubmit={async (v) => {
              setState({ checking: true });
              const r = await checkKey(name, v);
              if (r.needsTfa) { setKey(name, v.trim()); return setState({ tfa: v.trim() }); }
              if (!r.ok) return setState({ error: r.message });
              const where = setKey(name, v.trim());
              setState({ ok: `${r.message} Key ${where}.` });
            }} />
        </Box>
      </Box>
    </Screen>
  );
}

export function KeysScreen({ ctx, onDone }: { ctx: Ctx; onDone: () => void }) {
  const [editing, setEditing] = useState<KeyName>();
  const [manage, setManage] = useState<KeyName>();
  const [tfa, setTfa] = useState<string>();
  const [checking, setChecking] = useState(false);
  const [done, setDone] = useState<string>();
  const [, refresh] = useState(0);
  if (checking) return <Busy text="Checking your Vast.ai key…" />;
  if (tfa) return <VastTfa apiKey={tfa} onBack={() => setTfa(undefined)} onDone={(m) => { setTfa(undefined); setDone(m); }} />;
  if (editing) return <KeyPrompt name={editing} onBack={() => setEditing(undefined)} onDone={() => { setEditing(undefined); refresh((x) => x + 1); }} />;
  if (manage) {
    return (
      <Screen title={KEY_INFO[manage].title} note="This key is saved.">
        <Select items={[{ value: "replace", label: "Replace it" }, { value: "remove", label: "Remove it", color: C.warn }, { value: "back", label: "Back" }]}
          onBack={() => setManage(undefined)}
          onSelect={(v) => {
            if (v === "replace") { setEditing(manage); setManage(undefined); }
            else if (v === "remove") { removeKey(manage); setManage(undefined); refresh((x) => x + 1); }
            else setManage(undefined);
          }} />
      </Screen>
    );
  }
  const names: KeyName[] = ["hf", "vast", "runpod"];
  return (
    <Screen title="Keys & tokens" note="Saved in your system keychain where possible. You only need the ones for what you use.">
      <Select labelWidth={26}
        items={[...names.map((n) => ({ value: n as string, label: KEY_INFO[n].title, right: getKey(n) ? <Text color={C.alive}>  saved</Text> : <Text dimColor>  not set</Text> })),
          { value: "back", label: "Back", separatorBefore: true }]}
        onBack={onDone}
        onSelect={async (v) => {
          if (v === "back") return onDone();
          const k = getKey(v as KeyName);
          if (!k) return setEditing(v as KeyName);
          if (v === "vast") {
            setChecking(true);
            const r = await checkKey("vast", k).catch(() => ({ ok: true, needsTfa: false, message: "" }));
            setChecking(false);
            if (r.needsTfa) return setTfa(k);
          }
          setManage(v as KeyName);
        }} />
      {done && <Text color={C.alive}>✓ {done}</Text>}
    </Screen>
  );
}

/** Vast.ai two-factor login: asks for a code once, then spunup uses the session Vast returns.
 *  Email works for every account, including ones that never set up two-factor login. */
export function VastTfa({ apiKey, onDone, onBack }: { apiKey: string; onDone: (msg: string) => void; onBack: () => void }) {
  const [kind, setKind] = useState<"email" | "totp" | "sms" | "backup">();
  const [phone, setPhone] = useState<string>();
  const [secret, setSecret] = useState<string>();
  const [busy, setBusy] = useState<string>();
  const [error, setError] = useState<string>();
  if (busy) return <Busy text={busy} />;
  if (!kind) {
    return (
      <Screen title="Vast.ai wants a quick login code" note="Vast.ai accepts your key but asks for a one-time code before spunup can use it. You only do this now and then.">
        {error && <Text color={C.error}>{error}</Text>}
        <Select items={[
          { value: "email", label: "Email me a code", hint: "works even if you never set up two-factor login" },
          { value: "totp", label: "Code from my authenticator app" },
          { value: "sms", label: "Code by text message" },
          { value: "backup", label: "A backup code" },
        ]} onBack={onBack} onSelect={async (v) => {
          setError(undefined);
          if (v === "email") {
            setBusy("Asking Vast.ai to email you a code…");
            try { setSecret(await vastSendEmail(apiKey)); setKind("email"); } catch (e) { setError((e as Error).message); }
            setBusy(undefined);
          } else setKind(v as any);
        }} />
      </Screen>
    );
  }
  if (kind === "sms" && !secret) {
    return (
      <Screen title="Vast.ai login code by text" note="Type the phone number on your Vast.ai account, with the country code (for example +44 7700 900123)." footer="Enter send · Esc back">
        {error && <Text color={C.error}>{error}</Text>}
        <TextInput initial={phone} onBack={() => setKind(undefined)} validate={(v) => (/^\+?[\d\s()-]{7,}$/.test(v.trim()) ? undefined : "Type the full number with the country code.")}
          onSubmit={async (v) => {
            const num = "+" + v.replace(/[^\d]/g, "");
            setPhone(v); setBusy("Asking Vast.ai to text you a code…");
            try { setSecret(await vastSendSms(apiKey, num)); setError(undefined); } catch (e) { setError((e as Error).message); }
            setBusy(undefined);
          }} />
      </Screen>
    );
  }
  const label = kind === "email" ? "Vast.ai has emailed you a code. Type it here (it lasts 10 minutes)"
    : kind === "totp" ? "Type the 6-digit code from your authenticator app"
    : kind === "sms" ? "Type the code Vast.ai just texted you" : "Type one of your backup codes";
  return (
    <Screen title="Vast.ai login code" note={label} footer="Enter · Esc back">
      {error && <Text color={C.error}>{error}</Text>}
      <TextInput onBack={() => { setKind(undefined); setSecret(undefined); }} validate={(v) => (v.trim() ? undefined : "Type the code first.")}
        onSubmit={async (v) => {
          setBusy("Checking the code…");
          try {
            await vastTfaLogin(apiKey, kind, v, secret);
            const r = await checkKey("vast", apiKey);
            setBusy(undefined);
            if (r.ok) onDone(`Logged in. ${r.message}`);
            else setError(r.message);
          } catch (e) { setBusy(undefined); setError((e as Error).message); }
        }} />
    </Screen>
  );
}
