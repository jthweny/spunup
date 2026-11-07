import { runSync } from "./proc.js";
import { P, IS_MAC, IS_WIN, IS_LINUX } from "./paths.js";
import { readJson, writeJson } from "./store.js";
import { http, HttpError } from "./net.js";

export type KeyName = "hf" | "vast" | "runpod";

/** What the user sees when spunup asks for a key. Names, pages and permissions
 *  were checked against each provider's docs on 2026-11-06 (see RESEARCH.md). */
export const KEY_INFO: Record<KeyName, { title: string; link: string; use: string; not: string; why: string }> = {
  hf: {
    title: "Hugging Face token",
    link: "https://huggingface.co/settings/tokens",
    use: 'An "Access Token" with the "Read" role (a fine-grained token also works if it can read gated repos)',
    not: "A Write token is not needed. First press \"Agree\" on the model's page, or the token cannot download it.",
    why: "Some models are gated (the maker wants you to accept their license first). The token lets spunup download them for you.",
  },
  vast: {
    title: "Vast.ai API key",
    link: "https://cloud.vast.ai/manage-keys/",
    use: 'The "API Keys" section on that page → "+New" (full access is the default)',
    not: 'Not an "SSH Key" from the same page. spunup makes its own SSH key.',
    why: "spunup uses it to find, rent, stop and delete machines for you.",
  },
  runpod: {
    title: "RunPod API key",
    link: "https://console.runpod.io/user/credentials",
    use: 'The "API Keys" tab → "Create API Key" with permission "All"',
    not: 'Not an "S3 API key" (starts with user_ or rps_), and not a "Read Only" key.',
    why: "spunup uses it to rent, stop and delete pods for you.",
  },
};

const SERVICE = "spunup";
// Tests set this so they never read or overwrite the user's real keys (the system keychain is
// shared by every spunup install, so SPUNUP_HOME alone does not keep a test away from it).
const NO_KEYCHAIN = !!process.env.SPUNUP_NO_KEYCHAIN;

function keychainSet(name: string, value: string): boolean {
  if (NO_KEYCHAIN) return false;
  if (IS_MAC) {
    const r = runSync("security", ["add-generic-password", "-U", "-a", SERVICE, "-s", `${SERVICE}:${name}`, "-w", value], { stdio: "ignore" });
    return r.status === 0;
  }
  if (IS_LINUX) {
    const r = runSync("secret-tool", ["store", "--label", `spunup ${name}`, "service", SERVICE, "key", name], { input: value, stdio: ["pipe", "ignore", "ignore"], timeout: 5000 });
    return r.status === 0;
  }
  if (IS_WIN) {
    // DPAPI: encrypted with the current Windows user's key, then kept in secrets.json.
    const r = runSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command",
      "$s = [Console]::In.ReadToEnd(); ConvertTo-SecureString $s -AsPlainText -Force | ConvertFrom-SecureString"],
      { input: value, encoding: "utf8", timeout: 15000 });
    if (r.status !== 0 || !r.stdout.trim()) return false;
    const s = readJson<Record<string, string>>(P.secrets, {});
    s[`dpapi:${name}`] = r.stdout.trim();
    delete s[name];
    writeJson(P.secrets, s);
    return true;
  }
  return false;
}

function keychainGet(name: string): string | undefined {
  if (NO_KEYCHAIN) return undefined;
  if (IS_MAC) {
    const r = runSync("security", ["find-generic-password", "-a", SERVICE, "-s", `${SERVICE}:${name}`, "-w"], { encoding: "utf8" });
    return r.status === 0 ? r.stdout.trim() || undefined : undefined;
  }
  if (IS_LINUX) {
    const r = runSync("secret-tool", ["lookup", "service", SERVICE, "key", name], { encoding: "utf8", timeout: 5000 });
    return r.status === 0 ? r.stdout.trim() || undefined : undefined;
  }
  if (IS_WIN) {
    const s = readJson<Record<string, string>>(P.secrets, {});
    const enc = s[`dpapi:${name}`];
    if (!enc) return undefined;
    const r = runSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command",
      "$e = [Console]::In.ReadToEnd().Trim(); $ss = ConvertTo-SecureString $e; [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($ss))"],
      { input: enc, encoding: "utf8", timeout: 15000 });
    return r.status === 0 ? r.stdout.trim() || undefined : undefined;
  }
  return undefined;
}

function keychainDelete(name: string): void {
  if (NO_KEYCHAIN) return;
  if (IS_MAC) runSync("security", ["delete-generic-password", "-a", SERVICE, "-s", `${SERVICE}:${name}`], { stdio: "ignore" });
  if (IS_LINUX) runSync("secret-tool", ["clear", "service", SERVICE, "key", name], { stdio: "ignore", timeout: 5000 });
}

const cache = new Map<string, string | undefined>();

export function getKey(name: KeyName): string | undefined {
  const env = { hf: process.env.HF_TOKEN, vast: process.env.VAST_API_KEY, runpod: process.env.RUNPOD_API_KEY }[name];
  if (env) return env;
  if (cache.has(name)) return cache.get(name);
  let v = keychainGet(name);
  if (!v) v = readJson<Record<string, string>>(P.secrets, {})[name];
  cache.set(name, v);
  return v;
}

/** Saves a key. Returns where it was stored, in plain words. */
export function setKey(name: KeyName, value: string): string {
  cache.set(name, value);
  if (keychainSet(name, value)) {
    const s = readJson<Record<string, string>>(P.secrets, {});
    if (s[name]) { delete s[name]; writeJson(P.secrets, s); }
    return IS_WIN ? "saved, locked to your Windows account" : "saved in your system keychain";
  }
  const s = readJson<Record<string, string>>(P.secrets, {});
  s[name] = value;
  writeJson(P.secrets, s, 0o600);
  return "saved in ~/.spunup (only your user can read it)";
}

export function removeKey(name: KeyName): void {
  cache.delete(name);
  keychainDelete(name);
  const s = readJson<Record<string, string>>(P.secrets, {});
  delete s[name];
  delete s[`dpapi:${name}`];
  writeJson(P.secrets, s);
}

export interface KeyCheck { ok: boolean; message: string; needsTfa?: boolean }

/** Checks a key straight away with one small API call. */
export async function checkKey(name: KeyName, value: string): Promise<KeyCheck> {
  value = value.trim();
  try {
    if (name === "hf") {
      if (!value.startsWith("hf_")) return { ok: false, message: 'That does not look like a Hugging Face token. Tokens start with "hf_".' };
      const r = await http<any>("https://huggingface.co/api/whoami-v2", { headers: { authorization: `Bearer ${value}` }, timeoutMs: 15000 });
      const role = r?.auth?.accessToken?.role;
      return { ok: true, message: `Token works (account: ${r.name}${role ? `, role: ${role}` : ""}).` };
    }
    if (name === "vast") {
      const session = getVastSession();
      let r: any;
      try {
        const auth = { authorization: `Bearer ${session || value}` };
        r = await http<any>("https://console.vast.ai/api/v0/users/current/", { headers: auth, timeoutMs: 15000 });
        // Reading the account can work while machine actions still need a two-factor login,
        // so also check a call spunup really uses (listing machines; free, changes nothing).
        await http<any>("https://console.vast.ai/api/v1/instances/?limit=1", { headers: auth, timeoutMs: 15000 });
      } catch (e) {
        if (e instanceof HttpError && /two factor/i.test(e.body)) {
          if (session) setVastSession(undefined); // expired session: log in again
          return { ok: false, needsTfa: true, message: "Your Vast.ai account uses two-factor login, so Vast asks for a code before this key can be used." };
        }
        throw e;
      }
      return { ok: true, message: `Key works (account: ${r.username || r.email || r.id}, credit: $${Number(r.credit ?? 0).toFixed(2)}).` };
    }
    if (value.startsWith("user_") || value.startsWith("rps_")) {
      return { ok: false, message: 'That is an "S3 API key". spunup needs a normal RunPod "API key" from the API Keys tab.' };
    }
    await http<any>("https://api.runpod.io/v2/pods", { headers: { authorization: `Bearer ${value}` }, timeoutMs: 15000 });
    return { ok: true, message: "Key works." };
  } catch (e) {
    if (e instanceof HttpError && (e.status === 401 || e.status === 403)) {
      const extra = name === "runpod" && e.status === 403 ? " It may be a Read Only key; spunup needs permission \"All\"." : "";
      return { ok: false, message: `The provider says this key is not valid (${e.status}).${extra}` };
    }
    return { ok: false, message: `Could not check the key: ${(e as Error).message}. Check your internet connection.` };
  }
}

// ------------------------------------------------------------------ Vast.ai two-factor login
// Accounts with 2FA reject a plain API key until a 2FA login is done; the login returns a
// session key that is then used as the bearer token (docs.vast.ai two-factor-authentication-endpoints).
const VAST_SESSION = "vast_session";

export function getVastSession(): string | undefined {
  if (cache.has(VAST_SESSION)) return cache.get(VAST_SESSION);
  const v = keychainGet(VAST_SESSION) || readJson<Record<string, string>>(P.secrets, {})[VAST_SESSION];
  cache.set(VAST_SESSION, v);
  return v;
}

export function setVastSession(v: string | undefined): void {
  cache.set(VAST_SESSION, v);
  if (!v) {
    keychainDelete(VAST_SESSION);
    const s = readJson<Record<string, string>>(P.secrets, {});
    delete s[VAST_SESSION]; delete s[`dpapi:${VAST_SESSION}`];
    writeJson(P.secrets, s);
    return;
  }
  if (!keychainSet(VAST_SESSION, v)) {
    const s = readJson<Record<string, string>>(P.secrets, {});
    s[VAST_SESSION] = v;
    writeJson(P.secrets, s, 0o600);
  }
}

const TFA_MESSAGES: Record<string, string> = {
  "2fa_verification_failed": "That code is not right. Check the code and try again.",
  "2fa_expired": "That code has expired. Ask for a new one.",
  invalid_backup_code: "That backup code is not right.",
  tfa_locked: "Too many wrong codes. Vast has locked this method for 15 minutes.",
  tfa_method_not_configured: "Your account has no authenticator app set up. Use a text message or a backup code.",
  invalid_tfa_method: "Your account does not use that kind of code. Try another kind.",
  team_user_forbidden: "Two-factor login must be done from your personal Vast account, not a team.",
};

function tfaError(e: unknown): Error {
  if (e instanceof HttpError) {
    try {
      const j = JSON.parse(e.body);
      return new Error(TFA_MESSAGES[j.error] || j.msg || `Vast said ${e.status}.`);
    } catch { return new Error(`Vast said ${e.status}.`); }
  }
  return e as Error;
}

/** Email a login code. Works for accounts with no 2FA method set up (vast-cli: `vastai tfa send-email`).
 *  Returns the challenge secret needed with the code. */
export async function vastSendEmail(apiKey: string): Promise<string> {
  try {
    const r = await http<any>("https://console.vast.ai/api/v0/tfa/email/", { method: "POST", body: {}, headers: { authorization: `Bearer ${apiKey}` }, timeoutMs: 20000 });
    return r.secret;
  } catch (e) { throw tfaError(e); }
}

/** Send a text-message code. Returns the challenge secret needed with the code. */
export async function vastSendSms(apiKey: string, phone: string): Promise<string> {
  try {
    const r = await http<any>("https://console.vast.ai/api/v0/tfa/sms/", { method: "POST", body: { phone_number: phone }, headers: { authorization: `Bearer ${apiKey}` }, timeoutMs: 20000 });
    return r.secret;
  } catch (e) { throw tfaError(e); }
}

/** Finish the 2FA login and save the session key. kind: "email", "totp" (authenticator app), "sms", "backup". */
export async function vastTfaLogin(apiKey: string, kind: "email" | "totp" | "sms" | "backup", code: string, secret?: string): Promise<void> {
  const body = kind === "backup" ? { backup_code: code.trim() }
    : kind === "totp" ? { tfa_method: "totp", code: code.trim() }
    : { tfa_method: kind, code: code.trim(), secret };
  try {
    const r = await http<any>("https://console.vast.ai/api/v0/tfa/", { method: "POST", body, headers: { authorization: `Bearer ${apiKey}` }, timeoutMs: 20000 });
    if (!r.session_key) throw new Error("Vast accepted the code but did not return a session. Try again.");
    setVastSession(r.session_key);
  } catch (e) { throw tfaError(e); }
}
