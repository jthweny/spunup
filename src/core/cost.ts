import { loadConfig, saveConfig, type Endpoint } from "./store.js";
import { http } from "./net.js";
import { costSoFar } from "./keeper.js";
import { provider } from "../providers/index.js";

/** Providers bill in US dollars. spunup shows the user's own currency (ECB rate via frankfurter.dev),
 *  converted once a day, and falls back to dollars when offline. */
const SYMBOL: Record<string, string> = { USD: "$", GBP: "£", EUR: "€" };

function localeCurrency(): "USD" | "GBP" | "EUR" {
  const loc = (Intl.DateTimeFormat().resolvedOptions().locale || "").toUpperCase();
  const region = loc.split("-").pop() || "";
  if (region === "GB" || region === "UK") return "GBP";
  if (["DE", "FR", "ES", "IT", "NL", "IE", "PT", "AT", "BE", "FI", "GR", "LU", "SK", "SI", "EE", "LV", "LT", "CY", "MT", "HR"].includes(region)) return "EUR";
  return "USD";
}

let rate = 1;
let cur: "USD" | "GBP" | "EUR" = "USD";

export async function initCurrency(): Promise<void> {
  const cfg = loadConfig();
  cur = cfg.currency && cfg.currency !== "USD" ? cfg.currency : localeCurrency();
  if (cur === "USD") { rate = 1; return; }
  if (cfg.fx && Date.now() - cfg.fx.at < 24 * 3600_000) { rate = cfg.fx.rate; return; }
  try {
    const r = await http<any>(`https://api.frankfurter.dev/v1/latest?base=USD&symbols=${cur}`, { timeoutMs: 5000 });
    rate = Number(r.rates?.[cur]) || 1;
    cfg.fx = { rate, at: Date.now() };
    cfg.currency = cur;
    saveConfig(cfg);
  } catch {
    if (cfg.fx) rate = cfg.fx.rate; else { cur = "USD"; rate = 1; }
  }
}

export function money(usd: number, digits = 2): string {
  const v = usd * rate;
  return `${SYMBOL[cur]}${v.toFixed(v > 0 && v < 0.01 ? 3 : digits)}`;
}

export const perHour = (usd: number) => `${money(usd)}/hr`;

/** Cost so far, checked against the provider's billing API where it has one. */
export async function checkedCost(e: Endpoint): Promise<{ usd: number; checked: boolean }> {
  const est = costSoFar(e);
  if (e.where === "local" || !e.instanceId) return { usd: 0, checked: false };
  const p = provider(e.where);
  if (!p.billed) return { usd: est, checked: false };
  try {
    const b = await p.billed(e.instanceId, e.createdAt);
    // Billing records lag up to an hour; trust them only when they are not behind the estimate by much.
    if (b !== undefined && b >= est * 0.8) return { usd: b, checked: true };
  } catch { /* fall back to estimate */ }
  return { usd: est, checked: false };
}
