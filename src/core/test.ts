/** Sends real requests to a new endpoint: checks it answers, accepts tools (so coding agents work),
 *  and measures generation and prefill speed on the actual machine.
 *
 *  How the speeds are worked out:
 *  - tokens/s: one long answer (ignore_eos, sized to ~4 s), counted from the first streamed token to
 *    the last. The wait before the first token is left out, so tunnel latency doesn't count.
 *  - prefill: a tiny prompt and a long prompt, each answered with 1 token. prefill = extra prompt
 *    tokens / extra time. The fixed cost of every request (tunnel round trip, scheduling, the one
 *    output token) is in both and cancels out. Each prompt starts with fresh random words, so prefix
 *    caching can't skip any of the work. Repeats take the fastest run: noise only ever adds time.
 *  - A warm-up request goes first, so first-request costs (graph capture, lazy loading) aren't measured. */
import crypto from "node:crypto";

export interface TestResult { ok: boolean; tps?: number; prefill?: number; toolsOk?: boolean; error?: string }

const WORDS = ("river map town note bridge field stone light water north south road hill tree house market " +
  "garden window paper letter story morning evening summer winter yellow green silver iron cloud rain").split(" ");

/** Plain words (~1 token each) that share no prefix with any earlier prompt. */
function freshText(words: number): string {
  const out: string[] = [crypto.randomUUID()];
  for (let i = 0; i < words; i++) out.push(WORDS[crypto.randomInt(WORDS.length)]);
  return out.join(" ");
}

class TestFailed extends Error {}

export async function testEndpoint(port: number, apiKey: string, model: string, context?: number, timeoutMs = 600_000): Promise<TestResult> {
  const base = `http://127.0.0.1:${port}/v1`;
  const auth = { authorization: `Bearer ${apiKey}`, "content-type": "application/json" };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);

  const post = (body: object) => fetch(`${base}/chat/completions`, { method: "POST", headers: auth, signal: ctrl.signal, body: JSON.stringify({ model, ...body }) });

  /** Streams an answer; returns when the first and last text arrived and the token counts. */
  async function streamed(content: string, maxTokens: number) {
    const t0 = performance.now();
    const res = await post({ messages: [{ role: "user", content }], max_tokens: maxTokens, temperature: 0.7, ignore_eos: true,
      stream: true, stream_options: { include_usage: true } });
    if (!res.ok || !res.body) throw new TestFailed(`Chat request failed (${res.status}): ${(await res.text()).slice(0, 200)}`);
    let first = 0, last = 0, chunks = 0, usage: any;
    const dec = new TextDecoder();
    let buf = "";
    const reader = res.body.getReader();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        try {
          const j = JSON.parse(data);
          if (j.usage) usage = j.usage;
          const d = j.choices?.[0]?.delta;
          if (d && (d.content || d.reasoning_content || d.reasoning)) {
            const now = performance.now();
            if (!first) first = now;
            last = now;
            chunks++;
          }
        } catch { /* ignore partial lines */ }
      }
    }
    if (!first) throw new TestFailed("The model returned no text.");
    return { t0, first, last, outTokens: (usage?.completion_tokens as number) || chunks };
  }

  /** tokens/s from one streamed answer, or undefined if it was too short to time. */
  const rate = (s: { first: number; last: number; outTokens: number }) =>
    s.last > s.first && s.outTokens > 1 ? (s.outTokens - 1) / ((s.last - s.first) / 1000) : undefined;

  /** Time for a fresh prompt of about `words` words, answered with 1 token (fastest of `runs`). */
  async function oneToken(words: number, runs: number) {
    let best = { ms: Infinity, tokens: 0 };
    for (let r = 0; r < runs; r++) {
      const t0 = performance.now();
      const res = await post({ messages: [{ role: "user", content: freshText(words) }], max_tokens: 1, temperature: 0 });
      if (!res.ok) throw new TestFailed(`Chat request failed (${res.status}): ${(await res.text()).slice(0, 200)}`);
      const j: any = await res.json();
      const ms = performance.now() - t0;
      if (ms < best.ms) best = { ms, tokens: j.usage?.prompt_tokens || Math.round(words * 1.3) };
    }
    return best;
  }

  try {
    const models = await fetch(`${base}/models`, { headers: auth, signal: ctrl.signal });
    if (!models.ok) return { ok: false, error: `The endpoint answered ${models.status} to /v1/models.` };
    const noKey = await fetch(`${base}/models`, { signal: ctrl.signal });
    if (noKey.ok) return { ok: false, error: "The endpoint answered without an API key. spunup will not hand out an open endpoint." };
    const listed: any = await models.json().catch(() => ({}));
    const ctx: number = listed?.data?.find?.((m: any) => m.id === model)?.max_model_len || context || 4096;

    // Warm-up, which also gives a rough speed to size the real run.
    const longAnswer = "Count upwards from 1, one number per line, and do not stop.";
    const rough = rate(await streamed(longAnswer, 32)) || 20;
    const outTokens = Math.max(64, Math.min(1024, Math.round(rough * 4), Math.floor(ctx / 2)));
    const tps = rate(await streamed(longAnswer, outTokens));

    // Prefill: tiny prompt vs long prompt. Words are ~1.3 tokens, and the long prompt keeps 20% of the context free.
    const maxWords = Math.max(64, Math.floor((ctx * 0.8) / 1.3));
    const small = await oneToken(16, 3);
    let words = Math.min(400, maxWords);
    let big = await oneToken(words, 1);
    const roughPrefill = (big.tokens - small.tokens) / Math.max(big.ms - small.ms, 1) * 1000;
    // Grow the long prompt until it takes ~2 s to read, so noise is small next to the difference.
    const target = Math.min(maxWords, 25_000, Math.round((roughPrefill * 2) / 1.3));
    if (target > words * 1.5) { words = target; big = await oneToken(words, 2); }
    else { const again = await oneToken(words, 1); if (again.ms < big.ms) big = again; }
    const prefill = big.ms > small.ms && big.tokens > small.tokens ? (big.tokens - small.tokens) / ((big.ms - small.ms) / 1000) : undefined;

    const tools = await post({ max_tokens: 64, messages: [{ role: "user", content: "What is the weather in Paris? Use the tool." }],
      tools: [{ type: "function", function: { name: "get_weather", description: "Get weather for a city", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } }] });
    const toolsOk = tools.ok;
    return { ok: true, tps: tps ? Math.round(tps * 10) / 10 : undefined, prefill: prefill ? Math.round(prefill) : undefined, toolsOk,
      error: toolsOk ? undefined : `Tool calling is not working (${tools.status}): ${(await tools.text()).slice(0, 160)}` };
  } catch (e) {
    if (e instanceof TestFailed) return { ok: false, error: e.message };
    return { ok: false, error: (e as Error).name === "AbortError" ? "The endpoint took too long to answer." : (e as Error).message };
  } finally {
    clearTimeout(timer);
  }
}

/** Waits until /v1/models answers 200 with the key. */
export async function waitHealthy(port: number, apiKey: string | undefined, isAlive: () => boolean | Promise<boolean>, timeoutMs: number, isCancelled: () => boolean): Promise<void> {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    if (isCancelled()) throw new Error("cancelled");
    if (!(await isAlive())) throw new Error("runtime-exited");
    try {
      const r = await fetch(`http://127.0.0.1:${port}/v1/models`, { headers: apiKey ? { authorization: `Bearer ${apiKey}` } : {}, signal: AbortSignal.timeout(5000) });
      if (r.ok) return;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 3000));
  }
  throw new Error("The runtime did not become ready in time.");
}
