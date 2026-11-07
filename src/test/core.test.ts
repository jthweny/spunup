import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { kvFromConfig } from "../core/kv.js";
import { buildLaunch, mergeArgs } from "../runtimes/index.js";
import { computeFit, estimateSpeed, calibrate, localDevices } from "../core/fit.js";
import { loadIndex, loadModelRecipe, parseHfId, type Recipe } from "../core/recipes.js";
import { PKG_ROOT } from "../core/paths.js";
import type { Hardware } from "../core/hardware.js";

test("KV: standard GQA (Llama 3.3 70B config) is 2 × layers × kv_heads × head_dim × 2 bytes", () => {
  const kv = kvFromConfig({ num_hidden_layers: 80, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 128, hidden_size: 8192 })!;
  assert.equal(kv.mbPer1k, Math.round((2 * 80 * 8 * 128 * 2 * 1024) / 1e6 * 100) / 100); // 335.54
});

test("KV: hybrid linear attention counts only full-attention layers", () => {
  const lt = Array.from({ length: 64 }, (_, i) => (i % 4 === 3 ? "full_attention" : "linear_attention"));
  const kv = kvFromConfig({ text_config: { num_hidden_layers: 64, num_attention_heads: 24, num_key_value_heads: 4, head_dim: 256, layer_types: lt } })!;
  assert.equal(kv.mbPer1k, Math.round((16 * 2 * 4 * 256 * 2 * 1024) / 1e6 * 100) / 100);
});

test("KV: sliding-window layers are a fixed cost, not per token", () => {
  const kv = kvFromConfig({ num_hidden_layers: 4, num_attention_heads: 8, num_key_value_heads: 2, head_dim: 64, sliding_window: 128, layer_types: ["sliding_attention", "full_attention", "sliding_attention", "full_attention"] })!;
  assert.equal(kv.mbPer1k, Math.round((2 * 2 * 2 * 64 * 2 * 1024) / 1e6 * 100) / 100);
  assert.ok(kv.fixedMb > 0);
});

test("KV: MLA uses the latent size", () => {
  const kv = kvFromConfig({ num_hidden_layers: 61, num_attention_heads: 128, kv_lora_rank: 512, qk_rope_head_dim: 64 })!;
  assert.equal(kv.mbPer1k, Math.round((61 * 576 * 2 * 1024) / 1e6 * 100) / 100);
});

test("mergeArgs: a user flag replaces the recipe flag with the same name", () => {
  assert.deepEqual(mergeArgs(["--a", "1", "--b", "2"], ["--a", "9"]), ["--b", "2", "--a", "9"]);
  assert.deepEqual(mergeArgs(["--x=1", "--y"], ["--x=2"]), ["--y", "--x=2"]);
});

test("parseHfId accepts links and names", () => {
  assert.equal(parseHfId("https://huggingface.co/Qwen/Qwen3.8-27B/tree/main"), "Qwen/Qwen3.8-27B");
  assert.equal(parseHfId("Qwen/Qwen3.8-27B"), "Qwen/Qwen3.8-27B");
  assert.equal(parseHfId("not a model"), undefined);
});

const hwMac8: Hardware = { os: "macos", arch: "arm64", cpu: { model: "Apple M2", cores: 8, avx2: false, avx512: false }, ramGb: 8, diskFreeGb: 100,
  gpus: [{ vendor: "apple", name: "Apple M2", vramGb: 5 }], apple: { chip: "Apple M2", unifiedGb: 8 }, notes: [] };
const hw4090: Hardware = { os: "linux", arch: "x64", cpu: { model: "AMD Ryzen 9", cores: 16, avx2: true, avx512: false }, ramGb: 64, diskFreeGb: 500,
  gpus: [{ vendor: "nvidia", name: "NVIDIA GeForce RTX 4090", vramGb: 24, cuda: "13.4", computeCap: "8.9" }], notes: [] };
const hwCpu: Hardware = { ...hw4090, gpus: [] };

test("fit: small model fits fully on GPU; big MoE spills to RAM (mixed); no GPU = CPU", () => {
  const small = loadModelRecipe("Qwen/Qwen3.5-9B")!;
  const q4 = small.quants.find((q) => q.format === "gguf" && q.label === "Q4")!;
  assert.equal(computeFit(small, q4, 32768, hw4090, "llamacpp").setup, "gpu");
  const big = loadModelRecipe("Qwen/Qwen3.5-122B-A10B")!;
  const bq = big.quants.find((q) => q.format === "gguf" && q.label === "Q4")!;
  assert.equal(computeFit(big, bq, 32768, hw4090, "llamacpp").setup, "none"); // 77 GB > 24 + 54
  const bq2 = big.quants.find((q) => q.format === "gguf" && q.label === "Q2")!;
  assert.equal(computeFit(big, bq2, 32768, hw4090, "llamacpp").setup, "mixed");
  assert.equal(computeFit(small, q4, 8192, hwCpu, "llamacpp").setup, "cpu");
  assert.equal(computeFit(small, q4, 8192, hwMac8, "llamacpp").ok, false);
});

test("speed: calibrated estimate matches the M2 measurement (Qwen3.5-2B Q4: 41.6 tokens/s)", () => {
  const r = loadModelRecipe("Qwen/Qwen3.5-2B")!;
  const q = r.quants.find((x) => x.format === "gguf" && x.label === "Q4")!;
  const s = estimateSpeed(r, q, { gpu: calibrate(100, 21.91, 179.57, {}, true), gpus: 1, cpuBwGb: 50, pcieGbps: 0 }, 1, 0.2);
  assert.ok(s.tpsMin <= 41.6 * 1.15 && s.tpsMax >= 41.6 * 0.85, `estimate ${s.tpsMin}-${s.tpsMax}`);
});

// ------------------------------------------------------------------ shipped recipes
const idx = loadIndex();
const recipes = idx.models.map((m) => loadModelRecipe(m.id)!) as Recipe[];

test("every listed model has a recipe file with sources", () => {
  assert.ok(recipes.length >= 90);
  for (const r of recipes) {
    assert.ok(r, "recipe missing");
    assert.ok(r.sources.length > 0, `${r.id}: no sources`);
    assert.ok(r.family, `${r.id}: no family`);
    assert.ok(fs.existsSync(path.join(PKG_ROOT, "recipes", "families", `${r.family}.json`)), `${r.id}: family file missing`);
    assert.ok(r.quants.length > 0, `${r.id}: no quantizations`);
    assert.ok(r.max_context && r.max_context >= 2048, `${r.id}: max context`);
    assert.ok(r.setups.gpu.best, `${r.id}: no GPU runtime`);
  }
});

test("every start command binds 127.0.0.1, requires an API key, and turns on tool calling", () => {
  for (const r of recipes) {
    for (const rt of ["vllm", "sglang", "llamacpp"] as const) {
      const rr = r.runtimes[rt];
      if (!rr.supported) continue;
      const q = r.quants.find((x) => x.runtimes.includes(rt)) || r.quants[0];
      const l = buildLaunch(rt, r, q, { modelPath: "/m", servedName: "m", host: "127.0.0.1", port: 8000, apiKey: "sk-test", context: 8192, gpus: 1, setup: "gpu" });
      const a = l.args.join(" ");
      assert.match(a, /--host 127\.0\.0\.1/, `${r.id} ${rt}`);
      assert.match(a, /--api-key sk-test/, `${r.id} ${rt}`);
      assert.doesNotMatch(a, /0\.0\.0\.0/, `${r.id} ${rt}`);
      if (rt === "vllm") assert.match(a, /--enable-auto-tool-choice --tool-call-parser \S+/, `${r.id} vllm tool parser`);
      if (rt === "sglang") assert.match(a, /--tool-call-parser \S+/, `${r.id} sglang tool parser`);
      if (rt === "llamacpp") assert.match(a, /--jinja/, `${r.id} llama.cpp needs --jinja for tool calls`);
    }
  }
});

test("speed: a small model that fits on the GPU gets prefill from GPU compute only (no 6× jumps between similar GPUs)", async () => {
  const { offerDevice } = await import("../core/plan.js");
  const r = loadModelRecipe("meta-llama/Llama-3.2-1B-Instruct")!;
  const q = r.quants.find((x) => x.label === "BF16")!;
  const mk = (gpuName: string, bw: number, fp32: number) => estimateSpeed(r, q, offerDevice({ provider: "vast", id: "x", gpuName, numGpus: 1, vramPerGpuGb: 8, ramGb: 16, diskGb: 50, pricePerHour: 0.1, gpuMemBw: bw, tflops: fp32 }), 1, 0.25);
  const g3060 = mk("RTX 3060", 319, 12.2), g4060 = mk("RTX 4060", 240, 15.3), g3070 = mk("RTX 3070", 389, 19.8);
  for (const s of [g4060, g3070]) assert.ok(s.prefill > g3060.prefill * 0.6 && s.prefill < g3060.prefill * 2.5, `prefill ${s.prefill} vs 3060 ${g3060.prefill}`);
  // generation follows memory bandwidth: 3070 (389 GB/s) > 3060 (319) > 4060 (240)
  assert.ok(g3070.tpsMax > g3060.tpsMax && g3060.tpsMax > g4060.tpsMax);
});

test("errors: the runtime's last lines are kept from the traceback on, without progress bars", async () => {
  const { logTail } = await import("../core/launch.js");
  const log = ["INFO starting", "Loading safetensors 40%|███", "Traceback (most recent call last):", '  File "x.py", line 3', "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", ""].join("\n");
  const out = logTail(log);
  assert.equal(out[0], "Traceback (most recent call last):");
  assert.ok(out.some((l) => /out of memory/.test(l)));
  assert.ok(!out.some((l) => /40%/.test(l)));
});

test("LoRA: vLLM rank rounds up to an allowed value; small adapters barely slow generation", async () => {
  const { vllmRank, adapterSpeedFactor } = await import("../core/lora.js");
  assert.equal(vllmRank(16), 16);
  assert.equal(vllmRank(24), 32);
  assert.equal(vllmRank(300), 320);
  assert.equal(vllmRank(undefined), 64);
  const f = adapterSpeedFactor(16, 0.17);  // 8B BF16 + a 170 MB rank-16 adapter
  assert.ok(f > 0.98 && f < 1);
});

test("labels: MLX precision comes from the repo name, then from bits per weight", async () => {
  const { mlxLabel } = await import("../core/recipes.js");
  assert.equal(mlxLabel("mlx-community/gpt-oss-20b-MXFP4-Q8", 12, 21), "MLX 4-bit");
  assert.equal(mlxLabel("ibm-granite/granite-4.2-8b-q4-mlx", 4.9, 8.7), "MLX 4-bit");
  assert.equal(mlxLabel("mlx-community/Qwen3.5-27B-bf16", 54.7, 27.8), "MLX 16-bit");
  assert.equal(mlxLabel("mlx-community/GLM-5.1", 1487.8, 754), "MLX 16-bit");
});

test("RunPod: disk is set by spunup, so machines are not dropped for disk and the price follows the size used", async () => {
  const { cloudOptions } = await import("../core/plan.js");
  const r = loadModelRecipe("meta-llama/Llama-3.2-1B-Instruct")!;
  const offer = { provider: "runpod" as const, id: "x", gpuName: "L4", numGpus: 1, vramPerGpuGb: 24, ramGb: 50, diskGb: 21, pricePerHour: 0.5, storagePerGbMonth: 0.1 };
  const o = cloudOptions(r, "vllm", 8192, [offer]);
  assert.equal(o.length, 1);
  assert.ok(o[0].offer.diskGb > 21 && o[0].offer.pricePerHour > 0.5);
});

test("a machine taken before it could be rented: nothing charged, offer to pick another", async () => {
  const { Launcher } = await import("../core/launch.js");
  const { NoCapacity } = await import("../providers/index.js");
  const r = loadModelRecipe("meta-llama/Llama-3.2-1B-Instruct")!;
  const l = new Launcher({ recipe: r, quant: r.quants[0], where: "runpod", runtime: "vllm", setup: "gpu", context: 8192 });
  (l as any).runCloud = async () => { throw new NoCapacity("RunPod has no free RTX A5000 right now."); };
  assert.equal(await l.run("cli.js"), undefined);
  assert.equal(l.error?.pickAnother, true);
  assert.equal(l.error?.machineAlive, false);
  assert.match(l.error!.next[0], /nothing is charged/);
});

test("speed: CPU estimate matches the Linux measurement (Qwen3.5-0.8B Q8, 4-core VM at 29 GB/s: 17.8 tokens/s, prefill 146)", () => {
  const r = loadModelRecipe("Qwen/Qwen3.5-0.8B")!;
  const q = r.quants.find((x) => x.format === "gguf" && x.label === "Q8")!;
  const cpuVm: Hardware = { ...hwCpu, cpu: { model: "Intel Xeon Processor (Cascadelake)", cores: 4, avx2: true, avx512: true, bwGb: 29 } };
  const s = estimateSpeed(r, q, localDevices(cpuVm), 0, 0.12);
  assert.ok(s.tpsMin <= 17.8 * 1.15 && s.tpsMax >= 17.8 * 0.85, `estimate ${s.tpsMin}-${s.tpsMax}`);
  assert.ok(s.prefill > 146 * 0.7 && s.prefill < 146 * 1.4, `prefill ${s.prefill}`);
});

test("recommended: the machine and size with the most quality for the price, not just the cheapest machine", async () => {
  const { cloudOptions, recommendedOption } = await import("../core/plan.js");
  const r = loadModelRecipe("Qwen/Qwen3-30B-A3B-Instruct-2507")!;
  const vast = (id: string, vram: number, price: number) => ({ provider: "vast" as const, id, gpuName: "RTX 3090", numGpus: 1, vramPerGpuGb: vram, ramGb: 64, diskGb: 200, pricePerHour: price, computeCap: 8.6 });
  const offers = [vast("16gb", 16, 0.08), vast("24gb", 24, 0.15), vast("48gb", 48, 0.4), vast("80gb", 80, 1.0)];
  const opts = cloudOptions(r, "llamacpp", 8192, offers);
  assert.equal(opts.find((o) => o.offer.id === "16gb")?.quant.label, "Q2");   // each machine gets the best size it fits
  const rec = recommendedOption(opts)!;
  assert.equal(rec.offer.id, "24gb");                                          // Q4 at $0.15 beats Q2 at $0.08 and Q8 at $0.40
  assert.equal(rec.quant.label, "Q4");
  // A bigger machine for barely more money wins with the bigger size.
  const better = recommendedOption(cloudOptions(r, "llamacpp", 8192, [...offers, vast("48cheap", 48, 0.17)]))!;
  assert.deepEqual([better.offer.id, better.quant.label], ["48cheap", "Q8"]);
});

test("speed: Windows GPU estimate matches the measurement (Qwen3.5-0.8B Q8 on an RTX PRO 6000, Windows 10: 433–444 tokens/s)", () => {
  const r = loadModelRecipe("Qwen/Qwen3.5-0.8B")!;
  const q = r.quants.find((x) => x.format === "gguf" && x.label === "Q8")!;
  const win: Hardware = { os: "windows", arch: "x64", cpu: { model: "Intel Xeon", cores: 16, avx2: true, avx512: true }, ramGb: 137, diskFreeGb: 500,
    gpus: [{ vendor: "nvidia", name: "NVIDIA RTX PRO 6000 Blackwell Server Edition", vramGb: 96, cuda: "13.2", computeCap: "12.0" }], notes: [] };
  const s = estimateSpeed(r, q, localDevices(win), 1, 0.12);
  assert.ok(s.tpsMin <= 440 * 1.15 && s.tpsMax >= 440 * 0.85, `estimate ${s.tpsMin}-${s.tpsMax}`);
});

// ---------------------------------------------------------------- the set-up page (core/tune.ts)
const tuneModel = () => loadModelRecipe("Qwen/Qwen3-30B-A3B-Instruct-2507")!;
const vastOffer = (id: string, vram: number, price: number) => ({ provider: "vast" as const, id, gpuName: "RTX 3090", numGpus: 1, vramPerGpuGb: vram, ramGb: 64, diskGb: 200, pricePerHour: price, computeCap: 8.6 });
const tuneOffers = { "16gb": vastOffer("16gb", 16, 0.08), "24gb": vastOffer("24gb", 24, 0.15), "48gb": vastOffer("48gb", 48, 0.4), "80gb": vastOffer("80gb", 80, 1.0) };
async function tuneRows(ctx: number, offers = Object.values(tuneOffers)) {
  const { cloudFits } = await import("../core/plan.js");
  return offers.map((offer) => ({ offer, fits: cloudFits(tuneModel(), "llamacpp", ctx, offer) })).filter((m) => m.fits.length);
}

test("tune: Smart squeeze picks the best value machine; a size you set narrows the machines; a machine you set gets its best size", async () => {
  const { resolveCloud } = await import("../core/tune.js");
  const r = tuneModel();
  const rows = await tuneRows(8192);
  const s = { r, hw: hwCpu, where: "vast" as const, locks: { runtime: "llamacpp" as const, context: 8192 }, adapterGb: 0 };
  const smart = resolveCloud(s, rows);
  assert.deepEqual([smart.chosen?.offer.id, smart.quant?.label, smart.context], ["24gb", "Q4", 8192]);
  const q8 = r.quants.find((q) => q.format === "gguf" && q.label === "Q8")!;
  const sized = resolveCloud({ ...s, locks: { ...s.locks, size: q8.id } }, rows);
  assert.deepEqual(sized.rows!.map((m) => m.offer.id).sort(), ["48gb", "80gb"]);   // only machines Q8 fits
  assert.equal(sized.chosen?.offer.id, "48gb");
  const mine = resolveCloud({ ...s, locks: { ...s.locks, machine: tuneOffers["48gb"] } }, rows);
  assert.deepEqual([mine.chosen?.offer.id, mine.quant?.label], ["48gb", "Q8"]);
});

test("tune: with a machine you picked, sizes and contexts it can't hold are greyed out with the reason", async () => {
  const { resolveCloud, sizeOptions, contextOptions } = await import("../core/tune.js");
  const r = tuneModel();
  const rows = await tuneRows(8192);
  const s = { r, hw: hwCpu, where: "vast" as const, locks: { runtime: "llamacpp" as const, context: 8192, machine: tuneOffers["24gb"] }, adapterGb: 0 };
  const res = resolveCloud(s, rows);
  const sizes = sizeOptions(s, res, rows);
  const q8 = sizes.find((c) => c.label.startsWith("Q8"))!;
  assert.equal(q8.ok, false);
  assert.match(q8.why!, /too big for the machine you picked/);
  assert.equal(sizes.find((c) => c.label.startsWith("Q4"))!.ok, true);
  // With the size left to Smart squeeze, a longer context can still fit by going one size down; with Q4 set, it can't.
  const longer = contextOptions(s, res).find((c) => typeof c.value === "number" && c.value > res.ctxMax)!;
  assert.equal(longer.ok, true);
  assert.match(longer.note!, /Smart squeeze switches to/);
  const q4 = r.quants.find((q) => q.format === "gguf" && q.label === "Q4")!;
  const fixed = { ...s, locks: { ...s.locks, size: q4.id } };
  const fixedRes = resolveCloud(fixed, rows);
  const big = contextOptions(fixed, fixedRes).find((c) => typeof c.value === "number" && c.value > fixedRes.ctxMax)!;
  assert.equal(big.ok, false);
  assert.match(big.why!, /doesn't fit the machine you picked with Q4/);
  // Left to Smart squeeze, the same long context is allowed: it picks a bigger machine instead.
  const free = { ...s, locks: { ...s.locks, machine: undefined } };
  const freeRes = resolveCloud(free, rows);
  const bigger = contextOptions(free, freeRes).find((c) => typeof c.value === "number" && c.value > freeRes.ctxMax)!;
  assert.equal(bigger.ok, true);
  assert.match(bigger.note!, /bigger machine/);
});

test("tune: runtime and size follow each other by file format, and clashes with what you set are greyed out", async () => {
  const { runtimeFor, runtimeOptions, sizeOptions, resolveCloud, keepLocks } = await import("../core/tune.js");
  const r = tuneModel();
  const gguf = r.quants.find((q) => q.format === "gguf" && q.label === "Q8")!;
  const s = { r, hw: hwCpu, where: "vast" as const, locks: {}, adapterGb: 0 };
  assert.equal(runtimeFor({ ...s, locks: { size: gguf.id } }), "llamacpp");                 // a GGUF size takes llama.cpp with it
  const vllm = runtimeOptions({ ...s, locks: { size: gguf.id } }).find((c) => c.value === "vllm")!;
  assert.equal(vllm.ok, false);
  assert.match(vllm.why!, /no Q8 file/);
  const locked = { ...s, locks: { runtime: "vllm" as const } };
  const ggufOpt = sizeOptions(locked, resolveCloud(locked, [])).find((c) => c.value === gguf.id)!;
  assert.equal(ggufOpt.ok, false);
  assert.match(ggufOpt.why!, /needs llama\.cpp \(you picked vLLM\)/);
  // Moving to a rented GPU lets go of what can't run there, and says so.
  const k = keepLocks({ ...s, locks: { runtime: "mlx" as const } });
  assert.equal(k.locks.runtime, undefined);
  assert.deepEqual(k.dropped, ["MLX"]);
});

test("tune: where Smart squeeze starts, and what a change moved", async () => {
  const { smartWhere, snap, moved, resolveCloud } = await import("../core/tune.js");
  const r = tuneModel();
  const rows = await tuneRows(32768);
  const s = { r, hw: hwCpu, where: "local" as const, locks: {}, adapterGb: 0 };
  const pick = smartWhere(s, { vast: rows });            // no GPU here: it runs on a rented one
  assert.equal(pick.where, "vast");
  assert.match(pick.why, /Best quality for the price/);
  const cloud = { ...s, where: "vast" as const, locks: { runtime: "llamacpp" as const } };
  const a = snap("vast", resolveCloud(cloud, rows));
  const b = snap("vast", resolveCloud({ ...cloud, locks: { ...cloud.locks, machine: tuneOffers["80gb"] } }, rows));
  assert.ok(moved(a, b).includes("machine") && moved(a, b).includes("price"));
  assert.deepEqual(moved(a, a), []);
});

test("LoRA: a GGUF file is an adapter only when its header says general.type = adapter (merged models are often named \"…-lora-…\")", async () => {
  const { ggufHeaderValue } = await import("../core/lora.js");
  const gguf = (kvs: [string, number, Buffer][]) => {
    const s = (v: string) => { const b = Buffer.from(v); const n = Buffer.alloc(8); n.writeBigUInt64LE(BigInt(b.length)); return Buffer.concat([n, b]); };
    const head = Buffer.alloc(24); head.write("GGUF", 0, "latin1"); head.writeUInt32LE(3, 4); head.writeBigUInt64LE(BigInt(kvs.length), 16);
    return Buffer.concat([head, ...kvs.map(([k, t, v]) => { const tb = Buffer.alloc(4); tb.writeUInt32LE(t); return Buffer.concat([s(k), tb, v]); })]);
  };
  const str = (v: string) => { const b = Buffer.from(v); const n = Buffer.alloc(8); n.writeBigUInt64LE(BigInt(b.length)); return Buffer.concat([n, b]); };
  const u32 = Buffer.alloc(4); u32.writeUInt32LE(7);
  const arr = Buffer.concat([Buffer.from([4, 0, 0, 0]), Buffer.from([2, 0, 0, 0, 0, 0, 0, 0]), u32, u32]); // array of 2 × u32
  assert.equal(ggufHeaderValue(gguf([["general.architecture", 8, str("qwen2")], ["x.n", 4, u32], ["x.list", 9, arr], ["general.type", 8, str("adapter")]]), "general.type"), "adapter");
  assert.equal(ggufHeaderValue(gguf([["general.type", 8, str("model")]]), "general.type"), "model");
  assert.equal(ggufHeaderValue(Buffer.from("not a gguf file at all, no"), "general.type"), undefined);
});

test("tune: on a big machine you picked, every smaller size can still be picked, and the machine stays even when a search leaves it out", async () => {
  const { resolveCloud, sizeOptions, runtimeFor } = await import("../core/tune.js");
  const r = tuneModel();
  const big = tuneOffers["80gb"];
  const s = { r, hw: hwCpu, where: "vast" as const, locks: { context: 8192, machine: big }, adapterGb: 0 };
  // Every GGUF size fits 80 GB, whichever runtime is in use now: none of them is greyed out.
  const rows = await tuneRows(8192);
  const ggufs = sizeOptions(s, resolveCloud(s, rows), rows).filter((c) => c.value && r.quants.find((q) => q.id === c.value)?.format === "gguf");
  assert.ok(ggufs.length >= 5);
  assert.deepEqual(ggufs.filter((c) => !c.ok).map((c) => c.label), []);
  // Pick the smallest. A new search for it only lists cheap small machines; the one you picked stays in use.
  const q2 = r.quants.find((q) => q.format === "gguf" && q.label === "Q2")!;
  const picked = { ...s, locks: { ...s.locks, size: q2.id } };
  assert.equal(runtimeFor(picked), "llamacpp");
  const cheapOnly = await tuneRows(8192, [tuneOffers["16gb"], tuneOffers["24gb"]]);
  const res = resolveCloud(picked, cheapOnly);
  assert.deepEqual([res.chosen?.offer.id, res.quant?.label, res.machineGone], ["80gb", "Q2", false]);
});

test("tune: on this computer with a big GPU, every GGUF size can be picked", async () => {
  const { resolveLocal, sizeOptions } = await import("../core/tune.js");
  const r = tuneModel();
  const bigLocal: Hardware = { ...hw4090, gpus: [{ vendor: "nvidia", name: "NVIDIA A100-SXM4-80GB", vramGb: 80, cuda: "13.4", computeCap: "8.0" }] };
  const s = { r, hw: bigLocal, where: "local" as const, locks: {}, adapterGb: 0 };
  const ggufs = sizeOptions(s, resolveLocal(s)).filter((c) => c.value && r.quants.find((q) => q.id === c.value)?.format === "gguf");
  assert.ok(ggufs.length >= 5);
  assert.deepEqual(ggufs.filter((c) => !c.ok).map((c) => `${c.label}: ${c.why}`), []);
});

test("downloads: follow a redirect to another host without the token, and resume a broken download with Range", async () => {
  const http = await import("node:http");
  const os = await import("node:os");
  const { downloadFiles } = await import("../core/hfdownload.js");
  const data = Buffer.alloc(300_000); for (let i = 0; i < data.length; i++) data[i] = i % 251;
  const seen: { auth?: string; range?: string }[] = [];
  let broke = false;
  const cdn = http.createServer((req, res) => {
    seen.push({ auth: req.headers.authorization, range: req.headers.range });
    const m = /bytes=(\d+)-/.exec(req.headers.range || "");
    const from = m ? Number(m[1]) : 0;
    res.writeHead(m ? 206 : 200, { "content-length": String(data.length - from) });
    if (!broke) { broke = true; res.write(data.subarray(from, from + 100_000)); setTimeout(() => res.destroy(), 50); return; }   // drop mid-file once
    res.end(data.subarray(from));
  });
  await new Promise<void>((r) => cdn.listen(0, "127.0.0.1", r));
  const cdnPort = (cdn.address() as any).port;
  const hub = http.createServer((req, res) => { res.writeHead(302, { location: `http://localhost:${cdnPort}/blob` }); res.end(); });
  await new Promise<void>((r) => hub.listen(0, "127.0.0.1", r));
  const dest = fs.mkdtempSync(path.join(os.tmpdir(), "spunup-dl-"));
  const { setKey } = await import("../core/keys.js");
  assert.ok(process.env.SPUNUP_NO_KEYCHAIN, "tests must not touch the system keychain");
  try {
    setKey("hf", "hf_test_token");   // goes to the test folder's secrets.json, not the keychain
    let last = 0;
    await downloadFiles([{ path: "model.gguf", size: data.length, url: `http://127.0.0.1:${(hub.address() as any).port}/file` }], dest, (d) => (last = d), () => false);
    assert.ok(fs.readFileSync(path.join(dest, "model.gguf")).equals(data), "file is complete and in order");
    assert.equal(last, data.length);
    assert.ok(seen.length >= 2 && seen[1].range === "bytes=100000-", "second try resumed where the first stopped");
    assert.ok(seen.every((s) => !s.auth), "token not sent to the other host");
  } finally {
    hub.close(); cdn.close(); fs.rmSync(dest, { recursive: true, force: true });
  }
});

test("LoRA on a GPU: each adapted weight adds a fixed cost per token (Qwen2.5-7B Q8 on an RTX PRO 6000, Windows: 150.5 tokens/s without the adapter, 88.5–91 with it)", async () => {
  const { withAdapter } = await import("../core/plan.js");
  const r = { id: "Qwen/Qwen2.5-7B-Instruct", params_b: 7.6, active_b: 7.6, moe: false, layers: 28 } as unknown as Recipe;
  const q = { id: "q8", label: "Q8", format: "gguf", repo: "x", size_gb: 8.1, runtimes: ["llamacpp"] } as any;
  const win: Hardware = { os: "windows", arch: "x64", cpu: { model: "AMD EPYC 9354", cores: 16, avx2: true, avx512: true }, ramGb: 137, diskFreeGb: 500,
    gpus: [{ vendor: "nvidia", name: "NVIDIA RTX PRO 6000 Blackwell Server Edition", vramGb: 96, cuda: "13.2", computeCap: "12.0" }], notes: [] };
  const dev = localDevices(win);
  const plain = estimateSpeed(r, q, dev, 1, 0.12);
  assert.ok(plain.tpsMin <= 150.5 * 1.15 && plain.tpsMax >= 150.5 * 0.85, `no adapter: ${plain.tpsMin}-${plain.tpsMax}`);
  const lora = withAdapter(plain, r, q, 0.36, dev.gpu!.overheadS, 1);
  assert.ok(lora.tpsMin <= 90 * 1.15 && lora.tpsMax >= 90 * 0.85, `with adapter: ${lora.tpsMin}-${lora.tpsMax}`);
  // On the CPU only the extra bytes count (Linux run: 2.49 tokens/s with the adapter, estimate 2).
  const cpu = withAdapter({ tpsMin: 2, tpsMax: 2.4, prefill: 15, basis: "" }, r, q, 0.36);
  assert.ok(cpu.tpsMax > 2.2, `cpu: ${cpu.tpsMax}`);
});

test("smart squeeze: a machine too slow to use loses to a faster one, unless nothing is faster", async () => {
  const { recommendedOption, cloudFits } = await import("../core/plan.js");
  const r = tuneModel();
  const q = r.quants.find((x) => x.format === "gguf" && x.label === "Q4")!;
  const o = (id: string, gpuName: string, bw: number, price: number) => ({ provider: "vast" as const, id, gpuName, numGpus: 1, vramPerGpuGb: 120, ramGb: 64, diskGb: 300, pricePerHour: price, computeCap: 8.6, gpuMemBw: bw });
  const opt = (x: ReturnType<typeof o>) => cloudFits(r, "llamacpp", 8192, x).find((c) => c.quant.id === q.id)!;
  const slow = opt(o("slow", "Unknown GPU", 20, 0.1)), fast = opt(o("fast", "RTX A6000", 768, 0.5));
  assert.ok(slow.speed.tpsMax < 10 && fast.speed.tpsMax >= 10);
  assert.equal(recommendedOption([slow, fast])!.offer.id, "fast");
  assert.equal(recommendedOption([slow])!.offer.id, "slow");
});

test("speed: big mixture-of-experts models land near SGLang's measured single-request speeds (cookbook, no speculation)", async () => {
  const { offerDevice } = await import("../core/plan.js");
  const points: [string, string, string, number, number][] = [
    ["moonshotai/Kimi-K3.1-Pro", "FP4", "B300", 8, 9.6],
    ["MiniMaxAI/MiniMax-M2.5", "FP8", "B200", 8, 9.89],
    ["MiniMaxAI/MiniMax-M2.5", "FP8", "H100 SXM", 8, 8.08],
    ["Qwen/Qwen3.8-Flash-Next", "NVFP4", "RTX PRO 6000", 1, 11.4],
  ];
  for (const [id, label, gpu, n, ms] of points) {
    const r = loadModelRecipe(id)!;
    const q = r.quants.find((x) => x.label === label)!;
    const dev = offerDevice({ provider: "runpod", id: "x", gpuName: gpu, numGpus: n, vramPerGpuGb: 200, ramGb: 512, diskGb: 2000, pricePerHour: 1 });
    const est = 1000 / estimateSpeed(r, q, dev, 1, 0.25).tpsMax;
    assert.ok(est > ms * 0.65 && est < ms * 1.5, `${id} on ${n}× ${gpu}: estimated ${est.toFixed(1)} ms a token, measured ${ms}`);
  }
});

test("tune: Max is the longest context you can really get, not just what the picked machine holds", async () => {
  const { resolveCloud, contextMax, contextOptions } = await import("../core/tune.js");
  const r = tuneModel();
  const rows = await tuneRows(8192);
  const s = { r, hw: hwCpu, where: "vast" as const, locks: { runtime: "llamacpp" as const, context: 8192 }, adapterGb: 0 };
  const res = resolveCloud(s, rows);
  const max = contextMax(s, res)!;                      // no full-context search: the most any listed machine holds
  assert.ok(max > res.ctxMax, `Max ${max} vs the picked machine's ${res.ctxMax}`);
  const opts = contextOptions(s, res, max);
  assert.equal(opts[1].label, `Max (${max.toLocaleString("en-GB")})`);
  assert.equal(opts[1].ok, true);
  for (const c of opts) if (typeof c.value === "number" && c.value > max) assert.equal(c.ok, false, `${c.label} is past Max but can be picked`);
});

test("sweep: a Q4 that spills into RAM beats a Q1 that fits the GPU; local ✦ context keeps the model on the GPU", async () => {
  const { preferredQuant } = await import("../core/plan.js");
  const { resolveLocal } = await import("../core/tune.js");
  const llama = loadModelRecipe("meta-llama/Llama-3.3-70B-Instruct")!;
  const p = preferredQuant(llama, "llamacpp", hw4090, 8192)!;
  assert.ok(!["Q1", "Q2"].includes(p.quant.label), `picked ${p.quant.label}`);
  const coder = loadModelRecipe("Qwen/Qwen3-Coder-30B-A3B-Instruct")!;
  const res = resolveLocal({ r: coder, hw: hw4090, where: "local", locks: {}, adapterGb: 0 });
  assert.equal(res.fit?.setup, "gpu", `✦ context ${res.context} spills: ${res.fit?.setup}`);
  assert.ok(res.ctxMax >= res.context);
});

test("sweep: vLLM / SGLang only rent 1, 2, 4 or 8 GPUs; a custom GGUF file is rated by its quant name", async () => {
  const { cloudMiss, quality } = await import("../core/plan.js");
  const r = tuneModel();
  const q = r.quants.find((x) => x.format === "safetensors")!;
  const o = (n: number) => ({ provider: "vast" as const, id: "x", gpuName: "RTX 3090", numGpus: n, vramPerGpuGb: 80, ramGb: 64, diskGb: 2000, pricePerHour: 1, computeCap: 8.6 });
  assert.equal(cloudMiss(r, "vllm", 8192, o(3), q), "gpus");
  assert.notEqual(cloudMiss(r, "vllm", 8192, o(4), q), "gpus");
  assert.notEqual(cloudMiss(r, "llamacpp", 8192, o(3), r.quants.find((x) => x.format === "gguf")!), "gpus");
  const custom = { id: "gguf:x", label: "Qwen3-8B-Q8_0", quant: "Q8_0", format: "gguf" as const, repo: "x", size_gb: 8, runtimes: ["llamacpp" as const] };
  assert.equal(quality(custom), 90);
});
