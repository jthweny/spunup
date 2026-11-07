import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import { P, IS_WIN } from "./paths.js";
import { diskNeeded } from "./plan.js";
import { runSync } from "./proc.js";
import { tensorParallelGpus } from "./fit.js";
import { upsertEndpoint, getEndpoint, loadConfig, saveConfig, loadHistory, logFile, type Endpoint, type RuntimeId, type Setup, type Where } from "./store.js";
import type { Recipe, Quant } from "./recipes.js";
import type { Hardware } from "./hardware.js";
import type { Fit, Speed } from "./fit.js";
import { freePort, portFree, download, sleep } from "./net.js";
import { getKey } from "./keys.js";
import { filesFor, localModelDir, downloadFiles, remoteDownloadScript } from "./hfdownload.js";
import { buildLaunch, commandString, cpuMoeLayers, IMAGES, type LaunchOptions } from "../runtimes/index.js";
import { installLlamaCpp, installPython, wslWrap, WSL_SETUP_SCRIPT } from "../runtimes/install.js";
import { ensureSshKey, sshExec, sshWrite, waitForSsh, type Remote } from "./ssh.js";
import { provider, NoCapacity, type Offer } from "../providers/index.js";
import { ensureKeeper, costSoFar } from "./keeper.js";
import { testEndpoint, waitHealthy } from "./test.js";
import { restoreRawMode } from "./proc.js";
import { adapterUrl, vllmRank, type LoraUse } from "./lora.js";

export interface Plan {
  recipe: Recipe;
  quant: Quant;
  where: Where;
  runtime: RuntimeId;
  setup: Setup;
  context: number;
  hw?: Hardware;
  fit?: Fit;
  speed?: Speed;
  offer?: Offer;
  existingInstanceId?: string;
  lora?: LoraUse;                // adapter found in the model's repo (on unless the user turned it off)
  advanced?: { extraArgs?: string[]; command?: string; lora?: LoraUse };  // Advanced: an adapter from elsewhere wins
  endpoint?: { port?: number; apiKey?: string; name?: string };   // your own port / API key / model name (else spunup's)
}

/** The adapter this setup uses, if any. */
export function planLora(p: Plan): LoraUse | undefined { return p.advanced?.lora || p.lora; }

export type StepState = "todo" | "doing" | "done" | "failed";

/** The runtime failed while starting: keeps what it printed so the Starting screen can show it. */
export class RuntimeFailed extends Error {
  constructor(message: string, public log: string[], public logPath: string) { super(message); }
}
export interface Step { key: string; label: string; state: StepState; detail?: string }

export function newApiKey(): string {
  return "sk-spunup-" + crypto.randomBytes(18).toString("base64url");
}

/** Same local port every time for the same model + place (spec 8: apps never change settings). */
export async function fixedPort(model: string, where: Where): Promise<number> {
  const cfg = loadConfig();
  const key = `${model}|${where}`;
  const used = new Set(Object.entries(cfg.ports).filter(([k]) => k !== key).map(([, v]) => v));
  const live = new Set(loadHistory().endpoints.filter((e) => e.status === "running" || e.status === "starting").map((e) => e.port));
  for (const j of activeJobs()) if (j.launcher.endpoint) live.add(j.launcher.endpoint.port);
  if (cfg.ports[key] && !live.has(cfg.ports[key]) && (await portFree(cfg.ports[key]))) return cfg.ports[key];
  const p = await freePort(cfg.firstPort || 8000, new Set([...used, ...live]));
  cfg.ports[key] = p;
  saveConfig(cfg);
  return p;
}

/** Whether this PID is a model runtime spunup starts (not a program that reused the number). */
function isRuntimeProcess(pid: number): boolean {
  const RT = /llama-server|vllm|sglang|mlx_lm|python|wsl/i;
  try {
    const r = IS_WIN ? runSync("tasklist", ["/FI", `PID eq ${pid}`, "/FO", "CSV", "/NH"], { encoding: "utf8" }) : runSync("ps", ["-p", String(pid), "-o", "command="], { encoding: "utf8" });
    return r.status === 0 && RT.test(r.stdout || "");
  } catch { return false; }
}

/** Why this port can't be the endpoint for this model here (undefined = it can). */
export async function checkPort(port: number, model: string, where: Where): Promise<string | undefined> {
  if (!Number.isInteger(port) || port < 1024 || port > 65535) return "Use a whole number from 1024 to 65535.";
  const owner = Object.entries(loadConfig().ports).find(([k, v]) => v === port && k !== `${model}|${where}`);
  if (owner) return `spunup keeps port ${port} for ${owner[0].split("|")[0]}, so apps set up for it keep working. Pick another.`;
  const live = loadHistory().endpoints.some((e) => e.port === port && (e.status === "running" || e.status === "starting")) || activeJobs().some((j) => j.launcher.endpoint?.port === port);
  if (live || !(await portFree(port))) return `Something on this computer is already using port ${port}. Pick another.`;
  return undefined;
}

/** The API key this model here will get unless you set one: last time's (so apps keep working), or a new one. */
export function keptApiKey(model: string, where: Where): string | undefined {
  return loadHistory().endpoints.find((e) => e.model === model && e.where === where && e.apiKey)?.apiKey;
}

function gpuGen(name: string): LaunchOptions["gpuGen"] {
  const n = name.toUpperCase();
  if (/H100|H200|GH200|H800|H20\b/.test(n)) return "hopper";
  if (/B200|B300|GB200|GB300|RTX 50|RTX PRO 6000|BLACKWELL/.test(n)) return "blackwell";
  if (/A100|A800|A10|A40|A6000|A5000|A4000|RTX 30/.test(n)) return "ampere";
  if (/RTX 40|L40|L4\b|6000 ADA|ADA/.test(n)) return "ada";
  return "other";
}

/** The model name apps send as "model" unless you set your own. */
export const defaultModelName = (recipeName: string) => slug(recipeName);

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9.]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
}

function fmtEta(s: number): string {
  if (!isFinite(s) || s <= 0) return "";
  return s < 90 ? `~${Math.round(s)} s left` : `~${Math.round(s / 60)} min left`;
}

export class Launcher extends EventEmitter {
  steps: Step[];
  endpoint!: Endpoint;
  cancelled = false;
  error?: { message: string; next: string[]; machineAlive: boolean; log?: string[]; logPath?: string; pickAnother?: boolean };
  private child?: ReturnType<typeof spawn>;

  constructor(public plan: Plan) {
    super();
    const cloud = plan.where !== "local";
    this.steps = cloud
      ? [
          { key: "rent", label: plan.existingInstanceId ? "Machine ready" : "Machine rented", state: "todo" },
          { key: "connect", label: "Connected (private SSH tunnel)", state: "todo" },
          { key: "download", label: "Downloading model", state: "todo" },
          { key: "start", label: "Starting runtime", state: "todo" },
          { key: "test", label: "Testing endpoint", state: "todo" },
        ]
      : [
          { key: "runtime", label: "Getting the runtime ready", state: "todo" },
          { key: "download", label: "Downloading model", state: "todo" },
          { key: "start", label: "Starting runtime", state: "todo" },
          { key: "test", label: "Testing endpoint", state: "todo" },
        ];
  }

  private lastLog = "";
  log(m: string) {
    if (m === this.lastLog) return;
    this.lastLog = m;
    try { fs.appendFileSync(logFile("launch.log"), `${new Date().toISOString()} ${this.endpoint?.id || "-"} ${m}\n`); } catch { /* ignore */ }
  }

  private set(key: string, state: StepState, detail?: string) {
    if (state !== "doing" || !detail || !/%/.test(detail)) this.log(`${key}: ${state}${detail ? ` (${detail})` : ""}`);
    const s = this.steps.find((x) => x.key === key)!;
    s.state = state;
    s.detail = detail;
    this.emit("update");
  }

  private check() {
    if (this.cancelled) throw new Error("cancelled");
  }

  cancel() {
    this.cancelled = true;
    if (this.child?.pid) { try { process.kill(this.child.pid); } catch { /* gone */ } }
    if (this.endpoint && this.plan.where === "local") {
      if (this.endpoint.pid) { try { process.kill(this.endpoint.pid); } catch { /* gone */ } }
      upsertEndpoint({ ...this.endpoint, status: "stopped" });
    }
    this.emit("update");
  }

  /** Stop the local runtime and anything it started (it runs detached, in its own process group). */
  private killChild(): void {
    const pid = this.child?.pid;
    if (!pid) return;
    try { if (!IS_WIN) process.kill(-pid); else throw new Error(); } catch { try { process.kill(pid); } catch { /* gone */ } }
  }

  async run(cliPath: string): Promise<Endpoint | undefined> {
    const { plan } = this;
    let port = plan.endpoint?.port;
    if (port) {   // your own port is kept for this model here from now on, like spunup's own pick (spec 8)
      const cfg = loadConfig();
      cfg.ports[`${plan.recipe.id}|${plan.where}`] = port;
      saveConfig(cfg);
    } else port = await fixedPort(plan.recipe.id, plan.where);
    const name = plan.endpoint?.name || defaultModelName(plan.recipe.name);
    this.endpoint = {
      id: `e-${Date.now().toString(36)}-${crypto.randomBytes(2).toString("hex")}`,
      model: plan.recipe.id, name, where: plan.where, runtime: plan.runtime, setup: plan.setup,
      quant: plan.quant.label, quantId: plan.quant.id, context: plan.context, port, apiKey: newApiKey(),
      createdAt: Date.now(), status: "starting",
      estimate: plan.speed ? { tpsMin: plan.speed.tpsMin, tpsMax: plan.speed.tpsMax, prefill: plan.speed.prefill } : undefined,
      autoStop: { enabled: false, minutes: 30, action: "stop" },
    };
    // Your own key, else the same key as last time, so apps keep working.
    this.endpoint.apiKey = plan.endpoint?.apiKey || keptApiKey(plan.recipe.id, plan.where) || this.endpoint.apiKey;
    try {
      if (plan.where === "local") await this.runLocal(cliPath);
      else await this.runCloud(cliPath);
      return this.endpoint;
    } catch (e) {
      const msg = (e as Error).message;
      this.log(`error: ${msg}`);
      if (msg === "cancelled") return undefined;
      const doing = this.steps.find((s) => s.state === "doing");
      if (doing) doing.state = "failed";
      const rf = e instanceof RuntimeFailed ? e : undefined;
      if (rf) this.log(`runtime output:\n${rf.log.join("\n")}`);
      const gone = e instanceof NoCapacity;
      // A local runtime that failed its checks would keep holding the GPU and the port, hidden from History.
      if (plan.where === "local") this.killChild();
      this.error = {
        message: msg, next: gone ? ["Nothing was rented, so nothing is charged. Pick another machine: the list is checked again."] : this.suggest(`${msg}\n${rf?.log.join("\n") || ""}`, doing?.key),
        // Only machines spunup rented are offered for deletion (never one you already had).
        machineAlive: plan.where !== "local" && !!this.endpoint.instanceId && !plan.existingInstanceId, log: rf?.log, logPath: rf?.logPath, pickAnother: gone,
      };
      this.endpoint.lastError = msg;
      // Nothing rented → failed. A rented machine stays listed as running so it can be deleted from History.
      const rented = plan.where !== "local" && !!this.endpoint.instanceId;
      upsertEndpoint({ ...this.endpoint, status: rented ? (this.endpoint.status === "starting" ? "running" : this.endpoint.status) : "failed" });
      this.emit("update");
      return undefined;
    }
  }

  private suggest(msg: string, step?: string): string[] {
    const out: string[] = [];
    if (/out of memory|CUDA out of memory|OOM|cudaMalloc|not enough memory|insufficient/i.test(msg)) out.push("Pick a smaller size (quantization) or a shorter context.");
    if (/gated|refused|401|403/i.test(msg)) out.push("Open the model page on Hugging Face, press Agree, and check your token in Keys & tokens.");
    // llama.cpp: "LoRA tensor 'output.weight' does not exist in base model". Seen with an adapter that
    // changes the output layer of a model that shares it with the input (tied embeddings, Qwen2.5-1.5B).
    if (/LoRA tensor .* does not exist/i.test(msg)) out.push("This adapter changes parts of the model that this copy does not have, so it was made for a different build of the model. Turn LoRA off, or use another adapter.");
    else if (/lora|adapter/i.test(msg)) out.push("The LoRA adapter did not load. Check it was made for this model, or turn it off in the LoRA row.");
    if (/unknown (model )?architecture|not supported|unsupported|no module named|KeyError: '\w+'/i.test(msg)) out.push("This runtime version cannot run this model. Pick another runtime in the Runtime row.");
    if (/address already in use|bind/i.test(msg)) out.push("Another program is using the port. Close it, or stop the other model in History.");
    if (step === "rent" || /taken|place this pod|capacity/i.test(msg)) out.push("Try another machine from the list.");
    if (step === "start" && this.plan.runtime !== "llamacpp") out.push("Try llama.cpp in the Runtime row.");
    if (!out.length) out.push("Try again, or pick another machine or size.");
    return out;
  }

  // ---------------------------------------------------------------- local
  private async runLocal(cliPath: string) {
    const { plan } = this;
    const e = this.endpoint;
    const hw = plan.hw!;
    const winLinux = IS_WIN && (plan.runtime === "vllm" || plan.runtime === "sglang");
    this.set("runtime", "doing");
    let bin: string;
    if (plan.runtime === "llamacpp") {
      bin = await installLlamaCpp(hw, (m, pct) => this.set("runtime", "doing", pct !== undefined ? `${m} ${pct}%` : m));
    } else if (winLinux) {
      this.set("runtime", "doing", "Installing inside WSL (first time only)");
      await new Promise<void>((resolve, reject) => {
        const w = spawn("wsl.exe", ["-e", "bash", "-lc", WSL_SETUP_SCRIPT(plan.runtime as "vllm" | "sglang")], { windowsHide: true });
        w.on("exit", restoreRawMode);
        let out = "";
        w.stdout.on("data", (b) => (out += b));
        w.stderr.on("data", (b) => (out += b));
        w.on("close", (c) => (c === 0 && out.includes("SPUNUP_OK") ? resolve() : reject(new Error(`Setting up ${plan.runtime} in WSL failed: ${out.trim().split("\n").slice(-2).join(" ")}`))));
      });
      bin = plan.runtime === "vllm" ? "~/.spunup/vllm-env/bin/vllm" : "~/.spunup/sglang-env/bin/python3";
    } else {
      const rr = plan.recipe.runtimes[plan.runtime];
      bin = await installPython(plan.runtime as "vllm" | "sglang" | "mlx", { nightly: rr.nightly, deps: rr.deps }, (m, pct) => this.set("runtime", "doing", pct !== undefined ? `${m}…` : m));
    }
    this.set("runtime", "done");
    this.check();

    this.set("download", "doing", "0%");
    this.log(`listing files in ${plan.quant.repo}`);
    const files = await filesFor(plan.quant);
    this.log(`${files.length} files, ${files.reduce((a, f) => a + f.size, 0)} bytes`);
    if (!files.length) throw new Error(`No model files found in ${plan.quant.repo}.`);
    const dest = localModelDir(plan.quant);
    const t0 = Date.now();
    let d0: number | undefined;   // bytes already on disk from an earlier try: not part of the speed
    await downloadFiles(files, dest, (d, t) => {
      const pct = t ? Math.floor((d / t) * 100) : 0;
      if (d0 === undefined) d0 = d;
      const rate = (d - d0) / Math.max(1, (Date.now() - t0) / 1000);
      this.set("download", "doing", `${pct}%  ${fmtEta((t - d) / rate)}`);
    }, () => this.cancelled);
    const rr = plan.recipe.runtimes[plan.runtime];
    let chatTemplatePath: string | undefined, pluginPath: string | undefined;
    if (plan.runtime === "vllm" && rr.chat_template?.startsWith("http")) {
      chatTemplatePath = path.join(dest, "spunup_tool_template.jinja");
      if (!fs.existsSync(chatTemplatePath)) await download(rr.chat_template!, chatTemplatePath);
    }
    if (plan.runtime === "vllm" && rr.reasoning_plugin) pluginPath = path.join(dest, rr.reasoning_plugin);
    const lora = await this.loraLocal();
    this.set("download", "done");
    this.check();

    this.set("start", "doing");
    const modelPath = plan.quant.format === "gguf" ? path.join(dest, files[0].path) : dest;
    const innerPort = plan.runtime === "mlx" ? await freePort(e.port + 1000) : undefined;
    const launch = buildLaunch(plan.runtime, plan.recipe, plan.quant, {
      modelPath: winLinux ? toWslPath(modelPath) : modelPath, servedName: e.name, host: "127.0.0.1", port: innerPort || e.port,
      apiKey: e.apiKey, context: plan.context, gpus: tensorParallelGpus(hw.gpus.filter((g) => g.vendor === "nvidia").length), setup: plan.setup,
      cpuMoeLayers: plan.setup === "mixed" && plan.fit ? cpuMoeLayers(plan.recipe, plan.quant, plan.fit.gpuShare) : undefined,
      chatTemplatePath, pluginPath, gpuGen: hw.gpus[0] ? gpuGen(hw.gpus[0].name) : "other",
      lora, extraArgs: plan.advanced?.extraArgs,
    });
    launch.bin = bin;
    let cmd = launch.bin, args = launch.args;
    if (plan.advanced?.command) {
      // Advanced: the user's own start command (placeholders filled in).
      const own = plan.advanced.command.replace("{model}", modelPath).replace("{port}", String(innerPort || e.port)).replace("{api_key}", e.apiKey).replace("{ctx}", String(plan.context)).replace("{name}", e.name);
      cmd = IS_WIN ? "cmd.exe" : "sh"; args = IS_WIN ? ["/c", own] : ["-c", own];
    } else if (winLinux) {
      const w = wslWrap(launch.bin, launch.args); cmd = w.bin; args = w.args;
    }
    const lf = logFile(`${e.id}.log`);
    const fd = fs.openSync(lf, "a");
    const child = spawn(cmd, args, { detached: true, stdio: ["ignore", fd, fd], env: { ...process.env, ...launch.env, ...(getKey("hf") ? { HF_TOKEN: getKey("hf")! } : {}) }, windowsHide: true });
    child.unref();
    this.child = child;
    let exited = false;
    child.on("exit", () => (exited = true));
    Object.assign(e, { pid: child.pid, innerPort, logFile: lf, command: commandString({ ...launch, bin: cmd === launch.bin ? launch.bin : cmd, args }),
      launch: { bin: cmd, args, env: launch.env }, startedAt: Date.now() });
    upsertEndpoint(e);
    try {
      await waitHealthy(innerPort || e.port, innerPort ? undefined : e.apiKey, () => !exited, 45 * 60_000, () => this.cancelled);
    } catch (err) {
      if ((err as Error).message === "runtime-exited") { const out = logTail(fs.existsSync(lf) ? fs.readFileSync(lf, "utf8") : ""); throw new RuntimeFailed(`The runtime stopped while loading the model: ${tidy(out.join("\n"))}`, out, lf); }
      throw err;
    }
    this.set("start", "done");
    this.check();
    if (innerPort) { e.status = "running"; upsertEndpoint(e); ensureKeeper(cliPath); await sleep(2500); }
    await this.testStep();
  }

  // ---------------------------------------------------------------- cloud
  private async runCloud(cliPath: string) {
    const { plan } = this;
    const e = this.endpoint;
    const prov = provider(plan.where as "vast" | "runpod");
    this.set("rent", "doing");
    const pub = ensureSshKey();
    await prov.registerKey(pub);
    const image = cloudImage(plan);
    const diskGb = diskNeeded(plan.quant);
    let id = plan.existingInstanceId;
    if (!id) {
      const r = await prov.create({ offer: plan.offer!, image, diskGb, label: `spunup-${e.name}`.slice(0, 60), publicKey: pub, keepDiskSeparately: prov.separateDisk });
      id = r.id;
      e.volumeId = r.volumeId;
    }
    Object.assign(e, {
      instanceId: id, machine: plan.offer ? `${plan.offer.numGpus}× ${plan.offer.gpuName}` : "your machine",
      pricePerHour: plan.offer?.pricePerHour, startedAt: Date.now(), remotePort: 8000,
      oneOffCost: 0, downloadPerGb: plan.offer?.downloadPerGb,   // download charges are read from the provider as they happen
    });
    upsertEndpoint(e);  // saved now, so the machine shows in History even if spunup is closed
    if (plan.existingInstanceId && prov.attachKey) await prov.attachKey(id, pub).catch(() => { /* RunPod: keys come with the pod */ });
    // A provider hiccup (timeout, 5xx, 429) while waiting is retried, not the end of the setup.
    const getInst = async () => { for (let i = 0; ; i++) { try { return await prov.get(id!); } catch (err) { if (i >= 5) throw err; await sleep(5000); } } };
    let inst = await getInst();
    const until = Date.now() + 25 * 60_000;
    while (!(inst.status === "running" && inst.ssh)) {
      this.check();
      if (inst.status === "gone" || inst.status === "error") throw new Error(`The machine failed to start (${inst.statusText || inst.status}).`);
      if (Date.now() > until) throw new Error("The machine did not start within 25 minutes.");
      this.set("rent", "doing", inst.statusText ? `waiting (${String(inst.statusText).slice(0, 50)})` : "waiting for the machine to boot");
      await sleep(5000);
      inst = await getInst();
    }
    if (inst.pricePerHour) e.pricePerHour = inst.pricePerHour;
    e.stoppedPricePerHour = inst.stoppedPricePerHour;
    if (inst.volumeId) e.volumeId = inst.volumeId;
    this.set("rent", "done");

    this.set("connect", "doing");
    const remote: Remote = inst.ssh!;
    await waitForSsh(remote, 10 * 60_000, () => this.cancelled);
    e.ssh = remote;
    e.innerPort = await freePort(e.port + 2000);
    this.check();   // cancelled meanwhile (Stop / Delete): don't save it back as running
    e.status = "running";
    upsertEndpoint(e);
    ensureKeeper(cliPath);
    this.set("connect", "done");
    this.check();

    // Download on the machine.
    this.set("download", "doing", "0%");
    const files = await filesFor(plan.quant);
    const total = files.reduce((a, f) => a + f.size, 0);
    const wd = prov.workDir;
    const modelDir = `${wd}/models/${plan.quant.repo.replace("/", "__")}`;
    const token = getKey("hf");
    const lu = planLora(plan);
    if (lu?.localPath) throw new Error("A LoRA adapter from a folder on this computer only works for Local. For a rented machine, use a Hugging Face adapter name.");
    const la = lu?.adapter;
    const loraDir = la ? `${wd}/loras/${la.repo.replace("/", "__")}` : "";
    const loraFiles = la ? la.files.map((f) => ({ ...f, url: adapterUrl(la, f.path), out: `${loraDir}/${f.path}` })) : [];
    if (token) await sshWrite(remote, `${wd}/.hf_token`, token);
    await sshWrite(remote, `${wd}/download.sh`, remoteDownloadScript([...files, ...loraFiles], modelDir, !!token), "700");
    // The exit code is written at the end, so a download that gave up (Hugging Face 429/5xx after curl's
    // retries, or wget, which prints nothing) is noticed instead of waiting forever while the machine bills.
    await sshExec(remote, `cd ${wd} && (${token ? `HF_TOKEN=$(cat ${wd}/.hf_token) ` : ""}nohup sh -c 'sh download.sh; echo SPUNUP_DOWNLOAD_EXIT:$?' > download.log 2>&1 < /dev/null &)`);
    const t0 = Date.now();
    let done0: number | undefined;
    let grew = { bytes: -1, at: Date.now() };
    for (;;) {
      this.check();
      await sleep(4000);
      const out = await sshExec(remote, `du -sb ${modelDir} 2>/dev/null | cut -f1; tail -c 400 ${wd}/download.log 2>/dev/null`).catch(() => "");
      const [first, ...rest] = out.split("\n");
      const done = Number(first) || 0;
      const tail = rest.join("\n");
      if (tail.includes("SPUNUP_DOWNLOAD_DONE")) break;
      const exit = /SPUNUP_DOWNLOAD_EXIT:(\d+)/.exec(tail);
      if (exit && exit[1] !== "0") {
        const why = tail.split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("SPUNUP_")).pop();
        throw new Error(`The download stopped on the machine${why ? `: ${why}` : ""}. Try again, or pick another machine.`);
      }
      if (done > grew.bytes) grew = { bytes: done, at: Date.now() };
      else if (Date.now() - grew.at > 10 * 60_000) throw new Error("The download made no progress for 10 minutes on the machine (its connection to Hugging Face may be down). Pick another machine.");
      if (/curl: \(22\).*(401|403)|403 Forbidden|401 Unauthorized/.test(tail)) throw new Error("Hugging Face refused the download (gated model). Accept the license on the model page and check your token.");
      if (/curl: \(\d+\)|wget: .*failed|No space left/.test(tail) && !/retry/i.test(tail)) throw new Error(`The download stopped on the machine: ${tail.trim().split("\n").pop()}`);
      const pct = total ? Math.min(99, Math.floor((done / total) * 100)) : 0;
      if (done0 === undefined) done0 = done;   // already there from an earlier try
      const rate = (done - done0) / Math.max(1, (Date.now() - t0) / 1000);
      this.set("download", "doing", `${pct}%  ${fmtEta((total - done) / rate)}`);
    }
    this.set("download", "done");
    await refreshDownloadCost(e);

    // Start the runtime on the machine, bound to 127.0.0.1 (reachable only through the tunnel).
    this.set("start", "doing");
    const rr = plan.recipe.runtimes[plan.runtime];
    let chatTemplatePath: string | undefined, pluginPath: string | undefined;
    if (plan.runtime === "vllm" && rr.chat_template?.startsWith("http")) {
      chatTemplatePath = `${wd}/tool_template.jinja`;
      await sshExec(remote, `curl -fsSL '${rr.chat_template}' -o ${chatTemplatePath} || wget -q '${rr.chat_template}' -O ${chatTemplatePath}`);
    }
    if (plan.runtime === "vllm" && rr.reasoning_plugin) pluginPath = `${modelDir}/${rr.reasoning_plugin}`;
    const modelPath = plan.quant.format === "gguf" ? `${modelDir}/${files[0].path}` : modelDir;
    const launch = buildLaunch(plan.runtime, plan.recipe, plan.quant, {
      modelPath, servedName: e.name, host: "127.0.0.1", port: 8000, apiKey: e.apiKey, context: plan.context,
      gpus: plan.offer?.numGpus || 1, setup: "gpu", chatTemplatePath, pluginPath, gpuGen: gpuGen(plan.offer?.gpuName || ""),
      lora: lu && la ? { name: lu.name, path: la.file ? `${loraDir}/${la.file}` : `${loraDir}${la.dir ? `/${la.dir}` : ""}`, rank: vllmRank(la.rank) } : undefined,
      extraArgs: plan.advanced?.extraArgs,
    });
    this.check();
    if (lu) { e.lora = { name: lu.name, path: la ? `${la.repo}${la.dir ? `/${la.dir}` : ""}` : lu.localPath! }; upsertEndpoint(e); }
    if (plan.runtime === "llamacpp") launch.bin = "/app/llama-server";
    let cmd = commandString(launch);
    if (plan.advanced?.command) cmd = plan.advanced.command.replace("{model}", modelPath).replace("{port}", "8000").replace("{api_key}", e.apiKey).replace("{ctx}", String(plan.context)).replace("{name}", e.name);
    await sshWrite(remote, `${wd}/start.sh`, startScript(wd, cmd, plan.runtime), "700");
    await sshExec(remote, `cd ${wd} && (nohup sh start.sh > runtime.log 2>&1 < /dev/null &)`);
    e.command = cmd;
    upsertEndpoint(e);
    // The exact process start.sh became (a pattern search would also match this check's own command).
    const alive = async () => {
      const o = await sshExec(remote, `kill -0 "$(cat ${wd}/runtime.pid 2>/dev/null)" 2>/dev/null && echo up || echo down`, 30_000).catch(() => "up");
      return !o.includes("down");
    };
    const started = Date.now();
    for (;;) {
      this.check();
      await sleep(6000);
      const code = await sshExec(remote, `curl -s -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer ${e.apiKey}' http://127.0.0.1:8000/v1/models || echo 000`).catch(() => "000");
      if (code.trim() === "200") break;
      // Keep the machine's runtime log on this computer too, so the error can be read after the machine is gone.
      const fail = async (why: string) => {
        const text = await sshExec(remote, `tail -n 200 ${wd}/runtime.log`).catch(() => "");
        const local = logFile(`${e.id}-runtime.log`);
        try { fs.writeFileSync(local, text); } catch { /* ignore */ }
        const out = logTail(text);
        return new RuntimeFailed(`${why}${out.length ? `: ${tidy(out.join("\n"))}` : ""}`, out, local);
      };
      if (!(await alive())) throw await fail("The runtime stopped while loading the model");
      if (Date.now() - started > 60 * 60_000) throw await fail("The runtime did not become ready within an hour");
      this.set("start", "doing", `loading the model (${Math.round((Date.now() - started) / 60_000)} min)`);
    }
    this.set("start", "done");
    await sleep(3000); // give the keeper's tunnel a moment
    await this.testStep();
  }

  /** Download the chosen LoRA adapter next to the models (Local). */
  private async loraLocal(): Promise<LaunchOptions["lora"]> {
    const u = planLora(this.plan);
    if (!u) return undefined;
    if (u.localPath) { this.endpoint.lora = { name: u.name, path: u.localPath }; return { name: u.name, path: u.localPath }; }
    const a = u.adapter!;
    const dest = path.join(P.models, `lora__${a.repo.replace("/", "__")}`);
    this.set("download", "doing", "LoRA adapter");
    await downloadFiles(a.files.map((f) => ({ ...f, url: adapterUrl(a, f.path) })), dest, () => {}, () => this.cancelled);
    this.endpoint.lora = { name: u.name, path: `${a.repo}${a.dir ? `/${a.dir}` : ""}` };
    return { name: u.name, path: a.file ? path.join(dest, a.file) : path.join(dest, a.dir), rank: vllmRank(a.rank) };
  }

  private async testStep() {
    const e = this.endpoint;
    this.set("test", "doing");
    // vLLM and SGLang serve the adapter under its own name; test that name so a broken adapter shows up now.
    const model = e.lora && (e.runtime === "vllm" || e.runtime === "sglang") ? e.lora.name : e.name;
    let r = await testEndpoint(e.port, e.apiKey, model, e.context);
    if (!r.ok && /fetch failed|ECONNREFUSED|503/.test(r.error || "")) { await sleep(8000); r = await testEndpoint(e.port, e.apiKey, model, e.context); }
    if (!r.ok) throw new Error(`The endpoint did not pass its test: ${r.error}`);
    e.measured = { tps: r.tps, prefill: r.prefill };
    e.status = "running";
    upsertEndpoint(e);
    this.set("test", "done", r.toolsOk ? "tool calling works" : r.error);
  }
}

// ---------------------------------------------------------------- setups running in the background
/** Each setup runs inside this spunup window, so people can leave the Starting screen
 *  and serve another model while it carries on. */
export interface Job { id: string; launcher: Launcher; result?: Endpoint; finished: boolean }
const jobs = new Map<string, Job>();
export const jobEvents = new EventEmitter();

export function startJob(plan: Plan, cliPath: string): string {
  const id = `j-${Date.now().toString(36)}-${crypto.randomBytes(2).toString("hex")}`;
  const launcher = new Launcher(plan);
  const job: Job = { id, launcher, finished: false };
  jobs.set(id, job);
  launcher.on("update", () => jobEvents.emit("change"));
  launcher.run(cliPath).then((e) => {
    job.result = e;
    job.finished = true;
    if (launcher.cancelled) jobs.delete(id);
    jobEvents.emit("change");
  });
  return id;
}

export function getJob(id: string): Job | undefined { return jobs.get(id); }
export function dropJob(id: string): void { jobs.delete(id); jobEvents.emit("change"); }
/** Still setting up. */
export function activeJobs(): Job[] { return [...jobs.values()].filter((j) => !j.finished); }
/** Setting up, or failed and not looked at yet. */
export function visibleJobs(): Job[] { return [...jobs.values()].filter((j) => !j.finished || j.launcher.error); }

/** The script that starts the runtime on a rented machine. An SSH session does not get the
 *  container's own environment (PATH, LD_LIBRARY_PATH, CUDA settings), so it is copied from PID 1;
 *  llama.cpp's images also keep their libraries next to the binary in /app without a search path. */
export function startScript(wd: string, cmd: string, rt: RuntimeId): string {
  return `#!/bin/sh
cd ${wd}
if [ -r /proc/1/environ ]; then
  tr '\\0' '\\n' < /proc/1/environ > .container_env
  while IFS= read -r line; do
    case "$line" in PATH=*|LD_LIBRARY_PATH=*|LIBRARY_PATH=*|CUDA_*|NVIDIA_*|PYTHONPATH=*|VIRTUAL_ENV=*) export "$line" ;; esac
  done < .container_env
fi
${rt === "llamacpp" ? 'export LD_LIBRARY_PATH="/app${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' : ""}export HF_HUB_OFFLINE=1
echo $$ > runtime.pid
exec ${/^[A-Za-z_][A-Za-z0-9_]*=/.test(cmd) ? "env " : ""}${cmd}
`;
}

export function cloudImage(plan: Plan): string {
  const rr = plan.recipe.runtimes[plan.runtime];
  if (plan.runtime === "vllm") return rr.image || (rr.nightly ? IMAGES.vllmNightly : IMAGES.vllm);
  if (plan.runtime === "sglang") return IMAGES.sglang;
  return (plan.offer?.cudaMax || 0) >= 13 ? IMAGES.llamacppCuda13 : IMAGES.llamacppCuda12;
}

/** The last lines a runtime printed, without progress bars and blank lines (shown on the Starting screen). */
export function logTail(text: string, n = 10): string[] {
  const lines = text.replace(/\r/g, "\n").split("\n").map((l) => l.replace(/\x1b\[[0-9;]*m/g, "").trimEnd())
    .filter((l) => l.trim() && !/^\s*(\d+%\||Loading .*\d+%|Fetching \d+ files)/.test(l));
  // Start at the traceback / first error line near the end if there is one, so the cause is kept.
  const tail = lines.slice(-60);
  const i = tail.findIndex((l) => /Traceback|^\S*Error|error:|ERROR|failed|exception/i.test(l));
  const from = i >= 0 ? tail.slice(i) : tail;
  return (from.length > n ? [...from.slice(0, 3), "…", ...from.slice(-(n - 4))] : from).map((l) => l.slice(0, 200));
}

/** Keep the most telling error line from a runtime log. */
function tidy(log: string): string {
  const lines = log.split("\n").map((l) => l.trim()).filter(Boolean);
  const err = [...lines].reverse().find((l) => /error|exception|out of memory|not supported|failed/i.test(l));
  return (err || lines.pop() || "no details").slice(0, 300);
}

function toWslPath(p: string): string {
  const m = /^([A-Za-z]):\\(.*)$/.exec(p);
  return m ? `/mnt/${m[1].toLowerCase()}/${m[2].replace(/\\/g, "/")}` : p;
}

// ---------------------------------------------------------------- stop / delete / restart / reconnect
export async function stopEndpoint(e: Endpoint): Promise<Endpoint> {
  if (e.where === "local") {
    // After a reboot the saved PID can belong to another program: only stop it if it is still a runtime.
    if (e.pid && isRuntimeProcess(e.pid)) { try { process.kill(e.pid); } catch { /* already gone */ } }
    const n = { ...e, status: "stopped" as const };
    upsertEndpoint(n);
    return n;
  }
  const cur = await refreshDownloadCost(e);
  await provider(e.where).stop(e.instanceId!);
  const n = { ...cur, status: "stopped" as const, costBefore: costSoFar(cur) - (cur.oneOffCost || 0), stoppedAt: Date.now() };
  upsertEndpoint(n);
  return n;
}

/** Update the download charges from the provider's own count (Vast). Returns the endpoint as saved. */
export async function refreshDownloadCost(e: Endpoint): Promise<Endpoint> {
  if (e.where !== "vast" || !e.instanceId || e.status === "deleted") return e;
  try {
    const inst = await provider(e.where).get(e.instanceId);
    if (inst.downloadCost === undefined || Math.abs(inst.downloadCost - (e.oneOffCost || 0)) < 0.0005) return e;
    e.oneOffCost = inst.downloadCost;
    upsertEndpoint({ ...(getEndpoint(e.id) || e), oneOffCost: inst.downloadCost });
  } catch { /* offline: keep the last count */ }
  return e;
}

export async function deleteEndpoint(e: Endpoint, keepDisk = false): Promise<Endpoint> {
  if (e.where === "local") return stopEndpoint(e);
  const cur = await refreshDownloadCost(e);   // last count before the machine (and its numbers) are gone
  await provider(e.where).destroy(e.instanceId!, { volumeId: e.volumeId, keepDisk });
  const n = { ...cur, status: "deleted" as const, costBefore: costSoFar(cur) - (cur.oneOffCost || 0), stoppedAt: undefined,
    stoppedPricePerHour: keepDisk ? cur.stoppedPricePerHour : 0, diskKept: keepDisk && !!cur.volumeId, deletedAt: Date.now() };
  upsertEndpoint(n);
  return n;
}

/** Delete a disk kept with "Machine only". */
export async function deleteKeptDisk(e: Endpoint): Promise<Endpoint> {
  const prov = provider(e.where as "vast" | "runpod");
  if (!e.volumeId || !prov.deleteDisk) throw new Error("This disk can't be deleted from spunup. Delete it on the provider's website.");
  await prov.deleteDisk(e.volumeId);
  const n = { ...e, costBefore: costSoFar(e) - (e.oneOffCost || 0), diskKept: false, stoppedPricePerHour: 0 };
  upsertEndpoint(n);
  return n;
}

/** Restart a stopped endpoint with the same settings, port and key. */
export async function restartEndpoint(e: Endpoint, cliPath: string, onStatus: (s: string) => void): Promise<Endpoint> {
  if (e.where === "local") {
    if (!e.launch && !e.command) throw new Error("No saved start command for this model.");
    onStatus("Starting the runtime again");
    const fd = fs.openSync(e.logFile || logFile(`${e.id}.log`), "a");
    // The program and its arguments as saved, with no shell in between (cmd.exe would keep POSIX quotes).
    const [bin, args, env] = e.launch ? [e.launch.bin, e.launch.args, e.launch.env] : [IS_WIN ? "cmd.exe" : "sh", IS_WIN ? ["/c", e.command!] : ["-c", e.command!], {}];
    const child = spawn(bin, args, { detached: true, stdio: ["ignore", fd, fd], windowsHide: true, env: { ...process.env, ...env, ...(getKey("hf") ? { HF_TOKEN: getKey("hf")! } : {}) } });
    child.unref();
    const n = { ...e, pid: child.pid, status: "running" as const, startedAt: Date.now() };
    upsertEndpoint(n);
    if (n.runtime === "mlx") ensureKeeper(cliPath);
    return n;
  }
  const prov = provider(e.where);
  onStatus("Asking the provider to start the machine");
  await prov.start(e.instanceId!);
  let inst = await prov.get(e.instanceId!);
  const until = Date.now() + 10 * 60_000;
  while (!(inst.status === "running" && inst.ssh)) {
    if (Date.now() > until) throw new Error("The machine did not start. Its GPU may have been taken by someone else while it was stopped. Delete it and start a new one.");
    await sleep(5000);
    inst = await prov.get(e.instanceId!);
    onStatus(`Waiting for the machine (${inst.statusText || inst.status})`);
  }
  const n: Endpoint = { ...e, ssh: inst.ssh, status: "running", costBefore: costSoFar(e) - (e.oneOffCost || 0), startedAt: Date.now(), stoppedAt: undefined };
  upsertEndpoint(n);
  onStatus("Starting the runtime");
  await waitForSsh(inst.ssh!, 5 * 60_000, () => false);
  await sshExec(inst.ssh!, `cd ${prov.workDir} && (nohup sh start.sh > runtime.log 2>&1 < /dev/null &)`);
  ensureKeeper(cliPath);
  return n;
}
