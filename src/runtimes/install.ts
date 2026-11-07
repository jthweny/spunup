import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawn } from "node:child_process";
import { runSync, restoreRawMode } from "../core/proc.js";
import { P, IS_WIN, IS_MAC, IS_LINUX, exe } from "../core/paths.js";
import { download } from "../core/net.js";
import { wslInfo, type Hardware, type WslInfo } from "../core/hardware.js";
import type { RuntimeId } from "../core/store.js";
import { VERSIONS } from "./index.js";

export type Progress = (msg: string, pct?: number) => void;

const GH = "https://github.com/ggml-org/llama.cpp/releases/download";

/** Pick the llama.cpp build for this computer. Asset names checked on release b12167 / b12110. */
export function llamaAssets(hw: Hardware): { files: string[]; flavor: string } {
  const b = VERSIONS.llamacppBuild;
  const arm = os.arch() === "arm64";
  const nv = hw.gpus.find((g) => g.vendor === "nvidia");
  const amd = hw.gpus.find((g) => g.vendor === "amd");
  const intel = hw.gpus.find((g) => g.vendor === "intel");
  const cuda = Number(nv?.cuda || 0);
  if (IS_MAC) return { files: [`llama-${b}-bin-macos-${arm ? "arm64" : "x64"}.tar.gz`], flavor: arm ? "Metal" : "CPU" };
  if (IS_WIN) {
    if (arm) return { files: [`llama-${b}-bin-win-cpu-arm64.zip`], flavor: "CPU" };
    if (nv && cuda >= 13.4) return { files: [`llama-${b}-bin-win-cuda-13.4-x64.zip`, "cudart-llama-bin-win-cuda-13.4-x64.zip"], flavor: "CUDA 13" };
    if (nv && cuda >= 12.4) return { files: [`llama-${b}-bin-win-cuda-12.4-x64.zip`, "cudart-llama-bin-win-cuda-12.4-x64.zip"], flavor: "CUDA 12" };
    if (nv || amd || intel) return { files: [`llama-${b}-bin-win-vulkan-x64.zip`], flavor: "Vulkan" };
    return { files: [`llama-${b}-bin-win-cpu-x64.zip`], flavor: "CPU" };
  }
  // Linux: the cudart- bundles carry the CUDA runtime, so no CUDA toolkit is needed.
  if (nv && cuda >= 13.4) return { files: [`cudart-llama-${b}-bin-ubuntu-cuda-13.4-${arm ? "arm64" : "x64"}.tar.gz`], flavor: "CUDA 13" };
  if (nv && cuda >= 12.8 && !arm) return { files: [`cudart-llama-${b}-bin-ubuntu-cuda-12.8-x64.tar.gz`], flavor: "CUDA 12" };
  if (amd && fs.existsSync("/opt/rocm") && !arm) return { files: [`llama-${b}-bin-ubuntu-rocm-10.0-x64.tar.gz`], flavor: "ROCm" };
  if (nv || amd || intel) return { files: [`llama-${b}-bin-ubuntu-vulkan-${arm ? "arm64" : "x64"}.tar.gz`], flavor: "Vulkan" };
  return { files: [`llama-${b}-bin-ubuntu-${arm ? "arm64" : "x64"}.tar.gz`], flavor: "CPU" };
}

function findFile(dir: string, name: string): string | undefined {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isFile() && e.name === name) return p;
    if (e.isDirectory()) { const f = findFile(p, name); if (f) return f; }
  }
  return undefined;
}

function extract(archive: string, dest: string): void {
  fs.mkdirSync(dest, { recursive: true });
  // tar ships with macOS, Linux and Windows 10+ (bsdtar, which also reads .zip).
  const r = runSync("tar", ["-xf", archive, "-C", dest], { stdio: "ignore" });
  if (r.status !== 0) throw new Error(`Could not unpack ${path.basename(archive)}.`);
}

// Two setups at once (spunup can serve several models) share one install instead of racing on its folder.
const inFlight = new Map<string, Promise<string>>();
function shared(key: string, progress: Progress, run: () => Promise<string>): Promise<string> {
  let p = inFlight.get(key);
  if (p) progress("Waiting for the other setup to finish installing it");
  else { p = run().finally(() => inFlight.delete(key)); inFlight.set(key, p); }
  return p;
}

export function installLlamaCpp(hw: Hardware, progress: Progress): Promise<string> {
  return shared("llamacpp", progress, async () => {
    const bin = await installLlamaCppNow(hw, progress);
    if (IS_LINUX) await ensureLinuxLibs(bin, progress);
    return bin;
  });
}

/** System libraries llama.cpp's Linux builds link against that minimal installs (cloud/server
 *  Ubuntu images) leave out, with the package that provides each. */
const LIB_PACKAGES: Record<string, Record<string, string>> = {
  "libgomp.so.1": { "apt-get": "libgomp1", dnf: "libgomp", yum: "libgomp", zypper: "libgomp1", pacman: "gcc-libs", apk: "libgomp" },
};

/** Check llama-server can load its libraries; install a missing one when the computer allows it
 *  without a password (root or passwordless sudo), otherwise say exactly what to type. */
export async function ensureLinuxLibs(bin: string, progress: Progress): Promise<void> {
  const missing = (runSync("ldd", [bin], { encoding: "utf8" }).stdout || "").split("\n")
    .map((l) => /^\s*(\S+) => not found/.exec(l)?.[1]).filter((x): x is string => !!x);
  if (!missing.length) return;
  const pm = ["apt-get", "dnf", "yum", "zypper", "pacman", "apk"].find((m) => runSync("sh", ["-c", `command -v ${m}`]).status === 0);
  const pkgs = [...new Set(missing.map((l) => (pm && LIB_PACKAGES[l]?.[pm]) || ""))];
  const known = pm && pkgs.every(Boolean);
  const installCmd = !known ? "" : pm === "apt-get" ? `apt-get install -y ${pkgs.join(" ")}` : pm === "pacman" ? `pacman -S --noconfirm ${pkgs.join(" ")}`
    : pm === "apk" ? `apk add ${pkgs.join(" ")}` : `${pm} install -y ${pkgs.join(" ")}`;
  const root = process.getuid?.() === 0;
  const sudoOk = root || runSync("sudo", ["-n", "true"]).status === 0;
  if (known && sudoOk) {
    progress(`Installing ${pkgs.join(", ")} (a system library llama.cpp needs)`);
    const cmd = `${root ? "" : "sudo -n "}${pm === "apt-get" ? `env DEBIAN_FRONTEND=noninteractive ${installCmd}` : installCmd}`;
    const r = await new Promise<number>((res) => spawn("sh", ["-c", pm === "apt-get" ? `${root ? "" : "sudo -n "}apt-get update -qq >/dev/null 2>&1; ${cmd}` : cmd], { stdio: "ignore" }).on("close", (c) => res(c ?? 1)));
    const still = (runSync("ldd", [bin], { encoding: "utf8" }).stdout || "").includes("not found");
    if (r === 0 && !still) return;
  }
  throw new Error(known
    ? `llama.cpp needs a system library this computer does not have (${missing.join(", ")}). Install it by typing: sudo ${installCmd} — then start again.`
    : `llama.cpp needs system libraries this computer does not have: ${missing.join(", ")}. Install them with your package manager, then start again.`);
}

async function installLlamaCppNow(hw: Hardware, progress: Progress): Promise<string> {
  const { files, flavor } = llamaAssets(hw);
  const dir = path.join(P.runtimes, `llama.cpp-${VERSIONS.llamacppBuild}-${flavor.replace(/\s/g, "").toLowerCase()}`);
  // Done only when every file was unpacked (Windows CUDA builds come in two zips).
  const done = path.join(dir, ".spunup-ok");
  const existing = fs.existsSync(done) && findFile(dir, exe("llama-server"));
  if (existing) return existing;
  fs.mkdirSync(dir, { recursive: true });
  for (const f of files) {
    const tmp = path.join(P.runtimes, f);
    progress(`Downloading llama.cpp (${flavor})`, 0);
    await download(`${GH}/${VERSIONS.llamacppBuild}/${f}`, tmp, (d, t) => t && progress(`Downloading llama.cpp (${flavor})`, Math.round((d / t) * 100)));
    progress("Unpacking llama.cpp");
    extract(tmp, dir);
    fs.rmSync(tmp, { force: true });
  }
  const bin = findFile(dir, exe("llama-server"));
  if (!bin) throw new Error("llama.cpp was downloaded but llama-server was not found inside it.");
  if (!IS_WIN) fs.chmodSync(bin, 0o755);
  if (IS_WIN) {
    // CUDA runtime DLLs must sit next to llama-server.exe.
    const binDir = path.dirname(bin);
    for (const e of fs.readdirSync(dir)) if (/\.dll$/i.test(e) && !fs.existsSync(path.join(binDir, e))) fs.copyFileSync(path.join(dir, e), path.join(binDir, e));
  }
  fs.writeFileSync(done, VERSIONS.llamacppBuild);
  return bin;
}

// ------------------------------------------------------------------ uv (private Python)
function uvTarget(): string {
  const arm = os.arch() === "arm64";
  if (IS_MAC) return arm ? "aarch64-apple-darwin" : "x86_64-apple-darwin";
  if (IS_WIN) return arm ? "aarch64-pc-windows-msvc" : "x86_64-pc-windows-msvc";
  return arm ? "aarch64-unknown-linux-gnu" : "x86_64-unknown-linux-gnu";
}

export async function ensureUv(progress: Progress): Promise<string> {
  const bin = path.join(P.bin, exe("uv"));
  if (fs.existsSync(bin)) return bin;
  const t = uvTarget();
  const name = `uv-${t}.${IS_WIN ? "zip" : "tar.gz"}`;
  const tmp = path.join(P.runtimes, name);
  progress("Downloading uv (private Python installer)", 0);
  await download(`https://github.com/astral-sh/uv/releases/download/${VERSIONS.uv}/${name}`, tmp, (d, tt) => tt && progress("Downloading uv", Math.round((d / tt) * 100)));
  const dir = path.join(P.runtimes, "uv");
  extract(tmp, dir);
  fs.rmSync(tmp, { force: true });
  const found = findFile(dir, exe("uv"));
  if (!found) throw new Error("uv download did not contain the uv program.");
  fs.copyFileSync(found, bin);
  if (!IS_WIN) fs.chmodSync(bin, 0o755);
  return bin;
}

function runLogged(cmd: string, args: string[], env: Record<string, string>, onLine: (l: string) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { env: { ...process.env, ...env }, windowsHide: true });
    p.on("exit", restoreRawMode);
    const feed = (b: Buffer) => b.toString().split(/\r?\n/).filter(Boolean).forEach(onLine);
    p.stdout.on("data", feed);
    p.stderr.on("data", feed);
    p.on("error", reject);
    p.on("close", (c) => (c === 0 ? resolve() : reject(new Error(`${path.basename(cmd)} exited with code ${c}`))));
  });
}

/** Python runtimes: one private environment each, with uv's own Python 3.12. */
export function installPython(rt: Exclude<RuntimeId, "llamacpp">, opts: { nightly?: boolean; deps?: string[] }, progress: Progress): Promise<string> {
  return shared(`${rt}${opts.nightly ? "-nightly" : ""}`, progress, () => installPythonNow(rt, opts, progress));
}

async function installPythonNow(rt: Exclude<RuntimeId, "llamacpp">, opts: { nightly?: boolean; deps?: string[] }, progress: Progress): Promise<string> {
  const uv = await ensureUv(progress);
  const ver = rt === "vllm" ? (opts.nightly ? "nightly" : VERSIONS.vllm) : rt === "sglang" ? VERSIONS.sglang : VERSIONS.mlxLm;
  const venv = path.join(P.runtimes, `${rt}-${ver}`);
  const binDir = path.join(venv, IS_WIN ? "Scripts" : "bin");
  const entry = { vllm: "vllm", sglang: "python3", mlx: "mlx_lm.server" }[rt];
  const done = path.join(venv, ".spunup-ok");
  if (fs.existsSync(done)) return path.join(binDir, exe(entry));
  const env = { UV_PYTHON_INSTALL_DIR: path.join(P.runtimes, "python"), VIRTUAL_ENV: venv };
  progress(`Setting up Python for ${rt}`);
  await runLogged(uv, ["venv", "--python", "3.12", venv], env, () => {});
  const pkgs = rt === "vllm"
    ? (opts.nightly ? ["-U", "vllm", "--pre", "--extra-index-url", "https://wheels.vllm.ai/nightly/cu130"] : [`vllm==${VERSIONS.vllm}`])
    : rt === "sglang" ? ["--prerelease=allow", `sglang==${VERSIONS.sglang}`] : [`mlx-lm==${VERSIONS.mlxLm}`];
  progress(`Installing ${rt} (first time only, this can take a few minutes)`);
  let lines = 0;
  await runLogged(uv, ["pip", "install", "--python", path.join(binDir, exe("python")), ...pkgs], env, () => progress(`Installing ${rt}`, Math.min(95, ++lines)));
  for (const d of opts.deps || []) {
    const m = /uv pip install (.+)$/.exec(d);
    if (m) await runLogged(uv, ["pip", "install", "--python", path.join(binDir, exe("python")), ...m[1].replace(/"/g, "").split(/\s+/)], env, () => {});
  }
  fs.writeFileSync(done, new Date().toISOString());
  return path.join(binDir, exe(entry));
}

// ------------------------------------------------------------------ Windows: WSL for vLLM / SGLang
export function wslReady(hw: Hardware): boolean {
  return hw.wsl?.state === "ready";
}

/** Sets up what WSL still lacks.
 *  - off: switching the Windows feature on needs admin rights (Windows asks) and a restart, because
 *    Windows only loads the virtual machine layer WSL runs on while it boots.
 *  - nodistro: installs Ubuntu. No admin, no restart. --no-launch skips the window that asks for a
 *    Linux user name; spunup's commands run without one. */
export async function startWslInstall(state: WslInfo["state"] | undefined, onLine: (l: string) => void = () => {}): Promise<{ ok: boolean; restart?: boolean; noVm?: boolean; message: string }> {
  if (state === "off" || !state) {
    const r = runSync("powershell.exe", ["-NoProfile", "-Command", "Start-Process wsl.exe -ArgumentList '--install','-d','Ubuntu','--no-launch' -Verb RunAs -Wait"], { encoding: "utf8" });
    if (r.status !== 0) return { ok: false, message: "WSL setup was cancelled or failed. You can run `wsl --install` yourself in an admin PowerShell." };
    return { ok: true, restart: true, message: "WSL is switched on. Windows only finishes this when it restarts: restart your computer, then run spunup again and pick this runtime." };
  }
  if (state === "nogpu") return { ok: false, message: "Linux in WSL cannot see your NVIDIA GPU. Update the NVIDIA driver on Windows (nvidia.com), then run spunup again." };
  const out: string[] = [];
  const code = await new Promise<number>((resolve) => {
    const w = spawn("wsl.exe", ["--install", "-d", "Ubuntu", "--no-launch"], { windowsHide: true, env: { ...process.env, WSL_UTF8: "1" } });
    const take = (b: Buffer) => {
      // wsl.exe draws progress with carriage returns; show the latest line.
      for (const l of b.toString("utf8").split(/[\r\n]+/).map((x) => x.replace(/\0/g, "").trim()).filter(Boolean)) { out.push(l); onLine(l); }
    };
    w.stdout.on("data", take);
    w.stderr.on("data", take);
    w.on("error", () => resolve(-1));
    w.on("close", (c) => { restoreRawMode(); resolve(c ?? -1); });
  });
  const now = wslInfo();
  if (now.state === "ready") return { ok: true, message: "Ubuntu is installed in WSL and can see your GPU." };
  if (now.state === "nogpu") return { ok: false, message: "Ubuntu is installed, but it cannot see your NVIDIA GPU. Update the NVIDIA driver on Windows (nvidia.com), then try again." };
  // WSL 2 runs Linux in a small virtual machine. Seen on a Windows cloud VM whose host does not pass
  // virtualization through (RESEARCH.md §7): the Windows features are on, but no VM can start.
  if (out.some((l) => /HCS_E_HYPERV_NOT_INSTALLED|HCS_E_SERVICE_NOT_AVAILABLE|VIRTUALIZATION|0x80370102/i.test(l)))
    return { ok: false, noVm: true, message: "WSL cannot start here: it runs Linux in a small virtual machine, and this computer does not allow that. If this computer is itself a virtual machine (a cloud or company Windows machine), its host must turn on nested virtualization; on your own PC, turn on virtualization (Intel VT-x / AMD-V) in the BIOS. Until then, use llama.cpp." };
  return { ok: false, message: `Installing Ubuntu in WSL failed${code ? ` (exit ${code})` : ""}: ${out.slice(-2).join(" ") || "no output"}. You can run \`wsl --install -d Ubuntu\` yourself.` };
}

/** Run a Linux runtime inside WSL. Returns the command prefix. */
export function wslWrap(cmd: string, args: string[]): { bin: string; args: string[] } {
  // A leading ~/ stays outside the quotes (bash does not expand ~ inside them).
  const q = (a: string) => (a.startsWith("~/") ? `"$HOME"/'${a.slice(2).replace(/'/g, `'\\''`)}'` : `'${a.replace(/'/g, `'\\''`)}'`);
  const quoted = [cmd, ...args].map(q).join(" ");
  return { bin: "wsl.exe", args: ["-e", "bash", "-lc", quoted] };
}

export const WSL_SETUP_SCRIPT = (rt: "vllm" | "sglang") => `set -e
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/.spunup && cd ~/.spunup
[ -d ${rt}-env ] || uv venv --python 3.12 ${rt}-env
uv pip install --python ${rt}-env/bin/python ${rt === "vllm" ? `vllm==${VERSIONS.vllm}` : `--prerelease=allow sglang==${VERSIONS.sglang}`}
echo SPUNUP_OK`;

export function hasCommand(cmd: string): boolean {
  const r = runSync(IS_WIN ? "where" : "which", [cmd], { stdio: "ignore" });
  return r.status === 0;
}

export const PLATFORM_OK = { linux: IS_LINUX, mac: IS_MAC, win: IS_WIN };
