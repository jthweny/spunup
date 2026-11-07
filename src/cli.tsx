#!/usr/bin/env node
import "./nodecheck.js";
import React from "react";
import { render } from "ink";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { ensureDirs } from "./core/paths.js";
import { detectBackground } from "./ui/theme.js";
import { initCurrency } from "./core/cost.js";
import { updateRecipes } from "./core/recipes.js";
import { runKeeper } from "./core/keeper.js";
import { fromNpx, installedVersion, installedCli, selfInstall } from "./core/selfinstall.js";

let cliPath = fileURLToPath(import.meta.url);
const pkg = createRequire(import.meta.url)("../package.json") as { version: string };
const arg = process.argv[2];

async function main() {
  ensureDirs();
  if (arg === "__keeper") return runKeeper();
  if (arg === "--version" || arg === "-v") { console.log(pkg.version); return; }
  if (arg === "--help" || arg === "-h") {
    console.log(`spunup ${pkg.version}\n\nType "spunup" to open the tool. Everything happens on its screens.\n\n  spunup                  open spunup\n  spunup update-recipes   download the newest model recipes now\n  spunup --version        show the version\n\nFiles: ~/.spunup`);
    return;
  }
  if (arg === "update-recipes") {
    const ok = await updateRecipes(true);
    console.log(ok ? "Recipes updated." : "Recipes are already up to date (or the recipe server could not be reached).");
    return;
  }
  if (!process.stdin.isTTY) {
    console.error("spunup needs an interactive terminal. Open a terminal window and type: spunup");
    process.exit(1);
  }
  if (fromNpx()) {
    if (installedVersion() !== pkg.version) {
      console.log("Setting up the spunup command (only this once)…");
      await selfInstall(pkg.version);
    }
    // The keeper outlives this window, so it runs from the installed copy, not npx's cache.
    cliPath = installedCli() || cliPath;
  }
  await Promise.all([detectBackground(), initCurrency()]);
  updateRecipes().catch(() => {});
  const { App } = await import("./app.js");
  // Start on a clean terminal: clear the screen and its scrollback, cursor to the top.
  if (process.stdout.isTTY) process.stdout.write("\x1b[2J\x1b[3J\x1b[H");
  const app = render(<App cliPath={cliPath} />, { exitOnCtrlC: true });
  await app.waitUntilExit();
  // Local runtimes and the keeper keep running in the background; spunup itself just closes.
  process.exit(0);
}

main().catch((e) => {
  console.error(`spunup hit a problem: ${(e as Error).message}`);
  process.exit(1);
});
