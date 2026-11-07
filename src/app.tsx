import React, { useEffect, useState } from "react";
import { useApp, useInput } from "ink";
import { detectHardware, measureCpuBw, type Hardware } from "./core/hardware.js";
import { StartScreen } from "./screens/start.js";
import { KeysScreen } from "./screens/keys.js";
import { ServeFlow } from "./screens/serve.js";
import { EndpointScreen } from "./screens/endpoint.js";
import { StartingScreen } from "./screens/starting.js";
import { Busy } from "./ui/components.js";

export type Route =
  | { name: "start" }
  | { name: "keys" }
  | { name: "serve"; n?: number }       // n: a new number starts a fresh serve flow
  | { name: "starting"; id: string }     // a setup running in the background
  | { name: "endpoint"; id: string };

export interface Ctx {
  hw?: Hardware;
  cliPath: string;
  go: (r: Route) => void;
  quit: () => void;
}

export function App({ cliPath }: { cliPath: string }) {
  const { exit } = useApp();
  // Keep keyboard input on for the whole session. Between screens (a "Looking up…" moment) no screen
  // listens, Ink then pauses input, and on Windows it did not come back: arrows and Esc stopped working.
  useInput(() => {});
  const [route, setRoute] = useState<Route>({ name: "start" });
  const [hw, setHw] = useState<Hardware>();
  useEffect(() => {
    // Hardware check runs once, in the background, so the start screen appears at once.
    setTimeout(() => {
      const h = detectHardware();
      setHw(h);
      measureCpuBw(h).then((bw) => { if (bw) setHw({ ...h, cpu: { ...h.cpu, bwGb: bw } }); });
    }, 10);
  }, []);
  const ctx: Ctx = { hw, cliPath, go: setRoute, quit: () => exit() };
  switch (route.name) {
    case "start": return <StartScreen ctx={ctx} />;
    case "keys": return <KeysScreen ctx={ctx} onDone={() => setRoute({ name: "start" })} />;
    case "serve": return hw ? <ServeFlow key={route.n || 0} ctx={ctx} /> : <Busy text="Checking this computer's hardware…" />;
    case "starting": return <StartingScreen key={route.id} ctx={ctx} jobId={route.id} />;
    case "endpoint": return <EndpointScreen ctx={ctx} id={route.id} />;
  }
}
