import { vast } from "./vast.js";
import { runpod } from "./runpod.js";
import type { Provider } from "./types.js";
export type { Provider, Offer, Instance, RunpodCloud } from "./types.js";
export { NoCapacity } from "./types.js";

export function provider(id: "vast" | "runpod"): Provider {
  return id === "vast" ? vast : runpod;
}
