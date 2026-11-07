import React, { useEffect, useState } from "react";
import { Box, Text, useInput, usePaste } from "ink";
import { C, strongSpeed, softSpeed, fmtInt } from "./theme.js";

export interface Item<T = string> {
  value: T;
  label: string;
  hint?: string;          // dim text after the label
  after?: React.ReactNode; // straight after the hint (the label keeps its natural width)
  right?: React.ReactNode;
  separatorBefore?: boolean | string;   // a string is drawn as the separator line itself
  disabled?: boolean;
  color?: string;
  indent?: number;
}

/** Arrow keys to move, Enter to pick, Esc to go back. */
export function Select<T>({ items, onSelect, onBack, onKey, onMove, initial = 0, active = true, labelWidth }: {
  items: Item<T>[];
  onSelect: (v: T, i: number) => void;
  onBack?: () => void;
  onKey?: (input: string, index: number) => void;
  onMove?: (index: number) => void;   // the row the cursor is on, whenever it changes
  initial?: number;
  active?: boolean;
  labelWidth?: number;
}) {
  const firstOk = Math.max(0, items.findIndex((x) => !x.disabled));
  const start = Math.min(Math.max(initial, firstOk), Math.max(0, items.length - 1));
  const [idx, setIdx] = useState(start);
  // A new screen often reuses this component in the same place: a different list starts at its own first row.
  const sig = items.map((x) => String(x.value)).join("|");
  const [lastSig, setLastSig] = useState(sig);
  if (sig !== lastSig) { setLastSig(sig); setIdx(start); }
  const i = Math.min(idx, Math.max(0, items.length - 1));
  useEffect(() => { onMove?.(i); }, [i]);
  useInput((input, key) => {
    if (!items.length) { if (key.escape) onBack?.(); return; }
    const move = (d: number) => {
      let n = i;
      for (let k = 0; k < items.length; k++) {
        n = (n + d + items.length) % items.length;
        if (!items[n].disabled) break;
      }
      setIdx(n);
    };
    if (key.upArrow) move(-1);
    else if (key.downArrow) move(1);
    else if (key.return && !items[i].disabled) onSelect(items[i].value, i);
    else if (key.escape) onBack?.();
    else if (key.leftArrow) onKey?.("left", i);
    else if (key.rightArrow) onKey?.("right", i);
    else if (input) onKey?.(input, i);
  }, { isActive: active });
  return (
    <Box flexDirection="column">
      {items.map((it, k) => (
        <Box key={k} flexDirection="column">
          {it.separatorBefore && <Text dimColor>{typeof it.separatorBefore === "string" ? it.separatorBefore : "   " + "─".repeat(13)}</Text>}
          <Box>
            <Text color={k === i ? C.accent : undefined}>{" ".repeat(it.indent || 0)}{k === i ? " › " : "   "}</Text>
            <Box width={it.right ? labelWidth : undefined}>
              <Text color={it.disabled ? "gray" : it.color} bold={k === i && !it.disabled}>{it.label}</Text>
            </Box>
            {it.hint && <Text dimColor>{"  " + it.hint}</Text>}
            {it.after}
            {it.right}
          </Box>
        </Box>
      ))}
    </Box>
  );
}

/** Single-line text input. `mask` hides what is typed (for keys). */
export function TextInput({ onSubmit, onBack, mask, placeholder, initial = "", active = true, validate }: {
  onSubmit: (v: string) => void;
  onBack?: () => void;
  mask?: boolean;
  placeholder?: string;
  initial?: string;
  active?: boolean;
  validate?: (v: string) => string | undefined;
}) {
  const [v, setV] = useState(initial);
  const [err, setErr] = useState<string>();
  usePaste((text) => setV((x) => x + text.replace(/[\r\n]+/g, "").trim()), { isActive: active });
  useInput((input, key) => {
    if (key.escape) return onBack?.();
    if (key.return) {
      const e = validate?.(v);
      if (e) return setErr(e);
      return onSubmit(v);
    }
    if (key.backspace || key.delete) return setV((x) => x.slice(0, -1));
    if (key.ctrl && input === "u") return setV("");
    if (input && !key.ctrl && !key.meta) setV((x) => x + input.replace(/[\r\n]/g, ""));
  }, { isActive: active });
  const shown = mask ? (v.length > 8 ? v.slice(0, 4) + "•".repeat(Math.min(24, v.length - 8)) + v.slice(-4) : "•".repeat(v.length)) : v;
  return (
    <Box flexDirection="column">
      <Text>{shown || <Text dimColor>{placeholder || ""}</Text>}<Text color={C.accent}>_</Text></Text>
      {err && <Text color={C.warn}>{err}</Text>}
    </Box>
  );
}

/** Title + short summary of picks so far (spec 11: one question per screen). */
export function Screen({ title, note, summary, children, footer }: {
  title: string; note?: string; summary?: [string, string][]; children: React.ReactNode; footer?: string;
}) {
  return (
    <Box flexDirection="column" paddingX={1} paddingY={0}>
      {summary && summary.length > 0 && (
        <Box marginBottom={1}>
          <Text dimColor>{summary.map(([k, v]) => `${k}: ${v}`).join("  ·  ")}</Text>
        </Box>
      )}
      <Text bold>{title}</Text>
      {note && <Text dimColor>{note}</Text>}
      <Box marginTop={1} flexDirection="column">{children}</Box>
      <Box marginTop={1}><Text dimColor>{footer ?? "↑↓ move · Enter pick · Esc back"}</Text></Box>
    </Box>
  );
}

/** "38–52 tokens/s" in the strongest colour. */
export function Tps({ min, max }: { min?: number; max?: number }) {
  if (!max) return <Text dimColor>—</Text>;
  const a = Math.max(1, Math.round(min ?? max)), b = Math.max(1, Math.round(max));
  return <Text {...strongSpeed()}>{a === b ? `${b}` : `${a}–${b}`} tokens/s</Text>;
}

/** "prefill 1,800 tokens/s" — always less prominent than tokens/s. */
export function Prefill({ value, label = true }: { value?: number; label?: boolean }) {
  if (!value) return null;
  return <Text {...softSpeed()}>{label ? "prefill " : ""}{fmtInt(value)} tokens/s</Text>;
}

export function Spinner() {
  const frames = ["◐", "◓", "◑", "◒"];
  const [f, setF] = useState(0);
  React.useEffect(() => { const t = setInterval(() => setF((x) => (x + 1) % 4), 150); return () => clearInterval(t); }, []);
  return <Text color={C.accent}>{frames[f]}</Text>;
}

export function Busy({ text }: { text: string }) {
  return <Box paddingX={1}><Spinner /><Text> {text}</Text></Box>;
}

export function Err({ text }: { text: string }) {
  return <Text color={C.error}>{text}</Text>;
}
