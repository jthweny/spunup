/** KV cache size from a model's config.json.
 *  Mirrors research/scripts/build_recipes.py:kv_estimate — keep them in sync.
 *  Returns megabytes per 1,024 tokens (16-bit cache) plus a fixed part for
 *  sliding-window layers, which stop growing once the window is full. */
export interface KvInfo { mbPer1k: number; fixedMb: number; method: string; estimate: boolean }

const FULL = new Set(["full_attention", "attention", "global", "deepseek_sparse_attention", "indexed_attention", "full"]);
const SLIDE = new Set(["sliding_attention", "sliding_window", "local", "chunked_attention"]);

export function kvFromConfig(cfg: any): KvInfo | undefined {
  if (!cfg || typeof cfg !== "object") return undefined;
  const c: any = { ...cfg };
  for (const k of ["text_config", "llm_config", "language_config"]) if (cfg[k] && typeof cfg[k] === "object") Object.assign(c, cfg[k]);
  if (Array.isArray(c.layers_block_type) && !Array.isArray(c.layer_types)) c.layer_types = c.layers_block_type;
  const L: number = c.num_hidden_layers || c.n_layers || (Array.isArray(c.layer_types) ? c.layer_types.length : 0);
  if (!L) return undefined;
  const H: number = c.num_attention_heads || 1;
  const KV: number = c.num_key_value_heads || c.num_attention_groups || c.multi_query_group_num || H;
  const hd: number = c.head_dim || (c.hidden_size ? Math.floor(c.hidden_size / H) : 128);
  const win: number = c.sliding_window || c.sliding_window_size || c.attention_chunk_size || 0;
  const B = 2;
  const method: string[] = [];
  let full = L, slide = 0;
  const lac = c.linear_attn_config;
  if (lac?.full_attn_layers?.length) { full = lac.full_attn_layers.length; method.push("full-attention layers from linear_attn_config"); }
  else if (Array.isArray(c.layer_types) && c.layer_types.length) {
    const lt = c.layer_types.slice(0, L).map(String);
    full = lt.filter((x: string) => FULL.has(x)).length;
    slide = lt.filter((x: string) => SLIDE.has(x)).length;
    method.push("layer_types");
  } else if (typeof c.hybrid_override_pattern === "string") { full = (c.hybrid_override_pattern.match(/\*/g) || []).length; method.push("hybrid pattern"); }
  else if (Array.isArray(c.local_layer_ids)) { slide = c.local_layer_ids.length; full = L - slide; method.push("local_layer_ids"); }
  else if (c.model_type === "llama4" && c.attention_chunk_size) { full = Math.floor(L / 4); slide = L - full; method.push("Llama 4 chunked"); }
  if (c.num_kv_shared_layers) { const f = (L - c.num_kv_shared_layers) / L; full = Math.round(full * f); slide = Math.round(slide * f); method.push("shared KV layers"); }
  if (Array.isArray(c.kv_source_layers)) { full = Math.min(full, new Set(c.kv_source_layers).size); method.push("kv_source_layers"); }

  if (Array.isArray(c.compress_ratios)) {
    const r: number[] = c.compress_ratios.slice(0, L);
    const idx: number[] = Array.isArray(c.kv_source_layer_ids) ? c.kv_source_layer_ids : r.map((_, i) => i);
    const per = idx.reduce((a, i) => a + (r[i] > 0 ? (hd * B) / r[i] : 0), 0);
    const fixed = r.reduce((a, x) => a + (x === 0 ? hd * B : 0), 0) * win;
    return { mbPer1k: round2((per * 1024) / 1e6), fixedMb: round1(fixed / 1e6), method: "DeepSeek V4 compressed KV", estimate: true };
  }
  let perFull: number, perSlide: number;
  if (c.kv_lora_rank) {
    perFull = (c.kv_lora_rank + (c.qk_rope_head_dim || 64)) * B + (c.index_head_dim || 0);
    perSlide = perFull;
    method.push("MLA latent");
  } else {
    perFull = 2 * (c.num_global_key_value_heads || KV) * (c.global_head_dim || hd) * B;
    perSlide = 2 * (c.swa_num_key_value_heads || KV) * (c.swa_head_dim || hd) * B;
    method.push("2 x kv_heads x head_dim");
    const sac = c.sparse_attention_config;
    if (sac?.use_sparse_attention) perFull += (sac.sparse_index_dim || 0) * (sac.sparse_num_index_heads || 0) * B;
  }
  let bpt = full * perFull;
  let fixed = slide * perSlide * win;
  if (!win && slide) { bpt += slide * perSlide; fixed = 0; }
  return { mbPer1k: round2((bpt * 1024) / 1e6), fixedMb: round1(fixed / 1e6), method: method.join("; "), estimate: method.length > 2 };
}

/** Rough KV when no config is available: scale from parameter count. */
export function kvFallback(paramsB: number): KvInfo {
  // Dense GQA models measure 5–17 MB per 1K tokens per billion params (Llama 3.3 70B: 335, Llama 3.1 8B: 134),
  // so 8 per billion is within about 2x for both; it errs high for small models, which is the safe side.
  return { mbPer1k: Math.max(16, Math.round(paramsB * 8)), fixedMb: 0, method: "size-based guess", estimate: true };
}

const round1 = (x: number) => Math.round(x * 10) / 10;
const round2 = (x: number) => Math.round(x * 100) / 100;
