"""Build spunup recipe files from the checked research data.

Inputs (all under research/raw, fetched 2026-11-06):
  models/, configs/   Hugging Face API metadata and config.json per model
  vllm-recipes/       clone of github.com/vllm-project/recipes
  search/picked.json  GGUF / MLX repos chosen per model, with real file sizes
  vllm_supported_models.md, llama-arch.cpp, mlx model list (in this script)

Output: ../recipes/models/<org>__<name>.json, ../recipes/families/<id>.json,
        ../recipes/index.json

Run: research/.venv/bin/python research/scripts/build_recipes.py
"""
import json, os, re, sys, glob, math, urllib.request
import yaml
sys.path.insert(0, os.path.dirname(__file__))
from families import FAMILIES, MODEL_OVERRIDES, ORG_FAMILY_OVERRIDE

R = os.path.join(os.path.dirname(__file__), "..", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "recipes")
CHECKED = "2026-11-06"

# Total / active parameters (billions) from model cards or vLLM recipes, for models whose
# config.json does not state them or whose checkpoint packs FP4/INT4 weights (safetensors
# element counts then undercount parameters).
MODEL_FACTS = {
    "deepseek-ai/DeepSeek-V4.1-Flash": (552, 16, "model card: 552B backbone; activated 8B / 16B"),
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": (552, 16, "model card: 552B backbone; activated 8B / 16B"),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (284, 13, "vLLM recipe DeepSeek-V4-Flash: 284B / 13B"),
    "deepseek-ai/DeepSeek-V4-Pro-0813": (1600, 49, "vLLM recipe DeepSeek-V4-Pro: 1600B / 49B"),
    "deepseek-ai/DeepSeek-V4-Pro": (1600, 49, "vLLM recipe: 1600B / 49B"),
    "moonshotai/Kimi-K3": (2800, 104, "model card: 2.8T total, 104B activated"),
    "moonshotai/Kimi-K3.1-Pro": (2800, 104, "model card: 104B activated; SGLang cookbook: 2.8T"),
    "moonshotai/Kimi-K2.6": (1000, 32, "vLLM recipe: 1T / 32B"),
    "moonshotai/Kimi-K2.7-Code": (1000, 32, "vLLM recipe: 1T / 32B"),
    "Qwen/Qwen3-Coder-Next": (80, 3, "model card: 80B total, 3B activated"),
    "zai-org/GLM-4.7-Flash": (30, 3, "model card: 30B-A3B"),
    "zai-org/GLM-4.5-Air": (106, 12, "model card: 106B total, 12B active"),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": (400, 17, "model name / Meta card: 17B active, 128 experts"),
    "MiniMaxAI/MiniMax-M3": (428, 23, "model card: ~428B total, ~23B activated"),
    "MiniMaxAI/MiniMax-M3.1-Flash": (428, 23, "model card: ~428B total, 23B activated"),
    "ornith-ai/Ornith-1.5-122B": (122, 10, "Qwen3.5-122B-A10B architecture (config identical)"),
    "ornith-ai/Ornith-1.5-397B-FP8": (397, 17, "Qwen3.5-397B-A17B architecture (config identical)"),
    "mistralai/Mistral-Large-4-1T-A52B": (1000, 52, "model card: ~1T total / ~52B active"),
    "XiaomiMiMo/MiMo-V2.6-Flash-RL": (309, 15, "vLLM recipe: 309B / 15B"),
    "XiaomiMiMo/MiMo-V2.6-Pro-RL": (1000, 42, "vLLM recipe: 1T / 42B"),
    "meta-models/Muse-Spark-Open-816B-A42B": (816, 42, "vLLM recipe: 816B / 42B"),
    "thinkingmachines/Inkling": (975, 41, "vLLM recipe Inkling-NVFP4: 975B / 41B"),
    "thinkingmachines/Inkling-Small": (276, 12, "vLLM recipe Inkling-Small-NVFP4: 276B / 12B"),
}
PACKED = {"U8", "I8", "I32"}

def load(p, default=None):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return default

_HF_CACHE_DIR = os.path.join(R, "hfcache")
def hf_get(url):
    import hashlib
    os.makedirs(_HF_CACHE_DIR, exist_ok=True)
    cf = os.path.join(_HF_CACHE_DIR, hashlib.sha1(url.encode()).hexdigest() + ".json")
    if os.path.exists(cf):
        return load(cf)
    v = _hf_get(url)
    if v is not None:
        with open(cf, "w") as f: json.dump(v, f)
    return v

def _hf_get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "spunup-research"}), timeout=60) as r:
            return json.load(r)
    except Exception:
        return None

# ---------------------------------------------------------------- support tables
def vllm_arch_table():
    """Architecture -> LoRA support, from vLLM docs/models/supported_models.md."""
    t = {}
    for line in open(os.path.join(R, "vllm_supported_models.md")):
        if not line.startswith("| `"): continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        archs = re.findall(r"`([A-Za-z0-9_]+)`", cols[0])
        lora = "✅" in cols[-2] if len(cols) >= 4 else False
        for a in archs: t[a] = lora
    return t
VLLM_ARCH = vllm_arch_table()

LLAMA_ARCHS = set(re.findall(r'\{ LLM_ARCH_[A-Z0-9_]+, +"([a-z0-9_.-]+)"', open(os.path.join(R, "llama-arch.cpp")).read()))

# mlx_lm/models/*.py on main, 2026-11-06 (github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models)
MLX_TYPES = set("""afm7 afmoe apertus baichuan_m1 bailing_hybrid bailing_moe bailing_moe_linear bailing_moe_v3 bitnet cohere cohere2 dbrx deepseek deepseek_v2 deepseek_v3 deepseek_v32 deepseek_v41 dots1 ernie4_5 ernie4_5_moe exaone exaone4 exaone_moe falcon_h1 flint gemma gemma2 gemma3 gemma3_text gemma3n gemma4 gemma4_text glm glm4 glm4_moe glm4_moe_lite glm_moe_dsa gpt2 gpt_bigcode gpt_neox gpt_oss gptj granite granitemoe granitemoehybrid helium hunyuan hunyuan_v1_dense internlm2 internlm3 jamba kimi_k25 kimi_k3 kimi_linear kimi_vl laguna lfm2 lfm2_moe llama llama4 llama4_text longcat_flash mamba mamba2 mellum mimo mimo_v2_flash minicpm minicpm3 minimax minimax_m3_vl ministral3 mistral3 mistral4 mixtral muse_glimmer muse_spark nanbeige nemotron nemotron_h olmo olmo2 olmo3 olmoe phi phi3 phimoe plamo plamo2 plamo3 qwen qwen2 qwen2_moe qwen2_vl qwen3 qwen3_5 qwen3_5_moe qwen3_moe qwen3_next qwen3_vl qwen3_vl_moe seed_oss smollm3 stablelm starcoder2 step3p5 telechat3""".split())

# ---------------------------------------------------------------- KV cache
FULL = {"full_attention", "attention", "global", "deepseek_sparse_attention", "indexed_attention", "full"}
SLIDE = {"sliding_attention", "sliding_window", "local", "chunked_attention"}

def kv_estimate(cfg):
    """Return dict: bytes per token (16-bit cache) for context-growing layers,
    fixed bytes (sliding-window layers at full window), and the method used.
    Mirrors src/core/kv.ts; keep the two in sync."""
    c = dict(cfg)
    for k in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(k), dict): c.update(cfg[k])
    lbt = c.get("layers_block_type") if isinstance(c.get("layers_block_type"), list) else None
    if lbt and not isinstance(c.get("layer_types"), list): c["layer_types"] = lbt
    L = c.get("num_hidden_layers") or c.get("n_layers") or (len(c["layer_types"]) if isinstance(c.get("layer_types"), list) else 0)
    H = c.get("num_attention_heads") or 1
    KV = c.get("num_key_value_heads") or c.get("num_attention_groups") or c.get("multi_query_group_num") or H
    hd = c.get("head_dim") or ((c.get("hidden_size") or 0) // H if H else 0)
    win = c.get("sliding_window") or c.get("sliding_window_size") or c.get("attention_chunk_size") or 0
    B = 2  # bytes per element, 16-bit cache
    method = []
    full_n, slide_n = L, 0
    lt = c.get("layer_types")
    lac = c.get("linear_attn_config") if isinstance(c.get("linear_attn_config"), dict) else None
    hop = c.get("hybrid_override_pattern")
    if lac and lac.get("full_attn_layers"):
        full_n, slide_n = len(lac["full_attn_layers"]), 0; method.append("full-attention layers from linear_attn_config")
    elif isinstance(lt, list) and lt:
        full_n = sum(1 for x in lt[:L or len(lt)] if str(x) in FULL)
        slide_n = sum(1 for x in lt[:L or len(lt)] if str(x) in SLIDE)
        method.append("layer_types")
    elif isinstance(hop, str):
        full_n = hop.count("*"); method.append("hybrid_override_pattern '*' layers")
    elif isinstance(c.get("local_layer_ids"), list):
        slide_n = len(c["local_layer_ids"]); full_n = L - slide_n; method.append("local_layer_ids")
    elif c.get("model_type") == "llama4" and c.get("attention_chunk_size"):
        full_n = L // 4; slide_n = L - full_n; method.append("Llama 4: 1 in 4 layers global, rest chunked")
    if c.get("num_kv_shared_layers"):
        share = c["num_kv_shared_layers"]; frac = (L - share) / L if L else 1
        full_n = round(full_n * frac); slide_n = round(slide_n * frac); method.append(f"{share} layers reuse earlier KV")
    if isinstance(c.get("kv_source_layers"), list):
        uniq = len(set(c["kv_source_layers"])); full_n = min(full_n, uniq); method.append("shared kv_source_layers")
    per_full = per_slide = 0
    if c.get("compress_ratios"):  # DeepSeek V4 / V4.1 compressed sparse attention
        ratios = c["compress_ratios"][:L]
        src = c.get("kv_source_layer_ids")
        idx = range(L) if not src else src
        per = sum((hd * B) / ratios[i] for i in idx if i < len(ratios) and ratios[i] > 0)
        method.append("DeepSeek V4 compressed KV: head_dim/ratio per layer" + (" (KV source layers only)" if src else ""))
        slide_bytes = sum(hd * B for i in range(L) if i < len(ratios) and ratios[i] == 0) * win
        return dict(bytes_per_token=round(per), fixed_bytes=round(slide_bytes), method="; ".join(method), estimate=True)
    if c.get("kv_lora_rank"):
        per_full = (c["kv_lora_rank"] + (c.get("qk_rope_head_dim") or 64)) * B
        if c.get("index_head_dim"): per_full += c["index_head_dim"]  # fp8 indexer keys
        method.append("MLA latent (kv_lora_rank + rope dim)")
    else:
        g_kv = c.get("num_global_key_value_heads") or KV
        g_hd = c.get("global_head_dim") or hd
        per_full = 2 * g_kv * g_hd * B
        sk = c.get("swa_num_key_value_heads") or KV; shd = c.get("swa_head_dim") or hd
        per_slide = 2 * sk * shd * B
        method.append("2 x kv_heads x head_dim")
        sac = c.get("sparse_attention_config") if isinstance(c.get("sparse_attention_config"), dict) else None
        if sac and sac.get("use_sparse_attention"):
            per_full += sac.get("sparse_index_dim", 0) * sac.get("sparse_num_index_heads", 0) * B
            method.append("plus sparse index keys")
    bpt = full_n * per_full
    fixed = slide_n * per_slide * (win or 0)
    if not win and slide_n:
        bpt += slide_n * per_slide; fixed = 0
    est = bool(set(method) - {"2 x kv_heads x head_dim", "layer_types"}) or not lt
    return dict(bytes_per_token=round(bpt), fixed_bytes=round(fixed), method="; ".join(method), estimate=est)

# ---------------------------------------------------------------- sizes
DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1, "I32": 4, "I64": 8, "F64": 8, "I16": 2, "BOOL": 1}
def weight_gb(meta):
    st = (meta or {}).get("safetensors") or {}
    p = st.get("parameters") or {}
    if not p: return None
    return round(sum(v * DT_BYTES.get(k, 2) for k, v in p.items()) / 1e9, 1)

_size_cache = {}
def repo_size_gb(repo):
    """Real download size: sum of the repo's .safetensors files (HF tree API)."""
    if repo in _size_cache: return _size_cache[repo]
    t = hf_get(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true") or []
    st = [f for f in t if isinstance(f, dict) and f.get("path", "").endswith(".safetensors")]
    root = [f for f in st if "/" not in f["path"]]
    # Mistral repos ship the weights twice (consolidated.safetensors + sharded model-0000x files); spunup
    # downloads only the sharded ones (hfdownload.filesFor), so only those count.
    if any(re.match(r"model-\d+-of-\d+\.safetensors$", f["path"]) for f in root):
        root = [f for f in root if not f["path"].startswith("consolidated")]
    tot = sum(f.get("size", 0) for f in (root or st))  # gpt-oss also ships original/ and metal/ copies
    m = hf_get(f"https://huggingface.co/api/models/{repo}?expand[]=gated") or {}
    _size_cache[repo] = (round(tot / 1e9, 1) if tot else None, m.get("gated"))
    return _size_cache[repo]

# ---------------------------------------------------------------- GGUF quant choice
GGUF_LADDER = [  # (label shown to user, candidate names in preference order, approx class)
    ("Q8", ["Q8_0", "UD-Q8_K_XL"]),
    ("Q6", ["Q6_K", "UD-Q6_K_XL", "UD-Q6_K"]),
    ("Q5", ["Q5_K_M", "UD-Q5_K_XL", "UD-Q5_K_M"]),
    ("Q4", ["Q4_K_M", "UD-Q4_K_XL", "UD-Q4_K_M", "Q4_K", "MXFP4_MOE", "MXFP4", "Q4_0", "Q4_K_S"]),
    ("Q3", ["Q3_K_M", "UD-Q3_K_XL", "UD-Q3_K_M", "IQ3_M", "UD-IQ3_S", "IQ3_S", "UD-IQ3_XXS", "IQ3_XXS"]),
    ("Q2", ["UD-Q2_K_XL", "Q2_K_L", "Q2_K", "UD-IQ2_M", "IQ2_M", "UD-IQ2_XXS", "IQ2_XXS"]),
    ("Q1", ["UD-IQ1_M", "UD-IQ1_S", "IQ1_M", "UD-Q1_0", "UD-TQ1_0"]),
]
def gguf_arch(repo, path):
    """Read general.architecture from the GGUF header (first 8 KB), cached."""
    import hashlib, struct
    cf = os.path.join(_HF_CACHE_DIR, "arch-" + hashlib.sha1((repo + path).encode()).hexdigest() + ".txt")
    if os.path.exists(cf): return open(cf).read().strip()
    try:
        req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{path}", headers={"User-Agent": "spunup-research", "Range": "bytes=0-16383"})
        with urllib.request.urlopen(req, timeout=60) as r: b = r.read()
        i = b.find(b"general.architecture")
        if i < 0: return ""
        j = i + len("general.architecture") + 4          # value type (uint32)
        n = struct.unpack("<Q", b[j:j + 8])[0]           # string length (uint64)
        arch = b[j + 8:j + 8 + n].decode()
        open(cf, "w").write(arch)
        return arch
    except Exception:
        return ""

def pick_ggufs(files, params_b):
    # Per step, the first file that is really smaller (>7%) than the step above: a size that is not
    # smaller is no use (gpt-oss ships MXFP4, so every "quant" is ~63 GB; DeepSeek-V4.1-Flash's UD-Q4_K_XL
    # and UD-Q3_K_XL are 98% and 95% of its Q8). Trying the step's other files then finds the real middle
    # sizes, e.g. its UD-IQ3_S at 83%, which taking only the first file name that exists used to skip.
    out = []
    for label, names in GGUF_LADDER:
        for n in names:
            if n not in files: continue
            f = files[n]
            if out and f["gb"] >= out[-1]["size_gb"] * 0.93: continue
            bits = round(f["gb"] * 8 / params_b, 2) if params_b else None
            # Label by the file's own quant name (Q4_K_M → Q4). Bits per weight mislabels small
            # models with big embeddings and models whose parameter count is off.
            out.append(dict(label=label, name=n, size_gb=f["gb"], parts=f["parts"], first_file=f["files"][0], bits=bits))
            break
    # MXFP4-native models (gpt-oss) only publish MXFP4
    if not out and files:
        n, f = sorted(files.items(), key=lambda x: x[1]["gb"])[0]
        out.append(dict(label="Q4", name=n, size_gb=f["gb"], parts=f["parts"], first_file=f["files"][0], bits=None))
    return out

def mlx_label(repo, size, params_b):
    """MLX repos say their precision in the name (…-4bit, …-q4-mlx, …-mxfp4, …-bf16); otherwise use bits per weight."""
    n = repo.lower()
    if re.search(r"mxfp4|nvfp4", n): return "MLX 4-bit"   # before "-q8": gpt-oss "MXFP4-Q8" is 4-bit weights
    m = re.search(r"(\d+)[-_]?bit", n) or re.search(r"[-_]q(\d)(?![\d])", n)
    if m: return f"MLX {m.group(1)}-bit"
    if re.search(r"bf16|fp16|f16", n): return "MLX 16-bit"
    if size and params_b:
        b = size * 8 / params_b
        return "MLX 16-bit" if b >= 12 else "MLX 8-bit" if b >= 6.5 else "MLX 6-bit" if b >= 5.5 else "MLX 4-bit"
    return "MLX"

PREC_LABEL = {"bf16": "BF16", "fp16": "FP16", "fp8": "FP8", "nvfp4": "NVFP4", "mxfp4": "MXFP4", "fp4": "FP4", "int4": "INT4", "int8": "INT8", "awq": "AWQ", "gptq": "GPTQ", "mxfp8": "MXFP8", "int2/4/8": "QAT"}

def family_for(mid, cfg_mt):
    if mid in ORG_FAMILY_OVERRIDE: return ORG_FAMILY_OVERRIDE[mid]
    org = mid.split("/")[0]
    for fid, f in FAMILIES.items():
        if org in f["orgs"]: return fid
    for fid, f in FAMILIES.items():
        if cfg_mt in f["model_types"]: return fid
    return None

# ---------------------------------------------------------------- per model
def vllm_recipe(mid):
    p = os.path.join(R, "vllm-recipes", "models", mid + ".yaml")
    if not os.path.exists(p): return None
    return yaml.safe_load(open(p)), "https://github.com/vllm-project/recipes/blob/main/models/" + mid + ".yaml"

def feat_args(feats, key):
    v = (feats or {}).get(key)
    return list(v.get("args") or []) if isinstance(v, dict) else []

def parse_parser(args, flag):
    if flag in args:
        i = args.index(flag)
        return args[i + 1] if i + 1 < len(args) else None
    return None

def build_model(mid, picked, popular=True):
    fname = mid.replace("/", "_")
    meta = load(os.path.join(R, "models", fname + ".json"), {}) or {}
    cfg = load(os.path.join(R, "configs", fname + ".json"), {}) or {}
    if "error" in cfg: cfg = {}
    tc = cfg.get("text_config") or {}
    mt = cfg.get("model_type")
    arch = (cfg.get("architectures") or [None])[0]
    fam_id = family_for(mid, mt)
    fam = FAMILIES[fam_id]
    ov = MODEL_OVERRIDES.get(mid, {})
    vr = vllm_recipe(mid)
    rec, rec_url = (vr if vr else (None, None))
    m = (rec or {}).get("model", {})
    feats = (rec or {}).get("features", {})
    sources = list(fam["sources"]) + [f"https://huggingface.co/{mid}"]
    if rec_url: sources.insert(0, rec_url)
    issues = list(ov.get("issues", []))

    # parameters
    total_b = meta.get("safetensors", {}).get("total") if meta.get("safetensors") else None
    params_b = round(total_b / 1e9, 1) if total_b else None
    def num(s):
        if s is None: return None
        s = str(s).strip().upper()
        mm = re.match(r"^([\d.]+)\s*([BT])", s)
        if not mm: return None
        v = float(mm.group(1)); return v * 1000 if mm.group(2) == "T" else v
    rec_params = num(m.get("parameter_count")); rec_active = num(m.get("active_parameters"))
    packed = bool(PACKED & set(((meta.get("safetensors") or {}).get("parameters") or {}).keys()))
    if rec_params and (packed or params_b is None): params_b = rec_params
    fact = MODEL_FACTS.get(mid)
    if fact:
        params_b, rec_active = float(fact[0]), float(fact[1]); sources.append("params: " + fact[2])
    name_active = re.search(r"-A(\d+(?:\.\d+)?)B", mid)
    active_b = rec_active or (float(name_active.group(1)) if name_active else None)
    E = (tc.get("n_routed_experts") or tc.get("num_experts") or tc.get("num_local_experts") or cfg.get("n_routed_experts")
         or cfg.get("num_experts") or cfg.get("num_local_experts") or cfg.get("moe_num_experts") or tc.get("moe_num_experts"))
    moe = bool(E) or m.get("architecture") == "moe"
    if not moe: active_b = params_b or rec_params
    if params_b is None: params_b = rec_params
    if moe and active_b is None:
        issues.append("Active parameter count not published in config; speed estimates use 10% of total.")
    ctx = (tc.get("max_position_embeddings") or cfg.get("max_position_embeddings") or m.get("context_length"))
    if m.get("context_length") and ctx and m["context_length"] < ctx and ctx > 2_000_000:
        ctx = m["context_length"]  # Llama 4 Scout config says 10M; recipe documents the served length
    long_ctx = None
    lc = (feats or {}).get("long_context")
    if isinstance(lc, dict):
        a = lc.get("args") or []
        if "--max-model-len" in a: long_ctx = int(a[a.index("--max-model-len") + 1])
    kv = kv_estimate(cfg) if cfg else None
    if kv is None or kv["bytes_per_token"] == 0:
        issues.append("KV cache size could not be computed from config.json; spunup uses a size-based fallback.")

    # ---------------- runtimes
    runtimes = {}
    # vLLM
    fv = dict(fam["vllm"]); fv.update(ov.get("vllm", {}))
    base_args = list(m.get("base_args") or [])
    tool = parse_parser(feat_args(feats, "tool_calling"), "--tool-call-parser")
    reason = parse_parser(feat_args(feats, "reasoning"), "--reasoning-parser")
    extra_reason = [a for a in feat_args(feats, "reasoning") if a not in ("--reasoning-parser", reason)]
    reason_plugin = parse_parser(feat_args(feats, "reasoning"), "--reasoning-parser-plugin")
    if "vllm" in ov and "tool" in ov["vllm"]: tool = ov["vllm"]["tool"]
    if "vllm" in ov and "reasoning" in ov["vllm"]: reason = ov["vllm"]["reasoning"]
    if tool is None and not rec: tool = fv.get("tool")
    if reason is None and not rec and "reasoning" in fv: reason = fv.get("reasoning")
    if rec and tool is None: tool = fv.get("tool")  # recipe lists no tool parser (Llama, Kimi Linear)
    chat_template = None
    ct_args = feat_args(feats, "tool_calling")
    if "--chat-template" in ct_args:
        ct = ct_args[ct_args.index("--chat-template") + 1]
        chat_template = "https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/" + ct if ct.startswith("examples/") else ct
    if fv.get("chat_template_url"): chat_template = fv["chat_template_url"]
    tok_args = [a for a in ct_args if a not in ("--enable-auto-tool-choice", "--tool-call-parser", tool, "--chat-template") and not str(a).startswith("examples/")]
    args = base_args + tok_args + [a for a in extra_reason if a not in ("--reasoning-parser-plugin", reason_plugin)]
    if not rec: args = list(fv.get("args", []))
    elif "args" in ov.get("vllm", {}):
        for a in ov["vllm"]["args"]:
            if a not in args: args.append(a)
    vllm_supported = arch in VLLM_ARCH or rec is not None
    hw = {}
    for k, v in ((rec or {}).get("hardware_overrides") or {}).items():
        if k in ("hopper", "blackwell", "nvidia", "ampere", "ada") and isinstance(v, dict) and (v.get("extra_args") or v.get("extra_env")):
            hw[k] = dict(args=v.get("extra_args") or [], env=v.get("extra_env") or {})
    lora_ok = VLLM_ARCH.get(arch, False)
    runtimes["vllm"] = dict(
        supported=vllm_supported, min_version=m.get("min_vllm_version"), nightly=bool(m.get("nightly_required")),
        image=(m.get("docker_image") if isinstance(m.get("docker_image"), str) else (m.get("docker_image") or {}).get("nvidia")),
        args=args, env=dict(m.get("base_env") or {}), tool_parser=tool, reasoning_parser=reason,
        reasoning_plugin=(fv.get("reasoning_plugin") or reason_plugin), chat_template=chat_template, hardware=hw,
        deps=[d.get("command") for d in ((rec or {}).get("dependencies") or []) if not d.get("optional")],
        lora=dict(supported=lora_ok, args=["--enable-lora", "--lora-modules", "{lora_name}={lora_path}", "--max-lora-rank", "{lora_rank}"]),
        caching="Prefix caching is on by default (vllm/config/cache.py: enable_prefix_caching=True).",
        long_context=(dict(args=lc.get("args"), env=lc.get("env") or {}) if isinstance(lc, dict) else None),
    )
    if runtimes["vllm"]["reasoning_plugin"]:
        issues.append(f"vLLM reasoning parser {reason} is a plugin file ({runtimes['vllm']['reasoning_plugin']}) shipped in the model repo; spunup downloads it next to the weights.")
    if runtimes["vllm"]["nightly"]:
        issues.append(f"vLLM support needs a nightly build (recipe min version: {m.get('min_vllm_version')}).")
    # SGLang
    fs = dict(fam["sglang"]); fs.update(ov.get("sglang", {}))
    runtimes["sglang"] = dict(supported=True, tool_parser=fs.get("tool"), reasoning_parser=fs.get("reasoning"),
        args=(["--trust-remote-code"] if "--trust-remote-code" in args else []),
        lora=dict(supported=lora_ok, args=["--enable-lora", "--lora-paths", "{lora_name}={lora_path}", "--max-lora-rank", "{lora_rank}"]),
        caching="Radix (prefix) cache is on by default.")
    # llama.cpp
    pk = picked.get(mid) or {}
    ggufs = pick_ggufs(pk.get("gguf_files") or {}, params_b)
    gg_repo = (pk.get("gguf_repos") or [None])[0]
    lc_arch = gguf_arch(gg_repo, ggufs[0]["first_file"]) if ggufs and gg_repo else ""
    llama_supported = bool(ggufs) and (not lc_arch or lc_arch in LLAMA_ARCHS)
    if ggufs and not llama_supported:
        issues.append(f"GGUF files exist ({gg_repo}) but use architecture \"{lc_arch}\", which mainline llama.cpp ({len(LLAMA_ARCHS)} architectures, checked {CHECKED}) does not support yet.")
        ggufs = []
    if llama_supported and gg_repo:
        sources.append(f"https://huggingface.co/{gg_repo}")
    runtimes["llamacpp"] = dict(supported=llama_supported, repo=gg_repo, min_build="b12110", gguf_arch=lc_arch or None,
        args=["--jinja", "--reasoning-format", "deepseek"],
        lora=dict(supported=llama_supported, args=["--lora", "{lora_path}"], note="Adapter must be in GGUF format (convert with llama.cpp's convert_lora_to_gguf.py)."),
        caching="--cache-prompt is on by default; spunup adds --cache-reuse 256 so shifted prompts reuse the cache.")
    # MLX
    mlx_repos = pk.get("mlx_repos") or []
    mlx_ok = bool(mlx_repos) and (mt in MLX_TYPES or (tc.get("model_type") in MLX_TYPES))
    runtimes["mlx"] = dict(supported=mlx_ok, repos=mlx_repos[:3], args=[],
        lora=dict(supported=mlx_ok, args=["--adapter-path", "{lora_path}"]),
        caching="Prompt cache is built in (mlx_lm.server --prompt-cache-size).",
        note="mlx_lm.server has no API key option; spunup puts its small key-checking proxy in front.")

    # ---------------- quantizations
    quants = []
    base_gb = repo_size_gb(mid)[0] or weight_gb(meta)
    base_prec = "bf16"
    dts = (meta.get("safetensors") or {}).get("parameters") or {}
    if dts:
        big = max(dts.items(), key=lambda x: x[1])[0]
        base_prec = {"F8_E4M3": "fp8", "U8": "fp4", "I8": "int8", "I32": "int4"}.get(big, "bf16")
        if mt == "gpt_oss": base_prec = "mxfp4"
    seen = set()
    def add_st(repo, prec, size, extra=None, gated=None):
        if repo in seen: return
        seen.add(repo)
        quants.append(dict(id=f"{prec}:{repo}", label=PREC_LABEL.get(prec, prec.upper()), format="safetensors", repo=repo,
                           size_gb=size, runtimes=["vllm", "sglang"], extra_args=extra or [], gated=gated))
    for vk, v in ((rec or {}).get("variants") or {}).items():
        if any(x in vk for x in ("ascend", "amd", "rocm", "xpu", "tpu", "reap")): continue
        repo = v.get("model_id") or mid
        if repo == mid:
            size = base_gb
        else:
            size, _g = repo_size_gb(repo)
        if size is None and v.get("vram_minimum_gb"): size = round(v["vram_minimum_gb"] / 1.2, 1)
        add_st(repo, v.get("precision", "bf16"), size, v.get("extra_args"))
    if mid not in seen and base_gb:
        add_st(mid, base_prec, base_gb)
    for g in ggufs:
        quants.append(dict(id=f"gguf:{g['name']}", label=g["label"], format="gguf", repo=gg_repo, file=g["first_file"], parts=g["parts"],
                           size_gb=g["size_gb"], bits=g["bits"], quant=g["name"], runtimes=["llamacpp"]))
    if mlx_ok:
        for repo in mlx_repos[:2]:
            size, _g = repo_size_gb(repo)
            lab = mlx_label(repo, size, params_b)
            quants.append(dict(id=f"mlx:{repo}", label=lab, format="mlx", repo=repo, size_gb=size, runtimes=["mlx"]))

    # one entry per (format, label): the recipe lists the preferred repo first
    seen_l, uq = set(), []
    for q in quants:
        k = (q["format"], q["label"])
        if k in seen_l: continue
        seen_l.add(k); uq.append(q)
    quants = uq

    # ---------------- setups
    v_ok = runtimes["vllm"]["supported"]
    gpu_best = "vllm" if v_ok and not runtimes["vllm"]["nightly"] else ("sglang" if fs.get("tool") else ("vllm" if v_ok else "llamacpp"))
    if not v_ok and not runtimes["sglang"]["supported"]: gpu_best = "llamacpp"
    gpu_others = [r for r in ("vllm", "sglang", "llamacpp") if r != gpu_best and runtimes[r]["supported"]]
    cpu_best = "llamacpp" if llama_supported else None
    mac_best = "llamacpp" if llama_supported else ("mlx" if mlx_ok else None)
    mac_others = [r for r in ("mlx", "llamacpp") if r != mac_best and runtimes[r]["supported"]]
    setups = dict(
        gpu=dict(best=gpu_best, others=gpu_others),
        cpu=dict(best=cpu_best, others=[]),
        mixed=dict(best=cpu_best, others=[], note=("MoE: expert layers go to RAM first (--n-cpu-moe)." if moe else "Dense: whole layers split between GPU and RAM (-ngl).")),
        mac=dict(best=mac_best, others=mac_others),
    )
    for s in setups.values():
        s["command"] = command_template(s["best"], runtimes, moe) if s["best"] else None

    if not llama_supported:
        issues.append("No GGUF build from a trusted publisher yet, so CPU, mixed GPU+RAM and Mac setups are not available.")
    if meta.get("gated"):
        issues.append("Gated on Hugging Face: accept the license on the model page, then use a Read token.")
    if fam.get("notes"): issues = fam["notes"] + issues

    return dict(
        schema=1, id=mid, name=mid.split("/")[1], family=fam_id, popular=popular, checked=CHECKED,
        architecture=arch, model_type=mt or tc.get("model_type"), moe=moe,
        params_b=params_b, active_b=active_b, layers=(tc.get("num_hidden_layers") or cfg.get("num_hidden_layers")),
        experts=E, max_context=ctx, max_context_extended=long_ctx,
        gated=bool(meta.get("gated")), license=(meta.get("cardData") or {}).get("license"),
        multimodal=bool(cfg.get("vision_config") or "ConditionalGeneration" in (arch or "")),
        kv=(dict(mb_per_1k=round(kv["bytes_per_token"] * 1024 / 1e6, 2), fixed_mb=round(kv["fixed_bytes"] / 1e6, 1),
                 dtype="16-bit", method=kv["method"], estimate=kv["estimate"]) if kv else None),
        quants=quants, runtimes=runtimes, setups=setups, known_issues=issues,
        benchmarks=[], sources=sorted(set(sources)),
    )

def command_template(rt, runtimes, moe):
    r = runtimes[rt]
    if rt == "vllm":
        parts = ["vllm serve {model} --host 127.0.0.1 --port {port} --api-key {api_key} --served-model-name {name}",
                 "--max-model-len {ctx} --gpu-memory-utilization 0.90 --tensor-parallel-size {gpus}"] + r["args"]
        if r["tool_parser"]: parts += ["--enable-auto-tool-choice --tool-call-parser " + r["tool_parser"]]
        if r["reasoning_parser"]: parts += ["--reasoning-parser " + r["reasoning_parser"]]
        if r["chat_template"]: parts += ["--chat-template {chat_template}"]
        return " ".join(parts)
    if rt == "sglang":
        parts = ["python3 -m sglang.launch_server --model-path {model} --host 127.0.0.1 --port {port} --api-key {api_key}",
                 "--served-model-name {name} --context-length {ctx} --tp-size {gpus}"] + r["args"]
        if r["tool_parser"]: parts += ["--tool-call-parser " + r["tool_parser"]]
        if r["reasoning_parser"]: parts += ["--reasoning-parser " + r["reasoning_parser"]]
        return " ".join(parts)
    if rt == "llamacpp":
        return ("llama-server -m {gguf} --host 127.0.0.1 --port {port} --api-key {api_key} --alias {name} -c {ctx} "
                "--jinja --reasoning-format deepseek --cache-reuse 256 -np 1 --fit on {gpu_layers}" + (" {cpu_moe}" if moe else ""))
    if rt == "mlx":
        return "mlx_lm.server --model {model} --host 127.0.0.1 --port {inner_port}  (behind spunup key proxy on {port})"

def build_family(fid, f, members):
    def tmpl(rt, runtimes):
        return command_template(rt, runtimes, False)
    rts = dict(
        vllm=dict(supported=True, tool_parser=f["vllm"].get("tool"), reasoning_parser=f["vllm"].get("reasoning"),
                  args=f["vllm"].get("args", []), chat_template=f["vllm"].get("chat_template_url"), env={},
                  lora=dict(args=["--enable-lora", "--lora-modules", "{lora_name}={lora_path}", "--max-lora-rank", "{lora_rank}"]),
                  caching="Prefix caching is on by default."),
        sglang=dict(supported=True, tool_parser=f["sglang"].get("tool"), reasoning_parser=f["sglang"].get("reasoning"),
                    args=(["--trust-remote-code"] if "--trust-remote-code" in f["vllm"].get("args", []) else []),
                    lora=dict(args=["--enable-lora", "--lora-paths", "{lora_name}={lora_path}"]), caching="Radix cache is on by default."),
        llamacpp=dict(supported=True, args=["--jinja", "--reasoning-format", "deepseek"], lora=dict(args=["--lora", "{lora_path}"]),
                      caching="--cache-prompt on by default; spunup adds --cache-reuse 256."),
        mlx=dict(supported=True, args=[], lora=dict(args=["--adapter-path", "{lora_path}"]), caching="Prompt cache built in."),
    )
    setups = dict(
        gpu=dict(best="vllm", others=["sglang", "llamacpp"], command=tmpl("vllm", rts)),
        cpu=dict(best="llamacpp", others=[], command=tmpl("llamacpp", rts)),
        mixed=dict(best="llamacpp", others=[], command=tmpl("llamacpp", rts) + " {cpu_moe}"),
        mac=dict(best="llamacpp", others=["mlx"], command=tmpl("llamacpp", rts)),
    )
    return dict(schema=1, id=fid, name=f["name"], hf_orgs=f["orgs"], model_types=f["model_types"], checked=CHECKED,
                runtimes=rts, setups=setups, notes=f.get("notes", []), members=sorted(members), sources=f["sources"],
                gguf_publishers=["ggml-org", "unsloth", "bartowski", "lmstudio-community"] + f["orgs"],
                mlx_publishers=["mlx-community", "lmstudio-community"] + f["orgs"])

def main():
    picked = load(os.path.join(R, "search", "picked.json"), {})
    ids = [l.strip() for l in open(os.path.join(R, "..", "candidates.txt")) if l.strip()]
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "families"), exist_ok=True)
    members = {k: [] for k in FAMILIES}
    index = []
    for mid in ids:
        r = build_model(mid, picked)
        members[r["family"]].append(mid)
        with open(os.path.join(OUT, "models", mid.replace("/", "__") + ".json"), "w") as f:
            json.dump(r, f, indent=1)
        meta = load(os.path.join(R, "models", mid.replace("/", "_") + ".json"), {}) or {}
        index.append(dict(id=mid, family=r["family"], params_b=r["params_b"], active_b=r["active_b"], moe=r["moe"],
                          max_context=r["max_context"], gated=r["gated"], popular=True, downloads=meta.get("downloads", 0),
                          file="models/" + mid.replace("/", "__") + ".json"))
        print(f"{mid:55s} fam={r['family']:9s} P={r['params_b']} A={r['active_b']} ctx={r['max_context']} kv={r['kv'] and r['kv']['mb_per_1k']}MB/1k "
              f"gpu={r['setups']['gpu']['best']} mac={r['setups']['mac']['best']} q={len(r['quants'])} tool={r['runtimes']['vllm']['tool_parser']}")
    for fid, f in FAMILIES.items():
        with open(os.path.join(OUT, "families", fid + ".json"), "w") as fh:
            json.dump(build_family(fid, f, members[fid]), fh, indent=1)
    with open(os.path.join(OUT, "index.json"), "w") as fh:
        # Families ordered by how much their models are downloaded (popular first, spec 11.2).
        fam_dl = {f: sum(m["downloads"] for m in index if m["family"] == f) for f in FAMILIES}
        fams = sorted(FAMILIES, key=lambda f: -fam_dl[f])
        index.sort(key=lambda m: (fams.index(m["family"]), -m["downloads"]))
        json.dump(dict(schema=1, checked=CHECKED, families=fams, family_names={f: FAMILIES[f]["name"] for f in fams}, models=index), fh, indent=1)

if __name__ == "__main__":
    main()
