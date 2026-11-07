# spunup recipe format

Recipes are plain JSON. They ship with the tool (`recipes/`) and can be updated from the
project's repo without a new release (`spunup update-recipes`, or automatically once a day).
Updated copies are kept in `~/.spunup/recipes/` and win when their `checked` date is newer.

Settings are chosen in this order; the first match wins (spec 5.1):

1. **User** — Advanced settings in the tool (extra flags, own start command, quant file, LoRA).
2. **Model** — `models/<org>__<name>.json`, hand-checked for a popular model.
3. **Family** — `families/<id>.json`, the general way to serve any model from that family.
4. **Generic** — built in (`src/core/recipes.ts`): SGLang and llama.cpp with tool and reasoning
   formats detected from the model's chat template, KV size from `config.json`.

A LoRA adapter is added on top of whichever layer is used (`runtimes.<rt>.lora.args`).

## `index.json`

```json
{ "schema": 1, "checked": "2026-11-06",
  "families": ["qwen", "gemma", ...],              // ordered by downloads: popular first
  "family_names": { "qwen": "Qwen", ... },
  "models": [ { "id": "Qwen/Qwen3.8-27B", "family": "qwen", "params_b": 27.8, "active_b": 27.8,
                "moe": false, "max_context": 262144, "gated": false, "popular": true,
                "downloads": 7845288, "file": "models/Qwen__Qwen3.8-27B.json" } ] }
```

## Model recipe

| Field | Meaning |
|---|---|
| `id`, `name`, `family`, `popular`, `checked` | Hugging Face id, display name, family id, listed as popular, date checked |
| `architecture`, `model_type` | from `config.json` |
| `moe`, `params_b`, `active_b`, `layers`, `experts` | size; `active_b` = parameters used per token (MoE) |
| `max_context`, `max_context_extended` | native context; extended = with the recipe's YaRN override |
| `gated`, `license` | gated on Hugging Face (needs a Read token and accepting the license) |
| `kv` | `mb_per_1k`: KV cache MB per 1,024 tokens (16-bit cache); `fixed_mb`: sliding-window layers at full window; `method`; `estimate: true` for architectures where the formula is approximate |
| `quants[]` | one per size: `label` (BF16, FP8, NVFP4, Q8…Q1, MLX 4-bit…), `format` (`safetensors`, `gguf`, `mlx`), `repo`, `file`/`parts` (GGUF), `size_gb` (real file sizes), `bits` (bits per weight), `runtimes`, `extra_args` |
| `runtimes.vllm` | `supported`, `min_version`, `nightly`, `image`, `args`, `env`, `tool_parser`, `reasoning_parser`, `reasoning_plugin`, `chat_template` (URL), `hardware` (per GPU generation: hopper/blackwell…), `deps`, `lora`, `caching`, `long_context` |
| `runtimes.sglang` | `supported`, `tool_parser`, `reasoning_parser`, `args`, `lora`, `caching` |
| `runtimes.llamacpp` | `supported`, `repo`, `gguf_arch`, `min_build`, `args`, `lora`, `caching` |
| `runtimes.mlx` | `supported`, `repos`, `args`, `lora`, `caching`, `note` |
| `setups.{gpu,cpu,mixed,mac}` | `best` runtime, `others`, `command` (start command template), `note` |
| `known_issues[]` | plain-English problems and fixes |
| `benchmarks[]` | measured speeds where published (`gpu`, `quant`, `tps`, `prefill`, `source`) |
| `sources[]` | links every setting came from |

Command templates use placeholders filled in by the tool: `{model} {port} {api_key} {name} {ctx}
{gpus} {gguf} {gpu_layers} {cpu_moe} {chat_template} {inner_port}`. Every template binds
`127.0.0.1` and requires the API key.

## Family recipe

`id`, `name`, `hf_orgs` (Hugging Face organisations watched for new releases), `model_types`,
`runtimes` (same shape as above, general settings), `setups`, `notes`, `members`,
`gguf_publishers` / `mlx_publishers` (whose quantized repos are trusted), `sources`.

## Regenerating

The recipe files are built from checked sources by `research/scripts/build_recipes.py`
(see `RESEARCH.md`). Hand-checked family settings and per-model overrides live in
`research/scripts/families.py`.
