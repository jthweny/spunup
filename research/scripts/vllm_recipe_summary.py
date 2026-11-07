import yaml,sys,glob,json
for f in sys.argv[1:]:
    d=yaml.safe_load(open(f))
    m=d.get('model',{}); feats=d.get('features',{}) or {}
    out={'id':m.get('model_id'),'min':m.get('min_vllm_version'),'nightly':m.get('nightly_required'),'docker':m.get('docker_image'),
      'arch':m.get('architecture'),'params':m.get('parameter_count'),'active':m.get('active_parameters'),'ctx':m.get('context_length'),
      'base_args':m.get('base_args'),'base_env':m.get('base_env'),
      'feat':{k:(v.get('args'),v.get('env')) for k,v in feats.items() if isinstance(v,dict)},
      'optin':d.get('opt_in_features'),
      'variants':{k:{kk:vv for kk,vv in v.items() if kk in('model_id','precision','vram_minimum_gb','extra_args','extra_env')} for k,v in (d.get('variants') or {}).items()},
      'hw':d.get('hardware_overrides'),'deps':[x.get('command') for x in (d.get('dependencies') or [])],'kvoff':d.get('kv_offload_support')}
    print(json.dumps(out,separators=(',',':')))
