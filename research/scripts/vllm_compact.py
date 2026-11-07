import yaml,sys,json
for f in sys.argv[1:]:
    d=yaml.safe_load(open(f)); m=d.get('model',{}); fe=d.get('features',{}) or {}
    def a(k):
        v=fe.get(k); return ' '.join(v.get('args') or []) if isinstance(v,dict) else None
    tc=a('tool_calling'); rp=a('reasoning')
    vs=[]
    for k,v in (d.get('variants') or {}).items():
        if any(x in k for x in ('ascend','amd','rocm','xpu','tpu')): continue
        vs.append(f"{v.get('precision')}:{v.get('model_id','=')}:{v.get('vram_minimum_gb')}" + (f"[{' '.join(v.get('extra_args'))}]" if v.get('extra_args') else ''))
    hw={k:' '.join(v.get('extra_args') or []) for k,v in (d.get('hardware_overrides') or {}).items() if k in('hopper','blackwell','nvidia','ampere','ada') and isinstance(v,dict)}
    print(f"{m.get('model_id')} | vllm>={m.get('min_vllm_version')}{' NIGHTLY' if m.get('nightly_required') else ''} img={json.dumps(m.get('docker_image'))} | {m.get('architecture')} {m.get('parameter_count')}/{m.get('active_parameters')} ctx={m.get('context_length')} | base={' '.join(m.get('base_args') or [])} env={m.get('base_env') or ''} | TOOL={tc} | REASON={rp} | other_feats={[k for k in fe if k not in('tool_calling','reasoning','long_context','text_only','encoder_parallel')]} | V={'; '.join(vs)} | HW={hw or ''} | deps={[x.get('command') for x in (d.get('dependencies') or []) if not x.get('optional')]}")
