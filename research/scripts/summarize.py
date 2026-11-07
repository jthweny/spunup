import json,sys,os,glob
def g(c,*ks):
    for k in ks:
        if k in c and c[k] is not None: return c[k]
    return None
rows=[]
for f in sorted(glob.glob(sys.argv[1]+'/models/*.json')):
    name=os.path.basename(f)[:-5]
    try: m=json.load(open(f))
    except Exception: print(name,'BADMETA'); continue
    try: c=json.load(open(sys.argv[1]+'/configs/'+name+'.json'))
    except Exception: c={}
    if 'error' in c: c={}
    full=c
    tc=c.get('text_config') or c.get('llm_config') or c.get('language_config') or {}
    if tc: c={**c,**tc}
    st=m.get('safetensors') or {}
    tot=st.get('total')
    dt=st.get('parameters')
    arch=(full.get('architectures') or [''])[0]
    L=g(c,'num_hidden_layers','n_layers','num_layers')
    H=g(c,'num_attention_heads','n_heads')
    KV=g(c,'num_key_value_heads','n_kv_heads','multi_query_group_num')
    hd=g(c,'head_dim')
    hs=g(c,'hidden_size','d_model')
    if hd is None and hs and H: hd=hs//H
    ctx=g(c,'max_position_embeddings','max_sequence_length','seq_length','model_max_length')
    rs=g(c,'rope_scaling','rope_parameters')
    E=g(c,'n_routed_experts','num_experts','num_local_experts','moe_num_experts')
    topk=g(c,'num_experts_per_tok','moe_topk','num_experts_per_token','top_k')
    kvl=g(c,'kv_lora_rank'); rope=g(c,'qk_rope_head_dim')
    lt=g(c,'layer_types','hybrid_override_pattern','layers_block_type','full_attn_idxs','attn_type_list')
    if isinstance(lt,list):
        from collections import Counter; lt=dict(Counter(map(str,lt)))
    sw=g(c,'sliding_window')
    q=g(full,'quantization_config')
    q=q.get('quant_method') if isinstance(q,dict) else None
    print(f"{m.get('id')} | gated={m.get('gated')} | tot={tot and round(tot/1e9,1)}B dt={dt and {k:round(v/1e9,1) for k,v in dt.items()}} | {arch} model_type={full.get('model_type')} | L={L} H={H} KV={KV} hd={hd} hs={hs} | ctx={ctx} rope={json.dumps(rs)[:120] if rs else None} | E={E} k={topk} | mla={kvl},{rope} | sw={sw} lt={lt} | q={q} | dl={m.get('downloads')} | lic={(m.get('cardData') or {}).get('license')}")
