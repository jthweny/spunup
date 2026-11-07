import json,sys,re,urllib.request
d=json.load(open('raw/search/quants.json'))
def get(u):
    try:
        with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'spunup-research'}),timeout=60) as r: return json.load(r)
    except Exception as e: return None
TRUST=['ggml-org','unsloth','bartowski','lmstudio-community','mlx-community']
BAD=re.compile(r'abliter|uncensor|heretic|obliter|humanlike|penclaw|distill|reap|derisk|mtp-only|dspark|dflash|-layers|be-gguf|iq1|nvfp4|gsq|swift|gyro|maya|exl|cyber',re.I)
def pick(mid,lst,kind):
    org,name=mid.split('/')
    base=name.lower()
    cands=[i for i,_ in lst if not BAD.search(i)]
    base2=re.sub(r'-(bf16)$','',base)
    SUF=re.compile(r'^([-_](it|instruct))?([-_](qat|ud|text|mlx|gguf|q4_0|unquantized))*([-_]?(\d+bit|\d+-bit|bf16|mxfp4|mxfp8|q\d|mlx|gguf|dwq|mixed|q4-mlx|mlx-\d+bit|4bit-dwq))*$')
    def ok(i):
        n=i.split('/')[1].lower()
        n=n.replace(org.lower()+'_','').replace(org.lower()+'.','')
        for b in (base,base2):
            if n.startswith(b) and SUF.match(n[len(b):]): return True
        return False
    cands=[i for i in cands if ok(i)]
    def rank(i):
        o=i.split('/')[0]
        if o==org: return -1
        return TRUST.index(o) if o in TRUST else 99
    cands.sort(key=rank)
    return [c for c in cands if rank(c)<99][:4]
QRE=re.compile(r'(?<![A-Za-z0-9])(UD-)?(IQ[1-4]_[A-Z]+|Q[1-8]_K(_[A-Z]+)?|Q[1-8]_[01]|MXFP4(_MOE)?|BF16|F16|F32|TQ[12]_0|PTQ1_0|PQ2_0|FP8(_E4M3)?)(?![A-Za-z0-9])',re.I)
SIDE=re.compile(r'mmproj|mtp|draft|assistant|dflash|dspark|eagle|imatrix',re.I)
out={}
for mid,v in d.items():
    g=pick(mid,v['gguf'],'gguf'); m=pick(mid,v['mlx'],'mlx')
    best=None
    for repo in g:
      files={}
      t=get(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true")
      for f in (t or []):
        p=f.get('path','')
        if not p.endswith('.gguf') or SIDE.search(p.split('/')[-1]): continue
        ms=list(QRE.finditer(p.split('/')[-1])) or list(QRE.finditer(p))
        if not ms: continue
        q=ms[-1].group(0).upper()
        files.setdefault(q,[0,[]]); files[q][0]+=f.get('size',0); files[q][1].append(p)
      if best is None or len(files)>=len(best[1])+3: best=(repo,files)
    files=best[1] if best else {}
    if best: g=[best[0]]+[x for x in g if x!=best[0]]
    out[mid]={'gguf_repos':g,'mlx_repos':m,'gguf_files':{q:{'gb':round(s/1e9,2),'files':fs[:1],'parts':len(fs)} for q,(s,fs) in files.items()}}
    print(mid.split('/')[1],'| G:',g[:1],'| M:',m[:1],'|',' '.join(f"{q}={round(s/1e9,1)}" for q,(s,_) in sorted(files.items(),key=lambda x:x[1][0])))
json.dump(out,open('raw/search/picked.json','w'),indent=1)
