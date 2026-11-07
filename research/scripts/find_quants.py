import json,sys,urllib.request,urllib.parse,os
def get(u):
    try:
        with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'spunup-research'}),timeout=30) as r: return json.load(r)
    except Exception as e: return []
out={}
for line in open(sys.argv[1]):
    mid=line.strip()
    if not mid: continue
    name=mid.split('/')[1]
    q=urllib.parse.quote(name)
    gg=get(f"https://huggingface.co/api/models?search={q}&filter=gguf&sort=downloads&limit=15")
    ml=get(f"https://huggingface.co/api/models?search={q}&filter=mlx&sort=downloads&limit=10")
    out[mid]={'gguf':[(m['id'],m.get('downloads',0)) for m in gg],'mlx':[(m['id'],m.get('downloads',0)) for m in ml]}
json.dump(out,open(sys.argv[2],'w'),indent=0)
