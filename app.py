import os, io, math, requests
import numpy as np
from PIL import Image, ImageFilter
from flask import Flask, request, jsonify, render_template_string

app=Flask(__name__)
BASE=os.getenv('FRINKIAC_BASE','https://frinkiac.com').rstrip('/')
UA={'User-Agent':'kent-brockman-zine-research/2.0'}
EP_COUNTS={1:13,2:22,3:24,4:22,5:22,6:25,7:25,8:25,9:25,10:23}


def get(url, timeout=25):
    r=requests.get(url,timeout=timeout,headers=UA); r.raise_for_status(); return r

def api_json(url): return get(url).json()
def tc(ms):
    s,m=divmod(int(ms),1000); return f'{s//60:02d}:{s%60:02d}.{m:03d}'
def img_url(ep,ts): return f'{BASE}/img/{ep}/{int(ts)}.jpg'

def norm_search(data):
    rows=data if isinstance(data,list) else (data.get('Results',[]) if isinstance(data,dict) else [])
    out=[]
    for r in rows:
        if not isinstance(r,dict): continue
        ep=r.get('Episode',''); ep=ep.get('Key','') if isinstance(ep,dict) else ep
        try: ts=int(r.get('Timestamp',0))
        except: continue
        if ep: out.append({'episode':ep,'timestamp':ts})
    return out

def norm_frames(data, episode):
    rows=data
    if isinstance(data,dict):
        for k in ('Frames','frames','Nearby','nearby','Results','results'):
            if isinstance(data.get(k),list): rows=data[k]; break
    if not isinstance(rows,list): return []
    out=[]
    for r in rows:
        if not isinstance(r,dict): continue
        ep=r.get('Episode') or r.get('episode') or episode
        if isinstance(ep,dict): ep=ep.get('Key') or ep.get('key') or episode
        try: ts=int(r.get('Timestamp',r.get('timestamp')))
        except: continue
        out.append({'episode':ep,'timestamp':ts,'timecode':tc(ts),'image':img_url(ep,ts)})
    return out

def caption(ep,ts):
    try:
        d=api_json(f'{BASE}/api/caption?e={ep}&t={ts}'); subs=d.get('Subtitles',[]) if isinstance(d,dict) else []
        return ' '.join(str(x.get('Content','')).strip() for x in subs if isinstance(x,dict)).strip()
    except: return ''

def feature_from_bytes(b):
    im=Image.open(io.BytesIO(b)).convert('RGB').resize((48,36))
    arr=np.asarray(im,dtype=np.float32)/255.0
    # Coarse spatial colour layout.
    small=np.asarray(im.resize((12,9)),dtype=np.float32).reshape(-1)/255.0
    # Per-channel histograms.
    hist=[]
    for c in range(3):
        h,_=np.histogram(arr[:,:,c],bins=12,range=(0,1),density=True); hist.extend(h.tolist())
    # Edge layout catches desk/card composition.
    gray=im.convert('L').filter(ImageFilter.FIND_EDGES).resize((12,9))
    edge=np.asarray(gray,dtype=np.float32).reshape(-1)/255.0
    f=np.concatenate([small,np.asarray(hist,dtype=np.float32)*0.12,edge*0.7])
    n=np.linalg.norm(f); return f/n if n else f

def feature_from_url(url): return feature_from_bytes(get(url,timeout=20).content)
def cosine(a,b): return float(np.dot(a,b))

def model_score(f,pos,neg):
    if not pos: return 0.0
    ps=sorted((cosine(f,p) for p in pos),reverse=True)[:min(5,len(pos))]
    p=float(np.mean(ps))
    if neg:
        ns=sorted((cosine(f,n) for n in neg),reverse=True)[:min(5,len(neg))]
        n=float(np.mean(ns))
        # Similarity plus margin from negatives.
        return 0.7*p+0.3*((p-n+1)/2)
    return p

HTML=r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kent Brockman Frame Finder v2</title>
<style>body{font-family:system-ui,sans-serif;max-width:1320px;margin:26px auto;padding:0 18px;color:#111}h1{margin:0}.sub{color:#555}.tabs{display:flex;gap:8px;margin:20px 0}.tabs button,.btn{padding:9px 12px;font:inherit}.panel{display:none}.panel.on{display:block}.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin:16px 0}input,select,button{font:inherit;padding:8px}label{display:flex;gap:6px;align-items:center}.scene{border-top:2px solid #111;margin:26px 0;padding-top:14px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}.card{border:1px solid #ccc;border-radius:8px;padding:8px}.card img{width:100%;display:block;background:#eee;min-height:110px}.meta{font-size:13px;margin-top:6px}.actions{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:7px}.actions button{font-size:12px;padding:7px}.good,.zine{background:#111;color:white}.bad{background:#ddd}.status{padding:10px;background:#f4f4f4;margin:12px 0}.error{background:#fee;padding:10px}.score{font-weight:700}.tiny{font-size:13px;color:#555}</style></head><body>
<h1>Kent Brockman Frame Finder <small>v2</small></h1><div class="sub">Dialogue search and trainable visual scan feed the same zine library.</div>
<div class="tabs"><button onclick="tab('dialogue')">Dialogue Search</button><button onclick="tab('visual')">Visual Scan</button><button onclick="tab('library')">Training + Zine</button></div>
<section id="dialogue" class="panel on"><form class="bar" method="get"><input type="hidden" name="mode" value="dialogue"><input name="q" value="{{q}}" placeholder="e.g. this is Kent Brockman" size="38"><label>Seconds either side <input name="window" value="{{window}}" type="number" min="1" max="30" style="width:65px"></label><label>Max scenes <input name="limit" value="{{limit}}" type="number" min="1" max="50" style="width:60px"></label><button>Search</button></form>
{% if error %}<div class="error">{{error}}</div>{% endif %}
{% for s in scenes %}<section class="scene"><h2>{{s.episode}} · {{s.hit_timecode}}</h2><div class="tiny">{{s.dialogue}}</div><div class="grid">{% for x in s.frames %}<div class="card" data-id="{{x.episode}}-{{x.timestamp}}"><a href="{{x.image}}" target="_blank"><img src="{{x.image}}" loading="lazy"></a><div class="meta"><b>{{x.episode}}</b> · {{x.timecode}}<br>{{x.timestamp}} ms</div><div class="actions"><button onclick="labelFrame(this,'good')" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}">Good example</button><button onclick="labelFrame(this,'bad')" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}">Not relevant</button><button onclick="labelFrame(this,'zine')" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}">Save to zine</button></div></div>{% endfor %}</div></section>{% endfor %}</section>
<section id="visual" class="panel"><div class="status" id="trainStatus"></div><div class="bar"><label>Season <select id="season">{% for n in range(1,11) %}<option value="{{n}}">Season {{n}}</option>{% endfor %}</select></label><label>Sample every <select id="interval"><option value="30">30 sec</option><option value="20">20 sec</option><option value="15">15 sec</option><option value="10">10 sec</option></select></label><label>Top results <input id="topn" type="number" value="60" min="10" max="200" style="width:65px"></label><button onclick="visualScan()">Run visual scan</button></div><div class="tiny">Start with 20+ Good examples. Saved zine frames count as strong positive examples. More varied examples help it find different Channel 6, Smartline and Eye on Springfield layouts.</div><div id="scanMsg" class="status" style="display:none"></div><div id="visualGrid" class="grid"></div></section>
<section id="library" class="panel"><div class="bar"><button onclick="exportZine()">Export zine CSV</button><button onclick="clearLabels()">Clear all labels</button></div><div id="libraryStatus" class="status"></div><div id="libraryGrid" class="grid"></div></section>
<script>
const K='kent-v2-labels';function labels(){try{return JSON.parse(localStorage.getItem(K)||'{}')}catch(e){return {}}}function saveLabels(x){localStorage.setItem(K,JSON.stringify(x));refresh()}
function tab(id){document.querySelectorAll('.panel').forEach(x=>x.classList.remove('on'));document.getElementById(id).classList.add('on');refresh()}
function dataFrom(b){return {episode:b.dataset.ep,timestamp:+b.dataset.ts,timecode:b.dataset.tc,image:b.dataset.img}}
function labelFrame(b,label){let x=labels(),d=dataFrom(b),id=d.episode+'-'+d.timestamp;if(x[id]&&x[id].label===label)delete x[id];else x[id]={...d,label};saveLabels(x)}
function refresh(){let x=labels(),a=Object.values(x),g=a.filter(v=>v.label==='good').length,b=a.filter(v=>v.label==='bad').length,z=a.filter(v=>v.label==='zine').length;document.getElementById('trainStatus').textContent=`Training set: ${g} good + ${z} zine positives, ${b} negatives.`;document.getElementById('libraryStatus').textContent=`${g} good examples · ${b} negatives · ${z} saved zine frames`;document.querySelectorAll('.card').forEach(c=>{let v=x[c.dataset.id];c.querySelectorAll('.actions button').forEach(q=>q.classList.remove('good','bad','zine'));if(v){let btn=[...c.querySelectorAll('.actions button')].find(q=>q.textContent.toLowerCase().includes(v.label==='good'?'good':v.label==='bad'?'not':'zine'));if(btn)btn.classList.add(v.label)}});renderLibrary()}
function cardHTML(v,score=''){return `<div class="card" data-id="${v.episode}-${v.timestamp}"><a href="${v.image}" target="_blank"><img src="${v.image}" loading="lazy"></a><div class="meta"><b>${v.episode}</b> · ${v.timecode||''}${score?`<br><span class="score">Visual score ${score}</span>`:''}</div><div class="actions"><button data-ep="${v.episode}" data-ts="${v.timestamp}" data-tc="${v.timecode||''}" data-img="${v.image}" onclick="labelFrame(this,'good')">Good example</button><button data-ep="${v.episode}" data-ts="${v.timestamp}" data-tc="${v.timecode||''}" data-img="${v.image}" onclick="labelFrame(this,'bad')">Not relevant</button><button data-ep="${v.episode}" data-ts="${v.timestamp}" data-tc="${v.timecode||''}" data-img="${v.image}" onclick="labelFrame(this,'zine')">Save to zine</button></div></div>`}
function renderLibrary(){let el=document.getElementById('libraryGrid');if(!el)return;el.innerHTML=Object.values(labels()).map(v=>cardHTML(v,`label: ${v.label}`)).join('')}
async function visualScan(){let x=Object.values(labels()),pos=x.filter(v=>v.label==='good'||v.label==='zine'),neg=x.filter(v=>v.label==='bad');if(pos.length<3){alert('Add at least 3 Good example or Save to zine frames first. 20+ is recommended.');return}let msg=document.getElementById('scanMsg');msg.style.display='block';msg.textContent='Scanning. A full season can take a few minutes on Railway…';document.getElementById('visualGrid').innerHTML='';try{let r=await fetch('/api/visual-scan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({season:+document.getElementById('season').value,interval:+document.getElementById('interval').value,top_n:+document.getElementById('topn').value,positives:pos,negatives:neg})});let d=await r.json();if(!r.ok)throw new Error(d.error||'Scan failed');msg.textContent=`Scanned ${d.scanned} real frames from Season ${d.season}. Showing ${d.results.length} highest-ranked candidates.`;document.getElementById('visualGrid').innerHTML=d.results.map(v=>cardHTML(v,(v.score*100).toFixed(1)+'%')).join('');refresh()}catch(e){msg.textContent='Visual scan failed: '+e.message}}
function exportZine(){let a=Object.values(labels()).filter(v=>v.label==='zine'),cols=['episode','timestamp','timecode','image'];let esc=v=>/[",\n]/.test(String(v))?'"'+String(v).replaceAll('"','""')+'"':v;let csv=[cols.join(','),...a.map(v=>cols.map(c=>esc(v[c]||'')).join(','))].join('\n');let u=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),q=document.createElement('a');q.href=u;q.download='kent-brockman-zine.csv';q.click();URL.revokeObjectURL(u)}
function clearLabels(){if(confirm('Clear training labels and saved zine frames?')){localStorage.removeItem(K);refresh()}}
refresh();
</script></body></html>'''

@app.route('/')
def index():
    q=request.args.get('q',''); scenes=[]; error=None
    try: window=max(1,min(30,int(request.args.get('window','8'))))
    except: window=8
    try: limit=max(1,min(50,int(request.args.get('limit','12'))))
    except: limit=12
    if q:
        try:
            for h in norm_search(api_json(f'{BASE}/api/search?q={requests.utils.quote(q)}'))[:limit]:
                ms=window*1000; raw=api_json(f'{BASE}/api/frames/{h["episode"]}/{h["timestamp"]}/{ms}/{ms}')
                seen=set(); frames=[]
                for f in norm_frames(raw,h['episode']):
                    k=(f['episode'],f['timestamp'])
                    if k not in seen: seen.add(k); frames.append(f)
                scenes.append({'episode':h['episode'],'hit_timecode':tc(h['timestamp']),'dialogue':caption(h['episode'],h['timestamp']),'frames':frames})
        except Exception as e: error=f'Frinkiac request failed: {e}'
    return render_template_string(HTML,q=q,window=window,limit=limit,scenes=scenes,error=error)

@app.route('/api/visual-scan',methods=['POST'])
def visual_scan():
    d=request.get_json(silent=True) or {}
    season=int(d.get('season',1)); interval=max(10,min(60,int(d.get('interval',30)))); top_n=max(10,min(200,int(d.get('top_n',60))))
    if season not in EP_COUNTS:return jsonify({'error':'season must be 1-10'}),400
    positives=d.get('positives',[]); negatives=d.get('negatives',[])
    if len(positives)<3:return jsonify({'error':'At least 3 positive examples are required'}),400
    try:
        pos=[feature_from_url(x['image']) for x in positives[:80]]; neg=[feature_from_url(x['image']) for x in negatives[:120]]
        candidates={}
        # Probe every N seconds. The Frinkiac frame endpoint converts each probe into real stored frames.
        # 24 minutes covers S1-S10 episodes with margin; failed/out-of-range probes are ignored.
        for e in range(1,EP_COUNTS[season]+1):
            ep=f'S{season:02d}E{e:02d}'
            for sec in range(0,24*60,interval):
                ts=sec*1000
                try:
                    raw=api_json(f'{BASE}/api/frames/{ep}/{ts}/1000/1000')
                    fs=norm_frames(raw,ep)
                    if fs:
                        f=min(fs,key=lambda x:abs(x['timestamp']-ts)); candidates[(ep,f['timestamp'])]=f
                except: pass
        scored=[]
        for f in candidates.values():
            try:
                feat=feature_from_url(f['image']); score=model_score(feat,pos,neg); scored.append({**f,'score':round(score,6)})
            except: pass
        scored.sort(key=lambda x:x['score'],reverse=True)
        return jsonify({'season':season,'scanned':len(scored),'results':scored[:top_n]})
    except Exception as e:return jsonify({'error':str(e)}),502

@app.route('/health')
def health(): return {'ok':True,'version':'2.0','methods':['dialogue','visual-training']}

if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.getenv('PORT','5000')))
