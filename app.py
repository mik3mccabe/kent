import os, requests
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
BASE = os.getenv('FRINKIAC_BASE', 'https://frinkiac.com').rstrip('/')
UA = {'User-Agent':'kent-brockman-zine-research/1.2'}

def api_json(url):
    r = requests.get(url, timeout=25, headers=UA)
    r.raise_for_status()
    return r.json()

def tc(ms):
    s, milli = divmod(int(ms),1000)
    return f'{s//60:02d}:{s%60:02d}.{milli:03d}'

def norm_search(data):
    rows = data if isinstance(data,list) else (data.get('Results',[]) if isinstance(data,dict) else [])
    out=[]
    for r in rows:
        if not isinstance(r,dict): continue
        ep=r.get('Episode','')
        if isinstance(ep,dict): ep=ep.get('Key') or ep.get('key') or ''
        try: ts=int(r.get('Timestamp',0))
        except: continue
        if ep: out.append({'episode':ep,'timestamp':ts})
    return out

def norm_frames(data, episode):
    rows=data
    if isinstance(data,dict):
        for key in ('Frames','frames','Nearby','nearby','Results','results'):
            if isinstance(data.get(key),list): rows=data[key]; break
    if not isinstance(rows,list): return []
    out=[]
    for r in rows:
        if not isinstance(r,dict): continue
        ep=r.get('Episode') or r.get('episode') or episode
        if isinstance(ep,dict): ep=ep.get('Key') or ep.get('key') or episode
        try: ts=int(r.get('Timestamp',r.get('timestamp')))
        except: continue
        out.append({'episode':ep,'timestamp':ts,'timecode':tc(ts),'image':f'{BASE}/img/{ep}/{ts}.jpg'})
    return out

def caption(ep,ts):
    try:
        d=api_json(f'{BASE}/api/caption?e={ep}&t={ts}')
        subs=d.get('Subtitles',[]) if isinstance(d,dict) else []
        return ' '.join(str(x.get('Content','')).strip() for x in subs if isinstance(x,dict)).strip()
    except Exception:
        return ''

HTML=r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kent Brockman Frame Finder</title>
<style>body{font-family:system-ui,sans-serif;max-width:1280px;margin:28px auto;padding:0 18px;color:#111}h1{margin-bottom:4px}.search{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin:22px 0}input,button{font:inherit;padding:9px}label{display:flex;gap:7px;align-items:center}.scene{border-top:2px solid #111;padding-top:16px;margin:28px 0}.scenehead{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}.dialogue{margin:7px 0 14px;color:#444}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:12px}.card{border:1px solid #ccc;padding:8px;border-radius:8px}.card img{width:100%;display:block;background:#eee;min-height:100px}.meta{font-size:13px;margin-top:6px}.save{width:100%;margin-top:7px}.saved{background:#111;color:#fff}.tools{display:flex;gap:8px;margin:18px 0}.error{padding:12px;background:#fee}.hint{color:#555;font-size:14px}</style></head><body>
<h1>Kent Brockman Frame Finder</h1><p>Search spoken dialogue, then inspect actual Frinkiac frames around each hit.</p>
<form class="search" method="get"><input name="q" value="{{q}}" placeholder="e.g. this is Kent Brockman" size="38"><label>Seconds either side <input name="window" value="{{window}}" type="number" min="1" max="30" style="width:70px"></label><label>Max scenes <input name="limit" value="{{limit}}" type="number" min="1" max="50" style="width:60px"></label><button>Search</button></form>
<div class="tools"><button type="button" onclick="exportSaved()">Export saved CSV</button><button type="button" onclick="clearSaved()">Clear saved</button><span id="count" class="hint"></span></div>
{% if error %}<div class="error">{{error}}</div>{% endif %}
{% for s in scenes %}<section class="scene"><div class="scenehead"><h2>{{s.episode}} · hit {{s.hit_timecode}}</h2><span>{{s.frames|length}} real Frinkiac frames</span></div>{% if s.dialogue %}<div class="dialogue">{{s.dialogue}}</div>{% endif %}<div class="grid">
{% for x in s.frames %}<div class="card"><a href="{{x.image}}" target="_blank"><img src="{{x.image}}" loading="lazy"></a><div class="meta"><b>{{x.episode}}</b> · {{x.timecode}}<br>{{x.timestamp}} ms</div><button class="save" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}" data-dialogue="{{s.dialogue|e}}" onclick="toggleSave(this)">Save to zine</button></div>{% endfor %}</div></section>{% endfor %}
<script>
const KEY='kent-zine-saved-v1';
function getSaved(){try{return JSON.parse(localStorage.getItem(KEY)||'[]')}catch(e){return []}}
function setSaved(x){localStorage.setItem(KEY,JSON.stringify(x)); refresh()}
function idFor(b){return b.dataset.ep+'-'+b.dataset.ts}
function toggleSave(b){let a=getSaved(),id=idFor(b),i=a.findIndex(x=>x.id===id);if(i>=0)a.splice(i,1);else a.push({id,episode:b.dataset.ep,timestamp_ms:b.dataset.ts,timecode:b.dataset.tc,image_url:b.dataset.img,dialogue:b.dataset.dialogue,graphic_text:'',category:'',notes:''});setSaved(a)}
function refresh(){let a=getSaved(),ids=new Set(a.map(x=>x.id));document.querySelectorAll('.save').forEach(b=>{let on=ids.has(idFor(b));b.textContent=on?'Saved ✓':'Save to zine';b.classList.toggle('saved',on)});document.getElementById('count').textContent=a.length+' saved frame'+(a.length===1?'':'s')}
function esc(v){v=String(v??'');return /[",\n]/.test(v)?'"'+v.replaceAll('"','""')+'"':v}
function exportSaved(){let a=getSaved(),cols=['episode','timestamp_ms','timecode','image_url','dialogue','graphic_text','category','notes'];let csv=[cols.join(','),...a.map(x=>cols.map(c=>esc(x[c])).join(','))].join('\n');let blob=new Blob([csv],{type:'text/csv'}),u=URL.createObjectURL(blob),link=document.createElement('a');link.href=u;link.download='kent-brockman-zine.csv';link.click();URL.revokeObjectURL(u)}
function clearSaved(){if(confirm('Clear all saved zine frames?'))setSaved([])}
refresh();
</script></body></html>'''

@app.route('/')
def index():
    q=request.args.get('q','')
    try: window=max(1,min(30,int(request.args.get('window','8'))))
    except: window=8
    try: limit=max(1,min(50,int(request.args.get('limit','12'))))
    except: limit=12
    scenes=[]; error=None
    if q:
        try:
            hits=norm_search(api_json(f'{BASE}/api/search?q={requests.utils.quote(q)}'))[:limit]
            for h in hits:
                before=after=window*1000
                raw=api_json(f'{BASE}/api/frames/{h["episode"]}/{h["timestamp"]}/{before}/{after}')
                frames=norm_frames(raw,h['episode'])
                # De-duplicate while preserving API order.
                seen=set(); unique=[]
                for f in frames:
                    k=(f['episode'],f['timestamp'])
                    if k not in seen: seen.add(k); unique.append(f)
                scenes.append({'episode':h['episode'],'hit_timestamp':h['timestamp'],'hit_timecode':tc(h['timestamp']),'dialogue':caption(h['episode'],h['timestamp']),'frames':unique})
        except Exception as e: error=f'Frinkiac request failed: {e}'
    return render_template_string(HTML,q=q,window=window,limit=limit,scenes=scenes,error=error)

@app.route('/api/search')
def api_search():
    q=request.args.get('q','')
    if not q:return jsonify({'error':'q is required'}),400
    try:return jsonify(norm_search(api_json(f'{BASE}/api/search?q={requests.utils.quote(q)}')))
    except Exception as e:return jsonify({'error':str(e)}),502

@app.route('/api/frames')
def api_frames():
    ep=request.args.get('episode'); ts=request.args.get('timestamp',type=int); seconds=request.args.get('seconds',8,type=int)
    if not ep or ts is None:return jsonify({'error':'episode and timestamp are required'}),400
    seconds=max(1,min(30,seconds)); ms=seconds*1000
    try:return jsonify(norm_frames(api_json(f'{BASE}/api/frames/{ep}/{ts}/{ms}/{ms}'),ep))
    except Exception as e:return jsonify({'error':str(e)}),502

@app.route('/health')
def health(): return {'ok':True,'version':'1.2'}
