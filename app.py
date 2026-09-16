import os, io, math, requests, zipfile, re, json, traceback, time
import numpy as np
from PIL import Image, ImageFilter
from flask import Flask, request, jsonify, render_template_string, send_file, Response, stream_with_context

app=Flask(__name__)
BASE=os.getenv('FRINKIAC_BASE','https://frinkiac.com').rstrip('/')
UA={'User-Agent':'kent-brockman-zine-research/2.4'}
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

HTML=r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kent Brockman Frame Finder v2.4</title>
<style>body{font-family:system-ui,sans-serif;max-width:1320px;margin:26px auto;padding:0 18px;color:#111}h1{margin:0}.sub{color:#555}.tabs{display:flex;gap:8px;margin:20px 0}.tabs button,.btn{padding:9px 12px;font:inherit}.panel{display:none}.panel.on{display:block}.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin:16px 0}input,select,button{font:inherit;padding:8px}label{display:flex;gap:6px;align-items:center}.scene{border-top:2px solid #111;margin:26px 0;padding-top:14px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}.card{border:1px solid #ccc;border-radius:8px;padding:8px}.card img{width:100%;display:block;background:#eee;min-height:110px}.meta{font-size:13px;margin-top:6px}.actions{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:7px}.actions button{font-size:12px;padding:7px}.good,.zine{background:#111;color:white}.bad{background:#ddd}.status{padding:10px;background:#f4f4f4;margin:12px 0}.error{background:#fee;padding:10px}.score{font-weight:700}.tiny{font-size:13px;color:#555}</style></head><body>
<h1>Kent Brockman Frame Finder <small>v2.4</small></h1><div class="sub">Dialogue search and trainable visual scan feed the same zine library.</div>
<div class="tabs"><button onclick="tab('dialogue')">Dialogue Search</button><button onclick="tab('visual')">Visual Scan</button><button onclick="tab('library')">Zine Library</button></div>
<section id="dialogue" class="panel on"><form class="bar" method="get"><input type="hidden" name="mode" value="dialogue"><input name="q" value="{{q}}" placeholder="e.g. this is Kent Brockman" size="38"><label>Seconds either side <input name="window" value="{{window}}" type="number" min="1" max="30" style="width:65px"></label><label>Max raw matches <input name="limit" value="{{limit}}" type="number" min="1" max="500" style="width:72px"><label><input type="checkbox" name="all" value="1" {% if scan_all %}checked{% endif %}> Scan all matches</label><label><input type="checkbox" id="hideReviewed" onchange="applyReviewed()"> Hide reviewed scenes</label></label><button>Search</button></form>
{% if error %}<div class="error">{{error}}</div>{% endif %}
{% for s in scenes %}<section class="scene" data-scene="{{s.scene_id}}"><h2>{{s.episode}} · {{s.start_timecode}}–{{s.end_timecode}}</h2><div class="tiny">{{s.dialogue}}</div><div class="bar"><button type="button" onclick="markScene(this)">Mark scene reviewed</button><button type="button" onclick="expandScene(this,'{{s.episode}}',{{s.midpoint}})">Expand ±30 sec</button></div><div class="grid">{% for x in s.frames %}<div class="card" data-id="{{x.episode}}-{{x.timestamp}}"><a href="{{x.image}}" target="_blank"><img src="{{x.image}}" loading="lazy"></a><div class="meta"><b>{{x.episode}}</b> · {{x.timecode}}<br>{{x.timestamp}} ms</div><div class="actions"><button onclick="labelFrame(this,'good')" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}">Good example</button><button onclick="labelFrame(this,'news')" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}">News, no graphic</button><button onclick="labelFrame(this,'bad')" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}">Not relevant</button><button onclick="labelFrame(this,'zine')" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}">Save to zine</button></div></div>{% endfor %}</div></section>{% endfor %}</section>
<section id="visual" class="panel"><div class="status" id="trainStatus"></div><div class="bar"><label>Season <select id="season">{% for n in range(1,11) %}<option value="{{n}}">Season {{n}}</option>{% endfor %}</select></label><label>Sample every <select id="interval"><option value="30">30 sec</option><option value="20">20 sec</option><option value="15">15 sec</option><option value="10">10 sec</option></select></label><label>Top results <input id="topn" type="number" value="60" min="10" max="200" style="width:65px"></label><button onclick="visualScan(false,false)">Start new scan</button><button onclick="resumeVisualScan()">Resume scan</button><button onclick="visualScan(true,false)">Diagnostic: 1 episode</button><button id="stopScan" onclick="stopVisualScan()" disabled>Stop scan</button><button onclick="clearScanProgress()">Clear scan progress</button></div><div class="tiny">Start with 20+ Good examples. Each episode is now a separate short request. Progress and ranked candidates are saved in this browser after every episode. Resume continues after a stop, timeout, refresh, or failed episode.</div><div id="scanMsg" class="status" style="display:none"></div><details open><summary>Visual Scan Log</summary><div class="bar"><button onclick="copyLogs()">Copy logs</button><button onclick="clearLogs()">Clear logs</button></div><pre id="scanLog" style="white-space:pre-wrap;background:#111;color:#eee;padding:12px;max-height:320px;overflow:auto;border-radius:6px">Ready.</pre></details><div id="visualGrid" class="grid"></div></section>
<section id="library" class="panel"><div class="bar"><button onclick="downloadZip()">Download Zine ZIP</button><button onclick="exportZine()">Export metadata CSV</button><button onclick="clearLabels()">Clear all labels</button></div><div id="libraryStatus" class="status"></div><div class="tiny">Saved zine frames appear here in season, episode and timestamp order. Add the exact on-screen wording before downloading your archive.</div><div id="libraryGrid" class="grid"></div></section>
<script>
const K='kent-v2-labels',R='kent-v21-reviewed';function labels(){try{return JSON.parse(localStorage.getItem(K)||'{}')}catch(e){return {}}}function saveLabels(x){localStorage.setItem(K,JSON.stringify(x));refresh()}
function tab(id){document.querySelectorAll('.panel').forEach(x=>x.classList.remove('on'));document.getElementById(id).classList.add('on');refresh()}
function dataFrom(b){return {episode:b.dataset.ep,timestamp:+b.dataset.ts,timecode:b.dataset.tc,image:b.dataset.img}}
function labelFrame(b,label){let x=labels(),d=dataFrom(b),id=d.episode+'-'+d.timestamp;if(x[id]&&x[id].label===label)delete x[id];else x[id]={...d,label};saveLabels(x)}
function refresh(){let x=labels(),a=Object.values(x),g=a.filter(v=>v.label==='good').length,b=a.filter(v=>v.label==='bad').length,z=a.filter(v=>v.label==='zine').length,n=a.filter(v=>v.label==='news').length;document.getElementById('trainStatus').textContent=`Training set: ${g} good + ${z} zine positives, ${n} news/no-graphic, ${b} negatives.`;document.getElementById('libraryStatus').textContent=`${g} good examples · ${n} news/no-graphic · ${b} negatives · ${z} saved zine frames`;document.querySelectorAll('.card').forEach(c=>{let v=x[c.dataset.id];c.querySelectorAll('.actions button').forEach(q=>q.classList.remove('good','bad','zine'));if(v){let btn=[...c.querySelectorAll('.actions button')].find(q=>q.textContent.toLowerCase().includes(v.label==='good'?'good':v.label==='bad'?'not':v.label==='news'?'news':'zine'));if(btn)btn.classList.add(v.label)}});renderLibrary()}
function cardHTML(v,score=''){return `<div class="card" data-id="${v.episode}-${v.timestamp}"><a href="${v.image}" target="_blank"><img src="${v.image}" loading="lazy"></a><div class="meta"><b>${v.episode}</b> · ${v.timecode||''}${score?`<br><span class="score">Visual score ${score}</span>`:''}</div><div class="actions"><button data-ep="${v.episode}" data-ts="${v.timestamp}" data-tc="${v.timecode||''}" data-img="${v.image}" onclick="labelFrame(this,'good')">Good example</button><button data-ep="${v.episode}" data-ts="${v.timestamp}" data-tc="${v.timecode||''}" data-img="${v.image}" onclick="labelFrame(this,'news')">News, no graphic</button><button data-ep="${v.episode}" data-ts="${v.timestamp}" data-tc="${v.timecode||''}" data-img="${v.image}" onclick="labelFrame(this,'bad')">Not relevant</button><button data-ep="${v.episode}" data-ts="${v.timestamp}" data-tc="${v.timecode||''}" data-img="${v.image}" onclick="labelFrame(this,'zine')">Save to zine</button></div></div>`}
function renderLibrary(){let el=document.getElementById('libraryGrid');if(!el)return;let a=Object.values(labels()).filter(v=>v.label==='zine').sort((a,b)=>a.episode.localeCompare(b.episode)||a.timestamp-b.timestamp);el.innerHTML=a.map(v=>`<div class="card" data-id="${v.episode}-${v.timestamp}"><a href="${v.image}" target="_blank"><img src="${v.image}" loading="lazy"></a><div class="meta"><b>${v.episode}</b> · ${v.timecode||''}<br>${v.timestamp} ms</div><input style="width:calc(100% - 18px);margin-top:7px" placeholder="Exact card wording" value="${escHtml(v.title||'')}" onchange="editZine('${v.episode}-${v.timestamp}','title',this.value)"><select style="width:100%;margin-top:6px" onchange="editZine('${v.episode}-${v.timestamp}','category',this.value)"><option ${!v.category?'selected':''}>Category</option><option ${v.category==='Story graphic'?'selected':''}>Story graphic</option><option ${v.category==='Programme title'?'selected':''}>Programme title</option><option ${v.category==='Smartline'?'selected':''}>Smartline</option><option ${v.category==='Eye on Springfield'?'selected':''}>Eye on Springfield</option><option ${v.category==='Channel 6'?'selected':''}>Channel 6</option></select><textarea style="width:calc(100% - 18px);margin-top:6px" placeholder="Notes" onchange="editZine('${v.episode}-${v.timestamp}','notes',this.value)">${escHtml(v.notes||'')}</textarea><div class="actions"><button onclick="downloadOne('${v.image}','${v.episode}',${v.timestamp})">Download JPG</button><button class="bad" onclick="removeZine('${v.episode}-${v.timestamp}')">Remove</button></div></div>`).join('')||'<div class="status">No saved zine frames yet.</div>'}
function escHtml(s){return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;')}
function editZine(id,k,v){let x=labels();if(x[id]){x[id][k]=v==='Category'?'':v;saveLabels(x)}}
function removeZine(id){let x=labels();delete x[id];saveLabels(x)}
function downloadOne(url,ep,ts){let a=document.createElement('a');a.href=url;a.download=`${ep}_${ts}.jpg`;a.target='_blank';a.click()}
async function downloadZip(){let a=Object.values(labels()).filter(v=>v.label==='zine');if(!a.length){alert('Save at least one frame to the zine first.');return}let r=await fetch('/api/zine-zip',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({items:a})});if(!r.ok){let d=await r.json().catch(()=>({}));alert(d.error||'ZIP download failed');return}let blob=await r.blob(),u=URL.createObjectURL(blob),q=document.createElement('a');q.href=u;q.download='kent-brockman-zine.zip';q.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
let scanController=null,seasonScan=null;
const SCANKEY='kent-v24-scan-progress';
function logLine(text){let el=document.getElementById('scanLog'),stamp=new Date().toLocaleTimeString();if(el.textContent==='Ready.')el.textContent='';el.textContent+=`[${stamp}] ${text}\n`;el.scrollTop=el.scrollHeight}
function clearLogs(){document.getElementById('scanLog').textContent='Ready.'}
async function copyLogs(){try{await navigator.clipboard.writeText(document.getElementById('scanLog').textContent);logLine('Logs copied to clipboard.')}catch(e){alert('Could not copy logs: '+e.message)}}
function stopVisualScan(){if(scanController)scanController.abort();if(seasonScan)seasonScan.stopped=true;logLine('Stop requested. Progress has been saved in this browser.');document.getElementById('stopScan').disabled=true}
function saveScanProgress(){if(seasonScan)localStorage.setItem(SCANKEY,JSON.stringify(seasonScan))}
function clearScanProgress(){localStorage.removeItem(SCANKEY);seasonScan=null;logLine('Saved scan progress cleared.');document.getElementById('scanMsg').textContent='Saved scan progress cleared.'}
function renderRanked(){if(!seasonScan)return;let top=+document.getElementById('topn').value||60, vals=Object.values(seasonScan.results||{}).sort((a,b)=>b.score-a.score).slice(0,top);document.getElementById('visualGrid').innerHTML=vals.map(v=>cardHTML(v,(v.score*100).toFixed(1)+'%')).join('');refresh()}
async function scanOneEpisode(ep,interval,pos,neg){scanController=new AbortController();let r=await fetch('/api/visual-episode',{method:'POST',headers:{'Content-Type':'application/json'},signal:scanController.signal,body:JSON.stringify({episode:ep,interval,positives:pos,negatives:neg})});let text=await r.text();if(!r.ok)throw new Error(`HTTP ${r.status}: ${text}`);let d=JSON.parse(text);if(d.logs)for(let x of d.logs)logLine(x);return d}
async function visualScan(diagnostic=false,resume=false){let x=Object.values(labels()),pos=x.filter(v=>v.label==='good'||v.label==='zine'),neg=x.filter(v=>v.label==='bad');if(pos.length<3){alert('Add at least 3 Good example or Save to zine frames first. 20+ is recommended.');return}let season=+document.getElementById('season').value,interval=+document.getElementById('interval').value,total={1:13,2:22,3:24,4:22,5:22,6:25,7:25,8:25,9:25,10:23}[season],msg=document.getElementById('scanMsg');msg.style.display='block';clearLogs();if(resume){try{seasonScan=JSON.parse(localStorage.getItem(SCANKEY)||'null')}catch(e){seasonScan=null}if(!seasonScan||seasonScan.season!==season||seasonScan.interval!==interval){alert('No matching saved scan to resume for this season and interval.');return}seasonScan.stopped=false;logLine(`Resuming Season ${season} after ${seasonScan.completed.length}/${total} episodes.`)}else{seasonScan={season,interval,completed:[],failed:[],results:{},stopped:false,started:new Date().toISOString()};saveScanProgress();document.getElementById('visualGrid').innerHTML='';logLine(`Starting ${diagnostic?'diagnostic':'resumable'} scan. ${pos.length} positives, ${neg.length} negatives.`)}document.getElementById('stopScan').disabled=false;let eps=diagnostic?[1]:Array.from({length:total},(_,i)=>i+1);try{for(let n of eps){if(seasonScan.stopped)break;let ep=`S${String(season).padStart(2,'0')}E${String(n).padStart(2,'0')}`;if(seasonScan.completed.includes(ep)){logLine(`${ep}: already complete, skipping.`);continue}msg.textContent=`Season ${season}: ${seasonScan.completed.length}/${diagnostic?1:total} episodes complete. Scanning ${ep}…`;logLine(`Requesting ${ep}. Each episode is a separate request.`);try{let d=await scanOneEpisode(ep,interval,pos,neg);for(let v of d.results)seasonScan.results[v.episode+'-'+v.timestamp]=v;seasonScan.completed.push(ep);seasonScan.failed=seasonScan.failed.filter(v=>v!==ep);saveScanProgress();logLine(`${ep}: complete. ${d.scanned} frames scored. ${Object.keys(seasonScan.results).length} season candidates saved.`);renderRanked()}catch(e){if(e.name==='AbortError'){seasonScan.stopped=true;saveScanProgress();logLine(`${ep}: stopped. Completed episodes are preserved.`);break}seasonScan.failed.push(ep);saveScanProgress();logLine(`${ep}: FAILED: ${e.message}`);logLine('Continuing to the next episode. Use Resume scan later to retry failures.')}}msg.textContent=seasonScan.stopped?`Stopped. ${seasonScan.completed.length}/${diagnostic?1:total} episodes saved. Click Resume scan to continue.`:`Finished. ${seasonScan.completed.length}/${diagnostic?1:total} episodes complete, ${seasonScan.failed.length} failed. Showing ranked candidates.`;if(!seasonScan.stopped)logLine(`Season pass finished. Completed ${seasonScan.completed.length}, failed ${seasonScan.failed.length}.`)}finally{scanController=null;document.getElementById('stopScan').disabled=true;renderRanked()}}
function resumeVisualScan(){visualScan(false,true)}
function exportZine(){let a=Object.values(labels()).filter(v=>v.label==='zine'),cols=['episode','timestamp','timecode','title','category','notes','image'];let esc=v=>/[",\n]/.test(String(v))?'"'+String(v).replaceAll('"','""')+'"':v;let csv=[cols.join(','),...a.map(v=>cols.map(c=>esc(v[c]||'')).join(','))].join('\n');let u=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),q=document.createElement('a');q.href=u;q.download='kent-brockman-zine.csv';q.click();URL.revokeObjectURL(u)}
function reviewed(){try{return JSON.parse(localStorage.getItem(R)||'{}')}catch(e){return {}}}
function markScene(b){let s=b.closest('.scene'),x=reviewed();x[s.dataset.scene]=true;localStorage.setItem(R,JSON.stringify(x));applyReviewed()}
function applyReviewed(){let h=document.getElementById('hideReviewed')?.checked,x=reviewed();document.querySelectorAll('.scene').forEach(s=>s.style.display=(h&&x[s.dataset.scene])?'none':'')}
async function expandScene(b,ep,ts){b.disabled=true;b.textContent='Loading…';try{let r=await fetch(`/api/nearby?episode=${ep}&timestamp=${ts}&window=30`),d=await r.json();if(!r.ok)throw new Error(d.error||'Failed');let grid=b.closest('.scene').querySelector('.grid');grid.innerHTML=d.frames.map(v=>cardHTML(v)).join('');refresh()}catch(e){alert(e.message)}finally{b.disabled=false;b.textContent='Expand ±30 sec'}}
function clearLabels(){if(confirm('Clear training labels and saved zine frames?')){localStorage.removeItem(K);refresh()}}
refresh();
</script></body></html>'''

@app.route('/')
def index():
    q=request.args.get('q',''); scenes=[]; error=None
    try: window=max(1,min(30,int(request.args.get('window','8'))))
    except: window=8
    try: limit=max(1,min(500,int(request.args.get('limit','100'))))
    except: limit=100
    scan_all=request.args.get('all')=='1'
    if q:
        try:
            hits=norm_search(api_json(f'{BASE}/api/search?q={requests.utils.quote(q)}'))
            if not scan_all: hits=hits[:limit]
            # Merge hits in the same episode when they are within 30 seconds.
            grouped=[]
            for h in sorted(hits,key=lambda x:(x['episode'],x['timestamp'])):
                if grouped and grouped[-1]['episode']==h['episode'] and h['timestamp']-grouped[-1]['last']<=30000:
                    grouped[-1]['hits'].append(h); grouped[-1]['last']=h['timestamp']
                else:
                    grouped.append({'episode':h['episode'],'first':h['timestamp'],'last':h['timestamp'],'hits':[h]})
            for g in grouped:
                midpoint=(g['first']+g['last'])//2
                before=midpoint-(g['first']-window*1000); after=(g['last']+window*1000)-midpoint
                raw=api_json(f'{BASE}/api/frames/{g["episode"]}/{midpoint}/{max(1000,before)}/{max(1000,after)}')
                seen=set(); frames=[]
                for f in norm_frames(raw,g['episode']):
                    k=(f['episode'],f['timestamp'])
                    if k not in seen: seen.add(k); frames.append(f)
                dialogues=[]
                for h in g['hits']:
                    c=caption(h['episode'],h['timestamp'])
                    if c and c not in dialogues: dialogues.append(c)
                scenes.append({'episode':g['episode'],'scene_id':f'{g["episode"]}-{g["first"]}-{g["last"]}',
                    'start_timecode':tc(g['first']),'end_timecode':tc(g['last']),'midpoint':midpoint,
                    'dialogue':' '.join(dialogues),'frames':frames})
        except Exception as e: error=f'Frinkiac request failed: {e}'
    return render_template_string(HTML,q=q,window=window,limit=limit,scan_all=scan_all,scenes=scenes,error=error)

@app.route('/api/nearby')
def nearby():
    try:
        ep=request.args['episode']; ts=int(request.args['timestamp']); w=max(1,min(60,int(request.args.get('window','30'))))*1000
        raw=api_json(f'{BASE}/api/frames/{ep}/{ts}/{w}/{w}')
        seen=set(); frames=[]
        for f in norm_frames(raw,ep):
            k=(f['episode'],f['timestamp'])
            if k not in seen: seen.add(k); frames.append(f)
        return jsonify({'frames':frames})
    except Exception as e:return jsonify({'error':str(e)}),502

@app.route('/api/visual-episode',methods=['POST'])
def visual_episode():
    d=request.get_json(silent=True) or {}; logs=[]
    try:
        ep=str(d.get('episode','')).upper(); interval=max(10,min(60,int(d.get('interval',30))))
        m=re.fullmatch(r'S(\d{2})E(\d{2})',ep)
        if not m:return jsonify({'error':'Invalid episode ID'}),400
        season,eno=int(m.group(1)),int(m.group(2))
        if season not in EP_COUNTS or eno<1 or eno>EP_COUNTS[season]:return jsonify({'error':'Episode outside S01-S10 range'}),400
        positives=d.get('positives',[]); negatives=d.get('negatives',[])
        if len(positives)<3:return jsonify({'error':'At least 3 positive examples are required'}),400
        logs.append(f'Loading training images for {ep}: {len(positives)} positive, {len(negatives)} negative.')
        pos=[]; neg=[]
        for i,x in enumerate(positives[:80],1):
            try: pos.append(feature_from_url(x['image']))
            except Exception as e: logs.append(f'Positive {i} failed: {type(e).__name__}: {e}')
        for i,x in enumerate(negatives[:120],1):
            try: neg.append(feature_from_url(x['image']))
            except Exception as e: logs.append(f'Negative {i} failed: {type(e).__name__}: {e}')
        if len(pos)<3:return jsonify({'error':'Fewer than 3 positive training images downloaded','logs':logs}),502
        candidates={}; probes=0; failures=0
        for sec in range(0,24*60,interval):
            ts=sec*1000; probes+=1
            try:
                fs=norm_frames(api_json(f'{BASE}/api/frames/{ep}/{ts}/1000/1000'),ep)
                if fs:
                    f=min(fs,key=lambda x:abs(x['timestamp']-ts)); candidates[(ep,f['timestamp'])]=f
            except Exception: failures+=1
        logs.append(f'{ep}: {probes} probes, {len(candidates)} unique real frames, {failures} probe failures.')
        scored=[]; image_failures=0
        for f in candidates.values():
            try:
                feat=feature_from_url(f['image']); scored.append({**f,'score':round(model_score(feat,pos,neg),6)})
            except Exception as e:
                image_failures+=1
                if image_failures<=5: logs.append(f'Image failed {f["timecode"]}: {type(e).__name__}: {e}')
        scored.sort(key=lambda x:x['score'],reverse=True)
        logs.append(f'{ep}: scored {len(scored)} frames, {image_failures} image failures.')
        return jsonify({'episode':ep,'scanned':len(scored),'results':scored,'logs':logs})
    except Exception as e:
        return jsonify({'error':f'{type(e).__name__}: {e}','trace':traceback.format_exc(limit=8),'logs':logs}),502

@app.route('/api/visual-scan-stream',methods=['POST'])
def visual_scan_stream():
    d=request.get_json(silent=True) or {}
    def emit(obj): return json.dumps(obj,separators=(',',':'))+'\n'
    @stream_with_context
    def generate():
        try:
            season=int(d.get('season',1)); interval=max(10,min(60,int(d.get('interval',30)))); top_n=max(10,min(200,int(d.get('top_n',60))))
            diagnostic=bool(d.get('diagnostic',False))
            if season not in EP_COUNTS:
                yield emit({'type':'error','message':'season must be 1-10'}); return
            positives=d.get('positives',[]); negatives=d.get('negatives',[])
            if len(positives)<3:
                yield emit({'type':'error','message':'At least 3 positive examples are required'}); return
            yield emit({'type':'log','message':f'Visual scan started. Season {season}, interval {interval}s, top {top_n}.'})
            yield emit({'type':'log','message':f'Loading training images: {len(positives)} positive, {len(negatives)} negative.'})
            pos=[]
            for i,x in enumerate(positives[:80],1):
                try: pos.append(feature_from_url(x['image']))
                except Exception as e:
                    yield emit({'type':'log','message':f'Positive image {i} failed: {type(e).__name__}: {e}'})
            neg=[]
            for i,x in enumerate(negatives[:120],1):
                try: neg.append(feature_from_url(x['image']))
                except Exception as e:
                    yield emit({'type':'log','message':f'Negative image {i} failed: {type(e).__name__}: {e}'})
            yield emit({'type':'log','message':f'Training images loaded: {len(pos)} positive, {len(neg)} negative.'})
            if len(pos)<3:
                yield emit({'type':'error','message':'Fewer than 3 positive training images could be downloaded. Check the image URLs shown in your saved examples.'}); return
            candidates={}; ep_total=1 if diagnostic else EP_COUNTS[season]
            for e in range(1,ep_total+1):
                ep=f'S{season:02d}E{e:02d}'; found=0; failures=0
                yield emit({'type':'log','message':f'Fetching {ep} ({e}/{ep_total})…'})
                for sec in range(0,24*60,interval):
                    ts=sec*1000
                    try:
                        raw=api_json(f'{BASE}/api/frames/{ep}/{ts}/1000/1000'); fs=norm_frames(raw,ep)
                        if fs:
                            f=min(fs,key=lambda x:abs(x['timestamp']-ts)); candidates[(ep,f['timestamp'])]=f; found+=1
                    except Exception: failures+=1
                yield emit({'type':'log','message':f'{ep}: {found} probes returned frames, {failures} probe failures, {len(candidates)} unique candidates total.'})
            yield emit({'type':'log','message':f'Analysing {len(candidates)} candidate images in small batches…'})
            scored=[]; failures=0
            vals=list(candidates.values())
            for i,f in enumerate(vals,1):
                try:
                    feat=feature_from_url(f['image']); score=model_score(feat,pos,neg); scored.append({**f,'score':round(score,6)})
                except Exception as e:
                    failures+=1
                    if failures<=10: yield emit({'type':'log','message':f'Image analysis failed for {f["episode"]} {f["timecode"]}: {type(e).__name__}: {e}'})
                if i%25==0 or i==len(vals): yield emit({'type':'log','message':f'Analysed {i}/{len(vals)} candidates. Successful {len(scored)}, failed {failures}.'})
            scored.sort(key=lambda x:x['score'],reverse=True)
            data={'season':season,'scanned':len(scored),'results':scored[:top_n]}
            yield emit({'type':'done','data':data})
        except Exception as e:
            yield emit({'type':'error','message':f'{type(e).__name__}: {e}','trace':traceback.format_exc(limit=8)})
    return Response(generate(),mimetype='application/x-ndjson',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

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

@app.route('/api/zine-zip',methods=['POST'])
def zine_zip():
    d=request.get_json(silent=True) or {}; items=d.get('items',[])
    if not isinstance(items,list) or not items:return jsonify({'error':'No zine items supplied'}),400
    if len(items)>300:return jsonify({'error':'Maximum 300 images per ZIP'}),400
    out=io.BytesIO(); index=[]
    try:
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            for x in sorted(items,key=lambda v:(str(v.get('episode','')),int(v.get('timestamp',0)))):
                ep=str(x.get('episode','')); ts=int(x.get('timestamp',0)); url=str(x.get('image',''))
                if not re.fullmatch(r'S\d{2}E\d{2}',ep) or url != img_url(ep,ts): continue
                title=re.sub(r'[^A-Za-z0-9._-]+','-',str(x.get('title','')).strip()).strip('-')[:70]
                name=f'{ep}/{tc(ts).replace(":","-").replace(".","-")}' + (f'_{title}' if title else '') + '.jpg'
                z.writestr(name,get(url,timeout=25).content)
                index.append({k:x.get(k,'') for k in ('episode','timestamp','timecode','title','category','notes','image')})
            import csv
            sio=io.StringIO(); w=csv.DictWriter(sio,fieldnames=['episode','timestamp','timecode','title','category','notes','image']); w.writeheader(); w.writerows(index)
            z.writestr('index.csv',sio.getvalue())
        out.seek(0); return send_file(out,mimetype='application/zip',as_attachment=True,download_name='kent-brockman-zine.zip')
    except Exception as e:return jsonify({'error':str(e)}),502

@app.route('/health')
def health(): return {'ok':True,'version':'2.4','methods':['dialogue','visual-training','zine-library','zip-download']}

if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.getenv('PORT','5000')))
