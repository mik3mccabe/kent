import os, io, math, requests, zipfile, re, json, traceback, time, html as htmlmod, csv, threading
from difflib import SequenceMatcher
import numpy as np
from PIL import Image, ImageFilter
from flask import Flask, request, jsonify, render_template_string, send_file, Response, stream_with_context

app=Flask(__name__)
BASE=os.getenv('FRINKIAC_BASE','https://frinkiac.com').rstrip('/')
UA={'User-Agent':'kent-brockman-zine-research/3.1'}
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

HTML=r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kent Brockman Frame Finder v3.1</title>
<style>body{font-family:system-ui,sans-serif;max-width:1320px;margin:26px auto;padding:0 18px;color:#111}h1{margin:0}.sub{color:#555}.tabs{display:flex;gap:8px;margin:20px 0}.tabs button,.btn{padding:9px 12px;font:inherit}.panel{display:none}.panel.on{display:block}.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin:16px 0}input,select,button{font:inherit;padding:8px}label{display:flex;gap:6px;align-items:center}.scene{border-top:2px solid #111;margin:26px 0;padding-top:14px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}.card{border:1px solid #ccc;border-radius:8px;padding:8px}.card img{width:100%;display:block;background:#eee;min-height:110px}.meta{font-size:13px;margin-top:6px}.actions{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:7px}.actions button{font-size:12px;padding:7px}.good,.zine{background:#111;color:white}.bad{background:#ddd}.status{padding:10px;background:#f4f4f4;margin:12px 0}.error{background:#fee;padding:10px}.score{font-weight:700}.tiny{font-size:13px;color:#555}.adjust-overlay[hidden]{display:none}.adjust-overlay{position:fixed;z-index:1000;inset:0;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:12px;overflow-y:auto}.adjust-dialog{position:relative;background:#fff;border-radius:10px;padding:18px;width:min(680px,100%);max-height:95vh;overflow-y:auto;box-shadow:0 10px 35px #1117}.adjust-preview{width:100%;max-height:360px;object-fit:contain;background:#eee;margin-top:8px}.adjust-control-row{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin:10px 0}.adjust-control-row button{padding:8px 10px}.adjust-control-row input{width:205px;max-width:100%}.adjust-dialog .secondary{font-size:13px;color:#555}.adjust-dialog .close{float:right}.adjust-actions{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}.adjust-actions button{padding:10px 15px}.adjust-actions .save{background:#111;color:#fff}.adjust-actions .save:disabled{opacity:.45;cursor:not-allowed}</style></head><body>
<h1>Kent Brockman Frame Finder <small>v3.1</small></h1><div class="sub">Dialogue search and trainable visual scan feed the same zine library.</div>
<div class="tabs"><button onclick="tab('dialogue')">Dialogue Search</button><button onclick="tab('visual')">Visual Scan</button><button onclick="tab('script')">Script Scan</button><button onclick="tab('library')">Zine Library</button></div>
<section id="dialogue" class="panel on"><form class="bar" method="get"><input type="hidden" name="mode" value="dialogue"><input name="q" value="{{q}}" placeholder="e.g. this is Kent Brockman" size="38"><label>Seconds either side <input name="window" value="{{window}}" type="number" min="1" max="30" style="width:65px"></label><label>Max raw matches <input name="limit" value="{{limit}}" type="number" min="1" max="500" style="width:72px"><label><input type="checkbox" name="all" value="1" {% if scan_all %}checked{% endif %}> Scan all matches</label><label><input type="checkbox" id="hideReviewed" onchange="applyReviewed()"> Hide reviewed scenes</label></label><button>Search</button></form>
{% if error %}<div class="error">{{error}}</div>{% endif %}
{% for s in scenes %}<section class="scene" data-scene="{{s.scene_id}}"><h2>{{s.episode}} · {{s.start_timecode}}–{{s.end_timecode}}</h2><div class="tiny">{{s.dialogue}}</div><div class="bar"><button type="button" onclick="markScene(this)">Mark scene reviewed</button><button type="button" onclick="expandScene(this,'{{s.episode}}',{{s.midpoint}})">Expand ±30 sec</button></div><div class="grid">{% for x in s.frames %}<div class="card" data-id="{{x.episode}}-{{x.timestamp}}"><a href="{{x.image}}" target="_blank"><img src="{{x.image}}" loading="lazy"></a><div class="meta"><b>{{x.episode}}</b> · {{x.timecode}}<br>{{x.timestamp}} ms</div><div class="tiny"><a href="https://frinkiac.com/caption/{{x.episode}}/{{x.timestamp}}" target="_blank" rel="noopener noreferrer">View on Frinkiac</a></div><div class="actions"><button onclick="labelFrame(this,'good')" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}">Good example</button><button onclick="labelFrame(this,'news')" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}">News, no graphic</button><button onclick="labelFrame(this,'bad')" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}">Not relevant</button><button onclick="labelFrame(this,'zine')" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}">Save to zine</button></div><button class="btn" type="button" style="margin-top:7px;width:100%;font-size:12px" data-ep="{{x.episode}}" data-ts="{{x.timestamp}}" data-tc="{{x.timecode}}" data-img="{{x.image}}" onclick="openAdjustFrame(this)">Adjust timestamp / Add to zine</button></div>{% endfor %}</div></section>{% endfor %}</section>
<section id="visual" class="panel"><div class="status" id="trainStatus"></div><div class="bar"><label>Season <select id="season">{% for n in range(1,11) %}<option value="{{n}}">Season {{n}}</option>{% endfor %}</select></label><label>Sample every <select id="interval"><option value="30">30 sec</option><option value="20">20 sec</option><option value="15">15 sec</option><option value="10">10 sec</option></select></label><label>Top results <input id="topn" type="number" value="60" min="10" max="200" style="width:65px"></label><button onclick="visualScan(false,false)">Start new scan</button><button onclick="resumeVisualScan()">Resume scan</button><button onclick="visualScan(true,false)">Diagnostic: 1 episode</button><button id="stopScan" onclick="stopVisualScan()" disabled>Stop scan</button><button onclick="clearScanProgress()">Clear scan progress</button></div><div class="tiny">Start with 20+ Good examples. Each episode is now a separate short request. Progress and ranked candidates are saved in this browser after every episode. Resume continues after a stop, timeout, refresh, or failed episode.</div><div id="scanMsg" class="status" style="display:none"></div><details open><summary>Visual Scan Log</summary><div class="bar"><button onclick="copyLogs()">Copy logs</button><button onclick="clearLogs()">Clear logs</button></div><pre id="scanLog" style="white-space:pre-wrap;background:#111;color:#eee;padding:12px;max-height:320px;overflow:auto;border-radius:6px">Ready.</pre></details><div id="visualGrid" class="grid"></div></section>
<section id="script" class="panel"><h2>Kent dialogue — speaker-attributed</h2><p><strong>Primary result: Kent's actual spoken lines.</strong> The source identifies Kent Brockman by character ID 211. Frinkiac matches are checked separately so slow image searches cannot hide the speaker-attributed lines. Use Match all to Frinkiac after loading dialogue. Approximate dataset times are unverified.</p><div class="bar"><label>Season <select id="scriptSeason" onchange="loadScriptEpisodes()">{% for n in range(1,11) %}<option value="{{n}}">Season {{n}}</option>{% endfor %}</select></label><label>Episode with Kent dialogue <select id="scriptEpisode" style="min-width:180px"><option value="">Load episodes first</option></select></label><button onclick="loadScriptEpisodes()">Load Kent episodes</button><button onclick="runScriptScan(true)">Test selected episode</button><button onclick="runScriptScan(false)">Scan season</button><button onclick="verifyAllScriptMatches()">Match all to Frinkiac</button><button onclick="exportKentDialogue()">Export Kent dialogue CSV</button><button onclick="stopScriptScan()">Stop</button><button onclick="clearScriptScan()">Clear script results</button></div><div id="scriptStatus" class="status">Select a season and click Load Kent episodes to see how many attributed lines are available.</div><div id="scriptCoverage" class="tiny"></div><div id="scriptResults"></div></section>
<section id="library" class="panel"><div class="bar"><button onclick="downloadZip()">Download Zine ZIP</button><button onclick="exportZine()">Export metadata CSV</button><button onclick="clearLabels()">Clear all labels</button></div><div id="libraryStatus" class="status"></div><div class="tiny">Saved zine frames appear here in season, episode and timestamp order. Add the exact on-screen wording before downloading your archive.</div><div id="libraryGrid" class="grid"></div></section>
<div id="adjustModal" class="adjust-overlay" hidden role="dialog" aria-modal="true" aria-labelledby="adjustHeading" onclick="if(event.target===this)closeAdjust()"><div class="adjust-dialog"><button class="close" type="button" onclick="closeAdjust()" aria-label="Close adjustment">Close ×</button><h2 id="adjustHeading">Adjust timestamp · Add to zine</h2><p class="secondary" id="adjustOriginal"></p><img class="adjust-preview" id="adjustImage" alt="Current selected Frinkiac frame"><p class="secondary" id="adjustCurrent">Preview the new timestamp before saving.</p><label for="adjustTime" style="display:block;margin-top:8px"><b>Time or Frinkiac page URL</b></label><div class="adjust-control-row"><input id="adjustTime" type="text" placeholder="05:12.000 or Frinkiac URL" oninput="invalidateAdjustPreview()" onkeydown="if(event.key==='Enter'){event.preventDefault();previewAdjust()}"><button type="button" onclick="previewAdjust()">Preview frame</button></div><div class="adjust-control-row"><button type="button" onclick="nudgeAdjust(-5000)">−5s</button><button type="button" onclick="nudgeAdjust(-1000)">−1s</button><button type="button" onclick="nudgeAdjust(-100)">−0.1s</button><button type="button" onclick="nudgeAdjust(100)">+0.1s</button><button type="button" onclick="nudgeAdjust(1000)">+1s</button><button type="button" onclick="nudgeAdjust(5000)">+5s</button></div><p class="secondary">Enter mm:ss.mmm, a millisecond timestamp, or paste a Frinkiac caption URL from the same episode. Preview snaps to the nearest available Frinkiac screenshot.</p><p id="adjustStatus" class="status" role="status">Choose a time, then preview the frame.</p><div class="adjust-actions"><a id="adjustFrinkiac" href="#" target="_blank" rel="noopener noreferrer">View on Frinkiac</a><button id="adjustSave" class="save" type="button" onclick="saveAdjustedFrame()" disabled>Save adjusted frame to zine</button></div></div></div>
<script>
const K='kent-v2-labels',R='kent-v21-reviewed';function labels(){try{return JSON.parse(localStorage.getItem(K)||'{}')}catch(e){return {}}}function saveLabels(x){localStorage.setItem(K,JSON.stringify(x));refresh()}
function tab(id){document.querySelectorAll('.panel').forEach(x=>x.classList.remove('on'));document.getElementById(id).classList.add('on');refresh()}
function dataFrom(b){return {episode:b.dataset.ep,timestamp:+b.dataset.ts,timecode:b.dataset.tc,image:b.dataset.img}}
function labelFrame(b,label){let x=labels(),d=dataFrom(b),id=d.episode+'-'+d.timestamp;if(x[id]&&x[id].label===label)delete x[id];else x[id]={...d,label};saveLabels(x)}
function refresh(){let x=labels(),a=Object.values(x),g=a.filter(v=>v.label==='good').length,b=a.filter(v=>v.label==='bad').length,z=a.filter(v=>v.label==='zine').length,n=a.filter(v=>v.label==='news').length;document.getElementById('trainStatus').textContent=`Training set: ${g} good + ${z} zine positives, ${n} news/no-graphic, ${b} negatives.`;document.getElementById('libraryStatus').textContent=`${g} good examples · ${n} news/no-graphic · ${b} negatives · ${z} saved zine frames`;document.querySelectorAll('.card').forEach(c=>{let v=x[c.dataset.id];c.querySelectorAll('.actions button').forEach(q=>q.classList.remove('good','bad','zine'));if(v){let btn=[...c.querySelectorAll('.actions button')].find(q=>q.textContent.toLowerCase().includes(v.label==='good'?'good':v.label==='bad'?'not':v.label==='news'?'news':'zine'));if(btn)btn.classList.add(v.label)}});renderLibrary()}
function cardHTML(v,score=''){return `<div class="card" data-id="${v.episode}-${v.timestamp}"><a href="${v.image}" target="_blank"><img src="${v.image}" loading="lazy"></a><div class="meta"><b>${v.episode}</b> · ${v.timecode||''}${score?`<br><span class="score">Visual score ${score}</span>`:''}</div><div class="tiny"><a href="https://frinkiac.com/caption/${encodeURIComponent(v.episode)}/${Number(v.timestamp)}" target="_blank" rel="noopener noreferrer">View on Frinkiac</a></div><div class="actions"><button data-ep="${v.episode}" data-ts="${v.timestamp}" data-tc="${v.timecode||''}" data-img="${v.image}" onclick="labelFrame(this,'good')">Good example</button><button data-ep="${v.episode}" data-ts="${v.timestamp}" data-tc="${v.timecode||''}" data-img="${v.image}" onclick="labelFrame(this,'news')">News, no graphic</button><button data-ep="${v.episode}" data-ts="${v.timestamp}" data-tc="${v.timecode||''}" data-img="${v.image}" onclick="labelFrame(this,'bad')">Not relevant</button><button data-ep="${v.episode}" data-ts="${v.timestamp}" data-tc="${v.timecode||''}" data-img="${v.image}" onclick="labelFrame(this,'zine')">Save to zine</button></div><button class="btn" style="margin-top:7px;width:100%;font-size:12px" data-ep="${v.episode}" data-ts="${v.timestamp}" data-tc="${v.timecode||''}" data-img="${v.image}" onclick="openAdjustFrame(this)">Adjust timestamp / Add to zine</button></div>`}
function renderLibrary(){let el=document.getElementById('libraryGrid');if(!el)return;let a=Object.values(labels()).filter(v=>v.label==='zine').sort((a,b)=>a.episode.localeCompare(b.episode)||a.timestamp-b.timestamp);el.innerHTML=a.map(v=>`<div class="card" data-id="${v.episode}-${v.timestamp}"><a href="${v.image}" target="_blank"><img src="${v.image}" loading="lazy"></a><div class="meta"><b>${v.episode}</b> · ${v.timecode||''}<br>${v.timestamp} ms${Number.isFinite(+v.original_timestamp)&&(+v.original_timestamp)!==(+v.timestamp)?`<br><span class="tiny">Original: ${formatAdjustTime(+v.original_timestamp)}</span>`:''}</div><input style="width:calc(100% - 18px);margin-top:7px" placeholder="Exact card wording" value="${escHtml(v.title||'')}" onchange="editZine('${v.episode}-${v.timestamp}','title',this.value)"><select style="width:100%;margin-top:6px" onchange="editZine('${v.episode}-${v.timestamp}','category',this.value)"><option ${!v.category?'selected':''}>Category</option><option ${v.category==='Story graphic'?'selected':''}>Story graphic</option><option ${v.category==='Programme title'?'selected':''}>Programme title</option><option ${v.category==='Smartline'?'selected':''}>Smartline</option><option ${v.category==='Eye on Springfield'?'selected':''}>Eye on Springfield</option><option ${v.category==='Channel 6'?'selected':''}>Channel 6</option></select><textarea style="width:calc(100% - 18px);margin-top:6px" placeholder="Notes" onchange="editZine('${v.episode}-${v.timestamp}','notes',this.value)">${escHtml(v.notes||'')}</textarea><div class="tiny"><a href="https://frinkiac.com/caption/${encodeURIComponent(v.episode)}/${Number(v.timestamp)}" target="_blank" rel="noopener noreferrer">View on Frinkiac</a></div><div class="actions"><button onclick="openAdjustSaved('${v.episode}-${v.timestamp}')">Adjust timestamp</button><button onclick="downloadOne('${v.image}','${v.episode}',${v.timestamp})">Download JPG</button><button class="bad" onclick="removeZine('${v.episode}-${v.timestamp}')">Remove</button></div></div>`).join('')||'<div class="status">No saved zine frames yet.</div>'}
function escHtml(s){return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;')}
function editZine(id,k,v){let x=labels();if(x[id]){x[id][k]=v==='Category'?'':v;saveLabels(x)}}
function removeZine(id){let x=labels();delete x[id];saveLabels(x)}
// One adjustment dialog works with Dialogue Search, Visual Scan, Script Scan and saved Zine Library items.
let adjustContext=null,adjustVerified=null,adjustRequestId=0;
function formatAdjustTime(ms){let secs=Math.floor(ms/1000),frac=Math.round(ms%1000);return `${String(Math.floor(secs/60)).padStart(2,'0')}:${String(secs%60).padStart(2,'0')}.${String(frac).padStart(3,'0')}`}
function parseAdjustTime(text,episode){let v=String(text||'').trim();if(!v)throw Error('Enter a time first.');let url=v.match(/(?:https?:\/\/)?(?:www\.)?frinkiac\.com\/caption\/(S\d{2}E\d{2})\/(\d+)/i);if(url){if(url[1].toUpperCase()!==episode)throw Error('That Frinkiac URL is from a different episode.');v=url[2]}
  let ms;if(/^\d+$/.test(v)){ms=Number(v)}else{let m=v.match(/^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$/);if(!m||Number(m[3])>=60||(m[1]!==undefined&&Number(m[2])>=60))throw Error('Use mm:ss.mmm, e.g. 08:24.500, or paste a Frinkiac page URL.');ms=((Number(m[1]||0)*3600)+(Number(m[2])*60)+Number(m[3]))*1000+Number((m[4]||'').padEnd(3,'0'))}
  if(!Number.isSafeInteger(ms)||ms<0||ms>3600000)throw Error('Timestamp must be within 00:00 and 60:00.');return ms}
function beginAdjust(d,existingId=null){if(!d||!/^S\d{2}E\d{2}$/.test(d.episode)||!Number.isFinite(+d.timestamp)){alert('Invalid frame details.');return}adjustContext={data:{...d},existingId};adjustVerified=null;adjustRequestId++;document.getElementById('adjustModal').hidden=false;document.getElementById('adjustHeading').textContent=`${d.episode} · Adjust timestamp`;let orig=Number.isFinite(+d.original_timestamp)?+d.original_timestamp:+d.timestamp;document.getElementById('adjustOriginal').textContent=`Original reference: ${formatAdjustTime(orig)} (${orig} ms). Current: ${formatAdjustTime(+d.timestamp)}.`;document.getElementById('adjustTime').value=formatAdjustTime(+d.timestamp);document.getElementById('adjustImage').src=d.image;document.getElementById('adjustCurrent').textContent='Current screenshot. Change the time and click Preview frame.';document.getElementById('adjustStatus').textContent='The saved frame will use the actual timestamp returned by Frinkiac.';document.getElementById('adjustSave').disabled=true;document.getElementById('adjustFrinkiac').href=`https://frinkiac.com/caption/${d.episode}/${Number(d.timestamp)}`;document.getElementById('adjustTime').focus()}
function openAdjustFrame(b){let d=dataFrom(b),id=d.episode+'-'+d.timestamp,saved=labels()[id];beginAdjust(saved&&saved.label==='zine'?saved:d,saved&&saved.label==='zine'?id:null)}
function openAdjustAtEpisode(ep,ts){if(!/^S\d{2}E\d{2}$/.test(ep)||!Number.isFinite(+ts))return;let t=Math.max(0,Math.round(+ts));beginAdjust({episode:ep,timestamp:t,timecode:formatAdjustTime(t),image:`https://frinkiac.com/img/${ep}/${t}.jpg`});previewAdjust()}
function openAdjustSaved(id){let d=labels()[id];if(!d||d.label!=='zine'){alert('Saved frame was not found.');return}beginAdjust(d,id)}
function closeAdjust(){adjustRequestId++;document.getElementById('adjustModal').hidden=true;adjustContext=null;adjustVerified=null}
function invalidateAdjustPreview(){adjustVerified=null;adjustRequestId++;document.getElementById('adjustSave').disabled=true;document.getElementById('adjustStatus').textContent='Time changed. Click Preview frame to confirm the new image.'}
function nudgeAdjust(delta){if(!adjustContext)return;try{let value=parseAdjustTime(document.getElementById('adjustTime').value,adjustContext.data.episode)+delta;if(value<0||value>3600000)throw Error('Timestamp is outside the allowed range.');document.getElementById('adjustTime').value=formatAdjustTime(value);invalidateAdjustPreview();previewAdjust()}catch(e){document.getElementById('adjustStatus').textContent=e.message}}
async function previewAdjust(){if(!adjustContext)return;let ep=adjustContext.data.episode,ms;try{ms=parseAdjustTime(document.getElementById('adjustTime').value,ep)}catch(e){invalidateAdjustPreview();document.getElementById('adjustStatus').textContent=e.message;return}let token=++adjustRequestId;adjustVerified=null;document.getElementById('adjustSave').disabled=true;document.getElementById('adjustStatus').textContent=`Finding nearest available screenshot to ${formatAdjustTime(ms)}…`;try{let r=await fetch('/api/frame-at-time?episode='+encodeURIComponent(ep)+'&timestamp='+ms);let d=await r.json();if(token!==adjustRequestId||!adjustContext)return;if(!r.ok)throw Error(d.error||`HTTP ${r.status}`);if(d.episode!==ep||!Number.isFinite(+d.timestamp))throw Error('Frinkiac returned an invalid frame.');adjustVerified={...d,requested_timestamp:ms};document.getElementById('adjustImage').src=d.image;document.getElementById('adjustCurrent').textContent=`Preview: ${d.episode} · ${d.timecode} (${d.timestamp} ms)`;document.getElementById('adjustStatus').textContent=`Requested ${formatAdjustTime(ms)}; nearest real Frinkiac frame ${d.timecode} (${Number(d.timestamp)-ms>=0?'+':''}${Number(d.timestamp)-ms} ms difference).`;document.getElementById('adjustFrinkiac').href=`https://frinkiac.com/caption/${ep}/${d.timestamp}`;document.getElementById('adjustSave').disabled=false}catch(e){if(token!==adjustRequestId)return;document.getElementById('adjustStatus').textContent=`Preview failed: ${e.message}. Frame was not saved.`}}
function saveAdjustedFrame(){if(!adjustContext||!adjustVerified){alert('Preview a valid frame before saving.');return}let ctx=adjustContext,x=labels(),orig=Number.isFinite(+ctx.data.original_timestamp)?+ctx.data.original_timestamp:+ctx.data.timestamp,d=adjustVerified,id=d.episode+'-'+d.timestamp;let existing=x[id];if(existing&&existing.label==='zine'&&ctx.existingId!==id&&!confirm('This screenshot is already in your Zine Library. Replace its details?'))return;
  let merged={...ctx.data,...d,label:'zine',episode:d.episode,timestamp:d.timestamp,timecode:d.timecode,image:d.image,original_timestamp:orig,original_timecode:formatAdjustTime(orig),requested_timestamp:d.requested_timestamp,requested_timecode:formatAdjustTime(d.requested_timestamp)};if(ctx.existingId&&ctx.existingId!==id)delete x[ctx.existingId];x[id]={...existing,...merged};closeAdjust();saveLabels(x);tab('library')}
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
async function exportZine(){let a=Object.values(labels()).filter(v=>v.label==='zine').sort((x,y)=>x.episode.localeCompare(y.episode)||x.timestamp-y.timestamp);if(!a.length){alert('Save at least one frame first.');return}let r=await fetch('/api/subtitles-batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({items:a.map(v=>({episode:v.episode,timestamp:v.timestamp}))})});if(!r.ok){alert('Could not fetch subtitles. Please try again.');return}let d=await r.json();let cols=['episode','timestamp','timecode','original_timestamp','original_timecode','adjusted_timestamp','adjusted_timecode','requested_timestamp','requested_timecode','title','category','notes','subtitle_text','image','frinkiac_url'];let field=(v,c)=>c==='original_timestamp'?(v.original_timestamp??v.timestamp):c==='original_timecode'?(v.original_timecode??v.timecode):c==='adjusted_timestamp'?v.timestamp:c==='adjusted_timecode'?v.timecode:c==='requested_timestamp'?(v.requested_timestamp??v.timestamp):c==='requested_timecode'?(v.requested_timecode??v.timecode):c==='frinkiac_url'?`https://frinkiac.com/caption/${v.episode}/${v.timestamp}`:c==='subtitle_text'?(d.subtitles[v.episode+'-'+v.timestamp]||''):v[c];let esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';let csv=[cols.join(','),...a.map(v=>cols.map(c=>esc(field(v,c))).join(','))].join('\r\n');let u=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'})),q=document.createElement('a');q.href=u;q.download='kent-brockman-zine.csv';q.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
function reviewed(){try{return JSON.parse(localStorage.getItem(R)||'{}')}catch(e){return {}}}
function markScene(b){let s=b.closest('.scene'),x=reviewed();x[s.dataset.scene]=true;localStorage.setItem(R,JSON.stringify(x));applyReviewed()}
function applyReviewed(){let h=document.getElementById('hideReviewed')?.checked,x=reviewed();document.querySelectorAll('.scene').forEach(s=>s.style.display=(h&&x[s.dataset.scene])?'none':'')}
async function expandScene(b,ep,ts){b.disabled=true;b.textContent='Loading…';try{let r=await fetch(`/api/nearby?episode=${ep}&timestamp=${ts}&window=30`),d=await r.json();if(!r.ok)throw new Error(d.error||'Failed');let grid=b.closest('.scene').querySelector('.grid');grid.innerHTML=d.frames.map(v=>cardHTML(v)).join('');refresh()}catch(e){alert(e.message)}finally{b.disabled=false;b.textContent='Expand ±30 sec'}}
function clearLabels(){if(confirm('Clear training labels and saved zine frames?')){localStorage.removeItem(K);refresh()}}
const SCRIPTKEY='kent-v30-script';let scriptStop=false,scriptEpisodeIndex=null;
function scriptData(){try{return JSON.parse(localStorage.getItem(SCRIPTKEY)||'null')}catch(e){return null}}
function clearScriptScan(){localStorage.removeItem(SCRIPTKEY);document.getElementById('scriptResults').innerHTML='';document.getElementById('scriptStatus').textContent='Script results cleared. Your saved zine frames and training labels are untouched.'}
function stopScriptScan(){scriptStop=true;document.getElementById('scriptStatus').textContent='Stopping after current episode. Completed episode results are saved.'}
async function loadScriptEpisodes(){let season=+document.getElementById('scriptSeason').value,status=document.getElementById('scriptStatus'),el=document.getElementById('scriptEpisode');status.textContent='Loading speaker-attributed dataset and finding Kent episodes…';try{let r=await fetch('/api/script-episodes?season='+season),d=await r.json();if(!r.ok)throw Error(d.error||'Dataset unavailable');scriptEpisodeIndex=d;el.innerHTML=d.episodes.map(e=>`<option value="${e.episode}">${e.episode} — ${e.line_count} Kent lines</option>`).join('')||'<option value="">No attributed Kent lines found</option>';document.getElementById('scriptCoverage').textContent=`${d.episodes.length} of ${d.total_episodes} episodes contain ${d.total_lines} attributed Kent lines. Character ID ${d.character_ids.join(', ')}. Source: ${d.source}`;status.textContent=d.episodes.length?`Ready. First available episode: ${d.episodes[0].episode}. Test this episode or scan the season.`:'No Kent-attributed lines in this season of this dataset. The dataset may be incomplete; Visual Scan is still useful.';return d}catch(e){status.textContent='Dataset load failed: '+e.message;el.innerHTML='<option value="">Unavailable</option>';return null}}
function scriptMatchKey(m){return m.episode+'-'+m.dataset_timestamp}
function showScriptResults(){let d=scriptData(),el=document.getElementById('scriptResults');if(!d){el.innerHTML='';return}el.innerHTML=d.matches.map(m=>{let time=m.timecode||'No verified Frinkiac match',status=m.match_status==='verified'?'Frinkiac subtitle verified':m.match_status==='pending'?'Frinkiac matching not yet attempted':'No verified subtitle match';let near=Number.isFinite(m.dataset_timestamp)?`<a href="https://frinkiac.com/caption/${encodeURIComponent(m.episode)}/${m.dataset_timestamp}" target="_blank" rel="noopener noreferrer">Inspect dataset time on Frinkiac (approximate)</a>`:'';let verified=m.timestamp!==null?`<a href="https://frinkiac.com/caption/${encodeURIComponent(m.episode)}/${m.timestamp}" target="_blank" rel="noopener noreferrer">Open verified Frinkiac match</a>`:'';let key=scriptMatchKey(m);return `<section class="scene"><h3>${escHtml(m.episode)} · ${escHtml(time)}</h3><p><b>${escHtml(status)}</b> · Kent Brockman (ID 211) · ${m.line_count} spoken lines<br><small>Dataset time: ${escHtml(m.dataset_timecode)}</small></p><p>${near}${verified?' · '+verified:''}</p><button onclick="verifyScriptGroup('${key.replaceAll("'",'')}')">${m.match_status==='verified'?'Recheck':'Find matching Frinkiac frames'}</button> <button onclick="openAdjustAtEpisode('${m.episode}',${m.timestamp!==null?m.timestamp:m.dataset_timestamp})">Choose timestamp / Add to zine</button>${m.matched_subtitle?`<p class="tiny">Matched Frinkiac subtitle: ${escHtml(m.matched_subtitle)} (score ${m.match_score})</p>`:''}<details open><summary>Read ${m.line_count} Kent-attributed lines</summary><ol>${m.lines.map(l=>`<li><strong>${escHtml(l.timecode)}</strong> ${escHtml(l.dialogue)}</li>`).join('')}</ol></details>${m.frames&&m.frames.length?`<div class="grid">${m.frames.map(f=>cardHTML(f)).join('')}</div>`:'<div class="tiny">No verified screenshots yet. Approximate dataset timestamps may not align perfectly with Frinkiac.</div>'}</section>`}).join('')||'<div class="status">No Kent dialogue saved for this scan. Use Load Kent episodes to check dataset coverage.</div>';refresh()}
async function verifyScriptGroup(key){let d=scriptData();if(!d)return false;let m=d.matches.find(x=>scriptMatchKey(x)===key);if(!m)return false;let status=document.getElementById('scriptStatus');status.textContent=`Matching Kent's dialogue in ${m.episode} to Frinkiac…`;try{let r=await fetch('/api/script-match',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode:m.episode,line_ids:m.lines.map(x=>x.line_id)})});let x=await r.json();if(!r.ok)throw Error(x.error||`HTTP ${r.status}`);Object.assign(m,x);localStorage.setItem(SCRIPTKEY,JSON.stringify(d));showScriptResults();status.textContent=`${m.episode}: ${m.match_status==='verified'?'Verified Frinkiac subtitle match at '+m.timecode:'No exact Frinkiac subtitle match. Kent’s original dialogue is still shown.'}`;return true}catch(e){status.textContent=`Frinkiac lookup failed: ${e.message}. Kent dialogue is preserved.`;return false}}
async function verifyAllScriptMatches(){let d=scriptData();if(!d||!d.matches.length){document.getElementById('scriptStatus').textContent='Load Kent dialogue first.';return}scriptStop=false;let remaining=d.matches.filter(m=>m.match_status==='pending');for(let i=0;i<remaining.length;i++){if(scriptStop)break;await verifyScriptGroup(scriptMatchKey(remaining[i]));document.getElementById('scriptStatus').textContent=`Frinkiac matching: ${i+1}/${remaining.length} scenes processed. ${scriptData().matches.filter(m=>m.match_status==='verified').length} verified. All attributed dialogue remains saved.`}if(!scriptStop)document.getElementById('scriptStatus').textContent=`Matching complete. ${scriptData().matches.filter(m=>m.match_status==='verified').length} verified scene groups. Unmatched groups retain their attributed dialogue.`}
function exportKentDialogue(){let d=scriptData();if(!d||!d.matches.length){alert('Load Kent dialogue first.');return}let columns=['episode','character_id','dataset_timestamp','dataset_timecode','spoken_words','frinkiac_timestamp','match_status','frinkiac_url'];let rows=[];for(let m of d.matches)for(let line of m.lines)rows.push({episode:m.episode,character_id:211,dataset_timestamp:line.timestamp,dataset_timecode:line.timecode,spoken_words:line.dialogue,frinkiac_timestamp:m.timestamp??'',match_status:m.match_status,frinkiac_url:m.timestamp!==null?'https://frinkiac.com/caption/'+m.episode+'/'+m.timestamp:''});let quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';let csv=[columns.join(','),...rows.map(r=>columns.map(k=>quote(r[k])).join(','))].join('\r\n');let blob=new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='kent-brockman-attributed-dialogue.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
async function runScriptScan(one){let season=+document.getElementById('scriptSeason').value,status=document.getElementById('scriptStatus');if(!scriptEpisodeIndex||scriptEpisodeIndex.season!==season){if(!await loadScriptEpisodes())return}let eps=one?[document.getElementById('scriptEpisode').value]:scriptEpisodeIndex.episodes.map(e=>e.episode);eps=eps.filter(Boolean);if(!eps.length){status.textContent='No attributed Kent dialogue episodes found. Nothing to scan.';return}let d=scriptData();if(!d||d.season!==season)d={season,completed:[],matches:[],errors:[]};scriptStop=false;for(let ep of eps){if(scriptStop)break;if(d.completed.includes(ep)&&!one)continue;if(one){d.completed=d.completed.filter(x=>x!==ep);d.matches=d.matches.filter(x=>x.episode!==ep)}status.textContent=`Loading Kent's dialogue from ${ep} (${d.completed.length} episodes saved)…`;try{let r=await fetch('/api/script-episode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode:ep})});let x=await r.json();if(!r.ok)throw Error(x.error||`HTTP ${r.status}`);d.matches=d.matches.filter(x=>x.episode!==ep).concat(x.matches);d.completed.push(ep);localStorage.setItem(SCRIPTKEY,JSON.stringify(d));showScriptResults();status.textContent=`${ep}: ${x.labelled_turns} Kent-attributed lines, ${x.matches.length} scene groups, ${x.verified_matches} verified Frinkiac matches. ${d.completed.length}/${eps.length} episodes recorded. Results saved.`}catch(e){d.errors.push(ep+': '+e.message);localStorage.setItem(SCRIPTKEY,JSON.stringify(d));status.textContent=`${ep}: ${e.message}. Previous results saved. Continuing.`}}if(!scriptStop){let total=d.matches.reduce((n,m)=>n+(m.line_count||0),0),verified=d.matches.filter(m=>m.match_status==='verified').length;status.textContent=`Completed ${d.completed.length} Kent-containing episodes: ${total} attributed lines in ${d.matches.length} scene groups; ${verified} verified Frinkiac matches; ${d.errors.length} errors. Results are not limited to successful image matches.`}showScriptResults();refresh()}
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

@app.route('/api/frame-at-time')
def frame_at_time():
    """Snap a requested millisecond timestamp to the nearest actual Frinkiac frame."""
    ep=request.args.get('episode','').strip().upper()
    if not re.fullmatch(r'S\d{2}E\d{2}',ep):
        return jsonify({'error':'Invalid episode code'}),400
    try: ts=int(request.args.get('timestamp',''))
    except (ValueError,TypeError): return jsonify({'error':'Timestamp must be milliseconds'}),400
    if not 0 <= ts <= 3600000:
        return jsonify({'error':'Timestamp outside allowed range'}),400
    try:
        # The first small window keeps edits responsive; the larger fallback covers gaps.
        frames=[]
        for radius in (2500,8000):
            response=api_json(f'{BASE}/api/frames/{ep}/{ts}/{radius}/{radius}')
            frames=[f for f in norm_frames(response,ep) if f['episode']==ep]
            if frames:break
        if not frames:return jsonify({'error':'No Frinkiac screenshots available near this timestamp'}),404
        frame=min(frames,key=lambda f:(abs(f['timestamp']-ts),f['timestamp']))
        if abs(frame['timestamp']-ts)>8000:
            return jsonify({'error':'No screenshot found within 8 seconds of the requested time'}),404
        return jsonify({**frame,'requested_timestamp':ts,'offset_ms':frame['timestamp']-ts})
    except requests.RequestException as e:
        return jsonify({'error':'Frinkiac could not be reached: '+str(e)}),502
    except (ValueError,TypeError,KeyError) as e:
        return jsonify({'error':'Invalid Frinkiac response: '+str(e)}),502

@app.route('/api/subtitles-batch',methods=['POST'])
def subtitles_batch():
    d=request.get_json(silent=True) or {}
    items=d.get('items',[])
    if not isinstance(items,list) or len(items)>300:
        return jsonify({'error':'Expected at most 300 frames'}),400
    out={}
    for item in items:
        try:
            ep=str(item.get('episode','')); ts=int(item.get('timestamp',0))
            if not re.fullmatch(r'S\d{2}E\d{2}',ep) or ts<0: continue
            out[f'{ep}-{ts}']=caption(ep,ts)
        except (TypeError,ValueError,AttributeError): continue
    return jsonify({'subtitles':out})

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
                index.append({**{k:x.get(k,'') for k in ('episode','timestamp','timecode','title','category','notes','image')},'original_timestamp':x.get('original_timestamp',ts),'original_timecode':x.get('original_timecode',tc(ts)),'adjusted_timestamp':ts,'adjusted_timecode':tc(ts),'requested_timestamp':x.get('requested_timestamp',ts),'requested_timecode':x.get('requested_timecode',tc(ts)),'subtitle_text':caption(ep,ts),'frinkiac_url':f'https://frinkiac.com/caption/{ep}/{ts}'})
            import csv
            sio=io.StringIO(); w=csv.DictWriter(sio,fieldnames=['episode','timestamp','timecode','original_timestamp','original_timecode','adjusted_timestamp','adjusted_timecode','requested_timestamp','requested_timecode','title','category','notes','subtitle_text','image','frinkiac_url']); w.writeheader(); w.writerows(index)
            z.writestr('index.csv',sio.getvalue())
        out.seek(0); return send_file(out,mimetype='application/zip',as_attachment=True,download_name='kent-brockman-zine.zip')
    except Exception as e:return jsonify({'error':str(e)}),502


# Speaker-attributed dataset originally collected from Simpsons World.
# Fetch once per Railway instance; never treat an unavailable dataset as zero matches.
DATA_ROOT='https://raw.githubusercontent.com/sujanjoejacob/Text-mining-with-Simpsons-Data/master/'
_DATA_LOCK=threading.Lock()
_DATA_INDEX=None
_DATA_INFO=None

def load_kent_dataset():
    global _DATA_INDEX,_DATA_INFO
    if _DATA_INDEX is not None:return _DATA_INDEX,_DATA_INFO
    with _DATA_LOCK:
        if _DATA_INDEX is not None:return _DATA_INDEX,_DATA_INFO
        cache_dir=os.getenv('KENT_DATA_DIR','/tmp/kent-brockman-data')
        os.makedirs(cache_dir,exist_ok=True)
        sources={
            'episodes':'simpsons_episodes.csv',
            'characters':'simpsons_characters.csv',
            'lines':'simpsons_script_lines.csv.zip',
        }
        for name,filename in sources.items():
            path=os.path.join(cache_dir,filename)
            if not os.path.exists(path) or os.path.getsize(path)==0:
                r=get(DATA_ROOT+filename,timeout=90)
                temp=path+'.part'
                with open(temp,'wb') as f:f.write(r.content)
                os.replace(temp,path)
        def read_csv_bytes(data):
            return csv.DictReader(io.StringIO(data.decode('utf-8-sig',errors='replace'),newline=''))
        with open(os.path.join(cache_dir,sources['episodes']),'rb') as f:
            eps=list(read_csv_bytes(f.read()))
        with open(os.path.join(cache_dir,sources['characters']),'rb') as f:
            chars=list(read_csv_bytes(f.read()))
        kent_ids={str(c.get('id','')).strip() for c in chars if (c.get('normalized_name') or c.get('name') or '').strip().casefold()=='kent brockman'}
        if not kent_ids:raise ValueError('Kent Brockman not found in character table')
        episodes={}
        for e in eps:
            try:
                season=int(float(e['season'])); number=int(float(e['number_in_season']))
                if 1<=season<=10:
                    episodes[str(e['id'])]=f'S{season:02d}E{number:02d}'
            except (ValueError,KeyError,TypeError):pass
        index={}
        with zipfile.ZipFile(os.path.join(cache_dir,sources['lines'])) as z:
            members=[x for x in z.namelist() if x.lower().endswith('.csv')]
            if not members:raise ValueError('No CSV in script archive')
            with z.open(members[0]) as f:
                reader=csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',errors='replace',newline=''))
                for row in reader:
                    ep=episodes.get(str(row.get('episode_id','')).strip())
                    if not ep or str(row.get('character_id','')).strip() not in kent_ids:continue
                    phrase=(row.get('spoken_words') or '').strip()
                    if not phrase:continue
                    try:timestamp=int(float(row.get('timestamp_in_ms') or 0))
                    except (TypeError,ValueError):timestamp=0
                    index.setdefault(ep,[]).append({'dialogue':phrase,'dataset_timestamp':timestamp,'line_id':row.get('id','')})
        if not index:raise ValueError('No Kent-attributed dialogue found for Seasons 1–10')
        _DATA_INDEX=index
        _DATA_INFO={'source':DATA_ROOT,'kent_character_ids':sorted(kent_ids),'episodes_with_kent':len(index),'lines':sum(map(len,index.values())),'coverage':'Seasons 1–10'}
        return index,_DATA_INFO

def _words(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?",str(s).casefold())


def _phrase_score(source, subtitle):
    """Score only subtitle-supported matches; 0 means do not claim it is Kent."""
    a=_words(source); b=_words(subtitle)
    if len(a)<3 or len(b)<3:return 0.0
    if len(a)>len(b): longer=a; shorter=b
    else: longer=b; shorter=a
    ratio=SequenceMatcher(None,' '.join(a),' '.join(b),autojunk=False).ratio()
    # Compare the transcript with every equal-sized segment of the subtitle.
    window=max(3,min(len(a),len(b)))
    best=0.0
    for x in range(len(longer)-window+1):
        best=max(best,SequenceMatcher(None,shorter,longer[x:x+window],autojunk=False).ratio())
    # Guard against short common phrases matching another character.
    same_four=any(a[i:i+4]==b[j:j+4] for i in range(max(0,len(a)-3)) for j in range(max(0,len(b)-3)))
    if same_four:return max(best,ratio,0.81)
    return max(best,ratio)*0.85


def _search_phrases(dialogue):
    words=_words(dialogue)
    if len(words)<4:return []
    # Short fragments are more reliable than v2.9's ten-word exact subtitle search.
    lengths=[5,4] if len(words)>=5 else [4]
    offsets=[0, max(0,(len(words)-5)//2),max(0,len(words)-5)]
    out=[]
    for length in lengths:
        for start in offsets:
            q=' '.join(words[start:start+length])
            if len(q)>=15 and q not in out:out.append(q)
    return out[:4]


def find_frinkiac_match(ep,lines):
    """Only return a timestamp when Frinkiac subtitles support the actual Kent line."""
    checked=set(); queries=0; checked_captions=0
    for line in lines[:4]:
        speech=line['dialogue']
        for query in _search_phrases(speech)[:3]:
            if queries>=9 or checked_captions>=32:break
            queries+=1
            try:
                hits=[h for h in norm_search(api_json(f'{BASE}/api/search?q={requests.utils.quote(query)}')) if h['episode']==ep]
            except Exception:continue
            hits.sort(key=lambda h:abs(h['timestamp']-line['dataset_timestamp']))
            for h in hits[:5]:
                ts=h['timestamp']
                if ts in checked:continue
                checked.add(ts);checked_captions+=1
                sub=caption(ep,ts)
                score=_phrase_score(speech,sub)
                if score>=0.73:
                    return {'timestamp':ts,'subtitle':sub,'score':round(score,3)}
        if queries>=9 or checked_captions>=32:break
    return None


@app.route('/api/script-dataset-status')
def script_dataset_status():
    try:
        index,info=load_kent_dataset()
        return jsonify(ok=True,**info)
    except Exception as e:return jsonify(ok=False,error=str(e),source=DATA_ROOT),503


@app.route('/api/script-episodes')
def script_episodes():
    try:
        season=int(request.args.get('season','1'))
        if season not in EP_COUNTS:return jsonify(error='Season must be 1–10'),400
        index,info=load_kent_dataset()
        eps=[]
        for n in range(1,EP_COUNTS[season]+1):
            ep=f'S{season:02d}E{n:02d}'
            rows=index.get(ep,[])
            if rows:eps.append({'episode':ep,'line_count':len(rows),'first_line':rows[0]['dialogue'][:130]})
        return jsonify(season=season,episodes=eps,total_episodes=EP_COUNTS[season],
            total_lines=sum(e['line_count'] for e in eps),
            character_ids=info['kent_character_ids'],source=DATA_ROOT)
    except Exception as exc:return jsonify(error=f'{type(exc).__name__}: {exc}'),503


@app.route('/api/script-episode',methods=['POST'])
def script_episode():
    try:
        ep=str((request.get_json(silent=True) or {}).get('episode',''))
        match=re.fullmatch(r'S(0[1-9]|10)E(0[1-9]|[12][0-9])',ep)
        if not match:return jsonify(error='Invalid episode ID'),400
        season,num=int(match[1]),int(match[2])
        if num>EP_COUNTS[season]:return jsonify(error='Episode out of range'),400
        index,info=load_kent_dataset()
        rows=index.get(ep,[])
        groups=[]
        for line in sorted(rows,key=lambda x:x['dataset_timestamp']):
            if groups and line['dataset_timestamp']-groups[-1][-1]['dataset_timestamp']<=35000:
                groups[-1].append(line)
            else:groups.append([line])
        matches=[]
        for group in groups:
            # No Frinkiac requests here. Always deliver attributed dialogue quickly.
            matches.append({'episode':ep,'timestamp':None,'timecode':None,
                'dataset_timestamp':group[0]['dataset_timestamp'],'dataset_timecode':tc(group[0]['dataset_timestamp']),
                'dialogue':' '.join(line['dialogue'] for line in group),
                'lines':[{'dialogue':line['dialogue'],'timestamp':line['dataset_timestamp'],
                         'timecode':tc(line['dataset_timestamp']),'line_id':line['line_id']} for line in group],
                'source':'Kent Brockman (character ID 211) — speaker-attributed dataset',
                'match_status':'pending','matched_subtitle':None,'match_score':None,
                'attribution':'explicit','line_count':len(group),'frames':[]})
        return jsonify(episode=ep,matches=matches,source=DATA_ROOT,
            note=f'{len(rows)} Kent-attributed lines in {len(groups)} dialogue groups. Frinkiac checking is separate.',
            verified_matches=0,attribution='explicit',labelled_turns=len(rows),dataset=info)
    except Exception as exc:
        return jsonify(error=f'{type(exc).__name__}: {exc}',trace=traceback.format_exc()[-1200:]),503


@app.route('/api/script-match',methods=['POST'])
def script_match():
    try:
        payload=request.get_json(silent=True) or {}
        ep=str(payload.get('episode',''))
        ids=payload.get('line_ids',[])
        if not re.fullmatch(r'S(?:0[1-9]|10)E(?:0[1-9]|[12][0-9])',ep):return jsonify(error='Invalid episode'),400
        if not isinstance(ids,list) or not 1<=len(ids)<=40:return jsonify(error='Send 1 to 40 line IDs'),400
        index,_=load_kent_dataset()
        valid={str(row['line_id']):row for row in index.get(ep,[])}
        if any(str(i) not in valid for i in ids):return jsonify(error='Lines are not attributed to Kent in this episode'),400
        group=[valid[str(i)] for i in ids]
        found=find_frinkiac_match(ep,group)
        ts=found['timestamp'] if found else None
        frames=[]
        if found:
            try:frames=norm_frames(api_json(f'{BASE}/api/frames/{ep}/{ts}/7000/7000'),ep)[:15]
            except Exception:frames=[{'episode':ep,'timestamp':ts,'timecode':tc(ts),'image':img_url(ep,ts)}]
        return jsonify(timestamp=ts,timecode=tc(ts) if ts is not None else None,
            match_status='verified' if found else 'unverified',
            matched_subtitle=found['subtitle'] if found else None,
            match_score=found['score'] if found else None,frames=frames)
    except Exception as e:return jsonify(error=f'{type(e).__name__}: {e}'),503


@app.route('/health')
def health(): return {'ok':True,'version':'3.1','methods':['dialogue','visual-training','zine-library','zip-download','speaker-attributed-script-scan']}

if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.getenv('PORT','5000')))
