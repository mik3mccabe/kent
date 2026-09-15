import os, csv, io, requests
from flask import Flask, request, jsonify, render_template_string, Response

app = Flask(__name__)
BASE = os.getenv('FRINKIAC_BASE', 'https://frinkiac.com')

HTML = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kent Brockman Frame Finder</title><style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:30px auto;padding:0 18px}form{display:flex;gap:8px;flex-wrap:wrap}input,button{font:inherit;padding:9px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;margin-top:22px}.card{border:1px solid #ccc;padding:10px;border-radius:8px}img{width:100%;height:auto}.meta{font-size:13px;word-break:break-word}textarea{width:100%;min-height:60px}</style></head><body><h1>Kent Brockman Frame Finder</h1><p>Search dialogue to locate a scene, then inspect nearby clean Frinkiac frames.</p><form method="get"><input name="q" value="{{q}}" placeholder="e.g. Kent Brockman" size="35"><label>Seconds either side <input name="window" value="{{window}}" type="number" min="0" max="30" title="Seconds before and after each dialogue hit" style="width:70px"></label><button>Search</button></form>{% if error %}<p>{{error}}</p>{% endif %}<div class="grid">{% for x in items %}<div class="card"><a href="{{x.image}}" target="_blank"><img src="{{x.image}}" loading="lazy"></a><div class="meta"><b>{{x.episode}}</b> · {{x.timecode}} · {{x.timestamp}} ms<br>{{x.text}}</div></div>{% endfor %}</div></body></html>'''

def api_get(path, params=None):
    r=requests.get(BASE+path, params=params, timeout=20, headers={'User-Agent':'kent-brockman-zine-research/1.0'})
    r.raise_for_status(); return r.json()

def tc(ms):
    s=ms//1000; return f'{s//60:02d}:{s%60:02d}.{ms%1000:03d}'

def normalize_search(data):
    # Frinkiac can return Episode as either a string (e.g. "S06E18")
    # or an object containing Key/key. Accept both forms.
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get('Results', data.get('results', []))
    else:
        rows = []
    out=[]
    for r in rows:
        if not isinstance(r, dict):
            continue
        ep=r.get('Episode') or r.get('episode') or ''
        if isinstance(ep, str):
            episode=ep
        elif isinstance(ep, dict):
            episode=ep.get('Key') or ep.get('key') or ''
        else:
            episode=''
        episode=episode or r.get('EpisodeKey') or r.get('episodeKey') or ''
        ts=r.get('Timestamp', r.get('timestamp', 0))
        text=r.get('Text') or r.get('text') or r.get('Subtitle') or ''
        try:
            ts=int(ts)
        except (TypeError, ValueError):
            continue
        if episode:
            out.append({'episode':episode,'timestamp':ts,'text':text})
    return out

@app.route('/')
def index():
    q=request.args.get('q','')
    window=int(request.args.get('window','3') or 3)
    items=[]; error=None
    if q:
        try:
            hits=normalize_search(api_get('/api/search', {'q':q}))
            # Sample around each hit at one-second intervals. This is deliberately visual-review oriented.
            seen=set()
            for h in hits[:80]:
                for off in range(-window, window+1):
                    ts=max(0,h['timestamp']+off*1000); key=(h['episode'],ts)
                    if key in seen: continue
                    seen.add(key)
                    items.append({**h,'timestamp':ts,'timecode':tc(ts),'image':f'{BASE}/img/{h["episode"]}/{ts}.jpg'})
        except Exception as e: error=f'Frinkiac request failed: {e}'
    return render_template_string(HTML,q=q,window=window,items=items,error=error)

@app.route('/api/search')
def search():
    q=request.args.get('q','')
    if not q:return jsonify({'error':'q is required'}),400
    try:return jsonify(normalize_search(api_get('/api/search',{'q':q})))
    except Exception as e:return jsonify({'error':str(e)}),502

@app.route('/api/nearby')
def nearby():
    episode=request.args.get('episode'); timestamp=request.args.get('timestamp',type=int)
    if not episode or timestamp is None:return jsonify({'error':'episode and timestamp are required'}),400
    try:return jsonify(api_get('/api/nearby',{'e':episode,'t':timestamp}))
    except Exception as e:return jsonify({'error':str(e)}),502

@app.route('/export.csv')
def export_csv():
    q=request.args.get('q','Kent Brockman')
    try: hits=normalize_search(api_get('/api/search',{'q':q}))
    except Exception as e:return str(e),502
    buf=io.StringIO(); w=csv.writer(buf); w.writerow(['episode','timestamp_ms','timecode','dialogue','image_url'])
    for h in hits:w.writerow([h['episode'],h['timestamp'],tc(h['timestamp']),h['text'],f'{BASE}/img/{h["episode"]}/{h["timestamp"]}.jpg'])
    return Response(buf.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=frinkiac-results.csv'})

@app.route('/health')
def health(): return {'ok':True}
