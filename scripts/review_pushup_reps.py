#!/usr/bin/env python3
"""Local-only MV7 repetition review. Run from pose_training_offline.
python scripts/review_pushup_reps.py [--port 8765]
No training, no network upload. Review data saved atomically on disk.
"""
import argparse, csv, json, re, hashlib, mimetypes, os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

BASE=Path('results/pushup/side_eval/evaluation')
OUT=Path('results/pushup/rep_review')

def read_csv(p):
    with p.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def atomic(p, data):
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(data,encoding='utf-8'); tmp.replace(p)

def load(base):
    v=read_csv(base/'per_video.csv'); m=read_csv(base/'manual_counts.csv') if (base/'manual_counts.csv').exists() else read_csv(base.parent/'manual_counts.csv')
    mm={r['sample_id']:r for r in m}; rr=read_csv(base/'candidate_reps.csv'); result={}
    for r in v:
        sid=r['sample_id']; p=Path(mm[sid]['video_path']); n=int(r['frames']); fps=float(r['fps'])
        if not p.is_file(): raise FileNotFoundError(p)
        if int(mm[sid]['manual_rep_count'])!=int(r['manual_count']): raise ValueError(sid+' manual count mismatch')
        reps=[dict(id='auto_'+x['rep'],start=int(x['start_frame']),end=int(x['end_frame']),label='',note='') for x in rr if x['sample_id']==sid]
        if len(reps)!=int(r['causal_count']) or any(not 0<=x['start']<x['end']<n for x in reps): raise ValueError(sid+' candidate mismatch')
        result[sid]=dict(sample_id=sid,group=r['group'],split=r['split'],fps=fps,frames=n,manual=int(r['manual_count']),path=str(p),clip_label=int(r['clip_label']),reps=reps)
    if len(result)!=23 or sum(len(x['reps']) for x in result.values())!=258: raise ValueError('Expected frozen 23 clips / 258 candidates')
    return result

def validate(s, v):
    if not isinstance(s,dict) or not isinstance(s.get('reps'),list): raise ValueError('invalid review')
    reps=[]; ids=set(); expected={x['id'] for x in v['reps']}
    for x in s['reps']:
        ident=x['id']; a=x['start']; b=x['end']; label=x['label']; note=x.get('note','')
        if not isinstance(ident,str) or not re.fullmatch(r'(auto|manual)_\d+',ident) or ident in ids or (ident.startswith('auto_') and ident not in expected): raise ValueError('invalid ID')
        if type(a)!=int or type(b)!=int or not 0<=a<b<v['frames']: raise ValueError('invalid frame interval')
        if label not in ('','good','bad','uncertain','reject') or not isinstance(note,str) or len(note)>1000: raise ValueError('invalid label/note')
        ids.add(ident); reps.append(dict(id=ident,start=a,end=b,label=label,note=note))
    if not expected<=ids: raise ValueError('Original candidates must remain; use reject')
    active=sorted([x for x in reps if x['label']!='reject'],key=lambda x:x['start'])
    complete=bool(s.get('complete')); note=s.get('note',''); revised=s.get('revised_count',v['manual'])
    if type(revised)!=int or revised<0 or not isinstance(note,str) or len(note)>2000: raise ValueError('invalid count/note')
    if complete:
        if any(not x['label'] for x in reps): raise ValueError('Unreviewed candidates remain')
        if len(active)!=revised: raise ValueError('Accepted count must match reviewed manual count')
        if revised!=v['manual'] and not note.strip(): raise ValueError('Explain revised manual count')
        if any(y['start']<x['end'] for x,y in zip(active,active[1:])): raise ValueError('Overlapping repetitions; fix boundaries')
    return dict(reps=reps,complete=complete,note=note,revised_count=revised)

def export(state, videos, out):
    p=out/'reviewed_reps.csv'; tmp=p.with_suffix('.tmp')
    fields=['sample_id','group','split','rep_id','start_frame','end_frame','fps','review_label','note','source_clip_label','video_complete']
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for sid,s in state.items():
            v=videos[sid]
            for r in s['reps']:
                w.writerow(dict(sample_id=sid,group=v['group'],split=v['split'],rep_id=r['id'],start_frame=r['start'],end_frame=r['end'],fps=v['fps'],review_label=r['label'],note=r['note'],source_clip_label=v['clip_label'],video_complete=s['complete']))
    tmp.replace(p)

HTML=r'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Push-up 반복 검수</title><style>
body{font:15px system-ui;margin:24px;background:#f4f6f8;color:#1c2531}h1{font-size:22px}button,select,input{font:inherit;padding:6px;margin:3px}button{cursor:pointer}button:disabled{opacity:.5}main{display:grid;grid-template-columns:minmax(400px,1.3fr) minmax(330px,1fr);gap:18px}section{background:white;border:1px solid #d0d6df;padding:16px;border-radius:10px}video{width:100%;max-height:460px;background:black}label{display:inline-block}textarea{width:96%;height:50px}#rows{max-height:300px;overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:6px;border-bottom:1px solid #e1e5eb;text-align:left}tr.cur{background:#dcecff}tr{cursor:pointer}#msg{white-space:pre-wrap;color:#944421}#progress{font-weight:bold}small{color:#586579}@media(max-width:850px){main{display:block}}
</style><h1>푸시업 반복 구간 및 자세 검수</h1><p id="progress"></p><select id="vid"></select><button id="nextvid">다음 영상</button><span id="info"></span><main><section><video id="video" controls preload="metadata"></video><div><button id="full">전체 영상 재생</button><button id="play">현재 반복 재생</button><button id="prev">이전 반복</button><button id="next">다음 반복</button></div><div><button id="minus">−1 frame</button><button id="plus">+1 frame</button><span id="frame"></span></div><p><small>구간을 확인하고 필요 시 시작/끝 프레임을 수정하세요. 원본 영상의 자세 Label은 화면에 표시하지 않습니다. 단축키: 1 Good, 2 Bad, 3 보류, 4 제외, N 다음 반복, P 반복 재생.</small></p></section><section><div id="rows"></div><div><button id="add">누락 반복 추가 (현재 프레임에서)</button></div><hr><div id="editor"><strong id="repname"></strong><div><label>Start <input id="start" type="number" min="0" style="width:75px"></label><button id="setstart">현재 프레임</button></div><div><label>End <input id="end" type="number" min="0" style="width:75px"></label><button id="setend">현재 프레임</button></div><div><button data-label="good">Good</button><button data-label="bad">Bad</button><button data-label="uncertain">판정 보류</button><button data-label="reject">오검출 제외</button></div><label>판정: <b id="label"></b></label><textarea id="repnote" placeholder="반복별 특이사항 (선택)"></textarea></div><hr><p id="counts"></p><label>검수 후 실제 반복 수 <input id="revised" type="number" min="0" style="width:65px"></label><textarea id="note" placeholder="영상 전체 특이사항 / 수동 횟수 수정 이유"></textarea><button id="save">임시 저장</button><button id="done">영상 검수 완료</button><p id="msg"></p></section></main>
<script>
const $=id=>document.getElementById(id);let data, states, v,s,idx=0, stop=null,dirty=false;const video=$('video');
const names={'':'미검수',good:'Good',bad:'Bad',uncertain:'보류',reject:'제외'};
function frame(){return Math.min(v.frames-1,Math.max(0,Math.floor(video.currentTime*v.fps+1e-5)))}
function seek(f){stop=null;video.pause();video.currentTime=(Math.min(v.frames-1,Math.max(0,f))+.05)/v.fps;}
function render(){ $('progress').textContent=`완료 ${Object.values(states).filter(x=>x.complete).length} / ${data.length} 영상`; const r=s.reps[idx]; $('rows').innerHTML='<table><thead><tr><th>구간</th><th>시작</th><th>끝</th><th>판정</th></tr></thead><tbody></tbody></table>';const body=$('rows').querySelector('tbody'); s.reps.forEach((x,i)=>{const tr=document.createElement('tr');if(i===idx)tr.className='cur';[x.id,x.start,x.end,names[x.label]].forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td)});tr.onclick=()=>{idx=i;render();seek(s.reps[i].start)};body.append(tr)}); $('repname').textContent=r.id; $('start').value=r.start;$('end').value=r.end;$('label').textContent=names[r.label];$('repnote').value=r.note; $('revised').value=s.revised_count;$('note').value=s.note; $('counts').textContent=`기존 수동 ${v.manual}회 / 자동 ${v.reps.length}회 / 검수 확정 ${s.reps.filter(x=>x.label&&x.label!=='reject').length}회 / 미검수 ${s.reps.filter(x=>!x.label).length}개`; $('info').textContent=`${v.fps} FPS · ${v.frames} frames · ${s.complete?'완료':'검수 중'}`; }
function change(){dirty=true;s.complete=false;} function load(id){v=data.find(x=>x.sample_id===id);s=states[id]||{reps:structuredClone(v.reps),complete:false,note:'',revised_count:v.manual};idx=0;dirty=false;stop=null;video.src='/video?id='+encodeURIComponent(id);$('msg').textContent='';render();}
async function save(complete){try{const res=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:v.sample_id,state:{...s,complete}})});const ret=await res.json();if(!res.ok)throw Error(ret.error);s=ret.state;states[v.sample_id]=s;dirty=false;render();$('msg').textContent=complete?'검수 완료 저장됨':'임시 저장됨';return true;}catch(e){$('msg').textContent=e.message;return false;}}
async function switchvid(id){if(dirty&&!await save(false))return;$('vid').value=id;load(id)}
$('vid').onchange=()=>{const id=$('vid').value;$('vid').value=v.sample_id;switchvid(id)};$('nextvid').onclick=()=>{const i=data.indexOf(v);if(i+1<data.length)switchvid(data[i+1].sample_id)};
$('full').onclick=()=>{stop=null;video.currentTime=0;video.play()};$('play').onclick=()=>{const r=s.reps[idx];video.currentTime=r.start/v.fps;stop=(r.end+1)/v.fps;video.play()};video.ontimeupdate=()=>{if(stop!==null&&video.currentTime>=stop){video.pause();stop=null}$('frame').textContent=`Frame ${frame()}`};$('minus').onclick=()=>seek(frame()-1);$('plus').onclick=()=>seek(frame()+1);
$('prev').onclick=()=>{if(idx>0){idx--;render();seek(s.reps[idx].start)}};$('next').onclick=()=>{if(idx+1<s.reps.length){idx++;render();seek(s.reps[idx].start)}};
function setBound(key,val){if(!Number.isInteger(val)||val<0||val>=v.frames){$('msg').textContent='프레임 범위 오류';render();return} const r=s.reps[idx];if((key==='start'&&val>=r.end)||(key==='end'&&val<=r.start)){$('msg').textContent='시작은 끝보다 앞서야 합니다';render();return}r[key]=val;change();render()}
$('start').onchange=()=>setBound('start',Number($('start').value));$('end').onchange=()=>setBound('end',Number($('end').value));$('setstart').onclick=()=>setBound('start',frame());$('setend').onclick=()=>setBound('end',frame());document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>{s.reps[idx].label=b.dataset.label;change();render()});$('repnote').oninput=()=>{s.reps[idx].note=$('repnote').value;change()};$('note').oninput=()=>{s.note=$('note').value;change()};$('revised').onchange=()=>{s.revised_count=Number($('revised').value);change();render()};$('add').onclick=()=>{const f=Math.min(frame(),v.frames-2);let i=1;while(s.reps.some(x=>x.id==='manual_'+i))i++;s.reps.push({id:'manual_'+i,start:f,end:Math.min(v.frames-1,f+Math.round(v.fps)),label:'',note:''});idx=s.reps.length-1;change();render()};$('save').onclick=()=>save(false);$('done').onclick=()=>save(true);document.addEventListener('keydown',e=>{if(!s||['INPUT','TEXTAREA','SELECT','BUTTON','VIDEO'].includes(e.target.tagName)||e.ctrlKey||e.altKey||e.metaKey)return;const l={'1':'good','2':'bad','3':'uncertain','4':'reject'}[e.key];if(l){e.preventDefault();s.reps[idx].label=l;change();render()}else if(e.key.toLowerCase()==='n'){e.preventDefault();$('next').click()}else if(e.key.toLowerCase()==='p'){e.preventDefault();$('play').click()}});window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue=''}};
(async()=>{const r=await fetch('/data');const d=await r.json();data=d.videos;states=d.states;for(const x of data){const o=document.createElement('option');o.value=x.sample_id;o.textContent=x.sample_id.replace('MV7_','').replace(/_(good|bad)_side/,'')+' · '+(data.indexOf(x)+1);$('vid').append(o)}const first=data.find(x=>!states[x.sample_id]?.complete)||data[0];$('vid').value=first.sample_id;load(first.sample_id)})().catch(e=>$('msg').textContent=e.message);
</script></html>'''

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base',type=Path,default=BASE);ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--port',type=int,default=8765);a=ap.parse_args()
    videos=load(a.base); a.out.mkdir(parents=True,exist_ok=True); p=a.out/'review_state.json'; signature=hashlib.sha256((a.base/'candidate_reps.csv').read_bytes()+(a.base/'per_video.csv').read_bytes()).hexdigest()
    doc=json.loads(p.read_text()) if p.exists() else {'source_sha256':signature,'states':{}}
    if doc['source_sha256']!=signature: raise ValueError('Candidate source changed; use a different review directory')
    state=doc['states']
    for k,s in state.items(): validate(s,videos[k])
    import threading
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,body,ctype='application/json'):
            if isinstance(body,str): body=body.encode()
            self.send_response(code);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            u=urlsplit(self.path)
            if u.path=='/': return self.send(200,HTML,'text/html; charset=utf-8')
            if u.path=='/data':
                vv=[{k:x for k,x in v.items() if k not in ('path','clip_label')} for v in videos.values()]
                return self.send(200,json.dumps({'videos':vv,'states':state}))
            if u.path!='/video':return self.send(404,'{}')
            sid=parse_qs(u.query).get('id',[''])[0]
            if sid not in videos:return self.send(404,'{}')
            path=Path(videos[sid]['path']); size=path.stat().st_size; lo=0; hi=size-1; rng=self.headers.get('Range'); status=200
            if rng:
                m=re.fullmatch(r'bytes=(\d*)-(\d*)',rng)
                if not m or not any(m.groups()):return self.send(416,'{}')
                if not m[1]:lo=max(0,size-int(m[2]))
                else:lo=int(m[1]);hi=min(hi,int(m[2])) if m[2] else hi
                if lo>hi:return self.send(416,'{}')
                status=206
            self.send_response(status);self.send_header('Content-Type',mimetypes.guess_type(path)[0] or 'video/mp4');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(hi-lo+1));
            if status==206:self.send_header('Content-Range',f'bytes {lo}-{hi}/{size}')
            self.end_headers()
            try:
                with path.open('rb') as f:
                    f.seek(lo);remain=hi-lo+1
                    while remain:
                        b=f.read(min(remain,262144));self.wfile.write(b);remain-=len(b)
                        if not b:break
            except (BrokenPipeError,ConnectionResetError):pass
        def do_POST(self):
            try:
                if self.path!='/save':return self.send(404,'{}')
                if self.headers.get('Origin') not in (f'http://localhost:{a.port}',f'http://127.0.0.1:{a.port}'):return self.send(403,'{"error":"local origin required"}')
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<1000000:raise ValueError('Invalid request length')
                d=json.loads(self.rfile.read(n));sid=d['id'];s=validate(d['state'],videos[sid])
                with lock:
                    state[sid]=s;atomic(p,json.dumps(doc,ensure_ascii=False,indent=2));export(state,videos,a.out)
                self.send(200,json.dumps({'state':s},ensure_ascii=False))
            except (ValueError,KeyError,TypeError) as e:self.send(400,json.dumps({'error':str(e)},ensure_ascii=False))
    print(f'Open http://localhost:{a.port} in Windows browser. Ctrl+C stops server. Saved in {a.out}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
