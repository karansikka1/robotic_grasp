"""Build a self-contained local video review page for the categorized demonstrations."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import h5py

from v4.perturbations import CATEGORIES


DESCRIPTIONS = {
    'clean': ('Clean', 'The teacher command is executed unchanged.'),
    'gaussian_noise': ('Gaussian noise', 'Small XYZ noise on every command: standard deviation 0.015, clipped at ±0.045 normalized units (0.75 mm / 2.25 mm commanded translation).'),
    'dropped_commands': ('Dropped commands', 'Occasional zero XYZ commands for 1–3 control steps (50–150 ms). A burst starts with probability 3% on each idle step.'),
    'delayed_commands': ('Delayed commands', 'XYZ commands arrive one control step late (50 ms). The initial movement command is zero.'),
    'sustained_bias': ('Sustained bias', 'A fixed random-direction XYZ offset of magnitude 0.02 (1 mm commanded translation) lasts 5–10 steps. A burst starts with probability 2% on each idle step.'),
}


def build(root):
    root = Path(root).resolve()
    summary = json.loads((root / 'summary.json').read_text())
    videos_path = root / 'videos.json'
    if videos_path.exists():
        entries = json.loads(videos_path.read_text())['videos']
    else:
        entries = []
        for path in sorted((root / 'jobs').glob('*.json')):
            state = json.loads(path.read_text())
            result = state['result']
            if result is not None and result['video']:
                entries.append({'path': result['trajectory'], 'video': result['video'],
                                'seed': result['seed'], 'category': state['slot']['category'],
                                'partition': state['slot']['partition'], 'steps': result['steps']})
    records = []
    for entry in entries:
        with h5py.File(entry['path'], 'r') as f:
            records.append({**entry, 'video': os.path.relpath(entry['video'], root),
                            'path': os.path.relpath(entry['path'], root),
                            'actions': f['actions'][:].round(6).tolist(),
                            'executed': f['executed_actions'][:].round(6).tolist(),
                            'stages': f['stage'].asstr()[:].tolist(),
                            'active': f['perturbation/active'][:].tolist(),
                            'changed': f['perturbation/changed'][:].tolist(),
                            'events': f['perturbation/event_id'][:].tolist(),
                            'changed_steps': int(f['perturbation/changed'][:].sum())})
    records.sort(key=lambda r: (CATEGORIES.index(r['category']), r['seed']))
    data = {'summary': summary, 'categories': [{'id': c, 'name': DESCRIPTIONS[c][0],
                                             'description': DESCRIPTIONS[c][1]} for c in CATEGORIES],
            'videos': records}
    serialized = json.dumps(data, separators=(',', ':')).replace('<', '\\u003c')
    output = TEMPLATE.replace('__DATA__', serialized)
    if (root / 'visible_pilots' / 'index.html').exists():
        output = output.replace('<body><main>', '<body><main><p class="panel"><a href="visible_pilots/">New: three visible disturbance previews — physical block drop, Gaussian burst, and sideways motion</a></p>')
    if (root / 'retirement.json').exists():
        output = output.replace('<body><main>', '<body><main><p class="panel">Archived mild dataset: 290 training/validation HDF5 files were deleted to free storage. These ten review examples remain. <a href="recovery300/">Open the stronger recovery dataset</a>.</p>')
    (root / 'index.html').write_text(output)
    return {'page': str(root / 'index.html'), 'videos': len(records),
            'categories': dict(Counter(r['category'] for r in records))}


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BC trajectory review</title>
<style>
:root{color-scheme:dark;--bg:#10151d;--panel:#19212c;--line:#303e4e;--text:#eef2f6;--muted:#acbbc9;--blue:#68d5e3;--orange:#ffb274}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,sans-serif}main{max-width:1320px;margin:auto;padding:32px 28px 60px}a{color:var(--blue)}h1{font-size:34px;line-height:1.15;margin:8px 0 12px;font-weight:650;letter-spacing:-1px}h2{font-size:20px;margin:0}p{margin:8px 0;color:var(--muted)}.eyebrow{font-size:12px;letter-spacing:2px;color:var(--blue);text-transform:uppercase}.header{display:flex;justify-content:space-between;gap:24px;align-items:center;margin-bottom:26px}.stats{display:flex;gap:28px}.stat strong{display:block;font-size:27px}.stat span{font-size:12px;color:var(--muted)}.tabs{display:flex;flex-wrap:wrap;gap:8px;margin:24px 0 20px}button,select{font:inherit;color:var(--text);border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:9px 13px;cursor:pointer}button:hover{border-color:var(--blue)}button[aria-pressed=true]{background:#24424c;border-color:var(--blue);color:#b5f5ff}button:disabled{opacity:.4;cursor:default}button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid var(--blue);outline-offset:3px}.layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:20px}.viewer{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#080b10}video{display:block;width:100%;aspect-ratio:2/1;background:#05080c}.camera-labels{display:flex;border-top:1px solid var(--line);font-size:11px;color:var(--muted);text-align:center;padding:7px}.camera-labels span{width:50%}.controls{padding:14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}.controls select{margin-left:auto}.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px}.pill{display:inline-block;padding:3px 9px;background:#233d35;color:#ace7c7;border-radius:20px;font-size:12px;margin-bottom:12px}.meta{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:22px 0}.meta strong{display:block;font-size:20px}.meta span{font-size:12px;color:var(--muted)}.stage{min-height:75px;padding:12px;background:#111a25;border-radius:8px;margin:16px 0;overflow-wrap:anywhere}.stage small{display:block;color:var(--muted);font-size:11px;letter-spacing:1px;text-transform:uppercase}.active{color:var(--orange)}.samples{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}.plot{margin-top:20px;padding:18px 20px;background:var(--panel);border:1px solid var(--line);border-radius:14px}.plot-head{display:flex;justify-content:space-between;gap:10px;align-items:baseline;flex-wrap:wrap}.legend{font-size:12px;color:var(--muted)}.teacher{color:var(--blue)}.executed{color:var(--orange)}canvas{width:100%;height:210px;display:block;cursor:crosshair;margin-top:10px}.hint{font-size:12px;color:var(--muted)}.footer{margin-top:25px;border-top:1px solid var(--line);padding-top:18px;display:flex;gap:20px;flex-wrap:wrap;font-size:13px}.empty{padding:50px;text-align:center;color:var(--muted)}#error{color:#ffaaa3}.collection{margin-top:30px}.collection table{border-collapse:collapse;width:100%;font-size:13px}.collection th,.collection td{padding:10px 8px;border-bottom:1px solid var(--line);text-align:right}.collection th:first-child,.collection td:first-child{text-align:left}.collection th{color:var(--muted);font-weight:500}
@media(max-width:850px){.layout{grid-template-columns:1fr}.header{display:block}.stats{margin-top:24px}.panel{padding:16px}.meta{grid-template-columns:repeat(4,1fr)}main{padding:22px 16px}.stage{min-height:0}}
@media(max-width:450px){.meta{grid-template-columns:1fr 1fr}.stats{gap:20px}.tabs button{font-size:13px;padding:8px}.controls button{font-size:12px}}
</style></head>
<body><main>
<div class="header"><div><div class="eyebrow">Robotic grasp / behavior cloning</div><h1>Trajectory review</h1><p>Inspect the disturbances and the teacher’s response, frame by frame.</p></div><div class="stats"><div class="stat"><strong id="total">—</strong><span>demonstrations saved</span></div><div class="stat"><strong id="validation">—</strong><span>validation trajectories</span></div><div class="stat"><strong id="video-count">—</strong><span>example videos</span></div></div></div>
<p class="hint">These are successful scripted-teacher demonstrations, not learned-policy evaluations. Rotation and gripper commands are unchanged in every category. Both cameras are shown.</p>
<nav class="tabs" id="categories" aria-label="Perturbation category"></nav>
<div id="empty" class="empty" hidden>No example is ready in this category yet. Refresh after collection progresses.</div>
<div id="review"><div class="layout"><div><div class="viewer"><video id="video" controls playsinline preload="metadata" aria-label="Teacher trajectory, front camera and wrist camera"></video><div class="camera-labels"><span>FRONT CAMERA</span><span>WRIST CAMERA</span></div><div class="controls"><button id="prev" title="Previous frame">← Frame</button><button id="next" title="Next frame">Frame →</button><button id="event">Next disturbance</button><select id="speed" aria-label="Playback speed"><option value="0.25">0.25× speed</option><option value="0.5">0.5× speed</option><option value="1" selected>1× speed</option><option value="2">2× speed</option></select></div></div><div class="samples" id="samples" aria-label="Example trajectory"></div><p id="error" role="status"></p></div>
<aside class="panel"><span class="pill">Stable full stack</span><h2 id="name"></h2><p id="description"></p><div class="meta"><div><strong id="duration"></strong><span>simulated seconds</span></div><div><strong id="steps"></strong><span>control actions</span></div><div><strong id="changed"></strong><span>disturbed actions</span></div><div><strong id="partition"></strong><span>dataset partition</span></div></div><div class="stage"><small id="frame"></small><div id="stage"></div><div id="disturbance" class="hint"></div><div id="delta" class="hint"></div></div><p class="hint" id="seed"></p><a id="download" download>Download video</a></aside></div>
<div class="plot"><div class="plot-head"><h2>Movement commands</h2><div class="legend"><span class="teacher">━ Teacher target</span> &nbsp; <span class="executed">┄ Executed</span> &nbsp; Shading: scheduled disturbance</div></div><canvas id="plot" aria-label="XYZ commands over time; click to seek in the video"></canvas><p class="hint">Click the plot to seek. Commands are normalized; 1 unit corresponds to 5 cm of commanded translation, not measured displacement. The marker identifies the command applied after the displayed frame.</p></div></div>
<div class="collection"><h2>Collection outcomes</h2><p class="hint">Teacher completion counts include rejected attempts. The dataset and examples retain successful trajectories only; these percentages are not learned-policy success rates.</p><table><thead><tr><th>Category</th><th>Saved</th><th>Attempts</th><th>Teacher completion</th><th>Validation</th></tr></thead><tbody id="outcomes"></tbody></table></div>
<div class="footer"><a href="summary.json">Collection summary</a><a href="plan.json">Frozen collection plan</a><a href="videos.json">Video index</a><a href="validation.json">Artifact checks</a><a href="replay_validation.json">Simulator replay checks</a></div>
<noscript>This page needs JavaScript for category selection and synchronized command plots. Videos are listed in videos.json.</noscript>
</main><script id="dataset" type="application/json">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('dataset').textContent);
const el=id=>document.getElementById(id), video=el('video'), canvas=el('plot');
let current=null;
el('total').textContent=data.summary.successful_trajectories+' / '+data.summary.target_trajectories;
el('validation').textContent=Object.values(data.summary.categories).reduce((n,c)=>n+c.validation_trajectories,0);
el('video-count').textContent=data.videos.length;
for(const c of data.categories){const button=document.createElement('button');button.textContent=c.name;button.dataset.category=c.id;button.setAttribute('aria-pressed','false');button.onclick=()=>selectCategory(c.id);el('categories').append(button);
const stats=data.summary.categories[c.id],row=document.createElement('tr');for(const value of [c.name,stats.successes,stats.attempts,stats.attempts?(100*stats.successes/stats.attempts).toFixed(1)+'%':'—',stats.validation_trajectories]){const cell=document.createElement('td');cell.textContent=value;row.append(cell)}el('outcomes').append(row)}
function selectCategory(id){document.querySelectorAll('[data-category]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.category===id)));const examples=data.videos.filter(v=>v.category===id);el('samples').replaceChildren();examples.forEach((v,i)=>{const b=document.createElement('button');b.textContent='Example '+(i+1)+' · seed '+v.seed;b.onclick=()=>selectVideo(v,i);el('samples').append(b)});el('empty').hidden=examples.length>0;el('review').hidden=!examples.length;if(examples.length)selectVideo(examples[0],0);else{video.pause();current=null}}
function selectVideo(v,index){current=v;video.pause();video.src=v.video;video.playbackRate=Number(el('speed').value);el('error').textContent='';const c=data.categories.find(c=>c.id===v.category);el('name').textContent=c.name;el('description').textContent=c.description;el('duration').textContent=(v.steps/20).toFixed(2);el('steps').textContent=v.steps;el('changed').textContent=v.changed_steps;el('partition').textContent=v.partition;el('seed').textContent='Layout seed '+v.seed+' · 20 Hz · one initial observation frame';el('download').href=v.video;el('samples').querySelectorAll('button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));el('event').disabled=!['dropped_commands','sustained_bias'].includes(v.category);update()}
function step(){return current?Math.min(current.steps,Math.max(0,Math.floor(video.currentTime*20+0.001))):0}
function seek(s){if(current){video.pause();video.currentTime=Math.max(0,Math.min(current.steps,s))/20;update()}}
el('prev').onclick=()=>seek(step()-1);el('next').onclick=()=>seek(step()+1);el('speed').onchange=()=>{video.playbackRate=Number(el('speed').value)};
el('event').onclick=()=>{if(!current)return;const s=step();for(let i=s+1;i<current.steps;i++)if(current.active[i]&&(i===0||current.events[i]!==current.events[i-1])){seek(i);return}seek(0)};
function update(){if(!current)return;const s=step(),terminal=s===current.steps;el('frame').textContent=terminal?'Terminal frame':'Command '+(s+1)+' / '+current.steps;el('stage').textContent=terminal?'Stack complete':current.stages[s].replaceAll('_',' ');el('disturbance').textContent=terminal?'Ten consecutive official success observations':current.changed[s]?'Movement command disturbed':current.active[s]?'Disturbance scheduled; command unchanged':'Teacher command executed unchanged';el('disturbance').className='hint'+(!terminal&&current.changed[s]?' active':'');el('delta').textContent=terminal?'':'Command offset XYZ: '+current.executed[s].slice(0,3).map((v,i)=>((v-current.actions[s][i])*50).toFixed(2)).join(', ')+' mm';draw(s)}
function draw(s){if(!current)return;const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=210;if(!w)return;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const left=30,right=w-12,top=12,bottom=h-25,row=(bottom-top)/3,x=i=>left+i/current.steps*(right-left),y=(value,axis)=>top+row*(axis+0.5)-value*(row-14)/1.5;
for(let i=0;i<current.steps;i++)if(current.active[i]){ctx.fillStyle='rgba(255,178,116,0.075)';ctx.fillRect(x(i),top,Math.max(1,x(i+1)-x(i)),bottom-top)}
for(let axis=0;axis<3;axis++){ctx.strokeStyle='#3a4858';ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(left,y(0,axis));ctx.lineTo(right,y(0,axis));ctx.stroke();ctx.fillStyle='#aabbca';ctx.font='12px system-ui';ctx.fillText(['X','Y','Z'][axis],5,y(0,axis)+4);for(const [key,color,dash]of[['actions','#68d5e3',[]],['executed','#ffb274',[3,3]]]){ctx.strokeStyle=color;ctx.lineWidth=1.4;ctx.setLineDash(dash);ctx.beginPath();current[key].forEach((a,i)=>{if(i)ctx.lineTo(x(i),y(a[axis],axis));else ctx.moveTo(x(i),y(a[axis],axis))});ctx.stroke()}}
ctx.setLineDash([]);ctx.strokeStyle='#eef2f6';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(s),top);ctx.lineTo(x(s),bottom);ctx.stroke();ctx.fillStyle='#aabbca';ctx.font='11px system-ui';ctx.fillText('0 s',left,h-7);ctx.textAlign='right';ctx.fillText((current.steps/20).toFixed(2)+' s',right,h-7)}
canvas.onclick=e=>{const rect=canvas.getBoundingClientRect();seek(Math.round((e.clientX-rect.left-30)/(rect.width-42)*current.steps))};
video.addEventListener('timeupdate',update);video.addEventListener('seeked',update);video.addEventListener('loadedmetadata',update);video.addEventListener('ended',update);video.addEventListener('error',()=>el('error').textContent='Video could not load. Open this page through the local server or download the MP4 directly.');window.addEventListener('resize',update);
if('requestVideoFrameCallback'in video){const frame=()=>{update();video.requestVideoFrameCallback(frame)};video.requestVideoFrameCallback(frame)}
selectCategory(data.categories[0].id);
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('v4/trajectories/robustness300'))
    args = parser.parse_args()
    print(json.dumps(build(args.root), indent=2))


if __name__ == '__main__':
    main()
